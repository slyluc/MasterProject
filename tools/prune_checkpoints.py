"""Thin checkpoint folders to one snapshot per million time units, and gather
them, thinned, into one folder and a zip archive.

Each snapshot holds the random state that continued its run, so any snapshot
can be rebuilt exactly by replaying from an earlier one in the same folder
(``MS.recreate_checkpoint``). In every folder that holds snapshots this tool
keeps

- the first snapshot of each whole million of time units, which is the one on
  the million itself when it exists, so no snapshot is ever more than one
  million time units of replay away (about ten minutes); and
- the folder's newest snapshot, so the run loads and resumes as before.

Every other file is kept: analysis records, batch plans, capture series and
features, videos. Before a snapshot is left out, its patch count must be in
the folder's ``analysis.npz``; missing counts are counted and recorded first,
and a snapshot the record does not track is kept. Recording changes no
snapshot, so a report-only run does it too. The tool also replays from kept
snapshots to the next left-out one and requires an exact match of the
lattice, Gamma, species, histories and random state; a folder that fails is
kept whole.

Nothing is deleted unless ``--apply`` is given:

    python tools/prune_checkpoints.py "checkpoints/*"            # report only
    python tools/prune_checkpoints.py "checkpoints/*" --apply    # thin in place

``--into`` instead copies the given folders, thinned, into one destination and
leaves the sources untouched; ``--zip`` then writes ``<destination>.zip``:

    python tools/prune_checkpoints.py "checkpoints/*" "E:/MasterProject/checkpoints/*" \\
        --into E:/MasterProject/export/checkpoints --zip
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import glob as globbing
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sim_modules as MS  # noqa: E402

EVERY = 1_000_000
# Already compressed, so deflating them only costs time.
STORED_SUFFIXES = {".mp4", ".png", ".jpg", ".jpeg", ".gif", ".zip"}
# What the random state drives; patch counts are derived from the lattice.
REPLAYED_KEYS = ("lattice", "Gamma", "current_species", "newest_species",
                 "timestep", "rng_state", "tracked_timesteps",
                 "diversity_history")


def resolve_directories(patterns):
    """Expand the command line into directories.

    The shell is not relied on to do this. cmd.exe does no globbing at all,
    and PowerShell does none for a native command, so ``checkpoints/*``
    arrives here as a literal string on Windows and as already-expanded
    names on a Unix shell. Both have to work.
    """
    directories = []
    seen = set()
    for pattern in patterns:
        matches = (
            [pattern]
            if Path(pattern).is_dir()
            else sorted(globbing.glob(pattern))
        )
        usable = [Path(match) for match in matches if Path(match).is_dir()]
        if not usable:
            print(f"warning: {pattern} matched no directory", file=sys.stderr)
            continue
        for path in usable:
            key = str(path.resolve()).casefold()
            if key not in seen:
                seen.add(key)
                directories.append(path)
    return directories


def _format_size(byte_count):
    """Render a byte count in the largest unit that keeps it readable."""
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024


def _key(path):
    return os.path.normcase(os.path.abspath(path))


def _skipped(path):
    """Files never copied: locks (.job.lock, .sweep.lock, .capture.lock), which
    are recreated on use, and leftovers of interrupted atomic writes."""
    return path.name.startswith(".") and path.name.endswith((".lock", ".tmp.npz"))


def keep_mask(timesteps, every=EVERY):
    """Which of a folder's sorted snapshot times to keep.

    The first snapshot in each block of ``every`` time units, and the newest.
    """
    timesteps = np.asarray(timesteps, dtype=np.int64)
    if not timesteps.size:
        return np.zeros(0, dtype=bool)
    blocks = timesteps // int(every)
    keep = np.r_[True, blocks[1:] != blocks[:-1]]
    keep[-1] = True
    return keep


def snapshot_folders(top):
    """Every folder under top, itself included, that holds snapshots."""
    return sorted({path.parent for path in Path(top).rglob("checkpoint_*.npz")})


def plan_folder(folder, every=EVERY):
    """Which snapshots of one folder to keep and which to leave out."""
    files = sorted(Path(folder).glob("checkpoint_*.npz"), key=MS._checkpoint_timestep)
    files = [path for path in files if MS._checkpoint_timestep(path) >= 0]
    keep = keep_mask([MS._checkpoint_timestep(path) for path in files], every)
    return {
        "folder": Path(folder),
        "files": files,
        "keep": [path for path, kept in zip(files, keep) if kept],
        "drop": [path for path, kept in zip(files, keep) if not kept],
        "counted": 0,
        "unrecordable": 0,
        "checks": [],
        "failed": [],
    }


def _count_file_patches(path):
    """Return one snapshot's timestep and patch count."""
    with np.load(path, allow_pickle=False) as saved:
        return int(saved["timestep"]), MS.count_patches(saved["lattice"])


def _recorded_counts(record, timesteps):
    """The record's patch count at each timestep: -1 when uncounted, -2 when untracked."""
    counts = np.full(len(timesteps), -2, dtype=np.int64)
    recorded_times = np.asarray(record["tracked_timesteps"], dtype=np.int64)
    if not recorded_times.size:
        return counts
    timesteps = np.asarray(timesteps, dtype=np.int64)
    position = np.minimum(np.searchsorted(recorded_times, timesteps),
                          recorded_times.size - 1)
    tracked = recorded_times[position] == timesteps
    counts[tracked] = np.asarray(record["patch_history"], dtype=np.int64)[position[tracked]]
    return counts


def _new_record(newest_file):
    """An analysis record built from a folder's newest snapshot."""
    state = MS.load_checkpoint(newest_file)
    record = dict(state)
    record["event_timesteps"] = MS.detect_diversity_swaps(
        state["tracked_timesteps"], state["diversity_history"], MS.RetentionPolicy()
    )
    record["lattice_shape"] = np.asarray(state["lattice"]).shape
    return record


def complete_record(plan, workers=8):
    """Make the folder's analysis.npz hold the patch count of every snapshot left out.

    Missing counts are counted from the snapshots and the record is replaced
    atomically; a folder without a record gets one from its newest snapshot.
    A snapshot the record does not track at all cannot be recorded, so it is
    kept. Rerunning reuses the counts already written.
    """
    if not plan["drop"]:
        return
    folder = plan["folder"]
    record = MS.load_analysis(folder)
    had_record = record is not None
    if not had_record:
        record = _new_record(plan["files"][-1])
    times = [MS._checkpoint_timestep(path) for path in plan["drop"]]
    counts = _recorded_counts(record, times)
    missing = [path for path, count in zip(plan["drop"], counts) if count == -1]
    if missing or not had_record:
        patches = np.asarray(record["patch_history"], dtype=np.int64).copy()
        recorded_times = np.asarray(record["tracked_timesteps"], dtype=np.int64)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for timestep, count in pool.map(_count_file_patches, missing):
                patches[np.searchsorted(recorded_times, timestep)] = count
        record["patch_history"] = patches
        MS._write_analysis(folder, record)
        plan["counted"] = len(missing)
        counts = _recorded_counts(record, times)
    untracked = {_key(path) for path, count in zip(plan["drop"], counts) if count < 0}
    if untracked:
        plan["unrecordable"] = len(untracked)
        plan["keep"] = sorted(
            plan["keep"] + [path for path in plan["drop"] if _key(path) in untracked],
            key=MS._checkpoint_timestep,
        )
        plan["drop"] = [path for path in plan["drop"] if _key(path) not in untracked]


def choose_checks(plan, checks):
    """Up to ``checks`` (kept snapshot, next snapshot) pairs, spread over the folder.

    Each pair replays one tracking interval from a kept snapshot to a
    snapshot that will be left out.
    """
    dropped = {_key(path) for path in plan["drop"]}
    files = plan["files"]
    pairs = [
        (files[index], files[index + 1])
        for index in range(len(files) - 1)
        if _key(files[index]) not in dropped and _key(files[index + 1]) in dropped
    ]
    if checks <= 0 or not pairs:
        return []
    picks = np.unique(np.linspace(0, len(pairs) - 1, min(checks, len(pairs))).round())
    return [pairs[int(index)] for index in picks]


def replay_matches(start, target):
    """Replay from one snapshot to a later saved one; return the first difference, or None."""
    began = time.perf_counter()
    replayed = MS.recreate_checkpoint(target, start=start)
    saved = MS.load_checkpoint(target)
    for key in REPLAYED_KEYS:
        first, second = saved[key], replayed[key]
        if isinstance(first, dict) or isinstance(second, dict):
            same = first == second
        else:
            same = np.array_equal(np.asarray(first), np.asarray(second))
        if not same:
            return key, time.perf_counter() - began
    return None, time.perf_counter() - began


def run_checks(plans, workers):
    """Run every planned replay check in parallel; a failing folder is kept whole."""
    jobs = [(plan, start, target) for plan in plans for start, target in plan["checks"]]
    if not jobs:
        return
    workers = max(1, min(workers, len(jobs)))
    print(f"replaying {len(jobs)} checks on {workers} processes...", flush=True)

    def outcomes():
        if workers == 1:
            for _, start, target in jobs:
                yield lambda start=start, target=target: replay_matches(start, target)
            return
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(replay_matches, start, target)
                           for _, start, target in jobs]:
                yield future.result

    for (plan, start, target), outcome in zip(jobs, outcomes()):
        try:
            difference, _ = outcome()
        except Exception as error:  # A replay that cannot run proves nothing.
            difference = f"error: {error}"
        if difference is not None:
            plan["failed"].append(f"{start.name} -> {target.name}: {difference}")
    for plan in plans:
        if plan["failed"]:
            plan["keep"], plan["drop"] = plan["files"], []


def gather(tops, plans, destination):
    """Copy each top folder, without its left-out snapshots, to destination/<name>.

    Files already copied with the same size are skipped, so an interrupted
    gather can be rerun. Returns the destination paths of every file this
    gather covers.
    """
    left_out = {_key(path) for plan in plans for path in plan["drop"]}
    covered = set()
    copied = skipped = 0
    for top in tops:
        for path in sorted(Path(top).rglob("*")):
            if not path.is_file() or _skipped(path) or _key(path) in left_out:
                continue
            out = destination / top.name / path.relative_to(top)
            covered.add(_key(out))
            if out.is_file() and out.stat().st_size == path.stat().st_size:
                skipped += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            copied += 1
        print(f"  gathered {top.name}", flush=True)
    print(f"copied {copied} files, {skipped} already there")
    return covered


def write_zip(destination):
    """Zip destination as <destination>.zip, with the folder itself at the top.

    The archive is written under a temporary name, every entry's checksum is
    tested, and only then is it renamed into place.
    """
    archive = destination.with_name(destination.name + ".zip")
    partial = archive.with_name(archive.name + ".partial")
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    with zipfile.ZipFile(partial, "w", allowZip64=True) as bundle:
        for path in files:
            name = (Path(destination.name) / path.relative_to(destination)).as_posix()
            if path.suffix.lower() in STORED_SUFFIXES:
                bundle.write(path, name, compress_type=zipfile.ZIP_STORED)
            else:
                bundle.write(path, name, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=6)
    with zipfile.ZipFile(partial) as bundle:
        broken = bundle.testzip()
        entries = len(bundle.namelist())
    if broken is not None or entries != len(files):
        raise RuntimeError(f"{partial} failed its check ({broken or entries}); kept for inspection")
    os.replace(partial, archive)
    return archive, len(files)


def _inside(path, folder):
    path, folder = Path(path).resolve(), Path(folder).resolve()
    return path == folder or folder in path.parents


def report(top, plans):
    """One line per top folder, plus any folder that needs attention."""
    files = sum(len(plan["files"]) for plan in plans)
    kept = sum(len(plan["keep"]) for plan in plans)
    kept_bytes = sum(path.stat().st_size for plan in plans for path in plan["keep"])
    left_bytes = sum(path.stat().st_size for plan in plans for path in plan["drop"])
    checks = sum(len(plan["checks"]) for plan in plans)
    failed = sum(len(plan["failed"]) for plan in plans)
    counted = sum(plan["counted"] for plan in plans)
    line = (f"{top.name}: {len(plans)} folders, {files} snapshots; keep {kept} "
            f"({_format_size(kept_bytes)}), leave out {files - kept} "
            f"({_format_size(left_bytes)})")
    if checks:
        line += f"; replay {checks - failed}/{checks} exact"
    if counted:
        line += f"; recorded {counted} patch counts"
    print(line, flush=True)
    for plan in plans:
        name = plan["folder"].relative_to(top) if plan["folder"] != top else Path(".")
        if plan["unrecordable"]:
            print(f"    {name}: kept {plan['unrecordable']} snapshots its record does not track")
        for failure in plan["failed"]:
            print(f"    {name}: kept whole, replay differs at {failure}")
    return kept_bytes, left_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Keep one snapshot per million time units (and each folder's "
            "newest) in checkpoint folders, in place or in a thinned copy."
        )
    )
    parser.add_argument(
        "directories", nargs="+",
        help="run, batch or capture folders, or quoted patterns such as \"checkpoints/*\"",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true",
                      help="delete the left-out snapshots in place")
    mode.add_argument("--into", type=Path, metavar="FOLDER",
                      help="copy the folders, thinned, into FOLDER; the sources are untouched")
    parser.add_argument("--zip", action="store_true",
                        help="with --into, also write FOLDER.zip")
    parser.add_argument("--every", type=int, default=EVERY,
                        help="keep the first snapshot of each block of this many time units")
    parser.add_argument("--verify", type=int, default=1, metavar="N",
                        help="replay checks per folder before anything is left out (0: none)")
    parser.add_argument("--workers", type=int, default=8,
                        help="processes for the replay checks and threads for counting")
    arguments = parser.parse_args(argv)
    if arguments.zip and arguments.into is None:
        parser.error("--zip needs --into")
    if arguments.every <= 0:
        parser.error("--every must be positive")

    tops = resolve_directories(arguments.directories)
    if not tops:
        print("no folders to process", file=sys.stderr)
        return 1
    names = [top.name.casefold() for top in tops]
    if len(set(names)) != len(names):
        parser.error("two folders share a name; pass their contents instead, "
                     "for example \"checkpoints/*\"")
    if arguments.into is not None:
        for top in tops:
            if _inside(arguments.into, top) or _inside(top, arguments.into):
                parser.error(f"--into must lie outside {top}, and the other way round")

    plans_by_top = {}
    for top in tops:
        plans = [plan_folder(folder, arguments.every) for folder in snapshot_folders(top)]
        for plan in plans:
            complete_record(plan, arguments.workers)
            plan["checks"] = choose_checks(plan, arguments.verify)
        plans_by_top[top] = plans
    run_checks([plan for plans in plans_by_top.values() for plan in plans],
               arguments.workers)

    total_kept = total_left = 0
    for top, plans in plans_by_top.items():
        kept, left = report(top, plans)
        total_kept += kept
        total_left += left
    print(f"snapshots kept {_format_size(total_kept)}, left out {_format_size(total_left)}")
    all_plans = [plan for plans in plans_by_top.values() for plan in plans]

    if arguments.apply:
        for plan in all_plans:
            for path in plan["drop"]:
                path.unlink(missing_ok=True)
        print(f"deleted {sum(len(plan['drop']) for plan in all_plans)} snapshots")
    elif arguments.into is not None:
        destination = arguments.into
        destination.mkdir(parents=True, exist_ok=True)
        covered = gather(tops, all_plans, destination)
        extra = [path for path in destination.rglob("*")
                 if path.is_file() and _key(path) not in covered]
        if extra:
            print(f"{len(extra)} files in {destination} are not part of this gather, "
                  f"for example {extra[0]}; remove them or gather into a new folder")
            return 1
        print(f"gathered into {destination}")
        if arguments.zip:
            archive, entries = write_zip(destination)
            print(f"wrote {archive}: {entries} files, {_format_size(archive.stat().st_size)}, "
                  f"every checksum tested")
    else:
        print("report only; nothing was deleted or copied. Use --apply to thin "
              "in place, or --into FOLDER to make a thinned copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
