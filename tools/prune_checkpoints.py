"""Shrink existing checkpoint directories to an analysis record plus the
snapshots that are still worth reading.

A finished run keeps one full lattice per tracking interval, which for a
100M-timestep run at ``track_every=10000`` is 10,000 files of 0.33 MB. Almost
all of that is only ever read to recount patches for a plot. This tool counts
the patches once, writes them into the run's ``analysis.npz`` alongside the
living-species series, and then deletes every snapshot except the newest one
and the ones surrounding a large swap in living species.

It reports what it would delete and changes nothing until ``--apply`` is
passed. The analysis record is always written before anything is deleted, so
an interrupted run of this tool can lose files but never the counts.

    python tools/prune_checkpoints.py checkpoints/*            # dry run
    python tools/prune_checkpoints.py checkpoints/* --apply    # delete
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob as globbing
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sim_modules as MS  # noqa: E402


def resolve_directories(patterns):
    """Expand the command line into run directories.

    The shell is not relied on to do this. cmd.exe does no globbing at all,
    and PowerShell does none for a native command, so ``checkpoints/*``
    arrives here as a literal string on Windows and as six already-expanded
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


def _count_file_patches(path):
    """Return one snapshot's timestep and patch count."""
    with np.load(path, allow_pickle=False) as saved:
        return int(saved["timestep"]), MS.count_patches(saved["lattice"])


def _existing_patch_history(directory, tracked_timesteps):
    """Reuse counts from an earlier migration, so a rerun is not a recount."""
    blank = np.full(
        tracked_timesteps.size, MS._MISSING_PATCH_COUNT, dtype=np.int64
    )
    analysis = MS.load_analysis(directory)
    if analysis is None:
        return blank
    recorded_times = np.asarray(analysis["tracked_timesteps"], dtype=np.int64)
    recorded_patches = np.asarray(analysis["patch_history"], dtype=np.int64)
    if recorded_times.size != recorded_patches.size:
        return blank
    # Counts are matched by timestep rather than position, so a record from a
    # differently tracked leg cannot silently shift the series.
    positions = np.searchsorted(tracked_timesteps, recorded_times)
    inside = positions < tracked_timesteps.size
    positions = positions[inside]
    recorded_times = recorded_times[inside]
    recorded_patches = recorded_patches[inside]
    aligned = tracked_timesteps[positions] == recorded_times
    blank[positions[aligned]] = recorded_patches[aligned]
    return blank


def survey_run(directory, policy, workers=8):
    """Work out what one run directory would keep, without changing it."""
    directory = Path(directory)
    checkpoint_files = sorted(
        directory.glob("checkpoint_*.npz"), key=MS._checkpoint_timestep
    )
    if not checkpoint_files:
        return None

    newest = checkpoint_files[-1]
    state = MS.load_checkpoint(newest)
    tracked_timesteps = np.asarray(state["tracked_timesteps"], dtype=np.int64)
    diversity_history = np.asarray(state["diversity_history"], dtype=np.int64)

    patch_history = _existing_patch_history(directory, tracked_timesteps)

    def sample_index(timestep):
        """Return the tracked sample a timestep belongs to, or -1."""
        position = int(np.searchsorted(tracked_timesteps, timestep))
        if (
            position < tracked_timesteps.size
            and int(tracked_timesteps[position]) == timestep
        ):
            return position
        return -1

    uncounted = [
        path
        for path in checkpoint_files
        if sample_index(MS._checkpoint_timestep(path)) >= 0
        and patch_history[sample_index(MS._checkpoint_timestep(path))] < 0
    ]

    print(f"  counting patches in {len(uncounted)} snapshots...", flush=True)
    if uncounted:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for timestep, patches in pool.map(_count_file_patches, uncounted):
                position = sample_index(timestep)
                if position >= 0:
                    patch_history[position] = patches

    events = MS.detect_diversity_swaps(
        tracked_timesteps, diversity_history, policy
    )

    window = int(policy.event_window)
    newest_timestep = int(tracked_timesteps[-1])
    keep = []
    drop = []
    for path in checkpoint_files:
        timestep = MS._checkpoint_timestep(path)
        position = sample_index(timestep)
        counted = position >= 0 and int(patch_history[position]) >= 0
        near_event = events.size and bool(
            np.any(np.abs(events - timestep) <= window)
        )
        # A snapshot nobody could count stays: deleting it would destroy the
        # only copy of a patch count that is not in the record.
        if not counted or near_event or timestep >= newest_timestep:
            keep.append(path)
        else:
            drop.append(path)

    return {
        "directory": directory,
        "state": state,
        "tracked_timesteps": tracked_timesteps,
        "diversity_history": diversity_history,
        "patch_history": patch_history,
        "events": events,
        "keep": keep,
        "drop": drop,
    }


def report(survey):
    """Print what a survey found for one run directory."""
    events = survey["events"]
    keep = survey["keep"]
    drop = survey["drop"]
    freed = sum(path.stat().st_size for path in drop)
    kept_bytes = sum(path.stat().st_size for path in keep)
    diversity = survey["diversity_history"]

    print(f"  samples:          {survey['tracked_timesteps'].size}")
    print(
        f"  living species:   min {diversity.min()}, "
        f"median {int(np.median(diversity))}, max {diversity.max()}"
    )
    print(f"  swaps detected:   {events.size}")
    for timestep in events[:12]:
        print(f"      at timestep {int(timestep):,}")
    if events.size > 12:
        print(f"      ... and {events.size - 12} more")
    print(f"  keep:             {len(keep)} files ({_format_size(kept_bytes)})")
    print(f"  delete:           {len(drop)} files ({_format_size(freed)})")
    return freed


def write_record(survey):
    """Save the counts. Non-destructive, so a dry run does this too.

    Counting the patches is the slow half of the job, and the record is what
    makes the run plottable once its lattices are gone. Writing it during a
    dry run means the later --apply pass only has to delete.
    """
    state = survey["state"]
    return MS._write_analysis(
        survey["directory"],
        {
            "tracked_timesteps": survey["tracked_timesteps"],
            "diversity_history": survey["diversity_history"],
            "patch_history": survey["patch_history"],
            "event_timesteps": survey["events"],
            "timestep": int(survey["tracked_timesteps"][-1]),
            "target_timestep": int(state["target_timestep"]),
            "gamma": float(state["gamma"]),
            "alpha": float(state["alpha"]),
            "track_every": int(state["track_every"]),
            "populate_first_100": bool(state["populate_first_100"]),
            "lattice_shape": np.asarray(state["lattice"]).shape,
        },
    )


def delete_snapshots(survey):
    """Delete the snapshots the survey judged redundant."""
    for path in survey["drop"]:
        path.unlink(missing_ok=True)
    print(f"  deleted {len(survey['drop'])} files")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Replace a run's redundant lattice snapshots with a small "
            "analysis record, keeping the newest state and the snapshots "
            "around each large swap in living species."
        )
    )
    parser.add_argument(
        "directories", nargs="+", help="checkpoint directories to shrink"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete; without it nothing is changed",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=MS.RetentionPolicy.event_window,
        help="simulation time kept each side of a swap",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=MS.RetentionPolicy.smooth_samples,
        help="moving-average width in samples",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=MS.RetentionPolicy.low_fraction,
        help="low regime threshold, as a fraction of median diversity",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=MS.RetentionPolicy.high_fraction,
        help="high regime threshold, as a fraction of median diversity",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="patch counting threads"
    )
    arguments = parser.parse_args(argv)

    policy = MS.RetentionPolicy(
        event_window=arguments.window,
        smooth_samples=arguments.smooth,
        low_fraction=arguments.low,
        high_fraction=arguments.high,
    )

    directories = resolve_directories(arguments.directories)
    if not directories:
        print("no run directories to process", file=sys.stderr)
        return 1

    total_freed = 0
    surveys = []
    for path in directories:
        print(f"{path}")
        survey = survey_run(path, policy, workers=arguments.workers)
        if survey is None:
            print("  no checkpoints found")
            continue
        total_freed += report(survey)
        write_record(survey)
        print(f"  wrote {MS._ANALYSIS_FILENAME}")
        surveys.append(survey)
        print()

    print(f"total reclaimable: {_format_size(total_freed)}")
    if not arguments.apply:
        print(
            "dry run; the analysis records were saved but no snapshot was "
            "deleted. Re-run with --apply to delete."
        )
        return 0

    for survey in surveys:
        print(f"applying to {survey['directory']}")
        delete_snapshots(survey)
    print(f"freed {_format_size(total_freed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
