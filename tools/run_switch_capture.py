"""Continue one run, keeping every state, until it has risen to and fallen from high diversity.

The default continues the gamma = 0.03 control of the gamma swap in analysis/sim_analysis.ipynb
(p = 0.3, alpha = 0.0125, largest cluster only, directed cycles of length 2-6 rejected) from
its last state at t = 150M. The saved RNG is restored, so this is the control's own
trajectory, recorded every 1,000 steps instead of every 10,000. Every state is kept.

The run stops --after-down steps after it has first risen to the high regime and then fallen
back to the low regime. Regimes follow analysis/sim_analysis.ipynb: living species on its
10,000-step grid, Gaussian-smoothed over 0.2M steps, high from reaching 50 until falling to 10.

Preview: python tools/run_switch_capture.py --dry-run
Run:     python tools/run_switch_capture.py
Resume:  python tools/run_switch_capture.py --resume E:/MasterProject/checkpoints/<folder>
"""

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sim_modules as MS  # noqa: E402
from tools.run_gamma_sweep import nonnegative_int, positive_int, write_json  # noqa: E402


DEFAULT_SOURCE = (PROJECT_ROOT / "checkpoints"
                  / "no_short_cycles_gamma_p30_from_g003_20261022_005553_5940261e"
                  / "job_00001_seed_56052066336814670902865051663650190248")
DEFAULT_PARENT = Path("E:/MasterProject/checkpoints")
PLAN, PROGRESS, SERIES = "capture_plan.json", "progress.json", "series.npz"
# The regime rule of analysis/sim_analysis.ipynb, judged on its sample grid. The plan keeps a
# copy, so a capture is always judged by the rule it started with.
REGIME_RULE = {"low": 10, "high": 50, "sample_every": 10_000, "smooth_sigma": 20}
TRUNCATE = 4.0  # MS._gaussian_smooth's kernel reach, in widths.
# Options that only decide when the capture stops; --resume may change them.
STOP_OPTIONS = ("after_down", "max_steps", "min_free_gb")


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", type=Path, help="continue an existing capture folder")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="checkpoint file, or a run folder whose newest checkpoint is used")
    parser.add_argument("--output-root", type=Path,
                        help=f"new capture folder (default: a fresh folder in {DEFAULT_PARENT})")
    parser.add_argument("--track-every", type=positive_int, default=1_000,
                        help="steps between kept states (default: 1,000)")
    parser.add_argument("--segment", type=positive_int, default=1_000_000,
                        help="steps per segment folder (default: 1,000,000)")
    parser.add_argument("--after-down", type=nonnegative_int, default=2_000_000,
                        help="steps kept after the collapse (default: 2,000,000)")
    parser.add_argument("--max-steps", type=positive_int, default=200_000_000,
                        help="stop after this many steps without both switches (default: 200M)")
    parser.add_argument("--min-free-gb", type=float, default=20.0,
                        help="pause before a segment when the drive has less free space")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    given = {argument.split("=", 1)[0] for argument in argv}
    if args.resume is not None and given & {"--source", "--output-root", "--track-every",
                                            "--segment"}:
        parser.error("--resume keeps the saved source, tracking and segments; only --after-down, "
                     "--max-steps, --min-free-gb and --dry-run may change")
    args.overrides = {name: getattr(args, name) for name in STOP_OPTIONS
                      if "--" + name.replace("_", "-") in given}
    return args


def regime_switches(timesteps, diversity, rule):
    """Regime switches of a living-species series, each marked settled or not.

    Samples on the rule's grid are smoothed, then walked: the run switches up on reaching
    ``high`` and down on falling to ``low``, and its start regime comes from the raw first
    sample, as in analysis/sim_analysis.ipynb. Smoothing reads TRUNCATE widths ahead, so a
    switch closer than that to the newest sample can still move or vanish as the run goes on.
    """
    timesteps = np.asarray(timesteps, dtype=np.int64)
    on_grid = (timesteps - timesteps[0]) % rule["sample_every"] == 0
    times = timesteps[on_grid]
    values = np.asarray(diversity, dtype=np.float64)[on_grid]
    smoothed = MS._gaussian_smooth(values, rule["smooth_sigma"], truncate=TRUNCATE)
    reach = max(int(TRUNCATE * rule["smooth_sigma"] + 0.5), 1)
    high = bool(values[0] >= 0.5 * (rule["low"] + rule["high"]))
    switches = []
    for index, value in enumerate(smoothed):
        if (high and value <= rule["low"]) or (not high and value >= rule["high"]):
            high = not high
            switches.append({"timestep": int(times[index]), "up": high,
                             "settled": index < times.size - reach})
    return switches, float(smoothed[-1]), high


def capture_status(timesteps, diversity, plan):
    """The first rise, the first collapse after it, and the time the capture may stop."""
    switches, smoothed, high = regime_switches(timesteps, diversity, plan["regime_rule"])
    rise = next((switch for switch in switches if switch["up"]), None)
    collapse = None if rise is None else next(
        (switch for switch in switches
         if not switch["up"] and switch["timestep"] > rise["timestep"]), None)
    stop_at = None
    if rise and rise["settled"] and collapse and collapse["settled"]:
        stop_at = collapse["timestep"] + plan["after_down"]
    return {"rise": rise, "collapse": collapse, "stop_at": stop_at, "smoothed": smoothed,
            "regime": "high" if high else "low"}


def check_steps(plan):
    """Keep every boundary on the tracking grid, and the tracking grid on the rule's grid."""
    track = plan["track_every"]
    if plan["regime_rule"]["sample_every"] % track:
        raise ValueError(f"--track-every must divide {plan['regime_rule']['sample_every']:,}, "
                         "the spacing the regime rule is judged on")
    for name in ("segment", "after_down", "max_steps"):
        if plan[name] % track:
            raise ValueError(f"--{name.replace('_', '-')} must be a multiple of --track-every")


def new_capture(args):
    state = MS.load_checkpoint(args.source.expanduser().resolve())
    checkpoint = Path(state["checkpoint_path"])
    root = args.output_root
    if root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = DEFAULT_PARENT / (f"switches_p{state.get('p', 0.0):g}_a{state['alpha']:g}"
                                 f"_g{state['gamma']:g}_{stamp}_{uuid4().hex[:8]}")
    root = root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"choose a new --output-root; already exists: {root}")
    plan = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint),
        "source_fingerprint": MS._branch_state_fingerprint(state),
        "initial_state": str(Path("initial_state") / checkpoint.name),
        "rng": "restored from the source checkpoint, so its own trajectory continues",
        "start": {"timestep": state["timestep"], "living_species": state["diversity"],
                  "patches": MS.count_patches(state["lattice"])},
        "simulation": {
            "gamma": state["gamma"], "alpha": state["alpha"], "p": state.get("p", 0.0),
            "lattice_shape": list(state["lattice"].shape),
            "usable_sites": state.get("usable_sites"),
            "largest_cluster_only": bool(state.get("largest_cluster_only", True)),
            "max_forbidden_cycle_length": state.get("max_forbidden_cycle_length"),
            "max_cycle_rejection_attempts": state.get("max_cycle_rejection_attempts", 1_000_000),
        },
        "track_every": args.track_every, "segment": args.segment,
        "after_down": args.after_down, "max_steps": args.max_steps,
        "min_free_gb": args.min_free_gb, "regime_rule": dict(REGIME_RULE),
        "output_root": str(root),
    }
    check_steps(plan)
    return plan, root, state


def simulation_config(plan):
    settings = plan["simulation"]
    rows, columns = settings["lattice_shape"]
    return MS.SimulationConfig(
        L_col=columns, L_row=rows, D=1, gamma=settings["gamma"], alpha=settings["alpha"], T=0,
        track_every=plan["track_every"], progress=True, p=settings["p"],
        largest_cluster_only=settings["largest_cluster_only"],
        max_forbidden_cycle_length=settings["max_forbidden_cycle_length"],
        max_cycle_rejection_attempts=settings["max_cycle_rejection_attempts"],
        retention=MS.RetentionPolicy(keep_all=True),
    )


def segment_dirs(root):
    return sorted((root / "segments").glob("segment_*"))


def has_states(directory):
    return any(directory.glob("checkpoint_*.npz"))


def finish_last_segment(root, config):
    """Complete a segment an interruption left short of its target, or behind its record.

    Resume restores the newest state's RNG, so the repeated interval follows the same random
    sequence; with nothing left to run it only rewrites the analysis record.
    """
    segments = segment_dirs(root)
    if not segments or not has_states(segments[-1]):
        return
    newest = MS.load_checkpoint(segments[-1])
    record = MS.load_analysis(segments[-1])
    behind = record is None or int(record["timestep"]) < newest["timestep"]
    if newest["timestep"] < newest["target_timestep"] or behind:
        print(f"Finishing {segments[-1].name} from t = {newest['timestep'] / 1e6:.3f}M.",
              flush=True)
        replace(config, checkpoint_dir=str(segments[-1])).resume()


def load_series(root, plan):
    """Living species and patches of the whole capture, from its segments' analysis records.

    Each segment starts on the previous one's last state, so only the first segment's starting
    sample is kept, and the plan holds that one.
    """
    start = plan["start"]
    times, diversity, patches = [[start["timestep"]]], [[start["living_species"]]], [[start["patches"]]]
    owner = [[-1]]
    for index, directory in enumerate(segment_dirs(root)):
        record = MS.load_analysis(directory)
        if record is None:
            continue
        segment_times = np.asarray(record["tracked_timesteps"], dtype=np.int64)
        if segment_times[0] != times[-1][-1]:
            raise ValueError(f"{directory.name} starts at t = {segment_times[0]:,}, not where the "
                             f"capture stands (t = {times[-1][-1]:,})")
        times.append(segment_times[1:])
        diversity.append(np.asarray(record["diversity_history"], dtype=np.int64)[1:])
        patches.append(np.asarray(record["patch_history"], dtype=np.int64)[1:])
        owner.append(np.full(segment_times.size - 1, index))
    return {"timesteps": np.concatenate(times), "living_species": np.concatenate(diversity),
            "patches": np.concatenate(patches), "segment": np.concatenate(owner)}


def write_npz(path, **arrays):
    """Replace a series file atomically, so a reader never sees half of one."""
    descriptor, name = tempfile.mkstemp(prefix=".series_", suffix=".tmp.npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def folder_bytes(root):
    return sum(entry.stat().st_size for entry in root.rglob("*") if entry.is_file())


def describe(switch):
    if switch is None:
        return "not yet"
    return f"{switch['timestep'] / 1e6:.2f}M" + ("" if switch["settled"] else " (provisional)")


def start_next_segment(root, plan, config, status):
    segments = segment_dirs(root)
    # A segment that was started but never saved a state is started again in its own folder.
    index = len(segments) - 1 if segments and not has_states(segments[-1]) else len(segments)
    begin = MS.load_checkpoint(segments[index - 1] if index else root / plan["initial_state"])
    end = plan["start"]["timestep"] + plan["max_steps"]
    if status["stop_at"] is not None:
        end = min(end, status["stop_at"])
    steps = min(plan["segment"], end - begin["timestep"])
    directory = root / "segments" / f"segment_{index:04d}_t{begin['timestep']:012d}"
    replace(config, T=steps, checkpoint_dir=str(directory)).run_percolation(initial_state=begin)


def capture(root, plan):
    """Run segments until the capture is complete or a limit stops it; return the outcome."""
    config = simulation_config(plan)
    start = plan["start"]["timestep"]
    session = None
    while True:
        finish_last_segment(root, config)
        series = load_series(root, plan)
        timesteps, diversity = series["timesteps"], series["living_species"]
        now = int(timesteps[-1])
        status = capture_status(timesteps, diversity, plan)
        free = shutil.disk_usage(root).free
        if status["stop_at"] is not None and now >= status["stop_at"]:
            outcome = "complete"
        elif now - start >= plan["max_steps"]:
            outcome = "stopped: reached --max-steps"
        elif free < plan["min_free_gb"] * 1e9:
            outcome = "stopped: drive below --min-free-gb"
        else:
            outcome = "running"
        used = folder_bytes(root)
        write_npz(root / SERIES, **series)
        write_json(root / PROGRESS, {
            "status": outcome, "updated": datetime.now().isoformat(timespec="seconds"),
            "timestep": now, "steps_since_start": now - start,
            "living_species": int(diversity[-1]), "smoothed": round(status["smoothed"], 2),
            "regime": status["regime"], "rise": status["rise"], "collapse": status["collapse"],
            "stop_at": status["stop_at"], "segments": len(segment_dirs(root)),
            "kept_states": int(timesteps.size - 1), "disk_bytes": used, "free_bytes": free,
        })
        if session is None:
            session = (time.monotonic(), now)
        hours = (time.monotonic() - session[0]) / 3600
        rate = f" | {(now - session[1]) / 1e6 / hours:.1f}M steps/h" if now > session[1] else ""
        print(f"t = {now / 1e6:.2f}M (+{(now - start) / 1e6:.2f}M) | {int(diversity[-1])} species, "
              f"smoothed {status['smoothed']:.1f} ({status['regime']}) | rise: "
              f"{describe(status['rise'])} | collapse: {describe(status['collapse'])} | "
              f"{used / 1e9:.2f} GB, {free / 1e9:.0f} GB free{rate}", flush=True)
        if outcome != "running":
            return outcome, status
        start_next_segment(root, plan, config, status)


def dry_run_report(plan, state):
    """Disk per million steps, and how long the source run's own regimes lasted."""
    source = Path(plan["source_checkpoint"])
    history_bytes = 3 * 8 * state["tracked_timesteps"].size
    average_history = 3 * 8 * plan["segment"] // plan["track_every"] // 2
    per_state = source.stat().st_size - history_bytes + average_history
    per_million = per_state * 1_000_000 / plan["track_every"]
    print(f"About {per_state / 1e3:.0f} KB per kept state: {per_million / 1e9:.2f} GB per 1M steps.")
    record = MS.load_analysis(source.parent)
    if record is not None:
        switches, _, _ = regime_switches(record["tracked_timesteps"], record["diversity_history"],
                                         plan["regime_rule"])
        lows = [(after["timestep"] - before["timestep"]) / 1e6
                for before, after in zip(switches, switches[1:]) if after["up"]]
        highs = [(after["timestep"] - before["timestep"]) / 1e6
                 for before, after in zip(switches, switches[1:]) if not after["up"]]
        span = (f"{record['tracked_timesteps'][0] / 1e6:g}-"
                f"{record['tracked_timesteps'][-1] / 1e6:g}M")
        print(f"Source run {span}, same rule: {len(switches)} switches; low spells "
              f"{', '.join(f'{value:.1f}' for value in lows) or '-'}M steps; high spells "
              f"{', '.join(f'{value:.1f}' for value in highs) or '-'}M steps.")
    print(f"Destination drive: {shutil.disk_usage(Path(plan['output_root']).anchor).free / 1e9:.0f}"
          " GB free.")


def main(argv=None):
    args = parse_args(argv)
    if args.resume is not None:
        root = args.resume.expanduser().resolve()
        plan = json.loads((root / PLAN).read_text(encoding="utf-8"))
        plan.update(args.overrides)
        check_steps(plan)
        initial = MS.load_checkpoint(root / plan["initial_state"])
        if MS._branch_state_fingerprint(initial) != plan["source_fingerprint"]:
            raise ValueError(f"{plan['initial_state']} no longer matches the source state")
        state = None
    else:
        plan, root, state = new_capture(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        if state is not None:
            dry_run_report(plan, state)
        print("Dry run: no simulation started and no files written.")
        return 0
    if args.resume is None:
        (root / "initial_state").mkdir(parents=True)
        shutil.copy2(plan["source_checkpoint"], root / plan["initial_state"])
    try:
        with MS._simulation_directory_lock(root, filename=".capture.lock"):
            # Saved before running, so a resumed capture keeps any changed stop options.
            write_json(root / PLAN, plan)
            print(f'Capture folder: {root}\nResume: python tools/run_switch_capture.py --resume '
                  f'"{root}"', flush=True)
            outcome, status = capture(root, plan)
    except KeyboardInterrupt:
        print("\nCapture paused. Kept states stay on disk; the unfinished interval repeats with "
              "the same random sequence.", flush=True)
        print(f'Resume: python tools/run_switch_capture.py --resume "{root}"', flush=True)
        return 130
    if outcome != "complete":
        print(f"Capture {outcome}. Resume with a higher limit or more free space: "
              f'python tools/run_switch_capture.py --resume "{root}"', flush=True)
        return 2
    print(f"Capture complete: rise at {describe(status['rise'])}, collapse at "
          f"{describe(status['collapse'])}. Series: {root / SERIES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
