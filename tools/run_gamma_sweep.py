"""Run the requested 200x200 percolation sweep from one shared checkpoint.

Inspect the setup without starting simulations or writing files::

    python tools/run_gamma_sweep.py --dry-run

Run 12 gamma values, two independent branches each, for 50M additional units::

    python tools/run_gamma_sweep.py

Resume the same saved sweep after stopping it or restarting the computer::

    python tools/run_gamma_sweep.py --resume checkpoints/my_gamma_sweep_01

The process pool uses CPUs. The guarded entry point supports Windows spawn.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sim_modules as MS  # noqa: E402


GAMMAS = (0.069, 0.07, 0.071, 0.072, 0.073,
          0.09, 0.092, 0.094, 0.096, 0.098, 0.1, 0.12)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "gamma_07_alpha_0125_perc_05"
    / "checkpoint_000020000000.npz"
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume an existing sweep folder with its saved settings and seeds")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--workers", type=positive_int, default=None,
                        help="concurrent CPU processes (default: up to 6)")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="new batch folder (default: uniquely named under checkpoints/)")
    parser.add_argument("--timesteps", type=positive_int, default=50_000_000,
                        help="additional model time units per job (default: 50,000,000)")
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--track-every", type=positive_int, default=10_000)
    parser.add_argument("--base-seed", type=nonnegative_int, default=67)
    parser.add_argument("--dry-run", action="store_true",
                        help="read and inspect the starting state; do not run or write files")
    args = parser.parse_args(argv)
    if args.resume is not None:
        new_run_options = {
            "--checkpoint", "--output-root", "--timesteps", "--repeats",
            "--track-every", "--base-seed",
        }
        if any(argument.split("=", 1)[0] in new_run_options for argument in argv):
            parser.error("--resume uses saved settings; only --workers and --dry-run may override it")
    return args


def write_json(path, data):
    """Replace the result index atomically so an interrupted write is harmless."""
    descriptor, name = tempfile.mkstemp(prefix=".sweep_", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_results(output_root, results):
    records = []
    for result in results:
        metadata = result["batch_metadata"]
        records.append({
            "gamma": metadata["config"]["gamma"],
            "repeat": metadata["repeat_index"] + 1,
            "seed": metadata["seed"],
            "checkpoint_dir": metadata["checkpoint_dir"],
            "timestep": int(result["timestep"]),
            "final_diversity": int(result["diversity"]),
        })
    write_json(output_root / "sweep_results.json", records)


def prepare_resume(args):
    """Read the saved plan; CLI defaults never replace the original settings."""
    root = args.resume.expanduser().resolve()
    summary = json.loads((root / "sweep_setup.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "batch_plan.json").read_text(encoding="utf-8"))
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = min(args.workers or min(6, cpu_count), len(plan["jobs"]))
    statuses = []
    needs_initial_state = False
    for metadata in plan["jobs"]:
        directory = Path(metadata["checkpoint_dir"])
        candidates = list(directory.glob("checkpoint_*.npz"))
        if candidates:
            state = MS.load_checkpoint(directory)
            timestep = int(state["timestep"])
            target = int(state["target_timestep"])
            status = "complete" if timestep == target else "resume"
            if status == "complete":
                analysis = MS.load_analysis(directory)
                if analysis is None or (
                    int(analysis["timestep"]) != timestep
                    or not np.array_equal(analysis["tracked_timesteps"], state["tracked_timesteps"])
                    or not np.array_equal(analysis["diversity_history"], state["diversity_history"])
                    or analysis["patch_history"][-1] < 0
                ):
                    status = "finalize"
        else:
            timestep = int(summary["start_timestep"])
            status = "not_started"
            needs_initial_state = plan.get("initial_state") is not None
        statuses.append({
            "job_index": metadata["job_index"], "gamma": metadata["config"]["gamma"],
            "repeat": metadata["repeat_index"] + 1, "status": status,
            "timestep": timestep,
        })
    initial_state = None
    if needs_initial_state:
        saved_initial = summary.get("initial_checkpoint")
        if saved_initial:
            initial_state = MS.load_checkpoint(root / saved_initial)
        else:
            # Compatibility with sweeps created before the resume command.
            initial_state = MS.keep_largest_cluster(summary["source_checkpoint"])
        if int(initial_state["timestep"]) != int(summary["start_timestep"]):
            raise ValueError("saved starting state does not match this sweep's start timestep")
    summary = {
        **summary, "checkpoint_root": str(root), "workers": workers,
        "resume": True, "job_statuses": statuses,
        "already_complete": sum(job["status"] == "complete" for job in statuses),
        "unfinished": sum(job["status"] != "complete" for job in statuses),
    }
    return initial_state, root, summary


def prepare_sweep(args):
    """Read and prune once so every independent branch starts identically."""
    checkpoint = args.checkpoint.expanduser().resolve()
    source = MS.load_checkpoint(checkpoint)
    if source["lattice"].shape != (200, 200):
        raise ValueError("this sweep requires a 200x200 checkpoint")
    state = MS.keep_largest_cluster(source)
    sites = state["lattice"].size
    blocked_before = int(np.count_nonzero(source["lattice"] == -1))
    blocked_after = int(np.count_nonzero(state["lattice"] == -1))
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = args.workers if args.workers is not None else min(6, cpu_count)
    workers = min(workers, len(GAMMAS) * args.repeats)
    output_root = args.output_root
    if output_root is None:
        label = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (
            PROJECT_ROOT / "checkpoints"
            / f"gamma_sweep_perc05_largest_{label}_{uuid4().hex[:8]}"
        )
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"choose a new --output-root; already exists: {output_root}")
    alpha = float(source["alpha"])
    configs = [MS.SimulationConfig(
        L_col=200, L_row=200, D=len(state["current_species"]),
        gamma=gamma, alpha=alpha, T=args.timesteps, p=0.05,
        track_every=args.track_every, seed=None, progress=False,
        populate_first_100=False,
    ) for gamma in GAMMAS]
    summary = {
        "source_checkpoint": str(checkpoint),
        "gammas": list(GAMMAS),
        "repeats": args.repeats,
        "jobs": len(configs) * args.repeats,
        "workers": workers,
        "base_seed": args.base_seed,
        "alpha": alpha,
        "nominal_p": 0.05,
        "lattice_shape": [200, 200],
        "blocked_sites_before": blocked_before,
        "blocked_sites_after": blocked_after,
        "additional_blocked_sites": blocked_after - blocked_before,
        "effective_p": blocked_after / sites,
        "initial_species": len(state["current_species"]),
        "start_timestep": int(state["timestep"]),
        "additional_timesteps": args.timesteps,
        "target_timestep": int(state["timestep"]) + args.timesteps,
        "track_every": args.track_every,
        "checkpoint_root": str(output_root),
    }
    return state, configs, output_root, summary


def main(argv=None):
    args = parse_args(argv)
    if args.resume is not None:
        state, output_root, summary = prepare_resume(args)
        print(json.dumps(summary, indent=2), flush=True)
        if args.dry_run:
            print("Dry run: no simulations resumed and no files written.")
            return 0
        try:
            with MS._simulation_directory_lock(output_root, filename=".sweep.lock"):
                results = MS.resume_simulations(
                    output_root, max_workers=summary["workers"], initial_state=state,
                )
                write_results(output_root, results)
        except KeyboardInterrupt:
            print_pause_command(output_root)
            return 130
        print(f"Sweep complete: {len(results)} jobs. Results: {output_root}", flush=True)
        return 0

    state, configs, output_root, summary = prepare_sweep(args)
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: no simulations started and no files written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        with MS._simulation_directory_lock(output_root, filename=".sweep.lock"):
            # Keep the common starting lattice locally for jobs that were still
            # queued when the PC stopped, even if the original source is moved.
            prepared = {
                **state, "rng_state": None, "target_timestep": summary["target_timestep"],
                "track_every": args.track_every, "populate_first_100": False,
                "p": 0.05, "p_applied": False,
            }
            initial_checkpoint = MS._write_checkpoint(output_root / "initial_state", prepared)
            summary["initial_checkpoint"] = str(Path(initial_checkpoint).relative_to(output_root))
            write_json(output_root / "sweep_setup.json", summary)
            print(f"Starting {summary['jobs']} independent jobs on {summary['workers']} CPU processes.",
                  flush=True)
            print(f"Resume this sweep with --resume \"{output_root}\"", flush=True)
            results = MS.run_simulations(
                configs, kind="percolation", repeats=args.repeats,
                base_seed=args.base_seed, max_workers=summary["workers"],
                initial_state=state, checkpoint_root=output_root,
            )
            write_results(output_root, results)
    except KeyboardInterrupt:
        print_pause_command(output_root)
        return 130
    print(f"Completed {len(results)} jobs. Settings, seeds and results: {output_root}",
          flush=True)
    return 0


def print_pause_command(output_root, script="tools/run_gamma_sweep.py"):
    print("\nSweep stopped. Saved checkpoints are retained; unfinished chunks will be repeated.",
          flush=True)
    print(f'Resume: python {script} --resume "{output_root}"', flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
