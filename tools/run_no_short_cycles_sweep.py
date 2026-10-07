"""Fresh largest-cluster percolation sweep rejecting directed cycles 2--6.

Preview: python tools/run_no_short_cycles_sweep.py --dry-run
Run on the 32-core machine: python tools/run_no_short_cycles_sweep.py --workers 24
Resume: python tools/run_no_short_cycles_sweep.py --resume checkpoints/my_sweep
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sim_modules as MS  # noqa: E402
from tools.run_gamma_sweep import (  # noqa: E402
    GAMMAS, nonnegative_int, positive_int, prepare_resume,
    print_pause_command, write_json, write_results,
)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=positive_int,
                        help="concurrent CPU processes (default: up to 24 available CPUs)")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--timesteps", type=positive_int, default=50_000_000)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--track-every", type=positive_int, default=10_000)
    parser.add_argument("--base-seed", type=nonnegative_int, default=67)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.resume is not None and any(argument.split("=", 1)[0] in {
        "--output-root", "--timesteps", "--repeats", "--track-every", "--base-seed",
    } for argument in argv):
        parser.error("--resume uses saved settings; only --workers and --dry-run may override it")
    return args


def prepare_sweep(args):
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = min(args.workers or min(24, cpu_count), len(GAMMAS) * args.repeats)
    root = args.output_root
    if root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = PROJECT_ROOT / "checkpoints" / f"no_short_cycles_perc05_{stamp}_{uuid4().hex[:8]}"
    root = root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"choose a new --output-root; already exists: {root}")
    configs = [MS.SimulationConfig(
        L_col=200, L_row=200, D=1, gamma=gamma, alpha=0.0125,
        T=args.timesteps, track_every=args.track_every, p=0.05,
        largest_cluster_only=True, max_forbidden_cycle_length=6,
        max_cycle_rejection_attempts=1_000_000, populate_first_100=False,
        progress=False,
    ) for gamma in GAMMAS]
    summary = {
        "simulation": "percolation_without_short_directed_cycles",
        "fresh_start": True, "gammas": list(GAMMAS), "repeats": args.repeats,
        "jobs": len(configs) * args.repeats, "workers": workers,
        "base_seed": args.base_seed, "alpha": 0.0125, "nominal_p": 0.05,
        "lattice_shape": [200, 200], "initial_species": 1,
        "largest_cluster_only": True, "max_forbidden_cycle_length": 6,
        "max_cycle_rejection_attempts": 1_000_000,
        "start_timestep": 0, "additional_timesteps": args.timesteps,
        "target_timestep": args.timesteps, "track_every": args.track_every,
        "checkpoint_root": str(root),
    }
    return configs, root, summary


def main(argv=None):
    args = parse_args(argv)
    if args.resume is not None:
        if args.workers is None:
            cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
            args.workers = min(24, cpu_count)
        state, root, summary = prepare_resume(args)
        if summary.get("simulation") != "percolation_without_short_directed_cycles":
            raise ValueError("this command resumes only the no-short-cycles sweep")
        configs = None
    else:
        configs, root, summary = prepare_sweep(args)
        state = None
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: no simulations started and no files written.")
        return 0
    if args.resume is None:
        root.mkdir(parents=True, exist_ok=False)
    try:
        with MS._simulation_directory_lock(root, filename=".sweep.lock"):
            if args.resume is None:
                write_json(root / "sweep_setup.json", summary)
            print(f"Running {summary['jobs']} jobs with up to {summary['workers']} CPU workers.",
                  flush=True)
            print(f'Resume: python tools/run_no_short_cycles_sweep.py --resume "{root}"', flush=True)
            if args.resume is not None:
                results = MS.resume_simulations(root, max_workers=summary["workers"],
                                                initial_state=state)
            else:
                results = MS.run_simulations(
                    configs, kind="percolation", repeats=args.repeats,
                    base_seed=args.base_seed, max_workers=summary["workers"],
                    checkpoint_root=root,
                )
            write_results(root, results)
    except KeyboardInterrupt:
        print_pause_command(root, script="tools/run_no_short_cycles_sweep.py")
        return 130
    print(f"Sweep complete: {len(results)} jobs. Results: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
