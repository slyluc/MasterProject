from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
import multiprocessing
import os
import signal
from pathlib import Path
import tempfile
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:  # Keep the module importable for a helpful runtime error.
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function

        return decorator


@dataclass
class SimulationConfig:
    """Mutable simulation settings for convenient use in notebooks.

    Change any attribute between runs, then call ``run_main()`` or
    ``run_percolation()``. Validation is performed by the simulation function
    when the run starts. Set ``checkpoint_dir`` to enable disk snapshots. A
    saved or previous result can be continued with
    ``run_main(initial_state=state)``. Continuing into the folder the state
    came from extends that run's history; continuing into any other folder
    starts a new history at the state's timestep. Pass ``continue_history``
    to choose explicitly.
    """

    L_col: int
    L_row: int
    D: int
    gamma: float
    alpha: float
    T: int
    track_every: int = 1
    seed: int | None = None
    progress: bool = False
    p: float = 0.0
    # Force one new-species introduction in each of the first 100 model
    # time units, in addition to the ordinary alpha * gamma process.
    populate_first_100: bool = False
    # When set, save an atomic state file after every ``track_every`` interval.
    checkpoint_dir: str | None = None
    # How many of those state files survive. None uses the default policy:
    # the newest state, plus one window either side of each detected swap in
    # living species. Pass RetentionPolicy(keep_all=True) to keep every one.
    retention: "RetentionPolicy | None" = None
    # Keep only the largest periodic four-neighbour usable component at start.
    largest_cluster_only: bool = True
    # Reject new-species link proposals that create a directed cycle of lengths
    # 2 through this inclusive bound. None leaves the original model unchanged.
    max_forbidden_cycle_length: int | None = None
    max_cycle_rejection_attempts: int = 1_000_000

    def run_main(self, initial_state=None, continue_history=None):
        """Run the ordinary simulation using the current settings."""
        return main_simulation(
            L_col=self.L_col,
            L_row=self.L_row,
            D=self.D,
            gamma=self.gamma,
            alpha=self.alpha,
            T=self.T,
            track_every=self.track_every,
            seed=self.seed,
            progress=self.progress,
            populate_first_100=self.populate_first_100,
            initial_state=initial_state,
            checkpoint_dir=self.checkpoint_dir,
            retention=self.retention,
            continue_history=continue_history,
            max_forbidden_cycle_length=self.max_forbidden_cycle_length,
            max_cycle_rejection_attempts=self.max_cycle_rejection_attempts,
        )

    def run_percolation(self, initial_state=None, continue_history=None):
        """Run the percolation simulation using the current settings."""
        return percolation_simulation(
            L_col=self.L_col,
            L_row=self.L_row,
            D=self.D,
            gamma=self.gamma,
            alpha=self.alpha,
            T=self.T,
            p=self.p,
            track_every=self.track_every,
            seed=self.seed,
            progress=self.progress,
            populate_first_100=self.populate_first_100,
            initial_state=initial_state,
            checkpoint_dir=self.checkpoint_dir,
            retention=self.retention,
            continue_history=continue_history,
            largest_cluster_only=self.largest_cluster_only,
            max_forbidden_cycle_length=self.max_forbidden_cycle_length,
            max_cycle_rejection_attempts=self.max_cycle_rejection_attempts,
        )

    def run_many(
        self, repeats=1, *, kind="main", base_seed=None, max_workers=None,
        checkpoint_root=None, initial_state=None,
    ):
        """Run independent repetitions; see :func:`run_simulations`."""
        return run_simulations(
            self, kind=kind, repeats=repeats, base_seed=base_seed,
            max_workers=max_workers, checkpoint_root=checkpoint_root,
            initial_state=initial_state,
        )

    def resume(self, checkpoint=None):
        """Resume the newest checkpoint up to its original target timestep."""
        checkpoint = self.checkpoint_dir if checkpoint is None else checkpoint
        if checkpoint is None:
            raise ValueError("set checkpoint_dir or pass a checkpoint path")
        state = load_checkpoint(checkpoint)
        output_dir = self.checkpoint_dir
        if output_dir is None:
            checkpoint_path = Path(checkpoint).expanduser()
            output_dir = (
                checkpoint_path
                if checkpoint_path.is_dir()
                else checkpoint_path.parent
            )
        target = int(state.get("target_timestep", self.T))
        completed = int(state.get("timestep", 0))
        remaining = target - completed
        if remaining < 0:
            raise ValueError("checkpoint is already beyond its target timestep")
        return simulation_from_state(
            state,
            gamma=self.gamma,
            alpha=self.alpha,
            T=remaining,
            track_every=self.track_every,
            seed=self.seed,
            progress=self.progress,
            populate_first_100=self.populate_first_100,
            checkpoint_dir=output_dir,
            retention=self.retention,
            target_timestep=target,
            continue_history=True,
            max_forbidden_cycle_length=self.max_forbidden_cycle_length,
            max_cycle_rejection_attempts=self.max_cycle_rejection_attempts,
        )


_BATCH_INITIAL_STATE = None


def _batch_settings_instance(value, settings_class):
    """Recognize settings retained by a notebook across reload(sim_modules)."""
    return isinstance(value, settings_class) or (
        is_dataclass(value)
        and type(value).__module__ == settings_class.__module__
        and type(value).__qualname__ == settings_class.__qualname__
    )


def _copy_batch_settings(value, settings_class):
    # Spawn workers must receive instances of the currently imported classes.
    return settings_class(**{
        field.name: getattr(value, field.name) for field in fields(settings_class)
        if hasattr(value, field.name)
    })


def _batch_integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _batch_json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"cannot record batch setting {type(value).__name__}")


def _validate_batch_config(config, kind, initial_state):
    """Reject configuration errors before any potentially long job starts."""
    _validate_simulation_options(
        config.alpha, config.T, config.track_every, config.progress,
        config.populate_first_100,
    )
    _validate_gamma(config.gamma)
    cycle_limit = _validate_cycle_rejection_options(
        config.max_forbidden_cycle_length, config.max_cycle_rejection_attempts
    )
    if initial_state is not None and cycle_limit and _has_forbidden_directed_cycle(
        initial_state["Gamma"], cycle_limit
    ):
        raise ValueError("initial Gamma contains a forbidden directed cycle")
    if config.retention is not None and not isinstance(
        config.retention, RetentionPolicy
    ):
        raise TypeError("retention must be a RetentionPolicy or None")
    if kind == "percolation":
        if not isinstance(config.largest_cluster_only, (bool, np.bool_)):
            raise TypeError("largest_cluster_only must be a boolean")
        if isinstance(config.p, (bool, np.bool_)) or not isinstance(
            config.p, (int, float, np.integer, np.floating)
        ):
            raise TypeError("p must be a real number")
        if not 0 <= config.p <= 1:
            raise ValueError("p must be between 0 and 1")
    if initial_state is None:
        columns = _batch_integer(config.L_col, "L_col", minimum=1)
        rows = _batch_integer(config.L_row, "L_row", minimum=1)
        diversity = _batch_integer(
            config.D, "D", minimum=1 if kind == "main" else 0
        )
        sites = rows * columns
        if diversity > sites:
            raise ValueError("D cannot exceed the number of lattice sites")
        usable_sites = sites if kind == "main" else None
    else:
        sites = initial_state["lattice"].size
        usable_sites = np.count_nonzero(initial_state["lattice"] != -1)
    if cycle_limit and config.gamma == 1:
        if initial_state is None and config.D > 1:
            raise ValueError("gamma=1 cannot initialize multiple species without forbidden cycles")
        start = 0 if initial_state is None else initial_state["timestep"]
        if usable_sites and config.T and (
            config.alpha > 0 or (config.populate_first_100 and start < 100)
        ):
            raise ValueError("gamma=1 cannot support repeated introductions without forbidden cycles")
    if usable_sites and config.alpha * config.gamma / sites > 1:
        raise ValueError("alpha * gamma / N cannot exceed 1")


def _validate_batch_percolation_start(config):
    """Replay only site removal to check its seed-dependent initial capacity."""
    sites = int(config.L_col) * int(config.L_row)
    if config.p == 0:
        usable = sites
    elif config.p == 1:
        usable = 0
    elif config.largest_cluster_only:
        rng = np.random.default_rng(config.seed)
        mask = (rng.random(sites) >= config.p).reshape(
            int(config.L_row), int(config.L_col)
        )
        usable = np.count_nonzero(_largest_usable_cluster_mask(mask))
    else:
        rng = np.random.default_rng(config.seed)
        usable = 0
        # Avoid allocating another complete lattice for this early check.
        for start in range(0, sites, 1_000_000):
            usable += np.count_nonzero(
                rng.random(min(1_000_000, sites - start)) >= config.p
            )
    if usable and config.D == 0:
        raise ValueError("D must be positive when usable sites remain")
    if config.D > usable:
        raise ValueError(f"D={config.D} exceeds the {usable} usable sites")
    if usable and config.alpha * config.gamma / sites > 1:
        raise ValueError("alpha * gamma / N cannot exceed 1")
    if usable and config.max_forbidden_cycle_length is not None and config.gamma == 1 and config.T and (
        config.alpha > 0 or config.populate_first_100
    ):
        raise ValueError("gamma=1 cannot support repeated introductions without forbidden cycles")


def _initialize_simulation_worker(initial_state):
    # Send a potentially large branching state once per process, not per job.
    global _BATCH_INITIAL_STATE
    _BATCH_INITIAL_STATE = initial_state
    # Let the parent handle Ctrl+C and terminate workers. Atomic replacement
    # protects completed snapshots even if a temporary-file write is stopped.
    # Children receiving the same console signal would otherwise each raise
    # KeyboardInterrupt independently.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@contextmanager
def _simulation_directory_lock(directory, filename=".job.lock"):
    """Hold a crash-released advisory lock for a saved simulation directory."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / filename
    with lock_path.open("a+b") as lock_file:
        if lock_file.seek(0, os.SEEK_END) == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"simulation directory is already in use: {directory}; "
                "stop its running process before resuming"
            ) from error
        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _simulation_job_lock(config):
    if config.checkpoint_dir is None:
        yield
    else:
        with _simulation_directory_lock(config.checkpoint_dir):
            yield


def _execute_simulation_job(job, initial_state):
    config, metadata = job
    try:
        run = config.run_main if metadata["kind"] == "main" else config.run_percolation
        with _simulation_job_lock(config):
            result = run(initial_state=initial_state, continue_history=False)
    except Exception as error:
        raise RuntimeError(
            f"simulation job {metadata['job_index']} "
            f"(config {metadata['config_index']}, repeat {metadata['repeat_index']}, "
            f"seed={config.seed}, checkpoint_dir={config.checkpoint_dir!r}) "
            f"failed: {error}"
        ) from error
    result["batch_metadata"] = {**metadata, "worker_pid": os.getpid()}
    return result


def _process_simulation_job(job):
    """Importable entry point required by spawned Windows/notebook workers."""
    return _execute_simulation_job(job, _BATCH_INITIAL_STATE)


def _branch_state_fingerprint(state):
    """Identify the model state used by an independent branch, excluding RNG."""
    digest = hashlib.sha256()
    for key in ("lattice", "Gamma", "current_species"):
        values = np.ascontiguousarray(state[key], dtype="<i8")
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    digest.update(json.dumps({
        "timestep": int(state["timestep"]),
        "newest_species": int(state["newest_species"]),
    }, sort_keys=True).encode("ascii"))
    return digest.hexdigest()


def _stop_simulation_executor(executor):
    """Stop spawned workers before returning from an interrupted saved batch.

    Python 3.12's shutdown(cancel_futures=True) cancels only queued jobs. Its
    active processes must also be stopped, otherwise Ctrl+C leaves them writing
    to the same checkpoint folders that a subsequent resume will use.
    """
    processes = list((getattr(executor, "_processes", None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


def run_simulations(
    configs, *, kind="main", repeats=1, base_seed=None, max_workers=None,
    checkpoint_root=None, initial_state=None,
):
    """Run independent simulations on separate CPU processes.

    ``configs`` is a SimulationConfig or an iterable of them. Jobs are returned
    in configuration order, with ``repeats`` independent runs of each. Settings
    are copied before submission, including the mutable retention policy.
    ``max_workers=None`` uses the available CPU count, capped by the job count;
    set it lower to limit memory use, or to 1 to replay without subprocesses.

    Each job uses its own reproducible SeedSequence child seed. ``base_seed``
    overrides configuration seeds; otherwise each distinct configuration seed
    supplies a root, and configurations with the same seed share that root's
    child sequence. Unseeded configurations share a fresh entropy root. The
    actual seeds and settings are returned in each result's ``batch_metadata``.
    Repeating the same seeded batch reproduces it regardless of worker count;
    use a different base seed for another independent batch.

    ``checkpoint_root`` stores a batch_plan.json before execution and gives
    every job its own subfolder. Without it, each config's ``checkpoint_dir``
    serves as that config's parent folder. Existing job folders and batch plans
    are rejected before execution; use a new root for a new saved batch.

    ``initial_state`` deliberately creates independent branches: its saved RNG
    is replaced by each job's child stream, and history starts at the branch
    timestep. Single-run run_main/run_percolation/resume retain their usual
    exact continuation behavior. All branches share the same starting lattice.

    Workers import this module, so calls work from notebooks on Windows. In a
    Python script, put the call under ``if __name__ == '__main__':``. On failure
    or interruption, running workers are stopped and pending jobs are cancelled.
    """
    if kind not in ("main", "percolation"):
        raise ValueError("kind must be 'main' or 'percolation'")
    repeats = _batch_integer(repeats, "repeats", minimum=1)
    if base_seed is not None:
        base_seed = _batch_integer(base_seed, "base_seed")
    if max_workers is not None:
        max_workers = _batch_integer(max_workers, "max_workers", minimum=1)
        if os.name == "nt" and max_workers > 61:
            raise ValueError("max_workers cannot exceed 61 on Windows")
    if _batch_settings_instance(configs, SimulationConfig):
        configs = [configs]
    else:
        configs = list(configs)
    if not configs:
        raise ValueError("configs must contain at least one SimulationConfig")
    if any(not _batch_settings_instance(config, SimulationConfig) for config in configs):
        raise TypeError("configs must contain only SimulationConfig instances")
    configs = [_copy_batch_settings(config, SimulationConfig) for config in configs]
    for config in configs:
        if _batch_settings_instance(config.retention, RetentionPolicy):
            config.retention = _copy_batch_settings(config.retention, RetentionPolicy)
    state = None
    if initial_state is not None:
        state = _normalize_initial_state(initial_state)
        state["rng_state"] = None
        for config in configs:
            if config.max_forbidden_cycle_length is None:
                config.max_forbidden_cycle_length = state.get("max_forbidden_cycle_length")
    for index, config in enumerate(configs):
        try:
            _validate_batch_config(config, kind, state)
            if config.seed is not None:
                _batch_integer(config.seed, f"configs[{index}].seed")
        except (TypeError, ValueError) as error:
            raise type(error)(f"config {index}: {error}") from error
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if max_workers is None and hasattr(os, "sched_getaffinity"):
        cpu_count = len(os.sched_getaffinity(0)) or 1
    if os.name == "nt":
        cpu_count = min(cpu_count, 61)
    max_workers = min(max_workers or cpu_count, len(configs) * repeats)
    root = Path(checkpoint_root).expanduser().resolve() if checkpoint_root is not None else None
    plan_path = root / "batch_plan.json" if root is not None else None
    if plan_path is not None and plan_path.exists():
        raise FileExistsError(f"batch plan already exists: {plan_path}; use a new checkpoint_root")
    seed_roots = {}
    used_seeds = set()
    jobs = []
    directories = []
    for config_index, config in enumerate(configs):
        root_seed = base_seed if base_seed is not None else config.seed
        root_seed = int(root_seed) if root_seed is not None else None
        if root_seed not in seed_roots:
            seed_roots[root_seed] = np.random.SeedSequence(root_seed)
        sequence = seed_roots[root_seed]
        for repeat_index in range(repeats):
            child = sequence.spawn(1)[0]
            seed = int.from_bytes(child.generate_state(4).astype("<u4").tobytes(), "little")
            while seed in used_seeds:
                child = sequence.spawn(1)[0]
                seed = int.from_bytes(child.generate_state(4).astype("<u4").tobytes(), "little")
            used_seeds.add(seed)
            job_index = len(jobs)
            parent = root if root is not None else config.checkpoint_dir
            directory = None if parent is None else (
                Path(parent).expanduser().resolve() / f"job_{job_index:05d}_seed_{seed}"
            )
            job_config = replace(config, seed=seed,
                                 checkpoint_dir=str(directory) if directory is not None else None)
            if kind == "percolation" and state is None:
                try:
                    _validate_batch_percolation_start(job_config)
                except ValueError as error:
                    raise ValueError(f"job {job_index}, seed={seed}: {error}") from error
            metadata = {
                "job_index": job_index, "config_index": config_index,
                "repeat_index": repeat_index, "kind": kind, "seed": seed,
                "seed_root": int(sequence.entropy), "seed_spawn_key": list(child.spawn_key),
                "max_workers": max_workers, "checkpoint_dir": job_config.checkpoint_dir,
                "plan_path": str(plan_path) if plan_path is not None else None,
                "forked_rng": state is not None,
                "config": json.loads(json.dumps(asdict(job_config), default=_batch_json_default)),
            }
            jobs.append((job_config, metadata))
            if directory is not None:
                if directory.exists():
                    raise FileExistsError(f"job {job_index} checkpoint directory already exists: {directory}")
                for previous in directories:
                    if directory == previous or directory.is_relative_to(previous) or previous.is_relative_to(directory):
                        raise ValueError(f"batch checkpoint directories overlap: {previous} and {directory}")
                ancestor = directory.parent
                while not ancestor.exists():
                    ancestor = ancestor.parent
                if not ancestor.is_dir():
                    raise NotADirectoryError(f"checkpoint parent is not a directory: {ancestor}")
                directories.append(directory)
    # Reserve every destination and record the seeds before any worker starts.
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        with plan_path.open("x", encoding="utf-8") as plan_file:
            json.dump({
                "format_version": 1, "kind": kind, "repeats": repeats,
                "base_seed": base_seed, "max_workers": max_workers,
                "initial_state": None if state is None else {
                    "timestep": int(state["timestep"]),
                    "lattice_shape": list(state["lattice"].shape),
                    "source_directory": state["source_directory"],
                    "state_fingerprint": _branch_state_fingerprint(state),
                    "forked_rng": True,
                },
                "jobs": [metadata for _, metadata in jobs],
            }, plan_file, indent=2, default=_batch_json_default)
            plan_file.write("\n")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=False)
    if max_workers == 1:
        return [_execute_simulation_job(job, state) for job in jobs]
    executor = ProcessPoolExecutor(
        max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_simulation_worker, initargs=(state,),
    )
    futures = {}
    results = [None] * len(jobs)
    try:
        for index, job in enumerate(jobs):
            try:
                futures[executor.submit(_process_simulation_job, job)] = index
            except Exception as error:
                metadata = job[1]
                raise RuntimeError(
                    f"cannot submit simulation job {index}, seed={metadata['seed']}, "
                    f"checkpoint_dir={metadata['checkpoint_dir']!r}: {error}"
                ) from error
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                metadata = jobs[index][1]
                raise RuntimeError(
                    f"simulation job {index}, seed={metadata['seed']}, "
                    f"checkpoint_dir={metadata['checkpoint_dir']!r} failed: {error}"
                ) from error
    except BaseException:
        _stop_simulation_executor(executor)
        raise
    executor.shutdown(wait=True)
    return results


def _execute_resumed_simulation_job(job, initial_state):
    config, metadata, state, resume_metadata = job
    if state is None:
        result = _execute_simulation_job((config, metadata), initial_state)
    else:
        try:
            with _simulation_job_lock(config):
                result = simulation_from_state(
                    state, gamma=config.gamma, alpha=config.alpha,
                    T=resume_metadata["remaining_timesteps"],
                    track_every=config.track_every, seed=config.seed,
                    progress=config.progress,
                    populate_first_100=config.populate_first_100,
                    checkpoint_dir=config.checkpoint_dir,
                    retention=config.retention,
                    target_timestep=int(state["target_timestep"]),
                    continue_history=True,
                    max_forbidden_cycle_length=config.max_forbidden_cycle_length,
                    max_cycle_rejection_attempts=config.max_cycle_rejection_attempts,
                )
        except Exception as error:
            raise RuntimeError(
                f"cannot resume simulation job {metadata['job_index']}, "
                f"seed={config.seed}, checkpoint_dir={config.checkpoint_dir!r}: "
                f"{error}"
            ) from error
    result["resume_metadata"] = dict(resume_metadata)
    result["batch_metadata"] = {
        **metadata, **resume_metadata, "worker_pid": os.getpid(),
    }
    return result


def _process_resumed_simulation_job(job):
    return _execute_resumed_simulation_job(job, _BATCH_INITIAL_STATE)


def resume_simulations(checkpoint_root, *, max_workers=None, initial_state=None):
    """Resume a saved batch without changing its jobs, seeds or histories.

    Read the existing ``batch_plan.json`` and continue unfinished jobs from
    their newest atomic checkpoint, including its exact saved random state.
    Finished jobs are loaded without rewriting any of their files. A job that
    has not yet saved a checkpoint restarts using its original child seed.
    For such jobs in a branched batch, pass the original shared starting state
    as ``initial_state``. New batch plans fingerprint that state to detect an
    accidental change. No new seeds or checkpoint folders are generated.

    Validate every job and checkpoint before running any work. A corrupt newest
    checkpoint raises an error naming that file; it is never silently replaced
    with a new simulation. Results retain the original order and batch metadata,
    with ``resume_metadata`` describing resumed and already completed jobs.
    Spawned workers are stopped on interruption so the same batch can safely be
    resumed after this call exits. Do not resume while another process is still
    running the batch.
    """
    if max_workers is not None:
        max_workers = _batch_integer(max_workers, "max_workers", minimum=1)
        if os.name == "nt" and max_workers > 61:
            raise ValueError("max_workers cannot exceed 61 on Windows")
    root = Path(checkpoint_root).expanduser().resolve()
    plan_path = root / "batch_plan.json"
    with plan_path.open("r", encoding="utf-8") as plan_file:
        plan = json.load(plan_file)
    if not isinstance(plan, Mapping) or plan.get("format_version") != 1:
        raise ValueError("unsupported or invalid batch_plan.json format")
    kind = plan.get("kind")
    if kind not in ("main", "percolation"):
        raise ValueError("batch plan kind must be 'main' or 'percolation'")
    saved_jobs = plan.get("jobs")
    if not isinstance(saved_jobs, list) or not saved_jobs:
        raise ValueError("batch plan must contain at least one job")
    branch = plan.get("initial_state")
    if branch is not None and not isinstance(branch, Mapping):
        raise ValueError("batch plan initial_state must be a mapping or null")
    start_timestep = 0 if branch is None else _batch_integer(
        branch.get("timestep"), "initial_state.timestep"
    )
    source_state = None
    if initial_state is not None:
        if branch is None:
            raise ValueError("this batch started fresh; do not pass initial_state")
        source_state = _normalize_initial_state(initial_state)
        expected_shape = tuple(branch.get("lattice_shape", ()))
        if source_state["timestep"] != start_timestep or (
            source_state["lattice"].shape != expected_shape
        ):
            raise ValueError("initial_state does not match the batch's starting timestep/lattice shape")
        fingerprint = branch.get("state_fingerprint")
        if fingerprint is not None and _branch_state_fingerprint(source_state) != fingerprint:
            raise ValueError("initial_state fingerprint does not match the original batch starting state")
        source_state["rng_state"] = None

    results = [None] * len(saved_jobs)
    pending = []
    used_seeds = set()
    for index, saved_metadata in enumerate(saved_jobs):
        try:
            if not isinstance(saved_metadata, Mapping):
                raise ValueError("job metadata must be a mapping")
            metadata = deepcopy(dict(saved_metadata))
            if _batch_integer(metadata.get("job_index"), "job_index") != index:
                raise ValueError("job_index must match its position in the plan")
            _batch_integer(metadata.get("config_index"), "config_index")
            _batch_integer(metadata.get("repeat_index"), "repeat_index")
            if metadata.get("kind") != kind:
                raise ValueError("job kind does not match the batch")
            if metadata.get("forked_rng") != (branch is not None):
                raise ValueError("job forked_rng does not match the batch")
            seed = _batch_integer(metadata.get("seed"), "seed")
            if seed in used_seeds:
                raise ValueError("batch contains duplicate job seeds")
            used_seeds.add(seed)
            if "seed_root" in metadata and "seed_spawn_key" in metadata:
                seed_root = _batch_integer(metadata["seed_root"], "seed_root")
                spawn_key = tuple(_batch_integer(value, "seed_spawn_key")
                                  for value in metadata["seed_spawn_key"])
                sequence = np.random.SeedSequence(seed_root, spawn_key=spawn_key)
                expected_seed = int.from_bytes(
                    sequence.generate_state(4).astype("<u4").tobytes(), "little"
                )
                if seed != expected_seed:
                    raise ValueError("saved seed does not match its SeedSequence child")
            settings = deepcopy(metadata.get("config"))
            if not isinstance(settings, dict):
                raise ValueError("job config must be a mapping")
            if settings.get("retention") is not None:
                if not isinstance(settings["retention"], Mapping):
                    raise ValueError("saved retention must be a mapping or null")
                settings["retention"] = RetentionPolicy(**settings["retention"])
            config = SimulationConfig(**settings)
            _batch_integer(config.seed, "config.seed")
            if config.seed != seed or isinstance(config.seed, bool):
                raise ValueError("config seed does not match job seed")
            directory = root / f"job_{index:05d}_seed_{seed}"
            for location in (config.checkpoint_dir, metadata.get("checkpoint_dir")):
                if location is None or Path(location).expanduser().resolve() != directory:
                    raise ValueError("checkpoint directory does not match the batch job")
            saved_plan_path = metadata.get("plan_path")
            if saved_plan_path is None or Path(saved_plan_path).expanduser().resolve() != plan_path:
                raise ValueError("job plan_path does not match batch_plan.json")
            if directory.exists() and not directory.is_dir():
                raise NotADirectoryError(f"checkpoint folder is not a directory: {directory}")
            # Probe existing locks during preflight, including locks held by
            # orphaned workers after their parent was forcibly stopped. Do not
            # create new files in older, already completed job folders.
            if (directory / ".job.lock").exists():
                with _simulation_directory_lock(directory):
                    pass
            target_timestep = start_timestep + _batch_integer(config.T, "T")
            candidates = sorted(directory.glob("checkpoint_*.npz"),
                                key=_checkpoint_timestep) if directory.is_dir() else []
            state = None
            checkpoint_path = None
            if candidates:
                checkpoint_path = candidates[-1]
                try:
                    state = load_checkpoint(checkpoint_path)
                    _normalize_initial_state(state)
                    if state["rng_state"] is None:
                        raise ValueError("checkpoint is missing its saved RNG state")
                    _make_rng(None, state["rng_state"])
                except Exception as error:
                    raise ValueError(f"invalid latest checkpoint {checkpoint_path}: {error}") from error
                if state["timestep"] != _checkpoint_timestep(checkpoint_path):
                    raise ValueError("checkpoint timestep does not match its filename")
                if not start_timestep <= state["timestep"] <= target_timestep:
                    raise ValueError("checkpoint timestep is outside the saved job interval")
                if state["target_timestep"] != target_timestep:
                    raise ValueError("checkpoint target_timestep does not match the saved job")
                if int(state["tracked_timesteps"][0]) != start_timestep:
                    raise ValueError("checkpoint history does not start at the batch starting timestep")
                for key in ("gamma", "alpha", "track_every", "populate_first_100"):
                    if state[key] != getattr(config, key):
                        raise ValueError(f"checkpoint {key} does not match the saved job")
                for key, default in (("max_forbidden_cycle_length", None),
                                     ("max_cycle_rejection_attempts", 1_000_000)):
                    if state.get(key, default) != getattr(config, key):
                        raise ValueError(f"checkpoint {key} does not match the saved job")
                if kind == "percolation":
                    for key in ("p", "largest_cluster_only"):
                        if key in state and state[key] != getattr(config, key):
                            raise ValueError(f"checkpoint {key} does not match the saved job")
                expected_shape = (config.L_row, config.L_col) if branch is None else tuple(
                    branch.get("lattice_shape", ())
                )
                if state["lattice"].shape != expected_shape:
                    raise ValueError("checkpoint lattice shape does not match the saved job")
                validation_state = _normalize_initial_state(state)
            else:
                if (directory / _ANALYSIS_FILENAME).exists():
                    raise ValueError("analysis exists but no checkpoint remains; refusing to restart this job")
                if branch is not None and source_state is None:
                    raise ValueError("initial_state is required to restart a never-started branch")
                validation_state = source_state
            _validate_batch_config(config, kind, validation_state)
            if state is None and kind == "percolation" and source_state is None:
                _validate_batch_percolation_start(config)
            remaining = target_timestep - (state["timestep"] if state is not None else start_timestep)
            finalizing = False
            if state is not None and remaining == 0:
                analysis = load_analysis(directory)
                finalizing = analysis is None or (
                    analysis["timestep"] != target_timestep
                    or not np.array_equal(analysis["tracked_timesteps"], state["tracked_timesteps"])
                    or not np.array_equal(analysis["diversity_history"], state["diversity_history"])
                    or analysis["patch_history"][-1] < 0
                )
            resume_metadata = {
                "resumed": state is not None and (remaining > 0 or finalizing),
                "from_checkpoint": state is not None,
                "already_completed": state is not None and remaining == 0 and not finalizing,
                "finalizing": finalizing,
                "resume_checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
                "remaining_timesteps": remaining,
            }
            if resume_metadata["already_completed"]:
                result = state
                if result["patch_history"][-1] < 0:
                    result["patch_history"][-1] = count_patches(result["lattice"])
                result["checkpoint_files"] = [str(path.resolve()) for path in candidates]
                result["removed_checkpoint_files"] = []
                result["resume_metadata"] = dict(resume_metadata)
                result["batch_metadata"] = {**metadata, **resume_metadata, "worker_pid": os.getpid()}
                results[index] = result
            else:
                pending.append((config, metadata, state, resume_metadata))
        except (TypeError, ValueError, OSError) as error:
            raise type(error)(f"job {index}: {error}") from error

    if not pending:
        return results
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if max_workers is None and hasattr(os, "sched_getaffinity"):
        cpu_count = len(os.sched_getaffinity(0)) or 1
    if os.name == "nt":
        cpu_count = min(cpu_count, 61)
    max_workers = min(max_workers or cpu_count, len(pending))
    # All validation is complete before folders are recreated or work begins.
    for config, _, _, _ in pending:
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if max_workers == 1:
        for job in pending:
            results[job[1]["job_index"]] = _execute_resumed_simulation_job(job, source_state)
        return results
    executor = ProcessPoolExecutor(
        max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_simulation_worker, initargs=(source_state,),
    )
    futures = {}
    try:
        for job in pending:
            futures[executor.submit(_process_resumed_simulation_job, job)] = job
        for future in as_completed(futures):
            job = futures[future]
            results[job[1]["job_index"]] = future.result()
    except BaseException:
        _stop_simulation_executor(executor)
        raise
    executor.shutdown(wait=True)
    return results


_CHECKPOINT_VERSION = 1
_PERCOLATION_METADATA_KEYS = (
    "p", "p_applied", "blocked_sites", "largest_cluster_only",
    "original_usable_sites", "usable_sites", "removed_cluster_sites",
    "effective_p",
)
_CYCLE_METADATA_KEYS = ("max_forbidden_cycle_length", "max_cycle_rejection_attempts")


def _cycle_metadata(state, *, stored=False):
    metadata = {key: state[key] for key in _CYCLE_METADATA_KEYS if key in state}
    if stored and metadata.get("max_forbidden_cycle_length", 0) is None:
        metadata["max_forbidden_cycle_length"] = 0
    return metadata


def _percolation_metadata(state):
    return {key: state[key] for key in _PERCOLATION_METADATA_KEYS if key in state}


def _checkpoint_timestep(path):
    """Return the integer suffix from ``checkpoint_<timestep>.npz``."""
    try:
        return int(path.stem.removeprefix("checkpoint_"))
    except ValueError:
        return -1


def _initial_newest_species(record):
    """Return the largest species ID at the start of a record's history.

    Histories written before this was stored always began at their run's own
    start, where the largest ID equals the number of species present.
    """
    initial_newest = record.get("initial_newest_species")
    if initial_newest is None:
        initial_newest = np.asarray(record["diversity_history"])[0]
    return int(initial_newest)


def _state_directory(state):
    """Return the checkpoint folder a loaded state or result came from."""
    for key in ("checkpoint_path", "analysis_path"):
        location = state.get(key)
        if location:
            return Path(location).expanduser().parent
    checkpoint_files = state.get("checkpoint_files") or ()
    if checkpoint_files:
        return Path(checkpoint_files[-1]).expanduser().parent
    if state.get("source_directory") is not None:
        return Path(state["source_directory"]).expanduser()
    return None


def _same_directory(first, second):
    """Report whether two optional folders name the same place."""
    if first is None or second is None:
        return first is None and second is None
    # Comparing the folders themselves, not their spellings, survives
    # relative paths and Windows short names such as ``USERNA~1``.
    try:
        return os.path.samefile(
            Path(first).expanduser(), Path(second).expanduser()
        )
    except OSError:
        # A folder that does not exist yet is not the one a state came from.
        return False


def load_checkpoint(path):
    """Load one checkpoint, or the newest checkpoint in a directory.

    The returned dictionary can be passed directly to ``run_main`` or
    ``run_percolation`` as ``initial_state``.
    """
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        candidates = list(checkpoint_path.glob("checkpoint_*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"no checkpoint_*.npz files found in {checkpoint_path}"
            )
        checkpoint_path = max(candidates, key=_checkpoint_timestep)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    with np.load(checkpoint_path, allow_pickle=False) as saved:
        required = {
            "format_version",
            "lattice",
            "Gamma",
            "current_species",
            "newest_species",
            "timestep",
            "target_timestep",
            "tracked_timesteps",
            "diversity_history",
            "rng_state",
            "gamma",
            "alpha",
            "track_every",
            "populate_first_100",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                f"checkpoint is missing: {', '.join(sorted(missing))}"
            )
        version = int(saved["format_version"])
        if version != _CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version {version}; "
                f"expected {_CHECKPOINT_VERSION}"
            )
        state = {
            "lattice": saved["lattice"].copy(),
            "Gamma": saved["Gamma"].copy(),
            "current_species": saved["current_species"].astype(
                np.int64
            ).tolist(),
            "diversity": int(saved["current_species"].size),
            "newest_species": int(saved["newest_species"]),
            "timestep": int(saved["timestep"]),
            "target_timestep": int(saved["target_timestep"]),
            "tracked_timesteps": saved["tracked_timesteps"].copy(),
            "diversity_history": saved["diversity_history"].copy(),
            "rng_state": json.loads(str(saved["rng_state"].item())),
            "gamma": float(saved["gamma"]),
            "alpha": float(saved["alpha"]),
            "track_every": int(saved["track_every"]),
            "populate_first_100": bool(saved["populate_first_100"]),
            "checkpoint_path": str(checkpoint_path.resolve()),
        }
        if "initial_newest_species" in saved.files:
            state["initial_newest_species"] = int(
                saved["initial_newest_species"]
            )
        if "patch_history" in saved.files:
            state["patch_history"] = saved["patch_history"].astype(
                np.int64, copy=True
            )
        state.update({
            key: saved[key].item() for key in _PERCOLATION_METADATA_KEYS
            if key in saved.files
        })
        if "max_forbidden_cycle_length" in saved.files:
            state["max_forbidden_cycle_length"] = int(saved["max_forbidden_cycle_length"]) or None
        if "max_cycle_rejection_attempts" in saved.files:
            state["max_cycle_rejection_attempts"] = int(saved["max_cycle_rejection_attempts"])
    state["initial_newest_species"] = _initial_newest_species(state)

    # The analysis record includes the count of its own snapshot; a checkpoint
    # additionally stores the known earlier counts for crash recovery. Older
    # checkpoints rely entirely on analysis. Trim that record to this snapshot's
    # time, preserving any known prefix if analysis has not caught up yet.
    tracked_count = state["tracked_timesteps"].size
    patch_history = state.get("patch_history")
    if patch_history is None:
        patch_history = np.full(tracked_count, _MISSING_PATCH_COUNT, dtype=np.int64)
    elif patch_history.ndim != 1 or patch_history.size != tracked_count:
        raise ValueError("checkpoint patch_history must match its tracked_timesteps")
    analysis = load_analysis(checkpoint_path)
    if analysis is not None:
        recorded_times = np.asarray(analysis["tracked_timesteps"])
        overlap = min(recorded_times.size, tracked_count)
        if overlap and np.array_equal(
            recorded_times[:overlap], state["tracked_timesteps"][:overlap]
        ):
            # A crash can occur after an atomic snapshot is written but before
            # its analysis record is replaced. Preserve all known old counts;
            # the current snapshot's missing count is recomputed on resume.
            recorded_patches = np.asarray(
                analysis["patch_history"], dtype=np.int64
            )[:overlap]
            known = recorded_patches >= 0
            patch_history[:overlap][known] = recorded_patches[known]
            state["analysis_path"] = analysis["analysis_path"]
            state["event_timesteps"] = analysis["event_timesteps"][
                analysis["event_timesteps"] <= state["timestep"]
            ].copy()
    state["patch_history"] = patch_history
    return state


def _normalize_initial_state(initial_state):
    """Validate and copy public/checkpoint state for a new simulation leg."""
    if isinstance(initial_state, (str, os.PathLike)):
        initial_state = load_checkpoint(initial_state)
    if not isinstance(initial_state, Mapping):
        raise TypeError("initial_state must be a result, checkpoint, or path")
    if "lattice" not in initial_state or "Gamma" not in initial_state:
        raise ValueError("initial_state must contain lattice and Gamma")

    lattice = np.asarray(initial_state["lattice"])
    if lattice.ndim != 2:
        raise ValueError("initial lattice must be two-dimensional")
    if not lattice.size:
        raise ValueError("initial lattice cannot be empty")
    if not np.issubdtype(lattice.dtype, np.integer):
        raise TypeError("initial lattice must contain integers")
    if np.any(lattice < -1):
        raise ValueError("initial lattice values cannot be below -1")
    lattice = lattice.astype(np.int64, copy=True)

    lattice_species = np.unique(lattice[lattice > 0]).astype(np.int64)
    if "current_species" in initial_state:
        current_species = list(initial_state["current_species"])
    else:
        # With raw arrays, Gamma rows are assumed to use ascending species IDs.
        current_species = lattice_species.tolist()
    if any(
        isinstance(species, (bool, np.bool_))
        or not isinstance(species, (int, np.integer))
        or species <= 0
        for species in current_species
    ):
        raise TypeError("current_species must contain positive integer IDs")
    current_species = [int(species) for species in current_species]
    if len(set(current_species)) != len(current_species):
        raise ValueError("current_species cannot contain duplicate IDs")
    if set(current_species) != set(lattice_species.tolist()):
        raise ValueError(
            "current_species must exactly match positive IDs in lattice"
        )

    Gamma = np.asarray(initial_state["Gamma"])
    expected_shape = (len(current_species), len(current_species))
    if Gamma.shape != expected_shape:
        raise ValueError(
            f"initial Gamma must have shape {expected_shape}, got {Gamma.shape}"
        )
    if not np.all((Gamma == 0) | (Gamma == 1)):
        raise ValueError("initial Gamma may contain only 0 and 1")
    Gamma = Gamma.astype(np.uint8, copy=True)

    minimum_newest = max(current_species, default=0)
    newest_species = initial_state.get("newest_species", minimum_newest)
    if isinstance(newest_species, (bool, np.bool_)) or not isinstance(
        newest_species, (int, np.integer)
    ):
        raise TypeError("newest_species must be an integer")
    newest_species = int(newest_species)
    if newest_species < minimum_newest:
        raise ValueError("newest_species cannot be below a live species ID")

    supplied_times = initial_state.get("tracked_timesteps")
    supplied_diversity = initial_state.get("diversity_history")
    if "timestep" in initial_state:
        timestep = initial_state["timestep"]
    elif supplied_times is not None and len(supplied_times):
        timestep = np.asarray(supplied_times)[-1]
    else:
        timestep = 0
    if isinstance(timestep, (bool, np.bool_)) or not isinstance(
        timestep, (int, np.integer)
    ):
        raise TypeError("initial timestep must be an integer")
    timestep = int(timestep)
    if timestep < 0:
        raise ValueError("initial timestep cannot be negative")

    if supplied_times is None and supplied_diversity is None:
        tracked_timesteps = np.array([timestep], dtype=np.int64)
        diversity_history = np.array(
            [len(current_species)], dtype=np.int64
        )
    elif supplied_times is None or supplied_diversity is None:
        raise ValueError(
            "tracked_timesteps and diversity_history must be supplied together"
        )
    else:
        tracked_timesteps = np.asarray(supplied_times, dtype=np.int64)
        diversity_history = np.asarray(supplied_diversity, dtype=np.int64)
        if tracked_timesteps.ndim != 1 or diversity_history.ndim != 1:
            raise ValueError("tracking histories must be one-dimensional")
        if tracked_timesteps.size != diversity_history.size:
            raise ValueError("tracking histories must have equal lengths")
        if not tracked_timesteps.size:
            raise ValueError("tracking histories cannot be empty")
        if np.any(np.diff(tracked_timesteps) <= 0):
            raise ValueError("tracked_timesteps must be strictly increasing")
        if int(tracked_timesteps[-1]) != timestep:
            raise ValueError("tracking history must end at initial timestep")
        if int(diversity_history[-1]) != len(current_species):
            raise ValueError("diversity history does not match initial lattice")
        tracked_timesteps = tracked_timesteps.copy()
        diversity_history = diversity_history.copy()

    supplied_patches = initial_state.get("patch_history")
    if supplied_patches is None:
        patch_history = np.full(
            tracked_timesteps.size, _MISSING_PATCH_COUNT, dtype=np.int64
        )
    else:
        patch_history = np.asarray(supplied_patches, dtype=np.int64)
        if patch_history.ndim != 1:
            raise ValueError("patch_history must be one-dimensional")
        if patch_history.size != tracked_timesteps.size:
            raise ValueError(
                "patch_history must have one entry per tracked timestep"
            )
        patch_history = patch_history.copy()

    if supplied_times is None:
        initial_newest_species = newest_species
    else:
        initial_newest_species = _initial_newest_species(initial_state)
    if initial_newest_species > newest_species:
        raise ValueError("initial_newest_species cannot exceed newest_species")

    rng_state = initial_state.get("rng_state")
    if isinstance(rng_state, str):
        rng_state = json.loads(rng_state)
    if rng_state is not None and not isinstance(rng_state, Mapping):
        raise TypeError("rng_state must be a NumPy bit-generator state")

    return {
        "lattice": lattice,
        "Gamma": Gamma,
        "current_species": current_species,
        "newest_species": newest_species,
        "timestep": timestep,
        "target_timestep": initial_state.get("target_timestep"),
        "tracked_timesteps": tracked_timesteps,
        "diversity_history": diversity_history,
        "patch_history": patch_history,
        "initial_newest_species": initial_newest_species,
        "rng_state": dict(rng_state) if rng_state is not None else None,
        "source_directory": _state_directory(initial_state),
        # Transient provenance for an explicitly prepared branch. New
        # snapshots omit it so they can subsequently resume in their own folder.
        "_cluster_pruning_source_directory": initial_state.get(
            "_cluster_pruning_source_directory"
        ),
        **_percolation_metadata(initial_state),
        **_cycle_metadata(initial_state),
    }


def _make_rng(seed, rng_state=None):
    """Create a seeded generator or restore the exact saved generator state."""
    if rng_state is None:
        return np.random.default_rng(seed)
    bit_generator_name = rng_state.get("bit_generator")
    bit_generator_class = getattr(np.random, bit_generator_name, None)
    if bit_generator_class is None:
        raise ValueError(f"unknown NumPy bit generator: {bit_generator_name}")
    try:
        bit_generator = bit_generator_class()
        bit_generator.state = rng_state
    except (TypeError, ValueError) as error:
        raise ValueError("invalid saved NumPy RNG state") from error
    return np.random.Generator(bit_generator)


def _write_checkpoint(checkpoint_dir, state):
    """Atomically write one uncompressed checkpoint and return its path."""
    directory = Path(checkpoint_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestep = int(state["timestep"])
    destination = directory / f"checkpoint_{timestep:012d}.npz"
    if destination.exists():
        raise FileExistsError(
            f"checkpoint already exists: {destination}; use a new directory"
        )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checkpoint_", suffix=".tmp.npz", dir=directory
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as temporary_file:
            np.savez(
                temporary_file,
                format_version=np.int64(_CHECKPOINT_VERSION),
                lattice=state["lattice"],
                Gamma=state["Gamma"],
                current_species=np.asarray(
                    state["current_species"], dtype=np.int64
                ),
                newest_species=np.int64(state["newest_species"]),
                timestep=np.int64(timestep),
                target_timestep=np.int64(state["target_timestep"]),
                tracked_timesteps=state["tracked_timesteps"],
                diversity_history=state["diversity_history"],
                initial_newest_species=np.int64(
                    _initial_newest_species(state)
                ),
                rng_state=np.asarray(json.dumps(state["rng_state"])),
                gamma=np.float64(state["gamma"]),
                alpha=np.float64(state["alpha"]),
                track_every=np.int64(state["track_every"]),
                populate_first_100=np.bool_(state["populate_first_100"]),
                **({"patch_history": np.asarray(state["patch_history"], dtype=np.int64)}
                   if "patch_history" in state else {}),
                **_percolation_metadata(state),
                **_cycle_metadata(state, stored=True),
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(destination.resolve())


def _validate_checkpoint_destination(checkpoint_dir, initial_timestep):
    """Prevent a fresh run or branch from overwriting existing snapshots."""
    directory = Path(checkpoint_dir).expanduser()
    if not directory.is_dir():
        return
    existing_timesteps = [
        _checkpoint_timestep(path)
        for path in directory.glob("checkpoint_*.npz")
    ]
    existing_timesteps = [time for time in existing_timesteps if time >= 0]
    if any(time > int(initial_timestep) for time in existing_timesteps):
        raise FileExistsError(
            f"{directory} already contains later checkpoints; "
            "use a new checkpoint_dir for a fresh run or changed-rule branch"
        )


_ANALYSIS_FILENAME = "analysis.npz"
_ANALYSIS_VERSION = 1
# Stored in a patch history where the lattice behind a sample is already gone.
_MISSING_PATCH_COUNT = -1


@dataclass
class RetentionPolicy:
    """Rules deciding which lattice snapshots a run keeps on disk.

    A long run writes one full lattice every tracking interval, which is
    gigabytes of nearly redundant state. Only three things are ever read
    back: the newest state, so the run can be resumed or plotted; the
    diversity and patch series, which the analysis record holds; and the
    lattices surrounding a large swap in living species, which are the
    interesting part of the history. Everything else is deleted as the run
    produces it.

    ``event_window`` is kept on *each* side of a detected swap. Snapshots
    newer than one window are always held, because a swap detected later
    still needs its run-up. Set ``keep_all`` to restore the old behaviour of
    keeping every snapshot.
    """

    keep_all: bool = False
    # Simulation time kept on each side of a detected swap.
    event_window: int = 1_000_000
    # Width of the moving average applied before thresholding, in samples.
    smooth_samples: int = 5
    # Quantile of the diversity series taken as the high-regime level that
    # the two thresholds are fractions of. A median would sink along with a
    # long collapse, dragging both thresholds under the series and hiding the
    # recovery until its snapshots were already deleted.
    reference_quantile: float = 0.75
    # Regime thresholds, as fractions of that high-regime level.
    low_fraction: float = 0.25
    high_fraction: float = 0.60
    # Report nothing when the two thresholds are too close to separate real
    # regimes, which is the case for a run whose diversity barely moves.
    minimum_swing: float = 2.0
    # Re-run detection after this many new snapshots.
    detect_every: int = 10


def _moving_average(values, window):
    """Smooth a series with an edge-padded uniform window of odd width."""
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window)) | 1
    if window == 1 or values.size < 2:
        return values.astype(np.float64, copy=True)
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def detect_diversity_swaps(
    timesteps, diversity_history, policy=None, reference=None
):
    """Return the timesteps at which living species swap between regimes.

    The series is smoothed, then walked with two thresholds taken from its
    high-regime level: a swap is recorded when the smoothed curve falls to
    the low threshold while in the high regime, or rises to the high
    threshold while in the low regime. Using two thresholds rather than one
    stops a curve lingering near a single level from reporting the same
    transition over and over.

    ``reference`` overrides the high-regime level. A live run passes the
    largest level it has seen so far, so that thresholds established early do
    not drift downwards during a long collapse.
    """
    timesteps = np.asarray(timesteps, dtype=np.int64)
    diversity_history = np.asarray(diversity_history, dtype=np.float64)
    if timesteps.shape != diversity_history.shape:
        raise ValueError(
            "timesteps and diversity_history must have equal length"
        )
    if policy is None:
        policy = RetentionPolicy()
    if timesteps.size < 3:
        return np.empty(0, dtype=np.int64)

    smoothed = _moving_average(diversity_history, policy.smooth_samples)
    if reference is None:
        reference = regime_reference(diversity_history, policy)
    low_level = float(policy.low_fraction) * float(reference)
    high_level = float(policy.high_fraction) * float(reference)
    if high_level - low_level < float(policy.minimum_swing):
        return np.empty(0, dtype=np.int64)

    in_high_regime = bool(smoothed[0] > 0.5 * (low_level + high_level))
    events = []
    for index in range(smoothed.size):
        value = smoothed[index]
        if in_high_regime and value <= low_level:
            in_high_regime = False
            events.append(int(timesteps[index]))
        elif not in_high_regime and value >= high_level:
            in_high_regime = True
            events.append(int(timesteps[index]))
    return np.asarray(events, dtype=np.int64)


def _diversity_quantile(diversity_history, policy):
    """Take the policy's quantile of one diversity series."""
    diversity_history = np.asarray(diversity_history, dtype=np.float64)
    if not diversity_history.size:
        return 0.0
    return float(
        np.quantile(diversity_history, float(policy.reference_quantile))
    )


def regime_reference(diversity_history, policy=None):
    """Estimate the living-species level of a run's high regime.

    The estimate is the largest quantile reached over the run's growing
    prefixes rather than the quantile of the finished series. A collapse
    lasting most of a run would otherwise pull the plain quantile down into
    the low regime, putting both thresholds under the series and hiding the
    very swap that caused the collapse. Walking prefixes instead fixes the
    level once the run has shown a high regime, and matches what a live run
    computes as it goes.
    """
    if policy is None:
        policy = RetentionPolicy()
    diversity_history = np.asarray(diversity_history, dtype=np.float64)
    if not diversity_history.size:
        return 0.0
    step = max(1, int(policy.detect_every))
    ends = list(range(step, diversity_history.size, step))
    ends.append(diversity_history.size)
    return max(
        _diversity_quantile(diversity_history[:end], policy) for end in ends
    )


def _write_analysis(checkpoint_dir, record):
    """Atomically replace a run's analysis record and return its path."""
    directory = Path(checkpoint_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _ANALYSIS_FILENAME
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".analysis_", suffix=".tmp.npz", dir=directory
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as temporary_file:
            np.savez(
                temporary_file,
                format_version=np.int64(_ANALYSIS_VERSION),
                tracked_timesteps=np.asarray(
                    record["tracked_timesteps"], dtype=np.int64
                ),
                diversity_history=np.asarray(
                    record["diversity_history"], dtype=np.int64
                ),
                patch_history=np.asarray(
                    record["patch_history"], dtype=np.int64
                ),
                event_timesteps=np.asarray(
                    record["event_timesteps"], dtype=np.int64
                ),
                initial_newest_species=np.int64(
                    _initial_newest_species(record)
                ),
                timestep=np.int64(record["timestep"]),
                target_timestep=np.int64(record["target_timestep"]),
                gamma=np.float64(record["gamma"]),
                alpha=np.float64(record["alpha"]),
                track_every=np.int64(record["track_every"]),
                populate_first_100=np.bool_(record["populate_first_100"]),
                lattice_shape=np.asarray(
                    record["lattice_shape"], dtype=np.int64
                ),
                **_percolation_metadata(record),
                **_cycle_metadata(record, stored=True),
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(destination.resolve())


def load_analysis(path):
    """Load a run's analysis record, or ``None`` when the run has none.

    ``path`` may be the run's checkpoint directory, a checkpoint file inside
    it, or the analysis file itself. The record holds the full living-species
    and patch series for the run, so plots no longer need the lattices that
    produced them. A sample whose lattice was never counted stores -1 as its
    patch count.
    """
    analysis_path = Path(path).expanduser()
    if analysis_path.is_dir():
        analysis_path = analysis_path / _ANALYSIS_FILENAME
    elif analysis_path.name != _ANALYSIS_FILENAME:
        analysis_path = analysis_path.parent / _ANALYSIS_FILENAME
    if not analysis_path.is_file():
        return None

    with np.load(analysis_path, allow_pickle=False) as saved:
        version = int(saved["format_version"])
        if version != _ANALYSIS_VERSION:
            raise ValueError(
                f"unsupported analysis version {version}; "
                f"expected {_ANALYSIS_VERSION}"
            )
        record = {name: saved[name].copy() for name in saved.files}

    record["timestep"] = int(record["timestep"])
    record["target_timestep"] = int(record["target_timestep"])
    record["track_every"] = int(record["track_every"])
    record["gamma"] = float(record["gamma"])
    record["alpha"] = float(record["alpha"])
    record["populate_first_100"] = bool(record["populate_first_100"])
    record["initial_newest_species"] = _initial_newest_species(record)
    for key in _PERCOLATION_METADATA_KEYS:
        if key in record:
            record[key] = record[key].item()
    if "max_forbidden_cycle_length" in record:
        record["max_forbidden_cycle_length"] = int(record["max_forbidden_cycle_length"]) or None
    if "max_cycle_rejection_attempts" in record:
        record["max_cycle_rejection_attempts"] = int(record["max_cycle_rejection_attempts"])
    record["analysis_path"] = str(analysis_path.resolve())
    return record


def _directed_cycle_counts(Gamma, max_cycle_length):
    """Count simple directed cycles, identifying rotations of one cycle.

    The matrix formulas give exact counts through length four, including
    mutual-invasion pairs. Longer cycles use a bounded depth-first search.
    """
    adjacency = np.asarray(Gamma) != 0
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Gamma must be a square matrix")
    adjacency = adjacency.copy()
    np.fill_diagonal(adjacency, False)
    size = adjacency.shape[0]
    counts = {length: 0 for length in range(2, max_cycle_length + 1)}
    if size < 2:
        return counts

    reciprocal = int(np.count_nonzero(adjacency & adjacency.T))
    counts[2] = reciprocal // 2
    if max_cycle_length >= 3:
        matrix = adjacency.astype(np.float64)
        two_steps = matrix @ matrix
        counts[3] = int(round(float(np.sum(two_steps * matrix.T)))) // 3
    if max_cycle_length >= 4:
        closed_four = int(round(float(np.sum(two_steps * two_steps.T))))
        repeated_vertices = int(round(float(np.sum(np.diag(two_steps) ** 2))))
        counts[4] = (closed_four - 2 * repeated_vertices + reciprocal) // 4

    if max_cycle_length > 4:
        neighbors = [np.flatnonzero(row).tolist() for row in adjacency]
        search_steps = 0
        step_limit = 10_000_000

        def visit(start, vertex, visited, length):
            nonlocal search_steps
            for neighbor in neighbors[vertex]:
                search_steps += 1
                if search_steps > step_limit:
                    raise ValueError(
                        "cycle search exceeded 10 million steps; reduce "
                        "max_cycle_length or select fewer/smaller checkpoints"
                    )
                if neighbor == start:
                    if length >= 5:
                        counts[length] += 1
                elif (
                    neighbor > start
                    and neighbor not in visited
                    and length < max_cycle_length
                ):
                    visited.add(neighbor)
                    visit(start, neighbor, visited, length + 1)
                    visited.remove(neighbor)

        for start in range(size):
            visit(start, start, {start}, 1)
    return counts


@njit
def _log_add_cycles(first, second):
    """Add positive counts represented by their natural logarithms."""
    if first == -np.inf:
        return second
    if second == -np.inf:
        return first
    larger = max(first, second)
    return larger + np.log1p(np.exp(min(first, second) - larger))


@njit
def _sample_long_cycle_logs(adjacency, offsets, edges, samples,
                            max_length, seed):
    """Estimate all long simple-cycle counts with weighted random paths.

    A cycle is rooted at its smallest vertex. Each step chooses uniformly
    among unvisited larger neighbors. The product of branch counts is the
    inverse probability of that path, so its closure weight is an unbiased
    contribution to the count for that length.
    """
    np.random.seed(seed)
    size = adjacency.shape[0]
    log_totals = np.full(max_length + 1, -np.inf)
    hits = np.zeros(max_length + 1, dtype=np.int64)
    visited = np.zeros(size, dtype=np.bool_)
    path = np.empty(size, dtype=np.int64)
    candidates = np.empty(size, dtype=np.int64)
    if samples >= size:
        repetitions = (samples + size - 1) // size
        draws = size * repetitions
        log_root_weight = -np.log(float(repetitions))
    else:
        repetitions = 0
        draws = samples
        log_root_weight = np.log(float(size) / float(samples))

    for draw in range(draws):
        root = draw // repetitions if repetitions else np.random.randint(size)
        visited[root] = True
        path[0] = root
        vertex = root
        length = 1
        log_weight = log_root_weight
        while True:
            if length >= 5 and adjacency[vertex, root]:
                log_totals[length] = _log_add_cycles(
                    log_totals[length], log_weight
                )
                hits[length] += 1
            if length >= max_length:
                break
            choices = 0
            for edge_index in range(offsets[vertex], offsets[vertex + 1]):
                neighbor = edges[edge_index]
                if neighbor > root and not visited[neighbor]:
                    candidates[choices] = neighbor
                    choices += 1
            if choices == 0:
                break
            vertex = candidates[np.random.randint(choices)]
            log_weight += np.log(float(choices))
            visited[vertex] = True
            path[length] = vertex
            length += 1
        for path_index in range(length):
            visited[path[path_index]] = False
    return log_totals, hits


def _cycle_histogram_for_gamma(Gamma, max_cycle_length, samples, seed):
    """Return counts, natural-log counts, and long-cycle sample hits."""
    adjacency = np.asarray(Gamma) != 0
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Gamma must be a square matrix")
    size = adjacency.shape[0]
    limit = size if max_cycle_length is None else min(max_cycle_length, size)
    counts = _directed_cycle_counts(Gamma, min(max(limit, 2), 4))
    logs = {
        length: math.log(count) if count else -math.inf
        for length, count in counts.items()
    }
    hits = {}
    estimated = set()
    if limit <= 4:
        return counts, logs, hits, estimated

    # Small graphs can be enumerated in full. Larger graphs may contain an
    # exponential number of simple cycles, so sample long ones instead.
    if size <= 12:
        try:
            counts = _directed_cycle_counts(Gamma, limit)
        except ValueError as error:
            if "cycle search exceeded" not in str(error):
                raise
        else:
            logs = {
                length: math.log(count) if count else -math.inf
                for length, count in counts.items()
            }
            return counts, logs, hits, estimated

    adjacency = adjacency.copy()
    np.fill_diagonal(adjacency, False)
    degrees = np.count_nonzero(adjacency, axis=1)
    offsets = np.empty(size + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(degrees)
    edges = np.nonzero(adjacency)[1].astype(np.int64)
    sampled_logs, sampled_hits = _sample_long_cycle_logs(
        adjacency, offsets, edges, samples, limit, seed
    )
    for length in range(5, limit + 1):
        log_count = float(sampled_logs[length])
        logs[length] = log_count
        counts[length] = math.exp(log_count) if log_count < 700 else math.inf
        hits[length] = int(sampled_hits[length])
        estimated.add(length)
    return counts, logs, hits, estimated


def _fit_cycle_trend(log10_counts, model, min_length=2, max_length=None):
    """Fit either a power law or an exponential to positive cycle counts."""
    lengths = np.asarray(
        [length for length, value in log10_counts.items()
         if length >= min_length
         and (max_length is None or length <= max_length)
         and np.isfinite(value)], dtype=np.float64
    )
    if lengths.size < 2:
        raise ValueError(
            "a cycle-count fit needs at least two cycle lengths with "
            "positive counts"
        )
    log_counts = np.asarray(
        [log10_counts[int(length)] * math.log(10) for length in lengths]
    )
    predictor = np.log(lengths) if model == "power_law" else lengths
    slope, log_constant = np.polyfit(predictor, log_counts, 1)
    fitted = log_constant + slope * predictor
    residual = float(np.sum((log_counts - fitted) ** 2))
    total = float(np.sum((log_counts - np.mean(log_counts)) ** 2))
    fit = {
        "model": model,
        "constant": (math.exp(log_constant)
                     if -700 < log_constant < 700 else None),
        "log10_constant": float(log_constant / math.log(10)),
        "r_squared_log_space": 1.0 - residual / total if total else 1.0,
        "lengths_used": lengths.astype(np.int64),
    }
    if model == "power_law":
        fit["exponent"] = float(-slope)
    else:
        fit["rate"] = float(slope)
    return fit


def _fit_cycle_power_law(log10_counts, min_length=2, max_length=None):
    """Fit count = constant * length**(-exponent) in log-log space."""
    return _fit_cycle_trend(
        log10_counts, "power_law", min_length, max_length
    )


def _fit_cycle_exponential(log10_counts, min_length=2, max_length=None):
    """Fit count = constant * exp(rate * length) in semi-log space."""
    return _fit_cycle_trend(
        log10_counts, "exponential", min_length, max_length
    )


def show_gamma_cycle_histogram(
    checkpoint_dir,
    event_type="collapse",
    side=None,
    event_index="all",
    window=None,
    max_cycle_length=None,
    log_x=False,
    log_y=False,
    samples_per_checkpoint=5000,
    seed=0,
    fit_power_law=False,
    power_law_min_length=2,
    power_law_max_length=None,
    fit_model=None,
    fit_min_length=None,
    fit_max_length=None,
):
    """Plot directed invasion-cycle counts around diversity switches.

    ``checkpoint_dir`` is one run folder with ``analysis.npz`` and retained
    checkpoints. ``event_type`` is "collapse" or "generation" (recovery).
    ``event_index='all'`` (also ``None``) combines all matching switches;
    an integer chooses one, zero-based among switches of that type. The
    default side is before collapses and after generations. ``side`` can
    override it. ``window`` optionally limits distance from each switch.

    All possible cycle lengths are considered by default. Counts through
    length four are exact. Longer lengths are estimated by weighted sampling
    of simple paths when exhaustive counting is too large; increase
    ``samples_per_checkpoint`` for a more stable estimate. Rotations of a
    cycle count once per snapshot, and counts sum across snapshots. The
    returned ``sample_hits`` shows how many sampled paths closed at each
    estimated length; zero hits do not prove that length is absent.

    ``fit_power_law=True`` fits ``count = C * length**(-alpha)`` to all
    positive histogram bins between ``power_law_min_length`` and
    ``power_law_max_length`` (inclusive) by ordinary least squares in log-log
    space. ``fit_model='exponential'`` instead fits ``count = A * exp(k*length)``
    in semi-log space. ``fit_min_length`` and ``fit_max_length`` set the fit
    interval for either model. The curve and fitted parameters appear in the
    legend.
    """
    directory = Path(checkpoint_dir).expanduser()
    if not directory.is_dir():
        raise ValueError("checkpoint_dir must be a checkpoint folder")
    if event_type not in ("collapse", "generation"):
        raise ValueError("event_type must be 'collapse' or 'generation'")
    if side is None:
        side = "before" if event_type == "collapse" else "after"
    if side not in ("before", "after"):
        raise ValueError("side must be 'before' or 'after'")
    if event_index not in (None, "all") and (
        isinstance(event_index, (bool, np.bool_))
        or not isinstance(event_index, (int, np.integer))
        or event_index < 0
    ):
        raise ValueError(
            "event_index must be 'all', None, or a non-negative integer"
        )
    if window is not None and (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window <= 0
    ):
        raise ValueError("window must be a positive integer or None")
    if max_cycle_length is not None and (
        isinstance(max_cycle_length, (bool, np.bool_))
        or not isinstance(max_cycle_length, (int, np.integer))
        or max_cycle_length < 2
    ):
        raise ValueError(
            "max_cycle_length must be None or an integer of at least 2"
        )
    if (
        isinstance(samples_per_checkpoint, (bool, np.bool_))
        or not isinstance(samples_per_checkpoint, (int, np.integer))
        or samples_per_checkpoint < 1
    ):
        raise ValueError("samples_per_checkpoint must be a positive integer")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(log_x, (bool, np.bool_)) or not isinstance(
        log_y, (bool, np.bool_)
    ):
        raise TypeError("log_x and log_y must be True or False")
    if not isinstance(fit_power_law, (bool, np.bool_)):
        raise TypeError("fit_power_law must be True or False")
    if fit_model not in (None, "power_law", "exponential"):
        raise ValueError("fit_model must be None, 'power_law', or 'exponential'")
    if fit_power_law and fit_model == "exponential":
        raise ValueError("choose fit_power_law or fit_model='exponential'")
    selected_fit_model = (
        fit_model or ("power_law" if fit_power_law else None)
    )
    if (
        isinstance(power_law_min_length, (bool, np.bool_))
        or not isinstance(power_law_min_length, (int, np.integer))
        or power_law_min_length < 2
    ):
        raise ValueError("power_law_min_length must be an integer of at least 2")
    if power_law_max_length is not None and (
        isinstance(power_law_max_length, (bool, np.bool_))
        or not isinstance(power_law_max_length, (int, np.integer))
        or power_law_max_length < power_law_min_length
    ):
        raise ValueError(
            "power_law_max_length must be None or an integer at least "
            "power_law_min_length"
        )
    if fit_min_length is not None and (
        isinstance(fit_min_length, (bool, np.bool_))
        or not isinstance(fit_min_length, (int, np.integer))
        or fit_min_length < 2
    ):
        raise ValueError("fit_min_length must be None or an integer of at least 2")
    effective_fit_min = (
        power_law_min_length if fit_min_length is None else fit_min_length
    )
    effective_fit_max = (
        power_law_max_length if fit_max_length is None else fit_max_length
    )
    if effective_fit_max is not None and (
        isinstance(effective_fit_max, (bool, np.bool_))
        or not isinstance(effective_fit_max, (int, np.integer))
        or effective_fit_max < effective_fit_min
    ):
        raise ValueError(
            "fit_max_length must be None or an integer at least fit_min_length"
        )

    analysis = load_analysis(directory)
    if analysis is None:
        raise FileNotFoundError(f"analysis.npz not found in {directory}")
    times = np.asarray(analysis["tracked_timesteps"], dtype=np.int64)
    diversity = np.asarray(analysis["diversity_history"], dtype=np.float64)
    events = np.asarray(analysis["event_timesteps"], dtype=np.int64)
    if times.size == 0 or times.shape != diversity.shape:
        raise ValueError("analysis has no aligned diversity history")
    if not events.size:
        raise ValueError("analysis contains no diversity switches")

    # Stored events may include several nearby estimates of one transition.
    # The detector records the sample where a threshold is crossed, so the
    # smoothed change into that sample identifies its direction.
    smoothed = _moving_average(diversity, 5)
    directions = []
    for event in events:
        index = int(np.searchsorted(times, event))
        if index >= times.size or int(times[index]) != int(event):
            raise ValueError("event timestep is absent from tracked history")
        change = smoothed[index] - smoothed[max(0, index - 1)]
        if change == 0:
            change = smoothed[min(times.size - 1, index + 1)] - smoothed[
                max(0, index - 2)
            ]
        if change == 0:
            change = diversity[index] - diversity[max(0, index - 1)]
        directions.append("generation" if change > 0 else "collapse")

    matching = [i for i, direction in enumerate(directions)
                if direction == event_type]
    if not matching:
        raise ValueError(f"analysis contains no {event_type} events")
    if event_index not in (None, "all"):
        if event_index >= len(matching):
            raise IndexError(
                f"event_index {event_index} is out of range for "
                f"{len(matching)} {event_type} events"
            )
        matching = [matching[event_index]]

    selected_events = events[matching]
    checkpoint_paths = sorted(
        directory.glob("checkpoint_*.npz"), key=_checkpoint_timestep
    )
    selected_paths = {}
    for index in matching:
        event = int(events[index])
        if side == "before":
            opposite = next(
                (int(events[j]) for j in range(index - 1, -1, -1)
                 if directions[j] != event_type),
                int(times[0]) - 1,
            )
            selected = (
                path for path in checkpoint_paths
                if opposite <= _checkpoint_timestep(path) < event
                and (window is None or _checkpoint_timestep(path) >= event - window)
            )
        else:
            opposite = next(
                (int(events[j]) for j in range(index + 1, len(events))
                 if directions[j] != event_type),
                int(times[-1]) + 1,
            )
            selected = (
                path for path in checkpoint_paths
                if event <= _checkpoint_timestep(path) < opposite
                and (window is None or _checkpoint_timestep(path) <= event + window)
            )
        for path in selected:
            selected_paths[_checkpoint_timestep(path)] = path

    if not selected_paths:
        raise ValueError(
            "no retained Gamma checkpoints on the selected side of the "
            "switch; try another event, side, or window"
        )

    exact_counts = {}
    aggregate_logs = {}
    sample_hits = {}
    estimated_lengths = set()
    per_checkpoint_counts = {}
    checkpoint_timesteps = sorted(selected_paths)
    for timestep in checkpoint_timesteps:
        path = selected_paths[timestep]
        with np.load(path, allow_pickle=False) as saved:
            Gamma = saved["Gamma"]
            if int(saved["timestep"]) != timestep:
                raise ValueError(f"checkpoint timestep disagrees with {path}")
            snapshot_seed = np.random.SeedSequence(
                [int(seed), timestep & 0xffffffff, timestep >> 32]
            ).generate_state(1)[0]
            snapshot_counts, snapshot_logs, hits, estimated = (
                _cycle_histogram_for_gamma(
                    Gamma, max_cycle_length, samples_per_checkpoint,
                    int(snapshot_seed),
                )
            )
        per_checkpoint_counts[timestep] = snapshot_counts
        for length, log_count in snapshot_logs.items():
            aggregate_logs[length] = _log_add_cycles(
                aggregate_logs.get(length, -math.inf), log_count
            )
        for length in range(2, min(4, len(Gamma)) + 1):
            exact_counts[length] = exact_counts.get(length, 0) + int(
                snapshot_counts[length]
            )
        for length, hit_count in hits.items():
            sample_hits[length] = sample_hits.get(length, 0) + hit_count
        estimated_lengths.update(estimated)

    largest_length = max(aggregate_logs, default=2)
    counts = {}
    log10_counts = {}
    for length in range(2, largest_length + 1):
        if length in exact_counts:
            counts[length] = exact_counts[length]
            log_count = (math.log(counts[length]) if counts[length]
                         else -math.inf)
        else:
            log_count = aggregate_logs.get(length, -math.inf)
            counts[length] = (
                math.exp(log_count) if log_count < 700 else math.inf
            )
        log10_counts[length] = log_count / math.log(10)

    trend_fit = (
        _fit_cycle_trend(
            log10_counts, selected_fit_model,
            effective_fit_min, effective_fit_max,
        )
        if selected_fit_model is not None else None
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    use_log10_values = any(value >= 700 for value in aggregate_logs.values())
    exact_bars = [length for length in counts
                  if length not in estimated_lengths and counts[length] > 0]
    estimated_bars = [length for length in counts
                      if length in estimated_lengths and counts[length] > 0]
    if exact_bars or estimated_bars:
        def heights(lengths):
            return [log10_counts[length] if use_log10_values
                    else counts[length] for length in lengths]

        ax.bar(
            exact_bars, heights(exact_bars),
            width=0.8,
            color="tab:blue",
            edgecolor="white",
            label="Exact (lengths 2–4)",
        )
        ax.bar(
            estimated_bars, heights(estimated_bars),
            width=0.8,
            color="tab:orange",
            edgecolor="white",
            label="Estimated (longer cycles)",
        )
    else:
        ax.text(0.5, 0.5, "No cycles in selected snapshots",
                ha="center", va="center", transform=ax.transAxes)
    if len(counts) <= 20:
        ax.set_xticks(list(counts))
    elif not log_x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.set_xlabel("Cycle length (species)")
    ax.set_ylabel(
        "log10(count across Gamma snapshots)" if use_log10_values
        else "Count across Gamma snapshots"
    )
    ax.set_title(
        f"Invasion cycles {side} {event_type} "
        f"({len(selected_events)} events, {len(checkpoint_timesteps)} snapshots)"
    )
    if log_x:
        ax.set_xscale("log")
        if len(counts) <= 20:
            ax.set_xticks(list(counts), labels=[str(length) for length in counts])
    if log_y and not use_log10_values:
        ax.set_yscale("log")
        if not exact_bars and not estimated_bars:
            ax.set_ylim(0.8, 1.2)
    if trend_fit is not None:
        fit_lengths = trend_fit["lengths_used"]
        curve_lengths = np.linspace(
            float(fit_lengths[0]), float(fit_lengths[-1]), 200
        )
        log_curve = trend_fit["log10_constant"] * math.log(10)
        if selected_fit_model == "power_law":
            log_curve -= trend_fit["exponent"] * np.log(curve_lengths)
        else:
            log_curve += trend_fit["rate"] * curve_lengths
        curve_counts = (
            log_curve / math.log(10) if use_log10_values
            else np.exp(log_curve)
        )
        constant = trend_fit["constant"]
        constant_label = (
            f"{constant:.3g}" if constant is not None
            else f"10^{trend_fit['log10_constant']:.2f}"
        )
        if selected_fit_model == "power_law":
            fit_label = (
                f"Power law (L={fit_lengths[0]}–{fit_lengths[-1]}): "
                f"C={constant_label}, alpha={trend_fit['exponent']:.3g}"
            )
        else:
            fit_label = (
                f"Exponential (L={fit_lengths[0]}–{fit_lengths[-1]}): "
                f"A={constant_label}, k={trend_fit['rate']:.3g}"
            )
        ax.plot(
            curve_lengths, curve_counts, color="tab:red", lw=2,
            label=fit_label,
        )
    if estimated_bars or trend_fit is not None:
        ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()
    return {
        "cycle_counts": counts,
        "log10_cycle_counts": log10_counts,
        "per_checkpoint_counts": per_checkpoint_counts,
        "estimated_lengths": sorted(estimated_lengths),
        "sample_hits": sample_hits,
        "plot_uses_log10_counts": use_log10_values,
        "fit": trend_fit,
        "power_law_fit": (
            trend_fit if selected_fit_model == "power_law" else None
        ),
        "exponential_fit": (
            trend_fit if selected_fit_model == "exponential" else None
        ),
        "event_timesteps": selected_events.copy(),
        "checkpoint_timesteps": np.asarray(checkpoint_timesteps, dtype=np.int64),
        "figure": fig,
        "axes": ax,
    }


class _CheckpointRetention:
    """Delete the snapshots a run no longer needs, while it produces them.

    A snapshot survives when it is newer than one ``event_window`` (a swap
    detected later would need it as run-up), or when it sits inside the
    window around a detected swap. The newest snapshot therefore always
    survives, which is what keeps a pruned run resumable.

    A snapshot only becomes a deletion candidate once its patch count is in
    the analysis record. That makes the policy safe to point at a directory
    written before analysis records existed: nothing there is counted yet, so
    nothing there is deleted.
    """

    def __init__(self, checkpoint_dir, policy):
        self.directory = Path(checkpoint_dir).expanduser()
        self.policy = policy
        self.events = np.empty(0, dtype=np.int64)
        self.reference = 0.0
        self._pending = []
        # None forces a detection pass on the first recorded snapshot, so
        # adopted files are judged against real events, not an empty list.
        self._since_detection = None

    def adopt_existing(self):
        """Take responsibility for snapshots an earlier leg left behind."""
        if self.policy.keep_all or not self.directory.is_dir():
            return
        for path in self.directory.glob("checkpoint_*.npz"):
            timestep = _checkpoint_timestep(path)
            if timestep >= 0:
                self._pending.append((timestep, path))
        self._pending.sort()

    def record(self, path, timestep, timesteps, diversity, patches):
        """Register a new snapshot and drop whatever is now redundant."""
        if self.policy.keep_all:
            return []
        self._pending.append((int(timestep), Path(path)))
        interval = max(1, int(self.policy.detect_every))
        if self._since_detection is None or self._since_detection >= interval:
            self._since_detection = 0
            self._refresh_events(timesteps, diversity)
        else:
            self._since_detection += 1
        return self._prune(timesteps, patches)

    def finish(self, timesteps, diversity, patches):
        """Run a final detection and pruning pass once the leg has stopped."""
        if self.policy.keep_all:
            return []
        self._refresh_events(timesteps, diversity)
        return self._prune(timesteps, patches)

    def _refresh_events(self, timesteps, diversity):
        # The reference only ever rises. A run that has already shown a high
        # regime keeps judging against it, so a recovery out of a long
        # collapse is recognised while its snapshots are still on disk.
        self.reference = max(
            self.reference, _diversity_quantile(diversity, self.policy)
        )
        detected = detect_diversity_swaps(
            timesteps, diversity, self.policy, reference=self.reference
        )
        # Windows already granted are never revoked. A snapshot deleted on an
        # earlier estimate cannot be brought back, so keeping the union makes
        # the surviving set stable while the median is still settling.
        self.events = np.union1d(self.events, detected)

    def _survives(self, timestep, newest_timestep):
        window = int(self.policy.event_window)
        if timestep >= newest_timestep - window:
            return True
        if not self.events.size:
            return False
        return bool(np.any(np.abs(self.events - timestep) <= window))

    def _prune(self, timesteps, patches):
        timesteps = np.asarray(timesteps, dtype=np.int64)
        patches = np.asarray(patches, dtype=np.int64)
        if not timesteps.size:
            return []
        newest_timestep = int(timesteps[-1])

        removed = []
        survivors = []
        for timestep, path in self._pending:
            index = int(np.searchsorted(timesteps, timestep))
            counted = (
                index < timesteps.size
                and int(timesteps[index]) == timestep
                and int(patches[index]) >= 0
            )
            if not counted or self._survives(timestep, newest_timestep):
                survivors.append((timestep, path))
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            removed.append(str(path))
        self._pending = survivors
        return removed


def _align_patch_history(patch_history, sample_count):
    """Pad or trim a patch series to exactly one entry per tracked sample."""
    patch_history = np.asarray(patch_history, dtype=np.int64)
    if patch_history.size == sample_count:
        return patch_history
    if patch_history.size > sample_count:
        return patch_history[:sample_count].copy()
    return np.concatenate(
        [
            patch_history,
            np.full(
                sample_count - patch_history.size,
                _MISSING_PATCH_COUNT,
                dtype=np.int64,
            ),
        ]
    )


def create_lattice(L_col, L_row, D, rng=None):
    """Create a fully occupied lattice containing ``D`` initial species.

    Species are numbered from 1 to ``D`` and randomly distributed, with every
    species guaranteed to occupy at least one site. The paper's standard
    single-species initial condition is obtained with ``D=1``.
    """
    values = (L_col, L_row, D)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError("L_col, L_row, and D must be integers")
    if L_col <= 0 or L_row <= 0:
        raise ValueError("L_col and L_row must be positive")
    if D <= 0:
        raise ValueError("D must be positive")

    total_spots = L_row * L_col
    if D > total_spots:
        raise ValueError("D cannot exceed the number of lattice sites")

    random_source = np.random if rng is None else rng
    species_ids = np.arange(1, D + 1, dtype=np.int32)
    lattice = np.empty(total_spots, dtype=np.int32)
    lattice[:D] = species_ids
    if total_spots > D:
        lattice[D:] = random_source.choice(
            species_ids, size=total_spots - D
        )
    random_source.shuffle(lattice)
    lattice = lattice.reshape(L_row, L_col)

    # Keep this as a list so species can easily be added/removed later.
    current_species = list(range(1, D + 1))
    diversity = len(current_species)
    newest_species = D if D else 0

    return lattice, current_species, diversity, newest_species

def _validate_cycle_rejection_options(
    max_forbidden_cycle_length, max_cycle_rejection_attempts=1_000_000
):
    """Validate the optional directed-cycle restriction without drawing RNG."""
    if max_forbidden_cycle_length is None:
        max_cycle_length = 0
    else:
        if isinstance(max_forbidden_cycle_length, (bool, np.bool_)) or not isinstance(
            max_forbidden_cycle_length, (int, np.integer)
        ):
            raise TypeError("max_forbidden_cycle_length must be an integer or None")
        if max_forbidden_cycle_length < 2:
            raise ValueError("max_forbidden_cycle_length must be at least 2")
        max_cycle_length = int(max_forbidden_cycle_length)
    if isinstance(max_cycle_rejection_attempts, (bool, np.bool_)) or not isinstance(
        max_cycle_rejection_attempts, (int, np.integer)
    ):
        raise TypeError("max_cycle_rejection_attempts must be an integer")
    if max_cycle_rejection_attempts <= 0:
        raise ValueError("max_cycle_rejection_attempts must be positive")
    return max_cycle_length


@njit(cache=True)
def _has_forbidden_directed_cycle(Gamma, max_cycle_length):
    """Find a directed cycle of length 2 through the specified bound.

    Breadth-first searches stop after the bounded number of edges. Diagonal
    entries are excluded because species have no directed self-interaction.
    """
    if max_cycle_length < 2:
        return False
    species_count = Gamma.shape[0]
    queue = np.empty(species_count, dtype=np.int32)
    distances = np.empty(species_count, dtype=np.int32)
    for source in range(species_count):
        distances[:] = -1
        distances[source] = 0
        queue[0] = source
        queue_start = 0
        queue_end = 1
        while queue_start < queue_end:
            current = queue[queue_start]
            queue_start += 1
            next_distance = distances[current] + 1
            for target in range(species_count):
                if target == current or Gamma[current, target] == 0:
                    continue
                if target == source:
                    if 2 <= next_distance <= max_cycle_length:
                        return True
                elif next_distance < max_cycle_length and distances[target] == -1:
                    distances[target] = next_distance
                    queue[queue_end] = target
                    queue_end += 1
    return False


@njit(cache=True)
def _new_species_closes_forbidden_cycle(
    Gamma, active_slots, live_count, new_slot, max_cycle_length
):
    """Check only cycles through a candidate new species in active storage.

    A cycle new -> outgoing -> ... -> incoming -> new is forbidden when the
    old-species path has at most ``max_cycle_length - 2`` edges. Multi-source
    BFS gives that minimum path length while ignoring unused or extinct slots.
    """
    if max_cycle_length < 2 or live_count == 0:
        return False
    has_outgoing = False
    has_incoming = False
    for position in range(live_count):
        other_slot = active_slots[position]
        outgoing = Gamma[new_slot, other_slot] != 0
        incoming = Gamma[other_slot, new_slot] != 0
        has_outgoing = has_outgoing or outgoing
        has_incoming = has_incoming or incoming
        if outgoing and incoming:
            return True
    if max_cycle_length == 2 or not has_outgoing or not has_incoming:
        return False

    distances = np.full(live_count, -1, dtype=np.int32)
    queue = np.empty(live_count, dtype=np.int32)
    queue_start = 0
    queue_end = 0
    for position in range(live_count):
        if Gamma[new_slot, active_slots[position]]:
            distances[position] = 0
            queue[queue_end] = position
            queue_end += 1

    while queue_start < queue_end:
        current_position = queue[queue_start]
        queue_start += 1
        current_slot = active_slots[current_position]
        next_distance = distances[current_position] + 1
        if next_distance > max_cycle_length - 2:
            continue
        for target_position in range(live_count):
            if distances[target_position] != -1:
                continue
            target_slot = active_slots[target_position]
            if Gamma[current_slot, target_slot] == 0:
                continue
            if Gamma[target_slot, new_slot]:
                return True
            distances[target_position] = next_distance
            queue[queue_end] = target_position
            queue_end += 1
    return False


def update_Gamma(
    Gamma,
    gamma,
    current_species,
    newest_species,
    invaded_species=None,
    rng=None,
    max_forbidden_cycle_length=None,
    max_cycle_rejection_attempts=1_000_000,
):
    """Add one species to the directed invasion matrix.

    Rows and columns follow the order of ``current_species``. Thus,
    ``Gamma[i, j] == 1`` means ``current_species[i]`` can invade
    ``current_species[j]``. This keeps the matrix proportional to the number
    of live species instead of the largest species ID.

    If ``Gamma`` is ``None``, the initial interaction matrix is returned.
    Otherwise, one species is added and the function returns the updated
    matrix, live-species list, and newest species ID. The new species is
    guaranteed to invade ``invaded_species`` when it is not ``None`` or 0.
    When ``max_forbidden_cycle_length`` is enabled, all candidate connections
    are redrawn at the same ``gamma`` until no directed cycle of length 2
    through that bound is created. Longer cycles remain possible.
    """
    if (
        isinstance(gamma, (bool, np.bool_))
        or not isinstance(gamma, (int, float, np.integer, np.floating))
    ):
        raise TypeError("gamma must be a real number")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    max_cycle_length = _validate_cycle_rejection_options(
        max_forbidden_cycle_length, max_cycle_rejection_attempts
    )

    if isinstance(newest_species, (bool, np.bool_)) or not isinstance(
        newest_species, (int, np.integer)
    ):
        raise TypeError("newest_species must be an integer")
    if invaded_species is not None and (
        isinstance(invaded_species, (bool, np.bool_))
        or not isinstance(invaded_species, (int, np.integer))
    ):
        raise TypeError("invaded_species must be an integer or None")

    if newest_species < 0:
        raise ValueError("newest_species cannot be negative")

    if not isinstance(current_species, (list, tuple, np.ndarray)):
        raise TypeError("current_species must be a list or one-dimensional array")
    if isinstance(current_species, np.ndarray) and current_species.ndim != 1:
        raise ValueError("current_species must be one-dimensional")

    live_species = list(current_species)
    if any(
        isinstance(species, (bool, np.bool_))
        or not isinstance(species, (int, np.integer))
        for species in live_species
    ):
        raise TypeError("current_species must contain only integer species IDs")
    if len(set(live_species)) != len(live_species):
        raise ValueError("current_species cannot contain duplicate species IDs")
    if any(species < 1 or species > newest_species for species in live_species):
        raise ValueError("current_species contains an invalid species ID")
    if invaded_species not in (None, 0) and invaded_species not in live_species:
        raise ValueError("invaded_species must be a currently live species")

    new_species = newest_species + 1

    active_count = len(live_species)
    expected_shape = (active_count, active_count)
    random_source = np.random if rng is None else rng
    if Gamma is None:
        if max_cycle_length and gamma == 1 and active_count > 1:
            raise ValueError("gamma=1 cannot produce a directed-cycle-free initial Gamma")
        for attempt in range(max_cycle_rejection_attempts):
            Gamma = (random_source.random(expected_shape) < gamma).astype(np.uint8)
            np.fill_diagonal(Gamma, 0)
            if not max_cycle_length or not _has_forbidden_directed_cycle(
                Gamma, max_cycle_length
            ):
                return Gamma
        raise RuntimeError(
            "Initial Gamma rejection exceeded max_cycle_rejection_attempts; "
            "decrease gamma or increase the attempt limit"
        )
    else:
        Gamma = np.asarray(Gamma)
        if Gamma.shape != expected_shape:
            raise ValueError(
                f"Gamma must have shape {expected_shape}, got {Gamma.shape}"
            )
        if not np.all((Gamma == 0) | (Gamma == 1)):
            raise ValueError("Gamma may contain only 0 and 1")
        if max_cycle_length and _has_forbidden_directed_cycle(Gamma, max_cycle_length):
            raise ValueError(
                "Initial Gamma contains a forbidden directed cycle; "
                "start from a compatible state or one initial species"
            )
        if max_cycle_length and gamma == 1 and active_count:
            raise ValueError("gamma=1 cannot introduce a species without a directed 2-cycle")

    # uint8 uses one byte per edge and is sufficient for this binary matrix.
    new_size = active_count + 1
    new_Gamma = np.empty((new_size, new_size), dtype=np.uint8)
    new_Gamma[:-1, :-1] = Gamma

    # Draw both directed links with every already-existing species. The paper
    # does not define a self-interaction; it is dynamically irrelevant and 0.
    active_slots = np.arange(active_count, dtype=np.int32)
    invaded_index = (
        live_species.index(invaded_species) if invaded_species not in (None, 0) else -1
    )
    for attempt in range(max_cycle_rejection_attempts):
        new_edges = random_source.random(2 * active_count) < gamma
        new_Gamma[-1, :-1] = new_edges[:active_count]
        new_Gamma[:-1, -1] = new_edges[active_count:]
        new_Gamma[-1, -1] = 0
        if invaded_index >= 0:
            new_Gamma[-1, invaded_index] = 1
        if not max_cycle_length or not _new_species_closes_forbidden_cycle(
            new_Gamma, active_slots, active_count, active_count, max_cycle_length
        ):
            break
    else:
        raise RuntimeError(
            "Species introduction rejection exceeded max_cycle_rejection_attempts; "
            "decrease gamma or increase the attempt limit"
        )

    live_species.append(new_species)
    newest_species = new_species

    return new_Gamma, live_species, newest_species


def _validate_simulation_options(
    alpha, T, track_every, progress, populate_first_100
):
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
    ):
        raise TypeError("alpha must be a real number")
    if alpha < 0:
        raise ValueError("alpha cannot be negative")

    for name, value in (("T", T), ("track_every", track_every)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
    if T < 0:
        raise ValueError("T cannot be negative")
    if track_every <= 0:
        raise ValueError("track_every must be positive")
    if not isinstance(progress, (bool, np.bool_)):
        raise TypeError("progress must be True or False")
    if not isinstance(populate_first_100, (bool, np.bool_)):
        raise TypeError("populate_first_100 must be True or False")


def _validate_gamma(gamma):
    if (
        isinstance(gamma, (bool, np.bool_))
        or not isinstance(gamma, (int, float, np.integer, np.floating))
    ):
        raise TypeError("gamma must be a real number")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")


@njit(cache=True)
def _grow_species_storage(
    Gamma,
    active_slots,
    slot_species_ids,
    species_counts,
    slot_to_active_position,
    free_slots,
):
    """Double reusable species-slot storage when concurrent diversity grows."""
    old_capacity = Gamma.shape[0]
    new_capacity = max(4, old_capacity * 2)

    grown_Gamma = np.zeros((new_capacity, new_capacity), dtype=np.uint8)
    grown_Gamma[:old_capacity, :old_capacity] = Gamma

    grown_active_slots = np.empty(new_capacity, dtype=np.int32)
    grown_active_slots[:old_capacity] = active_slots

    grown_species_ids = np.full(new_capacity, -1, dtype=np.int64)
    grown_species_ids[:old_capacity] = slot_species_ids

    grown_counts = np.zeros(new_capacity, dtype=np.int64)
    grown_counts[:old_capacity] = species_counts

    grown_positions = np.full(new_capacity, -1, dtype=np.int32)
    grown_positions[:old_capacity] = slot_to_active_position

    grown_free_slots = np.empty(new_capacity, dtype=np.int32)
    grown_free_slots[:old_capacity] = free_slots

    return (
        grown_Gamma,
        grown_active_slots,
        grown_species_ids,
        grown_counts,
        grown_positions,
        grown_free_slots,
    )


@njit(cache=True, inline="never")
def _has_introduction(draws, probability, forced_timesteps, timestep):
    """Check already drawn trials without enlarging the compiled invasion loop."""
    if timestep <= forced_timesteps:
        return True
    if probability > 0:
        for draw in draws:
            if draw < probability:
                return True
    return False


@njit(cache=True)
def _run_compiled_simulation(
    slot_lattice,
    usable_sites,
    neighbours,
    rng,
    Gamma,
    active_slots,
    slot_species_ids,
    species_counts,
    slot_to_active_position,
    free_slots,
    live_count,
    next_unused_slot,
    free_count,
    newest_species,
    gamma,
    introduction_probability,
    forced_introduction_timesteps,
    timesteps,
    track_every,
    max_forbidden_cycle_length=0,
    max_cycle_rejection_attempts=1_000_000,
):
    """Execute sequential stochastic updates in compiled machine code."""
    total_sites = slot_lattice.size
    usable_count = usable_sites.size
    records = timesteps // track_every
    if timesteps % track_every:
        records += 1
    diversity_history = np.empty(records, dtype=np.int64)
    record_index = 0
    flat_neighbours = neighbours.ravel()

    for timestep in range(1, timesteps + 1):
        # Each code uniformly selects one of the N sites and one of its four
        # neighbors. This is distributionally identical to two separate draws.
        event_codes = rng.integers(0, 4 * total_sites, size=total_sites)
        introduction_draws = rng.random(total_sites)

        # With no species, or one species occupying every usable site, all
        # invasions are ineffective. Keep both complete random draws above,
        # and skip only time units without any introduction. Empty usable
        # sites disable the single-species shortcut because they can be filled.
        if live_count == 0 or (
            live_count == 1
            and species_counts[active_slots[0]] == usable_count
        ):
            if not _has_introduction(
                introduction_draws,
                introduction_probability,
                forced_introduction_timesteps,
                timestep,
            ):
                if timestep % track_every == 0 or timestep == timesteps:
                    diversity_history[record_index] = live_count
                    record_index += 1
                continue

        for event in range(total_sites):
            source_site = event_codes[event] >> 2
            source_slot = slot_lattice[source_site]

            if source_slot >= 0:
                # The code is already 4 * source_site + direction.
                target_site = flat_neighbours[event_codes[event]]
                target_slot = slot_lattice[target_site]

                if target_slot == -1:  # Empty, usable site.
                    slot_lattice[target_site] = source_slot
                    species_counts[source_slot] += 1
                elif (
                    target_slot >= 0
                    and target_slot != source_slot
                    and Gamma[source_slot, target_slot] != 0
                ):
                    slot_lattice[target_site] = source_slot
                    species_counts[source_slot] += 1
                    species_counts[target_slot] -= 1

                    if species_counts[target_slot] == 0:
                        extinct_position = slot_to_active_position[target_slot]
                        last_slot = active_slots[live_count - 1]
                        active_slots[extinct_position] = last_slot
                        slot_to_active_position[last_slot] = extinct_position
                        live_count -= 1
                        slot_to_active_position[target_slot] = -1
                        slot_species_ids[target_slot] = -1
                        free_slots[free_count] = target_slot
                        free_count += 1

            # A Bernoulli introduction trial follows every invasion attempt.
            if (
                timestep <= forced_introduction_timesteps and event == 0
            ) or introduction_draws[event] < introduction_probability:
                introduction_site = usable_sites[
                    rng.integers(0, usable_count)
                ]
                replaced_slot = slot_lattice[introduction_site]

                if free_count:
                    free_count -= 1
                    new_slot = free_slots[free_count]
                else:
                    if next_unused_slot == Gamma.shape[0]:
                        (
                            Gamma,
                            active_slots,
                            slot_species_ids,
                            species_counts,
                            slot_to_active_position,
                            free_slots,
                        ) = _grow_species_storage(
                            Gamma,
                            active_slots,
                            slot_species_ids,
                            species_counts,
                            slot_to_active_position,
                            free_slots,
                        )
                    new_slot = next_unused_slot
                    next_unused_slot += 1

                newest_species += 1
                slot_species_ids[new_slot] = newest_species
                species_counts[new_slot] = 1

                # Draw all candidate relationships in the model's original
                # order. With the optional restriction, reject the whole
                # candidate and retry at the same gamma. Existing edges and
                # the introduction site stay fixed throughout these retries.
                attempts = 0
                while True:
                    for position in range(live_count):
                        other_slot = active_slots[position]
                        Gamma[new_slot, other_slot] = rng.random() < gamma
                    Gamma[new_slot, new_slot] = 0
                    for position in range(live_count):
                        other_slot = active_slots[position]
                        Gamma[other_slot, new_slot] = rng.random() < gamma

                    if replaced_slot >= 0:
                        Gamma[new_slot, replaced_slot] = 1
                    if not max_forbidden_cycle_length or not _new_species_closes_forbidden_cycle(
                        Gamma, active_slots, live_count, new_slot,
                        max_forbidden_cycle_length,
                    ):
                        break
                    attempts += 1
                    if attempts >= max_cycle_rejection_attempts:
                        raise RuntimeError(
                            "Species introduction rejection exceeded "
                            "max_cycle_rejection_attempts; decrease gamma "
                            "or increase the attempt limit"
                        )

                active_slots[live_count] = new_slot
                slot_to_active_position[new_slot] = live_count
                live_count += 1
                slot_lattice[introduction_site] = new_slot

                if replaced_slot >= 0:
                    species_counts[replaced_slot] -= 1
                    if species_counts[replaced_slot] == 0:
                        extinct_position = slot_to_active_position[replaced_slot]
                        last_slot = active_slots[live_count - 1]
                        active_slots[extinct_position] = last_slot
                        slot_to_active_position[last_slot] = extinct_position
                        live_count -= 1
                        slot_to_active_position[replaced_slot] = -1
                        slot_species_ids[replaced_slot] = -1
                        free_slots[free_count] = replaced_slot
                        free_count += 1

        if timestep % track_every == 0 or timestep == timesteps:
            diversity_history[record_index] = live_count
            record_index += 1

    return (
        slot_lattice,
        Gamma,
        active_slots,
        slot_species_ids,
        species_counts,
        slot_to_active_position,
        free_slots,
        live_count,
        next_unused_slot,
        free_count,
        newest_species,
        diversity_history,
    )


def _tracking_timesteps(T, track_every):
    tracked = np.arange(track_every, T + 1, track_every, dtype=np.int64)
    if T > 0 and (tracked.size == 0 or tracked[-1] != T):
        tracked = np.append(tracked, T)
    return tracked


def _export_simulation_state(state, rows, columns):
    """Convert reusable internal slots to the small public state format."""
    (
        slot_lattice,
        Gamma,
        active_slots,
        slot_species_ids,
        species_counts,
        slot_to_active_position,
        free_slots,
        live_count,
        next_unused_slot,
        free_count,
        newest_species,
    ) = state
    del species_counts, slot_to_active_position, free_slots
    del next_unused_slot, free_count

    live_slots = active_slots[:live_count].astype(np.intp)
    current_species = slot_species_ids[live_slots].astype(np.int64).tolist()
    # Advanced indexing already produces an independent copy.
    compact_Gamma = Gamma[np.ix_(live_slots, live_slots)]

    external_flat = np.zeros(slot_lattice.size, dtype=np.int64)
    external_flat[slot_lattice == -2] = -1
    occupied_sites = slot_lattice >= 0
    external_flat[occupied_sites] = slot_species_ids[
        slot_lattice[occupied_sites]
    ]
    return {
        "lattice": external_flat.reshape(rows, columns),
        "Gamma": compact_Gamma,
        "current_species": current_species,
        "diversity": int(live_count),
        "newest_species": int(newest_species),
    }


def _simulate_lattice(
    lattice,
    current_species,
    newest_species,
    gamma,
    alpha,
    T,
    track_every,
    rng,
    progress,
    populate_first_100,
    initial_Gamma=None,
    initial_timestep=0,
    prior_tracked_timesteps=None,
    prior_diversity_history=None,
    prior_patch_history=None,
    initial_newest_species=None,
    checkpoint_dir=None,
    retention=None,
    target_timestep=None,
    geometry_metadata=None,
    max_forbidden_cycle_length=None,
    max_cycle_rejection_attempts=1_000_000,
):
    """Prepare state and run the exact sequential model in compiled code."""
    cycle_limit = _validate_cycle_rejection_options(
        max_forbidden_cycle_length, max_cycle_rejection_attempts
    )
    cycle_options = {
        "max_forbidden_cycle_length": cycle_limit,
        "max_cycle_rejection_attempts": int(max_cycle_rejection_attempts),
    } if cycle_limit else {}
    geometry_metadata = {
        **({} if geometry_metadata is None else geometry_metadata),
        "max_forbidden_cycle_length": cycle_limit or None,
        "max_cycle_rejection_attempts": int(max_cycle_rejection_attempts),
    }
    if not _NUMBA_AVAILABLE:
        raise ImportError(
            "Fast simulations require numba; install it with 'pip install numba'"
        )

    if initial_Gamma is None:
        initial_Gamma = update_Gamma(
            None,
            gamma,
            current_species,
            newest_species,
            invaded_species=None,
            rng=rng,
            **cycle_options,
        )
    else:
        initial_Gamma = np.asarray(initial_Gamma, dtype=np.uint8)
        if cycle_limit and _has_forbidden_directed_cycle(initial_Gamma, cycle_limit):
            raise ValueError("initial Gamma contains a forbidden directed cycle")

    rows, columns = lattice.shape
    total_sites = lattice.size
    external_flat = lattice.ravel()
    usable_sites = np.flatnonzero(external_flat != -1).astype(np.intp)
    if cycle_limit and gamma == 1 and usable_sites.size and T and (
        alpha > 0 or (populate_first_100 and initial_timestep < 100)
    ):
        raise ValueError("gamma=1 cannot support repeated introductions without forbidden cycles")
    introduction_probability = alpha * gamma / total_sites
    if usable_sites.size == 0:
        introduction_probability = 0.0
    if introduction_probability > 1:
        raise ValueError(
            "alpha * gamma / N cannot exceed 1; reduce alpha or gamma"
        )
    forced_introduction_timesteps = 0
    if populate_first_100 and usable_sites.size:
        forced_introduction_timesteps = min(
            int(T), max(0, 100 - int(initial_timestep))
        )
    if target_timestep is None:
        target_timestep = int(initial_timestep) + int(T)
    target_timestep = int(target_timestep)
    if target_timestep != int(initial_timestep) + int(T):
        raise ValueError("target_timestep must equal initial timestep plus T")
    if initial_newest_species is None:
        initial_newest_species = newest_species
    initial_newest_species = int(initial_newest_species)
    if checkpoint_dir is not None and T:
        _validate_checkpoint_destination(checkpoint_dir, initial_timestep)

    # Internal lattice values are reusable Gamma slots: >=0 is a species,
    # -1 is empty, and -2 is permanently blocked. Permanent species IDs are
    # stored separately, so old IDs never force Gamma to grow.
    slot_lattice = np.full(total_sites, -1, dtype=np.int32)
    slot_lattice[external_flat == -1] = -2
    species_to_slot = {
        int(species): slot for slot, species in enumerate(current_species)
    }
    occupied_sites = external_flat > 0
    occupied_species, inverse_slots = np.unique(
        external_flat[occupied_sites], return_inverse=True
    )
    if set(species_to_slot) != set(occupied_species.tolist()):
        raise ValueError("current_species does not match the species in lattice")
    # Map the sorted IDs back to the supplied species order. This keeps Gamma
    # slots identical while moving the per-site work out of Python.
    ordered_slots = np.asarray(
        [species_to_slot[int(species)] for species in occupied_species],
        dtype=np.int32,
    )
    slot_lattice[occupied_sites] = ordered_slots[inverse_slots]
    del occupied_sites, occupied_species, inverse_slots, ordered_slots
    del external_flat, species_to_slot

    # The table is read at random for every invasion attempt. Smaller indices
    # halve its memory footprint and improve cache use on ordinary lattices.
    site_index_dtype = (
        np.int32 if total_sites <= np.iinfo(np.int32).max else np.intp
    )
    site_numbers = np.arange(total_sites, dtype=site_index_dtype).reshape(
        rows, columns
    )
    neighbours = np.empty((total_sites, 4), dtype=site_index_dtype)
    neighbours[:, 0] = np.roll(site_numbers, 1, axis=0).ravel()
    neighbours[:, 1] = np.roll(site_numbers, -1, axis=1).ravel()
    neighbours[:, 2] = np.roll(site_numbers, -1, axis=0).ravel()
    neighbours[:, 3] = np.roll(site_numbers, 1, axis=1).ravel()
    del site_numbers

    live_count = len(current_species)
    capacity = 4
    while capacity < live_count:
        capacity *= 2

    Gamma = np.zeros((capacity, capacity), dtype=np.uint8)
    Gamma[:live_count, :live_count] = initial_Gamma
    active_slots = np.empty(capacity, dtype=np.int32)
    active_slots[:live_count] = np.arange(live_count, dtype=np.int32)
    slot_species_ids = np.full(capacity, -1, dtype=np.int64)
    slot_species_ids[:live_count] = np.asarray(current_species, dtype=np.int64)
    species_counts = np.zeros(capacity, dtype=np.int64)
    if live_count:
        species_counts[:live_count] = np.bincount(
            slot_lattice[slot_lattice >= 0], minlength=live_count
        )
    slot_to_active_position = np.full(capacity, -1, dtype=np.int32)
    slot_to_active_position[:live_count] = np.arange(live_count, dtype=np.int32)
    free_slots = np.empty(capacity, dtype=np.int32)
    next_unused_slot = live_count
    free_count = 0

    if prior_tracked_timesteps is None:
        prior_tracked_timesteps = np.array(
            [int(initial_timestep)], dtype=np.int64
        )
    else:
        prior_tracked_timesteps = np.asarray(
            prior_tracked_timesteps, dtype=np.int64
        ).copy()
    if prior_diversity_history is None:
        prior_diversity_history = np.array(
            [len(current_species)], dtype=np.int64
        )
    else:
        prior_diversity_history = np.asarray(
            prior_diversity_history, dtype=np.int64
        ).copy()

    if prior_patch_history is None:
        prior_patch_history = np.full(
            prior_tracked_timesteps.size, _MISSING_PATCH_COUNT, dtype=np.int64
        )
    else:
        prior_patch_history = _align_patch_history(
            prior_patch_history, prior_tracked_timesteps.size
        )
    if checkpoint_dir is not None and prior_patch_history.size:
        # The starting lattice is in hand, so its patch count costs nothing
        # here and makes the sample the leg resumes from safe to delete.
        prior_patch_history[-1] = count_patches(lattice)
    del lattice

    if retention is None:
        retention = RetentionPolicy()
    snapshot_retention = None
    if checkpoint_dir is not None:
        snapshot_retention = _CheckpointRetention(checkpoint_dir, retention)
        snapshot_retention.adopt_existing()
        # Retention grants around already detected swaps survive a restart.
        # A later estimate cannot revoke them: some intervening snapshots may
        # already have been pruned under that earlier decision.
        existing_analysis = load_analysis(checkpoint_dir)
        if existing_analysis is not None:
            recorded_times = np.asarray(existing_analysis["tracked_timesteps"])
            overlap = min(recorded_times.size, prior_tracked_timesteps.size)
            if overlap and np.array_equal(
                recorded_times[:overlap], prior_tracked_timesteps[:overlap]
            ):
                snapshot_retention.events = np.asarray(
                    existing_analysis["event_timesteps"], dtype=np.int64
                )[
                    existing_analysis["event_timesteps"] <= int(initial_timestep)
                ].copy()

    progress_bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as error:
            raise ImportError(
                "progress=True requires tqdm; install it with 'pip install tqdm'"
            ) from error
        progress_bar = tqdm(total=int(T), desc="Simulation", unit="time unit")

    state = (
        slot_lattice,
        Gamma,
        active_slots,
        slot_species_ids,
        species_counts,
        slot_to_active_position,
        free_slots,
        live_count,
        next_unused_slot,
        free_count,
        newest_species,
    )

    new_timesteps = int(initial_timestep) + _tracking_timesteps(
        int(T), int(track_every)
    )
    prior_sample_count = prior_tracked_timesteps.size
    sample_count = prior_sample_count + new_timesteps.size
    tracked_timesteps = np.empty(sample_count, dtype=np.int64)
    tracked_timesteps[:prior_sample_count] = prior_tracked_timesteps
    tracked_timesteps[prior_sample_count:] = new_timesteps
    diversity_history = np.empty(sample_count, dtype=np.int64)
    diversity_history[:prior_sample_count] = prior_diversity_history
    patch_history = np.full(
        sample_count, _MISSING_PATCH_COUNT, dtype=np.int64
    )
    patch_history[:prior_sample_count] = prior_patch_history
    recorded_samples = prior_sample_count
    checkpoint_files = []
    removed_files = set()

    if not progress and checkpoint_dir is None:
        if T:
            result = _run_compiled_simulation(
                state[0],
                usable_sites,
                neighbours,
                rng,
                *state[1:],
                gamma,
                introduction_probability,
                forced_introduction_timesteps,
                int(T),
                int(track_every),
                **cycle_options,
            )
            state = result[:-1]
            diversity_history[recorded_samples:] = result[-1]
            recorded_samples = sample_count
    else:
        previous_timestep = 0
        if checkpoint_dir is not None:
            # A state can only be written after compiled work returns, so end
            # each chunk exactly where a snapshot is requested.
            chunk_limit = int(track_every)
        else:
            # Limit expensive Python/UI refreshes to roughly 1,000.
            minimum_chunk = max(1, (int(T) + 999) // 1000)
            chunk_limit = max(int(track_every), minimum_chunk)
            chunk_limit = (
                (chunk_limit + int(track_every) - 1) // int(track_every)
            ) * int(track_every)
        try:
            while previous_timestep < T:
                chunk_size = min(
                    chunk_limit, int(T) - previous_timestep
                )
                result = _run_compiled_simulation(
                    state[0],
                    usable_sites,
                    neighbours,
                    rng,
                    *state[1:],
                    gamma,
                    introduction_probability,
                    min(forced_introduction_timesteps, chunk_size),
                    chunk_size,
                    int(track_every),
                    **cycle_options,
                )
                state = result[:-1]
                next_recorded_samples = recorded_samples + result[-1].size
                diversity_history[recorded_samples:next_recorded_samples] = (
                    result[-1]
                )
                recorded_samples = next_recorded_samples
                if progress_bar is not None:
                    progress_bar.update(chunk_size)
                forced_introduction_timesteps = max(
                    0, forced_introduction_timesteps - chunk_size
                )
                previous_timestep += chunk_size

                if checkpoint_dir is not None:
                    exported = _export_simulation_state(
                        state, rows, columns
                    )
                    # Only expose the filled prefix. Earlier samples stay in
                    # place instead of being recopied after every checkpoint.
                    tracked_so_far = tracked_timesteps[:recorded_samples]
                    diversity_so_far = diversity_history[:recorded_samples]
                    snapshot_timestep = (
                        int(initial_timestep) + previous_timestep
                    )
                    exported.update(
                        {
                            "timestep": snapshot_timestep,
                            "target_timestep": target_timestep,
                            "tracked_timesteps": tracked_so_far,
                            "diversity_history": diversity_so_far,
                            # Preserve the known prefix even if the subsequent
                            # atomic analysis write is interrupted. The current
                            # snapshot is counted immediately after this write.
                            "patch_history": patch_history[:recorded_samples],
                            "initial_newest_species": initial_newest_species,
                            "rng_state": rng.bit_generator.state,
                            "gamma": float(gamma),
                            "alpha": float(alpha),
                            "track_every": int(track_every),
                            "populate_first_100": bool(populate_first_100),
                            **geometry_metadata,
                        }
                    )
                    written_file = _write_checkpoint(checkpoint_dir, exported)
                    checkpoint_files.append(written_file)

                    # Patches are counted here, while the lattice is in
                    # memory, because retention deletes most snapshots and
                    # the count could not be recovered afterwards.
                    patch_history[recorded_samples - 1] = count_patches(
                        exported["lattice"]
                    )
                    patch_so_far = patch_history[:recorded_samples]
                    removed_files.update(
                        snapshot_retention.record(
                            written_file,
                            snapshot_timestep,
                            tracked_so_far,
                            diversity_so_far,
                            patch_so_far,
                        )
                    )
                    _write_analysis(
                        checkpoint_dir,
                        {
                            "tracked_timesteps": tracked_so_far,
                            "diversity_history": diversity_so_far,
                            "patch_history": patch_so_far,
                            "event_timesteps": snapshot_retention.events,
                            "initial_newest_species": initial_newest_species,
                            "timestep": snapshot_timestep,
                            "target_timestep": target_timestep,
                            "gamma": float(gamma),
                            "alpha": float(alpha),
                            "track_every": int(track_every),
                            "populate_first_100": bool(populate_first_100),
                            "lattice_shape": (rows, columns),
                            **geometry_metadata,
                        },
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

    analysis_path = None
    event_timesteps = np.empty(0, dtype=np.int64)
    if snapshot_retention is not None:
        # The last detection pass sees the whole leg, so a swap that was
        # still inside the rolling window at the final snapshot is caught.
        removed_files.update(
            snapshot_retention.finish(
                tracked_timesteps, diversity_history, patch_history
            )
        )
        event_timesteps = snapshot_retention.events
        analysis_path = _write_analysis(
            checkpoint_dir,
            {
                "tracked_timesteps": tracked_timesteps,
                "diversity_history": diversity_history,
                "patch_history": patch_history,
                "event_timesteps": event_timesteps,
                "initial_newest_species": initial_newest_species,
                "timestep": target_timestep,
                "target_timestep": target_timestep,
                "gamma": float(gamma),
                "alpha": float(alpha),
                "track_every": int(track_every),
                "populate_first_100": bool(populate_first_100),
                "lattice_shape": (rows, columns),
                **geometry_metadata,
            },
        )
        checkpoint_files = [
            path for path in checkpoint_files if path not in removed_files
        ]

    final_result = _export_simulation_state(state, rows, columns)
    final_result.update(geometry_metadata)
    final_result.update(
        {
            "introduction_probability": float(introduction_probability),
            "gamma": float(gamma),
            "alpha": float(alpha),
            "track_every": int(track_every),
            "populate_first_100": bool(populate_first_100),
            "forced_initial_introductions": min(
                int(T), max(0, 100 - int(initial_timestep))
            )
            if populate_first_100 and usable_sites.size
            else 0,
            "start_timestep": int(initial_timestep),
            "timestep": target_timestep,
            "target_timestep": target_timestep,
            "tracked_timesteps": tracked_timesteps,
            "diversity_history": diversity_history,
            "patch_history": patch_history,
            "initial_newest_species": initial_newest_species,
            "event_timesteps": event_timesteps,
            "rng_state": rng.bit_generator.state,
            "checkpoint_files": checkpoint_files,
            "removed_checkpoint_files": sorted(removed_files),
            "analysis_path": analysis_path,
        }
    )
    return final_result


def simulation_from_state(
    initial_state,
    gamma,
    alpha,
    T,
    track_every=1,
    seed=None,
    progress=False,
    populate_first_100=False,
    checkpoint_dir=None,
    retention=None,
    target_timestep=None,
    continue_history=None,
    max_forbidden_cycle_length=None,
    max_cycle_rejection_attempts=None,
):
    """Run ``T`` additional time units from a result or saved checkpoint.

    Existing interactions in ``Gamma`` are kept. ``gamma`` controls only the
    interactions drawn for species introduced during this new simulation leg.
    A saved RNG is restored automatically; ``seed`` is used for raw states
    that do not contain one. When raw arrays omit ``current_species``, Gamma
    rows are assumed to follow the ascending positive IDs in the lattice.

    ``continue_history`` decides whether the state's recorded history is
    carried into this leg. By default it is carried only when the leg writes
    to the folder the state came from, which is an extension of that run. A
    leg written anywhere else is a branch: its history, analysis record and
    species-introduced count start at the state's timestep, which keeps its
    original simulation time.
    """
    _validate_simulation_options(
        alpha, T, track_every, progress, populate_first_100
    )
    _validate_gamma(gamma)
    state = _normalize_initial_state(initial_state)
    if max_forbidden_cycle_length is None:
        max_forbidden_cycle_length = state.get("max_forbidden_cycle_length")
    if max_cycle_rejection_attempts is None:
        max_cycle_rejection_attempts = state.get("max_cycle_rejection_attempts", 1_000_000)
    _validate_cycle_rejection_options(max_forbidden_cycle_length, max_cycle_rejection_attempts)
    pruning_source = state.get("_cluster_pruning_source_directory")
    if pruning_source is not None and checkpoint_dir is not None and _same_directory(
        pruning_source, checkpoint_dir
    ):
        raise ValueError("cluster pruning requires a new checkpoint_dir")
    if continue_history is None:
        continue_history = _same_directory(
            state["source_directory"], checkpoint_dir
        )
    elif not isinstance(continue_history, (bool, np.bool_)):
        raise TypeError("continue_history must be True, False, or None")
    if not continue_history:
        # The branch keeps only the sample it starts from.
        for key in ("tracked_timesteps", "diversity_history", "patch_history"):
            state[key] = state[key][-1:].copy()
        state["initial_newest_species"] = state["newest_species"]
    rng = _make_rng(seed, state["rng_state"])
    if target_timestep is None:
        target_timestep = state["timestep"] + int(T)
    result = _simulate_lattice(
        state["lattice"],
        state["current_species"],
        state["newest_species"],
        gamma,
        alpha,
        T,
        track_every,
        rng,
        progress,
        populate_first_100,
        initial_Gamma=state["Gamma"],
        initial_timestep=state["timestep"],
        prior_tracked_timesteps=state["tracked_timesteps"],
        prior_diversity_history=state["diversity_history"],
        prior_patch_history=state["patch_history"],
        initial_newest_species=state["initial_newest_species"],
        checkpoint_dir=checkpoint_dir,
        retention=retention,
        target_timestep=target_timestep,
        geometry_metadata=_percolation_metadata(state),
        max_forbidden_cycle_length=max_forbidden_cycle_length,
        max_cycle_rejection_attempts=max_cycle_rejection_attempts,
    )
    if checkpoint_dir is None and pruning_source is not None:
        result["_cluster_pruning_source_directory"] = pruning_source
    return result


def main_simulation(
    L_col,
    L_row,
    D,
    gamma,
    alpha,
    T,
    track_every=1,
    seed=None,
    progress=False,
    populate_first_100=False,
    initial_state=None,
    checkpoint_dir=None,
    retention=None,
    continue_history=None,
    max_forbidden_cycle_length=None,
    max_cycle_rejection_attempts=1_000_000,
):
    """Run the spatial invasion simulation.

    One model time unit contains ``N = L_col * L_row`` microscopic updates.
    Diversity is recorded at time 0, every ``track_every`` time units, and at
    the final time. The lattice has periodic boundaries.

    After each microscopic invasion attempt, an already-successful new species
    is introduced with the paper's probability ``alpha * gamma / N``. Thus,
    introductions average ``alpha * gamma`` per model time unit. Supplying
    ``seed`` makes a run reproducible.
    Set ``populate_first_100=True`` to force one new-species introduction
    during each of the first 100 model time units, in addition to ordinary
    stochastic introductions.
    Set ``progress=True`` to display a progress bar. For long runs, visual
    refreshes are capped at roughly 1,000 while every requested diversity
    sample is still recorded.
    Set ``checkpoint_dir`` to save one state at every tracking interval. Pass
    a previous result, checkpoint dictionary, or checkpoint path as
    ``initial_state`` to run ``T`` additional time units from that state; see
    ``simulation_from_state`` for when its history is carried over.
    """
    _validate_simulation_options(
        alpha, T, track_every, progress, populate_first_100
    )
    _validate_gamma(gamma)
    _validate_cycle_rejection_options(max_forbidden_cycle_length, max_cycle_rejection_attempts)
    if initial_state is not None:
        return simulation_from_state(
            initial_state,
            gamma=gamma,
            alpha=alpha,
            T=T,
            track_every=track_every,
            seed=seed,
            progress=progress,
            populate_first_100=populate_first_100,
            checkpoint_dir=checkpoint_dir,
            retention=retention,
            continue_history=continue_history,
            max_forbidden_cycle_length=max_forbidden_cycle_length,
            max_cycle_rejection_attempts=max_cycle_rejection_attempts,
        )

    rng = _make_rng(seed)
    lattice, current_species, _, newest_species = create_lattice(
        L_col, L_row, D, rng=rng
    )
    return _simulate_lattice(
        lattice,
        current_species,
        newest_species,
        gamma,
        alpha,
        T,
        track_every,
        rng,
        progress,
        populate_first_100,
        checkpoint_dir=checkpoint_dir,
        retention=retention,
        max_forbidden_cycle_length=max_forbidden_cycle_length,
        max_cycle_rejection_attempts=max_cycle_rejection_attempts,
    )


@njit(cache=True)
def _largest_usable_cluster_mask(usable):
    """Select the largest four-neighbour component on a periodic lattice.

    Equal-sized components are resolved by their first row-major site. This
    uses no randomness and treats empty and occupied usable sites alike.
    """
    rows, columns = usable.shape
    parent = np.arange(usable.size, dtype=np.int64)
    sizes = np.ones(usable.size, dtype=np.int64)
    for row in range(rows):
        for column in range(columns):
            if not usable[row, column]:
                continue
            site = row * columns + column
            right = (column + 1) % columns
            down = (row + 1) % rows
            if usable[row, right]:
                _merge_patch_sites(parent, sizes, site, row * columns + right)
            if usable[down, column]:
                _merge_patch_sites(parent, sizes, site, down * columns + column)

    best_root = -1
    best_size = 0
    for site in range(usable.size):
        if usable[site // columns, site % columns]:
            root = _patch_root(parent, site)
            if sizes[root] > best_size:
                best_root = root
                best_size = sizes[root]
    selected = np.zeros(usable.shape, dtype=np.bool_)
    for site in range(usable.size):
        if usable[site // columns, site % columns]:
            selected[site // columns, site % columns] = (
                _patch_root(parent, site) == best_root
            )
    return selected


def keep_largest_cluster(initial_state):
    """Copy a state, block smaller usable components, and start a new history.

    Connectivity follows the model's four neighbours and periodic boundaries,
    regardless of species. Empty sites (0) also connect; blocked sites (-1) do
    not. Gamma is restricted to surviving species in their original order.
    Species IDs, simulation time, and the saved random state are preserved.
    The source checkpoint and input arrays are never modified.
    """
    if isinstance(initial_state, (str, os.PathLike)):
        initial_state = load_checkpoint(initial_state)
    state = _normalize_initial_state(initial_state)
    selected = _largest_usable_cluster_mask(state["lattice"] != -1)
    current_usable = int(np.count_nonzero(state["lattice"] != -1))
    original_usable = current_usable
    if state.get("largest_cluster_only"):
        original_usable = int(state.get("original_usable_sites", original_usable))
    state["lattice"][~selected] = -1
    surviving_ids = set(np.unique(state["lattice"][state["lattice"] > 0]).tolist())
    surviving_positions = [
        index for index, species in enumerate(state["current_species"])
        if species in surviving_ids
    ]
    state["Gamma"] = state["Gamma"][np.ix_(surviving_positions, surviving_positions)]
    state["current_species"] = [state["current_species"][index] for index in surviving_positions]
    state["rng_state"] = deepcopy(state["rng_state"])
    state["tracked_timesteps"] = np.array([state["timestep"]], dtype=np.int64)
    state["diversity_history"] = np.array([len(surviving_positions)], dtype=np.int64)
    state["patch_history"] = np.array([_MISSING_PATCH_COUNT], dtype=np.int64)
    state["initial_newest_species"] = state["newest_species"]
    usable = int(np.count_nonzero(selected))
    # Explicit preparation resets histories even if the mask was connected.
    # It must never replace the original checkpoint folder's analysis record.
    if state["source_directory"] is not None:
        state["_cluster_pruning_source_directory"] = state["source_directory"]
    state.update({
        "diversity": len(surviving_positions),
        "largest_cluster_only": True,
        "original_usable_sites": original_usable,
        "usable_sites": usable,
        "removed_cluster_sites": original_usable - usable,
        "blocked_sites": selected.size - usable,
        "effective_p": (selected.size - usable) / selected.size,
    })
    result = {**initial_state, **state}
    # A prepared state carries provenance, but never the old run's plot data.
    for key in ("checkpoint_path", "analysis_path", "checkpoint_files",
                "removed_checkpoint_files", "event_timesteps", "batch_metadata"):
        result.pop(key, None)
    return result


def percolation_simulation(
    L_col,
    L_row,
    D,
    gamma,
    alpha,
    T,
    p,
    track_every=1,
    seed=None,
    progress=False,
    populate_first_100=False,
    initial_state=None,
    checkpoint_dir=None,
    retention=None,
    continue_history=None,
    largest_cluster_only=True,
    max_forbidden_cycle_length=None,
    max_cycle_rejection_attempts=1_000_000,
):
    """Run the simulation with permanent random site removal.

    This is an extension of the paper's base model. Each site is independently
    blocked with probability ``p`` before initialization. By default only the
    largest remaining periodic four-neighbour component is kept; all other
    sites are permanently blocked too. Set ``largest_cluster_only=False`` to
    retain the earlier independent-removal behavior. Blocked sites remain
    -1 and cannot invade, be invaded, or receive introductions. A blocked
    source draw is simply a null event, so one time unit still contains ``N``
    microscopic updates and uses introduction probability ``alpha*gamma/N``.
    """
    _validate_simulation_options(
        alpha, T, track_every, progress, populate_first_100
    )
    _validate_gamma(gamma)
    _validate_cycle_rejection_options(max_forbidden_cycle_length, max_cycle_rejection_attempts)
    if (
        isinstance(p, (bool, np.bool_))
        or not isinstance(p, (int, float, np.integer, np.floating))
    ):
        raise TypeError("p must be a real number")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if not isinstance(largest_cluster_only, (bool, np.bool_)):
        raise TypeError("largest_cluster_only must be a boolean")

    if initial_state is not None:
        # Site removal is an initialization rule. Existing -1 cells are kept;
        # p is not drawn again when branching from an existing geometry.
        state = _normalize_initial_state(initial_state)
        original_usable = int(np.count_nonzero(state["lattice"] != -1))
        if largest_cluster_only:
            prepared = keep_largest_cluster(state)
            if prepared["usable_sites"] < original_usable:
                if continue_history is not None and not isinstance(
                    continue_history, (bool, np.bool_)
                ):
                    raise TypeError("continue_history must be True, False, or None")
                if continue_history:
                    raise ValueError("cluster pruning changes the geometry; use a new history")
                if checkpoint_dir is not None and _same_directory(
                    state["source_directory"], checkpoint_dir
                ):
                    raise ValueError("cluster pruning requires a new checkpoint_dir")
                state = prepared
                continue_history = False
            else:
                state.update(_percolation_metadata(prepared))
        else:
            original_usable = int(state.get("original_usable_sites", original_usable))
            usable = int(np.count_nonzero(state["lattice"] != -1))
            state.update({
                "largest_cluster_only": False,
                "original_usable_sites": original_usable,
                "usable_sites": usable,
                "removed_cluster_sites": original_usable - usable,
                "blocked_sites": state["lattice"].size - usable,
                "effective_p": (state["lattice"].size - usable) / state["lattice"].size,
            })
        state.update({"p": float(p), "p_applied": False})
        return simulation_from_state(
            state,
            gamma=gamma,
            alpha=alpha,
            T=T,
            track_every=track_every,
            seed=seed,
            progress=progress,
            populate_first_100=populate_first_100,
            checkpoint_dir=checkpoint_dir,
            retention=retention,
            continue_history=continue_history,
            max_forbidden_cycle_length=max_forbidden_cycle_length,
            max_cycle_rejection_attempts=max_cycle_rejection_attempts,
        )

    # Reuse create_lattice for input validation without consuming randomness.
    values = (L_col, L_row, D)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError("L_col, L_row, and D must be integers")
    if L_col <= 0 or L_row <= 0:
        raise ValueError("L_col and L_row must be positive")
    if D < 0:
        raise ValueError("D cannot be negative")

    rng = _make_rng(seed)
    total_sites = L_row * L_col
    blocked = rng.random(total_sites) < p
    original_usable = int(np.count_nonzero(~blocked))
    if largest_cluster_only:
        blocked = ~_largest_usable_cluster_mask((~blocked).reshape(L_row, L_col)).ravel()
    active_sites = np.flatnonzero(~blocked)
    geometry_metadata = {
        "p": float(p),
        "p_applied": True,
        "largest_cluster_only": bool(largest_cluster_only),
        "original_usable_sites": original_usable,
        "usable_sites": int(active_sites.size),
        "removed_cluster_sites": original_usable - int(active_sites.size),
        "blocked_sites": int(blocked.sum()),
        "effective_p": int(blocked.sum()) / total_sites,
    }
    if active_sites.size > 0 and D == 0:
        raise ValueError("D must be positive when usable sites remain")
    if D > active_sites.size:
        raise ValueError(
            f"D={D} exceeds the {active_sites.size} usable sites generated "
            f"for p={p}"
        )

    lattice = np.full(total_sites, -1, dtype=np.int32)
    if active_sites.size:
        species_ids = np.arange(1, D + 1, dtype=np.int32)
        active_values = np.empty(active_sites.size, dtype=np.int32)
        active_values[:D] = species_ids
        if active_sites.size > D:
            active_values[D:] = rng.choice(
                species_ids, size=active_sites.size - D
            )
        rng.shuffle(active_values)
        lattice[active_sites] = active_values
    lattice = lattice.reshape(L_row, L_col)

    result = _simulate_lattice(
        lattice,
        list(range(1, D + 1)),
        D,
        gamma,
        alpha,
        T,
        track_every,
        rng,
        progress,
        populate_first_100,
        checkpoint_dir=checkpoint_dir,
        retention=retention,
        geometry_metadata=geometry_metadata,
        max_forbidden_cycle_length=max_forbidden_cycle_length,
        max_cycle_rejection_attempts=max_cycle_rejection_attempts,
    )
    return result


@njit(cache=True)
def _patch_root(parent, site):
    """Return a union-find root while shortening the traversed path."""
    while parent[site] != site:
        parent[site] = parent[parent[site]]
        site = parent[site]
    return site


@njit(cache=True)
def _merge_patch_sites(parent, sizes, first, second):
    """Merge two union-find components and report whether they differed."""
    first_root = _patch_root(parent, first)
    second_root = _patch_root(parent, second)
    if first_root == second_root:
        return False
    if sizes[first_root] < sizes[second_root]:
        first_root, second_root = second_root, first_root
    parent[second_root] = first_root
    sizes[first_root] += sizes[second_root]
    return True


@njit(cache=True)
def _count_lattice_patches(lattice):
    """Count same-species components with four-neighbour periodic edges."""
    rows, columns = lattice.shape
    total_sites = lattice.size
    parent = np.arange(total_sites, dtype=np.int64)
    sizes = np.ones(total_sites, dtype=np.int64)
    patches = 0

    for row in range(rows):
        for column in range(columns):
            species = lattice[row, column]
            if species <= 0:
                continue

            patches += 1
            site = row * columns + column
            right_column = 0 if column + 1 == columns else column + 1
            if lattice[row, right_column] == species:
                right_site = row * columns + right_column
                if _merge_patch_sites(parent, sizes, site, right_site):
                    patches -= 1

            down_row = 0 if row + 1 == rows else row + 1
            if lattice[down_row, column] == species:
                down_site = down_row * columns + column
                if _merge_patch_sites(parent, sizes, site, down_site):
                    patches -= 1

    return patches


def count_patches(lattice):
    """Return the number of spatial species patches in a lattice.

    A patch is a maximal four-neighbour connected region occupied by one
    positive species ID. Connections wrap across the periodic lattice edges;
    empty (0) and blocked (-1) sites are not counted.
    """
    lattice = np.asarray(lattice)
    if lattice.ndim != 2:
        raise ValueError("lattice must be two-dimensional")
    if not lattice.size:
        raise ValueError("lattice cannot be empty")
    if not np.issubdtype(lattice.dtype, np.integer):
        raise TypeError("lattice must contain integers")
    return int(_count_lattice_patches(lattice))


def _patchiness_checkpoint_files(results, checkpoint_dir=None):
    """Find the snapshots available for a patchiness time series."""
    if checkpoint_dir is not None:
        source = Path(checkpoint_dir).expanduser()
        if source.is_dir():
            candidates = list(source.glob("checkpoint_*.npz"))
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"checkpoint path not found: {source}")
    else:
        checkpoint_files = results.get("checkpoint_files", ())
        if checkpoint_files:
            directories = {
                Path(path).expanduser().parent for path in checkpoint_files
            }
            candidates = [
                path
                for directory in directories
                for path in directory.glob("checkpoint_*.npz")
            ]
        elif results.get("checkpoint_path"):
            checkpoint_path = Path(results["checkpoint_path"]).expanduser()
            candidates = list(
                checkpoint_path.parent.glob("checkpoint_*.npz")
            )
        else:
            raise ValueError(
                "show_patchiness=True requires checkpoints; pass "
                "checkpoint_dir or use a result/state that contains "
                "checkpoint metadata"
            )

    # normcase(abspath(...)) deduplicates with string work alone; resolve()
    # costs one filesystem call per path, which is seconds for a long run.
    unique_candidates = {
        os.path.normcase(os.path.abspath(path)): path for path in candidates
    }
    checkpoint_files = sorted(
        unique_candidates.values(), key=_checkpoint_timestep
    )
    if not checkpoint_files:
        raise FileNotFoundError("no checkpoint_*.npz files found")
    return checkpoint_files


# Checkpoints are written once and never rewritten, so a patch count keyed on
# a file's identity, size and modification time stays valid for the session.
# Re-plotting a long run would otherwise re-read every lattice from disk.
_PATCH_COUNT_CACHE = {}


def _snapshot_patch_count(checkpoint_file, expected_shape, result_timestep):
    """Count one snapshot's patches, reusing a cached count when possible.

    Returns ``(timestep, patches)``, or ``None`` when the snapshot is later
    than the result time and therefore not part of the history.
    """
    status = os.stat(checkpoint_file)
    cache_key = (
        os.path.normcase(os.path.abspath(checkpoint_file)),
        status.st_size,
        status.st_mtime_ns,
    )
    cached = _PATCH_COUNT_CACHE.get(cache_key)
    if cached is not None:
        timestep, patches, cached_shape = cached
        if cached_shape != expected_shape:
            raise ValueError(
                f"checkpoint {checkpoint_file} has lattice shape "
                f"{cached_shape}; expected {expected_shape}"
            )
        return None if timestep > result_timestep else (timestep, patches)

    with np.load(checkpoint_file, allow_pickle=False) as saved:
        missing = {"lattice", "timestep"}.difference(saved.files)
        if missing:
            raise ValueError(
                f"checkpoint {checkpoint_file} is missing: "
                f"{', '.join(sorted(missing))}"
            )
        timestep = int(saved["timestep"])
        if timestep > result_timestep:
            return None
        checkpoint_lattice = saved["lattice"]
        if checkpoint_lattice.shape != expected_shape:
            raise ValueError(
                f"checkpoint {checkpoint_file} has lattice shape "
                f"{checkpoint_lattice.shape}; expected {expected_shape}"
            )
        patches = count_patches(checkpoint_lattice)

    _PATCH_COUNT_CACHE[cache_key] = (
        timestep, patches, checkpoint_lattice.shape
    )
    return timestep, patches


def _result_timestep(results):
    """Return the simulation time a result or state was taken at."""
    result_timestep = results.get("timestep")
    if result_timestep is None:
        result_timestep = np.asarray(results["tracked_timesteps"])[-1]
    return int(result_timestep)


def _stored_patch_series(timesteps, patches, result_timestep):
    """Keep the counted samples of a stored series up to the result time."""
    timesteps = np.asarray(timesteps, dtype=np.int64)
    patches = np.asarray(patches, dtype=np.int64)
    if timesteps.size != patches.size or not timesteps.size:
        return None
    usable = (patches >= 0) & (timesteps <= result_timestep)
    if not np.any(usable):
        return None
    return timesteps[usable].copy(), patches[usable].copy()


def _recorded_patchiness_history(results, checkpoint_dir=None):
    """Read the patch series a run stored, or None when it stored none.

    Reading the record is what makes a pruned run plottable: the lattices
    that produced the counts are gone, but the counts themselves were saved
    while those lattices were still in memory.
    """
    result_timestep = _result_timestep(results)

    # An explicitly named directory wins over whatever the result remembers.
    if checkpoint_dir is not None:
        analysis = load_analysis(checkpoint_dir)
        if analysis is not None:
            return _stored_patch_series(
                analysis["tracked_timesteps"],
                analysis["patch_history"],
                result_timestep,
            )
        return None

    if results.get("patch_history") is not None:
        series = _stored_patch_series(
            results["tracked_timesteps"],
            results["patch_history"],
            result_timestep,
        )
        if series is not None:
            return series

    for key in ("analysis_path", "checkpoint_path"):
        location = results.get(key)
        if location:
            analysis = load_analysis(location)
            if analysis is not None:
                return _stored_patch_series(
                    analysis["tracked_timesteps"],
                    analysis["patch_history"],
                    result_timestep,
                )
    for path in results.get("checkpoint_files", ()) or ():
        analysis = load_analysis(path)
        if analysis is not None:
            return _stored_patch_series(
                analysis["tracked_timesteps"],
                analysis["patch_history"],
                result_timestep,
            )
    return None


def _load_patchiness_history(results, checkpoint_dir=None):
    """Return the patch series, preferring the one the run recorded.

    Runs written before analysis records existed, and legs that never
    checkpointed, fall back to counting each snapshot still on disk.
    """
    recorded = _recorded_patchiness_history(
        results, checkpoint_dir=checkpoint_dir
    )
    if recorded is not None:
        return recorded

    checkpoint_files = _patchiness_checkpoint_files(
        results, checkpoint_dir=checkpoint_dir
    )
    expected_shape = np.asarray(results["lattice"]).shape
    result_timestep = _result_timestep(results)

    # Checkpoint filenames carry their own simulation time, so snapshots past
    # the result time can be dropped without reading their lattices. A name
    # that does not parse is still opened and judged on its stored timestep.
    eligible_files = [
        checkpoint_file
        for checkpoint_file in checkpoint_files
        if _checkpoint_timestep(checkpoint_file) <= result_timestep
    ]

    def count_one(checkpoint_file):
        return _snapshot_patch_count(
            checkpoint_file, expected_shape, result_timestep
        )

    # Reading a snapshot is dominated by file I/O, which releases the GIL, so
    # a small thread pool cuts the wall time of a long history several-fold.
    worker_count = min(8, len(eligible_files))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            counted = list(pool.map(count_one, eligible_files))
    else:
        counted = [count_one(path) for path in eligible_files]

    patches_by_timestep = {
        timestep: patches for timestep, patches in filter(None, counted)
    }

    if not patches_by_timestep:
        raise ValueError(
            "no checkpoint snapshots are available through the result time"
        )
    timesteps = np.asarray(sorted(patches_by_timestep), dtype=np.int64)
    patchiness = np.asarray(
        [patches_by_timestep[timestep] for timestep in timesteps],
        dtype=np.int64,
    )
    return timesteps, patchiness


def _load_lattice_snapshot(results, lattice_timestep, checkpoint_dir=None):
    """Load the available lattice nearest to a requested timestep."""
    if isinstance(lattice_timestep, (bool, np.bool_)) or not isinstance(
        lattice_timestep, (int, float, np.integer, np.floating)
    ):
        raise TypeError("lattice_timestep must be a real number")

    requested_timestep = float(lattice_timestep)
    if not np.isfinite(requested_timestep):
        raise ValueError("lattice_timestep must be finite")
    if requested_timestep < 0:
        raise ValueError("lattice_timestep cannot be negative")

    result_lattice = np.asarray(results["lattice"])
    result_timestep = results.get("timestep")
    if result_timestep is None:
        result_timestep = np.asarray(results["tracked_timesteps"])[-1]
    result_timestep = int(result_timestep)

    # The in-memory result is the nearest eligible state at or beyond the
    # completed result time, so no checkpoint metadata is needed there.
    if requested_timestep >= result_timestep:
        return result_lattice, result_timestep

    try:
        checkpoint_files = _patchiness_checkpoint_files(
            results, checkpoint_dir=checkpoint_dir
        )
    except ValueError as error:
        raise ValueError(
            "selecting a historical lattice requires checkpoints; pass "
            "checkpoint_dir or use a result/state with checkpoint metadata"
        ) from error

    # Checkpoint filenames are written from their integer simulation time.
    # Inspecting these names avoids loading every (potentially large) lattice.
    files_by_timestep = {
        timestep: checkpoint_file
        for checkpoint_file in checkpoint_files
        if 0 <= (timestep := _checkpoint_timestep(checkpoint_file))
        <= result_timestep
        and timestep != result_timestep
    }
    available_timesteps = [result_timestep, *files_by_timestep]
    selected_timestep = min(
        available_timesteps,
        key=lambda timestep: (
            abs(timestep - requested_timestep),
            timestep,
        ),
    )

    # The in-memory result is also an available lattice and is preferred at
    # its own timestep, even if a duplicate final checkpoint exists.
    if selected_timestep == result_timestep:
        return result_lattice, result_timestep

    checkpoint_file = files_by_timestep[selected_timestep]
    with np.load(checkpoint_file, allow_pickle=False) as saved:
        missing = {"lattice", "timestep"}.difference(saved.files)
        if missing:
            raise ValueError(
                f"checkpoint {checkpoint_file} is missing: "
                f"{', '.join(sorted(missing))}"
            )
        saved_timestep = int(saved["timestep"])
        if saved_timestep != selected_timestep:
            raise ValueError(
                f"checkpoint {checkpoint_file} contains timestep "
                f"{saved_timestep}; expected {selected_timestep}"
            )
        lattice = saved["lattice"].copy()

    if lattice.shape != result_lattice.shape:
        raise ValueError(
            f"checkpoint {checkpoint_file} has lattice shape "
            f"{lattice.shape}; expected {result_lattice.shape}"
        )
    return lattice, selected_timestep


def _validate_smoothing_sigma(value):
    """Return one optional positive, finite Gaussian smoothing width."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("smooth_sigma must be a real number or None")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("smooth_sigma must be finite")
    if value <= 0:
        raise ValueError("smooth_sigma must be positive")
    return value


def _gaussian_smooth(values, sigma, truncate=4.0):
    """Smooth a time series with a Gaussian kernel of width ``sigma``.

    ``sigma`` is measured in samples of the series, matching the convention
    of ``scipy.ndimage.gaussian_filter1d``. The kernel is cut off at
    ``truncate`` standard deviations and the ends are reflected, so the first
    and last points are not dragged towards zero.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return values.copy()
    radius = max(int(truncate * sigma + 0.5), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _series_from(timesteps, values, start_timestep, name):
    """Keep the samples of a time series at or after ``start_timestep``."""
    timesteps = np.asarray(timesteps)
    values = np.asarray(values)
    if start_timestep is None:
        return timesteps, values
    shown = timesteps >= start_timestep
    if not np.any(shown):
        raise ValueError(
            f"no {name} samples at or after start_timestep "
            f"{start_timestep:,.0f}"
        )
    return timesteps[shown], values[shown]


def show_results(
    results,
    show_patchiness=False,
    checkpoint_dir=None,
    lattice_timestep=None,
    smooth_sigma=None,
    start_timestep=None,
):
    """Plot a lattice snapshot and the full diversity/patchiness history.

    When ``show_patchiness`` is true, patch counts are calculated from the
    available checkpoint lattices and drawn against a separate right-hand
    y-axis. ``checkpoint_dir`` can explicitly name a checkpoint directory or
    file; otherwise checkpoint metadata in ``results`` is used. When
    ``lattice_timestep`` is supplied, the lattice nearest to that simulation
    time is loaded from the available checkpoints.

    ``start_timestep`` drops the time-series samples before that simulation
    time, for example the history a branch inherited from the run it was
    started from. The curves still end at the result time, and the lattice
    shown is unaffected.

    ``smooth_sigma`` draws a Gaussian-smoothed living-species curve, and a
    smoothed patchiness curve when one is shown, over a faded copy of the
    raw series. The width is given in tracked samples rather than timesteps,
    so a run tracked every ``track_every`` steps is smoothed over roughly
    ``smooth_sigma * track_every`` simulation time. Smoothing is applied
    after ``start_timestep`` is, so earlier samples do not bleed into it.
    """
    if not isinstance(show_patchiness, (bool, np.bool_)):
        raise TypeError("show_patchiness must be True or False")
    smooth_sigma = _validate_smoothing_sigma(smooth_sigma)
    start_timestep = _validate_animation_timestep(
        start_timestep, "start_timestep"
    )
    diversity_timesteps, diversity_history = _series_from(
        results["tracked_timesteps"],
        results["diversity_history"],
        start_timestep,
        "living-species",
    )

    lattice = results["lattice"]
    selected_timestep = None
    if lattice_timestep is not None:
        lattice, selected_timestep = _load_lattice_snapshot(
            results,
            lattice_timestep,
            checkpoint_dir=checkpoint_dir,
        )

    species_ids = np.unique(lattice[lattice > 0])
    number_of_species = species_ids.size

    patchiness_history = None
    if show_patchiness:
        patchiness_history = _series_from(
            *_load_patchiness_history(results, checkpoint_dir=checkpoint_dir),
            start_timestep,
            "patchiness",
        )

    # Compact historical IDs so every currently living species gets a color.
    display_lattice = np.ones(lattice.shape, dtype=np.int32)  # Empty = 1
    display_lattice[lattice == -1] = 0                       # Blocked = 0
    occupied = lattice > 0
    display_lattice[occupied] = (
        np.searchsorted(species_ids, lattice[occupied]) + 2
    )

    hues = (np.arange(max(number_of_species, 1)) * 0.61803398875) % 1
    species_colors = plt.colormaps["hsv"](hues)
    colors = np.vstack(([0, 0, 0, 1], [0.92, 0.92, 0.92, 1], species_colors))
    colors = colors[:number_of_species + 2]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(number_of_species + 3) - 0.5, cmap.N)

    fig, (ax_lattice, ax_diversity) = plt.subplots(1, 2, figsize=(13, 5))
    ax_lattice.imshow(
        display_lattice, cmap=cmap, norm=norm, interpolation="nearest"
    )
    if selected_timestep is None:
        lattice_title = "Final lattice"
    else:
        lattice_title = f"Lattice at timestep {selected_timestep:,}"
    ax_lattice.set_title(
        f"{lattice_title}: {number_of_species:,} living species"
    )
    ax_lattice.set_axis_off()

    if smooth_sigma is None:
        diversity_line, = ax_diversity.plot(
            diversity_timesteps,
            diversity_history,
            color="tab:blue",
            label="Living species",
            lw=1.5,
        )
    else:
        # The raw series stays visible underneath the smoothed curve.
        ax_diversity.plot(
            diversity_timesteps,
            diversity_history,
            color="tab:blue",
            lw=1.0,
            alpha=0.25,
        )
        diversity_line, = ax_diversity.plot(
            diversity_timesteps,
            _gaussian_smooth(diversity_history, smooth_sigma),
            color="tab:blue",
            label="Living species",
            lw=1.8,
        )
    ax_diversity.set(xlabel="Timestep", ylabel="Living species", title="Diversity")
    ax_diversity.grid(alpha=0.25)

    if patchiness_history is not None:
        patch_timesteps, patch_counts = patchiness_history
        ax_patchiness = ax_diversity.twinx()
        if smooth_sigma is None:
            patchiness_line, = ax_patchiness.plot(
                patch_timesteps,
                patch_counts,
                color="tab:orange",
                label="Species patches",
                lw=1.5,
            )
        else:
            ax_patchiness.plot(
                patch_timesteps,
                patch_counts,
                color="tab:orange",
                lw=1.0,
                alpha=0.25,
            )
            patchiness_line, = ax_patchiness.plot(
                patch_timesteps,
                _gaussian_smooth(patch_counts, smooth_sigma),
                color="tab:orange",
                label="Species patches",
                lw=1.8,
            )
        ax_patchiness.set_ylabel("Species patches", color="tab:orange")
        ax_patchiness.tick_params(axis="y", labelcolor="tab:orange")
        ax_patchiness.yaxis.set_major_formatter(
            StrMethodFormatter("{x:,.0f}")
        )
        ax_diversity.set_title("Diversity and patchiness")
        ax_diversity.legend(
            handles=[diversity_line, patchiness_line], loc="best"
        )

    if smooth_sigma is not None:
        ax_diversity.set_title(
            f"{ax_diversity.get_title()} "
            f"(Gaussian sigma = {smooth_sigma:g} samples)"
        )

    plt.tight_layout()
    plt.show()

    initial_newest_species = _initial_newest_species(results)
    if selected_timestep is None:
        print(f"Living species:        {results['diversity']:,}")
        print(f"Largest species ID:    {results['newest_species']:,}")
    else:
        print(f"Displayed timestep:    {selected_timestep:,}")
        print(f"Displayed species:     {number_of_species:,}")
        print(f"Final living species:  {results['diversity']:,}")
        print(f"Final largest ID:      {results['newest_species']:,}")
    print(
        "Species introduced:    "
        f"{results['newest_species'] - initial_newest_species:,}"
    )


def _validate_animation_timestep(value, name):
    """Return one optional finite, non-negative animation boundary."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number or None")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _animation_frame_sources(
    source,
    checkpoint_dir=None,
    start_timestep=None,
    end_timestep=None,
    frame_stride=1,
    max_frames=150,
):
    """Return a result and its selected, ordered animation frame sources."""
    if isinstance(source, (str, os.PathLike)):
        results = load_checkpoint(source)
    elif isinstance(source, Mapping):
        results = source
    else:
        raise TypeError(
            "source must be a result/state mapping or checkpoint path"
        )

    if isinstance(frame_stride, (bool, np.bool_)) or not isinstance(
        frame_stride, (int, np.integer)
    ):
        raise TypeError("frame_stride must be an integer")
    frame_stride = int(frame_stride)
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if max_frames is not None:
        if isinstance(max_frames, (bool, np.bool_)) or not isinstance(
            max_frames, (int, np.integer)
        ):
            raise TypeError("max_frames must be an integer or None")
        max_frames = int(max_frames)
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2 or None")

    start_timestep = _validate_animation_timestep(
        start_timestep, "start_timestep"
    )
    end_timestep = _validate_animation_timestep(
        end_timestep, "end_timestep"
    )
    if (
        start_timestep is not None
        and end_timestep is not None
        and start_timestep > end_timestep
    ):
        raise ValueError("start_timestep cannot exceed end_timestep")

    result_lattice = np.asarray(results["lattice"])
    if result_lattice.ndim != 2:
        raise ValueError("result lattice must be two-dimensional")
    result_timestep = results.get("timestep")
    if result_timestep is None:
        result_timestep = np.asarray(results["tracked_timesteps"])[-1]
    result_timestep = int(result_timestep)

    try:
        checkpoint_files = _patchiness_checkpoint_files(
            results, checkpoint_dir=checkpoint_dir
        )
    except ValueError as error:
        raise ValueError(
            "animating lattice evolution requires checkpoints; pass "
            "checkpoint_dir, a checkpoint path, or a result/state with "
            "checkpoint metadata"
        ) from error

    files_by_timestep = {}
    for checkpoint_file in checkpoint_files:
        timestep = _checkpoint_timestep(checkpoint_file)
        if 0 <= timestep < result_timestep:
            files_by_timestep[timestep] = checkpoint_file

    # Prefer the supplied in-memory endpoint over a duplicate final file.
    frames = sorted(files_by_timestep.items())
    frames.append((result_timestep, None))
    available_timesteps = [timestep for timestep, _ in frames]
    if start_timestep is None:
        start_index = 0
    else:
        start_index = next(
            (
                index
                for index, timestep in enumerate(available_timesteps)
                if timestep >= start_timestep
            ),
            len(frames),
        )
    if end_timestep is None:
        end_index = len(frames) - 1
    else:
        end_index = next(
            (
                index
                for index in range(len(frames) - 1, -1, -1)
                if available_timesteps[index] <= end_timestep
            ),
            -1,
        )
    selected_frames = frames[start_index:end_index + 1]
    if not selected_frames:
        raise ValueError("the requested timestep range contains no snapshots")

    endpoint = selected_frames[-1]
    selected_frames = selected_frames[::frame_stride]
    if selected_frames[-1][0] != endpoint[0]:
        selected_frames.append(endpoint)
    if max_frames is not None and len(selected_frames) > max_frames:
        selected_indices = np.linspace(
            0,
            len(selected_frames) - 1,
            num=max_frames,
            dtype=np.int64,
        )
        selected_frames = [
            selected_frames[int(index)] for index in selected_indices
        ]
    if len(selected_frames) < 2:
        raise ValueError(
            "an animation requires at least two checkpoint states in the "
            "selected timestep range"
        )
    return results, selected_frames


def _load_animation_lattice(results, frame, expected_shape):
    """Load and validate one lazy animation frame."""
    expected_timestep, checkpoint_file = frame
    if checkpoint_file is None:
        lattice = np.asarray(results["lattice"])
        saved_timestep = expected_timestep
    else:
        with np.load(checkpoint_file, allow_pickle=False) as saved:
            missing = {"lattice", "timestep"}.difference(saved.files)
            if missing:
                raise ValueError(
                    f"checkpoint {checkpoint_file} is missing: "
                    f"{', '.join(sorted(missing))}"
                )
            saved_timestep = int(saved["timestep"])
            lattice = saved["lattice"].copy()

    if saved_timestep != expected_timestep:
        raise ValueError(
            f"checkpoint {checkpoint_file} contains timestep "
            f"{saved_timestep}; expected {expected_timestep}"
        )
    if lattice.shape != expected_shape:
        raise ValueError(
            f"checkpoint {checkpoint_file} has lattice shape "
            f"{lattice.shape}; expected {expected_shape}"
        )
    return lattice


def _lattice_rgba(lattice):
    """Map a lattice to stable colors based on permanent species IDs."""
    lattice = np.asarray(lattice)
    rgba = np.empty(lattice.shape + (4,), dtype=np.float64)
    rgba[...] = (0.92, 0.92, 0.92, 1.0)  # Empty sites.
    rgba[lattice < 0] = (0.0, 0.0, 0.0, 1.0)  # Blocked sites.

    occupied = lattice > 0
    if np.any(occupied):
        species_ids = lattice[occupied].astype(np.float64, copy=False)
        hues = np.remainder(species_ids * 0.61803398875, 1.0)
        rgba[occupied] = plt.colormaps["hsv"](hues)
    return rgba


def animate_lattice(
    source,
    checkpoint_dir=None,
    *,
    start_timestep=None,
    end_timestep=None,
    frame_stride=1,
    max_frames=150,
    interval=150,
    repeat=True,
    save_path=None,
    display=True,
    dpi=100,
):
    """Animate lattice evolution from a result or checkpoint sequence.

    ``source`` may be a simulation result/state, a checkpoint directory, or
    one checkpoint file. A specific checkpoint animates its sibling snapshots
    only through that checkpoint's timestep. Start/end bounds are inclusive.
    ``frame_stride`` keeps every nth available snapshot while always retaining
    the selected endpoint. ``max_frames`` automatically downsamples long runs
    to help keep notebook playback compact; set it to ``None`` to use every
    selected frame.

    In a notebook, ``display=True`` renders JavaScript playback controls.
    Supplying ``save_path`` additionally writes a GIF (Pillow) or MP4 (FFmpeg).
    The returned Matplotlib ``FuncAnimation`` can be reused or saved again.
    """
    if isinstance(interval, (bool, np.bool_)) or not isinstance(
        interval, (int, float, np.integer, np.floating)
    ):
        raise TypeError("interval must be a real number")
    interval = float(interval)
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be finite and positive")
    if not isinstance(repeat, (bool, np.bool_)):
        raise TypeError("repeat must be True or False")
    if not isinstance(display, (bool, np.bool_)):
        raise TypeError("display must be True or False")
    if isinstance(dpi, (bool, np.bool_)) or not isinstance(
        dpi, (int, float, np.integer, np.floating)
    ):
        raise TypeError("dpi must be a real number")
    dpi = float(dpi)
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be finite and positive")

    results, frames = _animation_frame_sources(
        source,
        checkpoint_dir=checkpoint_dir,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    expected_shape = np.asarray(results["lattice"]).shape
    first_lattice = _load_animation_lattice(
        results, frames[0], expected_shape
    )

    from matplotlib.animation import FuncAnimation

    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(
        _lattice_rgba(first_lattice), interpolation="nearest"
    )
    title = axis.set_title("")
    axis.set_axis_off()

    first_frame_cache = {0: first_lattice}

    def draw_frame(frame_index):
        lattice = first_frame_cache.get(frame_index)
        if lattice is None:
            lattice = _load_animation_lattice(
                results, frames[frame_index], expected_shape
            )
        timestep = frames[frame_index][0]
        number_of_species = np.unique(lattice[lattice > 0]).size
        image.set_data(_lattice_rgba(lattice))
        title.set_text(
            f"Lattice at timestep {timestep:,}: "
            f"{number_of_species:,} living species"
        )
        return image, title

    def initialize():
        return draw_frame(0)

    animation = FuncAnimation(
        figure,
        draw_frame,
        frames=range(len(frames)),
        init_func=initialize,
        interval=interval,
        repeat=bool(repeat),
        blit=True,
        cache_frame_data=False,
    )
    plt.tight_layout()

    if save_path is not None:
        if not isinstance(save_path, (str, os.PathLike)):
            raise TypeError("save_path must be a path or None")
        destination = Path(save_path).expanduser()
        suffix = destination.suffix.lower()
        if suffix not in {".gif", ".mp4"}:
            raise ValueError("save_path must end in .gif or .mp4")
        destination.parent.mkdir(parents=True, exist_ok=True)
        frames_per_second = 1000.0 / interval

        if suffix == ".gif":
            from matplotlib.animation import PillowWriter

            writer = PillowWriter(fps=frames_per_second)
        else:
            from matplotlib.animation import FFMpegWriter

            if not FFMpegWriter.isAvailable():
                raise RuntimeError(
                    "saving MP4 animations requires FFmpeg; save as GIF "
                    "or install FFmpeg"
                )
            writer = FFMpegWriter(fps=frames_per_second)
        animation.save(str(destination), writer=writer, dpi=dpi)

    if display:
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display as notebook_display
        except ImportError:
            plt.show()
        else:
            if get_ipython() is None:
                plt.show()
            else:
                playback_mode = "loop" if repeat else "once"
                notebook_display(
                    HTML(animation.to_jshtml(default_mode=playback_mode))
                )
                plt.close(figure)
    else:
        plt.close(figure)

    return animation
