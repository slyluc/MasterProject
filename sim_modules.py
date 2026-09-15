from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import StrMethodFormatter
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
    ``run_main(initial_state=state)``.
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

    def run_main(self, initial_state=None):
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
        )

    def run_percolation(self, initial_state=None):
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
        )


_CHECKPOINT_VERSION = 1


def _checkpoint_timestep(path):
    """Return the integer suffix from ``checkpoint_<timestep>.npz``."""
    try:
        return int(path.stem.removeprefix("checkpoint_"))
    except ValueError:
        return -1


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

    # The patch series lives in the run's analysis record rather than in the
    # snapshot, because most snapshots are deleted once they are counted. A
    # snapshot from the middle of a run only carries the history up to its own
    # time, so the record is trimmed to match before it is attached.
    tracked_count = state["tracked_timesteps"].size
    patch_history = np.full(tracked_count, _MISSING_PATCH_COUNT, dtype=np.int64)
    analysis = load_analysis(checkpoint_path)
    if analysis is not None:
        recorded_times = np.asarray(analysis["tracked_timesteps"])
        if recorded_times.size >= tracked_count and np.array_equal(
            recorded_times[:tracked_count], state["tracked_timesteps"]
        ):
            patch_history = np.asarray(
                analysis["patch_history"], dtype=np.int64
            )[:tracked_count].copy()
            state["analysis_path"] = analysis["analysis_path"]
            state["event_timesteps"] = analysis["event_timesteps"].copy()
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
        "rng_state": dict(rng_state) if rng_state is not None else None,
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
                rng_state=np.asarray(json.dumps(state["rng_state"])),
                gamma=np.float64(state["gamma"]),
                alpha=np.float64(state["alpha"]),
                track_every=np.int64(state["track_every"]),
                populate_first_100=np.bool_(state["populate_first_100"]),
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
                timestep=np.int64(record["timestep"]),
                target_timestep=np.int64(record["target_timestep"]),
                gamma=np.float64(record["gamma"]),
                alpha=np.float64(record["alpha"]),
                track_every=np.int64(record["track_every"]),
                populate_first_100=np.bool_(record["populate_first_100"]),
                lattice_shape=np.asarray(
                    record["lattice_shape"], dtype=np.int64
                ),
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
    record["analysis_path"] = str(analysis_path.resolve())
    return record


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

def update_Gamma(
    Gamma,
    gamma,
    current_species,
    newest_species,
    invaded_species=None,
    rng=None,
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
    """
    if (
        isinstance(gamma, (bool, np.bool_))
        or not isinstance(gamma, (int, float, np.integer, np.floating))
    ):
        raise TypeError("gamma must be a real number")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")

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
        Gamma = (random_source.random(expected_shape) < gamma).astype(np.uint8)
        np.fill_diagonal(Gamma, 0)
        return Gamma
    else:
        Gamma = np.asarray(Gamma)
        if Gamma.shape != expected_shape:
            raise ValueError(
                f"Gamma must have shape {expected_shape}, got {Gamma.shape}"
            )
        if not np.all((Gamma == 0) | (Gamma == 1)):
            raise ValueError("Gamma may contain only 0 and 1")

    # uint8 uses one byte per edge and is sufficient for this binary matrix.
    new_size = active_count + 1
    new_Gamma = np.empty((new_size, new_size), dtype=np.uint8)
    new_Gamma[:-1, :-1] = Gamma

    # Draw both directed links with every already-existing species. The paper
    # does not define a self-interaction; it is dynamically irrelevant and 0.
    new_edges = random_source.random(2 * active_count) < gamma
    new_Gamma[-1, :-1] = new_edges[:active_count]
    new_Gamma[:-1, -1] = new_edges[active_count:]
    new_Gamma[-1, -1] = 0

    if invaded_species not in (None, 0):
        invaded_index = live_species.index(invaded_species)
        new_Gamma[-1, invaded_index] = 1

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
):
    """Execute sequential stochastic updates in compiled machine code."""
    total_sites = slot_lattice.size
    usable_count = usable_sites.size
    records = timesteps // track_every
    if timesteps % track_every:
        records += 1
    diversity_history = np.empty(records, dtype=np.int64)
    record_index = 0

    for timestep in range(1, timesteps + 1):
        # Each code uniformly selects one of the N sites and one of its four
        # neighbors. This is distributionally identical to two separate draws.
        event_codes = rng.integers(0, 4 * total_sites, size=total_sites)
        introduction_draws = rng.random(total_sites)

        for event in range(total_sites):
            source_site = event_codes[event] >> 2
            direction = event_codes[event] & 3
            source_slot = slot_lattice[source_site]

            if source_slot >= 0:
                target_site = neighbours[source_site, direction]
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

                # Draw the new directed relationships independently. Existing
                # relationships are never regenerated or altered.
                for position in range(live_count):
                    other_slot = active_slots[position]
                    Gamma[new_slot, other_slot] = rng.random() < gamma
                Gamma[new_slot, new_slot] = 0
                for position in range(live_count):
                    other_slot = active_slots[position]
                    Gamma[other_slot, new_slot] = rng.random() < gamma

                if replaced_slot >= 0:
                    Gamma[new_slot, replaced_slot] = 1

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
    compact_Gamma = Gamma[np.ix_(live_slots, live_slots)].copy()

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
    checkpoint_dir=None,
    retention=None,
    target_timestep=None,
):
    """Prepare state and run the exact sequential model in compiled code."""
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
        )
    else:
        initial_Gamma = np.asarray(initial_Gamma, dtype=np.uint8)

    rows, columns = lattice.shape
    total_sites = lattice.size
    external_flat = lattice.ravel()
    usable_sites = np.flatnonzero(external_flat != -1).astype(np.intp)
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
    for site in np.flatnonzero(external_flat > 0):
        slot_lattice[site] = species_to_slot[int(external_flat[site])]

    site_numbers = np.arange(total_sites, dtype=np.intp).reshape(rows, columns)
    neighbours = np.empty((total_sites, 4), dtype=np.intp)
    neighbours[:, 0] = np.roll(site_numbers, 1, axis=0).ravel()
    neighbours[:, 1] = np.roll(site_numbers, -1, axis=1).ravel()
    neighbours[:, 2] = np.roll(site_numbers, -1, axis=0).ravel()
    neighbours[:, 3] = np.roll(site_numbers, 1, axis=1).ravel()

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

    if set(species_to_slot) != set(int(value) for value in external_flat if value > 0):
        raise ValueError("current_species does not match the species in lattice")

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

    if retention is None:
        retention = RetentionPolicy()
    snapshot_retention = None
    if checkpoint_dir is not None:
        snapshot_retention = _CheckpointRetention(checkpoint_dir, retention)
        snapshot_retention.adopt_existing()

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

    diversity_chunks = []
    timestep_chunks = []
    patch_counts = []
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
            )
            state = result[:-1]
            diversity_chunks.append(result[-1])
            timestep_chunks.append(
                int(initial_timestep)
                + _tracking_timesteps(int(T), int(track_every))
            )
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
                )
                state = result[:-1]
                diversity_chunks.append(result[-1])
                timestep_chunks.append(
                    int(initial_timestep)
                    + previous_timestep
                    + _tracking_timesteps(chunk_size, int(track_every))
                )
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
                    tracked_so_far = np.concatenate(
                        [prior_tracked_timesteps, *timestep_chunks]
                    )
                    diversity_so_far = np.concatenate(
                        [prior_diversity_history, *diversity_chunks]
                    )
                    snapshot_timestep = (
                        int(initial_timestep) + previous_timestep
                    )
                    exported.update(
                        {
                            "timestep": snapshot_timestep,
                            "target_timestep": target_timestep,
                            "tracked_timesteps": tracked_so_far,
                            "diversity_history": diversity_so_far,
                            "rng_state": rng.bit_generator.state,
                            "gamma": float(gamma),
                            "alpha": float(alpha),
                            "track_every": int(track_every),
                            "populate_first_100": bool(populate_first_100),
                        }
                    )
                    written_file = _write_checkpoint(checkpoint_dir, exported)
                    checkpoint_files.append(written_file)

                    # Patches are counted here, while the lattice is in
                    # memory, because retention deletes most snapshots and
                    # the count could not be recovered afterwards.
                    patch_counts.append(count_patches(exported["lattice"]))
                    patch_so_far = _align_patch_history(
                        np.concatenate(
                            [
                                prior_patch_history,
                                np.asarray(patch_counts, dtype=np.int64),
                            ]
                        ),
                        tracked_so_far.size,
                    )
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
                            "timestep": snapshot_timestep,
                            "target_timestep": target_timestep,
                            "gamma": float(gamma),
                            "alpha": float(alpha),
                            "track_every": int(track_every),
                            "populate_first_100": bool(populate_first_100),
                            "lattice_shape": (rows, columns),
                        },
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

    tracked_timesteps = np.concatenate(
        [prior_tracked_timesteps, *timestep_chunks]
    )
    diversity_history = np.concatenate(
        [prior_diversity_history, *diversity_chunks]
    )
    # A leg that never wrote snapshots leaves its new samples uncounted, so
    # the series is padded rather than assumed complete.
    patch_history = _align_patch_history(
        np.concatenate(
            [prior_patch_history, np.asarray(patch_counts, dtype=np.int64)]
        ),
        tracked_timesteps.size,
    )

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
                "timestep": target_timestep,
                "target_timestep": target_timestep,
                "gamma": float(gamma),
                "alpha": float(alpha),
                "track_every": int(track_every),
                "populate_first_100": bool(populate_first_100),
                "lattice_shape": (rows, columns),
            },
        )
        checkpoint_files = [
            path for path in checkpoint_files if path not in removed_files
        ]

    final_result = _export_simulation_state(state, rows, columns)
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
):
    """Run ``T`` additional time units from a result or saved checkpoint.

    Existing interactions in ``Gamma`` are kept. ``gamma`` controls only the
    interactions drawn for species introduced during this new simulation leg.
    A saved RNG is restored automatically; ``seed`` is used for raw states
    that do not contain one. When raw arrays omit ``current_species``, Gamma
    rows are assumed to follow the ascending positive IDs in the lattice.
    """
    _validate_simulation_options(
        alpha, T, track_every, progress, populate_first_100
    )
    _validate_gamma(gamma)
    state = _normalize_initial_state(initial_state)
    rng = _make_rng(seed, state["rng_state"])
    if target_timestep is None:
        target_timestep = state["timestep"] + int(T)
    return _simulate_lattice(
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
        checkpoint_dir=checkpoint_dir,
        retention=retention,
        target_timestep=target_timestep,
    )


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
    ``initial_state`` to run ``T`` additional time units from that state.
    """
    _validate_simulation_options(
        alpha, T, track_every, progress, populate_first_100
    )
    _validate_gamma(gamma)
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
    )


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
):
    """Run the simulation with permanent random site removal.

    This is an extension of the paper's base model. Each site is independently
    blocked with probability ``p`` before initialization. Blocked sites remain
    -1 and cannot invade, be invaded, or receive introductions. A blocked
    source draw is simply a null event, so one time unit still contains ``N``
    microscopic updates and uses introduction probability ``alpha*gamma/N``.
    """
    _validate_simulation_options(
        alpha, T, track_every, progress, populate_first_100
    )
    _validate_gamma(gamma)
    if (
        isinstance(p, (bool, np.bool_))
        or not isinstance(p, (int, float, np.integer, np.floating))
    ):
        raise TypeError("p must be a real number")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    if initial_state is not None:
        # Site removal is an initialization rule. Existing -1 cells are kept;
        # changing p cannot reblock an already-running lattice.
        result = simulation_from_state(
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
        )
        result["p"] = float(p)
        result["p_applied"] = False
        result["blocked_sites"] = int(np.count_nonzero(result["lattice"] == -1))
        return result

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
    active_sites = np.flatnonzero(~blocked)
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
    )
    result["p"] = float(p)
    result["p_applied"] = True
    result["blocked_sites"] = int(blocked.sum())
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


def show_results(
    results,
    show_patchiness=False,
    checkpoint_dir=None,
    lattice_timestep=None,
    smooth_sigma=None,
):
    """Plot a lattice snapshot and the full diversity/patchiness history.

    When ``show_patchiness`` is true, patch counts are calculated from the
    available checkpoint lattices and drawn against a separate right-hand
    y-axis. ``checkpoint_dir`` can explicitly name a checkpoint directory or
    file; otherwise checkpoint metadata in ``results`` is used. When
    ``lattice_timestep`` is supplied, the lattice nearest to that simulation
    time is loaded from the available checkpoints. The time-series plots are
    still drawn through the full result time.

    ``smooth_sigma`` draws a Gaussian-smoothed living-species curve, and a
    smoothed patchiness curve when one is shown, over a faded copy of the
    raw series. The width is given in tracked samples rather than timesteps,
    so a run tracked every ``track_every`` steps is smoothed over roughly
    ``smooth_sigma * track_every`` simulation time.
    """
    if not isinstance(show_patchiness, (bool, np.bool_)):
        raise TypeError("show_patchiness must be True or False")
    smooth_sigma = _validate_smoothing_sigma(smooth_sigma)

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
        patchiness_history = _load_patchiness_history(
            results, checkpoint_dir=checkpoint_dir
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

    diversity_timesteps = results["tracked_timesteps"]
    diversity_history = results["diversity_history"]
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

    initial_species = int(results["diversity_history"][0])
    if selected_timestep is None:
        print(f"Living species:        {results['diversity']:,}")
        print(f"Largest species ID:    {results['newest_species']:,}")
    else:
        print(f"Displayed timestep:    {selected_timestep:,}")
        print(f"Displayed species:     {number_of_species:,}")
        print(f"Final living species:  {results['diversity']:,}")
        print(f"Final largest ID:      {results['newest_species']:,}")
    print(f"Species introduced:    {results['newest_species'] - initial_species:,}")


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
