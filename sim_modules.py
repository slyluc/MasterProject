from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from matplotlib.colors import BoundaryNorm, ListedColormap
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
    checkpoint_dir=None,
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
    checkpoint_files = []

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
                    exported.update(
                        {
                            "timestep": int(initial_timestep)
                            + previous_timestep,
                            "target_timestep": target_timestep,
                            "tracked_timesteps": np.concatenate(
                                [prior_tracked_timesteps, *timestep_chunks]
                            ),
                            "diversity_history": np.concatenate(
                                [prior_diversity_history, *diversity_chunks]
                            ),
                            "rng_state": rng.bit_generator.state,
                            "gamma": float(gamma),
                            "alpha": float(alpha),
                            "track_every": int(track_every),
                            "populate_first_100": bool(populate_first_100),
                        }
                    )
                    checkpoint_files.append(
                        _write_checkpoint(checkpoint_dir, exported)
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
            "rng_state": rng.bit_generator.state,
            "checkpoint_files": checkpoint_files,
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
        checkpoint_dir=checkpoint_dir,
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
    )
    result["p"] = float(p)
    result["p_applied"] = True
    result["blocked_sites"] = int(blocked.sum())
    return result


def show_results(results):
    lattice = results["lattice"]
    species_ids = np.unique(lattice[lattice > 0])
    number_of_species = species_ids.size

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
    ax_lattice.set_title(f"Final lattice: {number_of_species:,} living species")
    ax_lattice.set_axis_off()

    ax_diversity.plot(
        results["tracked_timesteps"], results["diversity_history"], lw=1.5
    )
    ax_diversity.set(xlabel="Timestep", ylabel="Living species", title="Diversity")
    ax_diversity.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    initial_species = int(results["diversity_history"][0])
    print(f"Living species:        {results['diversity']:,}")
    print(f"Largest species ID:    {results['newest_species']:,}")
    print(f"Species introduced:    {results['newest_species'] - initial_species:,}")
