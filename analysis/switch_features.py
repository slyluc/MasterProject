"""Per-state measures of a switch capture (tools/run_switch_capture.py), for switch_analysis.ipynb.

One pass reads every kept state and records community, interaction-network and lattice
measures, plus per-species values in a compressed-row layout: the species of state k are
``species_id[offsets[k]:offsets[k + 1]]``. The pass takes a few minutes, so its result is
cached next to the capture as features.npz and reused while the capture is unchanged.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

from numba import njit
import numpy as np

import sim_modules as MS
from sim_modules import _gaussian_smooth, _merge_patch_sites, _patch_root
from tools.run_switch_capture import regime_switches


CACHE = "features.npz"
CACHE_VERSION = 2
SCALARS = ("timestep", "species", "introduced", "largest_share", "effective_species", "links",
           "scc_largest", "scc_species", "top_species", "top_share", "boundaries",
           "active_boundaries", "patches", "single_patch_share", "largest_patch_share")
PER_SPECIES = ("species_id", "abundance", "front", "in_degree", "out_degree", "in_largest_scc",
               "patches_per_species")


def state_files(root):
    """The capture's starting state, then every segment's kept states, in time order.

    A capture thinned by tools/prune_checkpoints.py keeps only some states; the rest are
    replayed from them (tools/render_switch_videos.py, sim_modules.recreate_checkpoint).
    """
    root = Path(root)
    plan = json.loads((root / "capture_plan.json").read_text(encoding="utf-8"))
    files = [root / plan["initial_state"]]
    files += sorted(root.glob("segments/*/checkpoint_*.npz"))
    return files


def states_at(root, timesteps):
    """The capture's state at each of the given non-decreasing timesteps.

    A kept state is loaded. Any other is replayed in memory from the newest kept state before
    it, or onwards from the previous state yielded when that is later; the saved random state
    makes the replay exact. In a capture thinned to one state per million steps, a replay can
    cover up to a million steps, about ten minutes.
    """
    root = Path(root)
    settings = json.loads((root / "capture_plan.json").read_text(encoding="utf-8"))["simulation"]
    files = state_files(root)
    kept = np.array([MS._checkpoint_timestep(path) for path in files], dtype=np.int64)
    state = None
    for target in np.asarray(timesteps, dtype=np.int64).tolist():
        if state is not None and target < state["timestep"]:
            raise ValueError("states_at needs non-decreasing timesteps")
        index = int(np.searchsorted(kept, target, side="right")) - 1
        if index < 0:
            raise ValueError(f"t = {target:,} is before the capture's first state")
        if state is None or state["timestep"] < kept[index]:
            state = MS.load_checkpoint(files[index])
        if state["timestep"] < target:
            steps = target - state["timestep"]
            result = MS.simulation_from_state(
                state, gamma=settings["gamma"], alpha=settings["alpha"], T=steps,
                track_every=steps,
                max_forbidden_cycle_length=settings["max_forbidden_cycle_length"],
                max_cycle_rejection_attempts=settings["max_cycle_rejection_attempts"])
            state = {key: result[key] for key in ("lattice", "Gamma", "newest_species",
                                                  "timestep", "rng_state")}
            state["current_species"] = [int(species) for species in result["current_species"]]
            state["diversity"] = len(state["current_species"])
        yield state


def switch_points(timesteps, diversity, rule, smooth=200):
    """Sample indices of a capture's five snapshots and of its rise and collapse windows.

    On the living-species curve smoothed over ``smooth`` samples: the low point the rise starts
    from, the steepest rise up to the peak it reaches, the most species between the rise and
    the collapse, the steepest decline from the last peak before the collapse, and the fewest
    species after the collapse. The rise and collapse are the capture's regime switches.
    """
    timesteps = np.asarray(timesteps, dtype=np.int64)
    smoothed = _gaussian_smooth(np.asarray(diversity, dtype=np.float64), smooth)
    rate = np.gradient(smoothed, timesteps / 1e6)
    switches, _, _ = regime_switches(timesteps, diversity, rule)
    rise = next(switch["timestep"] for switch in switches if switch["up"])
    collapse = next(switch["timestep"] for switch in switches
                    if not switch["up"] and switch["timestep"] > rise)
    i_rise, i_collapse = (int(index) for index in np.searchsorted(timesteps, [rise, collapse]))

    def walk(index, step, sign):
        # Follow the curve from index while sign × living species keeps growing; a flat
        # stretch ends the walk, so a plateau's near end counts as the trough or peak.
        while (0 <= index + step < smoothed.size
               and sign * smoothed[index + step] > sign * smoothed[index]):
            index += step
        return index

    trough, first_peak = walk(i_rise, -1, -1), walk(i_rise, 1, 1)
    last_high = i_collapse - int(np.argmax(smoothed[i_collapse::-1] >= rule["high"]))
    last_peak = walk(last_high, -1, 1)
    low = i_collapse + int(np.argmin(smoothed[i_collapse:]))
    return {
        "rise": i_rise, "collapse": i_collapse,
        "snapshots": {
            "Before the rise": trough,
            "Steepest rise": trough + int(np.argmax(rate[trough:first_peak + 1])),
            "High diversity": i_rise + int(np.argmax(smoothed[i_rise:i_collapse])),
            "Steepest collapse": last_peak + int(np.argmin(rate[last_peak:i_collapse + 1])),
            "Low after the collapse": low,
        },
        "windows": {"Rise": (trough, first_peak), "Collapse": (last_peak, low)},
    }


@njit(cache=True)
def patch_labels(lattice):
    """Root site of each site's same-species patch, -1 off the lattice; patches wrap at the edges.

    The same four-neighbour union-find as sim_modules.count_patches, kept per site.
    """
    rows, columns = lattice.shape
    parent = np.arange(lattice.size, dtype=np.int64)
    sizes = np.ones(lattice.size, dtype=np.int64)
    for row in range(rows):
        for column in range(columns):
            species = lattice[row, column]
            if species <= 0:
                continue
            site = row * columns + column
            right = column + 1 if column + 1 < columns else 0
            if lattice[row, right] == species:
                _merge_patch_sites(parent, sizes, site, row * columns + right)
            down = row + 1 if row + 1 < rows else 0
            if lattice[down, column] == species:
                _merge_patch_sites(parent, sizes, site, down * columns + column)
    labels = np.full(lattice.size, -1, dtype=np.int64)
    flat = lattice.ravel()
    for site in range(lattice.size):
        if flat[site] > 0:
            labels[site] = _patch_root(parent, site)
    return labels.reshape(rows, columns)


def strong_components(Gamma):
    """Size of each species' strongly connected component in the invasion network.

    Reachability is closed by repeated squaring of (I + Gamma); two species share a component
    when each reaches the other. A size above one means the species lies on a directed cycle.
    """
    size = Gamma.shape[0]
    reach = (Gamma > 0) | np.eye(size, dtype=bool)
    while True:
        wider = (reach.astype(np.float32) @ reach.astype(np.float32)) > 0
        if np.array_equal(wider, reach):
            break
        reach = wider
    mutual = reach & reach.T
    return mutual.sum(axis=1), mutual


def state_features(path):
    """Community, network and lattice measures of one saved state."""
    with np.load(path) as saved:
        return features_of(saved["lattice"], saved["Gamma"], saved["current_species"],
                           int(saved["newest_species"]), int(saved["timestep"]))


def features_of(lattice, Gamma, species, introduced, timestep):
    """Community, network and lattice measures of one state held in memory."""
    Gamma = np.asarray(Gamma).astype(bool)
    species = np.asarray(species, dtype=np.int64)
    # Gamma rows follow current_species; map every occupied site to its species' row.
    order = np.argsort(species)
    occupied = lattice > 0
    row = np.full(lattice.shape, -1, dtype=np.int64)
    row[occupied] = order[np.searchsorted(species[order], lattice[occupied])]
    abundance = np.bincount(row[occupied], minlength=species.size)
    share = abundance / max(int(abundance.sum()), 1)
    held = share[share > 0]

    # Each unordered pair of neighbouring sites once: right and down, wrapping at the edges.
    front = np.zeros(species.size, dtype=np.int64)
    boundaries = active = 0
    for axis in (0, 1):
        neighbour = np.roll(row, -1, axis=axis)
        pair = (row >= 0) & (neighbour >= 0) & (row != neighbour)
        a, b = row[pair], neighbour[pair]
        forward, backward = Gamma[a, b], Gamma[b, a]
        boundaries += int(pair.sum())
        active += int(np.count_nonzero(forward | backward))
        # A species' front is how many neighbouring sites it could take over right now,
        # which sets how fast it spreads.
        front += np.bincount(a[forward], minlength=species.size)
        front += np.bincount(b[backward], minlength=species.size)

    # A newcomer takes over the whole patch it lands in (it always invades that species), so a
    # species with one patch is erased by an introduction landing anywhere on it.
    labels = patch_labels(lattice)
    rows_and_patches = np.unique(row[occupied] * lattice.size + labels[occupied])
    species_patches = np.bincount(rows_and_patches // lattice.size, minlength=species.size)
    patch_sizes = np.unique(labels[occupied], return_counts=True)[1]

    in_degree = Gamma.sum(axis=0)
    out_degree = Gamma.sum(axis=1)
    component, mutual = strong_components(Gamma)
    largest = int(component.max()) if species.size else 0
    in_largest = (mutual[int(np.argmax(component))] if largest > 1
                  else np.zeros(species.size, dtype=bool))
    top = in_degree == 0  # No living species can invade these.
    return {
        "timestep": timestep, "species": int(species.size), "introduced": introduced,
        "largest_share": float(share.max()) if species.size else 0.0,
        "effective_species": float(np.exp(-np.sum(held * np.log(held)))) if held.size else 0.0,
        "links": int(Gamma.sum()), "scc_largest": largest,
        "scc_species": int(np.count_nonzero(component > 1)),
        "top_species": int(top.sum()), "top_share": float(share[top].sum()),
        "boundaries": boundaries, "active_boundaries": active,
        "patches": int(patch_sizes.size),
        "single_patch_share": float(share[species_patches == 1].sum()),
        "largest_patch_share": float(patch_sizes.max() / max(int(abundance.sum()), 1))
        if patch_sizes.size else 0.0,
        "species_id": species, "abundance": abundance, "front": front,
        "in_degree": in_degree, "out_degree": out_degree, "in_largest_scc": in_largest,
        "patches_per_species": species_patches,
    }


def _write_npz(path, arrays):
    descriptor, name = tempfile.mkstemp(prefix=".features_", suffix=".tmp.npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_features(root, workers=None, refresh=False):
    """Features of every state of a capture, from its cache when that is still current.

    The cache is current when it covers every tracked sample in series.npz. Building it needs
    every state, so a thinned capture keeps its cache and refuses to rebuild it.
    """
    root = Path(root)
    files = state_files(root)
    cache = root / CACHE
    samples = len(files)
    if (root / "series.npz").exists():
        with np.load(root / "series.npz") as series:
            samples = series["timesteps"].size
    if cache.exists() and not refresh:
        with np.load(cache) as saved:
            if int(saved["version"]) == CACHE_VERSION and saved["timestep"].size == samples:
                return {key: saved[key] for key in saved.files}
    if len(files) < samples:
        raise FileNotFoundError(
            f"{cache} is missing or stale, and rebuilding it needs all {samples:,} states; "
            f"the capture keeps {len(files):,} (thinned by tools/prune_checkpoints.py)")
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        states = list(pool.map(state_features, files, chunksize=64))
    arrays = {key: np.array([state[key] for state in states]) for key in SCALARS}
    counts = np.array([state["species_id"].size for state in states])
    arrays["offsets"] = np.concatenate([[0], np.cumsum(counts)])
    for key in PER_SPECIES:
        arrays[key] = np.concatenate([state[key] for state in states])
    arrays["version"] = np.int64(CACHE_VERSION)
    _write_npz(cache, arrays)
    return arrays
