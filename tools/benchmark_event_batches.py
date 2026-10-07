"""Reproducible experiment: preserve exact trajectories with independent batches.

This is deliberately separate from the production simulator: thread dispatch and
dependency scanning may cost more than these very small invasion operations.
Run from the project directory, for example::

    python tools/benchmark_event_batches.py --side 200 --timesteps 100 --threads 1 2 4
    python tools/benchmark_event_batches.py --side 512 --diversity 40 --waves

Each greedy batch contains consecutive events with disjoint source/target sites.
Lattice invasion outcomes can then be computed concurrently. Species counts,
extinction bookkeeping, free-slot order, and introductions are still committed
in original event order. Every introduction is a barrier. Random draws are
identical to the production kernel, including their ordering.

With ``--waves``, a second candidate groups an entire introduction-free segment
by dependency level. Each event's level is one above the latest event touching
either its source or target. Independent later events may run before earlier
events on different sites; overlapping events retain their original order.
Species bookkeeping is replayed after all waves, before the next introduction.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import numba
from numba import njit, prange, set_num_threads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sim_modules as simulation


@njit(cache=True)
def _invade_serial(codes, start, end, lattice, neighbours, Gamma, sources, targets):
    for event in range(start, end):
        source_site = codes[event] >> 2
        target_site = neighbours[source_site, codes[event] & 3]
        source_slot = lattice[source_site]
        target_slot = lattice[target_site]
        sources[event] = -1
        if source_slot >= 0 and (
            target_slot == -1 or (
                target_slot >= 0 and target_slot != source_slot
                and Gamma[source_slot, target_slot] != 0
            )
        ):
            lattice[target_site] = source_slot
            sources[event] = source_slot
            targets[event] = target_slot


@njit(cache=True, parallel=True)
def _invade_parallel(codes, start, end, lattice, neighbours, Gamma, sources, targets):
    for event in prange(start, end):
        source_site = codes[event] >> 2
        target_site = neighbours[source_site, codes[event] & 3]
        source_slot = lattice[source_site]
        target_slot = lattice[target_site]
        sources[event] = -1
        if source_slot >= 0 and (
            target_slot == -1 or (
                target_slot >= 0 and target_slot != source_slot
                and Gamma[source_slot, target_slot] != 0
            )
        ):
            lattice[target_site] = source_slot
            sources[event] = source_slot
            targets[event] = target_slot


@njit(cache=True, parallel=True)
def _invade_wave(codes, order, start, end, lattice, neighbours, Gamma, sources, targets):
    for position in prange(start, end):
        event = order[position]
        source_site = codes[event] >> 2
        target_site = neighbours[source_site, codes[event] & 3]
        source_slot = lattice[source_site]
        target_slot = lattice[target_site]
        sources[event] = -1
        if source_slot >= 0 and (
            target_slot == -1 or (
                target_slot >= 0 and target_slot != source_slot
                and Gamma[source_slot, target_slot] != 0
            )
        ):
            lattice[target_site] = source_slot
            sources[event] = source_slot
            targets[event] = target_slot


@njit(cache=True)
def _invade_wave_serial(codes, order, start, end, lattice, neighbours, Gamma, sources, targets):
    for position in range(start, end):
        event = order[position]
        source_site = codes[event] >> 2
        target_site = neighbours[source_site, codes[event] & 3]
        source_slot = lattice[source_site]
        target_slot = lattice[target_site]
        sources[event] = -1
        if source_slot >= 0 and (
            target_slot == -1 or (
                target_slot >= 0 and target_slot != source_slot
                and Gamma[source_slot, target_slot] != 0
            )
        ):
            lattice[target_site] = source_slot
            sources[event] = source_slot
            targets[event] = target_slot


@njit(cache=True)
def run_batched(
    slot_lattice, usable_sites, neighbours, rng, Gamma, active_slots,
    slot_species_ids, species_counts, slot_to_active_position, free_slots,
    live_count, next_unused_slot, free_count, newest_species, gamma,
    introduction_probability, forced_introduction_timesteps, timesteps,
    track_every, parallel, statistics_out, waves=False,
):
    """Experimental exact kernel; ``parallel=False`` measures batching overhead."""
    total_sites = slot_lattice.size
    usable_count = usable_sites.size
    records = timesteps // track_every + (timesteps % track_every != 0)
    diversity_history = np.empty(records, dtype=np.int64)
    record_index = 0
    touched = np.zeros(total_sites, dtype=np.int64)
    sources = np.empty(total_sites, dtype=np.int32)
    targets = np.empty(total_sites, dtype=np.int32)
    last_layer = np.zeros(total_sites, dtype=np.int32)
    event_layer = np.empty(total_sites, dtype=np.int32)
    layer_counts = np.zeros(total_sites + 1, dtype=np.int64)
    layer_offsets = np.empty(total_sites + 2, dtype=np.int64)
    order = np.empty(total_sites, dtype=np.int64)
    batch_stamp = 0
    batch_count = 0
    batch_max = 0

    for timestep in range(1, timesteps + 1):
        event_codes = rng.integers(0, 4 * total_sites, size=total_sites)
        introduction_draws = rng.random(total_sites)
        start = 0
        while start < total_sites:
            batch_stamp += 1
            end = start
            introduction = False
            while end < total_sites:
                source_site = event_codes[end] >> 2
                target_site = neighbours[source_site, event_codes[end] & 3]
                if waves:
                    source_layer = 0
                    target_layer = 0
                    if touched[source_site] == batch_stamp:
                        source_layer = last_layer[source_site]
                    if touched[target_site] == batch_stamp:
                        target_layer = last_layer[target_site]
                    level = max(source_layer, target_layer) + 1
                    last_layer[source_site] = level
                    last_layer[target_site] = level
                    event_layer[end] = level
                    layer_counts[level] += 1
                elif touched[source_site] == batch_stamp or touched[target_site] == batch_stamp:
                    break
                touched[source_site] = batch_stamp
                touched[target_site] = batch_stamp
                introduction = (
                    timestep <= forced_introduction_timesteps and end == 0
                ) or introduction_draws[end] < introduction_probability
                end += 1
                if introduction:
                    break

            if waves:
                maximum_layer = 0
                for event in range(start, end):
                    maximum_layer = max(maximum_layer, event_layer[event])
                layer_offsets[1] = 0
                for level in range(1, maximum_layer + 1):
                    count = layer_counts[level]
                    layer_offsets[level + 1] = layer_offsets[level] + count
                    batch_max = max(batch_max, count)
                    layer_counts[level] = 0
                for event in range(start, end):
                    level = event_layer[event]
                    position = layer_offsets[level] + layer_counts[level]
                    order[position] = event
                    layer_counts[level] += 1
                for level in range(1, maximum_layer + 1):
                    if parallel:
                        _invade_wave(
                            event_codes, order, layer_offsets[level],
                            layer_offsets[level + 1], slot_lattice, neighbours,
                            Gamma, sources, targets,
                        )
                    else:
                        _invade_wave_serial(
                            event_codes, order, layer_offsets[level],
                            layer_offsets[level + 1], slot_lattice, neighbours,
                            Gamma, sources, targets,
                        )
                    layer_counts[level] = 0
                batch_count += maximum_layer
            elif parallel:
                _invade_parallel(
                    event_codes, start, end, slot_lattice, neighbours, Gamma,
                    sources, targets,
                )
            else:
                _invade_serial(
                    event_codes, start, end, slot_lattice, neighbours, Gamma,
                    sources, targets,
                )

            # Lattice writes are independent; shared species state is not.
            for event in range(start, end):
                source_slot = sources[event]
                if source_slot >= 0:
                    target_slot = targets[event]
                    species_counts[source_slot] += 1
                    if target_slot >= 0:
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

            if introduction:
                introduction_site = usable_sites[rng.integers(0, usable_count)]
                replaced_slot = slot_lattice[introduction_site]
                if free_count:
                    free_count -= 1
                    new_slot = free_slots[free_count]
                else:
                    if next_unused_slot == Gamma.shape[0]:
                        (
                            Gamma, active_slots, slot_species_ids, species_counts,
                            slot_to_active_position, free_slots,
                        ) = simulation._grow_species_storage(
                            Gamma, active_slots, slot_species_ids, species_counts,
                            slot_to_active_position, free_slots,
                        )
                    new_slot = next_unused_slot
                    next_unused_slot += 1

                newest_species += 1
                slot_species_ids[new_slot] = newest_species
                species_counts[new_slot] = 1
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
            if not waves:
                batch_count += 1
                batch_max = max(batch_max, end - start)
            start = end

        if timestep % track_every == 0 or timestep == timesteps:
            diversity_history[record_index] = live_count
            record_index += 1

    statistics_out[0] = batch_count
    statistics_out[1] = batch_max
    return (
        slot_lattice, Gamma, active_slots, slot_species_ids, species_counts,
        slot_to_active_position, free_slots, live_count, next_unused_slot,
        free_count, newest_species, diversity_history,
    )


def _capture_inputs(config, kind="main"):
    """Reuse production initialization, stopping just before the compiled call."""
    original = simulation._run_compiled_simulation
    captured = []

    class Captured(Exception):
        pass

    def capture(*args):
        captured.extend(args)
        raise Captured

    simulation._run_compiled_simulation = capture
    try:
        if kind == "percolation":
            config.run_percolation()
        else:
            config.run_main()
    except Captured:
        pass
    finally:
        simulation._run_compiled_simulation = original
    if not captured:
        raise RuntimeError("simulation did not invoke its compiled kernel")
    return tuple(captured)


def _clone_args(args):
    return tuple(copy.deepcopy(value) for value in args)


def _assert_exact(expected, actual, expected_rng, actual_rng, rows, columns):
    # Unused active/free slots contain uninitialized memory and are not state.
    for index in (0, 1, 3, 4, 5, 11):
        np.testing.assert_array_equal(actual[index], expected[index])
    for index in (7, 8, 9, 10):
        assert actual[index] == expected[index], index
    live_count, free_count = expected[7], expected[9]
    np.testing.assert_array_equal(actual[2][:live_count], expected[2][:live_count])
    np.testing.assert_array_equal(actual[6][:free_count], expected[6][:free_count])
    if hasattr(expected_rng, "bit_generator"):
        assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    else:
        assert actual_rng.draws == expected_rng.draws
    expected_public = simulation._export_simulation_state(expected[:-1], rows, columns)
    actual_public = simulation._export_simulation_state(actual[:-1], rows, columns)
    assert actual_public.keys() == expected_public.keys()
    for key, expected_value in expected_public.items():
        actual_value = actual_public[key]
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


class _FixedRng:
    """Deterministic Python-only oracle for deliberately interacting events."""

    def __init__(self, introduction_event):
        self.codes = np.array([1, 17, 5, 25, 29, 9, 13, 21], dtype=np.int64)
        self.trials = np.ones(8, dtype=np.float64)
        if introduction_event is not None:
            self.trials[introduction_event] = 0.0
        self.draws = []

    def integers(self, low, high, size=None):
        self.draws.append(("integers", low, high, size))
        return self.codes.copy() if size is not None else 3

    def random(self, size=None):
        self.draws.append(("random", size))
        if size is not None:
            return self.trials.copy()
        return 0.25 if len(self.draws) % 2 else 0.75


def _validate_fixed_events():
    """Disjoint singleton extinctions, dependent attacks, and slot-reuse barriers."""
    comparisons = 0
    sites = np.arange(8, dtype=np.int32)
    neighbours = np.column_stack((
        (sites - 1) % 8, (sites + 1) % 8,
        (sites - 1) % 8, (sites + 1) % 8,
    ))
    for introduction_event in (None, 1, 2, 7):
        Gamma = np.zeros((16, 16), dtype=np.uint8)
        Gamma[:8, :8] = 1
        np.fill_diagonal(Gamma, 0)
        active = np.full(16, -1, dtype=np.int32)
        active[:8] = sites
        species_ids = np.full(16, -1, dtype=np.int64)
        species_ids[:8] = np.arange(1, 9) * 10
        counts = np.zeros(16, dtype=np.int64)
        counts[:8] = 1
        positions = np.full(16, -1, dtype=np.int32)
        positions[:8] = sites
        args = (
            sites.copy(), sites.astype(np.intp), neighbours,
            _FixedRng(introduction_event), Gamma, active, species_ids, counts,
            positions, np.full(16, -1, dtype=np.int32), 8, 8, 0, 80,
            0.5, 0.5, 0, 1, 1,
        )
        expected_args = _clone_args(args)
        expected = simulation._run_compiled_simulation.py_func(*expected_args)
        for parallel, waves in ((False, False), (True, False), (False, True), (True, True)):
            actual_args = _clone_args(args)
            actual = run_batched.py_func(
                *actual_args, parallel, np.zeros(2, dtype=np.int64), waves,
            )
            _assert_exact(expected, actual, expected_args[3], actual_args[3], 1, 8)
            comparisons += 1
    return comparisons


def validate():
    """Cover extinction, growth, introductions, periodic self-neighbors and holes."""
    cases = [
        (1, 1, 1, 0.07, 0.0125, True, 0.0),
        (1, 7, 5, 0.5, 10.0, True, 0.0),
        (7, 1, 5, 0.9, 2.0, False, 0.0),
        (6, 7, 2, 0.3, 30.0, True, 0.0),
        (6, 7, 20, 1.0, 0.0, False, 0.0),
        (8, 9, 5, 0.6, 20.0, True, 0.3),
        (8, 9, 5, 0.0, 20.0, True, 0.3),
        (8, 9, 0, 0.3, 20.0, True, 1.0),
    ]
    comparisons = 0
    for rows, columns, diversity, gamma, alpha, populate, p in cases:
        for seed in (0, 1, 67):
            config = simulation.SimulationConfig(
                columns, rows, diversity, gamma, alpha, 12, track_every=5,
                seed=seed, populate_first_100=populate, p=p,
            )
            args = _capture_inputs(config, "percolation" if p else "main")
            expected_args = _clone_args(args)
            expected = simulation._run_compiled_simulation(*expected_args)
            for parallel, waves in ((False, False), (True, False), (False, True), (True, True)):
                actual_args = _clone_args(args)
                stats = np.zeros(2, dtype=np.int64)
                actual = run_batched(*actual_args, parallel, stats, waves)
                _assert_exact(expected, actual, expected_args[3], actual_args[3], rows, columns)
                comparisons += 1
    # Continuation with empty usable sites and permanent, noncontiguous IDs.
    config = simulation.SimulationConfig(
        7, 5, 3, 0.4, 2.0, 12, track_every=5, seed=67,
        populate_first_100=True,
    )
    args = list(_capture_inputs(config))
    args[0][::4] = -1
    args[7][:3] = np.bincount(args[0][args[0] >= 0], minlength=3)
    args[6][:3] = [17, 23, 100]
    args[13] = 100
    args[16] = 6  # Continue from t=94: forced introductions end after t=100.
    expected_args = _clone_args(args)
    expected = simulation._run_compiled_simulation(*expected_args)
    for parallel, waves in ((False, False), (True, False), (False, True), (True, True)):
        actual_args = _clone_args(args)
        actual = run_batched(*actual_args, parallel, np.zeros(2, dtype=np.int64), waves)
        _assert_exact(expected, actual, expected_args[3], actual_args[3], 5, 7)
        comparisons += 1
    return comparisons + _validate_fixed_events()


def benchmark(side, timesteps, diversity, gamma, alpha, threads, repeats, waves=False):
    maximum_threads = numba.config.NUMBA_NUM_THREADS
    if not threads or any(count < 1 or count > maximum_threads for count in threads):
        raise ValueError(f"thread counts must be between 1 and {maximum_threads}")
    config = simulation.SimulationConfig(
        side, side, diversity, gamma, alpha, timesteps, track_every=timesteps,
        seed=67, populate_first_100=True,
    )
    args = _capture_inputs(config)
    expected_args = _clone_args(args)
    expected = simulation._run_compiled_simulation(*expected_args)
    results = []
    modes = [("production", 1), ("batched_serial", 1)] + [
        ("batched_parallel", count) for count in threads
    ]
    if waves:
        modes.append(("dependency_waves_serial", 1))
        modes.extend(("dependency_waves", count) for count in threads)
    for name, thread_count in modes:
        set_num_threads(thread_count)
        times = []
        for repeat in range(repeats):
            actual_args = _clone_args(args)
            stats = np.zeros(2, dtype=np.int64)
            start = time.perf_counter()
            if name == "production":
                actual = simulation._run_compiled_simulation(*actual_args)
            else:
                actual = run_batched(
                    *actual_args, name not in ("batched_serial", "dependency_waves_serial"), stats,
                    name.startswith("dependency_waves"),
                )
            elapsed = time.perf_counter() - start
            _assert_exact(expected, actual, expected_args[3], actual_args[3], side, side)
            times.append(elapsed)
        result = {
            "name": name, "threads": thread_count,
            "median_seconds": statistics.median(times), "seconds": times,
            "exact": True,
        }
        if name != "production":
            result.update(
                batches=int(stats[0]), maximum_batch=int(stats[1]),
                mean_batch=side * side * timesteps / int(stats[0]),
            )
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--diversity", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.0125)
    parser.add_argument(
        "--threads", type=int, nargs="+",
        help="worker thread counts (default: available counts from 1, 2, 4)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--waves", action="store_true", help="also benchmark dependency-wave scheduling")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    maximum_threads = numba.config.NUMBA_NUM_THREADS
    if options.threads is None:
        options.threads = [count for count in (1, 2, 4) if count <= maximum_threads]
    elif any(count < 1 or count > maximum_threads for count in options.threads):
        parser.error(f"--threads counts must be between 1 and {maximum_threads}")
    set_num_threads(min(2, maximum_threads))
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__,
        "numba": numba.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "available_numba_threads": maximum_threads,
        "timing_excludes_initialization_and_jit": True,
    }
    print(json.dumps({"environment": metadata}), flush=True)
    validated = validate()
    print(json.dumps({"exact_validation_comparisons": validated}), flush=True)
    if options.validate_only:
        return
    results = benchmark(
        options.side, options.timesteps, options.diversity, options.gamma,
        options.alpha, options.threads, options.repeats, options.waves,
    )
    if options.output:
        options.output.write_text(json.dumps({
            "environment": metadata,
            "settings": vars(options) | {"output": str(options.output)},
            "exact_validation_comparisons": validated, "results": results,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
