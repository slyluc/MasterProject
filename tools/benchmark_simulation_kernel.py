"""Compare exact sequential kernels, including deliberately exploratory modes.

Run from the project root, for example::

    python tools/benchmark_simulation_kernel.py --variants baseline flat buffers segments --timesteps 1000 --repeats 3
    python tools/benchmark_simulation_kernel.py --variants baseline production --size 512 --timesteps 200 --repeats 5 --also-no-introductions

The baseline source is frozen below so rerunning after a production optimization
still compares against the pre-experiment kernel. Each timing starts with the
same seed, includes simulation setup, excludes compilation, and is checked for
identical lattice, Gamma, ordered species, histories, counters, and RNG state.
All updates in these modes are sequential; none parallelizes interacting events.
The int32 experiment is intended for small lattices whose event-code bound fits
in int32; production uses the original draw dtype. "flat", "noop", and
"segments" modifiers can be combined with underscores, such as "flat_noop".
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import numba
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sim_modules as MS


_BASELINE_SOURCE = '''def _run_compiled_simulation(
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
'''


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Unexpected baseline kernel source: " + before[:80])
    return source.replace(before, after, 1)


@njit(cache=True, inline="never")
def introductions_pending(draws, probability, forced_timesteps, timestep):
    """Isolate the rare-introduction scan from the generated invasion loop."""
    if timestep <= forced_timesteps:
        return True
    if probability > 0:
        for draw in draws:
            if draw < probability:
                return True
    return False


def candidate_source(mode):
    source = _BASELINE_SOURCE
    source = replace_once(source, "def _run_compiled_simulation(", "def candidate_kernel(")
    modifiers = set(mode.split("_"))
    if modifiers - {"baseline", "flat", "buffers", "int32", "segments", "noop", "helper"}:
        raise ValueError(mode)
    if {"buffers", "int32"} <= modifiers:
        raise ValueError("The buffer and int32 experiments are separate modes")
    if "flat" in modifiers:
        source = replace_once(source, "    record_index = 0\n", "    record_index = 0\n    flat_neighbours = neighbours.ravel()\n")
        source = replace_once(source, "neighbours[source_site, direction]", "flat_neighbours[event_codes[event]]")
    if "buffers" in modifiers:
        source = replace_once(source, "    record_index = 0\n", "    record_index = 0\n    event_codes = np.empty(total_sites, dtype=np.int64)\n    introduction_draws = np.empty(total_sites, dtype=np.float64)\n")
        source = replace_once(source, "        event_codes = rng.integers(0, 4 * total_sites, size=total_sites)\n        introduction_draws = rng.random(total_sites)\n", "        for index in range(total_sites):\n            event_codes[index] = rng.integers(0, 4 * total_sites)\n        for index in range(total_sites):\n            introduction_draws[index] = rng.random()\n")
    if "int32" in modifiers:
        source = replace_once(source, "event_codes = rng.integers(0, 4 * total_sites, size=total_sites)", "event_codes = rng.integers(0, 4 * total_sites, size=total_sites, dtype=np.int32)")
    if "noop" in modifiers:
        skip_proven_noops = """        if live_count == 0 or (
            live_count == 1 and species_counts[active_slots[0]] == usable_count
        ):
            introduction_pending = timestep <= forced_introduction_timesteps
            if not introduction_pending and introduction_probability > 0:
                for draw in introduction_draws:
                    if draw < introduction_probability:
                        introduction_pending = True
                        break
            if not introduction_pending:
                if timestep % track_every == 0 or timestep == timesteps:
                    diversity_history[record_index] = live_count
                    record_index += 1
                continue

"""
        if "helper" in modifiers:
            original_scan = """            introduction_pending = timestep <= forced_introduction_timesteps
            if not introduction_pending and introduction_probability > 0:
                for draw in introduction_draws:
                    if draw < introduction_probability:
                        introduction_pending = True
                        break
"""
            helper_call = """            introduction_pending = introductions_pending(
                introduction_draws, introduction_probability,
                forced_introduction_timesteps, timestep,
            )
"""
            skip_proven_noops = replace_once(skip_proven_noops, original_scan, helper_call)
        source = replace_once(source, "        for event in range(total_sites):\n", skip_proven_noops + "        for event in range(total_sites):\n")
    if "segments" in modifiers:
        loop_start = source.index("        for event in range(total_sites):\n")
        invasion_start = loop_start + len("        for event in range(total_sites):\n")
        intro_start = source.index("            # A Bernoulli introduction trial", invasion_start)
        intro_body_start = source.index("                introduction_site =", intro_start)
        loop_end = source.index("\n        if timestep % track_every", intro_body_start)
        invasion = source[invasion_start:intro_start].rstrip()
        introduction = source[intro_body_start:loop_end].rstrip()
        segment_invasion = "\n".join("    " + line for line in invasion.splitlines())
        intro_invasion = "\n".join(line for line in invasion.splitlines())
        unguarded_intro = "\n".join(line[4:] if line.startswith("    ") else line for line in introduction.splitlines())
        split_loop = """        event = 0
        while event < total_sites:
            segment_end = event
            if not (timestep <= forced_introduction_timesteps and event == 0):
                while segment_end < total_sites and not (introduction_draws[segment_end] < introduction_probability):
                    segment_end += 1
            for event in range(event, segment_end):
INVASION_SEGMENT
            if segment_end == total_sites:
                break
            event = segment_end
INVASION_INTRODUCTION
INTRODUCTION
            event += 1
""".replace("INVASION_SEGMENT", segment_invasion).replace("INVASION_INTRODUCTION", intro_invasion).replace("INTRODUCTION", unguarded_intro)
        source = source[:loop_start] + split_loop + source[loop_end:]
    return source


def compile_candidate(mode):
    if mode == "production":
        return MS._run_compiled_simulation
    namespace = {"np": np, "_grow_species_storage": MS._grow_species_storage,
                 "introductions_pending": introductions_pending}
    exec(compile(candidate_source(mode), "<simulation-kernel-" + mode + ">", "exec"), namespace)
    return njit(cache=False)(namespace["candidate_kernel"])


@contextmanager
def installed(kernel):
    original = MS._run_compiled_simulation
    MS._run_compiled_simulation = kernel
    try:
        yield
    finally:
        MS._run_compiled_simulation = original


def fingerprint(result):
    digest = hashlib.sha256()
    for name, dtype in (("lattice", "<i8"), ("Gamma", "u1"), ("current_species", "<i8"),
                        ("tracked_timesteps", "<i8"), ("diversity_history", "<i8")):
        array = np.asarray(result[name], dtype=dtype, order="C")
        digest.update(json.dumps([name, dtype, list(array.shape)]).encode())
        digest.update(array.tobytes())
    scalars = {name: int(result[name]) for name in ("newest_species", "diversity", "timestep",
                                                    "target_timestep", "initial_newest_species")}
    scalars["rng_state"] = result["rng_state"]
    def convert_numpy(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    digest.update(json.dumps(scalars, sort_keys=True, default=convert_numpy).encode())
    return digest.hexdigest()


def validate(kernels):
    cases = [
        ("main_simulation", dict(L_col=9, L_row=7, D=3, gamma=.65, alpha=12, T=80, track_every=4, seed=491)),
        ("main_simulation", dict(L_col=8, L_row=5, D=3, gamma=.45, alpha=.4, T=117, track_every=5, seed=1209, populate_first_100=True)),
        ("percolation_simulation", dict(L_col=7, L_row=5, D=4, gamma=.6, alpha=6, T=40, track_every=4, seed=823, p=.45)),
        ("main_simulation", dict(L_col=8, L_row=1, D=3, gamma=.7, alpha=4, T=31, track_every=3, seed=29)),
        ("main_simulation", dict(L_col=1, L_row=8, D=3, gamma=.7, alpha=4, T=31, track_every=3, seed=29)),
        ("main_simulation", dict(L_col=1, L_row=1, D=1, gamma=.65, alpha=.8, T=30, track_every=2, seed=998, populate_first_100=True)),
        ("main_simulation", dict(L_col=5, L_row=4, D=6, gamma=0, alpha=12, T=17, track_every=4, seed=114)),
        ("percolation_simulation", dict(L_col=4, L_row=3, D=0, gamma=.6, alpha=999, T=15, track_every=4, seed=867, p=1, populate_first_100=True)),
        ("percolation_simulation", dict(L_col=8, L_row=6, D=1, gamma=.6, alpha=2, T=25, track_every=4, seed=821, p=.4)),
        ("simulation_from_state", dict(initial_state={"lattice": np.array([[0, 99, 7, -1], [31, 0, 7, 99], [31, 7, 0, 99]], dtype=np.int64), "Gamma": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8), "current_species": [99, 7, 31], "newest_species": 104, "timestep": 94}, gamma=.55, alpha=6, T=19, track_every=3, seed=109, populate_first_100=True)),
        ("simulation_from_state", dict(initial_state={"lattice": np.array([[99, 0, 99, -1], [0, 99, 0, 99]], dtype=np.int64), "Gamma": np.zeros((1, 1), dtype=np.uint8), "current_species": [99], "newest_species": 104, "timestep": 0}, gamma=.6, alpha=0, T=19, track_every=3, seed=109)),
        ("simulation_from_state", dict(initial_state={"lattice": np.array([[0, 0, 0, -1], [0, 0, 0, 0]], dtype=np.int64), "Gamma": np.zeros((0, 0), dtype=np.uint8), "current_species": [], "newest_species": 104, "timestep": 0}, gamma=.6, alpha=2, T=19, track_every=3, seed=109)),
        ("main_simulation", dict(L_col=5, L_row=4, D=1, gamma=.7, alpha=float("nan"), T=17, track_every=4, seed=114)),
    ]
    for generator_name in ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"):
        for cached_half in (False, True):
            rng = np.random.Generator(getattr(np.random, generator_name)(113))
            if cached_half:
                rng.integers(0, 4, dtype=np.uint32)
            state = dict(lattice=np.array([[0, 99, 7, -1], [31, 0, 7, 99]], dtype=np.int64),
                         Gamma=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8),
                         current_species=[99, 7, 31], newest_species=104,
                         timestep=12, rng_state=rng.bit_generator.state)
            cases.append(("simulation_from_state", dict(initial_state=state, gamma=.55, alpha=6,
                                                        T=19, track_every=3, populate_first_100=True)))
    for function_name, options in cases:
        expected = None
        for mode, kernel in kernels.items():
            with installed(kernel):
                actual = fingerprint(getattr(MS, function_name)(**options))
            if expected is None:
                expected = actual
            elif actual != expected:
                raise AssertionError(f"{mode} changed seeded trajectory for {options}")
    print(f"Validated {len(cases)} edge cases across {len(kernels)} modes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["baseline", "production"])
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--diversities", nargs="+", type=int, default=[1, 40])
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--alpha", type=float, default=.0125)
    parser.add_argument("--also-no-introductions", action="store_true",
                        help="Also measure a fully occupied one-species alpha=0 run")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    modes = list(dict.fromkeys(["baseline"] + options.variants))
    kernels = {mode: compile_candidate(mode) for mode in modes}
    validate(kernels)
    rows = []
    scenarios = [(diversity, options.alpha) for diversity in options.diversities]
    if options.also_no_introductions and (1, 0) not in scenarios:
        scenarios.append((1, 0))
    for diversity, alpha in scenarios:
        run_options = dict(L_col=options.size, L_row=options.size, D=diversity,
                           gamma=.07, alpha=alpha, T=options.timesteps, track_every=10, seed=options.seed)
        times = {mode: [] for mode in modes}
        expected = None
        for repeat in range(options.repeats):
            order = modes if repeat % 2 == 0 else list(reversed(modes))
            for mode in order:
                with installed(kernels[mode]):
                    start = time.perf_counter()
                    result = MS.main_simulation(**run_options)
                    elapsed = time.perf_counter() - start
                actual = fingerprint(result)
                if expected is None:
                    expected = actual
                elif actual != expected:
                    raise AssertionError(f"{mode} changed D={diversity} seeded trajectory")
                times[mode].append(elapsed)
                print(f"D={diversity} alpha={alpha} repeat={repeat+1} {mode}: {elapsed:.6f}s", flush=True)
        baseline = statistics.median(times["baseline"])
        for mode in modes:
            median = statistics.median(times[mode])
            row = dict(D=diversity, alpha=alpha, mode=mode, seconds=times[mode], median=median,
                       speedup=baseline / median, fingerprint=expected)
            rows.append(row)
            print(f"D={diversity} alpha={alpha} {mode}: median={median:.6f}s speedup={baseline/median:.3f}x", flush=True)
    if options.output:
        metadata = dict(python=sys.version.split()[0], numpy=np.__version__,
                        numba=numba.__version__, platform=platform.platform(),
                        cpu_count=os.cpu_count(),
                        baseline_source_sha256=hashlib.sha256(_BASELINE_SOURCE.encode()).hexdigest())
        options.output.write_text(json.dumps(dict(settings=vars(options) | {"output": str(options.output)},
                                                 metadata=metadata, results=rows), indent=2))


if __name__ == "__main__":
    main()
