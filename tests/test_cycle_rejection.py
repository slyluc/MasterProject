"""Independent graph and RNG checks for rejection of short invasion cycles."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import sim_modules as MS


def has_short_cycle(matrix, maximum=6):
    """Enumerate bounded simple directed paths, independently of the BFS code."""
    matrix = np.asarray(matrix, dtype=bool)
    for origin in range(len(matrix)):
        pending = [(origin, (origin,))]
        while pending:
            vertex, path = pending.pop()
            for neighbor in np.flatnonzero(matrix[vertex]):
                if neighbor == origin and 2 <= len(path) <= maximum:
                    return True
                if neighbor not in path and len(path) < maximum:
                    pending.append((int(neighbor), path + (int(neighbor),)))
    return False


def directed_ring(length):
    matrix = np.zeros((length, length), dtype=np.uint8)
    for vertex in range(length):
        matrix[vertex, (vertex + 1) % length] = 1
    return matrix


def raw_state():
    return {
        "lattice": np.array([[1, 1, 2, 2], [1, 3, 3, 2], [-1, 0, 3, 0]]),
        "Gamma": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8),
        "current_species": [1, 2, 3],
        "newest_species": 3,
        "timestep": 105,
        "rng_state": np.random.default_rng(819).bit_generator.state,
    }


def rejection_reference(matrix, gamma, seed, forced_index, maximum=6):
    """Draw the complete candidate, then accept/reject its directed graph."""
    rng = np.random.default_rng(seed)
    size = len(matrix)
    for attempts in range(1, 100_001):
        edges = rng.random(2 * size) < gamma
        candidate = np.zeros((size + 1, size + 1), dtype=np.uint8)
        candidate[:size, :size] = matrix
        candidate[size, :size] = edges[:size]
        candidate[:size, size] = edges[size:]
        if forced_index is not None:
            candidate[size, forced_index] = 1
        if not has_short_cycle(candidate, maximum):
            return candidate, rng.bit_generator.state, attempts
    raise AssertionError("reference rejection sampler did not converge")


def microscopic_reference(state, gamma, alpha):
    """One sequential model time unit using public species IDs and dictionaries."""
    lattice = state["lattice"].copy()
    flat = lattice.ravel()
    species = list(state["current_species"])
    newest = state["newest_species"]
    interactions = {
        (first, second): int(state["Gamma"][i, j])
        for i, first in enumerate(species) for j, second in enumerate(species)
    }
    counts = {value: int(np.count_nonzero(flat == value)) for value in species}
    usable = np.flatnonzero(flat != -1)
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(state["rng_state"])
    event_codes = rng.integers(0, 4 * flat.size, size=flat.size)
    introductions = rng.random(flat.size)
    rejected = 0
    introduction_count = 0

    def remove_extinct(value):
        if counts[value] == 0:
            position = species.index(value)
            species[position] = species[-1]
            species.pop()

    rows, columns = lattice.shape
    for event, event_code in enumerate(event_codes):
        source_site, direction = divmod(int(event_code), 4)
        source = int(flat[source_site])
        if source > 0:
            row, column = divmod(source_site, columns)
            neighbor = (
                ((row - 1) % rows, column),
                (row, (column + 1) % columns),
                ((row + 1) % rows, column),
                (row, (column - 1) % columns),
            )[direction]
            target_site = neighbor[0] * columns + neighbor[1]
            target = int(flat[target_site])
            if target == 0 or (
                target > 0 and target != source and interactions[source, target]
            ):
                flat[target_site] = source
                counts[source] += 1
                if target > 0:
                    counts[target] -= 1
                    remove_extinct(target)

        if introductions[event] < alpha * gamma / flat.size:
            site = int(usable[rng.integers(0, len(usable))])
            replaced = int(flat[site])
            newest += 1
            introduction_count += 1
            old_count = len(species)
            old_matrix = np.array([
                [interactions[first, second] for second in species]
                for first in species
            ], dtype=np.uint8).reshape(old_count, old_count)
            for _ in range(100_000):
                edges = rng.random(2 * old_count) < gamma
                candidate = np.zeros((old_count + 1, old_count + 1), dtype=np.uint8)
                candidate[:-1, :-1] = old_matrix
                candidate[-1, :-1] = edges[:old_count]
                candidate[:-1, -1] = edges[old_count:]
                if replaced > 0:
                    candidate[-1, species.index(replaced)] = 1
                if not has_short_cycle(candidate):
                    break
                rejected += 1
            else:
                raise AssertionError("reference proposal did not converge")
            for position, other in enumerate(species):
                interactions[newest, other] = int(candidate[-1, position])
                interactions[other, newest] = int(candidate[position, -1])
            interactions[newest, newest] = 0
            species.append(newest)
            counts[newest] = 1
            flat[site] = newest
            if replaced > 0:
                counts[replaced] -= 1
                remove_extinct(replaced)

    matrix = np.array([
        [interactions[first, second] for second in species]
        for first in species
    ], dtype=np.uint8).reshape(len(species), len(species))
    return {
        "lattice": lattice, "Gamma": matrix,
        "current_species": species, "newest_species": newest,
        "timestep": state["timestep"] + 1,
        "tracked_timesteps": np.array([state["timestep"], state["timestep"] + 1]),
        "diversity_history": np.array([len(state["current_species"]), len(species)]),
        "rng_state": rng.bit_generator.state,
    }, rejected, introduction_count


class CycleGraphTests(unittest.TestCase):
    def test_directed_cycles_two_through_six_rejected_and_seven_allowed(self):
        for length in range(2, 8):
            with self.subTest(length=length):
                self.assertEqual(
                    bool(MS._has_forbidden_directed_cycle(directed_ring(length), 6)),
                    length <= 6,
                )

    def test_whole_graph_checker_matches_independent_simple_path_oracle(self):
        rng = np.random.default_rng(8832)
        for size in range(9):
            for repeat in range(10):
                matrix = (rng.random((size, size)) < 0.22).astype(np.uint8)
                np.fill_diagonal(matrix, 0)
                for maximum in (2, 3, 6):
                    with self.subTest(size=size, repeat=repeat, maximum=maximum):
                        self.assertEqual(
                            bool(MS._has_forbidden_directed_cycle(matrix, maximum)),
                            has_short_cycle(matrix, maximum),
                        )

    def test_new_species_checker_respects_active_slots_and_cycle_length(self):
        for cycle_length in range(2, 8):
            old_count = cycle_length - 1
            new_slot = old_count + 3
            capacity = old_count + 5
            active = np.arange(old_count, dtype=np.int64)[::-1].copy()
            matrix = np.zeros((capacity, capacity), dtype=np.uint8)
            for position in range(old_count - 1):
                matrix[active[position], active[position + 1]] = 1
            matrix[new_slot, active[0]] = 1
            matrix[active[-1], new_slot] = 1
            # Stale extinct-slot links form a shorter cycle but must be ignored.
            dead_slot = old_count + 1
            matrix[new_slot, dead_slot] = matrix[dead_slot, new_slot] = 1
            with self.subTest(cycle_length=cycle_length):
                self.assertEqual(
                    bool(MS._new_species_closes_forbidden_cycle(
                        matrix, active, old_count, new_slot, 6,
                    )),
                    cycle_length <= 6,
                )

    def test_candidate_checker_matches_oracle_with_shuffled_and_dead_slots(self):
        rng = np.random.default_rng(7943)
        for old_count in range(1, 9):
            for repeat in range(12):
                capacity = old_count + 5
                new_slot = capacity - 1
                active = rng.choice(capacity - 1, old_count, replace=False).astype(np.int64)
                old = np.triu(rng.random((old_count, old_count)) < 0.3, k=1)
                compact = np.zeros((old_count + 1, old_count + 1), dtype=np.uint8)
                compact[:-1, :-1] = old
                compact[-1, :-1] = rng.random(old_count) < 0.45
                compact[:-1, -1] = rng.random(old_count) < 0.45
                # Deliberately dirty all storage outside the live subgraph.
                stored = np.ones((capacity, capacity), dtype=np.uint8)
                slots = np.r_[active, new_slot]
                stored[np.ix_(slots, slots)] = compact
                with self.subTest(old_count=old_count, repeat=repeat):
                    self.assertEqual(
                        bool(MS._new_species_closes_forbidden_cycle(
                            stored, active, old_count, new_slot, 6,
                        )),
                        has_short_cycle(compact),
                    )

    def test_direction_matters_and_no_self_interaction_is_counted(self):
        matrix = np.triu(np.ones((7, 7), dtype=np.uint8))
        self.assertFalse(MS._has_forbidden_directed_cycle(matrix, 6))
        self.assertFalse(MS._has_forbidden_directed_cycle(directed_ring(3), 2))


class CycleProposalTests(unittest.TestCase):
    def test_full_rejection_proposal_matches_independent_rng_reference(self):
        matrix = raw_state()["Gamma"]
        original = matrix.copy()
        rejections = 0
        for seed in range(10):
            with self.subTest(seed=seed):
                expected, rng_state, attempts = rejection_reference(
                    matrix, 0.55, seed, forced_index=0,
                )
                rng = np.random.default_rng(seed)
                actual, species, newest = MS.update_Gamma(
                    matrix, 0.55, [12, 5, 21], 21,
                    invaded_species=12, rng=rng,
                    max_forbidden_cycle_length=6,
                )
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(rng.bit_generator.state, rng_state)
                self.assertEqual(species, [12, 5, 21, 22])
                self.assertEqual(newest, 22)
                self.assertEqual(actual[-1, 0], 1)
                np.testing.assert_array_equal(matrix, original)
                rejections += attempts - 1
        self.assertGreater(rejections, 0)

    def test_initial_matrix_rejection_uses_complete_original_bernoulli_draws(self):
        for seed in range(5):
            expected_rng = np.random.default_rng(seed)
            for _ in range(10000):
                expected = (expected_rng.random((4, 4)) < 0.5).astype(np.uint8)
                np.fill_diagonal(expected, 0)
                if not has_short_cycle(expected):
                    break
            else:
                self.fail("small initial graph did not converge")
            actual_rng = np.random.default_rng(seed)
            actual = MS.update_Gamma(
                None, 0.5, [1, 2, 3, 4], 4, rng=actual_rng,
                max_forbidden_cycle_length=6,
            )
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual_rng.bit_generator.state, expected_rng.bit_generator.state)

    def test_impossible_gamma_one_fails_and_rejection_limit_is_bounded(self):
        with self.assertRaises((ValueError, RuntimeError)):
            MS.update_Gamma(
                np.zeros((1, 1), dtype=np.uint8), 1.0, [1], 1,
                invaded_species=1, rng=np.random.default_rng(1),
                max_forbidden_cycle_length=6,
            )
        # Seed 0 produces an incoming edge to the forced prey on the first try.
        with self.assertRaisesRegex((ValueError, RuntimeError), "cycle|rejection|attempt"):
            MS.update_Gamma(
                np.zeros((1, 1), dtype=np.uint8), 0.99, [1], 1,
                invaded_species=1, rng=np.random.default_rng(0),
                max_forbidden_cycle_length=6, max_cycle_rejection_attempts=1,
            )


class CycleSimulationTests(unittest.TestCase):
    def assert_same_simulation(self, actual, expected):
        for key in ("lattice", "Gamma", "tracked_timesteps", "diversity_history"):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        for key in ("current_species", "newest_species", "timestep", "rng_state"):
            self.assertEqual(actual[key], expected[key], key)

    def test_compiled_events_match_independent_sequential_rejection_reference(self):
        rejected = 0
        for seed in range(12):
            with self.subTest(seed=seed):
                state = raw_state()
                state["rng_state"] = np.random.default_rng(seed).bit_generator.state
                expected, attempts_rejected, introduction_count = microscopic_reference(
                    state, gamma=0.35, alpha=10,
                )
                actual = MS.simulation_from_state(
                    state, gamma=0.35, alpha=10, T=1,
                    max_forbidden_cycle_length=6,
                )
                self.assert_same_simulation(actual, expected)
                self.assertEqual(actual["newest_species"], 3 + introduction_count)
                rejected += attempts_rejected
        self.assertGreater(rejected, 0)

    def test_constrained_percolation_is_reproducible_and_every_checkpoint_obeys_rule(self):
        options = dict(
            L_col=7, L_row=6, D=4, gamma=0.3, alpha=4.0, T=18,
            p=0.25, seed=783, track_every=1, populate_first_100=True,
            max_forbidden_cycle_length=6,
        )
        direct = MS.percolation_simulation(**options)
        with tempfile.TemporaryDirectory() as temporary:
            chunked = MS.percolation_simulation(
                **options, checkpoint_dir=temporary,
                retention=MS.RetentionPolicy(keep_all=True),
            )
            self.assert_same_simulation(chunked, direct)
            self.assertEqual(chunked["max_forbidden_cycle_length"], 6)
            self.assertTrue(chunked["largest_cluster_only"])
            self.assertGreater(chunked["newest_species"], options["D"])
            paths = sorted(Path(temporary).glob("checkpoint_*.npz"))
            self.assertEqual(len(paths), options["T"])
            for path in paths:
                saved = MS.load_checkpoint(path)
                self.assertFalse(has_short_cycle(saved["Gamma"]), str(path))
                self.assertEqual(saved["max_forbidden_cycle_length"], 6)

    def test_resume_inherits_saved_cycle_rule_and_preserves_exact_rng(self):
        state = raw_state()
        options = dict(gamma=0.3, alpha=7.0, track_every=3)
        direct = MS.simulation_from_state(
            state, **options, T=18, max_forbidden_cycle_length=6,
            max_cycle_rejection_attempts=123,
        )
        with tempfile.TemporaryDirectory() as temporary:
            first = MS.simulation_from_state(
                state, **options, T=18, max_forbidden_cycle_length=6,
                max_cycle_rejection_attempts=123,
                checkpoint_dir=temporary,
                retention=MS.RetentionPolicy(keep_all=True),
            )
            saved = MS.load_checkpoint(first["checkpoint_files"][2])
            self.assertEqual(saved["max_forbidden_cycle_length"], 6)
            self.assertEqual(saved["max_cycle_rejection_attempts"], 123)
            for checkpoint in Path(temporary).glob("checkpoint_*.npz"):
                if MS.load_checkpoint(checkpoint)["timestep"] > saved["timestep"]:
                    checkpoint.unlink()
            resumed = MS.simulation_from_state(
                saved, **options, T=9, checkpoint_dir=temporary,
                continue_history=True, retention=MS.RetentionPolicy(keep_all=True),
            )
            self.assert_same_simulation(resumed, direct)
            self.assertEqual(resumed["max_forbidden_cycle_length"], 6)
            self.assertEqual(resumed["max_cycle_rejection_attempts"], 123)
            self.assertEqual(resumed["target_timestep"], 123)
            self.assertFalse(has_short_cycle(resumed["Gamma"]))

    def test_zero_gamma_has_no_rejections_or_rng_changes(self):
        options = dict(
            L_col=6, L_row=5, D=3, gamma=0, alpha=0.1, T=15,
            p=0.1, seed=66, track_every=3, populate_first_100=True,
        )
        ordinary = MS.percolation_simulation(**options)
        constrained = MS.percolation_simulation(**options, max_forbidden_cycle_length=6)
        self.assert_same_simulation(constrained, ordinary)
        self.assertFalse(has_short_cycle(constrained["Gamma"]))

    def test_existing_forbidden_cycles_are_rejected_without_mutating_source(self):
        state = raw_state()
        state["Gamma"] = directed_ring(3)
        original = deepcopy(state)
        with self.assertRaisesRegex(ValueError, "cycle"):
            MS.simulation_from_state(
                state, gamma=0.3, alpha=1, T=0, max_forbidden_cycle_length=6,
            )
        np.testing.assert_array_equal(state["lattice"], original["lattice"])
        np.testing.assert_array_equal(state["Gamma"], original["Gamma"])
        self.assertEqual(state["rng_state"], original["rng_state"])

    def test_invalid_cycle_options_are_rejected_before_running(self):
        options = dict(L_col=3, L_row=3, D=1, gamma=0.2, alpha=1, T=0, p=0.1)
        for maximum in (True, False, 1.5, -1, 0, 1, "6"):
            with self.subTest(maximum=maximum):
                with self.assertRaises((TypeError, ValueError)):
                    MS.percolation_simulation(**options, max_forbidden_cycle_length=maximum)
        for attempts in (True, 0, -1, 1.5):
            with self.subTest(attempts=attempts):
                with self.assertRaises((TypeError, ValueError)):
                    MS.percolation_simulation(
                        **options, max_forbidden_cycle_length=6,
                        max_cycle_rejection_attempts=attempts,
                    )

    def test_spawned_batch_and_resume_preserve_rule_seeds_and_completed_jobs(self):
        config = MS.SimulationConfig(
            L_col=5, L_row=4, D=2, gamma=0.3, alpha=4,
            T=6, track_every=1, p=0.15, seed=311,
            max_forbidden_cycle_length=6,
            retention=MS.RetentionPolicy(keep_all=True),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            expected = MS.run_simulations(
                config, kind="percolation", repeats=2,
                max_workers=2, checkpoint_root=root,
            )
            first_directory = Path(expected[0]["batch_metadata"]["checkpoint_dir"])
            # Emulate a crash with a recoverable earlier atomic checkpoint.
            for checkpoint in first_directory.glob("checkpoint_*.npz"):
                if MS.load_checkpoint(checkpoint)["timestep"] > 2:
                    checkpoint.unlink()
            completed_directory = Path(expected[1]["batch_metadata"]["checkpoint_dir"])
            before = {path.name: path.read_bytes() for path in completed_directory.iterdir() if path.is_file()}
            actual = MS.resume_simulations(root, max_workers=2)
            for index, result in enumerate(actual):
                self.assert_same_simulation(result, expected[index])
                self.assertEqual(result["max_forbidden_cycle_length"], 6)
                self.assertFalse(has_short_cycle(result["Gamma"]))
                self.assertEqual(
                    result["batch_metadata"]["seed"], expected[index]["batch_metadata"]["seed"],
                )
            after = {path.name: path.read_bytes() for path in completed_directory.iterdir() if path.is_file()}
            self.assertEqual(after, before)
            self.assertNotEqual(expected[0]["batch_metadata"]["seed"], expected[1]["batch_metadata"]["seed"])


if __name__ == "__main__":
    unittest.main()
