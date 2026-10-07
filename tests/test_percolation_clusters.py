import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import sim_modules as MS


def largest_periodic_component(usable):
    """Small independent flood-fill oracle; ties follow row-major order."""
    usable = np.asarray(usable, dtype=bool)
    rows, columns = usable.shape
    visited = np.zeros_like(usable)
    largest = []
    for row, column in np.ndindex(usable.shape):
        if not usable[row, column] or visited[row, column]:
            continue
        component = []
        pending = [(row, column)]
        visited[row, column] = True
        while pending:
            current_row, current_column = pending.pop()
            component.append((current_row, current_column))
            for neighbor in (
                ((current_row - 1) % rows, current_column),
                ((current_row + 1) % rows, current_column),
                (current_row, (current_column - 1) % columns),
                (current_row, (current_column + 1) % columns),
            ):
                if usable[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(neighbor)
        if len(component) > len(largest):
            largest = component
    selected = np.zeros_like(usable)
    for site in largest:
        selected[site] = True
    return selected


def initial_state(lattice, *, species=None, timestep=20):
    lattice = np.asarray(lattice, dtype=np.int64)
    if species is None:
        species = np.unique(lattice[lattice > 0]).tolist()
    diversity = len(species)
    return {
        "lattice": lattice,
        "Gamma": np.zeros((diversity, diversity), dtype=np.uint8),
        "current_species": species,
        "newest_species": max(50, max(species, default=0)),
        "initial_newest_species": max(species, default=0),
        "timestep": timestep,
        "target_timestep": timestep,
        "tracked_timesteps": np.array([0, timestep], dtype=np.int64),
        "diversity_history": np.array([diversity, diversity], dtype=np.int64),
        "patch_history": np.array([diversity, diversity], dtype=np.int64),
        "rng_state": np.random.default_rng(123).bit_generator.state,
        "gamma": 0.7,
        "alpha": 0.125,
        "track_every": 1,
        "populate_first_100": False,
    }


class LargestPercolationClusterTests(unittest.TestCase):
    def assert_cluster(self, lattice, expected):
        state = initial_state(lattice)
        filtered = MS.keep_largest_cluster(state)
        expected = np.asarray(expected, dtype=bool)
        np.testing.assert_array_equal(filtered["lattice"] != -1, expected)
        np.testing.assert_array_equal(filtered["lattice"][expected], state["lattice"][expected])
        self.assertEqual(filtered["original_usable_sites"], np.count_nonzero(state["lattice"] != -1))
        self.assertEqual(filtered["usable_sites"], np.count_nonzero(expected))
        self.assertEqual(filtered["removed_cluster_sites"], np.count_nonzero(state["lattice"] != -1) - np.count_nonzero(expected))
        self.assertAlmostEqual(filtered["effective_p"], 1 - np.count_nonzero(expected) / expected.size)
        self.assertTrue(filtered["largest_cluster_only"])
        return filtered

    def test_periodic_connections_wrap_both_axes(self):
        lattice = np.full((5, 5), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[0, 4] = 2
        lattice[4, 0] = 3
        lattice[2, 2:4] = 4
        expected = np.zeros_like(lattice, dtype=bool)
        expected[0, 0] = expected[0, 4] = expected[4, 0] = True
        self.assert_cluster(lattice, expected)

    def test_diagonal_contact_does_not_connect(self):
        lattice = np.full((5, 5), -1, dtype=np.int64)
        lattice[0, 0] = lattice[1, 1] = 1
        lattice[3, 2:4] = 2
        expected = np.zeros_like(lattice, dtype=bool)
        expected[3, 2:4] = True
        self.assert_cluster(lattice, expected)

    def test_empty_sites_connect_different_species(self):
        lattice = np.full((5, 5), -1, dtype=np.int64)
        lattice[0, 0] = 2
        lattice[2, 1:4] = [9, 0, 7]
        expected = np.zeros_like(lattice, dtype=bool)
        expected[2, 1:4] = True
        self.assert_cluster(lattice, expected)

    def test_equal_size_tie_keeps_first_row_major_cluster(self):
        lattice = np.full((5, 5), -1, dtype=np.int64)
        lattice[0, 1:3] = 1
        lattice[3, 2:4] = 2
        expected = np.zeros_like(lattice, dtype=bool)
        expected[0, 1:3] = True
        self.assert_cluster(lattice, expected)

    def test_random_masks_and_single_row_or_column_match_flood_fill(self):
        rng = np.random.default_rng(267)
        for shape in ((1, 1), (1, 9), (9, 1), (2, 2), (3, 5), (8, 7)):
            for repeat in range(8):
                with self.subTest(shape=shape, repeat=repeat):
                    usable = rng.random(shape) >= 0.5
                    lattice = np.where(usable, rng.integers(0, 4, size=shape), -1)
                    self.assert_cluster(lattice, largest_periodic_component(usable))

    def test_species_and_gamma_are_pruned_in_original_survivor_order(self):
        lattice = np.full((5, 5), -1, dtype=np.int64)
        lattice[0, 0] = 2
        lattice[2, 1:4] = [9, 0, 7]
        state = initial_state(lattice, species=[9, 2, 7])
        state["Gamma"] = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.uint8)
        original = copy.deepcopy(state)
        filtered = MS.keep_largest_cluster(state)
        self.assertEqual(filtered["current_species"], [9, 7])
        np.testing.assert_array_equal(filtered["Gamma"], original["Gamma"][[0, 2]][:, [0, 2]])
        self.assertEqual(filtered["newest_species"], 50)
        self.assertEqual(filtered["rng_state"], original["rng_state"])
        np.testing.assert_array_equal(filtered["tracked_timesteps"], [20])
        np.testing.assert_array_equal(filtered["diversity_history"], [2])
        self.assertEqual(filtered["patch_history"].size, 1)
        self.assertEqual(filtered["initial_newest_species"], 50)
        for key in ("lattice", "Gamma", "tracked_timesteps", "diversity_history", "patch_history"):
            np.testing.assert_array_equal(state[key], original[key], err_msg=key)
            self.assertFalse(np.shares_memory(state[key], filtered[key]), key)
        self.assertEqual(state["current_species"], original["current_species"])
        self.assertEqual(state["rng_state"], original["rng_state"])

    def test_all_blocked_and_all_empty_lattices(self):
        for lattice in (np.full((4, 5), -1), np.zeros((4, 5), dtype=np.int64)):
            with self.subTest(value=lattice[0, 0]):
                filtered = self.assert_cluster(lattice, lattice != -1)
                self.assertEqual(filtered["current_species"], [])
                self.assertEqual(filtered["Gamma"].shape, (0, 0))
                np.testing.assert_array_equal(filtered["diversity_history"], [0])

    def test_loading_a_checkpoint_never_modifies_its_file(self):
        lattice = np.full((5, 5), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[2, 1:4] = 2
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(MS._write_checkpoint(temporary, initial_state(lattice)))
            before = path.read_bytes()
            filtered = MS.keep_largest_cluster(path)
            self.assertEqual(filtered["usable_sites"], 3)
            self.assertEqual(path.read_bytes(), before)
            np.testing.assert_array_equal(MS.load_checkpoint(path)["lattice"], lattice)

    def test_filter_is_idempotent_and_keeps_original_geometry_statistics(self):
        lattice = np.full((5, 5), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[2, 1:4] = 2
        filtered = MS.keep_largest_cluster(initial_state(lattice))
        repeated = MS.keep_largest_cluster(filtered)
        for key in ("lattice", "Gamma", "tracked_timesteps", "diversity_history"):
            np.testing.assert_array_equal(repeated[key], filtered[key], err_msg=key)
        for key in ("original_usable_sites", "usable_sites", "removed_cluster_sites", "effective_p", "rng_state"):
            self.assertEqual(repeated[key], filtered[key], key)
        self.assertEqual(repeated["original_usable_sites"], 4)
        self.assertEqual(repeated["removed_cluster_sites"], 1)


class LargestClusterSimulationTests(unittest.TestCase):
    @staticmethod
    def config(**changes):
        settings = {
            "L_col": 6, "L_row": 5, "D": 2, "gamma": 0.7,
            "alpha": 0.125, "T": 0, "p": 0.5, "seed": 1,
            "track_every": 1,
        }
        settings.update(changes)
        return MS.SimulationConfig(**settings)

    def test_fresh_default_filters_mask_before_simulation(self):
        config = self.config()
        random_mask = np.random.default_rng(config.seed).random((config.L_row, config.L_col)) >= config.p
        expected = largest_periodic_component(random_mask)
        result = config.run_percolation()
        np.testing.assert_array_equal(result["lattice"] != -1, expected)
        self.assertEqual(result["original_usable_sites"], np.count_nonzero(random_mask))
        self.assertEqual(result["usable_sites"], np.count_nonzero(expected))
        self.assertEqual(result["blocked_sites"], expected.size - np.count_nonzero(expected))
        self.assertEqual(result["removed_cluster_sites"], np.count_nonzero(random_mask & ~expected))
        self.assertAlmostEqual(result["effective_p"], 1 - np.count_nonzero(expected) / expected.size)
        self.assertTrue(result["p_applied"])
        self.assertTrue(result["largest_cluster_only"])

    def test_opt_out_keeps_all_bernoulli_usable_sites(self):
        config = self.config(largest_cluster_only=False)
        expected = np.random.default_rng(config.seed).random((config.L_row, config.L_col)) >= config.p
        result = config.run_percolation()
        np.testing.assert_array_equal(result["lattice"] != -1, expected)
        self.assertEqual(result["blocked_sites"], np.count_nonzero(~expected))
        self.assertFalse(result["largest_cluster_only"])

    def test_p_zero_leaves_legacy_trajectory_and_rng_unchanged(self):
        config = self.config(T=5, p=0)
        filtered = config.run_percolation()
        config.largest_cluster_only = False
        legacy = config.run_percolation()
        for key in ("lattice", "Gamma", "tracked_timesteps", "diversity_history"):
            np.testing.assert_array_equal(filtered[key], legacy[key], err_msg=key)
        for key in ("rng_state", "current_species", "newest_species"):
            self.assertEqual(filtered[key], legacy[key], key)
        self.assertEqual(filtered["removed_cluster_sites"], 0)
        self.assertEqual(filtered["effective_p"], 0)

    def test_already_connected_percolation_leaves_legacy_rng_unchanged(self):
        config = self.config(T=5, seed=0)
        mask = np.random.default_rng(config.seed).random((5, 6)) >= config.p
        np.testing.assert_array_equal(largest_periodic_component(mask), mask)
        filtered = config.run_percolation()
        config.largest_cluster_only = False
        legacy = config.run_percolation()
        for key in ("lattice", "Gamma", "tracked_timesteps", "diversity_history"):
            np.testing.assert_array_equal(filtered[key], legacy[key], err_msg=key)
        for key in ("rng_state", "current_species", "newest_species"):
            self.assertEqual(filtered[key], legacy[key], key)

    def test_largest_cluster_option_requires_a_boolean(self):
        for value in (0, 1, None, "yes"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "largest_cluster_only"):
                    self.config(largest_cluster_only=value).run_percolation()

    def test_p_one_with_zero_diversity_is_valid(self):
        result = self.config(T=3, p=1, D=0).run_percolation()
        np.testing.assert_array_equal(result["lattice"], -np.ones((5, 6)))
        np.testing.assert_array_equal(result["diversity_history"], [0, 0, 0, 0])
        self.assertEqual(result["Gamma"].shape, (0, 0))
        self.assertEqual(result["usable_sites"], 0)
        self.assertEqual(result["effective_p"], 1)

    def test_diversity_exceeding_largest_component_fails_before_dynamics(self):
        config = self.config()
        usable = np.random.default_rng(config.seed).random((config.L_row, config.L_col)) >= config.p
        capacity = np.count_nonzero(largest_periodic_component(usable))
        self.assertLess(capacity, np.count_nonzero(usable))
        config.D = int(capacity + 1)
        with patch.object(MS, "_simulate_lattice") as simulate:
            with self.assertRaisesRegex(ValueError, "D=.*exceeds"):
                config.run_percolation()
            simulate.assert_not_called()
        # Batch validation must also reject the seeded capacity before any worker starts.
        plan = MS.run_simulations(self.config(D=1), kind="percolation", base_seed=16, max_workers=1)
        child_seed = plan[0]["batch_metadata"]["seed"]
        batch_usable = np.random.default_rng(child_seed).random((config.L_row, config.L_col)) >= config.p
        config.D = int(np.count_nonzero(largest_periodic_component(batch_usable)) + 1)
        with patch.object(MS, "_execute_simulation_job") as execute:
            with self.assertRaisesRegex(ValueError, "job 0.*D=.*exceeds"):
                MS.run_simulations(config, kind="percolation", base_seed=16, max_workers=1)
            execute.assert_not_called()

    def test_pruned_and_original_blocked_sites_stay_blocked_during_dynamics(self):
        lattice = np.full((5, 6), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[2, 1:5] = [2, 0, 3, 2]
        state = initial_state(lattice)
        expected = largest_periodic_component(lattice != -1)
        result = self.config(T=6, alpha=2.0).run_percolation(initial_state=state)
        np.testing.assert_array_equal(result["lattice"] == -1, ~expected)
        self.assertEqual(result["tracked_timesteps"][0], 20)
        self.assertEqual(result["diversity_history"][0], 2)
        self.assertEqual(result["timestep"], 26)
        self.assertFalse(result["p_applied"])
        np.testing.assert_array_equal(state["lattice"], lattice)
        filtered = MS.keep_largest_cluster(state)
        direct = MS.simulation_from_state(filtered, gamma=0.7, alpha=2.0, T=6,
                                          track_every=1, continue_history=False)
        for key in ("lattice", "Gamma", "tracked_timesteps", "diversity_history"):
            np.testing.assert_array_equal(result[key], direct[key], err_msg=key)
        self.assertEqual(result["rng_state"], direct["rng_state"])

    def test_filter_runs_only_at_initialization_and_metadata_roundtrips(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(T=6, track_every=2, checkpoint_dir=temporary,
                                 retention=MS.RetentionPolicy(keep_all=True))
            with patch.object(MS, "_largest_usable_cluster_mask", wraps=MS._largest_usable_cluster_mask) as select:
                result = config.run_percolation()
                self.assertEqual(select.call_count, 1)
            saved = MS.load_checkpoint(temporary)
            analysis = MS.load_analysis(temporary)
            for key in ("p", "p_applied", "largest_cluster_only", "original_usable_sites",
                        "usable_sites", "removed_cluster_sites", "blocked_sites", "effective_p"):
                self.assertEqual(saved[key], result[key], key)
                self.assertEqual(analysis[key], result[key], key)
            self.assertGreater(result["removed_cluster_sites"], 0)

    def test_in_place_extension_of_filtered_checkpoint_keeps_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(T=2, checkpoint_dir=temporary,
                                 retention=MS.RetentionPolicy(keep_all=True))
            first = config.run_percolation()
            config.T = 2
            extended = config.run_percolation(initial_state=temporary)
            np.testing.assert_array_equal(extended["tracked_timesteps"], [0, 1, 2, 3, 4])
            np.testing.assert_array_equal(extended["diversity_history"][:3], first["diversity_history"])
            for key in ("original_usable_sites", "usable_sites", "removed_cluster_sites", "effective_p"):
                self.assertEqual(extended[key], first[key], key)

    def test_pruning_old_checkpoint_requires_new_folder_and_history(self):
        lattice = np.full((5, 6), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[2, 1:5] = [2, 0, 3, 2]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            checkpoint = Path(MS._write_checkpoint(source, initial_state(lattice)))
            original_file = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "new checkpoint_dir"):
                self.config(T=1, checkpoint_dir=str(source)).run_percolation(initial_state=checkpoint)
            with self.assertRaisesRegex(ValueError, "new history"):
                self.config(T=1, checkpoint_dir=str(Path(temporary) / "branch")).run_percolation(
                    initial_state=checkpoint, continue_history=True,
                )
            self.assertEqual(checkpoint.read_bytes(), original_file)
            self.assertEqual(len(list(source.glob("checkpoint_*.npz"))), 1)
            self.assertFalse((source / "analysis.npz").exists())

    def test_old_checkpoint_branch_writes_only_its_new_history(self):
        lattice = np.full((5, 6), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[2, 1:5] = [2, 0, 3, 2]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(MS._write_checkpoint(Path(temporary) / "source", initial_state(lattice)))
            original_file = checkpoint.read_bytes()
            destination = Path(temporary) / "branch"
            result = self.config(T=2, checkpoint_dir=str(destination)).run_percolation(initial_state=checkpoint)
            for run in (result, MS.load_checkpoint(destination), MS.load_analysis(destination)):
                np.testing.assert_array_equal(run["tracked_timesteps"], [20, 21, 22])
                self.assertEqual(run["diversity_history"][0], 2)
                self.assertEqual(run["removed_cluster_sites"], 1)
            self.assertEqual(checkpoint.read_bytes(), original_file)
            np.testing.assert_array_equal(MS.load_checkpoint(checkpoint)["tracked_timesteps"], [0, 20])

    def test_prepared_cluster_cannot_write_to_original_source_folder(self):
        lattice = np.full((5, 6), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[2, 1:5] = [2, 0, 3, 2]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            checkpoint = Path(MS._write_checkpoint(source, initial_state(lattice)))
            original_file = checkpoint.read_bytes()
            prepared = MS.keep_largest_cluster(checkpoint)
            config = self.config(T=1, checkpoint_dir=str(source))
            for kind in ("percolation", "main", "generic"):
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(ValueError, "new checkpoint_dir"):
                        if kind == "percolation":
                            config.run_percolation(initial_state=prepared)
                        elif kind == "main":
                            config.run_main(initial_state=prepared)
                        else:
                            MS.simulation_from_state(prepared, gamma=config.gamma,
                                                     alpha=config.alpha, T=1,
                                                     checkpoint_dir=source)
            self.assertEqual(checkpoint.read_bytes(), original_file)
            self.assertEqual(len(list(source.glob("checkpoint_*.npz"))), 1)
            self.assertFalse((source / "analysis.npz").exists())

    def test_prepared_cluster_can_write_and_resume_in_new_folders(self):
        lattice = np.full((5, 6), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[2, 1:5] = [2, 0, 3, 2]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            checkpoint = MS._write_checkpoint(source, initial_state(lattice))
            prepared = MS.keep_largest_cluster(checkpoint)
            for kind in ("percolation", "main", "generic"):
                with self.subTest(kind=kind):
                    destination = Path(temporary) / kind
                    config = self.config(T=1, checkpoint_dir=str(destination))
                    if kind == "percolation":
                        result = config.run_percolation(initial_state=prepared)
                    elif kind == "main":
                        result = config.run_main(initial_state=prepared)
                    else:
                        result = MS.simulation_from_state(prepared, gamma=config.gamma,
                                                          alpha=config.alpha, T=1,
                                                          checkpoint_dir=destination)
                    np.testing.assert_array_equal(result["tracked_timesteps"], [20, 21])
                    continued = config.run_percolation(initial_state=destination)
                    np.testing.assert_array_equal(continued["tracked_timesteps"], [20, 21, 22])
                    self.assertEqual(continued["removed_cluster_sites"], 1)

    def test_prepared_connected_checkpoint_requires_new_folder_for_reset_history(self):
        lattice = np.ones((5, 6), dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            checkpoint = Path(MS._write_checkpoint(source, initial_state(lattice)))
            original_file = checkpoint.read_bytes()
            prepared = MS.keep_largest_cluster(checkpoint)
            self.assertEqual(prepared["removed_cluster_sites"], 0)
            np.testing.assert_array_equal(prepared["tracked_timesteps"], [20])
            config = self.config(T=1, checkpoint_dir=str(source))
            for kind in ("percolation", "main", "generic"):
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(ValueError, "new checkpoint_dir"):
                        if kind == "percolation":
                            config.run_percolation(initial_state=prepared)
                        elif kind == "main":
                            config.run_main(initial_state=prepared)
                        else:
                            MS.simulation_from_state(prepared, gamma=config.gamma,
                                                     alpha=config.alpha, T=1,
                                                     checkpoint_dir=source)
            self.assertEqual(checkpoint.read_bytes(), original_file)
            self.assertEqual(len(list(source.glob("checkpoint_*.npz"))), 1)
            self.assertFalse((source / "analysis.npz").exists())

    def test_in_memory_prepared_branch_preserves_original_folder_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            checkpoint = Path(MS._write_checkpoint(source, initial_state(np.ones((5, 6), dtype=np.int64))))
            original_file = checkpoint.read_bytes()
            prepared = MS.keep_largest_cluster(checkpoint)
            branch = MS.simulation_from_state(prepared, gamma=0.7, alpha=0.125, T=1)
            np.testing.assert_array_equal(branch["tracked_timesteps"], [20, 21])
            with self.assertRaisesRegex(ValueError, "new checkpoint_dir"):
                MS.simulation_from_state(branch, gamma=0.7, alpha=0.125, T=1,
                                         checkpoint_dir=source)
            destination = Path(temporary) / "branch"
            saved_branch = MS.simulation_from_state(branch, gamma=0.7, alpha=0.125,
                                                    T=1, checkpoint_dir=destination)
            np.testing.assert_array_equal(saved_branch["tracked_timesteps"], [21, 22])
            self.assertEqual(checkpoint.read_bytes(), original_file)
            self.assertFalse((source / "analysis.npz").exists())

    def test_spawned_branches_have_unique_rng_and_isolated_new_histories(self):
        lattice = np.full((5, 6), -1, dtype=np.int64)
        lattice[0, 0] = 1
        lattice[2, 1:5] = [2, 0, 3, 2]
        state = initial_state(lattice)
        filtered = MS.keep_largest_cluster(state)
        saved_rng = json.dumps(filtered["rng_state"], sort_keys=True)
        config = self.config(T=2, alpha=1.0, retention=MS.RetentionPolicy(keep_all=True))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            results = MS.run_simulations(config, kind="percolation", repeats=2,
                                        base_seed=67, max_workers=2,
                                        checkpoint_root=root, initial_state=filtered)
            self.assertEqual(len({r["batch_metadata"]["seed"] for r in results}), 2)
            self.assertNotEqual(results[0]["rng_state"], results[1]["rng_state"])
            folders = []
            for result in results:
                metadata = result["batch_metadata"]
                self.assertNotEqual(metadata["worker_pid"], os.getpid())
                self.assertTrue(metadata["forked_rng"])
                folder = Path(metadata["checkpoint_dir"])
                folders.append(folder)
                self.assertEqual(folder.parent, root)
                saved = MS.load_checkpoint(folder)
                analysis = MS.load_analysis(folder)
                for run in (result, saved, analysis):
                    np.testing.assert_array_equal(run["tracked_timesteps"], [20, 21, 22])
                    self.assertEqual(run["diversity_history"][0], 2)
                    for key in ("original_usable_sites", "usable_sites", "removed_cluster_sites", "blocked_sites", "effective_p"):
                        self.assertEqual(run[key], filtered[key], key)
                np.testing.assert_array_equal(saved["lattice"], result["lattice"])
                np.testing.assert_array_equal(saved["lattice"] == -1, filtered["lattice"] == -1)
                self.assertEqual(saved["rng_state"], result["rng_state"])
                self.assertEqual(len(list(folder.glob("checkpoint_*.npz"))), 2)
            self.assertNotEqual(folders[0], folders[1])
        self.assertEqual(json.dumps(filtered["rng_state"], sort_keys=True), saved_rng)
        np.testing.assert_array_equal(filtered["tracked_timesteps"], [20])
        np.testing.assert_array_equal(state["lattice"], lattice)


if __name__ == "__main__":
    unittest.main()
