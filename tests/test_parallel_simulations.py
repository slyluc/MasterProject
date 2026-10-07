import json
from importlib import reload
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import sim_modules as MS


class ParallelSimulationTests(unittest.TestCase):
    @staticmethod
    def config(**changes):
        return replace(
            MS.SimulationConfig(
                L_col=5, L_row=4, D=2, gamma=0.65, alpha=2.0,
                T=4, seed=123, track_every=2,
            ),
            **changes,
        )

    def assert_same_simulation(self, first, second):
        for key in ("lattice", "Gamma", "tracked_timesteps", "diversity_history"):
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        for key in ("current_species", "newest_species", "timestep", "rng_state"):
            self.assertEqual(first[key], second[key], key)

    def test_spawned_main_and_percolation_match_direct_seeded_runs(self):
        configs = [self.config(), self.config(alpha=1.0, p=0.2)]
        for kind in ("main", "percolation"):
            with self.subTest(kind=kind):
                parallel = MS.run_simulations(configs, kind=kind, max_workers=2)
                sequential = MS.run_simulations(configs, kind=kind, max_workers=1)
                for index, (actual, replay) in enumerate(zip(parallel, sequential)):
                    self.assert_same_simulation(actual, replay)
                    self.assertEqual(actual["batch_metadata"]["job_index"], index)
                    self.assertNotEqual(actual["batch_metadata"]["worker_pid"], os.getpid())
                    child_config = replace(configs[index], seed=actual["batch_metadata"]["seed"])
                    direct = child_config.run_main() if kind == "main" else child_config.run_percolation()
                    self.assert_same_simulation(actual, direct)

    def test_repeats_and_duplicate_configs_get_distinct_reproducible_seeds(self):
        config = self.config(T=0)
        configs = [config, config, self.config(T=0, seed=456)]
        first = MS.run_simulations(configs, repeats=3, max_workers=1)
        second = MS.run_simulations(configs, repeats=3, max_workers=1)
        seeds = [result["batch_metadata"]["seed"] for result in first]
        self.assertEqual(len(set(seeds)), 9)
        self.assertEqual(seeds, [result["batch_metadata"]["seed"] for result in second])
        self.assertEqual(
            [(result["batch_metadata"]["config_index"], result["batch_metadata"]["repeat_index"])
             for result in first],
            [(config_index, repeat_index) for config_index in range(3) for repeat_index in range(3)],
        )
        self.assertEqual(config.seed, 123)
        self.assertIsNone(config.checkpoint_dir)

    def test_notebook_configs_and_retention_survive_module_reload(self):
        config = self.config(retention=MS.RetentionPolicy(keep_all=True))
        reload(MS)
        self.assertNotIsInstance(config, MS.SimulationConfig)
        results = config.run_many(2, max_workers=2)
        for result in results:
            metadata = result["batch_metadata"]
            current_config = MS.SimulationConfig(
                **{**vars(config), "retention": MS.RetentionPolicy(keep_all=True),
                   "seed": metadata["seed"]}
            )
            self.assert_same_simulation(result, current_config.run_main())
            self.assertTrue(metadata["config"]["retention"]["keep_all"])
            self.assertNotEqual(metadata["worker_pid"], os.getpid())

    def test_base_seed_overrides_config_roots(self):
        first = MS.run_simulations([self.config(T=0), self.config(T=0, seed=456)],
                                   base_seed=789, max_workers=1)
        second = MS.run_simulations([self.config(T=0, seed=999), self.config(T=0, seed=None)],
                                    base_seed=789, max_workers=1)
        self.assertEqual([r["batch_metadata"]["seed"] for r in first],
                         [r["batch_metadata"]["seed"] for r in second])
        self.assertTrue(all(result["batch_metadata"]["seed_root"] == 789 for result in first))

    def test_entropy_seeds_can_be_replayed_and_are_unique(self):
        config = self.config(seed=None)
        results = config.run_many(3, max_workers=1)
        self.assertEqual(len({result["batch_metadata"]["seed"] for result in results}), 3)
        for result in results:
            self.assert_same_simulation(
                result, replace(config, seed=result["batch_metadata"]["seed"]).run_main()
            )

    def test_checkpoint_plan_and_job_folders_isolate_each_run(self):
        config = self.config(T=2, retention=MS.RetentionPolicy(keep_all=True))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch"
            results = MS.run_simulations([config, config], repeats=2,
                                        max_workers=2, checkpoint_root=root)
            plan = json.loads((root / "batch_plan.json").read_text(encoding="utf-8"))
            directories = [Path(result["batch_metadata"]["checkpoint_dir"]) for result in results]
            self.assertEqual(len(set(directories)), 4)
            self.assertEqual(len(plan["jobs"]), 4)
            for index, (directory, result) in enumerate(zip(directories, results)):
                self.assertEqual(directory.parent, root)
                self.assertEqual(plan["jobs"][index]["seed"], result["batch_metadata"]["seed"])
                self.assertEqual(plan["jobs"][index]["config"]["T"], 2)
                saved = MS.load_checkpoint(directory)
                self.assert_same_simulation(result, saved)
            with patch.object(MS, "_execute_simulation_job") as execute:
                with self.assertRaisesRegex(FileExistsError, "batch plan already exists"):
                    MS.run_simulations(config, checkpoint_root=root, max_workers=1)
                execute.assert_not_called()
        self.assertIsNone(config.checkpoint_dir)
        self.assertTrue(config.retention.keep_all)

    def test_config_checkpoint_folder_is_a_parent_for_independent_repetitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(T=1, checkpoint_dir=temporary)
            results = config.run_many(2, max_workers=1)
            folders = [Path(result["batch_metadata"]["checkpoint_dir"]) for result in results]
            self.assertNotEqual(folders[0], folders[1])
            self.assertTrue(all(folder.parent == Path(temporary).resolve() for folder in folders))
            self.assertEqual(config.checkpoint_dir, temporary)
            with self.assertRaisesRegex(FileExistsError, "checkpoint directory already exists"):
                config.run_many(2, max_workers=1)

    def test_nested_checkpoint_destinations_are_rejected_before_work_starts(self):
        config = self.config(T=0)
        seed = config.run_many(max_workers=1)[0]["batch_metadata"]["seed"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new"
            first_folder = root / f"job_00000_seed_{seed}"
            configs = [replace(config, checkpoint_dir=root),
                       replace(config, checkpoint_dir=first_folder / "nested")]
            with patch.object(MS, "_execute_simulation_job") as execute:
                with self.assertRaisesRegex(ValueError, "directories overlap"):
                    MS.run_simulations(configs, max_workers=1)
                execute.assert_not_called()
            self.assertFalse(root.exists())

    def test_invalid_later_job_is_rejected_before_any_job_starts(self):
        configs = [self.config(), self.config(D=0, p=0.5)]
        with patch.object(MS, "_execute_simulation_job") as execute:
            with self.assertRaisesRegex(ValueError, "job 1, seed=.*D must be positive"):
                MS.run_simulations(configs, kind="percolation", max_workers=1)
            execute.assert_not_called()

    def test_saved_state_batch_forks_rng_and_single_run_continuation_keeps_rng(self):
        config = self.config(T=2)
        state = config.run_main()
        original_rng = json.dumps(state["rng_state"], sort_keys=True)
        results = config.run_many(2, max_workers=2, initial_state=state)
        for result in results:
            metadata = result["batch_metadata"]
            independent_state = {**state, "rng_state": None}
            expected = replace(config, seed=metadata["seed"]).run_main(
                initial_state=independent_state, continue_history=False
            )
            self.assert_same_simulation(result, expected)
            self.assertTrue(metadata["forked_rng"])
            self.assertNotEqual(metadata["worker_pid"], os.getpid())
            self.assertEqual(result["tracked_timesteps"][0], state["timestep"])
        self.assertNotEqual(results[0]["rng_state"], results[1]["rng_state"])
        self.assertEqual(json.dumps(state["rng_state"], sort_keys=True), original_rng)
        self.assert_same_simulation(config.run_main(initial_state=state),
                                    replace(config, seed=999).run_main(initial_state=state))

    def test_failure_contains_job_seed_and_folder_context(self):
        with patch.object(MS.SimulationConfig, "run_main", side_effect=ValueError("example failure")):
            with self.assertRaisesRegex(RuntimeError, r"job 0 .*seed=\d+, checkpoint_dir=None.*example failure"):
                self.config().run_many(max_workers=1)

    @unittest.skipUnless(os.name == "nt", "Windows process pool worker limit")
    def test_windows_worker_limit_is_validated_before_execution(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed 61 on Windows"):
            self.config().run_many(max_workers=62)


if __name__ == "__main__":
    unittest.main()
