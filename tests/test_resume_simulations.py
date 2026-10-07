"""Recovery tests use actual interrupted jobs and their recorded child seeds."""

from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

import sim_modules as MS


def _hold_job_lock_until_terminated(directory, ready_connection):
    """Importable spawned worker used to verify interruption cleanup."""
    with MS._simulation_directory_lock(directory):
        ready_connection.send(os.getpid())
        ready_connection.close()
        while True:
            time.sleep(30)


class ResumeSimulationTests(unittest.TestCase):
    @staticmethod
    def config(**changes):
        return replace(
            MS.SimulationConfig(
                L_col=5, L_row=4, D=3, gamma=0.65, alpha=5.0,
                T=8, seed=127, track_every=2,
                retention=MS.RetentionPolicy(keep_all=True),
            ),
            **changes,
        )

    def assert_same_simulation(self, actual, expected):
        for key in (
            "lattice", "Gamma", "tracked_timesteps", "diversity_history",
            "patch_history",
        ):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        for key in (
            "current_species", "newest_species", "initial_newest_species",
            "timestep", "target_timestep", "rng_state",
        ):
            self.assertEqual(actual[key], expected[key], key)

    @staticmethod
    def file_contents(directory):
        return {
            str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()
        }

    @staticmethod
    def read_plan(root):
        return json.loads((root / "batch_plan.json").read_text(encoding="utf-8"))

    @staticmethod
    def rewrite_checkpoint(path, **changes):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        arrays.update(changes)
        with path.open("wb") as saved_file:
            np.savez(saved_file, **arrays)

    def interrupt_at_analysis(
        self, root, *, initial_state=None, kind="main", repeats=3, crash_after=4,
    ):
        """First job finishes; second crashes after the second snapshot is atomic."""
        original_write = MS._write_analysis
        start = 0 if initial_state is None else int(initial_state["timestep"])

        def fail_second_job(directory, record):
            if Path(directory).name.startswith("job_00001_") and int(record["timestep"]) == start + crash_after:
                raise OSError("simulated power loss after atomic snapshot")
            return original_write(directory, record)

        with patch.object(MS, "_write_analysis", side_effect=fail_second_job):
            with self.assertRaisesRegex(RuntimeError, "simulated power loss"):
                MS.run_simulations(
                    self.config(p=0.2), kind=kind, repeats=repeats,
                    max_workers=1, checkpoint_root=root, initial_state=initial_state,
                )
        plan = self.read_plan(root)
        self.assertEqual(len(plan["jobs"]), repeats)
        self.assertEqual(MS.load_checkpoint(plan["jobs"][0]["checkpoint_dir"])["timestep"], start + 8)
        self.assertEqual(MS.load_checkpoint(plan["jobs"][1]["checkpoint_dir"])["timestep"], start + crash_after)
        if repeats > 2:
            self.assertFalse(list(Path(plan["jobs"][2]["checkpoint_dir"]).glob("checkpoint_*.npz")))
        return plan

    def expected_jobs(self, plan, temporary, initial_state=None):
        expected = []
        for job in plan["jobs"]:
            options = dict(job["config"])
            options["retention"] = MS.RetentionPolicy(**options["retention"])
            options["checkpoint_dir"] = str(Path(temporary) / f"reference_{job['job_index']}")
            config = MS.SimulationConfig(**options)
            source = None if initial_state is None else {**initial_state, "rng_state": None}
            run = config.run_main if job["kind"] == "main" else config.run_percolation
            expected.append(run(initial_state=source, continue_history=False))
        return expected

    def test_interrupted_fresh_sweep_resumes_and_replays_queued_job_with_same_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            plan = self.interrupt_at_analysis(root)
            plan_before = (root / "batch_plan.json").read_bytes()
            finished_directory = Path(plan["jobs"][0]["checkpoint_dir"])
            finished_before = self.file_contents(finished_directory)
            expected = self.expected_jobs(plan, temporary)

            # A new worker count must not affect RNG restoration or seed replay.
            actual = MS.resume_simulations(root, max_workers=2)

            self.assertEqual(len(actual), 3)
            self.assertEqual((root / "batch_plan.json").read_bytes(), plan_before)
            self.assertEqual(self.file_contents(finished_directory), finished_before)
            for index, (result, reference) in enumerate(zip(actual, expected)):
                self.assert_same_simulation(result, reference)
                self.assertEqual(result["batch_metadata"]["seed"], plan["jobs"][index]["seed"])
                self.assertEqual(result["batch_metadata"]["job_index"], index)
            self.assertTrue(actual[0]["resume_metadata"]["already_completed"])
            self.assertTrue(actual[1]["resume_metadata"]["from_checkpoint"])
            self.assertEqual(actual[1]["resume_metadata"]["remaining_timesteps"], 4)
            self.assertFalse(actual[2]["resume_metadata"]["from_checkpoint"])
            self.assertNotEqual(actual[1]["batch_metadata"]["worker_pid"], os.getpid())

    def test_interrupted_branches_keep_new_histories_and_source_rng_unchanged(self):
        for kind in ("main", "percolation"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                source = self.config(T=3, p=0.2).run_percolation()
                original_rng = json.dumps(source["rng_state"], sort_keys=True)
                original_lattice = source["lattice"].copy()
                root = Path(temporary) / "sweep"
                plan = self.interrupt_at_analysis(root, initial_state=source, kind=kind)
                expected = self.expected_jobs(plan, temporary, source)
                actual = MS.resume_simulations(root, max_workers=1, initial_state=source)
                for result, reference in zip(actual, expected):
                    self.assert_same_simulation(result, reference)
                    self.assertEqual(result["tracked_timesteps"][0], source["timestep"])
                    self.assertEqual(result["timestep"], source["timestep"] + 8)
                self.assertEqual(json.dumps(source["rng_state"], sort_keys=True), original_rng)
                np.testing.assert_array_equal(source["lattice"], original_lattice)

    def test_all_complete_sweep_is_read_only_and_needs_no_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            source = self.config(T=3).run_main()
            expected = MS.run_simulations(
                self.config(), repeats=2, max_workers=1,
                checkpoint_root=root, initial_state=source,
            )
            before = self.file_contents(root)
            with patch.object(MS, "ProcessPoolExecutor") as executor, patch.object(
                MS, "_execute_simulation_job"
            ) as execute:
                actual = MS.resume_simulations(root, max_workers=2)
                executor.assert_not_called()
                execute.assert_not_called()
            self.assertEqual(self.file_contents(root), before)
            for result, reference in zip(actual, expected):
                self.assert_same_simulation(result, reference)
                self.assertTrue(result["resume_metadata"]["already_completed"])

    def test_partial_branches_need_no_original_state_when_every_job_has_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            source = self.config(T=3).run_main()
            plan = self.interrupt_at_analysis(root, initial_state=source, repeats=2)
            expected = self.expected_jobs(plan, temporary, source)
            actual = MS.resume_simulations(root, max_workers=1)
            for result, reference in zip(actual, expected):
                self.assert_same_simulation(result, reference)
            self.assertTrue(actual[1]["resume_metadata"]["from_checkpoint"])

    def test_latest_corrupt_snapshot_rejects_whole_resume_before_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            plan = self.interrupt_at_analysis(root)
            damaged = Path(plan["jobs"][1]["checkpoint_dir"]) / "checkpoint_000000000004.npz"
            damaged.write_bytes(b"incomplete checkpoint bytes")
            before = self.file_contents(root)
            with patch.object(MS, "ProcessPoolExecutor") as executor, patch.object(
                MS, "simulation_from_state"
            ) as resume:
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    MS.resume_simulations(root, max_workers=2)
                executor.assert_not_called()
                resume.assert_not_called()
            self.assertEqual(self.file_contents(root), before)

    def test_checkpoint_target_and_settings_must_match_recorded_plan(self):
        for field, replacement in (
            ("target_timestep", np.int64(99)),
            ("gamma", np.float64(0.12)),
            ("alpha", np.float64(0.01)),
            ("track_every", np.int64(4)),
            ("populate_first_100", np.bool_(True)),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "sweep"
                plan = self.interrupt_at_analysis(root, repeats=2)
                damaged = Path(plan["jobs"][1]["checkpoint_dir"]) / "checkpoint_000000000004.npz"
                self.rewrite_checkpoint(damaged, **{field: replacement})
                before = self.file_contents(root)
                with patch.object(MS, "ProcessPoolExecutor") as executor:
                    with self.assertRaises(ValueError):
                        MS.resume_simulations(root, max_workers=2)
                    executor.assert_not_called()
                self.assertEqual(self.file_contents(root), before)

    def test_plan_seed_disagreement_rejects_resume_before_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            plan = self.interrupt_at_analysis(root)
            plan["jobs"][2]["config"]["seed"] += 1
            (root / "batch_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            before = self.file_contents(root)
            with patch.object(MS, "ProcessPoolExecutor") as executor:
                with self.assertRaisesRegex(ValueError, "seed"):
                    MS.resume_simulations(root, max_workers=2)
                executor.assert_not_called()
            self.assertEqual(self.file_contents(root), before)

    def test_wrong_fork_source_is_rejected_before_any_jobs_continue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            source = self.config(T=3).run_main()
            self.interrupt_at_analysis(root, initial_state=source)
            wrong = self.config(T=3, seed=777).run_main()
            before = self.file_contents(root)
            with patch.object(MS, "ProcessPoolExecutor") as executor:
                with self.assertRaises(ValueError):
                    MS.resume_simulations(root, max_workers=2, initial_state=wrong)
                executor.assert_not_called()
            self.assertEqual(self.file_contents(root), before)

    def test_active_job_lock_prevents_resume_before_workers_then_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            plan = self.interrupt_at_analysis(root, repeats=2)
            directory = Path(plan["jobs"][1]["checkpoint_dir"])
            before = self.file_contents(root)
            with MS._simulation_directory_lock(directory):
                with patch.object(MS, "ProcessPoolExecutor") as executor, patch.object(
                    MS, "_execute_resumed_simulation_job"
                ) as execute:
                    with self.assertRaisesRegex(RuntimeError, "already in use"):
                        MS.resume_simulations(root, max_workers=2)
                    executor.assert_not_called()
                    execute.assert_not_called()
            self.assertEqual(self.file_contents(root), before)
            # The persistent lock file is harmless after its owner exits.
            actual = MS.resume_simulations(root, max_workers=1)
            self.assertEqual(actual[1]["timestep"], 8)
            self.assertEqual(actual[1]["batch_metadata"]["seed"], plan["jobs"][1]["seed"])

    def test_atomic_snapshot_ahead_of_analysis_preserves_previous_patch_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            plan = self.interrupt_at_analysis(root, repeats=2)
            directory = Path(plan["jobs"][1]["checkpoint_dir"])
            checkpoint = MS.load_checkpoint(directory)
            analysis = MS.load_analysis(directory)
            self.assertEqual(checkpoint["timestep"], 4)
            self.assertEqual(analysis["timestep"], 2)
            np.testing.assert_array_equal(
                checkpoint["patch_history"][:2], analysis["patch_history"],
            )
            expected = self.expected_jobs(plan, temporary)
            actual = MS.resume_simulations(root, max_workers=1)
            self.assert_same_simulation(actual[1], expected[1])
            np.testing.assert_array_equal(actual[1]["tracked_timesteps"], [0, 2, 4, 6, 8])

    def test_final_snapshot_before_analysis_is_finalized_without_more_simulation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            plan = self.interrupt_at_analysis(root, repeats=2, crash_after=8)
            directory = Path(plan["jobs"][1]["checkpoint_dir"])
            completed_directory = Path(plan["jobs"][0]["checkpoint_dir"])
            self.assertEqual(MS.load_analysis(directory)["timestep"], 6)
            self.assertEqual(MS.load_checkpoint(directory)["timestep"], 8)
            expected = self.expected_jobs(plan, temporary)
            completed_before = self.file_contents(completed_directory)
            checkpoints_before = {
                path.name: path.read_bytes() for path in directory.glob("checkpoint_*.npz")
            }
            with patch.object(MS, "_run_compiled_simulation") as evolve:
                actual = MS.resume_simulations(root, max_workers=1)
                evolve.assert_not_called()
            self.assertEqual(self.file_contents(completed_directory), completed_before)
            self.assertEqual(
                {path.name: path.read_bytes() for path in directory.glob("checkpoint_*.npz")},
                checkpoints_before,
            )
            self.assert_same_simulation(actual[1], expected[1])
            analysis = MS.load_analysis(directory)
            self.assertEqual(analysis["timestep"], 8)
            np.testing.assert_array_equal(analysis["tracked_timesteps"], expected[1]["tracked_timesteps"])
            np.testing.assert_array_equal(analysis["patch_history"], expected[1]["patch_history"])

    def test_missing_analysis_is_rebuilt_from_snapshot_patch_history_without_evolving(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            expected = MS.run_simulations(
                self.config(), repeats=2, max_workers=1, checkpoint_root=root,
            )
            directory = Path(expected[0]["batch_metadata"]["checkpoint_dir"])
            other_directory = Path(expected[1]["batch_metadata"]["checkpoint_dir"])
            Path(expected[0]["analysis_path"]).unlink()
            self.assertIsNone(MS.load_analysis(directory))
            checkpoint = MS.load_checkpoint(directory)
            np.testing.assert_array_equal(
                checkpoint["patch_history"][:-1], expected[0]["patch_history"][:-1],
            )
            self.assertEqual(checkpoint["patch_history"][-1], -1)
            checkpoints_before = {
                path.name: path.read_bytes() for path in directory.glob("checkpoint_*.npz")
            }
            other_before = self.file_contents(other_directory)
            with patch.object(MS, "_run_compiled_simulation") as evolve:
                actual = MS.resume_simulations(root, max_workers=1)
                evolve.assert_not_called()
            for result, reference in zip(actual, expected):
                self.assert_same_simulation(result, reference)
            self.assertTrue(actual[0]["resume_metadata"]["finalizing"])
            self.assertEqual(self.file_contents(other_directory), other_before)
            self.assertEqual(
                {path.name: path.read_bytes() for path in directory.glob("checkpoint_*.npz")},
                checkpoints_before,
            )
            analysis = MS.load_analysis(directory)
            for key in ("tracked_timesteps", "diversity_history", "patch_history"):
                np.testing.assert_array_equal(analysis[key], expected[0][key], err_msg=key)

    def test_interruption_cleanup_terminates_spawned_workers_and_releases_job_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = multiprocessing.get_context("spawn")
            directories = [Path(temporary) / f"job_{index}" for index in range(2)]
            connections = [context.Pipe(duplex=False) for _ in directories]
            executor = ProcessPoolExecutor(
                max_workers=2, mp_context=context,
                initializer=MS._initialize_simulation_worker, initargs=(None,),
            )
            processes = []
            try:
                for directory, (_, sender) in zip(directories, connections):
                    executor.submit(_hold_job_lock_until_terminated, str(directory), sender)
                worker_pids = []
                for receiver, _ in connections:
                    self.assertTrue(receiver.poll(15), "spawned worker did not become ready")
                    worker_pids.append(receiver.recv())
                processes = list(executor._processes.values())
                self.assertEqual({process.pid for process in processes}, set(worker_pids))
                self.assertEqual(len(worker_pids), 2)
                self.assertTrue(all(process.is_alive() for process in processes))
                with self.assertRaises(KeyboardInterrupt):
                    try:
                        raise KeyboardInterrupt
                    except BaseException:
                        MS._stop_simulation_executor(executor)
                        raise
                self.assertTrue(all(not process.is_alive() for process in processes))
                self.assertTrue(all(process.exitcode is not None for process in processes))
                for directory in directories:
                    with MS._simulation_directory_lock(directory):
                        pass
            finally:
                # Also clean up if readiness or an assertion failed, so this
                # recovery test cannot leave background workers behind.
                remaining = processes or list((executor._processes or {}).values())
                if any(process.is_alive() for process in remaining):
                    MS._stop_simulation_executor(executor)
                else:
                    executor.shutdown(wait=True, cancel_futures=True)
                for receiver, sender in connections:
                    receiver.close()
                    sender.close()

    def test_legacy_branch_plan_requires_source_for_never_started_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            source = self.config(T=3).run_main()
            plan = self.interrupt_at_analysis(root, initial_state=source)
            # Version-1 plans historically recorded only source shape and time.
            plan["initial_state"] = {
                "timestep": int(source["timestep"]),
                "lattice_shape": list(source["lattice"].shape),
                "source_directory": None, "forked_rng": True,
            }
            (root / "batch_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            before = self.file_contents(root)
            with patch.object(MS, "ProcessPoolExecutor") as executor:
                with self.assertRaisesRegex(ValueError, "initial_state"):
                    MS.resume_simulations(root, max_workers=2)
                executor.assert_not_called()
            self.assertEqual(self.file_contents(root), before)
            actual = MS.resume_simulations(root, max_workers=1, initial_state=source)
            for result in actual:
                self.assertEqual(result["timestep"], int(source["timestep"]) + 8)


if __name__ == "__main__":
    unittest.main()
