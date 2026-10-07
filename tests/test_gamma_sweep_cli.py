"""Exercise interrupted CLI sweeps using tiny runs in temporary folders."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import sim_modules as MS
from tools import run_gamma_sweep as sweep


class GammaSweepCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output_root = self.root / "sweep"
        source_config = MS.SimulationConfig(
            L_col=200, L_row=200, D=2, gamma=0.07, alpha=0.0125,
            T=0, track_every=1, seed=123, p=0.05,
        )
        self.source_state = source_config.run_percolation()
        self.source_checkpoint = Path(MS._write_checkpoint(
            self.root / "source", self.source_state,
        ))
        self.gamma_patch = patch.object(sweep, "GAMMAS", (0.069, 0.12))
        self.gamma_patch.start()
        self.addCleanup(self.gamma_patch.stop)

    def fresh_arguments(self, **overrides):
        settings = {
            "checkpoint": self.source_checkpoint,
            "output-root": self.output_root,
            "workers": 1,
            "timesteps": 3,
            "repeats": 2,
            "track-every": 1,
            "base-seed": 67,
            **overrides,
        }
        arguments = []
        for option, value in settings.items():
            arguments.extend((f"--{option}", str(value)))
        return arguments

    def run_cli(self, arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            result = sweep.main(arguments)
        return result, output.getvalue()

    def read_json(self, name):
        return json.loads((self.output_root / name).read_text(encoding="utf-8"))

    def assert_same_simulation(self, actual, expected):
        for name in ("lattice", "Gamma", "tracked_timesteps", "diversity_history"):
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        for name in ("current_species", "newest_species", "timestep", "rng_state"):
            self.assertEqual(actual[name], expected[name], name)

    def stop_after_first_checkpoint(self):
        with patch.object(MS, "_write_analysis", side_effect=KeyboardInterrupt):
            status, output = self.run_cli(self.fresh_arguments())
        self.assertEqual(status, 130)
        self.assertIn("--resume", output)
        return self.read_json("batch_plan.json")

    def test_fresh_dry_run_does_not_write_any_files(self):
        before = self.source_checkpoint.read_bytes()
        status, output = self.run_cli(self.fresh_arguments() + ["--dry-run"])
        self.assertEqual(status, 0)
        self.assertIn('"jobs": 4', output)
        self.assertFalse(self.output_root.exists())
        self.assertEqual(self.source_checkpoint.read_bytes(), before)

    def test_resume_only_accepts_worker_and_dry_run_overrides(self):
        for option in ("--checkpoint", "--output-root", "--timesteps",
                       "--repeats", "--track-every", "--base-seed"):
            with self.subTest(option=option), redirect_stdout(io.StringIO()):
                with patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        sweep.parse_args(["--resume", "saved_sweep", f"{option}=1"])
                self.assertEqual(error.exception.code, 2)
        args = sweep.parse_args(["--resume", "saved_sweep", "--workers", "1", "--dry-run"])
        self.assertEqual(args.workers, 1)
        self.assertTrue(args.dry_run)

    def test_resume_dry_run_preserves_the_saved_folder(self):
        self.stop_after_first_checkpoint()
        files_before = {
            path.relative_to(self.output_root): path.read_bytes()
            for path in self.output_root.rglob("*") if path.is_file()
        }
        status, output = self.run_cli([
            "--resume", str(self.output_root), "--workers", "1", "--dry-run",
        ])
        self.assertEqual(status, 0)
        self.assertIn('"status": "resume"', output)
        self.assertIn('"status": "not_started"', output)
        files_after = {
            path.relative_to(self.output_root): path.read_bytes()
            for path in self.output_root.rglob("*") if path.is_file()
        }
        self.assertEqual(files_before, files_after)

    def test_interrupted_and_unstarted_jobs_resume_exactly_without_original_source(self):
        plan = self.stop_after_first_checkpoint()
        summary = self.read_json("sweep_setup.json")
        shared_path = self.output_root / summary["initial_checkpoint"]
        self.assertTrue(shared_path.is_file())
        self.assertFalse(Path(summary["initial_checkpoint"]).is_absolute())
        shared_state = MS.load_checkpoint(shared_path)
        self.assertIsNone(shared_state["rng_state"])
        self.source_checkpoint.unlink()
        with patch.object(sweep, "write_results", wraps=sweep.write_results) as writer:
            status, _ = self.run_cli(["--resume", str(self.output_root), "--workers", "1"])
        self.assertEqual(status, 0)
        resumed_results = writer.call_args.args[1]
        self.assertEqual(len(resumed_results), 4)
        for metadata, result in zip(plan["jobs"], resumed_results):
            config = MS.SimulationConfig(**metadata["config"])
            config = replace(config, checkpoint_dir=None, seed=metadata["seed"])
            expected = config.run_percolation(initial_state=shared_state, continue_history=False)
            self.assert_same_simulation(result, expected)
            self.assertEqual(result["batch_metadata"]["seed"], metadata["seed"])
        self.assertEqual(self.read_json("batch_plan.json"), plan)
        self.assertEqual(len(self.read_json("sweep_results.json")), 4)

    def test_completed_resume_recreates_result_index_without_repeating_simulations(self):
        status, _ = self.run_cli(self.fresh_arguments())
        self.assertEqual(status, 0)
        records = self.read_json("sweep_results.json")
        checkpoints = {
            path: path.read_bytes()
            for path in self.output_root.rglob("checkpoint_*.npz")
        }
        (self.output_root / "sweep_results.json").unlink()
        with patch.object(MS, "_run_compiled_simulation") as compiled:
            status, output = self.run_cli(["--resume", str(self.output_root), "--workers", "1"])
        self.assertEqual(status, 0)
        self.assertIn('"already_complete": 4', output)
        compiled.assert_not_called()
        self.assertEqual(self.read_json("sweep_results.json"), records)
        self.assertEqual(checkpoints, {path: path.read_bytes() for path in checkpoints})

    def test_active_sweep_lock_prevents_resume(self):
        self.stop_after_first_checkpoint()
        with MS._simulation_directory_lock(self.output_root, filename=".sweep.lock"):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                self.run_cli(["--resume", str(self.output_root), "--workers", "1"])

    def test_result_index_write_is_atomic_and_cleans_temporary_file_on_failure(self):
        self.output_root.mkdir()
        destination = self.output_root / "sweep_results.json"
        sweep.write_json(destination, [{"complete": False}])
        before = destination.read_bytes()
        with patch.object(sweep.os, "replace", side_effect=OSError("interrupted replace")):
            with self.assertRaisesRegex(OSError, "interrupted replace"):
                sweep.write_json(destination, [{"complete": True}])
        self.assertEqual(destination.read_bytes(), before)
        self.assertEqual(list(self.output_root.glob(".sweep_*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
