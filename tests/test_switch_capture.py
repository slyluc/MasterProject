"""Check the switch capture's stop rule, and that its segments chain into one exact run."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import sim_modules as MS
from tools import run_switch_capture as capture


def step_series(levels, start=150_000_000, every=1_000):
    """A living-species series holding each (value, steps) level in turn."""
    diversity = np.concatenate([np.full(steps // every, value) for value, steps in levels])
    return start + every * np.arange(diversity.size), diversity


class StopRuleTests(unittest.TestCase):
    plan = {"regime_rule": capture.REGIME_RULE, "after_down": 2_000_000}

    def test_rise_then_collapse_sets_the_stop(self):
        times, diversity = step_series([(5, 2_000_000), (100, 5_000_000), (5, 3_000_000)])
        status = capture.capture_status(times, diversity, self.plan)
        self.assertTrue(status["rise"]["settled"] and status["collapse"]["settled"])
        # Smoothing over 0.2M steps puts 50 just before the jump to 100, and 10 about 1.6
        # widths after the drop to 5.
        self.assertTrue(151_900_000 < status["rise"]["timestep"] <= 152_000_000)
        self.assertTrue(157_200_000 < status["collapse"]["timestep"] < 157_500_000)
        self.assertEqual(status["stop_at"], status["collapse"]["timestep"] + 2_000_000)
        self.assertEqual(status["regime"], "low")

    def test_a_collapse_near_the_end_is_not_settled(self):
        times, diversity = step_series([(5, 2_000_000), (100, 5_000_000), (5, 700_000)])
        status = capture.capture_status(times, diversity, self.plan)
        self.assertTrue(status["rise"]["settled"])
        self.assertFalse(status["collapse"]["settled"])
        self.assertIsNone(status["stop_at"])

    def test_a_collapse_before_the_rise_does_not_count(self):
        times, diversity = step_series([(100, 3_000_000), (5, 3_000_000), (100, 3_000_000)])
        status = capture.capture_status(times, diversity, self.plan)
        self.assertTrue(status["rise"]["timestep"] > 155_000_000)
        self.assertIsNone(status["collapse"])
        self.assertIsNone(status["stop_at"])

    def test_lingering_between_thresholds_is_not_a_switch(self):
        times, diversity = step_series([(5, 2_000_000), (40, 5_000_000), (12, 3_000_000)])
        status = capture.capture_status(times, diversity, self.plan)
        self.assertIsNone(status["rise"])


class CaptureChainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output_root = self.root / "capture"
        source = MS.SimulationConfig(
            L_col=16, L_row=16, D=2, gamma=0.3, alpha=2.0, T=0, track_every=1, seed=11, p=0.1,
            max_forbidden_cycle_length=6,
        ).run_percolation()
        self.source = Path(MS._write_checkpoint(self.root / "source", source))
        # A rule this run cannot reach, so only the step limit stops it.
        self.rule = patch.object(capture, "REGIME_RULE",
                                 {"low": 0, "high": 10**6, "sample_every": 10, "smooth_sigma": 1})
        self.rule.start()
        self.addCleanup(self.rule.stop)

    def run_cli(self, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return capture.main(arguments)

    def fresh(self, max_steps):
        return self.run_cli([
            "--source", str(self.source), "--output-root", str(self.output_root),
            "--track-every", "5", "--segment", "20", "--after-down", "0",
            "--max-steps", str(max_steps), "--min-free-gb", "0",
        ])

    def direct(self, steps):
        """The same steps as one run straight from the source, tracked differently."""
        config = MS.SimulationConfig(
            L_col=16, L_row=16, D=1, gamma=0.3, alpha=2.0, T=steps, track_every=10, p=0.1,
            max_forbidden_cycle_length=6,
        )
        return config.run_percolation(initial_state=MS.load_checkpoint(self.source))

    def assert_same_state(self, state, steps):
        expected = self.direct(steps)
        self.assertEqual(state["timestep"], expected["timestep"])
        np.testing.assert_array_equal(state["lattice"], expected["lattice"])
        np.testing.assert_array_equal(state["Gamma"], expected["Gamma"])
        self.assertEqual(state["current_species"], expected["current_species"])
        self.assertEqual(state["rng_state"], expected["rng_state"])

    def newest_state(self):
        return MS.load_checkpoint(capture.segment_dirs(self.output_root)[-1])

    def test_segments_chain_into_the_uninterrupted_run(self):
        self.assertEqual(self.fresh(50), 2)
        segments = capture.segment_dirs(self.output_root)
        self.assertEqual([path.name for path in segments],
                         ["segment_0000_t000000000000", "segment_0001_t000000000020",
                          "segment_0002_t000000000040"])
        self.assertEqual([len(list(path.glob("checkpoint_*.npz"))) for path in segments],
                         [4, 4, 2])
        self.assert_same_state(self.newest_state(), 50)
        with np.load(self.output_root / capture.SERIES) as series:
            np.testing.assert_array_equal(series["timesteps"], np.arange(0, 55, 5))
            self.assertTrue(np.all(series["patches"] >= 0))
        progress = json.loads((self.output_root / capture.PROGRESS).read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "stopped: reached --max-steps")
        self.assertEqual(progress["kept_states"], 10)

    def test_resume_finishes_an_interrupted_segment_then_extends(self):
        self.fresh(50)
        # An interruption before the last state was written leaves the segment short.
        last = capture.segment_dirs(self.output_root)[-1]
        max(last.glob("checkpoint_*.npz"), key=MS._checkpoint_timestep).unlink()
        self.assertEqual(self.run_cli(["--resume", str(self.output_root), "--max-steps", "70"]), 2)
        self.assertEqual(len(capture.segment_dirs(self.output_root)), 4)
        self.assert_same_state(self.newest_state(), 70)
        plan = json.loads((self.output_root / capture.PLAN).read_text(encoding="utf-8"))
        self.assertEqual(plan["max_steps"], 70)

    def test_stop_time_trims_the_last_segment(self):
        real_status = capture.capture_status

        def stop_at_30(timesteps, diversity, plan):
            return {**real_status(timesteps, diversity, plan), "stop_at": 30}

        with patch.object(capture, "capture_status", stop_at_30):
            self.assertEqual(self.fresh(1_000), 0)
        self.assertEqual(self.newest_state()["timestep"], 30)
        self.assert_same_state(self.newest_state(), 30)

    def test_resume_rejects_trajectory_options(self):
        self.fresh(20)
        with self.assertRaises(SystemExit):
            self.run_cli(["--resume", str(self.output_root), "--track-every", "10"])


if __name__ == "__main__":
    unittest.main()
