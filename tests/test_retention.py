"""Tests for snapshot retention, analysis records, and swap detection."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sim_modules as MS  # noqa: E402


def collapse_series(collapse_at=5000, recover_at=15000, end=20000, step=100):
    """Build a series that drops from 40 species to 1 and recovers."""
    timesteps = np.arange(0, end + 1, step, dtype=np.int64)
    diversity = np.where(
        (timesteps >= collapse_at) & (timesteps < recover_at), 1, 40
    ).astype(np.int64)
    return timesteps, diversity


class SwapDetectionTests(unittest.TestCase):
    def test_detects_a_collapse_and_a_recovery(self):
        timesteps, diversity = collapse_series()
        events = MS.detect_diversity_swaps(timesteps, diversity)
        self.assertEqual(events.size, 2)
        self.assertLess(abs(int(events[0]) - 5000), 1000)
        self.assertLess(abs(int(events[1]) - 15000), 1000)

    def test_a_flat_series_has_no_swaps(self):
        timesteps = np.arange(0, 5001, 100, dtype=np.int64)
        diversity = np.full(timesteps.size, 30, dtype=np.int64)
        self.assertEqual(
            MS.detect_diversity_swaps(timesteps, diversity).size, 0
        )

    def test_noise_around_one_level_does_not_report_swaps(self):
        timesteps = np.arange(0, 20001, 100, dtype=np.int64)
        generator = np.random.default_rng(3)
        diversity = generator.integers(28, 33, timesteps.size)
        self.assertEqual(
            MS.detect_diversity_swaps(timesteps, diversity).size, 0
        )

    def test_a_barely_moving_series_is_below_the_minimum_swing(self):
        timesteps = np.arange(0, 5001, 100, dtype=np.int64)
        diversity = np.where(timesteps > 2500, 2, 3).astype(np.int64)
        self.assertEqual(
            MS.detect_diversity_swaps(timesteps, diversity).size, 0
        )

    def test_lengths_must_agree(self):
        with self.assertRaises(ValueError):
            MS.detect_diversity_swaps(np.arange(5), np.arange(4))

    def test_reference_resists_a_long_collapse(self):
        # A median would sink to 1 here; the high-regime quantile must not.
        _, diversity = collapse_series(collapse_at=2000, recover_at=19000)
        self.assertGreater(MS.regime_reference(diversity), 10.0)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def write_snapshots(self, policy, timesteps, diversity, patches=None):
        """Feed a whole series through retention one snapshot at a time."""
        if patches is None:
            patches = np.full(timesteps.size, 7, dtype=np.int64)
        retention = MS._CheckpointRetention(self.directory, policy)
        for index, timestep in enumerate(timesteps):
            path = self.directory / f"checkpoint_{int(timestep):012d}.npz"
            path.write_bytes(b"snapshot")
            retention.record(
                path,
                int(timestep),
                timesteps[: index + 1],
                diversity[: index + 1],
                patches[: index + 1],
            )
        retention.finish(timesteps, diversity, patches)
        survivors = sorted(
            MS._checkpoint_timestep(path)
            for path in self.directory.glob("checkpoint_*.npz")
        )
        return retention, survivors

    def test_keeps_the_full_window_around_each_swap(self):
        policy = MS.RetentionPolicy(
            event_window=1000, smooth_samples=3, detect_every=1
        )
        timesteps, diversity = collapse_series()
        retention, survivors = self.write_snapshots(
            policy, timesteps, diversity
        )
        self.assertEqual(retention.events.size, 2)
        newest = int(timesteps[-1])
        for event in retention.events:
            expected = [
                timestep
                for timestep in timesteps
                if abs(int(timestep) - int(event)) <= policy.event_window
                and int(timestep) <= newest
            ]
            self.assertTrue(
                set(expected).issubset(survivors),
                f"snapshots missing around the swap at {int(event)}",
            )

    def test_keeps_the_newest_snapshot(self):
        policy = MS.RetentionPolicy(event_window=500, detect_every=1)
        timesteps, diversity = collapse_series()
        _, survivors = self.write_snapshots(policy, timesteps, diversity)
        self.assertIn(int(timesteps[-1]), survivors)

    def test_discards_most_of_an_uneventful_run(self):
        policy = MS.RetentionPolicy(event_window=500, detect_every=1)
        timesteps = np.arange(0, 20001, 100, dtype=np.int64)
        diversity = np.full(timesteps.size, 30, dtype=np.int64)
        _, survivors = self.write_snapshots(policy, timesteps, diversity)
        self.assertLess(len(survivors), 10)

    def test_keep_all_keeps_everything(self):
        policy = MS.RetentionPolicy(keep_all=True)
        timesteps, diversity = collapse_series()
        _, survivors = self.write_snapshots(policy, timesteps, diversity)
        self.assertEqual(len(survivors), timesteps.size)

    def test_never_deletes_an_uncounted_snapshot(self):
        # This is what makes the policy safe on directories written before
        # analysis records existed: no counts means no deletions.
        policy = MS.RetentionPolicy(event_window=500, detect_every=1)
        timesteps = np.arange(0, 20001, 100, dtype=np.int64)
        diversity = np.full(timesteps.size, 30, dtype=np.int64)
        patches = np.full(timesteps.size, MS._MISSING_PATCH_COUNT, np.int64)
        _, survivors = self.write_snapshots(
            policy, timesteps, diversity, patches
        )
        self.assertEqual(len(survivors), timesteps.size)


class AnalysisRecordTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_missing_record_reads_as_none(self):
        self.assertIsNone(MS.load_analysis(self.directory))

    def test_round_trips_every_field(self):
        record = {
            "tracked_timesteps": np.array([0, 10, 20], dtype=np.int64),
            "diversity_history": np.array([1, 5, 9], dtype=np.int64),
            "patch_history": np.array([1, 4, -1], dtype=np.int64),
            "event_timesteps": np.array([10], dtype=np.int64),
            "timestep": 20,
            "target_timestep": 40,
            "gamma": 0.07,
            "alpha": 0.0125,
            "track_every": 10,
            "populate_first_100": True,
            "lattice_shape": (4, 5),
        }
        MS._write_analysis(self.directory, record)
        loaded = MS.load_analysis(self.directory)
        np.testing.assert_array_equal(
            loaded["patch_history"], record["patch_history"]
        )
        np.testing.assert_array_equal(
            loaded["event_timesteps"], record["event_timesteps"]
        )
        self.assertEqual(loaded["timestep"], 20)
        self.assertEqual(loaded["target_timestep"], 40)
        self.assertEqual(loaded["track_every"], 10)
        self.assertAlmostEqual(loaded["gamma"], 0.07)
        self.assertTrue(loaded["populate_first_100"])

    def test_found_from_a_checkpoint_file_beside_it(self):
        MS._write_analysis(
            self.directory,
            {
                "tracked_timesteps": np.array([0], dtype=np.int64),
                "diversity_history": np.array([1], dtype=np.int64),
                "patch_history": np.array([1], dtype=np.int64),
                "event_timesteps": np.empty(0, dtype=np.int64),
                "timestep": 0,
                "target_timestep": 0,
                "gamma": 0.1,
                "alpha": 1.0,
                "track_every": 1,
                "populate_first_100": False,
                "lattice_shape": (2, 2),
            },
        )
        sibling = self.directory / "checkpoint_000000000000.npz"
        sibling.write_bytes(b"x")
        self.assertIsNotNone(MS.load_analysis(sibling))


class RunIntegrationTests(unittest.TestCase):
    """A real run must prune itself and stay readable and resumable."""

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp()) / "run"
        cls.config = MS.SimulationConfig(
            L_col=20,
            L_row=20,
            D=1,
            gamma=0.5,
            alpha=2.0,
            T=4000,
            track_every=50,
            seed=7,
            populate_first_100=True,
            checkpoint_dir=str(cls.directory),
            retention=MS.RetentionPolicy(event_window=200, smooth_samples=3),
        )
        cls.result = cls.config.run_main()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory.parent, ignore_errors=True)

    def test_prunes_snapshots_it_no_longer_needs(self):
        on_disk = list(self.directory.glob("checkpoint_*.npz"))
        self.assertLess(len(on_disk), self.result["tracked_timesteps"].size)
        self.assertTrue(self.result["removed_checkpoint_files"])

    def test_counts_every_patch_it_writes(self):
        patches = self.result["patch_history"]
        self.assertEqual(patches.size, self.result["tracked_timesteps"].size)
        self.assertFalse(np.any(patches < 0))

    def test_reported_checkpoint_files_all_still_exist(self):
        for path in self.result["checkpoint_files"]:
            self.assertTrue(Path(path).is_file(), path)

    def test_writes_an_analysis_record(self):
        record = MS.load_analysis(self.directory)
        self.assertIsNotNone(record)
        np.testing.assert_array_equal(
            record["patch_history"], self.result["patch_history"]
        )
        np.testing.assert_array_equal(
            record["diversity_history"], self.result["diversity_history"]
        )

    def test_patchiness_comes_from_the_record_not_the_lattices(self):
        state = MS.load_checkpoint(self.directory)
        timesteps, patches = MS._load_patchiness_history(state)
        # More samples than there are surviving lattices to count.
        self.assertGreater(
            timesteps.size,
            len(list(self.directory.glob("checkpoint_*.npz"))),
        )
        np.testing.assert_array_equal(
            patches, self.result["patch_history"][: patches.size]
        )

    def test_resumes_from_the_pruned_directory(self):
        state = MS.load_checkpoint(self.directory)
        self.assertEqual(state["timestep"], self.result["timestep"])
        self.assertEqual(
            state["patch_history"].size, state["tracked_timesteps"].size
        )


if __name__ == "__main__":
    unittest.main()
