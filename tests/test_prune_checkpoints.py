"""Check the pruning tool: which snapshots it keeps, that a pruned snapshot replays exactly,
and that a thinned copy and its zip hold what they should."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

import sim_modules as MS
from tools import prune_checkpoints as prune


def run_quietly(arguments):
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(io.StringIO()):
        code = prune.main(arguments)
    return code, output.getvalue()


def timesteps_in(folder):
    return sorted(MS._checkpoint_timestep(path) for path in Path(folder).glob("checkpoint_*.npz"))


class KeepMaskTests(unittest.TestCase):
    def test_first_of_each_block_and_the_newest(self):
        times = [0, 10, 990_000, 1_000_000, 1_010_000, 2_500_000, 2_510_000, 3_200_000]
        kept = np.asarray(times)[prune.keep_mask(times, 1_000_000)]
        self.assertEqual(kept.tolist(), [0, 1_000_000, 2_500_000, 3_200_000])

    def test_newest_kept_inside_a_block(self):
        self.assertEqual(prune.keep_mask([5, 10, 15], 20).tolist(), [True, False, True])

    def test_empty_and_single(self):
        self.assertEqual(prune.keep_mask([], 20).tolist(), [])
        self.assertEqual(prune.keep_mask([7], 20).tolist(), [True])


class PruneRunTests(unittest.TestCase):
    """A real run with a snapshot every 5 steps, pruned to one per 20."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = self.root / "checkpoints"
        self.run = self.sources / "run"
        MS.SimulationConfig(
            L_col=16, L_row=16, D=3, gamma=0.3, alpha=2.0, T=60, track_every=5, seed=5,
            p=0.1, max_forbidden_cycle_length=6, checkpoint_dir=str(self.run),
            retention=MS.RetentionPolicy(keep_all=True),
        ).run_percolation()
        (self.run / "notes.json").write_text("{}", encoding="utf-8")
        self.saved = {MS._checkpoint_timestep(path): MS.load_checkpoint(path)
                      for path in self.run.glob("checkpoint_*.npz")}

    def assert_same(self, recreated, saved):
        for key in prune.REPLAYED_KEYS + ("patch_history",):
            if isinstance(saved[key], dict):
                self.assertEqual(recreated[key], saved[key], key)
            else:
                np.testing.assert_array_equal(np.asarray(recreated[key]),
                                              np.asarray(saved[key]), err_msg=key)

    def test_report_only_changes_nothing(self):
        before = timesteps_in(self.run)
        code, output = run_quietly([str(self.sources / "*"), "--every", "20", "--workers", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(timesteps_in(self.run), before)
        self.assertIn("replay 1/1 exact", output)

    def test_apply_keeps_one_per_block_and_every_pruned_snapshot_replays(self):
        code, _ = run_quietly([str(self.run), "--every", "20", "--apply", "--workers", "1",
                               "--verify", "3"])
        self.assertEqual(code, 0)
        newest = max(self.saved)
        expected = sorted({min(t for t in self.saved if t // 20 == block)
                           for block in {t // 20 for t in self.saved}} | {newest})
        self.assertEqual(timesteps_in(self.run), expected)
        self.assertTrue((self.run / "notes.json").is_file())
        record = MS.load_analysis(self.run)
        for timestep, saved in self.saved.items():
            position = int(np.searchsorted(record["tracked_timesteps"], timestep))
            self.assertGreaterEqual(int(record["patch_history"][position]), 0)
            recreated = MS.recreate_checkpoint(self.run / f"checkpoint_{timestep:012d}.npz")
            self.assert_same(recreated, saved)
        self.assertEqual(MS.load_checkpoint(self.run)["timestep"], newest)

    def test_a_replay_that_differs_keeps_the_folder_whole(self):
        # Kept 0 (or the first) then the next file is the first check; make that file disagree.
        first, second = sorted(self.saved)[:2]
        path = self.run / f"checkpoint_{second:012d}.npz"
        with np.load(path) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays["newest_species"] = arrays["newest_species"] + 1
        with path.open("wb") as stream:
            np.savez(stream, **arrays)
        before = timesteps_in(self.run)
        code, output = run_quietly([str(self.run), "--every", "20", "--apply", "--workers", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(timesteps_in(self.run), before)
        self.assertIn("kept whole, replay differs", output)
        self.assertIn("newest_species", output)

    def test_into_copies_thinned_and_zips(self):
        (self.run / ".job.lock").write_bytes(b"\0")
        (self.sources / ".sweep.lock").write_bytes(b"\0")  # Outside the copied folders.
        (self.run / ".sweep.lock").write_bytes(b"\0")
        destination = self.root / "export" / "checkpoints"
        code, _ = run_quietly([str(self.sources / "*"), "--every", "20", "--workers", "1",
                               "--into", str(destination), "--zip"])
        self.assertEqual(code, 0)
        self.assertEqual(len(timesteps_in(self.run)), len(self.saved))  # Source untouched.
        copied = destination / "run"
        times = sorted(self.saved)
        self.assertEqual(timesteps_in(copied),
                         np.asarray(times)[prune.keep_mask(times, 20)].tolist())
        self.assertTrue((copied / "analysis.npz").is_file())
        self.assertTrue((copied / "notes.json").is_file())
        self.assertFalse((copied / ".job.lock").exists())
        self.assertFalse((copied / ".sweep.lock").exists())
        archive = destination.with_name("checkpoints.zip")
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            self.assertIsNone(bundle.testzip())
        self.assertIn("checkpoints/run/checkpoint_000000000060.npz", names)
        self.assertIn("checkpoints/run/analysis.npz", names)
        self.assertEqual(len(names), len([p for p in destination.rglob("*") if p.is_file()]))
        # A rerun copies nothing new and still succeeds.
        code, output = run_quietly([str(self.sources / "*"), "--every", "20", "--workers", "1",
                                    "--into", str(destination)])
        self.assertEqual(code, 0)
        self.assertIn("copied 0 files", output)

    def test_into_refuses_stray_files(self):
        destination = self.root / "export"
        (destination / "stray").mkdir(parents=True)
        (destination / "stray" / "old.txt").write_text("x", encoding="utf-8")
        code, output = run_quietly([str(self.run), "--every", "20", "--workers", "1",
                                    "--into", str(destination), "--zip"])
        self.assertEqual(code, 1)
        self.assertIn("not part of this gather", output)
        self.assertFalse(destination.with_name("export.zip").exists())

    def test_into_inside_a_source_is_refused(self):
        with self.assertRaises(SystemExit):
            run_quietly([str(self.sources), "--into", str(self.sources / "lean")])

    def test_a_folder_without_a_record_gets_one(self):
        (self.run / "analysis.npz").unlink()
        code, _ = run_quietly([str(self.run), "--every", "20", "--apply", "--workers", "1"])
        self.assertEqual(code, 0)
        record = MS.load_analysis(self.run)
        times = np.asarray(sorted(self.saved))
        left_out = times[~prune.keep_mask(times, 20)]
        position = np.searchsorted(record["tracked_timesteps"], left_out)
        np.testing.assert_array_equal(record["tracked_timesteps"][position], left_out)
        self.assertTrue(np.all(record["patch_history"][position] >= 0))


    def test_missing_counts_are_counted_before_anything_is_left_out(self):
        record = MS.load_analysis(self.run)
        record["patch_history"] = np.full_like(record["patch_history"], -1)
        MS._write_analysis(self.run, record)
        code, output = run_quietly([str(self.run), "--every", "20", "--apply", "--workers", "1"])
        self.assertEqual(code, 0)
        self.assertIn("recorded 8 patch counts", output)
        record = MS.load_analysis(self.run)
        for timestep, saved in self.saved.items():
            position = int(np.searchsorted(record["tracked_timesteps"], timestep))
            if not prune.keep_mask(sorted(self.saved), 20)[sorted(self.saved).index(timestep)]:
                self.assertEqual(int(record["patch_history"][position]),
                                 MS.count_patches(saved["lattice"]))


class RecreateCheckpointTests(unittest.TestCase):
    def test_existing_snapshot_is_loaded_and_missing_one_needs_an_earlier(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name) / "run"
            MS.SimulationConfig(L_col=12, L_row=12, D=2, gamma=0.3, alpha=2.0, T=20,
                                track_every=10, seed=3, checkpoint_dir=str(run),
                                retention=MS.RetentionPolicy(keep_all=True)).run_main()
            newest = max(timesteps_in(run))
            loaded = MS.recreate_checkpoint(run / f"checkpoint_{newest:012d}.npz")
            self.assertNotIn("replayed_from", loaded)
            self.assertEqual(loaded["timestep"], newest)
            with self.assertRaises(FileNotFoundError):
                MS.recreate_checkpoint(Path(name) / "empty", 5)
            with self.assertRaises(ValueError):
                MS.recreate_checkpoint(run, newest, start=run / f"checkpoint_{newest:012d}.npz")


if __name__ == "__main__":
    unittest.main()
