"""Check the video tool's age colours and highlighting on hand-made lattices, and its replay of
a capture thinned by tools/prune_checkpoints.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.colors import rgb_to_hsv
import numpy as np

from analysis import switch_features
import sim_modules as MS
from tools import prune_checkpoints as prune
from tools import render_switch_videos as V
from tools import run_switch_capture


class AgePaletteTests(unittest.TestCase):
    def test_species_are_ordered_by_id_once_each(self):
        ids, colours, plain = V.age_palette([57632, 57538, 57627, 57538])
        self.assertEqual(ids.tolist(), [57538, 57627, 57632])
        self.assertEqual(colours.shape, (3, 3))
        self.assertEqual(plain.shape, (3, 3))

    def test_hue_runs_from_violet_to_pink_by_age(self):
        _, _, plain = V.age_palette(np.arange(1, 31))
        hues = np.degrees(np.unwrap(2 * np.pi * rgb_to_hsv(plain)[:, 0])) / 360
        self.assertTrue(np.all(np.diff(hues) < 0))  # Violet, blue, ..., red, then pink.
        self.assertAlmostEqual(hues[0] - hues[-1], V.AGE_HUES[0] - V.AGE_HUES[1], places=6)

    def test_neighbouring_ages_differ_in_lightness(self):
        _, colours, plain = V.age_palette(np.arange(1, 31))
        np.testing.assert_allclose(colours[0::3], plain[0::3] * V.DARKER)
        np.testing.assert_allclose(colours[1::3], plain[1::3])
        self.assertTrue(np.all(colours[2::3].sum(axis=1) > plain[2::3].sum(axis=1)))
        self.assertGreater(np.linalg.norm(np.diff(colours, axis=0), axis=1).min(), 0.15)


class LatticeColourTests(unittest.TestCase):
    lattice = np.array([[-1, 0, 5],
                        [7, 5, 9]])

    def test_blocked_empty_and_species_sites(self):
        palette = V.age_palette([5, 7, 9])
        rgb = V.lattice_colours(self.lattice, palette)
        np.testing.assert_allclose(rgb[0, 0], 0.0)
        np.testing.assert_allclose(rgb[0, 1], V.EMPTY_COLOR)
        for (row, column), rank in (((0, 2), 0), ((1, 0), 1), ((1, 2), 2)):
            np.testing.assert_allclose(rgb[row, column], palette[1][rank])

    def test_highlight_keeps_plain_colours_and_mutes_the_rest(self):
        ids, colours, plain = V.age_palette([5, 7, 9])
        rgb = V.lattice_colours(self.lattice, (ids, colours, plain), highlight=[5, 9])
        np.testing.assert_allclose(rgb[0, 2], plain[0])
        np.testing.assert_allclose(rgb[1, 2], plain[2])
        np.testing.assert_allclose(rgb[1, 0], V.mute(colours[1]))
        np.testing.assert_allclose(rgb[0, 0], 0.0)  # Blocked sites stay black.

    def test_id_colours_unchanged_without_options(self):
        np.testing.assert_allclose(V.lattice_colours(self.lattice),
                                   MS._lattice_rgba(self.lattice)[..., :3])
        muted = V.lattice_colours(self.lattice, highlight=[7])
        np.testing.assert_allclose(muted[1, 0], MS._lattice_rgba(self.lattice)[1, 0, :3])
        np.testing.assert_allclose(muted[0, 2], V.mute(MS._lattice_rgba(self.lattice)[0, 2, :3]))


class PrunedCaptureReplayTests(unittest.TestCase):
    """A small capture, tracked every 5 steps in segments of 20, thinned to one state per 20."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        source = MS.SimulationConfig(
            L_col=16, L_row=16, D=2, gamma=0.3, alpha=2.0, T=0, track_every=1, seed=11, p=0.1,
            max_forbidden_cycle_length=6,
        ).run_percolation()
        source_file = MS._write_checkpoint(root / "source", source)
        cls.root = root / "capture"
        rule = {"low": 0, "high": 10**6, "sample_every": 10, "smooth_sigma": 1}
        quiet = io.StringIO()
        with patch.object(run_switch_capture, "REGIME_RULE", rule), \
                redirect_stdout(quiet), redirect_stderr(quiet):
            run_switch_capture.main([
                "--source", source_file, "--output-root", str(cls.root), "--track-every", "5",
                "--segment", "20", "--after-down", "0", "--max-steps", "50",
                "--min-free-gb", "0"])
            switch_features.load_features(cls.root, workers=1)
            cls.saved = {MS._checkpoint_timestep(path): MS.load_checkpoint(path)
                         for path in switch_features.state_files(cls.root)}
            prune.main([str(cls.root), "--every", "20", "--apply", "--workers", "1"])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def capture(self):
        # This short run has no rise or collapse to find; the replay does not need them.
        points = {"rise": 0, "collapse": 0, "windows": {}, "snapshots": {}}
        with patch.object(V.switch_features, "switch_points", return_value=points):
            return V.Capture(self.root)

    def assert_saved(self, state):
        saved = self.saved[state["timestep"]]
        np.testing.assert_array_equal(state["lattice"], saved["lattice"])
        np.testing.assert_array_equal(state["Gamma"], saved["Gamma"])
        self.assertEqual(state["current_species"].tolist(), saved["current_species"])
        self.assertEqual(state["rng_state"], saved["rng_state"])

    def test_the_capture_was_thinned_but_keeps_its_features(self):
        capture = self.capture()
        self.assertLess(capture.kept.size, capture.times.size)
        self.assertEqual(capture.features["timestep"].size, capture.times.size)

    def test_every_sample_replays_to_the_saved_state(self):
        capture = self.capture()
        states = list(V.replay(capture, capture.times))
        self.assertEqual([state["timestep"] for state in states], capture.times.tolist())
        for state in states:
            self.assert_saved(state)

    def test_consecutive_steps_across_a_left_out_stretch(self):
        capture = self.capture()
        steps = np.arange(23, 47)
        states = list(V.replay(capture, steps))
        self.assertEqual([state["timestep"] for state in states], steps.tolist())
        for state in states:
            if state["timestep"] in self.saved:
                self.assert_saved(state)
        # A later replay starts from the remembered first state when no kept state is newer.
        self.assertEqual(capture.memo["timestep"], 23)
        self.assertEqual(capture.start_for(24)["timestep"], 23)
        self.assertEqual(capture.start_for(30)["timestep"], 25)

    def test_a_replay_that_disagrees_with_the_features_stops(self):
        capture = self.capture()
        capture.features = dict(capture.features)
        index = int(np.flatnonzero(~np.isin(capture.times, capture.kept))[0])
        offsets = capture.features["offsets"]
        species = capture.features["species_id"].copy()
        species[offsets[index]] += 10_000
        capture.features["species_id"] = species
        with self.assertRaises(RuntimeError):
            list(V.replay(capture, capture.times[:index + 1]))

    def test_states_at_loads_kept_and_replays_left_out_states(self):
        capture = self.capture()
        wanted = [0, 10, 15, 15, 30, 35, 45, 50]
        self.assertTrue(set(wanted) - set(capture.kept.tolist()))  # Some must be replayed.
        states = list(switch_features.states_at(self.root, wanted))
        self.assertEqual([state["timestep"] for state in states], wanted)
        for state in states:
            state = {**state, "current_species": np.asarray(state["current_species"])}
            self.assert_saved(state)
            self.assertEqual(state["diversity"], len(self.saved[state["timestep"]]["current_species"]))
        with self.assertRaises(ValueError):
            list(switch_features.states_at(self.root, [30, 10]))

    def test_features_are_not_rebuilt_from_a_thinned_capture(self):
        with self.assertRaises(FileNotFoundError):
            switch_features.load_features(self.root, workers=1, refresh=True)


if __name__ == "__main__":
    unittest.main()
