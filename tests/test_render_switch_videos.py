"""Check the video tool's age colours and highlighting on hand-made lattices."""

import unittest

from matplotlib.colors import rgb_to_hsv
import numpy as np

import sim_modules as MS
from tools import render_switch_videos as V


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


if __name__ == "__main__":
    unittest.main()
