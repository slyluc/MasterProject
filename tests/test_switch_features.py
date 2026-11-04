"""Check the switch capture's per-state measures on hand-made states."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

import sim_modules as MS
from analysis import switch_features as F


class StateFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def features(self, lattice, Gamma, species):
        path = Path(self.temporary.name) / "state.npz"
        np.savez(path, lattice=np.asarray(lattice), Gamma=np.asarray(Gamma, dtype=np.uint8),
                 current_species=np.asarray(species), newest_species=np.int64(max(species)),
                 timestep=np.int64(5))
        return F.state_features(path)

    def test_a_cycle_of_seven_is_one_component(self):
        Gamma = np.zeros((8, 8), dtype=np.uint8)
        for species in range(7):
            Gamma[species, (species + 1) % 7] = 1
        Gamma[0, 7] = 1  # Species 7 is invaded, but closes no cycle.
        sizes, _ = F.strong_components(Gamma)
        self.assertEqual(sizes.tolist(), [7] * 7 + [1])

    def test_boundaries_fronts_and_uninvadable_species(self):
        # Rows wrap: 10 meets 20 below and 30 above; 10 can invade 20 only.
        Gamma = np.zeros((3, 3))
        Gamma[0, 1] = 1
        state = self.features([[10, 10, 10], [20, 20, -1], [30, 30, 30]], Gamma, [10, 20, 30])
        self.assertEqual((state["boundaries"], state["active_boundaries"]), (7, 2))
        self.assertEqual(state["front"].tolist(), [2, 0, 0])
        self.assertEqual(state["top_species"], 2)
        self.assertAlmostEqual(state["top_share"], 6 / 8)

    def test_patches_per_species_and_single_patch_share(self):
        # 20 is split in two by 30 and a blocked site; 10 and 30 each form one patch.
        state = self.features([[10, 10, 10, 10], [20, 30, 20, -1], [30, 30, 30, 30]],
                              np.zeros((3, 3)), [10, 20, 30])
        self.assertEqual(state["patches_per_species"].tolist(), [1, 2, 1])
        self.assertEqual(state["patches"], 4)
        self.assertAlmostEqual(state["single_patch_share"], 9 / 11)
        self.assertAlmostEqual(state["largest_patch_share"], 5 / 11)

    def test_switch_points_order_the_snapshots(self):
        # Low, a ramp up, high, a ramp down, low: one rise and one collapse.
        diversity = np.concatenate([np.full(2000, 5.0), np.linspace(5, 100, 1000),
                                    np.full(4000, 100.0), np.linspace(100, 5, 1000),
                                    np.full(2000, 5.0)])
        times = 150_000_000 + 1_000 * np.arange(diversity.size)
        points = F.switch_points(times, diversity, {"low": 10, "high": 50, "sample_every": 10_000,
                                                    "smooth_sigma": 20})
        snapshots = list(points["snapshots"].values())
        self.assertEqual(snapshots, sorted(snapshots))
        self.assertTrue(2000 <= points["snapshots"]["Steepest rise"] <= 3000)
        self.assertTrue(7000 <= points["snapshots"]["Steepest collapse"] <= 8000)
        rise, collapse = points["windows"]["Rise"], points["windows"]["Collapse"]
        self.assertTrue(rise[0] < points["rise"] < rise[1] <= collapse[0] < points["collapse"])
        self.assertEqual(collapse[1], points["snapshots"]["Low after the collapse"])

    def test_patch_total_matches_count_patches(self):
        rng = np.random.default_rng(3)
        lattice = rng.choice([-1, 1, 2, 3, 4], size=(20, 20), p=[0.3, 0.4, 0.2, 0.05, 0.05])
        state = self.features(lattice, np.zeros((4, 4)), [1, 2, 3, 4])
        self.assertEqual(state["patches"], MS.count_patches(lattice))
        self.assertEqual(int(state["patches_per_species"].sum()), state["patches"])


if __name__ == "__main__":
    unittest.main()
