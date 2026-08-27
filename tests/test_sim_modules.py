import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sim_modules as MS


class PatchCountingTests(unittest.TestCase):
    def test_periodic_four_neighbour_components(self):
        lattice = np.array(
            [
                [1, 2, 1],
                [2, 2, 2],
                [1, 2, 1],
            ],
            dtype=np.int64,
        )

        self.assertEqual(MS.count_patches(lattice), 2)

    def test_empty_and_blocked_sites_are_not_patches(self):
        lattice = np.array([[0, -1], [-1, 0]], dtype=np.int64)

        self.assertEqual(MS.count_patches(lattice), 0)

    def test_diagonal_sites_do_not_connect(self):
        lattice = np.array([[1, 2], [2, 1]], dtype=np.int64)

        self.assertEqual(MS.count_patches(lattice), 4)

    def test_large_unsigned_species_ids_are_preserved(self):
        species = 2**63 + 1
        lattice = np.array([[species, species]], dtype=np.uint64)

        self.assertEqual(MS.count_patches(lattice), 1)


class ShowResultsPatchinessTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _checkpoint_state(lattice, timestep, diversity_history):
        current_species = np.unique(lattice[lattice > 0]).astype(
            np.int64
        )
        return {
            "lattice": lattice,
            "Gamma": np.zeros(
                (current_species.size, current_species.size), dtype=np.uint8
            ),
            "current_species": current_species.tolist(),
            "newest_species": int(current_species.max(initial=0)),
            "timestep": timestep,
            "target_timestep": 2,
            "tracked_timesteps": np.arange(
                len(diversity_history), dtype=np.int64
            ),
            "diversity_history": np.asarray(
                diversity_history, dtype=np.int64
            ),
            "rng_state": np.random.default_rng(123).bit_generator.state,
            "gamma": 0.1,
            "alpha": 0.01,
            "track_every": 1,
            "populate_first_100": False,
        }

    @classmethod
    def _write_test_checkpoints(cls, directory):
        first = MS._write_checkpoint(
            directory,
            cls._checkpoint_state(
                np.ones((2, 2), dtype=np.int64), 1, [1, 1]
            ),
        )
        second = MS._write_checkpoint(
            directory,
            cls._checkpoint_state(
                np.array([[1, 2], [2, 1]], dtype=np.int64),
                2,
                [1, 1, 2],
            ),
        )
        return first, second

    def test_enabled_plot_uses_checkpoint_timesteps_on_right_axis(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_checkpoints(directory)
            state = MS.load_checkpoint(directory)

            with patch.object(MS.plt, "show"):
                MS.show_results(state, show_patchiness=True)

        figure = plt.gcf()
        self.assertEqual(len(figure.axes), 3)
        diversity_axis = figure.axes[1]
        patchiness_axis = figure.axes[2]
        self.assertEqual(diversity_axis.get_ylabel(), "Living species")
        self.assertEqual(patchiness_axis.get_ylabel(), "Species patches")
        self.assertTrue(
            diversity_axis.get_shared_x_axes().joined(
                diversity_axis, patchiness_axis
            )
        )
        np.testing.assert_array_equal(
            patchiness_axis.lines[0].get_xdata(), [1, 2]
        )
        np.testing.assert_array_equal(
            patchiness_axis.lines[0].get_ydata(), [1, 4]
        )

    def test_completed_result_metadata_discovers_sibling_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, second = self._write_test_checkpoints(
                Path(temporary_directory)
            )
            results = MS.load_checkpoint(second)
            results.pop("checkpoint_path")
            results["checkpoint_files"] = [second]

            timesteps, patchiness = MS._load_patchiness_history(results)

        np.testing.assert_array_equal(timesteps, [1, 2])
        np.testing.assert_array_equal(patchiness, [1, 4])

    def test_older_loaded_state_excludes_later_sibling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first, _ = self._write_test_checkpoints(
                Path(temporary_directory)
            )
            state = MS.load_checkpoint(first)

            timesteps, patchiness = MS._load_patchiness_history(state)

        np.testing.assert_array_equal(timesteps, [1])
        np.testing.assert_array_equal(patchiness, [1])

    def test_explicit_checkpoint_directory_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_checkpoints(directory)
            results = MS.load_checkpoint(directory)
            results.pop("checkpoint_path")

            timesteps, patchiness = MS._load_patchiness_history(
                results, checkpoint_dir=directory
            )

        np.testing.assert_array_equal(timesteps, [1, 2])
        np.testing.assert_array_equal(patchiness, [1, 4])

    def test_default_plot_does_not_read_checkpoint_metadata(self):
        results = self._checkpoint_state(
            np.ones((2, 2), dtype=np.int64), 1, [1, 1]
        )
        results["diversity"] = 1
        results["checkpoint_path"] = "missing/checkpoint.npz"

        with patch.object(MS.plt, "show"):
            MS.show_results(results)

        self.assertEqual(len(plt.gcf().axes), 2)

    def test_enabled_plot_requires_checkpoint_information(self):
        results = self._checkpoint_state(
            np.ones((2, 2), dtype=np.int64), 1, [1, 1]
        )
        results["diversity"] = 1

        with self.assertRaisesRegex(ValueError, "requires checkpoints"):
            MS.show_results(results, show_patchiness=True)

    def test_show_patchiness_must_be_boolean(self):
        results = self._checkpoint_state(
            np.ones((2, 2), dtype=np.int64), 1, [1, 1]
        )

        with self.assertRaisesRegex(TypeError, "must be True or False"):
            MS.show_results(results, show_patchiness="yes")


if __name__ == "__main__":
    unittest.main()
