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

    def test_requested_lattice_uses_nearest_checkpoint_and_full_histories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_checkpoints(directory)
            results = MS.load_checkpoint(directory)

            with patch.object(MS.plt, "show"):
                MS.show_results(
                    results,
                    show_patchiness=True,
                    lattice_timestep=1.25,
                )

        figure = plt.gcf()
        lattice_axis, diversity_axis, patchiness_axis = figure.axes
        self.assertIn("timestep 1", lattice_axis.get_title())
        self.assertIn("1 living species", lattice_axis.get_title())
        np.testing.assert_array_equal(
            lattice_axis.images[0].get_array(),
            np.full((2, 2), 2, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            diversity_axis.lines[0].get_xdata(), [0, 1, 2]
        )
        np.testing.assert_array_equal(
            diversity_axis.lines[0].get_ydata(), [1, 1, 2]
        )
        np.testing.assert_array_equal(
            patchiness_axis.lines[0].get_xdata(), [1, 2]
        )
        np.testing.assert_array_equal(
            patchiness_axis.lines[0].get_ydata(), [1, 4]
        )

    def test_nearest_lattice_tie_uses_earlier_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_checkpoints(directory)
            results = MS.load_checkpoint(directory)

            lattice, timestep = MS._load_lattice_snapshot(
                results, 1.5
            )

        self.assertEqual(timestep, 1)
        np.testing.assert_array_equal(lattice, np.ones((2, 2)))

    def test_later_sibling_is_not_used_for_older_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first, _ = self._write_test_checkpoints(directory)
            results = MS.load_checkpoint(first)

            lattice, timestep = MS._load_lattice_snapshot(results, 2)

        self.assertEqual(timestep, 1)
        np.testing.assert_array_equal(lattice, results["lattice"])

    def test_historical_lattice_requires_checkpoint_information(self):
        results = self._checkpoint_state(
            np.ones((2, 2), dtype=np.int64), 2, [1, 1, 1]
        )

        with self.assertRaisesRegex(ValueError, "requires checkpoints"):
            MS._load_lattice_snapshot(results, 1)

    def test_request_after_result_uses_final_lattice_without_checkpoints(self):
        results = self._checkpoint_state(
            np.ones((2, 2), dtype=np.int64), 2, [1, 1, 1]
        )

        lattice, timestep = MS._load_lattice_snapshot(results, 3)

        self.assertEqual(timestep, 2)
        np.testing.assert_array_equal(lattice, results["lattice"])

    def test_lattice_timestep_validation(self):
        results = self._checkpoint_state(
            np.ones((2, 2), dtype=np.int64), 2, [1, 1, 1]
        )

        for value in (True, "1"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    MS._load_lattice_snapshot(results, value)
        for value in (-1, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MS._load_lattice_snapshot(results, value)

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


class AnimateLatticeTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _checkpoint_state(lattice, timestep, diversity_history):
        return ShowResultsPatchinessTests._checkpoint_state(
            lattice, timestep, diversity_history
        )

    @staticmethod
    def _write_test_checkpoints(directory):
        return ShowResultsPatchinessTests._write_test_checkpoints(directory)

    def test_animation_uses_ordered_frames_and_stable_species_colors(self):
        from matplotlib.animation import FuncAnimation

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_checkpoints(directory)

            animation = MS.animate_lattice(directory, display=False)
            self.assertIsInstance(animation, FuncAnimation)
            self.assertEqual(animation._save_count, 2)

            first_artists = animation._init_func()
            first_colors = np.asarray(
                first_artists[0].get_array()
            ).copy()
            second_artists = animation._func(1)
            second_colors = np.asarray(
                second_artists[0].get_array()
            ).copy()

        np.testing.assert_array_equal(
            first_colors[0, 0], second_colors[0, 0]
        )
        self.assertFalse(
            np.array_equal(first_colors[0, 1], second_colors[0, 1])
        )
        self.assertIn("timestep 2", second_artists[1].get_text())
        self.assertIn("2 living species", second_artists[1].get_text())
        animation._draw_was_started = True

    def test_animation_stride_always_keeps_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for timestep in range(1, 5):
                state = self._checkpoint_state(
                    np.full((2, 2), timestep, dtype=np.int64),
                    timestep,
                    np.arange(1, timestep + 2),
                )
                state["target_timestep"] = 4
                MS._write_checkpoint(directory, state)
            results = MS.load_checkpoint(directory)

            _, frames = MS._animation_frame_sources(
                results, frame_stride=2
            )

        self.assertEqual([timestep for timestep, _ in frames], [1, 3, 4])

    def test_animation_timestep_bounds_do_not_include_outside_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for timestep in range(1, 5):
                state = self._checkpoint_state(
                    np.full((2, 2), timestep, dtype=np.int64),
                    timestep,
                    np.arange(1, timestep + 2),
                )
                state["target_timestep"] = 4
                MS._write_checkpoint(directory, state)
            results = MS.load_checkpoint(directory)

            _, frames = MS._animation_frame_sources(
                results,
                start_timestep=1.5,
                end_timestep=3.5,
            )

        self.assertEqual([timestep for timestep, _ in frames], [2, 3])

    def test_animation_max_frames_downsamples_and_keeps_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for timestep in range(1, 7):
                state = self._checkpoint_state(
                    np.full((2, 2), timestep, dtype=np.int64),
                    timestep,
                    np.arange(1, timestep + 2),
                )
                state["target_timestep"] = 6
                MS._write_checkpoint(directory, state)
            results = MS.load_checkpoint(directory)

            _, frames = MS._animation_frame_sources(
                results, max_frames=3
            )

        self.assertLessEqual(len(frames), 3)
        self.assertEqual(frames[0][0], 1)
        self.assertEqual(frames[-1][0], 6)

    def test_specific_checkpoint_excludes_later_siblings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first, second = self._write_test_checkpoints(directory)

            _, frames = MS._animation_frame_sources(second)
            with self.assertRaisesRegex(ValueError, "at least two"):
                MS._animation_frame_sources(first)

        self.assertEqual([timestep for timestep, _ in frames], [1, 2])

    def test_gif_save_uses_pillow_and_creates_animation_directory(self):
        from matplotlib.animation import PillowWriter

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self._write_test_checkpoints(directory)
            destination = directory / "animations" / "run.gif"

            with patch("matplotlib.animation.Animation.save") as save:
                animation = MS.animate_lattice(
                    directory,
                    interval=200,
                    save_path=destination,
                    display=False,
                )

            self.assertTrue(destination.parent.is_dir())
            save.assert_called_once()
            self.assertEqual(save.call_args.args[0], str(destination))
            self.assertIsInstance(
                save.call_args.kwargs["writer"], PillowWriter
            )
            self.assertEqual(save.call_args.kwargs["writer"].fps, 5)
            animation._draw_was_started = True

    def test_animation_requires_checkpoint_history(self):
        results = self._checkpoint_state(
            np.ones((2, 2), dtype=np.int64), 2, [1, 1, 1]
        )

        with self.assertRaisesRegex(ValueError, "requires checkpoints"):
            MS.animate_lattice(results, display=False)


if __name__ == "__main__":
    unittest.main()
