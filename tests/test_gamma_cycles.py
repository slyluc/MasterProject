"""Directed Gamma-cycle counts and event-side checkpoint selection."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sim_modules as MS


def cycle_matrix(length):
    matrix = np.zeros((4, 4), dtype=np.uint8)
    for vertex in range(length):
        matrix[vertex, (vertex + 1) % length] = 1
    return matrix


class DirectedCycleTests(unittest.TestCase):
    def test_exponential_fit_recovers_semi_log_line(self):
        counts = {
            length: np.log10(7 * np.exp(-0.4 * length))
            for length in range(2, 8)
        }
        fit = MS._fit_cycle_exponential(counts, min_length=3, max_length=6)
        self.assertEqual(fit["lengths_used"].tolist(), [3, 4, 5, 6])
        self.assertAlmostEqual(fit["constant"], 7.0)
        self.assertAlmostEqual(fit["rate"], -0.4)
        self.assertAlmostEqual(fit["r_squared_log_space"], 1.0)

    def test_power_law_fit_recovers_constant_and_exponent(self):
        counts = {
            length: np.log10(100 / length ** 2)
            for length in range(2, 7)
        }
        fit = MS._fit_cycle_power_law(counts)
        self.assertAlmostEqual(fit["constant"], 100.0)
        self.assertAlmostEqual(fit["exponent"], 2.0)
        self.assertAlmostEqual(fit["r_squared_log_space"], 1.0)
        tail = MS._fit_cycle_power_law(
            counts, min_length=4, max_length=5
        )
        self.assertEqual(tail["lengths_used"].tolist(), [4, 5])
        self.assertAlmostEqual(tail["constant"], 100.0)

    def test_counts_distinct_directed_cycles_by_length(self):
        triangle = cycle_matrix(3)
        square = cycle_matrix(4)
        mutual = np.zeros((4, 4), dtype=np.uint8)
        mutual[0, 1] = mutual[1, 0] = 1
        self.assertEqual(MS._directed_cycle_counts(triangle, 4),
                         {2: 0, 3: 1, 4: 0})
        self.assertEqual(MS._directed_cycle_counts(square, 4),
                         {2: 0, 3: 0, 4: 1})
        self.assertEqual(MS._directed_cycle_counts(mutual, 4),
                         {2: 1, 3: 0, 4: 0})

    def test_longer_cycle_search(self):
        matrix = np.zeros((5, 5), dtype=np.uint8)
        for vertex in range(5):
            matrix[vertex, (vertex + 1) % 5] = 1
        self.assertEqual(MS._directed_cycle_counts(matrix, 5)[5], 1)

    def test_all_lengths_sampler_finds_a_deterministic_five_cycle(self):
        matrix = np.zeros((13, 13), dtype=np.uint8)
        for vertex in range(5):
            matrix[vertex, (vertex + 1) % 5] = 1
        counts, logs, hits, estimated = MS._cycle_histogram_for_gamma(
            matrix, None, 100, 9
        )
        self.assertAlmostEqual(counts[5], 1.0)
        self.assertEqual(hits[5], 8)
        self.assertIn(5, estimated)
        self.assertEqual(counts[13], 0.0)
        self.assertAlmostEqual(logs[5], 0.0)


class GammaHistogramTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        times = np.arange(0, 201, 10, dtype=np.int64)
        diversity = np.where(
            ((times < 50) | ((times >= 100) & (times < 150))), 40, 1
        )
        MS._write_analysis(self.directory, {
            "tracked_timesteps": times,
            "diversity_history": diversity,
            "patch_history": np.ones(times.size, dtype=np.int64),
            "event_timesteps": np.array([50, 100, 150], dtype=np.int64),
            "timestep": 200,
            "target_timestep": 200,
            "gamma": 0.1,
            "alpha": 0.01,
            "track_every": 10,
            "populate_first_100": False,
            "lattice_shape": (2, 2),
        })
        for timestep, matrix in (
            (30, cycle_matrix(3)),
            (40, cycle_matrix(4)),
            (50, np.zeros((4, 4), dtype=np.uint8)),
            (60, np.zeros((4, 4), dtype=np.uint8)),
            (90, np.zeros((4, 4), dtype=np.uint8)),
            (100, cycle_matrix(3)),
            (110, cycle_matrix(3)),
            (140, cycle_matrix(4)),
            (150, np.zeros((4, 4), dtype=np.uint8)),
        ):
            np.savez(
                self.directory / f"checkpoint_{timestep:012d}.npz",
                timestep=np.int64(timestep), Gamma=matrix,
            )

    def run_histogram(self, **kwargs):
        with patch.object(MS.plt, "show"):
            result = MS.show_gamma_cycle_histogram(self.directory, **kwargs)
        self.addCleanup(plt.close, result["figure"])
        return result

    def test_default_side_of_collapse_and_log_axes(self):
        result = self.run_histogram(
            event_type="collapse", event_index=0, log_x=True, log_y=True
        )
        self.assertEqual(result["event_timesteps"].tolist(), [50])
        self.assertEqual(result["checkpoint_timesteps"].tolist(), [30, 40])
        self.assertEqual(result["cycle_counts"], {2: 0, 3: 1, 4: 1})
        self.assertEqual(result["per_checkpoint_counts"][30],
                         {2: 0, 3: 1, 4: 0})
        self.assertEqual(result["axes"].get_xscale(), "log")
        self.assertEqual(result["axes"].get_yscale(), "log")

    def test_generation_after_and_side_override(self):
        result = self.run_histogram(event_type="generation", event_index=0)
        self.assertEqual(result["event_timesteps"].tolist(), [100])
        self.assertEqual(result["checkpoint_timesteps"].tolist(),
                         [100, 110, 140])
        self.assertEqual(result["cycle_counts"], {2: 0, 3: 2, 4: 1})

        before = self.run_histogram(
            event_type="generation", event_index=0, side="before"
        )
        self.assertEqual(before["checkpoint_timesteps"].tolist(), [50, 60, 90])
        self.assertEqual(before["cycle_counts"], {2: 0, 3: 0, 4: 0})

    def test_window_limits_checkpoint_selection(self):
        result = self.run_histogram(
            event_type="collapse", event_index=0, window=10
        )
        self.assertEqual(result["checkpoint_timesteps"].tolist(), [40])
        self.assertEqual(result["cycle_counts"], {2: 0, 3: 0, 4: 1})

    def test_all_collapses_sum_each_selected_snapshot_once(self):
        result = self.run_histogram(event_type="collapse", event_index="all")
        self.assertEqual(result["event_timesteps"].tolist(), [50, 150])
        self.assertEqual(result["checkpoint_timesteps"].tolist(),
                         [30, 40, 100, 110, 140])
        self.assertEqual(result["cycle_counts"], {2: 0, 3: 3, 4: 2})

    def test_power_law_option_draws_label_and_returns_fit(self):
        result = self.run_histogram(
            event_type="collapse", event_index="all", fit_power_law=True,
            power_law_min_length=3, power_law_max_length=4,
            log_x=True, log_y=True,
        )
        fit = result["power_law_fit"]
        self.assertIsNotNone(fit)
        self.assertEqual(fit["lengths_used"].tolist(), [3, 4])
        self.assertEqual(len(result["axes"].lines), 1)
        labels = [text.get_text() for text in result["axes"].get_legend().texts]
        self.assertTrue(any("L=3–4" in label and "C=" in label
                            and "alpha=" in label
                            for label in labels))

    def test_exponential_option_draws_semi_log_fit(self):
        result = self.run_histogram(
            event_type="collapse", event_index="all",
            fit_model="exponential", fit_min_length=3, fit_max_length=4,
            log_y=True,
        )
        fit = result["exponential_fit"]
        self.assertIs(result["fit"], fit)
        self.assertIsNone(result["power_law_fit"])
        self.assertEqual(fit["lengths_used"].tolist(), [3, 4])
        self.assertEqual(result["axes"].get_xscale(), "linear")
        self.assertEqual(result["axes"].get_yscale(), "log")
        self.assertEqual(len(result["axes"].lines), 1)
        labels = [text.get_text() for text in result["axes"].get_legend().texts]
        self.assertTrue(any("Exponential" in label and "A=" in label
                            and "k=" in label for label in labels))

    def test_power_law_requires_two_positive_lengths(self):
        with patch.object(MS.plt, "show"):
            with self.assertRaisesRegex(ValueError, "at least two"):
                MS.show_gamma_cycle_histogram(
                    self.directory, event_type="collapse", event_index=0,
                    window=10, fit_power_law=True,
                )
        with self.assertRaisesRegex(ValueError, "power_law_max_length"):
            MS.show_gamma_cycle_histogram(
                self.directory, power_law_min_length=5,
                power_law_max_length=4,
            )


if __name__ == "__main__":
    unittest.main()
