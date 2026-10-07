"""Exact seeded trajectories frozen before the performance changes.

The golden SHA-256 values were generated from the unmodified ``sim_modules.py``
snapshot on 2026-10-06 using NumPy's PCG64 Generator. They deliberately include
the full RNG state: matching diversity alone would miss a change in draw order.
The original source SHA-256 was
``a870b9ee2d6dcedfe003aa459f3aaaf1d8ebfec2d4de851629a4e3eaf09e570c``.

Integer arrays use explicit little-endian dtypes and C order, and JSON uses
sorted keys. The fingerprints do not depend on native integer width, filesystem
paths, compression, progress display, or checkpoint retention metadata.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np

import sim_modules as MS


# These cases exercise species-storage growth and reuse, extinctions, periodic
# neighbours on thin lattices, empty sites, blocked sites and the timestep-100
# boundary. Single-species and empty-start cases exercise inactive time units
# followed by introductions, including states that must still fill empty sites.
# Seeds and case definitions are part of the frozen baseline.
_CASES = {
    "rectangular_growth_and_extinction": (
        "main_simulation",
        dict(L_col=9, L_row=7, D=3, gamma=0.65, alpha=12,
             T=80, track_every=4, seed=491),
    ),
    "forced_introduction_boundary": (
        "main_simulation",
        dict(L_col=8, L_row=5, D=3, gamma=0.45, alpha=0.4,
             T=117, track_every=5, seed=1209, populate_first_100=True),
    ),
    "percolation_with_blocked_sites": (
        "percolation_simulation",
        dict(L_col=7, L_row=5, D=4, gamma=0.6, alpha=6,
             T=40, track_every=4, seed=823, p=0.45),
    ),
    "single_row": (
        "main_simulation",
        dict(L_col=8, L_row=1, D=3, gamma=0.7, alpha=4,
             T=31, track_every=3, seed=29),
    ),
    "single_column": (
        "main_simulation",
        dict(L_col=1, L_row=8, D=3, gamma=0.7, alpha=4,
             T=31, track_every=3, seed=29),
    ),
    "single_site": (
        "main_simulation",
        dict(L_col=1, L_row=1, D=1, gamma=0.65, alpha=0.8,
             T=30, track_every=2, seed=998, populate_first_100=True),
    ),
    "no_interactions_or_introductions": (
        "main_simulation",
        dict(L_col=5, L_row=4, D=6, gamma=0, alpha=12,
             T=17, track_every=4, seed=114),
    ),
    "all_sites_blocked_no_species": (
        "percolation_simulation",
        dict(L_col=4, L_row=3, D=0, gamma=0.6, alpha=999,
             T=15, track_every=4, seed=867, p=1,
             populate_first_100=True),
    ),
    "raw_state_with_ordered_noncontiguous_ids": (
        "simulation_from_state",
        dict(gamma=0.55, alpha=6, T=19, track_every=3, seed=109,
             populate_first_100=True),
    ),
    "full_single_species_with_rare_introductions": (
        "main_simulation",
        dict(L_col=8, L_row=6, D=1, gamma=0.4, alpha=0.05,
             T=240, track_every=30, seed=729),
    ),
    "full_single_species_without_introductions": (
        "main_simulation",
        dict(L_col=8, L_row=6, D=1, gamma=0.6, alpha=0,
             T=80, track_every=8, seed=773),
    ),
    "single_species_with_blocked_sites": (
        "percolation_simulation",
        dict(L_col=8, L_row=6, D=1, gamma=0.4, alpha=0.05,
             T=240, track_every=30, seed=221, p=0.55),
    ),
    "single_species_with_empty_usable_sites": (
        "simulation_from_state",
        dict(
            initial_state={
                "lattice": np.array(
                    [[17, 0, 0, -1], [0, 0, -1, 0], [0, 0, 0, 0]],
                    dtype=np.int64,
                ),
                "Gamma": np.zeros((1, 1), dtype=np.uint8),
                "current_species": [17],
                "newest_species": 17,
                "timestep": 105,
            },
            gamma=0.4, alpha=0.02, T=180, track_every=10, seed=509,
        ),
    ),
    "empty_usable_and_blocked_sites": (
        "simulation_from_state",
        dict(
            initial_state={
                "lattice": np.array(
                    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0]],
                    dtype=np.int64,
                ),
                "Gamma": np.empty((0, 0), dtype=np.uint8),
                "current_species": [],
                "newest_species": 27,
                "timestep": 103,
            },
            gamma=0.4, alpha=0.2, T=90, track_every=10, seed=394,
        ),
    ),
}

_GOLDEN_TRAJECTORIES = {
    "rectangular_growth_and_extinction":
        "d6a9a920d4a2285d699f9e9134c8beb2a1875bc6316e5208a5842c5a5847a372",
    "forced_introduction_boundary":
        "b276530e4c63bd2941eca4eced16944b1fc79ce6c8e9883dd1b46b7f6177caff",
    "percolation_with_blocked_sites":
        "b3ba6c514ad8d2f12338fceba9f86b254e670e347eb564703e7f48ac4a16f61b",
    "single_row":
        "9668b965dd603bee57286ecc4132889b37cd90811bfc420712a110e35375a16e",
    "single_column":
        "3d72e3813a549b342ad4d17cd032bcfbd6faba1f2d6a22586fe59b11bfef096c",
    "single_site":
        "9d98129526f3622f3b0d8ce5ff68fa8bcdf849cc172f1f607a3f1968a433960d",
    "no_interactions_or_introductions":
        "bb01df3ebc9ee6e90553c62a27c4eb9f8b910c43e343fc42b8a9bf3ebe0ff81a",
    "all_sites_blocked_no_species":
        "1eb7b071fd0b42d25c64aa76db9367f00e9882f06d36b1010dc6720e0e97746e",
    "raw_state_with_ordered_noncontiguous_ids":
        "c39a20cff2277aa73e190d2128a18a4232ab6384a510e56b90dd155c7bce5588",
    "full_single_species_with_rare_introductions":
        "bbab069b653a91361eb521f103cd41635ea0b26d314f089266d5a447ef447820",
    "full_single_species_without_introductions":
        "48ddd401b011088ef55180d86571342dbca37117488db339d33c644ce960591b",
    "single_species_with_blocked_sites":
        "50a245c1f8066d377cf86cd796f62c8895111dccae98fd7a37c5d7d971ff1eea",
    "single_species_with_empty_usable_sites":
        "920f51dd15e0c104a374601b12a6dd44c1d89b909a70823848e8bf70481f8088",
    "empty_usable_and_blocked_sites":
        "c7029571be4776c0857894b645dbc057864475de15afd80ebf8a749c707682df",
}


def _raw_state():
    # Gamma follows [99, 7, 31], deliberately not sorted by species ID.
    return {
        "lattice": np.array(
            [[0, 99, 7, -1], [31, 0, 7, 99], [31, 7, 0, 99]],
            dtype=np.int64,
        ),
        "Gamma": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                          dtype=np.uint8),
        "current_species": [99, 7, 31],
        "newest_species": 104,
        "timestep": 94,
    }


def _run_case(module, name, **overrides):
    function_name, options = _CASES[name]
    options = dict(options, **overrides)
    if function_name == "percolation_simulation":
        # These frozen trajectories predate the requested cluster restriction.
        # Keep testing the original random stream and kernel using its opt-out.
        options.setdefault("largest_cluster_only", False)
    if function_name == "simulation_from_state":
        options.setdefault("initial_state", _raw_state())
    return getattr(module, function_name)(**options)


def _trajectory_digest(result):
    digest = hashlib.sha256()
    for name, dtype in (
        ("lattice", "<i8"),
        ("Gamma", "u1"),
        ("current_species", "<i8"),
        ("tracked_timesteps", "<i8"),
        ("diversity_history", "<i8"),
    ):
        array = np.asarray(result[name], dtype=dtype, order="C")
        header = json.dumps(
            [name, dtype, list(array.shape)], separators=(",", ":")
        ).encode("ascii")
        digest.update(len(header).to_bytes(4, "little"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    scalars = {
        key: int(result[key])
        for key in ("newest_species", "diversity", "timestep",
                    "target_timestep", "initial_newest_species")
    }
    digest.update(json.dumps(scalars, sort_keys=True,
                             separators=(",", ":")).encode("ascii"))
    digest.update(json.dumps(result["rng_state"], sort_keys=True,
                             separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


class SimulationExactnessTests(unittest.TestCase):
    def assert_frozen_trajectory(self, name, result):
        self.assertEqual(result["rng_state"]["bit_generator"], "PCG64")
        self.assertEqual(_trajectory_digest(result),
                         _GOLDEN_TRAJECTORIES[name], name)

    def test_seeded_trajectories_match_original_module(self):
        for name in _CASES:
            with self.subTest(case=name):
                self.assert_frozen_trajectory(name, _run_case(MS, name))

    def test_progress_chunking_preserves_trajectory_and_rng_state(self):
        for name in ("rectangular_growth_and_extinction",
                     "forced_introduction_boundary",
                     "full_single_species_with_rare_introductions",
                     "empty_usable_and_blocked_sites"):
            with self.subTest(case=name), patch("tqdm.auto.tqdm") as progress:
                result = _run_case(MS, name, progress=True)
                self.assert_frozen_trajectory(name, result)
                self.assertGreater(progress.return_value.update.call_count, 1)
                progress.return_value.close.assert_called_once()

    def test_checkpoint_chunking_preserves_trajectory_and_rng_state(self):
        for name in ("forced_introduction_boundary",
                     "percolation_with_blocked_sites",
                     "raw_state_with_ordered_noncontiguous_ids",
                     "full_single_species_without_introductions",
                     "single_species_with_blocked_sites",
                     "single_species_with_empty_usable_sites",
                     "empty_usable_and_blocked_sites"):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as folder:
                result = _run_case(
                    MS, name, checkpoint_dir=folder,
                    retention=MS.RetentionPolicy(keep_all=True),
                )
                self.assert_frozen_trajectory(name, result)
                self.assertGreater(len(result["checkpoint_files"]), 1)
                final_checkpoint = MS.load_checkpoint(folder)
                self.assert_frozen_trajectory(name, final_checkpoint)

    def test_continuation_preserves_draws_and_recording(self):
        for name, split in (
            ("rectangular_growth_and_extinction", 36),
            ("forced_introduction_boundary", 55),
            ("percolation_with_blocked_sites", 16),
            ("raw_state_with_ordered_noncontiguous_ids", 6),
            ("full_single_species_with_rare_introductions", 60),
            ("full_single_species_without_introductions", 32),
            ("single_species_with_blocked_sites", 60),
            ("single_species_with_empty_usable_sites", 10),
            ("empty_usable_and_blocked_sites", 10),
        ):
            # Splits are multiples of track_every so no extra intermediate
            # sample is introduced by ending the first simulation leg.
            with self.subTest(case=name):
                first_leg = _run_case(MS, name, T=split)
                full_T = _CASES[name][1]["T"]
                result = _run_case(
                    MS, name, initial_state=first_leg, T=full_T - split,
                    seed=999999, continue_history=True,
                )
                self.assert_frozen_trajectory(name, result)

    def test_disk_resume_restores_rng_instead_of_reseeding(self):
        name = "forced_introduction_boundary"
        with tempfile.TemporaryDirectory() as folder:
            _run_case(MS, name, T=55, checkpoint_dir=folder)
            state = MS.load_checkpoint(Path(folder))
            resumed = _run_case(
                MS, name, initial_state=state, T=62, seed=1,
                continue_history=True,
            )
            self.assert_frozen_trajectory(name, resumed)


if __name__ == "__main__":
    unittest.main()
