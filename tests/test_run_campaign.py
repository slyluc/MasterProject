"""The campaign tool: plan, scheduler with child processes, and exact continuation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis import campaign_runs
import sim_modules as MS
from tools import run_campaign as campaign


def small(p=0.2, gamma=0.05, **extra):
    return {"L_col": 20, "L_row": 20, "D": 1, "gamma": gamma, "alpha": 20.0, "p": p,
            "track_every": 100, "populate_first_100": False, "largest_cluster_only": True,
            "max_forbidden_cycle_length": 6, "max_cycle_rejection_attempts": 1_000_000,
            **extra}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        source = MS.SimulationConfig(**small(), T=2_000, seed=11,
                                     checkpoint_dir=str(self.base / "source"))
        source.run_percolation()
        self.source_file = max((self.base / "source").glob("checkpoint_*.npz"),
                               key=MS._checkpoint_timestep)
        self.groups = [{"group": "g", "question": "q", "design": "d", "runs": [
            {"label": "fresh", "mode": "fresh", "T": 1_500, "config": small(p=0.1)},
            {"label": "branch", "mode": "branch", "source": "S", "T": 1_500,
             "config": {key: value for key, value in small(gamma=0.08).items()
                        if key not in ("L_col", "L_row", "D", "p")}},
            {"label": "continue", "mode": "continue", "source": "S", "T": 2_000},
        ]}]
        self.sources = {"S": {"path": self.source_file}}

    def test_plan_gives_each_run_its_own_seed_and_interval(self):
        plan = campaign.materialize(self.groups, self.sources, seed_root=5)
        fresh, branch, continued = plan["runs"]
        self.assertEqual([fresh["base_seed"], branch["base_seed"]], [5000, 5001])
        self.assertIsNone(continued["base_seed"])
        self.assertEqual((fresh["start_timestep"], fresh["target_timestep"]), (0, 1_500))
        self.assertEqual((branch["start_timestep"], branch["target_timestep"]), (2_000, 3_500))
        self.assertEqual(continued["target_timestep"], 4_000)
        self.assertEqual(branch["config"]["p"], 0.2)  # Taken from the parent.

    def test_scheduler_runs_everything_and_a_continuation_is_exact(self):
        root, _ = campaign.create_campaign(self.base / "campaign", self.groups, self.sources,
                                           seed_root=5)
        self.assertTrue((root / "initial_states" / "S.npz").is_file())
        code = campaign.run_scheduler(root, 2, poll_seconds=0.2, status_every=0.0,
                                      retry_delay=0.0, max_attempts=1)
        self.assertEqual(code, 0, (root / "campaign.log").read_text(encoding="utf-8"))
        status = campaign._read_json(root / campaign.STATUS)
        self.assertEqual(status["counts"], {"done": 3})
        plan = campaign._read_json(root / campaign.PLAN)
        for run in plan["runs"]:
            self.assertEqual(campaign.newest_timestep(root, run), run["target_timestep"])

        # The continuation equals the source run carried on without a break.
        reference = MS.SimulationConfig(**small(), T=4_000, seed=11,
                                        checkpoint_dir=str(self.base / "reference"))
        expected = reference.run_percolation()
        continued = MS.load_checkpoint(root / "g" / "continue")
        np.testing.assert_array_equal(continued["lattice"], expected["lattice"])
        np.testing.assert_array_equal(continued["Gamma"], expected["Gamma"])
        self.assertEqual(continued["rng_state"], expected["rng_state"])
        np.testing.assert_array_equal(continued["diversity_history"],
                                      expected["diversity_history"])

        # The branch is a one-job batch that resume_simulations accepts as complete.
        state = MS.load_checkpoint(self.source_file)
        resumed = MS.resume_simulations(root / "g" / "branch", max_workers=1, initial_state=state)
        self.assertTrue(resumed[0]["resume_metadata"]["already_completed"])

        # Running a finished campaign again changes nothing.
        before = sorted(path.name for path in (root / "g" / "continue").glob("checkpoint_*"))
        self.assertEqual(campaign.run_scheduler(root, 2, poll_seconds=0.2, status_every=0.0), 0)
        after = sorted(path.name for path in (root / "g" / "continue").glob("checkpoint_*"))
        self.assertEqual(before, after)

        # The analysis loader reads done runs only, each over its own interval.
        status = campaign._read_json(root / campaign.STATUS)
        for row in status["runs"]:
            if row["id"] == "g/continue":
                row["state"] = "running"
        campaign._write_json(root / campaign.STATUS, status)
        done, waiting, _ = campaign_runs.load_done({"g"}, root=root)
        self.assertEqual(sorted(run["label"] for run in done), ["branch", "fresh"])
        self.assertEqual([run["label"] for run in waiting], ["continue"])
        branch = next(run for run in done if run["label"] == "branch")
        self.assertEqual((int(branch["times"][0]), branch["end_time"]), (2_000, 3_500))


if __name__ == "__main__":
    unittest.main()
