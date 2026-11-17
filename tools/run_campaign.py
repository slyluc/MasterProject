"""Queue a campaign of independent simulations on a fixed number of CPU slots.

Every run is either a single-job batch (its own batch_plan.json, job folder and seed, started
by MS.run_simulations and continued by MS.resume_simulations, exactly as a one-job sweep), or
an exact continuation of an earlier run into a new folder (its saved random state is restored,
so the continuation is that run's own trajectory). Each run is one child process. The
scheduler keeps --slots of them running, in the plan's order, so a slot freed by a finished run
goes straight to the next one.

A child that stops with an error is started again from its newest checkpoint, up to
--max-attempts times. That is exact: checkpoints hold the random state, and at most one tracking
interval (10,000 steps) is repeated. Children run at below-normal priority.

The 2026-11-09 campaign (see CAMPAIGN_GROUPS and RESEARCH_LOG.md) is the default.

Preview: python tools/run_campaign.py --dry-run
Start:   python tools/run_campaign.py --detach
Status:  python tools/run_campaign.py --status E:/MasterProject/checkpoints/<campaign>
Stop:    python tools/run_campaign.py --stop E:/MasterProject/checkpoints/<campaign>
Resume:  python tools/run_campaign.py --resume E:/MasterProject/checkpoints/<campaign> --detach

--detach starts the scheduler through WMI, so it is not a child of this terminal or editor and
keeps running when they close. Without it the scheduler runs in the foreground; Ctrl+C stops it
and its runs, and --resume continues them. Change the number of slots of a running scheduler by
writing a number to <campaign>/SLOTS.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sim_modules as MS  # noqa: E402


CHECKPOINTS = PROJECT_ROOT / "checkpoints"
DEFAULT_PARENT = Path("E:/MasterProject/checkpoints")
PLAN, STATUS, LOG = "campaign_plan.json", "campaign_status.json", "campaign.log"
STOP_FILE, SLOTS_FILE = "STOP", "SLOTS"
# Past seed roots: 6767 grid, 6768 alpha r1, 6769 alpha r2, 6770 gamma swap, 6771 fresh r3-r5.
SEED_ROOT = 6772
# Measured on this PC: about 5M steps per hour per run with 16 runs at once (RESEARCH_LOG.md).
STEPS_PER_HOUR_PER_RUN = 5_000_000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_NO_WINDOW = 0x08000000

GRID = CHECKPOINTS / "no_short_cycles_p_gamma_20261009_011738_ef440b70"
ALPHA_R1 = CHECKPOINTS / "no_short_cycles_alpha_p20_g003_20261021_011454_0f93dba1"
ALPHA_R2 = CHECKPOINTS / "no_short_cycles_alpha_p20_g003_rep2_20261022_005553_dfacd28f"
# Starting states. A fingerprint (lattice, Gamma, species, time; not the RNG) must match the one
# the earlier batches recorded for the same parent.
SOURCES = {
    "parent_A": {
        "path": GRID / "job_00004_seed_109676191229612237887150382465779759813"
        / "checkpoint_000050000000.npz",
        "fingerprint": "e4b53afae6706f8a30a7d3db70f25879d4911caa1e5516d59ec8c7229dfb4206",
        "note": "grid p = 0.2, gamma = 0.03 at 50M (129 species); parent of the alpha sweep",
    },
    "parent_B": {
        "path": GRID / "job_00008_seed_338341833799779468659849412261112410591"
        / "checkpoint_000050000000.npz",
        "fingerprint": "7b980673808a212ce529f72d444dcab50136f9d4d04a681044154c18ee46c9e0",
        "note": "grid p = 0.3, gamma = 0.03 at 50M (99 species); parent of the gamma swap",
    },
    "alpha_r1_a0.001": {
        "path": ALPHA_R1 / "job_00000_seed_171508993746080715750286890673418307787"
        / "checkpoint_000150000000.npz",
        "note": "alpha sweep r1, alpha = 0.001, final state at 150M",
    },
    "alpha_r1_a0.002": {
        "path": ALPHA_R1 / "job_00001_seed_230489409210967656605692221840594607897"
        / "checkpoint_000150000000.npz",
        "note": "alpha sweep r1, alpha = 0.002, final state at 150M",
    },
    "alpha_r2_a0.001": {
        "path": ALPHA_R2 / "job_00000_seed_164279621597021636181255278917113549378"
        / "checkpoint_000150000000.npz",
        "note": "alpha sweep r2, alpha = 0.001, final state at 150M",
    },
    "alpha_r2_a0.002": {
        "path": ALPHA_R2 / "job_00001_seed_121210882997407811484200316629206970519"
        / "checkpoint_000150000000.npz",
        "note": "alpha sweep r2, alpha = 0.002, final state at 150M",
    },
}

CONTROL_RATE = 0.0125 * 0.03  # alpha * gamma of the gamma-swap control: 3.75e-4 per unit.
NO_SHORT = {"max_forbidden_cycle_length": 6, "max_cycle_rejection_attempts": 10_000_000}
ALL_CYCLES = {"max_forbidden_cycle_length": None, "max_cycle_rejection_attempts": 10_000_000}


def _fresh(p, gamma, alpha=0.0125, **cycles):
    return {"L_col": 200, "L_row": 200, "D": 1, "gamma": gamma, "alpha": alpha, "p": p,
            "track_every": 10_000, "populate_first_100": False, "largest_cluster_only": True,
            **cycles}


def _branch(gamma, alpha):
    # A branch keeps its parent's lattice; p and the lattice size only label the run.
    return {"gamma": gamma, "alpha": alpha, "track_every": 10_000, "populate_first_100": False,
            "largest_cluster_only": True, **NO_SHORT}


def campaign_groups():
    """The 2026-11-09 campaign, in launch order. Longest runs first, then by priority."""
    groups = [{
        "group": "q4_extend_low_alpha",
        "question": "Q4: does an established community at alpha <= 0.002 persist, or only decay "
                    "slowly? 100M at alpha = 0.001 is ~3,000 newcomers, as many as 8M at 0.0125.",
        "design": "exact continuations of alpha sweep r1/r2 (siblings from parent A) from 150M "
                  "to 350M, on their own saved random state",
        "runs": [{"label": name.removeprefix("alpha_"), "mode": "continue", "source": name,
                  "T": 200_000_000}
                 for name in ("alpha_r1_a0.001", "alpha_r1_a0.002",
                              "alpha_r2_a0.001", "alpha_r2_a0.002")],
    }, {
        "group": "q5_fixed_rate_from_B",
        "question": "Q5: does gamma act through link density or the arrival rate alpha*gamma? "
                    "Gamma changed with alpha*gamma held at the control's 3.75e-4.",
        "design": "branches of parent B at 50M, 100M each; compare with the gamma swap "
                  "(same parent, alpha = 0.0125)",
        "runs": [{"label": f"g{gamma:g}_a{CONTROL_RATE / gamma:.4g}", "mode": "branch",
                  "source": "parent_B", "T": 100_000_000,
                  "config": _branch(gamma, CONTROL_RATE / gamma)}
                 for gamma in (0.04, 0.05, 0.07, 0.09)],
    }, {
        "group": "q5_raised_alpha_from_B",
        "question": "Q5: gamma = 0.03 with alpha raised to the arrival rate the gamma swap had at "
                    "gamma = 0.04, 0.05, 0.07 and 0.09. Also the first alpha above 0.0125.",
        "design": "branches of parent B at 50M, 100M each",
        "runs": [{"label": f"g0.03_a{0.0125 * rate_gamma / 0.03:.4g}_rate_of_g{rate_gamma:g}",
                  "mode": "branch", "source": "parent_B", "T": 100_000_000,
                  "config": _branch(0.03, 0.0125 * rate_gamma / 0.03)}
                 for rate_gamma in (0.04, 0.05, 0.07, 0.09)],
    }, {
        "group": "q7_threshold_p0",
        "question": "Q7: is the loss without short cycles due to the rejection threshold or the "
                    "full dynamics? Paper Fig. 6 (quasistatic): removing 2-4 raised diversity; "
                    "removing 2-5 sent gamma = 0.05 to one species, gamma = 0.025 stayed above.",
        "design": "fresh from one species, p = 0, alpha = 0.0125, 50M; threshold none/4/5",
        "runs": [{"label": f"g{gamma:g}_c{limit or 'all'}", "mode": "fresh", "T": 50_000_000,
                  "config": _fresh(0.0, gamma, max_forbidden_cycle_length=limit,
                                   max_cycle_rejection_attempts=10_000_000)}
                 for gamma in (0.025, 0.05) for limit in (None, 4, 5)],
    }, {
        "group": "d3_all_cycles_redo",
        "question": "D3 redo: does blocking raise diversity with all cycles allowed, under the "
                    "largest-cluster rule? (The September runs counted pocket species.)",
        "design": "fresh from one species, alpha = 0.0125, all cycles allowed, 50M; "
                  "gamma = 0.07 at p = 0 is the existing gamma_07_alpha_0125 run",
        "runs": [{"label": f"g{gamma:g}_p{p:g}", "mode": "fresh", "T": 50_000_000,
                  "config": _fresh(p, gamma, **ALL_CYCLES)}
                 for gamma, p in ((0.15, 0.0), (0.07, 0.1), (0.07, 0.2), (0.07, 0.3),
                                  (0.15, 0.1), (0.15, 0.2), (0.15, 0.3))],
    }, {
        "group": "q4_alpha_control_from_A",
        "question": "Q4/D7: the missing unchanged control of the alpha sweep.",
        "design": "two sibling branches of parent A at 50M with alpha = 0.0125, 100M each",
        "runs": [{"label": f"g0.03_a0.0125_s{index}", "mode": "branch", "source": "parent_A",
                  "T": 100_000_000, "config": _branch(0.03, 0.0125)} for index in (1, 2)],
    }, {
        "group": "q4_low_alpha_from_B",
        "question": "Q4: does low-alpha persistence hold for a second parent (B, p = 0.3)?",
        "design": "branches of parent B at 50M, 100M each",
        "runs": [{"label": f"g0.03_a{alpha:g}", "mode": "branch", "source": "parent_B",
                  "T": 100_000_000, "config": _branch(0.03, alpha)} for alpha in (0.001, 0.002)],
    }, {
        "group": "q3_p_ladder_g003",
        "question": "Q3/Q2/D6: is there an optimum in p, and does D6 replicate? p = 0 with short "
                    "cycles rejected has never been run and is Q2's baseline.",
        "design": "fresh from one species, gamma = 0.03, alpha = 0.0125, cycles 2-6 rejected, "
                  "100M; two independent seeds (own lattices) per p. Fresh p = 0.2 runs took "
                  "28-93M to first rise, so record the time to first rise.",
        "runs": [{"label": f"p{p:g}_s{index}", "mode": "fresh", "T": 100_000_000,
                  "config": _fresh(p, 0.03, **NO_SHORT)}
                 for p in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4) for index in (1, 2)],
    }]
    return groups, SOURCES


def _write_json(path, data):
    """Atomically replace a JSON file. Returns False when a reader held it open."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp",
                                             dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=MS._batch_json_default)
            handle.write("\n")
        for _ in range(20):
            try:
                os.replace(temporary, path)
                return True
            except PermissionError:
                time.sleep(0.25)
        return False
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _fingerprint(state):
    return MS._branch_state_fingerprint(MS._normalize_initial_state(state))


def _source_path(root, plan, name):
    """The original source checkpoint if it still exists, otherwise the campaign's copy."""
    source = plan["sources"][name]
    original = Path(source["original"])
    return original if original.is_file() else Path(root) / source["copy"]


def _load_source(root, plan, name):
    path = _source_path(root, plan, name)
    state = MS.load_checkpoint(path)
    expected = plan["sources"][name]["fingerprint"]
    if _fingerprint(state) != expected:
        raise ValueError(f"source {name} at {path} does not match its recorded fingerprint")
    return state


def materialize(groups, sources, seed_root=SEED_ROOT):
    """Give every run its folder, seed root, start and target, and check the sources."""
    used = {name for group in groups for name in
            (run.get("source") for run in group["runs"]) if name}
    source_records = {}
    for name in sorted(used):
        spec = sources[name]
        state = MS.load_checkpoint(spec["path"])
        fingerprint = _fingerprint(state)
        if spec.get("fingerprint") and fingerprint != spec["fingerprint"]:
            raise ValueError(f"source {name} has fingerprint {fingerprint}, "
                             f"expected {spec['fingerprint']}")
        source_records[name] = {
            "original": str(Path(spec["path"]).resolve()),
            "copy": f"initial_states/{name}.npz", "fingerprint": fingerprint,
            "timestep": int(state["timestep"]), "species": len(state["current_species"]),
            "state": {key: state.get(key) for key in (
                "gamma", "alpha", "p", "effective_p", "usable_sites", "track_every",
                "populate_first_100", "max_forbidden_cycle_length",
                "max_cycle_rejection_attempts")},
            "note": spec.get("note", ""),
        }
    runs = []
    for group in groups:
        for run in group["runs"]:
            index = len(runs)
            record = {"index": index, "id": f"{group['group']}/{run['label']}",
                      "group": group["group"], "label": run["label"], "mode": run["mode"],
                      "source": run.get("source"), "T": int(run["T"]), "kind": "percolation"}
            if run["mode"] == "continue":
                source = source_records[run["source"]]
                saved = source["state"]
                record["config"] = {
                    "L_col": 200, "L_row": 200, "D": 1, "gamma": saved["gamma"],
                    "alpha": saved["alpha"], "p": saved["p"], "track_every": saved["track_every"],
                    "populate_first_100": bool(saved["populate_first_100"]),
                    "largest_cluster_only": True,
                    "max_forbidden_cycle_length": saved["max_forbidden_cycle_length"],
                    "max_cycle_rejection_attempts": saved["max_cycle_rejection_attempts"],
                }
                record["base_seed"] = None  # The saved random state continues.
                start = source["timestep"]
            else:
                record["config"] = dict(run["config"])
                record["base_seed"] = seed_root * 1000 + index
                start = 0 if run["mode"] == "fresh" else source_records[run["source"]]["timestep"]
                if run["mode"] == "branch":
                    saved = source_records[run["source"]]["state"]
                    record["config"].setdefault("L_col", 200)
                    record["config"].setdefault("L_row", 200)
                    record["config"].setdefault("D", 1)
                    record["config"]["p"] = saved["p"]
            record["start_timestep"] = int(start)
            record["target_timestep"] = int(start + run["T"])
            runs.append(record)
    if len({run["id"] for run in runs}) != len(runs):
        raise ValueError("run labels must be unique within a group")
    # Check every configuration as the batch API would, before anything is written.
    for run in runs:
        config = MS.SimulationConfig(**run["config"], T=run["T"])
        MS._validate_batch_config(config, "percolation", None if run["mode"] == "fresh" else
                                  MS._normalize_initial_state(
                                      MS.load_checkpoint(sources[run["source"]]["path"])))
    return {"groups": [{key: group[key] for key in ("group", "question", "design")}
                       for group in groups],
            "sources": source_records, "runs": runs, "seed_root": seed_root}


def create_campaign(root, groups, sources, *, seed_root=SEED_ROOT, slots=16):
    root = Path(root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"choose a new campaign folder; already exists: {root}")
    plan = materialize(groups, sources, seed_root)
    root.mkdir(parents=True)
    (root / "initial_states").mkdir()
    (root / "logs").mkdir()
    for name, source in plan["sources"].items():
        shutil.copy2(source["original"], root / source["copy"])
    plan.update({"format_version": 1, "created": datetime.now().isoformat(timespec="seconds"),
                 "project_root": str(PROJECT_ROOT), "python": sys.executable,
                 "default_slots": slots,
                 "layout": "each run's folder is <group>/<label>: a one-job batch "
                           "(batch_plan.json + job_00000_seed_*) or, for mode 'continue', "
                           "a run folder whose history starts at the source run's start"})
    _write_json(root / PLAN, plan)
    return root, plan


# ----------------------------------------------------------------------------- one run


def run_folder(root, run):
    return Path(root) / run["folder"] if "folder" in run else Path(root) / run["id"]


def newest_timestep(root, run):
    """Newest checkpoint time of a run, from file names only (never opens a running run's files)."""
    folder = run_folder(root, run)
    if run["mode"] != "continue":
        jobs = [entry.path for entry in os.scandir(folder)
                if entry.is_dir() and entry.name.startswith("job_")] if folder.is_dir() else []
        folder = Path(jobs[0]) if jobs else None
    if folder is None or not folder.is_dir():
        return None
    times = [MS._checkpoint_timestep(Path(entry.name)) for entry in os.scandir(folder)
             if entry.name.startswith("checkpoint_") and entry.name.endswith(".npz")]
    times = [value for value in times if value >= 0]
    return max(times) if times else None


def run_child(root, run_id):
    """Run, continue or finish one run of the campaign, in this process."""
    root = Path(root).resolve()
    plan = _read_json(root / PLAN)
    run = next(item for item in plan["runs"] if item["id"] == run_id)
    folder = run_folder(root, run)
    config = MS.SimulationConfig(**run["config"], T=run["T"])
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} run {run_id} ({run['mode']}) in {folder}",
          flush=True)
    if run["mode"] == "continue":
        config.checkpoint_dir = str(folder)
        newest = newest_timestep(root, run)
        if newest is None:
            state = _load_source(root, plan, run["source"])
            result = config.run_main(initial_state=state, continue_history=True)
        else:
            analysis = MS.load_analysis(folder)
            if newest == run["target_timestep"] and analysis is not None and int(
                    analysis["timestep"]) == run["target_timestep"]:
                print("already complete", flush=True)
                return 0
            result = config.resume(folder)
    else:
        state = None if run["mode"] == "fresh" else _load_source(root, plan, run["source"])
        if (folder / "batch_plan.json").exists():
            result = MS.resume_simulations(folder, max_workers=1, initial_state=state)[0]
        else:
            result = MS.run_simulations([config], kind=run["kind"], repeats=1,
                                        base_seed=run["base_seed"], max_workers=1,
                                        checkpoint_root=folder, initial_state=state)[0]
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} finished at t = {int(result['timestep']):,} "
          f"with {len(result['current_species'])} living species", flush=True)
    return 0


# ----------------------------------------------------------------------------- scheduler


class Scheduler:
    def __init__(self, root, slots, *, min_free_gb=20.0, max_attempts=3, poll_seconds=20.0,
                 status_every=300.0, retry_delay=60.0):
        self.root = Path(root).resolve()
        self.plan = _read_json(self.root / PLAN)
        self.slots = slots
        self.min_free_gb = min_free_gb
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self.status_every = status_every
        self.retry_delay = retry_delay
        self.children = {}
        self.records = {run["id"]: {"state": "pending", "attempts": 0, "retry_after": 0.0}
                        for run in self.plan["runs"]}
        self.started = time.time()
        self._last_status = 0.0
        self._disk_warned = False

    def log(self, message):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
        if sys.stdout is not None:  # A detached scheduler may have no console.
            try:
                print(line, flush=True)
            except OSError:
                pass
        with (self.root / LOG).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def current_slots(self):
        path = self.root / SLOTS_FILE
        try:
            value = int(path.read_text(encoding="utf-8").strip())
            if value >= 0:
                return value
        except (OSError, ValueError):
            pass
        return self.slots

    def launch(self, run):
        record = self.records[run["id"]]
        record["attempts"] += 1
        log_path = self.root / "logs" / (run["id"].replace("/", "__") + ".log")
        handle = log_path.open("a", encoding="utf-8")
        flags = (BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW) if os.name == "nt" else 0
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(self.root), run["id"]],
            cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=flags)
        newest = newest_timestep(self.root, run)
        self.children[run["id"]] = (process, handle)
        record.update({"state": "running", "pid": process.pid,
                       "launched": datetime.now().isoformat(timespec="seconds"),
                       "launched_at": time.time(),
                       "timestep_at_launch": newest if newest is not None
                       else run["start_timestep"]})
        self.log(f"start {run['id']} (attempt {record['attempts']}, pid {process.pid})")

    def reap(self):
        for run_id, (process, handle) in list(self.children.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del self.children[run_id]
            record = self.records[run_id]
            record["exit_code"] = code
            if code == 0:
                record["state"] = "done"
                record["finished"] = datetime.now().isoformat(timespec="seconds")
                self.log(f"done  {run_id}")
            elif record["attempts"] < self.max_attempts:
                record["state"] = "pending"
                record["retry_after"] = time.time() + self.retry_delay
                self.log(f"exit {code} from {run_id}; restarting from its newest checkpoint "
                         f"in {self.retry_delay:.0f} s (see logs/)")
            else:
                record["state"] = "failed"
                self.log(f"FAILED {run_id} after {record['attempts']} attempts (see logs/)")

    def stop_children(self):
        for run_id, (process, handle) in list(self.children.items()):
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True)
                else:
                    process.kill()
                process.wait(timeout=30)
            handle.close()
            self.records[run_id]["state"] = "stopped"
            self.log(f"stopped {run_id}")
        self.children.clear()

    def write_status(self, final=False):
        now = time.time()
        if not final and now - self._last_status < self.status_every:
            return
        self._last_status = now
        rows = []
        for run in self.plan["runs"]:
            record = self.records[run["id"]]
            newest = newest_timestep(self.root, run)
            row = {"id": run["id"], "state": record["state"], "attempts": record["attempts"],
                   "start": run["start_timestep"], "target": run["target_timestep"],
                   "newest_checkpoint": newest}
            if record["state"] == "running" and newest is not None:
                hours = (now - record["launched_at"]) / 3600
                steps = newest - record["timestep_at_launch"]
                if hours > 0.05 and steps > 0:
                    row["steps_per_hour"] = round(steps / hours)
            for key in ("pid", "launched", "finished", "exit_code"):
                if key in record:
                    row[key] = record[key]
            rows.append(row)
        counts = {}
        for row in rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        _write_json(self.root / STATUS, {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "scheduler_pid": os.getpid(), "slots": self.current_slots(), "counts": counts,
            "runs": rows})

    def run(self):
        (self.root / STOP_FILE).unlink(missing_ok=True)
        self.log(f"scheduler pid {os.getpid()}: {len(self.plan['runs'])} runs, "
                 f"{self.slots} slots, campaign {self.root}")
        try:
            while True:
                if (self.root / STOP_FILE).exists():
                    self.log("STOP file found; stopping all runs")
                    self.stop_children()
                    self.write_status(final=True)
                    return 130
                self.reap()
                waiting = [run for run in self.plan["runs"]
                           if self.records[run["id"]]["state"] == "pending"]
                free_gb = shutil.disk_usage(self.root).free / 1e9
                for run in waiting:
                    if len(self.children) >= self.current_slots():
                        break
                    if free_gb < self.min_free_gb:
                        if not self._disk_warned:
                            self.log(f"only {free_gb:.1f} GB free; no new runs start below "
                                     f"{self.min_free_gb} GB")
                            self._disk_warned = True
                        break
                    if self.records[run["id"]]["retry_after"] <= time.time():
                        self.launch(run)
                self.write_status()
                if not self.children and not waiting:
                    break
                time.sleep(self.poll_seconds)
        except BaseException:
            self.log("scheduler interrupted; stopping all runs\n" + traceback.format_exc())
            self.stop_children()
            self.write_status(final=True)
            raise
        self.write_status(final=True)
        failed = [run_id for run_id, record in self.records.items()
                  if record["state"] == "failed"]
        self.log(f"campaign finished: {len(self.records) - len(failed)} done, "
                 f"{len(failed)} failed")
        return 1 if failed else 0


def run_scheduler(root, slots, **options):
    root = Path(root).resolve()
    try:
        with MS._simulation_directory_lock(root, filename=".campaign.lock"):
            return Scheduler(root, slots, **options).run()
    except Exception:
        with (root / LOG).open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} scheduler error\n"
                         + traceback.format_exc())
        raise


# ----------------------------------------------------------------------------- commands


def detach(argv):
    """Start this tool again through WMI, outside this process tree, and return its pid."""
    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *argv])
    if "'" in command or "'" in str(PROJECT_ROOT):
        raise ValueError("paths with single quotes cannot be detached")
    script = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
              f"@{{CommandLine='{command}'; CurrentDirectory='{PROJECT_ROOT}'}}; "
              "Write-Output \"$($r.ReturnValue) $($r.ProcessId)\"")
    output = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                            capture_output=True, text=True, check=True).stdout.split()
    if not output or output[0] != "0":
        raise RuntimeError(f"Win32_Process.Create failed: {output}")
    return int(output[1])


def print_plan(plan, slots):
    print(f"{'#':>3} {'run':<58} {'mode':<8} {'from':<16} {'gamma':>6} {'alpha':>9} "
          f"{'p':>5} {'cyc':>4} {'steps':>12}")
    for run in plan["runs"]:
        config = run["config"]
        limit = config.get("max_forbidden_cycle_length")
        print(f"{run['index']:>3} {run['id']:<58} {run['mode']:<8} {run['source'] or '-':<16} "
              f"{config['gamma']:>6g} {config['alpha']:>9.5g} {config['p']:>5g} "
              f"{limit if limit else 'all':>4} "
              f"{run['start_timestep'] / 1e6:>5.0f}-{run['target_timestep'] / 1e6:<4.0f}M")
    total = sum(run["T"] for run in plan["runs"])
    print(f"\n{len(plan['runs'])} runs, {total / 1e6:,.0f}M steps in all: about "
          f"{total / (STEPS_PER_HOUR_PER_RUN * slots):.0f} h at {slots} slots "
          f"({STEPS_PER_HOUR_PER_RUN / 1e6:g}M steps/h per run).")
    for name, source in plan["sources"].items():
        print(f"source {name}: t = {source['timestep'] / 1e6:g}M, {source['species']} species, "
              f"fingerprint {source['fingerprint'][:12]}  ({source['original']})")


def print_status(root):
    root = Path(root).resolve()
    plan = _read_json(root / PLAN)
    status = {}
    if (root / STATUS).exists():
        status = {row["id"]: row for row in _read_json(root / STATUS)["runs"]}
    counts = {}
    for run in plan["runs"]:
        row = status.get(run["id"], {})
        state = row.get("state", "pending")
        counts[state] = counts.get(state, 0) + 1
        newest = newest_timestep(root, run)
        done = 0 if newest is None else (newest - run["start_timestep"]) / run["T"]
        rate = row.get("steps_per_hour")
        print(f"{run['id']:<58} {state:<8} {done:>6.1%}  "
              f"t = {(newest or run['start_timestep']) / 1e6:7.2f}M"
              + (f"  {rate / 1e6:.2f}M/h" if rate else ""))
    print(counts)
    tail = (root / LOG).read_text(encoding="utf-8").splitlines()[-5:] if (
        root / LOG).exists() else []
    print("\n".join(tail))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", type=Path, help="run an existing campaign's remaining work")
    action.add_argument("--status", type=Path, help="print a campaign's progress")
    action.add_argument("--stop", type=Path, help="stop a running campaign's scheduler and runs")
    action.add_argument("--child", nargs=2, metavar=("ROOT", "RUN_ID"), help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path,
                        help=f"new campaign folder (default: a fresh folder in {DEFAULT_PARENT})")
    parser.add_argument("--slots", type=int, default=16,
                        help="runs at once (default 16: one per physical core on this PC)")
    parser.add_argument("--min-free-gb", type=float, default=20.0,
                        help="start no new run when the drive has less free space")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--detach", action="store_true",
                        help="run the scheduler outside this terminal's process tree")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return run_child(*args.child)
    if args.status:
        print_status(args.status)
        return 0
    if args.stop:
        (args.stop / STOP_FILE).write_text("stop\n", encoding="utf-8")
        print(f"Asked the scheduler of {args.stop} to stop; it does so within a minute. "
              f"Continue later with --resume.")
        return 0
    options = ["--slots", str(args.slots), "--min-free-gb", str(args.min_free_gb),
               "--max-attempts", str(args.max_attempts)]
    if args.resume:
        root = args.resume.expanduser().resolve()
        if not (root / PLAN).exists():
            raise FileNotFoundError(f"no {PLAN} in {root}")
        if args.dry_run:
            print_plan(_read_json(root / PLAN), args.slots)
            return 0
    else:
        groups, sources = campaign_groups()
        if args.dry_run:
            plan = materialize(groups, sources)
            print_plan(plan, args.slots)
            print(f"\nFree on {DEFAULT_PARENT.drive}: "
                  f"{shutil.disk_usage(DEFAULT_PARENT).free / 1e9:.0f} GB. "
                  "Dry run: nothing written.")
            return 0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = args.output_root or DEFAULT_PARENT / f"campaign_{stamp}_{uuid4().hex[:8]}"
        root, plan = create_campaign(root, groups, sources, slots=args.slots)
        print(f"Campaign created: {root} ({len(plan['runs'])} runs)")
    if args.detach:
        pid = detach(["--resume", str(root), *options])
        print(f"Scheduler started outside this terminal (pid {pid}).\n"
              f"Status: python tools/run_campaign.py --status \"{root}\"\n"
              f"Stop:   python tools/run_campaign.py --stop \"{root}\"")
        return 0
    try:
        return run_scheduler(root, args.slots, min_free_gb=args.min_free_gb,
                             max_attempts=args.max_attempts)
    except KeyboardInterrupt:
        print(f'Stopped. Resume: python tools/run_campaign.py --resume "{root}"')
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
