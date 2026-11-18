"""Finished runs of the 9 Nov campaign and of the earlier batches, for campaign_analysis.ipynb.

Only campaign runs that campaign_status.json marks as done are opened. A running run rewrites
its analysis.npz and checkpoints with os.replace, which fails on Windows while another process
holds the file open, so the run would stop. Its folder is therefore never touched here.

High and low states follow Uekermann, Mathiesen & Mitarai, Phys. Rev. E 95, 032409 (2017),
section III B (articles/PhysRevE.95.032409.md): take the high (right-hand) mode M1 of the
distribution of the state measure; a run enters the high state above 0.5 M1 and leaves it below
0.1 M1. The paper used the frozen boundary length; our records hold the living species, so the
rule is applied to those. The distribution comes from all alike runs (same gamma, p and cycle
rule), pooled over alpha, seed and parent; the paper found the high mode insensitive to alpha.
"""

import json
from pathlib import Path, PureWindowsPath

from matplotlib.lines import Line2D
import numpy as np

import sim_modules as MS

CAMPAIGN = Path(r"E:\MasterProject\checkpoints\campaign_20261109_161426_f78322dc")
CHECKPOINTS = Path(r"D:\Luc\ActualMasterProject\MasterProject\checkpoints")
SMOOTH_SIGMA = 20  # Gaussian width in tracked samples (0.2M steps) before the lines are applied.
MARK_EVERY = 500  # Samples between the sparse identity markers on a line (5M steps).

ENTER, EXIT = 0.5, 0.1  # Into the high state above ENTER x M1, out of it below EXIT x M1.
KERNEL_SHARE = 0.12  # The species distribution is smoothed by 12% of the count (at least 1).
MIN_MODE_MASS = 0.05  # A mode holds at least 5% of its pool's samples,
MIN_MODE_DIP = 0.10  # and the density dips by at least 10% between it and its neighbour.

# Finished batches outside the campaign that join the pools of alike runs.
REFERENCE_BATCHES = (
    "no_short_cycles_p_gamma_20261009_011738_ef440b70",
    "no_short_cycles_perc05_20261008_012527_5832a1ea",
    "no_short_cycles_perc05_20261008_151903_7dc539e7",
    "gamma_sweep_perc05_largest_20261006_113609_5a0ff663",
    "no_short_cycles_alpha_p20_g003_20261021_011454_0f93dba1",
    "no_short_cycles_alpha_p20_g003_rep2_20261022_005553_dfacd28f",
    "alpha_p20_g003_fresh_rep3-5_analysis",
    "no_short_cycles_gamma_p30_from_g003_20261022_005553_5940261e",
)
# Early single runs on the full lattice, all cycles allowed, alpha = 0.0125 (folder: gamma).
SINGLE_RUNS = {"gamma_07_alpha_0125": 0.07, "gamma_075_alpha_0125": 0.075,
               "gamma_080_alpha_0125": 0.08, "gamma_090_alpha_0125": 0.09}

PARENT_COLOR = "#52514e"  # References, parents and controls.
BAND_COLOR = "#ecebe6"  # Between the exit and entry lines of the high state.
SURFACE = "#fcfcfb"
RESULT_STYLE = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.facecolor": SURFACE,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.titlelocation": "left",
    "xtick.color": PARENT_COLOR, "ytick.color": PARENT_COLOR,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white",
}


# ------------------------------------------------------------------ campaign files

def plan(root=CAMPAIGN):
    """The campaign plan: groups, sources and runs."""
    return json.loads((Path(root) / "campaign_plan.json").read_text(encoding="utf-8"))


def campaign(root=CAMPAIGN):
    """Every planned run by id, with its state (done, running, pending, failed), and the time
    of the status file."""
    root = Path(root)
    status = json.loads((root / "campaign_status.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in status["runs"]}
    runs = {}
    for run in plan(root)["runs"]:
        row = rows.get(run["id"], {})
        runs[run["id"]] = {**run, "state": row.get("state", "pending"),
                           "newest": row.get("newest_checkpoint")}
    return runs, status["updated"]


def batch_jobs(folder):
    """(config, job folder) of every job in a finished batch folder."""
    plan_ = json.loads((Path(folder) / "batch_plan.json").read_text(encoding="utf-8"))
    return [(job["config"], Path(folder) / PureWindowsPath(job["checkpoint_dir"]).name)
            for job in plan_["jobs"]]


def _record_folder(root, run):
    """The folder holding a campaign run's analysis.npz: the run folder for a continuation,
    its one job folder for a one-job batch."""
    folder = Path(root) / run["id"]
    return folder if run["mode"] == "continue" else batch_jobs(folder)[0][1]


# ------------------------------------------------------------------ the catalogue of runs

def pool_of(config):
    """The pool of alike runs: gamma, p and the longest rejected cycle (None: all allowed)."""
    return (round(float(config["gamma"]), 4), round(float(config.get("p") or 0.0), 3),
            config.get("max_forbidden_cycle_length"))


def _entry(config, folder, start, end, **labels):
    record = MS.load_analysis(folder)
    return {**labels, "config": config, "gamma": float(config["gamma"]),
            "alpha": float(config["alpha"]), "p": float(config.get("p") or 0.0),
            "rule": config.get("max_forbidden_cycle_length"), "pool": pool_of(config),
            "start": int(start), "end": int(end), "record": record, "folder": Path(folder),
            "usable_sites": int(record["usable_sites"]) if "usable_sites" in record else 40_000}


def catalogue(root=CAMPAIGN):
    """Every finished run: the done campaign runs and the reference runs, each over its own
    interval. Returns (runs, campaign runs not done yet, status time)."""
    runs = []
    for name in REFERENCE_BATCHES:
        folder = CHECKPOINTS / name
        plan_ = json.loads((folder / "batch_plan.json").read_text(encoding="utf-8"))
        start = int((plan_.get("initial_state") or {}).get("timestep", 0))
        for config, job in batch_jobs(folder):
            runs.append(_entry(config, job, start, start + int(config["T"]), batch=name,
                               label=job.name, group=None, id=f"{name}/{job.name}"))
    for name, gamma in SINGLE_RUNS.items():
        config = {"gamma": gamma, "alpha": 0.0125, "p": 0.0, "max_forbidden_cycle_length": None}
        entry = _entry(config, CHECKPOINTS / name, 0, 0, batch=name, label=name, group=None, id=name)
        entry["end"] = int(entry["record"]["tracked_timesteps"][-1])
        runs.append(entry)
    planned, updated = campaign(root)
    waiting = [run for run in planned.values() if run["state"] != "done"]
    for run in planned.values():
        if run["state"] == "done":
            runs.append(_entry(run["config"], _record_folder(root, run), run["start_timestep"],
                               run["target_timestep"], batch=run["group"], label=run["label"],
                               group=run["group"], id=run["id"], mode=run["mode"],
                               source=run.get("source")))
    return runs, waiting, updated


def select(runs, **where):
    """The runs whose fields equal every given value, e.g. select(runs, group="q7_threshold_p0")."""
    return [run for run in runs if all(run.get(key) == value for key, value in where.items())]


def _samples(run):
    times = np.asarray(run["record"]["tracked_timesteps"], dtype=np.int64)
    keep = (times >= run["start"]) & (times <= run["end"])
    return np.asarray(run["record"]["diversity_history"])[keep]


# ------------------------------------------------------------------ modes and state lines

def find_modes(samples):
    """Modes of a distribution of living species: (modes, grid, density).

    The counts are smoothed by a Gaussian of KERNEL_SHARE x the count (at least 1 species), so
    the narrow low mode and the broad high mode are both resolved. Peaks that hold less than
    MIN_MODE_MASS of the samples, or that dip less than MIN_MODE_DIP below their own height
    towards a neighbour, are merged into that neighbour, weakest first.
    """
    values, counts = np.unique(np.asarray(samples).astype(np.int64), return_counts=True)
    step = 0.25
    grid = np.arange(0.0, values.max() * 1.3 + 10.0, step)
    width = np.maximum(1.0, KERNEL_SHARE * values)
    kernels = np.exp(-0.5 * ((grid[None, :] - values[:, None]) / width[:, None]) ** 2)
    density = (counts / width) @ kernels
    density /= density.sum() * step
    inner = np.arange(1, grid.size - 1)
    peaks = list(inner[(density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:])])
    while True:
        valleys = [a + int(np.argmin(density[a:b + 1])) for a, b in zip(peaks, peaks[1:])]
        bounds = [0] + valleys + [grid.size - 1]
        masses = [float(density[bounds[i]:bounds[i + 1] + 1].sum() * step)
                  for i in range(len(peaks))]
        weakest, worst = None, None
        for i, peak in enumerate(peaks):
            adjacent = valleys[max(i - 1, 0):i + 1]
            if not adjacent:
                break
            dip = 1.0 - max(density[v] for v in adjacent) / density[peak]
            score = min(masses[i] / MIN_MODE_MASS, dip / MIN_MODE_DIP)
            if score < 1.0 and (worst is None or score < worst):
                weakest, worst = i, score
        if weakest is None:
            break
        peaks.pop(weakest)
    modes = [{"location": float(grid[peak]), "height": float(density[peak]), "mass": mass}
             for peak, mass in zip(peaks, masses)]
    return modes, grid, density


def state_lines(runs):
    """The entry and exit lines of the high state for every pool of alike runs.

    kind "own": the pool has a high mode M1 with its low mode below EXIT x M1 (in the paper the
    low mode sits at zero, below that line); the lines are ENTER x M1 and EXIT x M1.
    kind "borrowed": the lines of the nearest "own" pool with the same cycle rule, at the same p
    and nearest gamma, else at the same gamma and nearest p (the lower value on a tie).
    kind "low only": no such pool, and every mode lies below the exit line of every "own"
    pool, so any of them would call it low. kind "unresolved": none of these; the runs are not
    classified.
    """
    pools = {}
    for run in runs:
        pools.setdefault(run["pool"], []).append(run)
    info = {}
    for key, members in pools.items():
        samples = np.concatenate([_samples(run) for run in members])
        modes, grid, density = find_modes(samples)
        item = {"pool": key, "runs": len(members), "samples": int(samples.size), "modes": modes,
                "grid": grid, "density": density, "kind": None, "source": None}
        high = modes[-1]["location"]
        if len(modes) >= 2 and modes[0]["location"] < EXIT * high:
            item.update(kind="own", high_mode=high, source=key)
        info[key] = item
    own = {key: item for key, item in info.items() if item["kind"] == "own"}
    low_ceiling = min(EXIT * item["high_mode"] for item in own.values())
    for key, item in info.items():
        if item["kind"] is not None:
            continue
        gamma, p, rule = key
        same_p = [k for k in own if k[2] == rule and k[1] == p]
        same_gamma = [k for k in own if k[2] == rule and k[0] == gamma]
        if same_p:
            source = min(same_p, key=lambda k: (abs(k[0] - gamma), k[0]))
        elif same_gamma:
            source = min(same_gamma, key=lambda k: (abs(k[1] - p), k[1]))
        else:
            source = None
        if source is not None:
            item.update(kind="borrowed", high_mode=own[source]["high_mode"], source=source)
        elif max(mode["location"] for mode in item["modes"]) < low_ceiling:
            item["kind"] = "low only"
        else:
            item["kind"] = "unresolved"
    for item in info.values():
        if "high_mode" in item:
            item["enter"], item["exit"] = ENTER * item["high_mode"], EXIT * item["high_mode"]
    return info


def pool_name(key):
    gamma, p, rule = key
    return f"γ {gamma:g}, p {p:g}, " + ("all cycles" if rule is None else f"2–{rule} rejected")


# ------------------------------------------------------------------ per-run summaries

def _window_mean(times, values, from_timestep):
    """Trapezoidal time mean of the samples at or after from_timestep."""
    keep = times >= from_timestep
    times, values = times[keep].astype(np.float64), values[keep]
    if times.size < 2:
        return np.nan
    return float(np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2.0)
                 / (times[-1] - times[0]))


def _regimes(smoothed, start_high, enter, exit_):
    """True where a run is in the high state: in above `enter`, out below `exit_`, so a run
    between the two lines keeps its state. The start state comes from the raw first sample,
    because smoothing would blend an early collapse into the start."""
    high = np.empty(smoothed.size, dtype=bool)
    state = bool(start_high)
    for index, value in enumerate(smoothed):
        if state and value < exit_:
            state = False
        elif not state and value > enter:
            state = True
        high[index] = state
    return high


def summarise(record, start, length, lines=None, late_from=None):
    """Living species of one record over start -> start + length, with its state summary.

    ``lines`` is the pool entry from state_lines (None or "unresolved": no states). Means and
    shares cover t >= start + late_from (default: the second half of the window).
    """
    times = np.asarray(record["tracked_timesteps"], dtype=np.int64)
    keep = (times >= start) & (times <= start + length)
    times = times[keep]
    diversity = np.asarray(record["diversity_history"], dtype=np.float64)[keep]
    smoothed = MS._gaussian_smooth(diversity, SMOOTH_SIGMA)
    kind = (lines or {}).get("kind")
    if kind in ("own", "borrowed"):
        midpoint = 0.5 * (lines["enter"] + lines["exit"])
        high = _regimes(smoothed, diversity[0] >= midpoint, lines["enter"], lines["exit"])
    elif kind == "low only":
        high = np.zeros(times.size, dtype=bool)
    else:
        high = None
    late_start = start + (length // 2 if late_from is None else late_from)
    late = times >= late_start
    summary = {
        "times": times, "elapsed": (times - start) / 1e6, "diversity": diversity,
        "smoothed": smoothed, "high": high, "late_start": late_start, "lines": lines,
        "mean": _window_mean(times, diversity, late_start),
        "final": int(diversity[-1]), "end_time": int(times[-1]),
        "share_high": np.nan, "switches": None, "first_rise": None, "first_collapse": None,
    }
    if high is not None:
        switches = np.flatnonzero(high[1:] != high[:-1]) + 1
        elapsed, up = (times[switches] - start) / 1e6, high[switches]
        summary.update(
            share_high=float(np.mean(high[late])) if late.any() else np.nan,
            switches=int(switches.size),
            first_rise=float(elapsed[up][0]) if up.any() else None,
            first_collapse=float(elapsed[~up][0]) if (~up).any() else None)
    return summary


def classify(runs, info, start=None, length=None, late_from=None):
    """Summarise each run with its pool's lines, over its own interval unless start and length
    are given. The summary fields are added to (a copy of) each run."""
    out = []
    for run in runs:
        begin = run["start"] if start is None else start
        span = run["end"] - run["start"] if length is None else length
        out.append({**run, **summarise(run["record"], begin, span, info[run["pool"]], late_from)})
    return out


def load_done(groups, root=CAMPAIGN):
    """The done campaign runs of the given groups (no state lines). Returns (done runs, the
    planned runs that are not done yet, status time)."""
    planned, updated = campaign(root)
    done, waiting = [], []
    for run in planned.values():
        if run["group"] not in groups:
            continue
        if run["state"] != "done":
            waiting.append(run)
            continue
        record = MS.load_analysis(_record_folder(root, run))
        start = int(run["start_timestep"])
        done.append({**run, **summarise(record, start, int(run["target_timestep"]) - start),
                     "record": record})
    return done, waiting, updated


# ------------------------------------------------------------------ drawing helpers

def draw_trace(ax, run, color, marker, label, x=None, raw=True, lw=1.4, mark_every=MARK_EVERY):
    """Raw living species faintly, the smoothed line on top, and sparse identity markers."""
    x = run["elapsed"] if x is None else x
    if raw:
        ax.plot(x, run["diversity"], color=color, lw=0.4, alpha=0.25, zorder=2)
    ax.plot(x, run["smoothed"], color=color, lw=lw, zorder=3, label=label, marker=marker,
            markevery=mark_every, ms=5, mec="white", mew=0.6)


def state_band(ax, runs):
    """The exit-to-entry band of the high state. Runs sharing one pair of lines get one grey
    band; differing lines are drawn per run in its colour (dashed: entry, dotted: exit)."""
    pairs = {}
    for run, color in runs:
        lines = run["lines"] or {}
        if "enter" in lines:
            pairs.setdefault((lines["enter"], lines["exit"]), []).append(color)
    if len(pairs) == 1:
        (enter, exit_), = pairs
        ax.axhspan(exit_, enter, color=BAND_COLOR, lw=0, zorder=0)
        return
    for (enter, exit_), colors in pairs.items():
        ax.axhline(enter, color=colors[0], ls="--", lw=0.9, alpha=0.8, zorder=1)
        ax.axhline(exit_, color=colors[0], ls=":", lw=0.9, alpha=0.8, zorder=1)


def handle(color, marker, label, ls="-"):
    """A legend entry: line plus marker."""
    return Line2D([], [], color=color, marker=marker, ms=6, mec="white", mew=0.6, lw=1.4,
                  ls=ls, label=label)


def note(ax, lines, y=0.96):
    """Small grey text at the right of a panel (top by default)."""
    if lines:
        ax.text(0.98, y, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=PARENT_COLOR)
