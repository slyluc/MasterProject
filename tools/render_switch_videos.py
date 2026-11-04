"""Render MP4 videos of a switch capture's lattice.

Two kinds of video:
- Sampled (default): the rise and collapse windows of analysis/switch_analysis.ipynb
  (switch_features.switch_points), padded on both sides, one frame per kept state (every 1,000
  steps by default).
- Every step (--every-step): --frames consecutive steps centred on each given time. The run is
  replayed in memory from the saved state before the window. The saved states hold the random
  number state, so the replay is the run itself; each replayed step that lands on a saved state
  is checked against it, and the render stops if one ever differs. Only the video is written.

Each frame shows the lattice, living species and patches with a cursor at the frame's time, the
frame's numbers, and a map of the sites a neighbour could invade at that moment and of the
youngest species.

Needs imageio-ffmpeg for its bundled ffmpeg: pip install imageio-ffmpeg

Sampled:     python tools/render_switch_videos.py
Every step:  python tools/render_switch_videos.py --every-step 153.9515 154.5 155
Test:        python tools/render_switch_videos.py --every-step 154.5 --max-frames 60
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sim_modules as MS  # noqa: E402
from analysis import switch_features  # noqa: E402


DEFAULT_CAPTURE = Path(
    r"E:\MasterProject\checkpoints\switches_p0.3_a0.0125_g0.03_20261029_150918_cd0568f6")
SIZE = (1600, 900)  # Pixels; even sides, as H.264 with 4:2:0 colour needs.
SMOOTH = 200  # Samples, as in the notebook: 0.2M steps.
# The notebook's colours and surfaces, so the video reads like its figures.
SPECIES_COLOR, PATCH_COLOR = "#6250d6", "#eb6834"
INK, BAND_COLOR, SURFACE = "#52514e", "#ecebe6", "#fcfcfb"
# Activity map: red and blue are the notebook's validated collapse and rise pair.
INVADABLE_COLOR = np.array([0.89, 0.29, 0.28])  # #e34948
NEWCOMER_COLOR = np.array([0.165, 0.47, 0.84])  # #2a78d6
USABLE_COLOR = np.array([0.88, 0.88, 0.86])
STATE_KEYS = ("lattice", "Gamma", "current_species", "newest_species", "timestep", "rng_state")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--window", choices=("rise", "collapse", "both"), default="both",
                        help="sampled videos to render (default: both)")
    parser.add_argument("--every-step", type=float, nargs="+", metavar="MILLIONS",
                        help="render every step around these times instead, e.g. 154.5")
    parser.add_argument("--frames", type=int, default=7200,
                        help="steps per every-step video (default: 7,200)")
    parser.add_argument("--fps", type=int,
                        help="frames per second (default: 30 sampled, 60 every step)")
    parser.add_argument("--before", type=float, default=0.3,
                        help="millions of steps shown before a sampled window (default: 0.3)")
    parser.add_argument("--after", type=float, default=0.2,
                        help="millions of steps shown after a sampled window (default: 0.2)")
    parser.add_argument("--out-dir", type=Path, help="default: the capture's videos/ folder")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames, for a test")
    parser.add_argument("--min-free-gb", type=float, default=20.0,
                        help="refuse to start, or stop, below this much free space (default: 20)")
    return parser.parse_args(argv)


def site_rows(lattice, species):
    """Each site's row in Gamma (the order of current_species), -1 where no species lives."""
    order = np.argsort(species)
    occupied = lattice > 0
    row = np.full(lattice.shape, -1, dtype=np.int64)
    row[occupied] = order[np.searchsorted(species[order], lattice[occupied])]
    return row


def invadable_sites(row, Gamma):
    """Sites that a neighbouring species could take over right now."""
    target = np.zeros(row.shape, dtype=bool)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        neighbour = np.roll(row, shift, axis=axis)  # Periodic, as in the model.
        pair = (row >= 0) & (neighbour >= 0) & (neighbour != row)
        target[pair] |= Gamma[neighbour[pair], row[pair]]
    return target


def species_color(species):
    """The colour sim_modules._lattice_rgba gives a species ID."""
    return plt.colormaps["hsv"](np.remainder(species * 0.61803398875, 1.0))


class Capture:
    """Series, per-state measures and state files of one capture."""

    def __init__(self, root):
        self.root = Path(root)
        self.plan = json.loads((self.root / "capture_plan.json").read_text(encoding="utf-8"))
        self.track = self.plan["track_every"]
        with np.load(self.root / "series.npz") as saved:
            self.times = saved["timesteps"]
            self.diversity = saved["living_species"].astype(np.float64)
            self.patches = saved["patches"].astype(np.float64)
        self.files = switch_features.state_files(self.root)
        self.features = switch_features.load_features(self.root)
        if not np.array_equal(self.features["timestep"], self.times):
            raise ValueError("features.npz does not match series.npz; delete it to rebuild")
        self.points = switch_features.switch_points(self.times, self.diversity,
                                                    self.plan["regime_rule"], smooth=SMOOTH)
        self.smooth_d = MS._gaussian_smooth(self.diversity, SMOOTH)
        self.smooth_p = MS._gaussian_smooth(self.patches, SMOOTH)
        self.first_new = int(self.features["introduced"][0]) + 1

    def sample(self, timestep):
        """Index of the last kept state at or before a timestep."""
        return int(np.searchsorted(self.times, timestep, side="right")) - 1

    def arrivals(self, species):
        """Latest possible arrival time of each species, at the kept states' spacing."""
        index = np.searchsorted(self.features["introduced"], species)
        known = (np.asarray(species) >= self.first_new) & (index < self.times.size)
        return np.where(known, self.times[np.minimum(index, self.times.size - 1)], -1)

    def regime(self, timestep):
        # The regime follows the rule's switches, which have hysteresis.
        index = self.sample(timestep)
        return ("high regime" if self.points["rise"] <= index < self.points["collapse"]
                else "low regime")

    def load(self, index):
        with np.load(self.files[index]) as saved:
            return {"lattice": saved["lattice"], "Gamma": saved["Gamma"],
                    "current_species": saved["current_species"].astype(np.int64),
                    "newest_species": int(saved["newest_species"]),
                    "timestep": int(saved["timestep"]),
                    "rng_state": json.loads(str(saved["rng_state"].item()))}


def replay(capture, first, last):
    """Every state from timestep first to last, replayed from the kept state before first.

    Each replayed state that lands on a kept state must equal it exactly, including the random
    number state; the replay then continues to the next kept state after last to check the end.
    """
    settings = capture.plan["simulation"]
    # Start strictly before first, so a window that begins on a kept state still shows it.
    state = capture.load(capture.sample(first - 1))
    end_check = capture.times[min(capture.sample(last - 1) + 1, capture.times.size - 1)]
    while state["timestep"] < max(last, end_check):
        result = MS.simulation_from_state(
            state, gamma=settings["gamma"], alpha=settings["alpha"], T=1, track_every=1,
            max_forbidden_cycle_length=settings["max_forbidden_cycle_length"],
            max_cycle_rejection_attempts=settings["max_cycle_rejection_attempts"])
        state = {key: result[key] for key in STATE_KEYS}
        state["current_species"] = np.asarray(state["current_species"], dtype=np.int64)
        if (state["timestep"] - capture.times[0]) % capture.track == 0:
            saved = capture.load(capture.sample(state["timestep"]))
            if not (np.array_equal(state["lattice"], saved["lattice"])
                    and np.array_equal(state["Gamma"], saved["Gamma"])
                    and np.array_equal(state["current_species"], saved["current_species"])
                    and state["newest_species"] == saved["newest_species"]
                    and state["rng_state"] == saved["rng_state"]):
                raise RuntimeError(f"replay differs from the kept state at t = "
                                   f"{state['timestep']:,}; stopping")
        if first <= state["timestep"] < last:
            yield state


def build_figure(capture, title, subtitle, trace, newcomer_age):
    """The static parts of a frame, and the artists each frame updates."""
    fig = plt.figure(figsize=(SIZE[0] / 100, SIZE[1] / 100), dpi=100, facecolor="white")
    side = 0.94 * SIZE[1] / SIZE[0]  # Square lattice in figure-width units.
    # Images start at the lattice's shape, so their extent fits every later frame.
    blank = np.zeros((*capture.plan["simulation"]["lattice_shape"], 3))
    ax_lattice = fig.add_axes([0.012, 0.03, side, 0.94])
    ax_lattice.set_axis_off()
    image = ax_lattice.imshow(blank, interpolation="nearest")
    right = 0.012 + side + 0.035
    fig.text(right, 0.965, title, fontsize=15, fontweight="bold", color="#1a1a19", va="top")
    fig.text(right, 0.925, subtitle, fontsize=10, color=INK, va="top")

    ax_trace = fig.add_axes([right + 0.04, 0.575, 0.93 - right - 0.04, 0.29])
    ax_trace.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax_trace.spines[spine].set_visible(False)
    twin = ax_trace.twinx()
    twin.spines["top"].set_visible(False)
    rule = capture.plan["regime_rule"]
    ax_trace.axhspan(rule["low"], rule["high"], color=BAND_COLOR, lw=0, zorder=0)
    x = trace["x"]
    for axis, raw, smooth, color, ls in (
            (ax_trace, trace["species"], trace.get("species_smooth"), SPECIES_COLOR, "-"),
            (twin, trace["patches"], trace.get("patches_smooth"), PATCH_COLOR, "--")):
        if smooth is None:
            axis.plot(x, raw, color=color, lw=1.2, ls=ls)
        else:
            axis.plot(x, raw, color=color, lw=0.5, alpha=0.35)
            axis.plot(x, smooth, color=color, lw=1.8, ls=ls)
        axis.set_ylim(0, 1.05 * np.max(raw))
        axis.tick_params(labelsize=9, colors=INK)
    for label, position in trace.get("marks", {}).items():
        ax_trace.axvline(position, color=INK, lw=0.8, ls=":")
        ax_trace.text(position, 1.02, label, transform=ax_trace.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
    cursor = ax_trace.axvline(x[0], color="#1a1a19", lw=1.4)
    ax_trace.set_xlim(x[0], x[-1])
    ax_trace.ticklabel_format(axis="x", useOffset=False)  # Plain times on short windows too.
    ax_trace.set_xlabel(trace["xlabel"], fontsize=9, color=INK)
    ax_trace.set_ylabel("Living species (solid)", fontsize=9, color=INK)
    twin.set_ylabel("Patches (dashed)", fontsize=9, color=INK)
    ax_trace.grid(True, color="#e1e0d9", lw=0.6)

    info = fig.text(right, 0.475, "", fontsize=12, color="#1a1a19", va="top", linespacing=1.55)
    swatch = fig.add_axes([right, 0.335, 0.012, 0.022])
    swatch.set_axis_off()
    swatch_patch = swatch.add_patch(plt.Rectangle((0, 0), 1, 1, color="white"))
    dominant_text = fig.text(right + 0.017, 0.346, "", fontsize=12, color="#1a1a19", va="center")

    map_height = 0.29
    map_width = map_height * SIZE[1] / SIZE[0]
    ax_map = fig.add_axes([right, 0.03, map_width, map_height])
    ax_map.set_axis_off()
    activity = ax_map.imshow(blank, interpolation="nearest")
    legend_x = right + map_width + 0.012
    fig.text(legend_x, 0.03 + map_height, "Where things happen", fontsize=11, fontweight="bold",
             color="#1a1a19", va="top")
    for row, (color, label) in enumerate(((INVADABLE_COLOR, "a neighbour can invade it now"),
                                          (NEWCOMER_COLOR, f"species younger than "
                                                           f"{newcomer_age:,} steps"),
                                          (USABLE_COLOR, "other sites"))):
        y = 0.03 + map_height - 0.05 - row * 0.035
        key = fig.add_axes([legend_x, y - 0.009, 0.01, 0.018])
        key.set_axis_off()
        key.add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        fig.text(legend_x + 0.015, y, label, fontsize=10.5, color="#1a1a19", va="center")
    map_text = fig.text(legend_x, 0.03 + map_height - 0.17, "", fontsize=10.5, color="#1a1a19",
                        va="top", linespacing=1.5)
    # These change every frame; the rest is drawn once and restored (blitting).
    moving = {"image": image, "cursor": cursor, "info": info, "swatch": swatch_patch,
              "dominant": dominant_text, "activity": activity, "map_text": map_text}
    for artist in moving.values():
        artist.set_animated(True)
    if fig.canvas.get_width_height() != SIZE:
        raise RuntimeError(f"frame is {fig.canvas.get_width_height()}, expected {SIZE}")
    return fig, moving


def encode(fig, art, frames, out, fps, newcomer_age, min_free_gb):
    """Draw each frame description into the figure and pipe it to ffmpeg; return the count.

    A frame description holds the state (lattice, Gamma, current_species), the cursor position,
    the information text, and the arrival time of each living species.
    """
    import imageio_ffmpeg

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".partial.mp4")
    # One keyframe per second keeps seeking quick in a player.
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-g", str(fps), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    count = 0
    try:
        for frame in frames:
            lattice, Gamma = frame["lattice"], np.asarray(frame["Gamma"]).astype(bool)
            species = np.asarray(frame["current_species"], dtype=np.int64)
            art["image"].set_data(MS._lattice_rgba(lattice)[..., :3])
            row = site_rows(lattice, species)
            invadable = invadable_sites(row, Gamma)
            newcomer = frame["arrivals"] > frame["timestep"] - newcomer_age
            colours = np.zeros(lattice.shape + (3,))
            colours[row >= 0] = USABLE_COLOR
            colours[(row >= 0) & newcomer[np.maximum(row, 0)]] = NEWCOMER_COLOR
            colours[invadable] = INVADABLE_COLOR
            art["activity"].set_data(colours)
            art["cursor"].set_xdata([frame["cursor"], frame["cursor"]])
            art["info"].set_text(frame["info"])
            held = np.bincount(row[row >= 0], minlength=species.size)
            top = int(np.argmax(held))
            art["swatch"].set_color(species_color(species[top]))
            arrived = frame["arrivals"][top]
            since = (f"arrived by t = {arrived:,}" if arrived >= 0
                     else "arrived before the capture")
            art["dominant"].set_text(f"Most sites: species {species[top]}, "
                                     f"{held[top] / held.sum():.0%} ({since})")
            sites = int(invadable.sum())
            art["map_text"].set_text(f"{sites:,} sites can be invaded now" if sites else
                                     "No site can be invaded:\nthe lattice is frozen until\n"
                                     "the next introduction")
            fig.canvas.restore_region(background)
            for artist in art.values():
                (artist.axes or fig).draw_artist(artist)
            encoder.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
            count += 1
            if count % 500 == 0:
                free = shutil.disk_usage(out.parent).free
                if free < min_free_gb * 1e9:
                    raise RuntimeError(f"only {free / 1e9:.0f} GB free on the output drive; stopping")
                print(f"  {out.stem}: {count:,} frames", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with code {encoder.returncode}")
        temporary.replace(out)
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        temporary.unlink(missing_ok=True)
        plt.close(fig)
    return count


def sampled_video(capture, name, args):
    lo, hi = capture.points["windows"][name]
    lo = max(lo - round(args.before * 1e6 / capture.track), 0)
    hi = min(hi + round(args.after * 1e6 / capture.track), capture.times.size - 1)
    if args.max_frames:
        hi = min(hi, lo + args.max_frames - 1)
    t_m = capture.times / 1e6
    part = slice(lo, hi + 1)
    trace = {"x": t_m[part], "species": capture.diversity[part], "patches": capture.patches[part],
             "species_smooth": capture.smooth_d[part], "patches_smooth": capture.smooth_p[part],
             "marks": {str(number): t_m[index] for number, index in
                       enumerate(capture.points["snapshots"].values(), 1) if lo <= index <= hi},
             "xlabel": "t (millions of steps)"}
    settings = capture.plan["simulation"]
    subtitle = (f"p = {settings['p']:g}, α = {settings['alpha']:g}, γ = {settings['gamma']:g}, "
                f"cycles 2-{settings['max_forbidden_cycle_length']} rejected · one frame every "
                f"{capture.track:,} steps")
    newcomer_age = 20_000
    fig, art = build_figure(capture, f"{name}: t = {t_m[lo]:.2f}M to {t_m[hi]:.2f}M",
                            subtitle, trace, newcomer_age)
    features = capture.features

    def frames():
        for frame, index in enumerate(range(lo, hi + 1), 1):
            state = capture.load(index)
            arrived = int(features["introduced"][index] - features["introduced"][max(index - 1, 0)])
            info = (f"t = {t_m[index]:.3f}M    frame {frame:,} of {hi - lo + 1:,}\n"
                    f"{capture.diversity[index]:.0f} living species (smoothed "
                    f"{capture.smooth_d[index]:.1f}, {capture.regime(state['timestep'])})\n"
                    f"{capture.patches[index]:.0f} patches · largest patch "
                    f"{features['largest_patch_share'][index]:.0%} of sites\n"
                    f"species with one patch hold {features['single_patch_share'][index]:.0%}"
                    f" · {arrived} new species since the last frame")
            yield {**state, "cursor": t_m[index], "info": info,
                   "arrivals": capture.arrivals(state["current_species"])}

    return fig, art, frames(), newcomer_age


def step_video(capture, centre, args):
    """Every step of --frames steps centred on a timestep, replayed twice: once to measure the
    window for the trace, once to draw it."""
    count = args.max_frames or args.frames
    first = centre - args.frames // 2
    last = first + count
    newcomer_age = 1_000
    lead = first - newcomer_age  # Replay this far back, so newcomers' exact arrivals are known.
    measures = {key: [] for key in ("timestep", "species", "patches", "largest", "single")}
    exact = {}  # Species ID -> the step it arrived in, for arrivals seen in the replay.
    newest = None
    print(f"  measuring {lead:,} to {last:,}", flush=True)
    for state in replay(capture, lead, last):
        if newest is not None:
            for species in range(newest + 1, state["newest_species"] + 1):
                exact[species] = state["timestep"]
        newest = state["newest_species"]
        if state["timestep"] >= first:
            measured = switch_features.features_of(state["lattice"], state["Gamma"],
                                                   state["current_species"],
                                                   state["newest_species"], state["timestep"])
            measures["timestep"].append(state["timestep"])
            measures["species"].append(measured["species"])
            measures["patches"].append(measured["patches"])
            measures["largest"].append(measured["largest_patch_share"])
            measures["single"].append(measured["single_patch_share"])
    measures = {key: np.asarray(values) for key, values in measures.items()}
    offset = measures["timestep"] - centre
    trace = {"x": offset, "species": measures["species"], "patches": measures["patches"],
             "xlabel": f"Steps from t = {centre:,}"}
    if first <= centre < last:
        trace["marks"] = {"centre": 0}
    settings = capture.plan["simulation"]
    subtitle = (f"p = {settings['p']:g}, α = {settings['alpha']:g}, γ = {settings['gamma']:g}, "
                f"cycles 2-{settings['max_forbidden_cycle_length']} rejected · every step, "
                f"replayed from the kept states")
    fig, art = build_figure(capture, f"Every step: t = {first:,} to {last - 1:,}", subtitle,
                            trace, newcomer_age)

    def frames():
        previous = None
        for frame, state in enumerate(replay(capture, first, last)):
            new = ([] if previous is None
                   else list(range(previous + 1, state["newest_species"] + 1)))
            previous = state["newest_species"]
            arrivals = capture.arrivals(state["current_species"])
            arrivals = np.array([exact.get(int(species), arrival)
                                 for species, arrival in zip(state["current_species"], arrivals)])
            info = (f"t = {state['timestep']:,}    step {frame + 1:,} of {count:,}\n"
                    f"{measures['species'][frame]} living species "
                    f"({capture.regime(state['timestep'])})\n"
                    f"{measures['patches'][frame]} patches · largest patch "
                    f"{measures['largest'][frame]:.0%} of sites\n"
                    f"species with one patch hold {measures['single'][frame]:.0%} · "
                    + (f"new species this step: {', '.join(map(str, new))}" if new
                       else "no new species this step"))
            yield {**state, "cursor": offset[frame], "info": info, "arrivals": arrivals}

    return fig, art, frames(), newcomer_age


def main(argv=None):
    args = parse_args(argv)
    capture = Capture(args.capture)
    out_dir = args.out_dir or capture.root / "videos"
    free = shutil.disk_usage(out_dir.anchor or out_dir).free
    if free < args.min_free_gb * 1e9:
        raise SystemExit(f"only {free / 1e9:.0f} GB free; need {args.min_free_gb:g} GB")
    if args.every_step:
        jobs = [(f"steps_{round(centre * 1e6):012d}", round(centre * 1e6))
                for centre in args.every_step]
        fps = args.fps or 60
    else:
        names = ("Rise", "Collapse") if args.window == "both" else (args.window.title(),)
        jobs = [(name.lower(), name) for name in names]
        fps = args.fps or 30
    for stem, what in jobs:
        out = out_dir / f"{stem}.mp4"
        print(f"{stem} -> {out}", flush=True)
        if args.every_step:
            fig, art, frames, age = step_video(capture, what, args)
        else:
            fig, art, frames, age = sampled_video(capture, what, args)
        count = encode(fig, art, frames, out, fps, age, args.min_free_gb)
        print(f"{stem}: {count:,} frames, {count / fps:.0f} s at {fps} fps, "
              f"{out.stat().st_size / 1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
