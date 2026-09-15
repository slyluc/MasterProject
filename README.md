# Simulation framework

## Setup

```powershell
pip install -r requirements.txt
```

## Configuration

```python
import sim_modules as MS

config = MS.SimulationConfig(
    L_col=200,
    L_row=200,
    D=1,
    gamma=0.1,
    alpha=0.01,
    T=10_000_000,
    track_every=10_000,
    seed=235235,
    progress=True,
    p=0.6,
    populate_first_100=True,
    checkpoint_dir="checkpoints/base_run",
)
```

| Option | Meaning |
|---|---|
| `L_col`, `L_row` | Lattice width and height. |
| `D` | Number of initial species. |
| `gamma` | Probability of a directed invasion link. Existing Gamma links do not change. |
| `alpha` | Species-introduction rate control; introductions average `alpha * gamma` per time unit. |
| `T` | Time units to run. When starting from a state, these are additional time units. |
| `track_every` | Interval for diversity records and checkpoints. |
| `seed` | Random seed; `None` gives a new random run. |
| `progress` | Show a progress bar. |
| `p` | Blocked-site probability for a new percolation lattice. |
| `populate_first_100` | Force one introduction during each of the first 100 time units. |
| `checkpoint_dir` | Snapshot folder; `None` disables checkpointing. |
| `retention` | Which snapshots survive; `None` uses the default policy. |

Configuration values can be changed directly between runs.

## Run

```python
results = config.run_main()
results_percolation = config.run_percolation()
```

## Checkpoints

When `checkpoint_dir` is set, an atomic `.npz` snapshot is written every
`track_every` time units and at the final time. Each snapshot contains the
lattice, Gamma, species order, history, elapsed time, and random-generator
state.

Most of those snapshots are then deleted again; see
[Snapshot retention](#snapshot-retention) for what survives and why.

```python
# Newest checkpoint in a folder
state = MS.load_checkpoint("checkpoints/base_run")

# A specific checkpoint
state = MS.load_checkpoint(
    "checkpoints/base_run/checkpoint_000000010000.npz"
)
```

Resume a crashed run to its original target time:

```python
results = config.resume()
```

## Snapshot retention

A full snapshot history is large. A 100M-timestep run tracked every 10,000
steps writes 10,000 snapshots: 4 GB at `gamma=0.07, alpha=0.0125`, and 57 GB
for the `p=0.50` percolation run, whose ~2,600 coexisting species make each
`Gamma` matrix 6.8 MB. `Gamma` grows with the square of living species, so it
dominates the lattice in any high-diversity run.

Almost all of that is only ever read back to recount patches for a plot, so a
run now counts the patches itself, records them, and deletes the snapshot.

Each run folder therefore ends up holding:

- `analysis.npz` — the complete living-species and patch series for the run,
  the detected swaps, and the run parameters. A few hundred KB.
- the newest snapshot, so the run stays resumable and plottable;
- every snapshot within one window either side of each large swap in living
  species.

A swap is a transition between the run's two diversity regimes. The series is
smoothed, and thresholds are placed at 25% and 60% of the run's high-regime
level; a swap is recorded when the smoothed curve falls to the low threshold
from the high regime, or rises to the high threshold from the low regime. For
`gamma=0.07, alpha=0.0125`, where living species repeatedly collapse from
around 40 to around 1 and recover, this finds each collapse and each recovery
and keeps 1M timesteps of context on both sides.

Tune or disable the policy with `retention`:

```python
config.retention = MS.RetentionPolicy(event_window=2_000_000)
config.retention = MS.RetentionPolicy(keep_all=True)   # keep every snapshot
```

| Field | Default | Meaning |
|---|---|---|
| `keep_all` | `False` | Keep every snapshot, as before. |
| `event_window` | `1_000_000` | Time kept on each side of a swap. |
| `smooth_samples` | `5` | Moving-average width, in tracked samples. |
| `reference_quantile` | `0.75` | Quantile taken as the high-regime level. |
| `low_fraction` | `0.25` | Low threshold, as a fraction of that level. |
| `high_fraction` | `0.60` | High threshold, as a fraction of that level. |
| `minimum_swing` | `2.0` | Report nothing when the thresholds are closer than this. |
| `detect_every` | `10` | Re-run detection after this many snapshots. |

A snapshot is never deleted until its patch count is recorded, so a folder
written before `analysis.npz` existed is left alone until it is migrated.

### Shrinking existing runs

`tools/prune_checkpoints.py` applies the same policy to folders that already
hold a full history. It counts the patches, writes `analysis.npz`, and then
deletes the redundant snapshots. Without `--apply` it reports what it would
delete and deletes nothing; it still writes the analysis record, because
counting is the slow half of the job and saving the counts is what makes the
later `--apply` pass instant:

```powershell
python tools/prune_checkpoints.py checkpoints/gamma_07_alpha_0125
python tools/prune_checkpoints.py checkpoints/gamma_07_alpha_0125 --apply
```

Several folders at once, quoting the pattern so the tool expands it rather
than the shell. `cmd.exe` does no globbing and PowerShell does none for a
native command, so an unquoted `checkpoints/*` reaches the tool as a literal
string on Windows:

```powershell
python tools/prune_checkpoints.py "checkpoints/*"
python tools/prune_checkpoints.py "checkpoints/*" --apply
```

The record is always written before anything is deleted, and rerunning the
tool reuses the counts it already made. `--window`, `--smooth`, `--low` and
`--high` override the policy defaults.

## Continue or change rules

```python
state = MS.load_checkpoint("checkpoints/base_run")

config.gamma = 0.2
config.T = 1_000_000
config.checkpoint_dir = "checkpoints/gamma_02_branch"

changed = config.run_main(initial_state=state)
```

The saved lattice and Gamma are retained. New settings apply from that state;
`gamma` affects links of newly introduced species. Existing blocked sites stay
blocked, and `p` is not reapplied. Use a new checkpoint folder for each branch.

A previous result can be used directly:

```python
changed = config.run_main(initial_state=results)
```

Raw arrays are also accepted:

```python
state = {"lattice": lattice, "Gamma": Gamma}
changed = config.run_main(initial_state=state)
```

Without `current_species`, Gamma rows are assumed to follow ascending positive
species IDs in the lattice.

## Plot

```python
MS.show_results(results)
state = MS.load_checkpoint("checkpoints/base_run")
MS.show_results(state)
```

Add the number of spatial species patches on a separate right-hand y-axis:

```python
MS.show_results(state, show_patchiness=True)
```

The patchiness series is read from the run's `analysis.npz`, which is what
lets a pruned run still plot its full history. A patch is one four-neighbour
connected region of a species, including connections across the periodic
lattice edges. Empty and blocked sites are not counted. Folders without an
analysis record fall back to counting the checkpoint lattices still present,
so the curve there starts at the first available checkpoint, which can be
later than the first diversity record. If checkpoint metadata is unavailable,
provide the folder explicitly:

```python
MS.show_results(
    results,
    show_patchiness=True,
    checkpoint_dir="checkpoints/base_run",
)
```

To display the lattice nearest to a particular simulation timestep, pass
`lattice_timestep`. The diversity and optional patchiness curves still cover
the complete result period:

```python
MS.show_results(
    results,
    show_patchiness=True,
    lattice_timestep=0.3e7,
)
```

Historical lattices come from the saved checkpoints, so the displayed time is
the closest available checkpoint time. The plot title reports the time that
was actually selected. Under the default retention policy the lattices that
survive are those near a swap, so a request far from one resolves to the
nearest kept snapshot. If checkpoint metadata is unavailable, use the same
`checkpoint_dir` argument shown above.

## Animate lattice evolution

Animate a result directly in a notebook. Frames are loaded lazily,
`frame_stride=10` uses every tenth checkpoint, and the endpoint is always
included:

```python
animation = MS.animate_lattice(
    results,
    frame_stride=10,
    interval=150,  # Milliseconds between frames
)
```

A checkpoint directory or individual checkpoint file can be passed instead.
An individual file animates the sibling snapshots only through that file's
timestep. Start/end bounds are inclusive, using the first checkpoint at or
after the start and the last checkpoint at or before the end:

```python
animation = MS.animate_lattice(
    "checkpoints/base_run",
    start_timestep=1e6,
    end_timestep=5e6,
    frame_stride=5,
)
```

Save an ignored GIF under `animations/` by adding `save_path`. MP4 output is
also supported when FFmpeg is installed:

```python
animation = MS.animate_lattice(
    results,
    frame_stride=10,
    save_path="animations/base_run.gif",
)
```

Animations draw on the same snapshots, so under the default retention policy
a run animates the windows around its swaps rather than its whole history.
Set `retention=MS.RetentionPolicy(keep_all=True)` on a run whose full
evolution you intend to animate.

Blocked sites remain black, empty sites remain light gray, and each permanent
species ID keeps the same color across the full animation. Long runs are
automatically limited to 150 evenly spaced frames to help keep notebook
playback compact; set `max_frames=None` to retain every selected checkpoint.
Set `display=False` when saving without inline notebook playback.
