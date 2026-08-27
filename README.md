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

The patchiness series is calculated from the sibling checkpoint lattices. A
patch is one four-neighbour connected region of a species, including
connections across the periodic lattice edges. Empty and blocked sites are
not counted. The patchiness curve starts at the first available checkpoint,
which can be later than the first diversity record. If checkpoint metadata is
unavailable, provide the folder explicitly:

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
was actually selected. If checkpoint metadata is unavailable, use the same
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

Blocked sites remain black, empty sites remain light gray, and each permanent
species ID keeps the same color across the full animation. Long runs are
automatically limited to 150 evenly spaced frames to help keep notebook
playback compact; set `max_frames=None` to retain every selected checkpoint.
Set `display=False` when saving without inline notebook playback.
