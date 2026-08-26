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
