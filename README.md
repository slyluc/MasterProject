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
| `largest_cluster_only` | Keep only the largest periodic four-neighbour usable cluster at initialization; defaults to `True` for percolation. |
| `populate_first_100` | Force one introduction during each of the first 100 time units. |
| `checkpoint_dir` | Snapshot folder; `None` disables checkpointing. |
| `retention` | Which snapshots survive; `None` uses the default policy. |

Configuration values can be changed directly between runs.

## Run

```python
results = config.run_main()
results_percolation = config.run_percolation()
```

Percolation first draws the blocked mask using `p`, then permanently blocks
usable regions outside the largest connected component. Connectivity includes
empty sites and follows the simulation's periodic boundaries; equal-sized
clusters are resolved by their first row-major site. This happens only before
the run begins. Every model time unit still contains `L_row * L_col` attempts.
Results, checkpoints and analysis records include `effective_p`,
`original_usable_sites`, `usable_sites` and `removed_cluster_sites`.
Set `largest_cluster_only=False` to reproduce the earlier removal rule.

When starting from a checkpoint, its existing blocked mask is retained and
`p` is not drawn again. Smaller usable components are removed if needed.
Changing the geometry starts a new history and requires a separate checkpoint
folder. `MS.keep_largest_cluster(state_or_path)` explicitly prepares a copied
state with a fresh history, without consuming randomness or editing its source.
Species absent after pruning are removed from Gamma in the original order.
An explicitly prepared checkpoint must also use a new output folder, since
preparation resets its history even when its usable mask is already connected.

## Parallel runs and parameter sweeps

Run independent repetitions on separate CPU processes, including from the
notebook on Windows:

```python
results = config.run_many(
    repeats=10,
    kind="main",                 # Or "percolation"
    base_seed=67,
    max_workers=4,               # None chooses available CPUs, capped by jobs
    checkpoint_root="checkpoints/replicates_01",
)
seeds = [result["batch_metadata"]["seed"] for result in results]
```

Each run remains sequential internally. Only independent simulations run in
parallel. `max_workers=1` runs the same jobs sequentially with the same seeds;
change the worker count to suit the machine's CPU, RAM and disk throughput.
Numba's initial compilation and process startup add overhead, so parallel
execution is most useful for long runs. Each worker holds its own lattice
and Gamma matrix, whose memory grows with the square of diversity.

For different settings, copy the configuration rather than mutating it during
the batch:

```python
from dataclasses import replace

gammas = [0.071, 0.0725, 0.075, 0.08, 0.10]
configs = [replace(config, gamma=gamma, progress=False) for gamma in gammas]
results = MS.run_simulations(
    configs,
    kind="percolation",
    repeats=2,
    base_seed=67,
    max_workers=4,
    checkpoint_root="checkpoints/gamma_sweep_01",
)
# Results follow config order, with that config's repetitions together.
results_by_gamma = {
    gamma: results[2 * index:2 * index + 2]
    for index, gamma in enumerate(gammas)
}
```

Seeds are generated with NumPy `SeedSequence.spawn`, giving every job its own
child stream even when configurations have identical `seed` values. An
explicit `base_seed` supplies the shared root. Without it, each config's
`seed` supplies its root; `seed=None` uses fresh system entropy. An individual
job can be reproduced with `replace(config, seed=seeds[index]).run_main()`
(or `.run_percolation()`). The original config objects are left unchanged.

With `checkpoint_root`, each job gets its own subfolder, and
`batch_plan.json` records the seeds, settings and destinations before work
starts. Without a root, a config's `checkpoint_dir`, when set, serves as the
parent for separate job subfolders. Existing batch plans and job
folders are rejected to avoid overwriting previous work; choose a fresh root
for a new batch. Results include each job's settings in `batch_metadata`.

Use `initial_state=checkpoint_or_result` to start all jobs from one state.
**Batch branches deliberately replace its saved RNG with independent child
streams.** The lattice, existing Gamma and elapsed time are retained, while
each branch starts its own history. Ordinary `run_main`, `run_percolation`
and `resume` still restore the saved RNG exactly, even if `config.seed` changes.
To reproduce one batch branch directly, remove `rng_state` from a copy of
the source state and pass the recorded child seed.

In a standalone Python script, put the batch call under
`if __name__ == "__main__":` so Windows workers can import the script safely.
Notebook calls do not need this guard; workers are defined in `sim_modules`.

### Prepared 200x200 percolation gamma sweep

The requested list, including `0.12`, has **12 gamma values**, so two repeats
give **24 independent simulations**:

```python
gammas = [0.069, 0.07, 0.071, 0.072, 0.073,
          0.09, 0.092, 0.094, 0.096, 0.098, 0.1, 0.12]
```

The ready-to-run script uses
`checkpoints/gamma_07_alpha_0125_perc_05/checkpoint_000020000000.npz`,
which contains a 200x200 lattice at timestep 20,000,000 and `alpha=0.0125`.
It first retains the largest four-neighbour connected region of usable sites,
including periodic edge connections, and permanently blocks all other regions.
Every job starts from this same prepared lattice and existing Gamma matrix.
Changing gamma affects links for subsequent species introductions.

The nominal `p` remains `0.05`; the checkpoint's existing blocked mask is
retained, rather than removing another random 5% of sites. The script prints
the effective blocked fraction after removing disconnected regions. In this
particular checkpoint all 37,972 usable sites already belong to one connected
component, so no additional sites are removed: its 2,028 blocked sites give an
effective fraction of **5.07%**. Its default `T=50_000_000` means **50M additional
model time units**, ending at timestep 70,000,000. Each model time unit still
contains 40,000 sequential attempts.

From the project directory, inspect the settings first, then launch:

```powershell
.\MSvenv\Scripts\python.exe tools/run_gamma_sweep.py --dry-run
.\MSvenv\Scripts\python.exe tools/run_gamma_sweep.py
```

The dry run reads the checkpoint and reports the setup without starting jobs
or writing files. Defaults use up to **six CPU processes** (fewer on machines
with fewer available CPUs), `repeats=2`, `base_seed=67` and
`track_every=10_000`. The process pool uses CPU cores; GPU core counts do not
control these workers. Each process has its own lattice and Gamma allocation;
the latter grows with species diversity. On a 32 GB machine, six workers are a
reasonable starting configuration for this 200x200 checkpoint. Use
`--workers 4` to reduce simultaneous CPU and memory use.

Every launch selects a fresh batch folder under `checkpoints/`; an explicit
`--output-root` must also name a new folder. Each job has separate checkpoint,
diversity and patch histories, beginning at the prepared state at timestep
20M. The source run's histories and RNG stream are not carried into the jobs.
All 24 jobs receive different reproducible child seeds, recorded before
simulation starts in `batch_plan.json`. Repeating the batch with `base_seed=67`
reproduces its trajectories; use `--base-seed 68` for another independent batch.
`sweep_setup.json` records the source and pruning details, and
`sweep_results.json` indexes completed jobs by gamma, repeat, seed and folder.

To choose a destination or adjust the same setup:

```powershell
.\MSvenv\Scripts\python.exe tools/run_gamma_sweep.py --workers 6 --output-root checkpoints/my_gamma_sweep_01
```

### Stop and resume a sweep

Press **Ctrl+C** in the terminal running the sweep to stop it. The command
stops its worker processes and prints the resume command. After a PC crash,
use the same command with the existing sweep folder:

```powershell
.\MSvenv\Scripts\python.exe tools/run_gamma_sweep.py --resume checkpoints/my_gamma_sweep_01
```

Replace that folder with the batch folder printed when your sweep started.
You can inspect its saved progress or change the number of simultaneous runs:

```powershell
.\MSvenv\Scripts\python.exe tools/run_gamma_sweep.py --resume checkpoints/my_gamma_sweep_01 --dry-run
.\MSvenv\Scripts\python.exe tools/run_gamma_sweep.py --resume checkpoints/my_gamma_sweep_01 --workers 4
```

Resume keeps the saved gamma values, targets, seeds and output folders.
Finished runs are skipped; unfinished runs restore their latest checkpoint's
lattice, Gamma, RNG and histories. Runs that had not saved a checkpoint start
with their original child seeds and the sweep's locally saved starting state.
It does not add another 50M timesteps. With `track_every=10_000`, a stopped run
repeats only its unfinished checkpoint interval, up to 10,000 timesteps,
following exactly the same random sequence. Reducing the tracking interval
before starting a new sweep reduces this amount at the cost of more disk work.

New sweeps keep a copy of their prepared starting state under `initial_state/`,
so pending runs can restart even if the original source checkpoint is moved.
Sweeps created before this command use the original source checkpoint when
pending runs need it. Atomic checkpoint and analysis writes protect completed
saves; interrupted temporary files are ignored. Resume also finishes an
analysis write interrupted after the final checkpoint, without further model
updates. File locks prevent a second command from overlapping active workers.

The general notebook API is `MS.resume_simulations(checkpoint_root,
max_workers=4, initial_state=original_shared_state)`. The starting state is
needed only for branched jobs without a checkpoint; the sweep command supplies
it automatically.

## Performance and reproducibility

The simulation keeps its existing microscopic update order and random draw
order. The microscopic rules follow the supplied
[model description](emergence_of_diversity_in_a_model_ecosystem.md#model-ecosystem):
each time unit consists of `N = L_row * L_col` invasion attempts, each followed
by an introduction trial, with successful changes visible to the next attempt.
The module's initialization settings are retained: it starts fully occupied
with `D` species, while the paper usually starts empty with an introduction
warm-up and reports equivalent subsequent dynamics for a single-species start.
Permanent site removal is the explicit percolation extension.

Lattice-to-species-slot setup uses NumPy arrays; ordinary lattices use
32-bit neighbor indices instead of 64-bit indices, halving the neighbor
table's memory; and tracking arrays are allocated once instead of rebuilding
the full history at every checkpoint. Exporting Gamma avoids a redundant
matrix copy, and setup temporaries are released before long runs.
Event-code and random-number generation
retain their existing types and ordering. Seeded regression tests compare
the lattice, ordered species, Gamma, histories and final RNG state with the
original implementation, including checkpointing and continuation.

The compiled loop also reads the neighbor table through its flat view: an
event code is already `4 * source_site + direction`, so it directly indexes
the chosen neighbor. Site choices and update order stay the same.

When there are no species, or exactly one species occupies every usable site,
all invasion attempts are ineffective until an introduction. The loop still
draws every event code and introduction trial. It skips the ineffective
invasions only when those already drawn trials contain no introduction and
the step has no forced introduction. Model time, diversity records and RNG
state advance exactly as before. A single-species lattice with empty usable
sites takes the ordinary path so that colonization proceeds immediately.

Earlier setup/memory benchmarks on this PC (median of five alternating runs,
Numba compilation warmed, seed 123, no checkpoints) measured:

| Lattice / initial species | Time units | Original | Optimized |
|---|---:|---:|---:|
| 200×200 / 1 (`gamma=0.07`, `alpha=0.0125`) | 1,000 | 0.531 s | 0.492 s |
| 200×200 / 40 (same rates) | 1,000 | 0.676 s | 0.635 s |
| 512×512 / 100 (`gamma=0.1`, `alpha=0.1`) | 100 | 0.955 s | 0.709 s |
| 512×512 / 100 (setup only) | 0 | 0.140 s | 0.027 s |

These include initialization; its improvement contributes less during a
multi-hour run. Hardware, diversity and checkpoint frequency affect the gain.
The parallel API adds throughput across independent runs.

### Experiments with parallel events

`tools/benchmark_event_batches.py` tests two ways of grouping independent
invasions, while preserving exact trajectories:

- Consecutive batches stop before source/target footprints overlap.
- Dependency waves retain the relative order of every pair of events that
  touches a common site, and group events at the same dependency level.

Only lattice reads and writes run in parallel. Species counts, extinctions and
free slots are committed in the original event order; even distant attacks
can affect the same species. Each introduction ends the current group before
new interactions are generated. Choices are pre-drawn for one time unit,
as in the ordinary kernel. Choices for later units are left until after
the current unit's conditional introduction draws.

Both schedulers passed exact state and RNG comparisons but were slower on
this six-CPU machine. With four threads, the larger dependency waves took
0.686 s versus 0.337 s for 200x200, 40 initial species and 500 time units;
for 512x512 and 100 units they took 1.643 s versus 0.677 s. Both used
`gamma=0.07`, `alpha=0.0125` and forced introductions for the first 100 units.
Dependency scanning, bucket construction, thread synchronization and ordered
bookkeeping outweighed the parallel work. These schedulers remain experiments;
production parallelism uses separate independent simulations.

To repeat the measurements on another machine:

```powershell
python tools/benchmark_event_batches.py --side 200 --diversity 40 --timesteps 500
python tools/benchmark_simulation_kernel.py --variants baseline production --size 200 --timesteps 1000 --repeats 5
```

The second tool compares the production loop with a frozen copy of the
earlier compiled kernel. Both use the current setup/history optimizations.
Compilation is excluded, repeated timings alternate order, and every result
is checked against the same seeded trajectory. Its other modes are exploratory
candidates; none changes the production configuration. JSON reports alongside
the tools retain settings, raw timings and checks.

With the final uniform-state shortcut, a 200x200 run starting from one species
(`seed=67`, `gamma=0.07`, `alpha=0.0125`, 1,000 units) took a median 0.290 s
versus 0.533 s for the earlier kernel: 1.84x faster. Starting with 40 species
took 0.694 s versus 0.702 s, within ordinary timing variation. These are five
alternating warmed repetitions, with identical full trajectories and RNG
states; the gain depends on time spent in completely occupied single-species
states. The trial scan is compiled separately to avoid enlarging the active
invasion loop. See `tools/benchmark_kernel_final_200_results.json` for raw data.
On the final 512x512, 40-species, 1,000-unit test, medians were 7.496 s versus
7.208 s across three pairs, about 4% slower; the shortcut offers its benefit
during uniform low-diversity periods. Host timing varied, so use the benchmark
on the target machine when estimating total runtime. That test is recorded in
`tools/benchmark_kernel_final_512_results.json`.

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

A branch into a new folder starts its own history. Its diversity and patch
series, `analysis.npz`, and species-introduced count begin at the state it
starts from, so none of the source run's history is copied in. Simulation time
is not reset: a branch from `checkpoint_000100000000.npz` with
`T = 50_000_000` records timesteps 100M to 150M.

Continuing into the folder the state came from extends that run instead, and
keeps its full history:

```python
config.checkpoint_dir = "checkpoints/base_run"
extended = config.run_main(initial_state=config.checkpoint_dir)
```

Pass `continue_history=True` or `False` to `run_main` or `run_percolation` to
override either default. `config.resume()` always keeps the history.

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

To plot the curves from a later time, pass `start_timestep`. Use it to hide
the copied source-run history in branch folders written before branches
started their own:

```python
state = MS.load_checkpoint("checkpoints/gamma_075_alpha_0125")
MS.show_results(
    state,
    show_patchiness=True,
    smooth_sigma=20,
    start_timestep=100_000_000,
)
```

Samples before `start_timestep` are dropped before smoothing, so they do not
bleed into the smoothed curve. The printed species-introduced count still
covers the whole recorded history.

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

### Invasion cycles around diversity switches

Use the run's recorded switches and surviving Gamma snapshots to count simple
directed invasion cycles. The default side is the high-diversity side: before
a collapse or after a generation/recovery event.

```python
cycles = MS.show_gamma_cycle_histogram(
    "checkpoints/gamma_07_alpha_0125",
    event_type="collapse",  # or "generation"
    event_index="all",      # combine all matching switches (the default)
    log_y=True,
    fit_model="exponential",
    fit_min_length=10,
    fit_max_length=30,
)
print(cycles["cycle_counts"])
print(cycles["exponential_fit"])
```

A straight trend with logarithmic count and linear cycle length corresponds
to an exponential fit, `count = A * exp(k * length)`. The legend shows `A`
and `k`. Set `fit_model="power_law"` (or the older `fit_power_law=True`) for
`count = C * length**(-alpha)`, which is a straight trend only when **both**
axes are logarithmic. `fit_min_length` and `fit_max_length` choose an
inclusive range for either model; omitting them fits all positive histogram
bins. At least two positive lengths are required. These are descriptive fits
to the chosen bins, including estimated long-cycle counts.

Set `event_index=0` for the first matching switch, `1` for the second, and so
on. By default every possible cycle length is considered, up to the number
of living species in each Gamma matrix. Counts for lengths 2–4 are exact;
longer cycles are **estimates** from weighted random paths. Exact counting
of all simple cycles can take
exponential time. Increase `samples_per_checkpoint` (default 5,000) to make
the longer-cycle estimates more stable; `seed` makes them reproducible.
`cycles["estimated_lengths"]` identifies estimated bars and
`cycles["sample_hits"]` reports how many sampled paths found each length.
Zero hits do not establish that a cycle length is absent. You can still set
`max_cycle_length` to focus on shorter cycles. For very large matrices,
estimated counts can exceed floating-point range; use
`cycles["log10_cycle_counts"]` in that case, which the plot switches to
automatically.

Pass `side="before"` or `side="after"` to override the default, and
`window=100_000` to consider only that many timesteps from each switch.
`log_x=True` also gives a logarithmic length axis. Counts sum over the
selected checkpoints: a cycle that persists for ten snapshots is counted
ten times. Rotating the same cycle does not count it again; opposite directed
cycles count separately. Only retained checkpoints can be analysed, so
pruned gaps contribute no Gamma matrices.

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
