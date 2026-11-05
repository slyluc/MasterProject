# Research log: emergence of diversity in a lichen-inspired lattice ecosystem

This log records the project's reasoning, not its code. For each step it gives the question asked,
why it was asked, what was done, what came out, and what that changed in the understanding. The
engineering detail lives in [README.md](README.md), the analyses in [analysis/](analysis/), and the
model's source paper in [emergence_of_diversity_in_a_model_ecosystem.md](emergence_of_diversity_in_a_model_ecosystem.md).

## Conventions

- **Times** are local Danish time: CEST (UTC+2) until 25 Oct 2026 and CET (UTC+1) after it. A commit is
  dated by its commit time. A batch run is dated by the timestamp in its folder name, which is when it
  started. Simulation time is written as t in model time units, where M = 10⁶. One unit is
  N = L_row·L_col invasion attempts, which is 40,000 on the 200 × 200 lattice used throughout.
- **Evidence keys:**
  - `git <hash>`: a commit.
  - `nb <notebook> §n / cell n`: a notebook section or cell.
  - `run <folder>`: a run folder under `checkpoints/` (or `E:\MasterProject\checkpoints\`).
  - `session <date>`: a Claude Code session on this PC. These hold intentions stated in writing at
    the time.
  - `paper Fig. n`: Mitarai, Mathiesen & Sneppen (2012).
- **Why a step was taken** is marked as one of:
  - **Recorded:** written down at the time, in a session or a commit message.
  - **Inferred:** reconstructed afterwards from the code and the order of events. The early phases
    (Aug–early Oct) left only commit messages, so most of their reasons are inferred.
- **Status of a finding:**
  - **Established:** replicated, or seen in many independent runs.
  - **Observed:** seen in one or a few runs.
  - **Hypothesis:** proposed but not yet tested.
- **Mean species** is the time-mean of living species over the second half of a run's own span, unless
  stated otherwise. **Regimes** follow `sim_analysis.ipynb`: the series is Gaussian-smoothed over 0.2M,
  "high" holds from reaching 50 species until falling to 10, and "low" otherwise.

---

## 0. Frame of reference

### The model

- **Lattice:** an L × L periodic lattice. Each site holds one sessile species or is empty.
- **Interaction network:** a directed matrix Γ. Each link i → j ("i can invade j") is drawn with
  probability γ when a species arrives, and links never change afterwards.
- **One time unit** is N invasion attempts. Each attempt picks a site and a neighbour. If Γ allows it,
  the neighbour is taken over.
- **Introductions:** new species arrive on random sites at an average of α·γ per time unit. A newcomer
  always gets a forced link to the species it lands on, so it takes that species' site.

This project adds three things to the model:
1. **Blocked sites:** a fraction p of the sites is permanently blocked (percolation).
2. **Largest cluster only** (from 7 Oct): usable sites outside the largest connected cluster are also
   blocked.
3. **Short-cycle rejection** (from 8 Oct): a newcomer's links are redrawn until Γ contains no directed
   cycle of length 2–6.

### What the paper claims

Mitarai, Mathiesen and Sneppen (2012) make four claims that this project tests:
1. **A transition:** for α → 0 there is a first-order transition from high to low diversity at
   γ_c ≈ 0.055. Near γ_c the system is bistable. Their Fig. 3 uses α = 0.0125, γ = 0.07, L = 200.
2. **Patchiness first:** in both directions, a change in patchiness (the number of single-species
   patches) comes before the change in diversity.
3. **Two routes to new patches:** patches form by stochastic fragmentation, either through a linear
   chain A → B → C or through the collapse of a cycle of length ≥ 4.
4. **Cycles of length 4–6 matter most:** they ran a quasistatic simulation (a newcomer arrives only once
   all invasion activity has stopped) with short cycles removed. Removing cycles of length 2–4 raised
   diversity. Removing everything shorter than 6 sent γ = 0.05 to a single species, while γ = 0.025
   stayed above one species. Their conclusion is that linear chains alone cannot keep diversity high,
   and that cycles of length 4–6 are central.

### Guiding question

*Inferred, from the order of experiments:* which mechanism creates and destroys the high-diversity
state, and can spatial structure in the habitat (blocked sites) do the job that short invasion cycles
do in the paper?

---

## 1. Timeline

### Phase I: building the model and reproducing the paper (26 Aug – 2 Oct 2026)

#### 2026-08-26 18:35: the model is implemented (`git cf2f7fd`, `git d6c185f` at 23:29)
- **Done:**
  - [sim_modules.py](sim_modules.py), a Numba kernel. Saved checkpoints include the random-number
    state, so any run can be resumed or branched into new parameters from any saved state.
  - Blocked-site (percolation) runs.
  - First runs: 200 × 200, γ = 0.1, α = 0.01; a branch from γ = 0.1 to γ = 0.2 (`run gamma_02_branch`);
    trial percolation at p = 0.2 and 0.6.
- **Why (inferred):** to have a faithful, reproducible model that can be branched mid-run. Branching
  makes it possible to ask whether an established community survives a change of parameters.

#### 2026-08-27 02:10–23:36: patchiness, animations, and the paper's Fig. 3 parameters (`git 5e6ff95`, `git 7935067`, `git f68b0b3`)
- **Done:**
  - A patch count P: four-neighbour connected regions held by one species, with periodic edges.
  - Lattice animations and the first unit tests.
  - Default parameters moved to α = 0.0125, γ = 0.07, L = 200, exactly those of paper Fig. 3.
- **Why (inferred):** the paper's central observable is P plotted against diversity D (Fig. 3), so the
  first goal was to reproduce Fig. 3.

#### 2026-08-30 20:31: first blocked-site runs (`git ab339b5`)
- **Done:** at γ = 0.07, α = 0.0125, runs at p = 0, 0.05, 0.2 and 0.5 (the p = 0 run was later
  extended to 100M).
- **Result:** means are 23.4 species (p = 0), 42.0 (p = 0.05), 64.5 (p = 0.2) and about 2,560 (p = 0.5).
  - The p = 0 run collapses and recovers repeatedly (10 recorded switches over 100M), as the paper
    describes. This reproduces the paper's bistable regime qualitatively.
  - Blocking sites raises diversity: the first sign of what became the project's main theme.
- **Later understanding:** at p = 0.5 only half the sites are usable, below the site-percolation
  threshold of about 59%. The habitat therefore breaks into many isolated pockets, and each keeps
  species of its own. The roughly 2,600 species are an artefact of fragmentation, which led to the
  largest-cluster rule (7 Oct). Its Γ matrices also made this one run's snapshots grow to 57 GB.
  The p = 0.2 count is inflated too: at its last snapshot 15 of its 64 species lived only in the 56
  cut-off sites. Both runs were removed on 5 Nov; the p = 0 and p = 0.05 runs are one cluster and stay.

#### 2026-09-15 14:18 / 17:25: faster plotting and snapshot retention (`git 3652822`, `git 5bfe5c0`)
- **Why (recorded in the commit and README):** storage was "bloating". A 100M-step run tracked every
  10,000 steps writes 10,000 snapshots: 4 GB at γ = 0.07, and 57 GB for the p = 0.5 run. Γ grows with
  the square of the number of species, about 6.8 MB per matrix at 2,600 species.
- **Done:** each run now keeps:
  - `analysis.npz`, with the full diversity and patch series;
  - its newest snapshot;
  - snapshots within ±1M of each detected regime swap.
  `tools/prune_checkpoints.py` applies the same policy to old runs.
- **Consequence:** lattices from before the capture of Phase V exist only near swaps. Watching a switch
  at fine resolution therefore needed a dedicated capture (29 Oct).

#### 2026-09-28 16:57–17:03: ladders in γ and in p (`git 8976d34`, `git 07ee0cf`)
- **Done:**
  - **γ branches at p = 0.05**, from the p = 0.05, γ = 0.07 run at t = 20M: γ = 0.071, 0.0725, 0.075,
    0.08 and 0.10.
  - **A γ ladder at p = 0:** γ = 0.075 from γ = 0.07 at t = 100M; then γ = 0.08 and 0.09 from γ = 0.075
    at t = 150M.
  - **A p ladder at γ = 0.15:** fresh 30M runs at p = 0.075, 0.1, 0.125, 0.15 and 0.2.
  - **Two branches at p = 0.15:** γ = 0.10 from the γ = 0.15 run at t = 30M, then γ = 0.30 from that
    branch at t = 80M.
- **Why (inferred):** to find where diversity is lost as γ rises, and whether blocked sites move that
  point.
- **Result:** one run per setting; means of each run's own span:

  | Setting | γ | Mean species |
  |---|---|---|
  | p = 0 | 0.075 / 0.08 / 0.09 | 7.7 / 11.8 / 4.6 (against 23.4 at 0.07) |
  | p = 0.05 | 0.071 / 0.0725 / 0.075 / 0.08 / 0.10 | 39.2 / 33.1 / 14.0 / 17.2 / 4.2 |
  | γ = 0.15 | p = 0.075 / 0.10 / 0.125 / 0.15 / 0.20 | 7.4 / 9.9 / 18.4 / 25.9 / 52.2 |
  | p = 0.15 | 0.10 / 0.30 | 45.6 / 28.1 |

- **Reading:**
  - Diversity falls as γ rises, but single runs are noisy: 0.08 came out above 0.075.
  - Blocking raises diversity steeply, even at γ = 0.15–0.3, far above the paper's γ_c.
  - **Caveat:** these runs predate the largest-cluster rule. At p ≤ 0.2 the cut-off pockets are small
    (97 of about 32,000 sites in the later p = 0.2 lattice), so a pocket artefact is unlikely. Still,
    γ ≥ 0.1 with blocking has not been repeated under the rule.
- **Later understanding (5 Nov):** the caveat was wrong. Each pocket, even a single site, keeps a
  species of its own, so the γ = 0.15 ladder counted mostly pocket species. At the last snapshots the
  largest cluster held 3–5 species for p = 0.075–0.15 and 17 at p = 0.2, against 6–62 counted. The
  p ladder and the p = 0.15 branches were removed (see 5 Nov, 14:19).

#### 2026-10-01 12:46 – 2026-10-02 14:30: counting cycles around switches (`git 56c55db`, `git ed9de9e`, `git 1297b35`)
- **Done:** `show_gamma_cycle_histogram` counts directed simple cycles in the Γ snapshots kept near
  swaps. Lengths 2–4 are counted exactly and longer ones estimated from sampled paths. It fits
  exponential or power-law curves to the counts. It was applied to collapses at γ = 0.08, p = 0.05 and
  at γ = 0.07, p = 0, and to rises at γ = 0.075, p = 0. The p = 0.05, γ = 0.0725 run was continued to
  120M (17 swaps).
- **Why (inferred):** the paper credits collapsing cycles of length ≥ 4 with creating patches. The
  question was whether Γ's cycle content differs around collapses and rises.
- **Result:** in all three runs the counts grow roughly exponentially with length, by about 2.6–2.8
  times per extra link between lengths 5 and 25. Counts above length 4 are estimates.
- **Reading (inferred, confirmed later in Phase V):** a diverse Γ always contains vast numbers of long
  cycles. A cycle in Γ is only a possibility, though; it does nothing unless its species are neighbours
  on the lattice. On 31 Oct about 80% of states turned out to be frozen, so counting cycles in Γ cannot
  by itself identify the mechanism.

### Phase II: tooling for controlled, replicated experiments (6–7 Oct)

#### 2026-10-06 11:36: replicated γ sweep with all cycles allowed (`run gamma_sweep_perc05_largest_20261006_113609_5a0ff663`)
- **Done:**
  - 12 values of γ (0.069–0.073 and 0.09–0.12) × 2 repeats, each with its own seed, 50M steps.
  - All runs branch from the p = 0.05, γ = 0.07 run at t = 20M. That lattice has 2,028 blocked sites
    (5.07%), and its usable sites already form a single cluster.
  - Run on another machine and copied to this PC on 9 Oct.
- **Why (inferred):** replicated averages on both sides of the transition seen in September.
- **Result:**
  - γ = 0.069–0.073: 24.7–42.1 species per run, with 0–23 swaps.
  - γ = 0.09–0.12: 4.4–12.5 species, with 13–41 swaps.
- **Conclusion:** with α = 0.0125 and p = 0.05 the high state is lost between γ = 0.073 and 0.09.
  **Established (24 runs).**

#### 2026-10-07 22:52: parallel runs and largest-cluster habitat (`git 9798582`)
- **Why (recorded):** "use all cores", and a "framework for ignoring stuff that isn't the largest
  cluster". A single p = 0.2 run takes about 11 hours per 50M steps.
- **Done:**
  - Parallel independent runs, each with its own reproducible seed, a `batch_plan.json` per batch, and
    resumable sweeps.
  - The largest-cluster-only habitat.
  - Regression tests that compare full trajectories, including the random-number state, against the
    original kernel.
  - A faster kernel: 1.84 × faster in phases where one species fills the lattice.
  - Two schemes for running invasion events in parallel were benchmarked and dropped, because they
    were slower (0.686 s against 0.337 s).
  - The paper itself, added to the repository as Markdown.
- **Significance:** from here on, a blocked lattice is one connected habitat, so isolated pockets can
  no longer create diversity.

### Phase III: closing the cycle route (8–19 Oct)

#### 2026-10-08 01:14–01:25: rejecting cycles of length 2–6 (`git aa9e386`, `run no_short_cycles_perc05_20261008_012527_5832a1ea`)
- **The rule:** when a species arrives, its whole row and column of Γ are redrawn until no directed
  cycle of length 2–6 exists among living species. Initial Γ matrices obey the same rule.
- **Comparison with the paper:** this matches the paper's Fig. 6 rule with cycles of length 7 and up
  allowed. Unlike the paper, it runs in the full (not quasistatic) dynamics, and it rejects length 6
  as well.
- **Why (recorded in part):** the commit calls these "noise sims". The README adds that the rule
  does not remove every cycle or "guarantee pure noise".
- **Why (inferred):** to remove the cycle route the paper calls central. What remains is fragmentation
  by chance, such as linear chains, and the question is whether that alone can sustain diversity.
- **Done:** p = 0.05, the same 12 γ values as the all-cycles sweep, × 2 repeats, run to about 33M.
- **Result:** every run sits at about 3.5–4 species (time means 3.47–3.96). Diversity is gone at every
  γ. This agrees with the paper's collapse to one species at γ = 0.05 with short cycles removed.

#### 2026-10-08 15:19: lower γ without short cycles (`run no_short_cycles_perc05_20261008_151903_7dc539e7`, `git 0ed99cb` at 16:11)
- **Done:** γ = 0.02, 0.03, 0.04 and 0.06 × 4 repeats, p = 0.05, 50M.
- **Why (inferred):** in the paper, γ = 0.025 stayed above one species with short cycles removed.
- **Result:** 3.4–4.3 species in all 16 runs. Diversity does not survive even at γ = 0.02. This is
  stricter than the paper's γ = 0.025 case, which still allowed cycles of length 6; it resembles the
  paper's finding that removing even longer cycles sends that case to one species too.

#### 2026-10-09 01:17: grid of p × γ without short cycles (`run no_short_cycles_p_gamma_20261009_011738_ef440b70`)
- **Done:** p ∈ {0.10, 0.20, 0.30, 0.50} × γ ∈ {0.03, 0.05, 0.07, 0.09}, one run each, from one
  species, 50M.
- **Why (inferred):** in September blocking raised diversity with all cycles allowed. Can blocking
  restore diversity once the cycle route is closed?
- **2026-10-09 11:14 (recorded, session):** the runs were paused because results were needed. The
  three sweeps above were plotted together in `nb sim_notebook` cells 16–21, committed on 19 Oct
  (`git d69a5b2`) after all three had finished.
- **Result:**

  | p | γ = 0.03 | 0.05 | 0.07 | 0.09 |
  |---|---|---|---|---|
  | 0.10 | 4.0 | 4.3 | 4.2 | 4.6 |
  | 0.20 | **94.5** | 11.5 | 7.7 | 7.0 |
  | 0.30 | **90.1** | 25.2 | 15.0 | 12.5 |
  | 0.50 | 2.3 | 2.3 | 2.4 | 2.3 |

  - At p = 0.5 the largest cluster is only a few hundred sites, so the habitat is effectively gone.
  - The p = 0.2, γ = 0.03 run rose by itself from about 10 to about 120 species around t = 27M.
- **Discovery:** blocking around 20–30% of the sites restores high diversity at low γ, without any
  cycle of length ≤ 6. With 10% blocked, or with 50% (the habitat falls apart), it does not.
  **Observed** (one run per cell, but p = 0.2 and p = 0.3 agree).

### Phase IV: introduction rate and link density, from established communities (21–26 Oct)

#### 2026-10-21 01:06: does the introduction rate keep diversity up? (`session 2026-10-21`, `nb sim_notebook cell 22`, `run no_short_cycles_alpha_p20_g003_20261021_011454_0f93dba1`)
- **Why (recorded):** to see whether a higher α is what keeps diversity possible.
- **Design:**
  - **Parent A:** the grid run at p = 0.2, γ = 0.03, at t = 50M with 129 species. Only α changes.
  - **α values:** 12, from 0.001 to 0.012, all below 0.0125. Each branch runs for 100M steps
    (t = 50M → 150M).
  - **Seeds:** root 6768. The grid's root, 6767, would have handed one branch the parent's own random
    sequence.
- **Follow-ups:**
  - **Repeat 2** (seed root 6769, the same parent state) was launched on 22 Oct at 00:55.
  - **Fresh runs r3–r5:** 36 runs from one species, each on its own lattice, 0 → 100M, seed 6771.
    They ran on another machine; their analysis records were copied here on 22 Oct at 23:12.

#### 2026-10-22 00:29–00:55: γ swap, to find where diversity is steady, bistable or absent (`session 2026-10-22`, `nb sim_notebook cell 23`, `run no_short_cycles_gamma_p30_from_g003_20261022_005553_5940261e`)
- **Why (recorded):** to find the γ ranges where diversity is constant, bistable and non-existent.
- **Design:**
  - **Parent B:** the grid run at p = 0.3, γ = 0.03, at t = 50M with 99 species.
  - **γ values:** 0.02, 0.03 (the control), 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.09, 0.12 and
    0.15, one run each.
  - **Redraw limit raised to 10⁷:** with short cycles forbidden, γ = 0.2 exhausts the limit of 10⁶
    redraws at once. γ = 0.15 needs about 20,000 redraws per newcomer when 99 species are alive.
- **Checked at 00:54 (recorded question):** every run simulates the largest cluster only. The effective
  blocked fraction is 20.4% for p = 0.2 (31,834 usable sites) and 31.3% for p = 0.3 (27,482 usable
  sites).
- **Side finding:** the cost of forbidding short cycles explodes with γ and with the number of
  species, which itself limits the parameters that can be reached. **Observed.**

#### 2026-10-23 10:53–15:09: the analysis notebook, and an audit of which run came from which (`session 2026-10-23`, `nb sim_analysis`, committed `git e6485f5` 26 Oct 13:35)
- **Done:** `alpha_branched_vs_fresh.ipynb`, renamed `sim_analysis.ipynb` once repeat 2 and the γ
  swap had finished.
- **Methodological discoveries:** these change how the runs may be compared.
  - Repeats r1 and r2 branch from the **same** parent state. They are sibling continuations of one
    community, not independent replicates, so averaging them is invalid.
  - No branch kept α = 0.0125, so the α sweep has no unchanged control.
  - Introductions arrive at α·γ, so changing γ also changes the arrival rate (up to 5 × at γ = 0.15).
  - Plots therefore give each run its own colour and marker and draw no group means; a mean of two
    siblings can show a value that no run ever held.
- **Results, α sweep** (`nb sim_analysis §14`):
  - **α ≤ 0.002:** all four branches stay diverse (92–113 species). Yet none of the 24 fresh runs at
    α ≤ 0.007 or α = 0.009 ever reaches 50 species. A diverse community persists here but does not
    form.
  - **α = 0.003–0.008:** 11 of 12 branches collapse, 5.4–93.0M steps after the branch.
  - **α ≥ 0.009:** branches move in both directions. At α = 0.012 all 3 fresh runs become diverse.
  - **Siblings often disagree:** at α = 0.005, 0.009 and 0.010 one branch collapses while the other
    stays diverse. Within 100M steps the outcome is largely chance.
- **Results, γ swap:**
  - **The control moves without any change:** it first collapses at 19.4M, switches 9 times, and spends
    51% of its second half in the high state (mean 38). Parent B was high for all of 25–50M.
  - **γ = 0.02** is the only value that stays diverse throughout (mean 112).
  - **γ = 0.035–0.05** keeps switching (6–22 switches).
  - **γ ≥ 0.055** holds no diverse state. From γ = 0.07 on, it collapses within 0.5–0.8M steps.
  - **At γ ≥ 0.05** the outcome matches the grid runs started from one species, so the starting state
    is forgotten.
- **Results, both experiments:**
  - No species alive at a branch point survives to t = 150M.
  - The number of introductions matches α·γ·T: the largest of 36 deviations is 2.0 √(α·γ·T).
  - Every final community with ≥ 30 species has a link density (connectance) below γ.

### Phase V: inside one switch (29 Oct – 4 Nov)

#### 2026-10-29 14:35: capturing one rise and one collapse at 1,000-step resolution (`session 2026-10-29`, `tools/run_switch_capture.py`, `run E:\MasterProject\checkpoints\switches_p0.3_a0.0125_g0.03_20261029_150918_cd0568f6`)
- **Why (recorded):** detailed videos of the low → high and high → low transitions, at p = 0.3,
  γ = 0.03, α = 0.0125, with short cycles rejected and the largest cluster only.
- **Design:**
  - Continue the γ = 0.03 control of the γ swap from t = 150M with its own saved random-number state,
    so this is the same trajectory.
  - Keep every state 1,000 steps apart. Stop 2M steps after the first collapse that follows a rise,
    using the notebook's regime rule; a switch counts once the run is 0.8M past it.
  - Verified that saving often and splitting the run into segments leave the trajectory unchanged.
- **Outcome:** started at 15:09. It kept 18,631 states over t = 150–168.63M (6.4 GB on E:) and ran at
  about 4.3M steps per hour.

#### 2026-10-31 00:31–00:56: analysis of the switch (`session 2026-10-31`, `nb analysis/switch_analysis.ipynb`, `analysis/switch_features.py`)
- **Done:** the notebooks moved to `analysis/`. Five lattice snapshots were chosen from the smoothed
  species curve:

  | # | Snapshot | t | Species |
  |---|---|---|---|
  | 1 | before the rise | 153.08M | 13 |
  | 2 | steepest rise | 154.61M | 65 |
  | 3 | peak | 156.29M | 120 |
  | 4 | steepest collapse | 163.95M | 83 |
  | 5 | low after the collapse | 166.74M | 4 |

- **Findings, rise (153.08 → 156.28M):**
  - The largest patch breaks up first, from 62% to 34% of the sites (half-way at 154.25M). The patch
    count follows (half-way at 154.29M), and species come last (half-way at 154.69M). For this rise,
    the paper's "patchiness first" holds.
  - Arrivals stay at about 377 per 1M steps, so the rise comes from fewer losses.
  - The share of newcomers that survive 0.1M steps goes from 7–26% before the rise to 44% during it.
- **Findings, establishment:** a newcomer survives 0.1M steps in 39% of cases when the largest patch
  covers under 20% of the sites, against 14% when it covers over 80%. By phase: 18% before the rise,
  32% while high, 17% after the collapse.
- **Findings, the two states:**
  - **The low state is a succession of sweeps:** the species holding the most sites changes 229–249
    times per 1M steps.
  - **The high state is a fast flow of newcomers:** none of the 13 species at snapshot 1 is alive at
    the peak, and 73% of the 120 species at the peak arrived within the previous 1M steps.
- **Findings, collapse (163.51 → 166.74M):** patchiness leads only at the start. Patches fall fastest at
  163.71M and species at 163.95M. Over the whole collapse, though, species are half-way down (164.49M)
  before patches are (164.87M), and before the largest patch has grown back from 35% to 72%.
- **Findings, cycles:**
  - A strongly connected component of 10 or more species (the species on a shared directed cycle) first
    appears at 154.20M, with 36 species. That is close to 1/γ ≈ 33, where a random network forms one.
  - At equal species count, fewer species lie on a cycle during the collapse than during the rise.
  - About 80% of states are frozen: no boundary allows any invasion.
- **Interpretation (hypothesis):** a feedback through the introduction rule.
  - A newcomer replaces the patch it lands on.
  - On a fragmented lattice, most newcomers land on a small patch of a species that has other patches.
    The newcomer is added rather than swapped in, and its small territory is unlikely to be hit by the
    next arrival. Fragmentation raises diversity, and more species keep the lattice fragmented.
  - Consolidation runs the same loop in reverse.

#### 2026-10-31 03:01–03:20: videos at 1,000-step resolution (`tools/render_switch_videos.py`; `rise.mp4`, `collapse.mp4`)
- **Observation:** a newcomer can sweep through a large patch in under 1,000 steps. At this spacing,
  sweeps show as jumps from one frame to the next, so the mechanism needs every single step.

#### 2026-10-31 17:11–17:36: every-step replays around the rise (`steps_000153951500.mp4`, `steps_000154500000.mp4`, `steps_000155000000.mp4`)
- **Method:** each window is replayed in memory from the nearest kept state, which stores the
  random-number state. Every replayed step that lands on a kept state matched it exactly.
- **Discovery:** the jump in patches at 153.951M is the paper's linear chain A → B → C, with each link
  confirmed in Γ.
  - Species 57627 holds 78% of the lattice in one patch.
  - A newcomer, 57633, sweeps into it (from 1,879 to 16,710 sites).
  - An older species, 57563, can invade the newcomer but not 57627. It follows right behind the
    newcomer's front (from 4,289 to 14,809 sites).
  - Remnants of 57627 survive as islands. Patches climb from about 280 to over 500, and the newcomer is
    extinct by 153.952M.
- The windows at 154.5M and 155.0M are mostly frozen.

#### 2026-10-31 17:46–18:06: every-step replays in the collapse (`steps_000164030500.mp4`, `steps_000164276500.mp4`, `steps_000165631500.mp4`)
- **A, extinction burst at 164.0305M:** a newcomer starts a relay through 6 species in turn. The last
  holds about 69% of the lattice, and 9 older species die as the relay passes.
- **B, the largest loss of patches at 164.2765M** (877 → 647 patches): two relays, each started by a
  newcomer. Islands are erased as the relays pass.
- **C, near-monoculture at 165.6315M:** a newcomer that can invade both remaining large species takes
  98% of the lattice.
- **Synthesis (Observed, 4 events):** both directions run as relays down chains of invasion links. What
  differs is the follower:
  - If the follower cannot invade the original species, remnants survive as islands: fragmentation,
    as in the rise.
  - If each follower takes over the ground in turn, islands are erased: consolidation, as in the
    collapse.

#### 2026-11-04: consolidation (`git 7340282`, `git 4332c16`)
- The capture, analysis and video tools were committed.
- Two tests that failed only because the Windows temp folder had two spellings were fixed; 168 of 168
  tests pass.
- This log was started.

#### 2026-11-04 14:08: the "wave" at t = 153,949,554 is a chain, not a 2-cycle (`session 2026-11-04`, every-step replay)
- **Why (recorded):** in the rise videos, a newly introduced species seemed to move through another
  species like a wave. That suggested a 2-cycle: two species that can each invade the other.
- **Done:** replayed every step from t = 153,947,954 to 153,951,153 from the kept states. Every replayed
  step that landed on a kept state matched it exactly. Γ was then read at 153,949,554.
- **Result:**
  - **No cycle at all:** none of the 28 living species can invade each other (no 2-cycles), and no
    species lies on any directed cycle. Connectance is 0.032 (γ = 0.03).
  - **The wave is a linear chain, 57627 → 57632 → 57538:**
    - The newcomer 57632 arrived at 153,949,351 on 57538, which held 71% of the sites.
    - 57632 invades 57538 through its forced link, and the older species 57627 invades 57632.
    - 57627 cannot invade 57538, and 57538 cannot invade either of them.
  - **The band:** 57632 is a band eaten from behind as it advances. It never held more than 4,331 sites
    and was extinct by 153,950,025, 674 steps after it arrived.
  - **The outcome:** 57627 went from 2,480 to 21,564 sites (78%). 57538 fell from 19,422 to 355 sites,
    which survive as 62 islands (16 before). Total patches went from 237 to 285.
  - **Consequence for the rise:** this relay made 57627 dominant. The next relay
    (57563 → 57633 → 57627, from 153,951,020) broke 57627 up in turn. The rise therefore starts with two
    back-to-back chain relays, each leaving islands of the species it passes over.
- **Why 2-cycles cannot occur here:** the newcomer's forced link to the species it lands on is set
  before the short-cycle check (`sim_modules.py`, introduction step). A newcomer whose prey could invade
  it back would be redrawn, so in these runs any travelling band is a chain, or a cycle of length ≥ 7.
- **Follow-up, 14:43: a re-render coloured by age**
  (`videos/steps_000153951500_age_highlight_57538_57627_57632.mp4`).
  - **Why (recorded):** with colours set by species ID, species could look alike. The request was a
    gradient that follows age, older on the left and newer on the right, with the three species above
    standing out.
  - **Colours:** hue runs from violet (oldest) to pink (newest), and lightness cycles over three levels
    so species of neighbouring ages differ.
  - **Highlighting:** 57538 (cyan), 57627 (red) and 57632 (magenta) are drawn at full colour, and all
    other species are faded.
  - **What it shows:** the magenta band moves through the cyan with red close behind. It is visible
    from 0:24 to 0:35.

#### 2026-11-05 11:29–12:40: one snapshot per million, gathered for the move to another machine (`session 2026-11-05`, `tools/prune_checkpoints.py`, `MS.recreate_checkpoint`)
- **Why (recorded):** the work moves to another environment. The request was to collect every
  checkpoint on this PC into one folder and one zip for Google Drive. Only one snapshot per million
  steps is kept, and the others are rebuilt from the saved random state when needed.
- **Done:**
  - All checkpoints on this PC are in two places: the project's `checkpoints/` (29 folders, 67 GB)
    and the switch capture on `E:` (7.4 GB). The only other copies are old runs in the Recycle Bin
    and a scratch run, which were left out.
  - `tools/prune_checkpoints.py` was rewritten. It used to keep snapshots around swaps; it now
    works per folder (run, batch job or capture segment) and keeps the first snapshot of each whole
    million plus the newest. Missing patch counts are recorded before a snapshot goes. It can thin
    in place (`--apply`), or copy the thinned folders into one folder and zip them
    (`--into … --zip`).
  - `MS.recreate_checkpoint` rebuilds a removed snapshot by replaying from the newest earlier one in
    its folder.
  - The video tool and `switch_analysis.ipynb` now replay states the capture no longer keeps.
    `features.npz` is protected from being rebuilt out of a thinned capture.
- **Result:**
  - **Replay is exact for every run family.** One check per folder, 155 in all, replayed from a kept
    snapshot to the next removed one. Lattice, Γ, species, histories and random state matched every
    time. This covered the early single runs, percolation, largest cluster, the no-short-cycle
    grids, the α and γ branches and the capture segments.
  - **Cost:** replay runs at 1,300–1,800 time units per second, so any snapshot is at most about
    13 minutes from its million.
  - **Size:** 875 MB of the 68.9 GB of snapshots are kept. With the analysis records, plans, capture
    series, features and videos, the copy holds 2,434 files (1.8 GB). `checkpoints.zip` is
    936 MB, and every entry's checksum was tested. The originals were left in place.
- **What it changes:**
  - Analyses that read many snapshots (lattice animations, Γ cycle counts summed over snapshots, the
    zoom on a swap in `sim_analysis`) see one snapshot per million unless states are rebuilt.
  - The capture's 1,000-step series and per-state features are kept whole, so its curves and
    statistics do not change.

#### 2026-11-05 14:19: September runs with cut-off pockets removed (`session 2026-11-05`)
- **Why (recorded):** runs from before the largest-cluster rule are "de facto wrong" and are to be
  redone. The alternative considered was to cut each run down to its largest cluster after the fact.
- **Checked:**
  - 21 run folders predate the rule; their snapshots carry no `largest_cluster_only` field. Blocked
    sites never change, so one snapshot gives each run's habitat.
  - **12 already form one cluster**, so the rule would have produced exactly the same run: every
    p = 0 run (`base_run`, `gamma_02_branch`, γ = 0.07–0.09) and every p = 0.05 run (γ = 0.07–0.10).
    They were kept.
  - **9 have usable sites outside the largest cluster.** At their last snapshots:

    | Run | Cut-off sites | Species in the largest cluster | Species only in pockets |
    |---|---|---|---|
    | γ = 0.07, p = 0.2 | 56 | 49 | 15 |
    | γ = 0.07, p = 0.5 | 19,556 | 1 | 2,603 |
    | γ = 0.15, p = 0.075 | 3 | 3 | 3 |
    | γ = 0.15, p = 0.10 | 4 | 3 | 4 |
    | γ = 0.15, p = 0.125 | 17 | 3 | 16 |
    | γ = 0.15, p = 0.15 | 24 | 5 | 21 |
    | γ = 0.15, p = 0.2 | 56 | 17 | 45 |
    | γ = 0.10, p = 0.15 (branch) | 24 | 25 | 24 |
    | γ = 0.30, p = 0.15 (branch) | 24 | 3 | 24 |

- **Why they were not repaired:**
  - The species and patch series were counted over the whole lattice every 10,000 steps, while
    snapshots exist only at some times. Cutting the snapshots down cannot correct the series.
  - A recount would mean replaying each run from its start and counting only the largest cluster:
    470M time units, roughly 70–100 CPU hours. Two runs no longer have their start state (γ = 0.07,
    p = 0.5 and γ = 0.15, p = 0.10; first snapshots at 8.65M and 29M).
  - Even that would not be the rule's model. The old runs spread introductions over every usable
    site, pockets included, so the largest cluster received 99.8% of them at p = 0.2 but only 2% at
    p = 0.5.
  - They are single runs that are to be redone anyway.
- **Done:**
  - The 9 runs (1.8 GB) were moved out of `checkpoints/`, and their thinned copies out of
    `E:\MasterProject\export`, into folders named `delete_me_pre_largest_cluster_runs` on D: (beside
    the project) and on E:, to be deleted.
  - The export and `checkpoints.zip` were rebuilt without them: 843 MB of 67.1 GB of snapshots kept,
    146 replay checks all exact, 2,370 files (1.8 GB), and a 933 MB zip with every checksum tested.
  - `nb sim_notebook` no longer loads or plots them.
- **What it changes:**
  - D3 is withdrawn. Whether blocking raises diversity when all cycles are allowed is open again.
  - The 28 Sep caveat that a pocket artefact was unlikely was wrong. Every pocket keeps a species of
    its own, however small it is.

---

## 2. Register of discoveries

| ID | Finding | Status | First seen | Evidence |
|---|---|---|---|---|
| D1 | At the paper's Fig. 3 parameters the model switches repeatedly between high and low diversity | Observed (1 run) | 30 Aug | `run gamma_07_alpha_0125` |
| D2 | With all cycles allowed, p = 0.05 and α = 0.0125, the high state is lost between γ = 0.073 and 0.09 | Established (24 runs) | 6 Oct | `run gamma_sweep_perc05_largest_…` |
| D3 | ~~Blocked sites raise diversity when all cycles are allowed (γ = 0.15: 7 → 52 species as p goes 0.075 → 0.2)~~ | Withdrawn 5 Nov: mostly species in cut-off pockets; the largest cluster held 3–5 species for p ≤ 0.15 and 17 at p = 0.2 | 28 Sep | `run gamma_15_alpha_0125_perc_*` (removed 5 Nov) |
| D4 | A habitat broken into pockets creates spurious diversity, even when the pockets are single sites; hence the largest-cluster rule | Established (methodological) | 30 Aug / 7 Oct; 5 Nov | `run gamma_07_alpha_0125_perc_50` (removed 5 Nov; 2,603 of 2,604 species outside the largest cluster), `git 9798582` |
| D5 | With cycles of length 2–6 rejected and p ≤ 0.1, diversity is lost at every γ from 0.02 to 0.12 | Established (44 runs) | 8 Oct | three sweeps, Phase III |
| D6 | With 20–30% of sites blocked, high diversity returns at γ = 0.03 without short cycles (90–95 species), but not at γ ≥ 0.05, nor at p = 0.1 or 0.5 | Observed (1 run per cell; p = 0.2 and 0.3 agree) | 9 Oct | `run no_short_cycles_p_gamma_…` |
| D7 | Hysteresis in α: at α ≤ 0.002 an established community persists, but fresh runs need α ≳ 0.008–0.012 to become diverse | Observed (2 siblings + 3 fresh runs per α) | 23 Oct | `nb sim_analysis §2–7` |
| D8 | A community's fate is largely chance: sibling branches diverge, and the unchanged control drops from 100% to 51% of its time high | Established for these runs | 23 Oct | `nb sim_analysis §7, §11` |
| D9 | γ swap from a diverse state: only γ = 0.02 holds, 0.035–0.05 flicker, γ ≥ 0.055 is lost, and at γ ≥ 0.05 the starting state is forgotten | Observed (1 run per γ) | 23 Oct | `nb sim_analysis §10–12` |
| D10 | The diverse state is a flow of newcomers, not a stable assembly: no species alive at a branch survives 100M, and 73% of peak species are under 1M steps old | Established (all branches) + Observed (capture) | 23 Oct / 31 Oct | `nb sim_analysis §9`, `nb switch_analysis §7` |
| D11 | The rise starts with the break-up of the dominant patch (patchiness first) | Observed (1 rise) | 31 Oct | `nb switch_analysis §3, §5` |
| D12 | Newcomers establish more often on a fragmented lattice (39% against 14%) | Observed (1 capture, many newcomers) | 31 Oct | `nb switch_analysis §5` |
| D13 | Without short cycles, patches form by linear chain relays whose follower cannot invade the original species; collapses are relays that erase islands. Travelling bands in the videos are such chains, not 2-cycles | Observed (5 events) | 31 Oct; 4 Nov | every-step videos; replay at 153,949,554 |
| D14 | Cycles in Γ track the species count (a large cycle-connected group appears near S ≈ 1/γ), and about 80% of states are frozen, so cycles in Γ are rarely active on the lattice | Observed (1 capture) | 31 Oct | `nb switch_analysis §6` |
| D15 | The cost of rejecting short cycles explodes with γ and species count (γ = 0.2 is unreachable at 99 species) | Observed (technical) | 22 Oct | `session 2026-10-22` |

**Relation to the paper.** D1, D2, D5 and D11 agree with it. D6 and D13 qualify its conclusion. In a
habitat with blocked sites, high diversity arises at low γ with no cycle shorter than 7, and the patch
creation observed so far runs through linear chain relays. The paper found chains alone insufficient
on an unblocked lattice. Cycles of length ≥ 7 are still allowed here, so their role is not excluded.

---

## 3. Open questions and hypotheses

- **Q1. What breaks up the dominant patch in a rise?**
  - One rise starts with two back-to-back linear chain relays: at 153.9494M and at 153.951M.
  - Is that the typical trigger? Answering this needs more rises, and an automated classifier for
    relay events: newcomer, follower, and whether the follower can invade the original species.
- **Q2. Why do blocked sites replace short cycles?** *Hypothesis:*
  - Blocked sites act as barriers that stop a sweep, or its follower, from reaching every remnant, so
    chains leave islands more often.
  - Blocked sites also shield islands from later sweeps.
  - *Test:* how island survival depends on distance to blocked sites; and how often relays leave
    islands at p = 0 compared with p = 0.3.
- **Q3. Is there an optimum in p?** No diversity at 0.1, high at 0.2–0.3, none at 0.5. This needs a
  finer grid in p with replicates, ideally compared at equal habitat size, since the largest cluster
  shrinks as p rises.
- **Q4. Is the hysteresis in α real bistability, or just too little time?** This needs longer runs, more
  independent parents, and a control that keeps α = 0.0125.
- **Q5. Does γ act through link density or through the arrival rate α·γ?** Separate the two by varying
  γ while holding α·γ fixed.
- **Q6. Does the follower relation predict the direction of a switch?** Compare the share of relays
  that leave islands with the share that erase them, over many rises and collapses.
- **Q7. Comparison with paper Fig. 6.** Our rejection allows cycles from length 7, in the full
  dynamics, whereas the paper's runs were quasistatic. Is the difference in outcome at p = 0 due to the
  dynamics, or only to the rejection threshold?

---

## 4. Methodological lessons

- **Branches:**
  - A branch keeps its parent's lattice and Γ and draws new random numbers from its own seed.
  - Branches from one state are siblings, not replicates. Their lineage is recorded in
    `batch_plan.json`, never in folder names.
  - Reusing a seed root reproduces a random sequence, which is why 6767 was avoided for branches of the
    grid.
- **Older branch folders** (from before branches started their own history) contain the source run's
  history. Use `start_timestep` when plotting them.
- **Changing γ** also changes the introduction rate α·γ.
- **Effective p** exceeds nominal p once the cut-off pockets are blocked: 20.4% for p = 0.2 and 31.3%
  for p = 0.3.
- **No pocket is too small to matter:** a cut-off pocket, even a single site, keeps a species of its own
  until a newcomer lands on it. A few dozen cut-off sites therefore add a few dozen species to the
  count. Only runs whose usable sites form one cluster are kept.
- **Two switch detectors:**
  - The simulator keeps snapshots around swaps found with thresholds relative to each run's own level.
  - The notebooks count regimes with fixed thresholds of 10 and 50 species.
  - The two do not always agree.
- **Cycle counts above length 4** are sampled estimates, and counts sum over snapshots.
- **Speed changes are verified bit-exact:** every kernel change is checked against full trajectories,
  including the random-number state.
- **Any snapshot can be rebuilt:** the random draws do not depend on how the steps are split between
  calls, so replaying from a saved snapshot reproduces the run exactly. Since 5 Nov only one snapshot
  per million is kept, and a rebuild costs up to about 13 minutes.
- **Plotting:** one colour and marker per run, kept fixed, and no means over n ≤ 3 sibling runs.

---

## 5. Where the data lives

| Data | Location |
|---|---|
| Early single runs (Aug–Sep), p = 0 and 0.05 only | `checkpoints/gamma_*`, `checkpoints/base_run`, `checkpoints/gamma_02_branch` |
| Sweeps and grids (Oct) | `checkpoints/gamma_sweep_perc05_largest_…`, `checkpoints/no_short_cycles_*` |
| Fresh α runs r3–r5 (analysis records only) | `checkpoints/alpha_p20_g003_fresh_rep3-5_analysis` |
| Switch capture, features and videos | `E:\MasterProject\checkpoints\switches_p0.3_a0.0125_g0.03_20261029_150918_cd0568f6` |
| All of the above, one snapshot per million (5 Nov) | `E:\MasterProject\export\checkpoints` and `checkpoints.zip` beside it; unzip at the project root elsewhere |
| Analyses | [analysis/sim_notebook.ipynb](analysis/sim_notebook.ipynb), [analysis/sim_analysis.ipynb](analysis/sim_analysis.ipynb), [analysis/switch_analysis.ipynb](analysis/switch_analysis.ipynb) |

---

## 6. Adding to this log

Add each new step at the end of section 1, under a heading with the date and local time. Give it:
- the question and **why** it was asked (recorded or inferred);
- what was done, with the run folder and seed root;
- the result, in numbers;
- what it changes in the understanding.

Then add any new finding to section 2 with its status, and move answered questions in section 3 to
the timeline entry that answered them.
