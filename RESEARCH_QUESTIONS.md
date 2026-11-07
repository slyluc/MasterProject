# Research question log
Place for all active questions being explored.

## Mechanics behind rise and fall of diversity (short cycles rejected)
At the core of the model is what environment allows for high diversity. To analyse this we try to find why and how high diversity can appear, and also how and why it collapses.

- **Setting:** invasion cycles of length 2–6 are rejected; cycles of length 7 and up are still allowed. p = 0.3, γ = 0.03, α = 0.0125, largest cluster only.
- **Evidence so far:** one rise (153.08 → 156.28M) and one collapse (163.51 → 166.74M) from the same run, with five events replayed step by step (see [RESEARCH_LOG.md](RESEARCH_LOG.md), Phase V). Everything below comes from this one pair of switches.

### Generator of high diversity

- High patchiness appears first, which seems to lay the ground for diversity to also rise. In the captured rise the largest patch breaks up first (62% → 34% of the sites), the patch count follows, and species come last. *Observed, one rise.*
- Patchiness is created by multi-species activity. Even without cycles, a chain A → B → C allows wave-like dynamics in which patches are created (replays at 153.9495M and 153.951M).
  - C starts off as a large cluster, which B begins eating away at. A follows right behind B, eating it up. Since A can't eat C, small pockets of C get left behind inside A even when all of B has been eaten. At 153.9495M, C held 71% of the sites, B was extinct 674 steps after it arrived, and C survived as 62 islands.
  - The relay also hands the ground to A: after the first relay, A held 78% of the lattice. It took a second relay of the same kind, right after, to break A up. The rise started with two relays back to back.
- The relays create patches, not species. Arrivals stayed at about 377 per 1M steps through the rise. What rose was the share of newcomers that survive 0.1M steps: 7–26% before the rise, 44% during it. The broken-up lattice is what lets newcomers establish.
- **Open:** are chain relays the usual trigger of a rise? Only one rise has been looked at (Q1 in the research log).
- **Open:** do blocked sites (higher p) make relays leave more islands? All replays so far are at p = 0.3, and the comparison with p = 0 has not been done (Q2 in the research log).

### Generator of diversity collapse

- The collapse also runs as relays down chains of invasion links. It is not mainly one species sweeping alone:
  - **164.0305M:** a newcomer starts a relay through 6 species. The last holds about 69% of the lattice, and 9 older species die as the relay passes.
  - **164.2765M:** two relays, each started by a newcomer, erase islands as they pass (877 → 647 patches).
  - **165.6315M:** a single newcomer that can invade both remaining large species takes 98% of the lattice.
- The single-species sweep came late. Species were already half-way down by 164.49M, before the largest patch had grown back. It looks like the last step of the collapse, not its main cause.
  - The ideal example of that last step: a newcomer that can eat the former largest species and some of the others, without being more prone to being eaten than the former largest species.
- What separates the two directions is the follower (*observed in 4 events*):
  - If the follower can't invade the original species, islands survive: fragmentation, as in the rise.
  - If each follower takes over the ground in turn, islands are erased: consolidation, as in the collapse.
- The low state itself is a succession of single-species sweeps: the species holding the most sites changes 229–249 times per 1M steps.

### Cycles

- The relay at 153.9495M happened with no cycles at all among the 28 living species.
- A group of 36 species sharing directed cycles (of length 7 or more) first appeared at 154.20M, during the rise. That is close to 1/γ ≈ 33, where a random network forms one.
- **Open:** do these long cycles help sustain the high state, or are they a side effect of having many species?

### Following questions

- These dynamics seem quite random, and can happen one after each other, so why do we see sharp periods of diversity emergence and sharp periods of diversity collapse, and not just randomly shifting diversity and patchiness?
  - Some underlying feature must push systems more towards rise than collapse and vice versa.
  - **Candidate** (hypothesis in the research log, 31 Oct): a feedback through the introduction rule. A newcomer replaces the patch it lands on. On a broken-up lattice that is usually a small patch of a species that has others, so the newcomer is added rather than swapped in, and its small territory is unlikely to be hit by the next arrival. Fragmentation raises diversity, and more species keep the lattice fragmented. Consolidation runs the same loop in reverse.
  - **Support:** a newcomer survives 0.1M steps in 39% of cases when the largest patch covers under 20% of the sites, against 14% when it covers over 80%.
- What decides whether a relay leaves islands or erases them?
  - Rough starting point: if links were independent, a follower A that can invade B fails to invade C with probability 1 − γ ≈ 0.97. Relays that leave islands should then be the usual kind, and erasing is what needs explaining.
  - One possibility is long relays: a relay of k species contains one that can invade the islands with probability about 1 − (1 − γ)^k.
  - Short-cycle rejection makes links not quite independent, so this is only a sketch.

### Towards mathematical models

- The evidence is one rise, one collapse and five replayed events, so the mechanisms above are hypotheses to test, not facts to fit.
- **Switching level:** model the species count as arrivals at rate α·γ, multiplied by an establishment probability that depends on how broken up the lattice is. The 39% and 14% figures are a first data point. Does the feedback produce two stable states with sharp switches between them?
- **Relay level:** model the B band moving between A and C, and how much of C is left as islands. How does that depend on p?
- Both levels need counts across many rises and collapses. A classifier for relay events (newcomer, follower, and whether the follower can invade the original species) would give them (Q1 and Q6 in the research log).
