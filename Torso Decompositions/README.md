# SpOC-3 Torso Decompositions — A Local-Search & Metaheuristics Study

> **Adrian Ymeri — PhD work (May 2026)**
> Progressive Hill Climbing variants plus a Simulated Annealing / GRASP /
> VNS metaheuristics chapter on ESA's
> [SpOC-3 Torso-Decompositions challenge](https://optimize.esa.int/challenge/spoc-3-torso-decompositions/About).

> **Note — this README covers the permutation-space chapter only.** The
> project's current best results come from the later *continuous-encoding +
> GBDT* paradigm (GBFC §11 and GBFC++ §11.6): **99.990 % / 98.14 % / 98.89 %**
> of the leaderboard top on small / medium / large. See
> **[docs/THESIS.md](docs/THESIS.md)** (the centerpiece) and
> **[docs/RESULTS.md](docs/RESULTS.md)**; the GBFC++ breakpoint-boosting method
> and its verified small-graph **−1,829,735** are in THESIS §11.6.

## Abstract

This repository accompanies the first chapter of a planned paper series
on multi-objective optimisation of treewidth-like graph decompositions.
It contains:

- A self-contained, bit-correct evaluator and hypervolume implementation
  (`core.py`) verified against an independent reference implementation
  (`tests/test_correctness.py`).
- **15 Hill Climbing variants** (`algorithms/hill_climbing/hc1_initial` … `hc15_hv_incremental`)
  covering the full classical taxonomy: random restarts, archive-based,
  warm-started, multi-operator (including 2-opt and 3-opt primitives),
  per-t local search, structural moves, late acceptance, HV-aware
  acceptance, torso-aware warm start, iterated local search,
  incremental evaluation, tabu memory, and stochastic best-improvement
  (steepest ascent).
- A **metaheuristics chapter** — **Simulated Annealing**, **GRASP**, and
  **Variable Neighborhood Search** — built on the same `core.py` /
  `meta_common.py` scaffolding (shared 15-operator pool, shared Pareto
  archive, shared HV-improvement acceptance) so the four families are
  directly comparable.
- 51 verified official-instance submissions (15 HC + 3 metaheuristics ×
  3 problems) and 260 verified runs on a 20-instance synthetic grid
  (13 variants × 20 instances) spanning sizes 200–1000 and densities 3–35.
- A correctness suite, a stand-alone submission re-evaluator, and a
  reproducible benchmark harness.

The codebase remains the foundation for follow-up chapters on NSGA-II and
HV-driven evolutionary algorithms; only the acceptance rule changes
between families.

## Headline results

Best result across **all four families** vs the official ESA leaderboard
(a more negative −HV is better):

| Problem | Our best (overall) | Best HC (chapter 1) | Leaderboard top | Gap (HV) | % of leaderboard |
|---|---:|---:|---:|---:|---:|
| small-graph (n = 1 357, sparse) | **−1 816 174** (`GRASP`) | −1 814 976 (`hc15`) | −1 829 919 | +13 745 | **99.25 %** |
| medium-graph (n = 1 399) | **−1 615 368** (`GRASP`) | −1 609 194 (`hc13`) | −1 745 122 | +129 754 | **92.57 %** |
| large-graph (n = 2 426, dense) | **−4 904 767** (`GRASP`) | −4 800 196 (`hc13`) | −5 493 062 | +588 295 | **89.29 %** |

> **GRASP wins all three official instances**, and its `large-graph`
> result (−4 904 767) is a **new project best — ~104 600 HV beyond the
> best HC** (hc13, −4 800 196) at an honest on-budget 25.0 s, robust
> across seeds 42/1/2 (−4 904 767 / −4 916 147 / −5 010 404).  The full
> metaheuristics scoreboard (GRASP > VNS > SA on every instance) and
> tuning are in [docs/RESULTS.md](docs/RESULTS.md) §4b.

### Hill-Climbing chapter (chapter 1)

Within the HC family alone, the best variants vs the leaderboard:

| Problem | Best HC | Leaderboard top | % of leaderboard | % behind |
|---|---:|---:|---:|---:|
| small-graph (n = 1 357, sparse) | −1 814 976 (`hc15` HV-accept + incremental) | −1 829 919 | **99.18 %** | 0.82 % |
| medium-graph (n = 1 399) | −1 609 194 (`hc13` Tabu Search) | −1 745 122 | **92.21 %** | 7.79 % |
| large-graph (n = 2 426, dense) | −4 800 196 (`hc13` Tabu Search) | −5 493 062 | **87.39 %** | 12.61 % |

> **Multiseed-confirmed (12 seeds), modest margin.**  `hc15`
> (HV-improvement acceptance running on hc12's incremental evaluator)
> edges the previous best `hc9` on the two sparse/medium official
> instances.  At seed 42 it leads by +51 HV on small (−1 814 976 vs
> −1 814 925) and +669 HV on medium (−1 608 632 vs −1 607 963), both
> submission-verified.  (On medium the overall canonical best is
> `hc13` at −1 609 194; hc15 leads the multi-seed *mean*, see below.)  The **multiseed sweep** (`make multiseed-official`,
> seeds 1–11 + canonical 42) confirms the direction: hc15 has the **best
> mean** of {hc5, hc9, hc11, hc14, hc15} on both small (−1 814 872 vs hc9
> −1 814 394) and medium (−1 609 238 vs hc9 −1 608 923), and wins the
> head-to-head against hc9 on **7/11** small and **8/11** medium seeds.
> The margin (≈ 300–500 HV, ≈ 0.02–0.03 %) is real but smaller than the
> seed-to-seed spread, so hc15 is best-by-a-whisker, not dominant.  On
> large, `hc13` (Tabu) remains the headline; hc15's 6-operator
> incremental set (−4 793 629) trails hc13's 15-operator structural
> moves by 6 567 HV, and across the 11 sweep seeds hc15 is the *most
> stable* large variant (sd 204) but not the strongest mean.

Reading: on `small-graph` our best HC achieves **99.18 %** of the
leaderboard top's HV — we are **0.82 % behind** the top.  The gap
widens on medium (**7.79 % behind**) and large (**12.61 % behind**)
because the leaderboard tops on those denser instances were almost
certainly produced by exact / near-exact treewidth solvers with
parallel compute, not single-thread HC; see
[docs/FUTURE.md](docs/FUTURE.md) §3 for the analysis.

On the 20 synthetic instances (1 s wall budget per run, 3 seeds per
cell, all 13 variants): by per-cell mean, **hc9** (HV-improvement
acceptance) wins **10 / 20**, the **hc12** (incremental evaluator)
wins **6 / 20** (the higher-density and larger cells where
moves-per-second binds), **hc2** (random restarts) wins **3 / 20**
(the three over-width cells, where every variant ties at the
fallback and hc2 takes the lexicographic tiebreak), and **hc10**
(per-t torso-aware warm start) wins **1 / 20** (the sparsest n = 500
cell).  A Friedman test
across the 13 variants gives χ² = 112.99 (df = 12, p = 1.5e-18); by the
Nemenyi critical difference at α = 0.05 (CD = 4.08) hc9 (mean rank
2.73) is significantly better than hc1, hc2, hc3, hc6, hc8 and hc10 but
not significantly better than hc4, hc5, hc7, hc11, hc12 or hc13.  The
critical-difference diagram is rendered by `tools/cd_diagram.py`
(`extra_instances/cd_diagram.png`); regenerate it on the reference
workstation with `make cd-diagram` so it reflects that machine's
`results.csv`.

The strongest algorithmic finding is that **HV-improvement
acceptance** (Beume et al. 2007 / Zitzler & Künzli 2004) wins on
small and medium official instances and 10 of the 20 synthetic
instances — confirming that when the scoring indicator is HV, the
working-solution acceptance rule should be HV too.  The refinement in
`hc15` sharpens this: running that same HV-acceptance rule on hc12's
**incremental evaluator** (more HV-improving moves per second) edges
plain hc9 on both sparse/medium official instances — by the best mean
across 12 seeds and on 7/11 (small) and 8/11 (medium) head-to-head
seeds — i.e. HV-aligned acceptance *and* evaluator throughput compose
rather than trade off (the margin is modest, ≈ 0.02–0.03 %, within the
seed-to-seed spread).  Adding an explicit **3-opt** primitive (double-reverse of
two adjacent segments) further improves hc9 on small by ≈ 1 500 HV.  On the
densest official instance (large), the seed-42 winner is
`hc13_tabu` (−4 800 196), narrowly ahead of a tight cluster of
hc11, hc7 and hc5 within ≈ 4 100 HV units — on this instance the
twelve-variant comparison is operator-driven and the ranking is not
statistically distinguishable across the top group.
The split between sparse-instance HV-acceptance dominance and
dense-instance structural-move dominance is the headline of the
chapter.

## Quick start

```bash
make help            # list all targets
make test-quick      # toy + HV edge cases (~1 s)
make test            # full correctness gauntlet (~30 s)
make verify          # re-verify every saved submission
make bench           # run hc5 + hc9 on the 20 extra instances
make matrix-small    # run every hc on small-graph
```

To run a single variant on a single problem:

```bash
python3 algorithms/hc9_hv_accept.py --problem small-graph --budget 25 --seed 42
```

To inspect the structure of a single solution end-to-end (the
canonical entry point if you are reading this project for the first
time):

```bash
python3 algorithms/hc1_initial.py --problem small-graph --budget 30 \
    --seed 42 --show-final
```

## Documentation

Detailed documentation lives under [`docs/`](docs).  Two reading
paths depending on your background:

- **Reviewer fluent in metaheuristics / HC, never heard of torso
  decomposition?**  Go to [docs/PROBLEM.md § 0](docs/PROBLEM.md#0-primer-for-reviewers-fluent-in-metaheuristics) — a ~5-minute primer that defines the problem in
  standard graph-theory vocabulary (elimination ordering, chordal
  fill-in, treewidth, hypervolume scoring), with anchor citations.
  Then jump to [docs/ALGORITHMS.md](docs/ALGORITHMS.md).
- **Never seen treewidth or hypervolume?**  Start at
  [docs/EXPLAINER.md](docs/EXPLAINER.md) — a from-scratch buildup
  with pictures, then come back to PROBLEM.md.

| Document | What it covers |
|---|---|
| [docs/EXPLAINER.md](docs/EXPLAINER.md) | **Start here if you have never heard of treewidth, chordal completion, or hypervolume.** Builds the problem up from "what is a graph?" with pictures, real-world analogies, and a step-by-step worked example. No background needed. |
| [docs/PROBLEM.md](docs/PROBLEM.md) | **§ 0 primer for fluent reviewers** + formal problem statement + constraints + hand-computed 10-vertex toy walkthrough. |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | The 15 Hill Climbing variants + the SA / GRASP / VNS metaheuristics chapter (§ 7), full catalogue, operator reference (incl. 2-opt / 3-opt), per-variant rationale, and what each taught us. |
| [docs/RESULTS.md](docs/RESULTS.md) | The cross-algorithm scoreboard (HC + metaheuristics § 4b) on official + synthetic instances, lessons learned, and reproducibility notes. |
| [docs/FUTURE.md](docs/FUTURE.md) | The publishable conclusions for both chapters, the planned next chapters (NSGA-II, SMS-EMOA, HV-EA), and the open gaps. |

## Project layout

```
core.py                         shared evaluator, HV, archive, warm starts
Makefile                        canonical task runner
README.md                       this file

algorithms/
  meta_common.py                  shared scaffolding for the metaheuristics
  hill_climbing/
    hc1_initial.py … hc15_hv_incremental.py  15 Hill Climbing variants
  simulated_annealing/sa.py       Simulated Annealing (chapter 2)
  grasp/grasp.py                  GRASP (chapter 2)
  vns/vns.py                      Variable Neighborhood Search (chapter 2)

tools/
  verify_submission.py          standalone re-evaluator for any submission JSON
  generate_instances.py         reproducible synthetic instance generator
  bench_extra.py                 13-algo × 20-instance benchmark harness
  multiseed.py                   3-seed sweep harness (hc5/hc9/hc11/hc14)
  convergence_plots.py           collect + plot archive-HV vs wall time

tests/
  test_correctness.py           evaluator + HV + submission round-trip suite

data/                           official ESA instance graphs
  small-graph.gr  medium-graph.gr  large-graph.gr

submissions/                    canonical-format JSON outputs
  small-graph/  medium-graph/  large-graph/
    hc1.json … hc15.json        one per HC variant, re-verifiable
    sa.json grasp.json vns.json metaheuristics (chapter 2)
  archived/                     Adrian's previous personal bests

extra_instances/                synthetic instances + benchmark output
  data/                         20 Erdős–Rényi random graphs
  instances.csv                 per-instance metadata
  results.csv                   220-run benchmark output
  summary.md                    pivot table + wins-per-algorithm

docs/                           detailed documentation (see table above)
```

## Reproducing the headline numbers

Every score in this repository can be re-verified end-to-end:

```bash
make test                 # confirms evaluator + HV + every saved JSON reproduce
make verify               # prints the score of every submission JSON
```

`make test` runs the **correctness gauntlet**, which checks:

1. The 10-vertex toy example reproduces the hand-computation
   (fitness `(2, 6)`, single-point HV 32, two-point HV 34).
2. Hypervolume edge cases (empty front, point at origin, point at
   reference, point past reference, duplicates, over-width point
   shadowed).
3. 100 random Pareto fronts agree with a brute-force grid-scan HV.
4. 50 / 10 / 5 random `(π, t)` pairs on small / medium / large
   agree with a slow set-based reference evaluator.
5. **Every saved submission re-evaluates to its claimed score** via
   `tools/verify_submission.py`.

If any of these regress, no other number in the repository should be
trusted.

## License & citation

This is research code accompanying an in-progress PhD thesis. If you
build on it, please cite the eventual paper (or contact Adrian for
the current draft).
