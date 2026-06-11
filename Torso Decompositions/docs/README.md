# SpOC-3 Torso Decompositions — Documentation

**Adrian Ymeri** · University of Prishtina

This is the complete documentation set for my work on the European Space
Agency's SpOC-3 Torso Decompositions problem. It is written to be read as a
single thesis: a formal problem statement, four chapters of permutation-space
search and the ceiling they reach, the continuous-encoding paradigm that breaks
through that ceiling, and a correctness audit underwriting every number.

## Reading order

| # | Document | What it is |
|---|---|---|
| 0 | [EXPLAINER.md](EXPLAINER.md) | The problem and the result in plain language — start here. |
| 1 | [PROBLEM.md](PROBLEM.md) | Formal problem statement: decision, objectives, hypervolume scoring. |
| 2 | [ALGORITHMS.md](ALGORITHMS.md) | The permutation-space families (hill climbing, SA, GRASP, VNS, ACO, NSGA-II/SMS) and their design. |
| 3 | [RESULTS.md](RESULTS.md) | The full scoreboard: per-family, per-instance, with statistical separation. |
| 4 | **[THESIS.md](THESIS.md)** | **The unified narrative** — permutation ceiling → continuous-encoding breakthrough → results → conclusion. The centrepiece. |
| 5 | [AUDIT.md](AUDIT.md) | Correctness and methodology audit: evaluator, hypervolume, archive, parameters. |
| 6 | [FUTURE.md](FUTURE.md) | Research log: continuous-encoding (§1g), GPU scale-up (§1h), GAPS (§1i), and GBFC + the hybrid scale-up (§1j). |

Figures referenced throughout live in [`figures/`](figures/) and are regenerated
from the verified results by `tools/make_figures.py`.

## The result in one table

The headline of the whole project: a continuous-encoding method, re-implemented
on a CPU from the two top leaderboard solutions, lifts every instance from the
permutation-space ceiling to within touching distance of the leaderboard top.

| Instance | Permutation portfolio | Continuous + GBDT | Leaderboard top | % of top |
|---|---:|---:|---:|---:|
| small  | −1,819,283 | **−1,828,451** | −1,829,919 | 99.92 % |
| medium | −1,617,086 | **−1,711,954** | −1,745,122 | 98.10 % |
| large  | −5,033,531 | **−5,399,072** | −5,493,062 | 98.29 % |
| large (GPU §9) | — | **−5,405,118** | −5,493,062 | **98.40 %** |
| large (GAPS §10) | — | **−5,431,595** | −5,493,062 | **98.88 %** |
| **small (GBFC §11)** | — | **−1,828,994** | −1,829,919 | **99.95 %** |
| **medium (GBFC §11)** | — | **−1,712,688** | −1,745,122 | **98.14 %** |
| **large (GBFC §11)** | — | **−5,431,924** | −5,493,062 | **98.89 %** |
| **small (GBFC++ §11.6)** | — | **−1,829,735** | −1,829,919 | **99.990 %** |

The **GBFC rows are the thesis's primary novel contribution** (§11):
*Gradient-Boosted Front Construction* reframes Pareto-front optimisation as a
boosting problem — GBDT weak learners fit to the lower-bound residual, decoded
into complementary specialist orderings. It is the only method here that
measurably improves the banked front, on every instance (verified
+688 / +734 / +329 HV; `tools/gbfc.py`). **GBFC++** (§11.6, `tools/gbfcpp.py`)
breaks GBFC's plateau on small by boosting at the granularity of single front
breakpoints, with the GBDT as a *learned move-proposal policy* inside an
incremental local search (C evaluation kernel, ~25× the Python walk): verified
**+741 HV over GBFC**, GBDT contribution isolated by paired same-seed ablation
(3/3 wins, mean +44.7 vs +18.7 HV per round); runs are checkpointed and
resumable, and gains had not flattened when the CPU budget ended.

Supporting novel pieces: **GAPS** (§10), a GBDT-augmented *nonlinear* decode that
lifts large to **98.88 %** with a controlled **+24,766 HV** from the GBDT column
alone (`tools/gaps_search.py`); the **GBDT adaptive constructor** (§6b,
+2,524 HV, the *generalising* GBDT result, LightGBM/XGBoost); and a validated
**GPU-parallel evaluator** (§9, +6,046 HV) whose plateau localised the residual
to a *decode-expressiveness* limit. None reaches #1 — the treewidth lower bound
(§5.2) certifies the dense core is provably optimal — which the docs report
plainly.

![Closing the gap.](figures/fig1_gap_closing.png)

All scores are canonical-scorer-verified (`tools/verify_submission.py`),
0-capped, and reproducible from the submissions in `submissions/`.
