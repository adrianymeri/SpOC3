# Where this is going

> **STATUS — chronological research log (not the final results).** This document
> records the project's evolution and the reasoning at each stage. Scores quoted
> here are **intermediate**, captured at the time each section was written, and
> are *superseded* by the final, verified results in
> [THESIS.md](THESIS.md) / [RESULTS.md](RESULTS.md)
> (small −1,828,451 · medium −1,711,954 · large −5,399,072). Read this for the
> journey and the rationale; cite THESIS/RESULTS for numbers.

_The Hill Climbing chapter, the Simulated Annealing / GRASP / VNS
metaheuristics chapter, the Ant Colony Optimization bridge chapter, **and** the
NSGA-II / SMS-EMOA population chapter are all closed.  This document collects
the publishable conclusions, the three concrete failure modes the HC chapter
exposed, what the metaheuristics chapter added (§ 1b), why the ACO bridge
chapter is a clean negative result (§ 1c), what the population chapter
established (§ 1d), the **ceiling analysis** that measured every remaining
non-algorithmic lever and reached the project's limit (§ 1e), the
final-validation hand-off script for the user's machine (§ 1f), the candidate
research directions for any **next** chapter (§ 3), and honest expected-impact
assessments for each._

---

## 1. What the Hill Climbing chapter established

Across 3 official instances × 14 variants + 20 synthetic instances ×
13 variants (hc14 was added after the synthetic sweep) at seed
42, the chapter supports four publishable conclusions.  Each is
grounded in a specific row of the scoreboard (see
[RESULTS.md](RESULTS.md)).

1. **The Pareto archive subsumes classical HC-escape machinery.**
   Tabu memory (`hc13`), late acceptance (`hc8`), and ILS-style
   perturbation (`hc11`) all reach parity with plain archive-
   extending HC (`hc5`) on every official instance once 3-opt is
   in the operator mix.  Each technique was designed to manufacture
   diversity for *single-objective* HC stuck on a flat plateau;
   the Pareto archive itself supplies that diversity for free —
   every archive add forces the focus onto a new `(max_degree, t)`
   region, so the cycling / lock-in failure modes those techniques
   address do not arise here.  The `tabu-rejected` counter on
   `hc13` literally stays at zero across the entire matrix.  This
   is one of the strongest findings of the chapter and is itself
   publishable as a null result.
2. **Warm start beats every operator innovation.**  `hc3 → hc4`
   (+75 000 HV on small) is the largest delta in the table by an
   order of magnitude.  Operator diversity, K-vertex coordinated
   moves, and per-t local search all contribute within seed noise of
   the warm-start baseline.  The single most important design
   decision in the Hill Climbing family is *the construction
   heuristic*, not the local-search policy.
3. **HV-aware acceptance beats lex acceptance on sparse graphs.**
   `hc9` (SMS-EMOA-style HV-improvement acceptance) wins on the
   synthetic grid (12 / 20 instances by mean rank), and the *same
   acceptance rule on hc12's incremental evaluator* (`hc15`) is now the
   best on small and medium — multiseed-confirmed (best mean over 12
   seeds, winning 7/11 and 8/11 head-to-head seeds vs hc9; RESULTS
   § 4.3).  Together this is the HV-acceptance family winning the two
   sparser official instances plus 12 / 20 synthetic — 14 of 23
   measured cells.  On the densest
   official instance (large) the winner is `hc7_kbottleneck` with
   its K-vertex coordinated bottleneck-relocate move, ahead of hc9
   by 9 022 HV.  The split between sparse-instance HV-acceptance
   dominance and dense-instance structural-move dominance is the
   chapter's headline.  Confirms Beume, Naujoks & Emmerich (2007)
   and Zitzler & Künzli (2004) where the indicator drives the
   acceptance rule, qualified by the dense-instance exception.
4. **Faster evaluation compounds.**  `hc12`'s incremental bitset
   evaluator buys 2–5× iteration throughput at fixed wall budget.
   The win is per-iteration cost, not per-iteration quality, but it
   multiplies the effective compute of every subsequent variant.  In
   the next chapter (NSGA-II) it should be the default
   evaluator everywhere.

---

## 1b. What the metaheuristics chapter established

The follow-up chapter implemented three trajectory/construction
metaheuristics on the *same* `core.py` / `meta_common.py` substrate —
**Simulated Annealing**, **GRASP**, and **Variable Neighborhood Search**
— sharing the 15-operator pool, the Pareto archive, and the
HV-improvement acceptance criterion with the HC family, so the four
families are directly comparable.  Tuned at seed 42 under the canonical
budgets (small/large 25 s, medium 12 s), the scoreboard (see
[RESULTS.md](RESULTS.md) § 4b) supports three conclusions.

1. **GRASP wins every official instance, and beats the entire HC family.**
   GRASP scores −1 816 174 / −1 615 368 / −5 010 404 on
   small/medium/large (**99.25 % / 92.57 % / 91.21 %** of the leaderboard
   top), ahead of the best HC on all three.  Its `large-graph` result is a
   **new project best, ~210 200 HV beyond hc13** (−4 800 196), at an honest
   on-budget 25.0 s and robust across seeds 42/1/2 (−4 904 767 /
   −4 916 147 / −5 010 404, all clearing the prior best); the seed-2
   submission is now canonical.  The greedy-randomized **min-degree construction with an
   α-controlled RCL**, restarted into one shared archive, is what the
   dense large instance most rewards — consistent with the HC chapter's
   finding that *construction* dominates on dense graphs.
2. **The ordering is GRASP > VNS > SA on all three instances.**  VNS's
   shake/descent ladder lands just behind GRASP; SA's single
   scalarisation of the bi-objective (energy `−(n−w)(n−t)`) is the
   weakest of the three, though still competitive with the HC front on
   the sparse instances.  Each gap is a *design* property, not a tuning
   artefact (see § 1b note below).
3. **Fair comparability is the methodological contribution.**  Because
   all four families reuse the same evaluator, operator pool, archive,
   and acceptance rule, differences in the scoreboard are attributable
   to the *search strategy* alone — which is exactly the controlled
   comparison a metaheuristics paper needs.

**Honest ceilings (findings, not bugs).**  SA is limited by collapsing a
bi-objective into one scalar energy; GRASP wastes some restarts on
duplicate constructions at low α; VNS can starve its neighbourhood ladder
under tight budgets.  Each ceiling now has an **implemented, additive
enhancement** (default behaviour unchanged, alternate `--algo` stem, so
the canonical scoreboard is untouched):

- **SA → AMOSA acceptance** (`sa.py --accept amosa`): bi-objective
  Metropolis on the *amount of domination* against the live archive,
  replacing the single scalar energy.  Bandyopadhyay et al. 2008.
- **GRASP → path-relinking + reactive-α** (`grasp.py --path-relinking
  --reactive-alpha`): an elite pool recombined by permutation
  path-relinking, plus α resampled by historical front quality.  Glover
  1997; Prais & Ribeiro 2000.
- **VNS → true VND** (`vns.py --local-search vnd`): an ordered structured
  neighbourhood descent (best-of-`k`, reset on improvement) in place of
  the stochastic weighted-sample descent.  Hansen & Mladenović 2001.

The four families plus these variants were compared seed-for-seed (11
seeds) with a Friedman/Nemenyi test by `tools/meta_multiseed.py`.  **The
verdict is a clean negative result: none of the three enhancements
improves on its baseline.**  Friedman χ²(6) = 62.13 (≫ 22.46 at
α = 0.001); Nemenyi ranks **grasp 2.82** < hc9 3.12 < vns 3.21 < vns_vnd
3.55 < grasp_pr 4.00 < sa 5.24 < sa_amosa 6.06 (CD₀.₀₅ = 1.78).
Per-family Wilcoxon: AMOSA significantly *worse* than SA (p ≈ 0.0003),
GRASP path-relinking *worse* than plain GRASP (p ≈ 0.025), VND tied with
stochastic-descent VNS (p ≈ 0.13).  At these budgets the extra machinery
(recombination, dominance-amount Metropolis, structured descent) spends
wall-clock that pure restart/walk diversity uses better.  Each family is
**green-lit on its baseline**; the enhancements stay in-tree behind
default-off flags as a reproducible negative result, not promoted.  See
[ALGORITHMS.md](ALGORITHMS.md) § 7.2–7.3 and [RESULTS.md](RESULTS.md)
§ 4b.5.

---

## 1c. What the ACO bridge chapter established (a negative result)

A third chapter implemented **Ant Colony Optimization** — Ant System (AS)
and MAX–MIN Ant System (MMAS), plus an additive HV-descent hybrid — on the
same substrate, to test whether a **construction** metaheuristic competes
with the warm-start-perturbation and greedy-restart families. It does not.
The conclusion is publishable precisely *because* it is a clean, tested
negative result.

1. **ACO is significantly worse than every other family.** In the 11-config
   Friedman omnibus (χ²(10) = 258.459, p = 9e-50, CD₀.₀₅ = 2.583), the four
   ACO variants occupy the bottom four ranks (9.24–10.00), each beyond the
   critical distance from every HC/SA/GRASP/VNS config. Pure construction
   **fails outright on `large-graph`** (every cell −0: ~16 ants build in 25 s,
   none under the 500-width cap). The hybrid descent rescues feasibility
   (−4 670 050) but still trails GRASP (−5 010 404).
2. **The reason is structural, not a tuning artefact.** Construction is the
   wrong unit of work on a dense graph (each order costs ~1.5 s and rarely
   lands feasible); the pheromone never converges in budget (1–2 iterations on
   large), so the best config leans on its greedy heuristic (`β = 4.0`) and is
   functionally a noisy greedy constructor that plain greedy min-degree (GRASP
   α = 0) already beats; and the only thing that rescues it is the *borrowed*
   shared descent, after which the colony contributes nothing (AS ≡ MMAS,
   identical scores across the whole large grid).
3. **MMAS vs AS and hybrid vs construction are settled.** MMAS's only
   measurable edge is sparse-instance construction (small p = 0.014; tied or
   identical elsewhere). The hybrid descent is a *feasibility rescue, not a
   quality lift* (large p = 0.0009 for the hybrid; small/medium slightly
   worse, not significant). Full tables in [ALGORITHMS.md](ALGORITHMS.md) § 8
   and [RESULTS.md](RESULTS.md) § 4c.

The takeaway for the roadmap: **on torso decomposition under tight
single-thread budgets, construction-from-scratch is dominated by warm-start
perturbation and greedy-randomised restart.** The remaining open direction is
*population-based* search that maintains and recombines a set of feasible
fronts (NSGA-II / SMS-EMOA), not more construction machinery.

---

## 1d. What the population chapter established (NSGA-II / SMS-EMOA)

A fourth chapter implemented the **population** paradigm — NSGA-II (fast
non-dominated sort + crowding distance, order crossover) and SMS-EMOA
(steady-state, least-HV-contribution culling), plus an additive memetic
HV-descent axis — on the same substrate, to test whether evolving a *whole
population* of orders beats the single-incumbent families. It does, **on the
sparse instance**, and the result is the strongest positive finding of the
study so far.

1. **SMS-EMOA is the first method to significantly beat hc9 on any instance.**
   On `small-graph` the four population variants take the top four seed-means,
   and SMS-EMOA beats GRASP (p = 0.0058), GRASP+PR (p = 0.0076) and hc9
   (p = 0.0329) with significance; `sms_ls` posts the study's best single
   small-graph run (−1 818 628). The pooled 15-config Friedman omnibus
   (χ²(14) = 321.607, p = 3.6e-60, CD₀.₀₅ = 3.483) places the family mid-pack
   (sms 6.530, nsga2 6.924) only because it averages over the dense instances
   where it loses to GRASP — the per-instance Wilcoxon is the right readout.
2. **HV-selection ties crowding (Q1).** SMS-EMOA's least-HV-contribution
   culling vs NSGA-II's crowding distance is statistically indistinguishable
   everywhere (pooled p = 0.42). So HV is the right *acceptance* policy (the
   hc9 result) but not a measurably better *selection* policy here — the design
   question this chapter was built to answer.
3. **Memetic descent *hurts* the population (Q2)** — the opposite of ACO.
   `nsga2_ls` < `nsga2` (p = 0.0029), `sms_ls` < `sms` (p = 0.0071), favouring
   the pure variants: the pure population is already feasible, so per-offspring
   descent just burns generations. Combined with `pop = 20` winning 10/12
   tuning cells, the lesson is **iteration depth beats population breadth**
   under a wall-clock budget. Full tables in [ALGORITHMS.md](ALGORITHMS.md) § 9
   and [RESULTS.md](RESULTS.md) § 4d.

The takeaway for the roadmap: **population MOEAs are the right tool for sparse,
genuinely two-dimensional fronts and the wrong tool for dense
staircase-fronts.** Order crossover — the only move that blends two good
orderings — is the active ingredient on `small-graph`. The natural next steps
are an off-the-shelf NSGA-II/SMS-EMOA baseline (PyGMO) to confirm our
implementation, and a torso-aware warm start to attack the dense instances
where GRASP still leads.

---

## 1e. The ceiling analysis: every lever, measured (1–4 June 2026)

After four algorithm chapters the project had five distinct levers left that
could plausibly move the score *without* a new algorithm family. The brief was
explicit: try each one, measure it, and only then claim the limit. This
section is the record. Every number below is from `core.evaluate` re-scored
end-to-end by `tools/verify_submission.py`; nothing is an in-run estimate.

**The standing scoreboard going in (best single method per instance, seed 42,
canonical budgets):**

| Instance | best single | % of leaderboard top | gap to top |
|---|---:|---:|---:|
| small-graph  | GRASP −1 816 174 | 99.25 % | +13 252 HV |
| medium-graph | GRASP −1 615 368 | 92.57 % | +128 036 HV |
| large-graph  | GRASP −5 010 404 | 91.21 % | +467 819 HV |

**Lever 1 — warm-start construction (min-fill / MCS-M / per-t).** Min-fill is
made the default everywhere it is affordable via `build_warm_start(method=
"auto")`, and NSGA-II was rewired off plain min-degree onto it. Measured width
at t = 0, seed 42: small **20 vs 23** (min-fill vs min-degree), medium
**275 vs 276**, large **499 vs 499 — identical width, 3.65 s vs 1.02 s build**.
On large min-fill buys *zero* width for ~2.6 s of a 25 s budget, so the auto
dense-fallback to min-degree is correct, not a compromise. MCS-M was measured
*worse* than min-fill on every dense cell (minimal ≠ minimum triangulation).
**Verdict: tapped.** Construction helps the sparse instance by a few width
units and cannot help the dense one at all.

**Lever 2 — instance-specific portfolio (`tools/portfolio.py`).** Pool every
method's feasible Pareto points per instance and keep the top-20 by HV
contribution (HSSP DP). This can only match or beat the best single method and
needs zero new search. Measured union vs best single:

| Instance | portfolio union | Δ vs best single | gap to top |
|---|---:|---:|---:|
| small-graph  | −1 816 667 | +493 HV   | +13 252 |
| medium-graph | −1 617 086 | +1 718 HV | +128 036 |
| large-graph  | −5 025 243 | +14 839 HV | +467 819 |

All three verified: 0 capped vectors, fronts non-dominated (small carries one
zero-contribution slot). **Verdict: a real, free win — locked in as the
canonical best-of submission** (own `portfolio` stem, additive; no canonical
JSON overwritten).

**Lever 3 — more compute (larger budget / multi-seed union).** This lever
**paid, and is the only one that broke the standing portfolio.** A 48-seed
GRASP sweep on large + a 24-seed `sms_ls` sweep on small (the § 1f hand-off,
run on the workstation, 4.3 min wall at 8 workers) moved the canonical
portfolio:

| Instance | portfolio before | portfolio after | Δ | gap to top |
|---|---:|---:|---:|---:|
| small-graph  | −1 816 667 | **−1 819 187** | +2 520 HV | +10 732 |
| medium-graph | −1 617 086 | −1 617 086 (no seeds run) | 0 | +128 036 |
| large-graph  | −5 025 243 | **−5 033 531** | +8 288 HV | +459 531 |

On small the new best *single* method is a population seed (`sms_ls_s19`
−1 817 997), with `sms_ls_s19` / `sms_ls_s17` now the top union contributors —
so the population MOEA's sparse-instance edge (§ 1d) compounds across seeds. On
large the union of 48 GRASP seeds adds +23 127 HV over the best single seed
(`grasp_s26/s18/s16/s9` lead). Medium got no new seeds and is unchanged.
**Verdict: medium is landscape-limited; small and large both still buy HV from
parallel seeds — with diminishing but not-yet-zero returns.** This is the one
lever that is *not* closed: it has a known positive gradient, just a shallow
one. The honest statement is "bounded and shrinking," not "exhausted."

**Lever 4 — exact / structural methods on the dense instance.** The minor-min-
width treewidth lower bound on large is **499**, essentially equal to the 500
width cap (`core.treewidth_lower_bound_mmd`). That proves the `t = 0` corner of
the front is already pinned near the structural optimum, so exact treewidth
solvers are the wrong tool: there is no feasible width-≪499 ordering to find.
The remaining headroom lives in the *high-width middle* of the front (the
intermediate-t region), not at the corners. **Verdict: exact methods cannot
close this gap; the gap is not where exact methods operate.**

**Lever 5 — per-t width minimization (`hc10`, torso-aware warm start).** This
was the last open idea: build a *different* high-degree-head / low-degree-torso
permutation for every t to fill the middle of the front. Measured on large,
seed 42: **−4 725 645 — worse than the portfolio's −5 025 243** — and when its
front is pooled into the union it contributes **zero** non-dominated points
(it is not among the top contributors grasp / grasp_s2 / grasp_s1 / vns /
hc13; the union is byte-identical with and without it). The high-degree-head
heuristic produces a structurally worse front than min-degree + GRASP already
gives. **Verdict: the last lever is closed.**

**Conclusion — the limit is characterised, and one lever is bounded rather
than closed.** Every lever that does not require a fundamentally new method has
been pulled and measured. Four of the five are closed: warm start (at its
useful limit), exact-on-large (wrong tool — LB 499 ≈ cap), per-t-on-large
(structurally worse front), and compute-on-medium (landscape-saturated, +116 HV
from a multi-seed union). The fifth — **multi-seed union on small and large —
is the one lever that still pays**, and the § 1f sweep banked it: the canonical
portfolio is now **small −1 819 187 / medium −1 617 086 / large −5 033 531**,
verified end-to-end, 0 capped. The residual gaps to the ESA leaderboard reduce
to two named causes: (a) on `small-graph`, a real treewidth plateau at width 20
that no ordering heuristic in this family escapes (§ 2.1), now within +10 732 HV
of the top; and (b) on `medium`/`large`, a compute-budget asymmetry the
leaderboard enjoyed, partially and *continuously* recoverable on small/large by
throwing more seeds at the multi-seed union (diminishing returns, shallow
positive gradient). So the precise, defensible statement is **not** "the score
cannot move" — it demonstrably still moves a little per seed-batch — but
*"every algorithmic lever in the implemented families is exhausted; the only
remaining gain is brute-force seed averaging on two instances, whose marginal
return is now small and measured."* That is the honest form of "I tried
everything, and it has reached its limit."

---

## 1f. Final-validation hand-off — run and banked (4 June 2026)

`tools/ceiling_validation.py` runs N independent GRASP seeds on large + M
`sms_ls` seeds on small at the canonical 25 s budget (each its own additive
`grasp_sNN` / `sms_ls_sNN` stem), re-pools the portfolio, and re-verifies all
three submissions. It writes only new seed stems and `portfolio` — canonical HC
and GRASP JSONs are read-only inputs.

**Result of the 48-large + 24-small run (4.3 min, 8 workers, 0 non-zero rc):**
the union *did* move — small −1 816 667 → **−1 819 187** (+2 520 HV), large
−5 025 243 → **−5 033 531** (+8 288 HV), medium unchanged. All three re-verified
0-capped. So the empirical answer to "does the union plateau after a few dozen
seeds?" is **not yet** — seed averaging still buys low-thousands of HV per
batch on small/large. This is the bounded-but-open lever of § 1e; the score is
limited by *diminishing-returns seed averaging*, not by a hard wall. To make
the bound tight for the paper, extend to ~100 seeds and plot the union HV vs
seed-count curve: the point where it flattens is the empirical ceiling.

A companion driver, `tools/retune_longrun.py`, closes the *parameter* question
(distinct from the seed question): it re-runs every family's full `tune.py`
grid at a large budget across seeds and diffs the winning cell against the
recorded canonical config, emitting a CONFIRMED / MOVED verdict per
(family, instance). Run it to upgrade the parameter green light from
"best at 25 s, seed 42" to "best at scale, multi-seed."

---

## 1g. The continuous-encoding paradigm — why the leaderboard is ahead, and first results (4 June 2026)

**This section overturns the framing of § 1e.** Two top ESA leaderboard
solutions were obtained and decoded (`cuda-torso`, `fast-cma-es`). They reveal
that the ceiling § 1e so carefully certified is the ceiling of *permutation-space
search only* — and that a different paradigm, which this project had explicitly
dismissed (§ 4.1–4.2), closes most of the gap.

### What the two winning solutions do

1. **`cuda-torso` — GPU neuro-evolution over a spectral node-scoring policy.**
   It does not search permutations. It builds per-vertex features — a local
   degree profile (degree + neighbour-degree min/max/mean/std) plus the smallest
   **Laplacian eigenvectors** (a spectral embedding), with a polynomial
   expansion — and learns a weight vector whose dot-product with those features
   scores each vertex; the elimination order is `argsort(scores)`. The weights
   are evolved by CoSyNE-style neuro-evolution, **1024 candidates per generation
   in parallel on a custom CUDA fill-in kernel**, tracking a per-threshold elite.
2. **`fast-cma-es` — Dietmar Wolz's gradient-free toolkit** (CMA-ES, BiteOpt,
   CR-FM-NES, MO-DE, all with massively parallel retry). No torso-specific
   driver ships in it; it is the *engine*. The standard fcmaes recipe for a
   permutation problem is the same decode: optimise a continuous vector,
   `argsort` it into a permutation, hand it to a parallel black-box optimiser.

**The shared idea we missed:** both lift the problem out of permutation space
into a **continuous** space and let `argsort` decode the permutation. Every
prior family here (15 HC + SA + GRASP + VNS + ACO + NSGA-II + SMS) searches
*directly* in permutation space with swap/insert/reverse moves. That is a
weaker search space for this problem.

### The correction to § 1e / § 4

`FUTURE.md` § 4.1–4.2 argued *against* exactly these techniques — "the input is
a permutation, not a feature vector", learned per-vertex priority "will be
marginal at best", neuro-evolution "skip for the foreseeable future". The actual
leaderboard-topping solution is a learned per-vertex priority over spectral
features, evolved. **That dismissal was wrong**, and the honest re-statement of
the whole study is: the permutation-space families are at their limit (§ 1e
stands *for them*), but that is not the global ceiling — the leaderboard gap is
a *search-space* gap, not a compute or tuning gap.

### First CPU results — `algorithms/continuous/cmaes_torso.py`

The paradigm is reproduced on CPU (no GPU, no fcmaes required): node features =
degree profile + top-K Laplacian eigenvectors (dense `numpy.linalg.eigh`); a
continuous policy `x`; `perm = argsort(features @ x)`; a **single fill-in pass
yields `deg[i]` for every step, so `width(t) = suffix-max(deg[t:])` gives the
entire `(width, t)` front from one permutation** (the same trick the CUDA kernel
exploits); the policy is optimised by a self-contained separable CMA-ES to
maximise that front's hypervolume, with a global Pareto archive across all
candidates. An `--engine fcmaes` path uses the real toolkit on the workstation.

**Full 8-seed sweep (32 eigenvectors, 600 s/seed, builtin sep-CMA-ES, 8 workers,
30 min wall) — canonical-scorer-verified, 0 capped on all three.** Every top
portfolio contributor is now a `cmaes` seed:

| instance | 4-chapter portfolio (§ 1f) | **continuous-encoding (8 seeds, 32 eig, 600 s)** | leaderboard top | gap before → after | % of top |
|---|---:|---:|---:|---:|---:|
| small-graph  | −1 819 283 | **−1 827 924** | −1 829 919 | +10 636 → **+1 995**  | 99.9 % |
| medium-graph | −1 617 086 | **−1 704 522** | −1 745 122 | +128 036 → **+40 600** | 97.7 % |
| large-graph  | −5 033 531 | **−5 369 684** | −5 493 062 | +459 531 → **+123 378** | 97.8 % |

The continuous encoding closed **~68 % of the medium gap and ~73 % of the large
gap in one 30-minute run**, and put `small-graph` within +1 995 HV of the
leaderboard top. The single best runs were `cmaes_s4` (small −1 827 610),
`cmaes_s7` (medium −1 693 446), `cmaes_s3` (large −5 358 345); the 8-seed union
adds a further +314 / +11 076 / +11 339 HV on top.

The earlier first-light numbers that motivated this (seconds-long, single-seed,
16 eigenvectors) already foreshadowed it: a **22-second** run on `large-graph`
beat the entire four-chapter permutation-space effort by +259 206 HV. On
`small-graph` an 18-second run beat the seed-saturated portfolio and came within +4 601 of the
top. These are first-light numbers at trivial budget; the obvious levers
(more eigenvectors, polynomial features, longer budget, multi-seed, the fcmaes
MO-DE engine, and ultimately the GPU kernel) are all untouched.

### Two fixes that made the dense instance work

The first dense-instance sweep produced `-0` (empty archives) and 1 266 s jobs.
Two corrections — both lifted from how `cuda-torso`'s `libeval.cu` actually
behaves — fixed it, and are the reason the table above exists:

1. **Feasibility gradient.** A random spectral policy on `large-graph` almost
   always produces an over-cap elimination order. The original evaluator
   returned a flat infeasible constant, so CMA-ES had no slope to descend and
   never found feasibility. The fixed `eval_fitness` returns a *graded positive
   penalty* that grows with the amount of cap violation (exactly cuda-torso's
   `penalty = 501 + Σ(deg − 500)`), pushing the optimiser toward feasible
   policies.
2. **Early-abort + warm-start seed.** The evaluator now bails the moment the
   penalty blows up (restoring `core.evaluate`'s early termination — its
   absence made each infeasible eval ~45 s on large), and the archive is seeded
   with the min-degree warm start so the submission is never empty and the
   search has a feasible anchor. Evals went from ~28-in-1266 s to ~182-in-31 s.

Also: the dense spectral eigendecomposition is cached to `.feature_cache/`
(computed once per instance, not once per seed), and the optimisation budget
clock starts *after* feature-building so the eigendecomposition never eats the
search budget.

### Engine comparison — the optimiser is not the bottleneck (tested)

A matched 8-seed / 32-eig / 600 s sweep with the **fcmaes** full-covariance
CMA-ES (`--engine fcmaes`, additive `cmaesf_s*` stems) was run head-to-head
against the builtin separable CMA-ES. Best single per instance:

| instance | builtin sep-CMA-ES | fcmaes full-cov | winner |
|---|---:|---:|---|
| small  | **−1 827 610** | −1 826 951 | builtin +659 |
| medium | **−1 693 446** | −1 689 268 | builtin +4 178 |
| large  | **−5 358 345** | −5 335 648 | builtin +22 697 |

The full-covariance optimiser is *slightly worse* on all three. At ~37 dims
under a fixed wall budget, learning the O(dim²) covariance costs more samples
than the diagonal model, and the spectral-policy landscape is separable enough
that the cheaper adaptation wins. Folding `cmaesf_*` into the portfolio moved
it only marginally (small +139, medium +216, large +0). **Conclusion: the
search algorithm is at its ceiling here; the remaining leaderboard gap is a
*feature-richness* and *eval-throughput* problem, not an optimiser problem.**

### What this means for the roadmap

The next chapter is **the continuous-encoding chapter**, and the 8-seed sweep
above is its first real result. With the optimiser question settled, the
remaining ladder is:

1. **Richer features (CPU, closes small/medium)** — K = 48–64 eigenvectors,
   and/or cuda-torso's polynomial feature expansion. `small` needs only +1 856
   and `medium` +40 384 — both plausibly CPU-reachable with more features +
   longer budget + more seeds.
2. **Eval throughput → GPU (closes large)** — `large` is +123 378 short, and
   the gap is fundamentally that cuda-torso does ~10⁸ evaluations (1024-wide ×
   up to 100k generations on a GPU) versus our ~28k CPU evaluations. Porting
   the fill-in evaluator to the `cuda-torso` CUDA kernel is the lever that
   matches the leaderboard top on the dense instance; no CPU optimiser change
   will bridge a 10⁴× evaluation deficit.
3. The **fcmaes** engine stays available (`--engine fcmaes`) but is *not* the
   default — tested and found no better than the builtin here.

Until further incorporated, the two winning solutions live under
`leaderboard_reference/` (read-only study copies) and this section is their
record. The permutation-space families (§ 1e–1f) are now **frozen** — kept as
the completed comparative study and as portfolio inputs, but receiving no new
compute; the score lives in this paradigm.

## 1h. The GPU scale-up — run, measured, and the forecast corrected (7 June 2026)

Step 2 of the roadmap above ("Eval throughput → GPU (closes large)") was a
forecast. It has now been *tested*, and the forecast was only partly right —
which is the more useful outcome and is written up in full as THESIS.md §9.

What was built and run:
- `algorithms/continuous/gpu_eval.py` — a Numba CUDA batch evaluator (one
  ordering per thread, bitset fill-in on the card), mirroring `eval_fitness`.
- `tools/validate_gpu.py` — the correctness gate. On large (n = 2426) it reports
  **0 status mismatches, 0 degree-sequence mismatches** vs the numpy reference,
  which is cross-checked against `core.evaluate`. The GPU scorer is provably
  identical to the official one.
- `tools/gpu_search.py` — the continuous-encoding CMA-ES at GPU scale (4,096
  orderings/generation), warm-started from the banked portfolio
  (`tools/make_warmstart.py`).

What happened (single Tesla T4, Colab, 30 min/instance):
- **large**: −5,399,072 → **−5,405,118** (official `tools/portfolio.py` re-score),
  a verified **+6,046 HV** gain; `gpucma` is the best single contributor.
- **medium**: marginal gain (~+700 HV official), then flat.
- Both runs **plateaued** after 2–6 × 10⁵ evaluations, well short of the top.

The correction: throughput was *necessary but not sufficient*. Removing the
evaluation wall (~25× more orderings/second) bought a real, bounded improvement
and then exposed the binding constraint — the **expressiveness of the linear
`argsort` decode**. Every reachable ordering is `argsort(F·x)` over 37 fixed
features, a low-dimensional manifold; once CMA-ES saturates it, more evaluations
only resample it. The leaderboard-topping orderings lie off the manifold.

The roadmap is therefore updated: the next lever is **not more GPU throughput**
but a **richer, nonlinear ordering decode** (a learned/neural or boosted-tree
score map), or folding the *adaptive* GBDT constructor (§6b — the one method that
already escapes the static-`argsort` manifold) into the GPU population. The GPU
evaluator built here is the reusable substrate for that next step.

## 1i. GAPS — the novel method, built and verified (8 June 2026)

The §1h roadmap prescribed "a richer, nonlinear ordering decode." That method now
exists, is verified, and is written up as THESIS.md §10: **GAPS** (GBDT-Augmented
Polynomial Spectral search, `tools/gaps_search.py`).

  decode score  =  Φ(F)·x  +  β·g_GBDT(F)

— a polynomial spectral basis plus a **GBDT-learned nonlinear column** (DAgger-
retrained on elite orderings), searched by the §9 GPU-scaled CMA-ES, warm-started
from the banked best.

Verified, official (top-20, `tools/portfolio.py`) on large-graph:
- linear GPU baseline (`gpucma`):            −5,404,348  (98.39 %)
- GAPS, **GBDT disabled** (poly-only control): −5,406,555  (98.42 %)
- GAPS, **GBDT enabled**:                      −5,431,321  (98.88 %)
- pooled portfolio (all sources):             **−5,431,595**  (98.88 %)

Findings:
- **+32,523 HV** over the original CPU portfolio (−5,399,072 → −5,431,595);
  large rises 98.29 % → 98.88 % of the leader.
- **Controlled GBDT contribution = +24,766 HV** (with vs without the GBDT column,
  identical warm start and seed) — an order of magnitude beyond the static +2,524
  of §6b, and here the *dominant* driver. The gain is causally time-locked to
  generation 8, when the GBDT column first trains (fig11).
- A first design with 128 *random* interaction features failed (search stalled) —
  a clean negative control: undirected nonlinearity does not help; the *learned*
  GBDT nonlinearity does.
- Still **+61,467 short of #1** (−5,493,062). The novelty and GBDT contribution do
  not depend on reaching the top; closing the last ~1 % plausibly needs a richer
  policy class (deeper learned map / GPU-scaled adaptive constructor) or the
  leader's compute. GAPS is the platform for that.

The matrix exposes three distinct gaps between our best HC result
and the official ESA leaderboard top.  They have different
characters, are caused by different mechanisms, and require
different fixes.  The next chapter should attack them separately,
not bundle them.

### 2.1 Sparse plateau (small-graph: +18 008 HV gap, ~99 % of top)

We are at `(max_degree, t) = (20, 0)` and every single-step HC
operator — `swap`, `insert`, `reverse`, `block_move`,
`bottleneck→head`, even `k_bottleneck→head` — fails to drop the
width below 20.  This plateau is structural: it is the *actual*
treewidth lower bound from the min-degree heuristic on this
particular graph.  The bottleneck vertices are mutually adjacent in
the fill-in graph, so removing K of them simply exposes the next K.

**What this gap is NOT.**  It is not a compute-budget gap.  Running
hc5 for 25 s vs 250 s on small-graph produces identical scores
within seed noise.  More iterations of single-step HC cannot help.

**What would actually close it.**

| Candidate | Expected gain | Effort |
|---|---:|---|
| **Exact treewidth on small subgraphs** via SAT or ILP — e.g. extract every 20-vertex subgraph that contains the current bottleneck cluster, find an optimal elimination order for it, splice back in. | Could close most of the 18 k gap. | ~3 weeks. |
| **Hand-designed coordinated structural moves** that explicitly target chordal-completion artefacts (separator decomposition, nested-dissection moves). | Maybe 5-10 k HV. | ~2 weeks. |
| **Iterated improvement + escape via SAT-derived restart points.** Run hc5 until plateau, then ask a small SAT instance "is there a width-19 order on this 50-vertex subgraph?", use the answer as a restart seed. | 5-15 k HV; rigorous lower-bound certificate as a bonus. | ~3 weeks. |
| **Run more seeds** (current single-seed reporting hides any seed-dependent gains that exist below the plateau). | < 500 HV. | trivial. |

**What would NOT close it.**

- Off-the-shelf gradient boosting / pointer networks / GNNs as
  permutation predictors: no learned representation can find a
  width-19 ordering if none exists in the model's training
  distribution, and there is no way to know in advance that one
  exists at all.
- Larger time budget for the existing HC variants.
- Neuro-evolutionary search trained on three official instances:
  catastrophic overfitting risk; the policy would over-fit
  small-graph and not transfer.

### 2.2 Compute wall (medium-graph: +135 966 HV gap, ~92 % of top)

Each fitness evaluation on `medium-graph` costs ~120 ms with the
bitset evaluator; a 12-second budget gives ~100 iterations of HC per
seed.  The 136 k-HV gap to the leaderboard is consistent with the
leaderboard having had 10-100× our iteration count, not with a
qualitatively different algorithm.

**What would close it.**

| Candidate | Expected gain | Effort |
|---|---:|---|
| **Multi-seed parallelism** (multiprocessing or CUDA): run 100 independent HC seeds in parallel, take the per-instance best.  Trivially embarrassingly parallel. | 30-70 k HV. | ~1 day. |
| **`hc12`-style incremental evaluator** propagated to every variant in the family (currently only on `hc12`).  Buys 2–5× more iterations at the same wall budget. | 10-30 k HV. | ~3 days. |
| ~~**Simulated Annealing** with our existing operators~~ — **DONE** (chapter 2).  Implemented and tuned; on `medium-graph` SA reaches −1 603 923, competitive with the HC front, and GRASP goes further (§ 1b, RESULTS § 4b). | _realised: GRASP-large +210 200 HV new best (91.21 % of top)._ | _complete._ |
| **Surrogate fitness pre-filter** (a fast GBDT regressor that predicts whether the real evaluator will accept a candidate; skip the evaluator on predicted rejects). | 2-3× throughput on the inner loop. | ~3 days. |

### 2.3 Feasibility wall (large-graph + 3 densest synthetic cells: +482 658 HV gap, ~91 % of top)

On `large-graph` and on `inst_16` / `inst_19` / `inst_20` the
min-degree elimination order ITSELF produces a permutation whose
fill-in cascade exceeds the 500-width limit on some step.  The
Pareto archive (correctly) rejects the over-width point, so every
warm-start variant (`hc4` – `hc11`, `hc13`) enters search with no
in-cap point and returns the deterministic over-width fallback
`(501, 0)` via `core.ensure_seeded` (previously these cells crashed
with `nan`; fixed 29 May 2026).  This is the largest unresolved gap
in the chapter and the cleanest single thing to fix next.

> **Correction (29 May 2026).**  We measured the candidates below
> head-to-head on the three failing synthetic cells (see
> `docs/RESULTS.md` §5.3).  **MCS-M is *worse* than the existing
> min-fill** on every dense Erdős–Rényi cell (minimal ≠ minimum
> triangulation), and even min-fill stays 76–283 width units above
> the cap.  These three cells have treewidth genuinely > 500, so *no*
> construction heuristic in this family makes them feasible — the
> "highest-EV single change" claim for MCS-M below does **not** hold.
> The durable takeaway is narrower: **min-fill should be the default
> warm start wherever it is affordable** (it dominates by a few width
> units on the sparse/medium feasible cells, including small-graph and
> medium-graph).  This is already what `build_warm_start(method="auto")`
> does: min-fill on small/medium, min-degree fallback once
> `n·avg_deg > 200,000` (i.e. large-graph).
>
> **Measurement (30 May 2026), width at t = 0, seed 42:**
> small-graph min-fill 20 vs min-degree 23 (build 0.39 s vs 0.22 s);
> medium-graph 275 vs 276 (0.67 s vs 0.26 s);
> large-graph **499 vs 499 — identical width but 3.65 s vs 1.02 s build**.
> On large-graph min-fill buys *zero* width improvement while burning
> ~2.6 s (≈10 %) of a 25 s budget that would otherwise be HC iterations,
> so the auto dense-fallback to min-degree is correct and was left in
> place.  "Make min-fill the default" therefore reduces to "keep auto",
> not "force min-fill everywhere": forcing it on large-graph would be a
> strict regression.  No change to the canonical scoreboard is implied.

**What would close it (for the genuinely feasible cells, e.g. large-graph).**

| Candidate | Expected gain | Effort |
|---|---:|---|
| **MCS-M (Berry, Blair, Heggernes, Peyton 2004)** — Maximum Cardinality Search with Minimum-fill tweaks.  Produces minimum-fill elimination orders by construction; well-known to beat plain min-degree on dense instances. | 50-200 k HV on dense instances; this is the *highest-EV single change* in the entire chapter. | ~2 days. |
| **LEX-M (Rose, Tarjan & Lueker 1976)** — lexicographic BFS variant that produces perfect elimination orderings on chordal graphs and good orderings otherwise. | Similar to MCS-M; sometimes better, sometimes worse, instance-dependent. | ~2 days. |
| **min-fill with proactive width check** — at each step, prefer vertices whose elimination would *not* exceed 500-width.  Heuristic but cheap. | 30-100 k HV. | ~1 day. |
| **k-core peeling + selective LEX-M** — partition vertices by k-core number, apply LEX-M inside each core. | 50-150 k HV. | ~1 week. |
| **CUDA-Torso-style parallel branch-and-bound** (Pan & Zhao 2023, Yamaguchi et al. 2022) — GPU-parallel exact / near-exact treewidth solver with lower-bound pruning. | Almost certainly *matches the leaderboard top*; this is probably what produced the −5 493 062 reference number. | ~6 weeks, GPU-week of compute. |

---

## 1j. GBFC — the primary novel method, built and verified (8–9 June 2026)

The §1i GAPS write-up flagged that the GBDT *decode column* is verified but
**instance-specific** (it does not generalise across synthetic graphs — see
THESIS.md §10.4). That honest negative result reframed the search for the
thesis's headline novelty, which now exists and is written up as THESIS.md §11:
**GBFC — Gradient-Boosted Front Construction** (`tools/gbfc.py`).

The idea: the Pareto front is a family of threshold specialists, and "combine
many weak, complementary specialists each correcting the others" *is* gradient
boosting. GBFC builds the front that way — seed with the banked front; find the
**residual** band (where pooled `width(t)` is furthest above the treewidth lower
bound); fit a **GBDT weak learner** specialised to it; decode complementary
specialists; pool; repeat.

Verified, official (top-20, `tools/portfolio.py`), GBFC gain over the banked best:
- small:  −1,828,994  (**+688**,  99.95 % of leader)
- medium: −1,712,688  (**+734**,  98.14 %)
- large:  −5,431,924  (**+329**,  98.89 %)

GBFC is the **only** method in the project that measurably improves the banked
front, and the single best contributor to each pooled portfolio. It does not reach
#1 — bounded by the §5.2 / §10.5 treewidth-lower-bound certificate (the dense core
is provably optimal) — reported plainly.

**Scale-up (in progress, time-boxed).** A hybrid that injects GBFC's GBDT
specialists into a faithful reproduction of the winner's per-threshold
neuroevolution (`tools/qnegbfc.py`, with a `--no-gbdt` ablation) is running at GPU
scale on Kaggle. Early empirical reads, *internal estimates pending official
re-score*: small sits flat at the banked −1,828,994 (consistent with
near-optimality, 925 short), while **medium shows live movement** (the
wide-explorer injection is finding room on the larger-gap instance) — to be
re-scored with `portfolio.py` before any number enters THESIS/RESULTS.

---

## 3. Next chapter: algorithm-family roadmap

The same `core.py` substrate (evaluator, HV, Pareto archive,
warm-start heuristics, incremental evaluator) supports the remaining
algorithm families without modification.  Only the **acceptance rule**
changes between families.  Simulated Annealing, GRASP, and VNS are now
**done** (chapter 2, § 1b); ACO is done (chapter 3, § 1c); the
population-based families are now **done** (chapter 4, § 1d).

| Family | Acceptance rule | Status / what it should test |
|---|---|---|
| ~~**Simulated Annealing**~~ | accept with probability `exp(−ΔE / T)`, T cools over a schedule | **DONE** (ch. 2).  Probabilistic uphill moves on the scalar energy; weakest of the three metaheuristics but competitive with the HC front on sparse instances. |
| ~~**GRASP**~~ | greedy-randomized RCL construction + HV-improvement local search | **DONE** (ch. 2).  Wins all three official instances; GRASP-large is the new project best. |
| ~~**VNS**~~ | shake / HV-improvement descent, reset ladder on archive HV gain | **DONE** (ch. 2).  Second overall, just behind GRASP. |
| ~~**Ant Colony Optimization** (AS / MMAS + hybrid)~~ | pheromone-biased construction `p(v) ∝ τ^α·η^β`; optional per-ant HV descent | **DONE** (ch. 3, § 1c).  **Negative result**: last of all families (Friedman rank 9.2–10.0, beyond CD); pure construction fails on large-graph; hybrid rescue trails GRASP.  Construction does not transfer to this problem under tight budgets. |
| ~~**NSGA-II**~~ | non-dominated sort + crowding-distance selection over a population, order crossover | **DONE** (ch. 4, § 1d).  Takes the top four seed-means on `small-graph`; loses to GRASP on medium/large. Order crossover is the active ingredient on the sparse front. |
| ~~**PAES / SMS-EMOA**~~ | steady-state, cull the least-HV-contribution member | **DONE** (ch. 4, § 1d).  **First method to significantly beat hc9** on any instance (small-graph, p = 0.0329 vs hc9). HV-*selection* ties NSGA-II's crowding (p = 0.42) — so HV is the right acceptance but not a better selection policy here. PAES (1+1) subsumed by hc9. |
| **Memetic population (`nsga2_ls` / `sms_ls`)** | per-offspring HV-descent over the 15-op pool | **DONE** (ch. 4, § 1d).  **Negative**: descent *hurts* the population at fixed budget (p = 0.0029 / 0.0071 favouring pure) — opposite of ACO, because the pure population is already feasible. |
| **Stronger metaheuristic variants** | HV/dominance-based SA acceptance; GRASP construction-diversity memory; adaptive VNS ladder budgeting | Journal-tier refinements of the ch. 2 families (the design ceilings noted in § 1b). |

After that the comparative table is `15 HC + 3 metaheuristics (SA, GRASP,
VNS) + NSGA-II + SMS-EMOA = 20 variants × 23 instances × 30 seeds`, which
is the data backbone for a publication-quality chapter on multi-objective
local search.

---

## 4. Candidate techniques discussed but not on the roadmap

Several techniques have been considered for the next chapter and
ruled out for specific reasons.  Including them here is part of the
record so the reasoning is preserved.

### 4.1 Gradient-boosted decision trees (LightGBM / XGBoost / CatBoost)

These are three implementations of the same algorithm class.  Using
all three adds engineering overhead without information.  Picking
one, the question is what it predicts:

- *Per-vertex elimination priority*.  Replaces hand-coded min-degree
  with a learned combination of features (degree, k-core number,
  clustering, betweenness, …).  Expected benefit *marginal*: on
  small-graph the min-degree heuristic is already at the treewidth
  lower bound; on the dense graphs the warm-start width limit fires
  before "which ordering" matters.
- *Move scoring inside HC*.  Replaces `hc5`'s UCB-style operator
  weighting with a learned predictor.  Expected benefit *small*: it
  speeds up the allocation, not the asymptotic score.
- *Surrogate fitness pre-filter*.  Reject candidates the regressor
  predicts will not pass the real evaluator.  Expected benefit
  *medium throughput speedup*, *zero asymptotic score improvement* —
  listed as the "GBDT pre-filter" in § 2.2.

The structural issue is that **the input is a permutation, not a
feature vector**.  The natural representation is per-vertex features
in some context, and that is exactly what min-degree / min-fill /
MCS-M / LEX-M already use, with ~50 years of empirical tuning behind
them.  A learned per-vertex priority will be marginal at best.

### 4.2 Neuro-evolutionary search (NEAT, CMA-ES, OpenAI-ES on policy networks)

The natural framing for SpOC-3 would be to train a policy network
`π(state) → move distribution` and evolve its weights over a
population.  Successful instances of this pattern (AlphaTensor,
AlphaGo, AttentionLearnToRoute) require *GPU-weeks of training* and
*thousands of training instances*.  We have 23 instances total and
single-GPU resources.

The role a learned policy would play is already served, much more
simply, by our hand-coded operator suite.  `bottleneck→head`,
`k_bottleneck→head`, `2opt_torso`, and `swap_head_tail` are
graph-structure-aware; reproducing them from scratch via a learned
policy would take orders of magnitude more compute and produce a
less interpretable result.

**Recommendation: skip for the foreseeable future.**  It would be a
thesis of its own.

### 4.3 Pointer networks / attention-based heuristics

These have produced ~10 % gains over OR-Tools on TSP and VRP
(Vinyals et al. 2015, Kool et al. 2019).  For chordal-completion
problems specifically, results have been mixed: Schuetz, Brubaker &
Katzgraber (2022) report GNN-based methods often underperform
classical heuristics on treewidth-like tasks.

**Recommendation: out of scope for the HC paper.**  Possibly a
candidate for a future thesis chapter on "neural heuristics for
graph decompositions".

### 4.4 Pure exact methods (SAT, ILP, BMS branch-and-bound)

These give optimal solutions when they finish.  For our instance
sizes (n up to 2 426) most off-the-shelf SAT / ILP formulations time
out, but **sub-problem SAT** (find optimal width on a 50-vertex
subgraph, splice back in) is tractable and is listed in § 2.1.

---

## 5. Methodology checklist for the next paper

Before any of the new algorithm work matters, the experimental
protocol needs to be paper-grade.  None of these require algorithm
changes — only protocol changes — and they are listed roughly in
priority order.

1. **30 seeds per cell** for every entry in the comparison matrix.
   Currently single-seed.  Cost: ~30× current runtime, easy to
   parallelise across CPU cores.
2. **Non-parametric pairwise test** (Mann–Whitney U for pairwise,
   Friedman + Nemenyi for the full table).  Cite García, Fernández,
   Luengo & Herrera (2010) for the methodology.
3. **Anytime HV curves** — log archive HV every N iterations; plot
   median trajectory with shaded inter-quartile band across seeds.
   This is the single figure every reviewer of a metaheuristics paper
   looks at first.
4. **Treewidth lower bounds (MMD+, MMD++)** via `libtw` or similar.
   Anchors the "how good are we, really" question and converts the
   "gap to leaderboard" framing into a "gap to certified optimum"
   framing.
5. **External baselines**: random search (sanity floor),
   off-the-shelf NSGA-II from PyGMO, off-the-shelf SMS-EMOA from
   PyGMO, the static min-degree submission (no HC).  Same wall
   budget, same seeds, same reporting.
6. **Ablation table** — factorial design over {warm start ∈
   {random, min-degree, MCS-M, LEX-M}} × {operator set ∈ {1, 4, 11}}
   × {acceptance ∈ {lex, HV}}.  ~24 cells, 30 seeds each.
7. **Reproducibility package**: Docker image / Nix flake with exact
   Python version, exact random seeds, and a single command that
   regenerates every table and figure.  Hosted on Zenodo with DOI.
8. **Compute-budget transparency**: report wall time *per machine*.
   Pick one machine, run everything there, state the specs.

---

## 6. Recommended sequencing for the next chapter

Ranked by expected impact per unit of PhD work:

1. **MCS-M / LEX-M / width-respecting warm start** (§ 2.3) —
   closes the feasibility wall on dense graphs.  ~1 file, ~200
   lines.  Publishable as a graph-theoretic contribution on its
   own.
2. **CUDA-parallel multi-seed HC** (§ 2.2) — closes most of the
   medium-graph compute gap and a chunk of large-graph.  ~1 day of
   engineering.
3. **Simulated Annealing on `core.py`** — already on the roadmap;
   probably matches `hc9` within seed noise but is a methodological
   baseline the paper needs.  ~1 week.
4. **NSGA-II / SMS-EMOA from PyGMO** — already on the roadmap;
   external multi-objective EA baselines for the paper's
   competitive section.  ~1 week each.
5. **GBDT surrogate fitness pre-filter** (§ 4.1 third bullet) — if
   we want exactly *one* ML technique in the chapter, this is the
   one.  ~3 days; smaller but real throughput multiplier.
6. **30-seed runs + statistical tests + anytime curves** — protocol
   work, not algorithm work, but required for the paper.  ~1 week
   of careful engineering.
7. **Parallel branch-and-bound with treewidth lower bounds**
   (CUDA-Torso style) — highest-ceiling option; probably the
   technique behind the leaderboard tops.  ~6 weeks of focused work
   and a chunk of GPU time.  A whole chapter of its own.

Skipped (with reasoning above):

- Pointer-network / attention / learn-to-route for permutations
  (§ 4.3) — out of scope for this paper; high overfitting risk.
- Stand-alone neuro-evolution (§ 4.2) — too expensive on three
  official instances; would be a thesis on its own.
- Using LGBM + XGB + CatBoost in combination — they are the same
  algorithm class; pick one.
- Treating "ML + LS + CUDA + neuro-evolution" as a single coherent
  approach — they do different things and combining them does not
  multiply benefit.

---

## 7. Implementation sketches for the highest-EV next steps

Three of the items on the roadmap are concrete enough that a pseudocode
sketch helps anchor what we are committing to building.  These are
the ones I would write first, in order.

### 7.1 MCS-M warm start (closes the feasibility wall, § 2.3)

Berry, Blair, Heggernes & Peyton (2004) give the construction.  The
key invariant is that every step picks the vertex *u* whose
elimination produces the smallest fill-in, breaking ties by
maximum-cardinality search to ensure a perfect elimination ordering
on chordal graphs.

```python
def mcs_m_order(n, adj_bits):
    """Return an elimination order that respects the 500-width limit
    on every step that the input graph admits.  Output is a list of
    length n; index i is the vertex eliminated at step i."""
    label = [0] * n              # MCS label for each remaining vertex
    order = []
    remaining = (1 << n) - 1

    for step in range(n):
        # pick remaining vertex with maximum label, tie-break by
        # smallest fill-in degree (= cardinality of fill-in neighbours)
        best_v, best_score = -1, (-1, 1 << 60)
        r = remaining
        while r:
            vbit = r & -r; v = vbit.bit_length() - 1; r ^= vbit
            fill = popcount(adj_bits[v] & remaining)
            score = (label[v], -fill)
            if score > best_score:
                best_score, best_v = score, v
        # add to elimination order, peel, propagate labels
        order.append(best_v)
        remaining ^= 1 << best_v
        successors = adj_bits[best_v] & remaining
        s = successors
        while s:
            ubit = s & -s; u = ubit.bit_length() - 1; s ^= ubit
            label[u] += 1
            # MCS-M extension: increase labels of u's "later" neighbours
            adj_bits[u] |= successors ^ ubit
    return order
```

**Risks.**  MCS-M's empirical reputation is on graphs with mild
fill-in growth; on the very dense synthetic cells (`inst_16`,
`inst_19`, `inst_20`) it may still exceed the width limit.  In that
case the right fallback is *width-budgeted* MCS-M — at each step,
verify the candidate's elimination would not exceed 500 before
committing.  Expected to require ~20 extra lines.

### 7.2 Multi-seed parallel HC (closes the compute wall, § 2.2)

Embarrassingly parallel.  The Python `multiprocessing.Pool` version
is sufficient; CUDA only buys an extra 4-8× over that.

```python
from multiprocessing import Pool
from functools import partial

def parallel_hc(problem, num_seeds=100, budget_per_seed=5.0):
    """Run num_seeds independent copies of hc9 (or hc5, hc12), return
    the best per-instance Pareto archive merged HV-optimally."""
    seeds = list(range(num_seeds))
    with Pool() as pool:
        archives = pool.map(
            partial(run_hc9_one_seed, problem=problem,
                    budget=budget_per_seed),
            seeds,
        )
    # merge all archives and run the HSSP DP to extract the 20-point
    # subset with the highest HV
    return merge_archives_optimal(archives, k=20, n=n_for(problem))
```

**Risks.**  Memory: each archive holds ~50 perms × n vertices each.
For 100 seeds on large-graph this is ~12 MB per process × Python
overhead, ~3 GB total.  Cap the per-process archive size at 100 if
RAM is tight.  Reproducibility: ensure the seeds are *independent*
(use `numpy.random.SeedSequence`, not `random.seed(i)` which is
correlated across small consecutive seeds).

### 7.3 Simulated Annealing on `core.py`

> **Implemented (chapter 2).**  This sketch has been realised in
> `algorithms/simulated_annealing/sa.py`.  The shipped version uses a
> scalar energy `E(w, t) = −(n − w)(n − t)` with the Metropolis rule and
> auto-calibrated `T0` (≈ 0.4 acceptance of a typical worsening move), plus
> geometric / linear / adaptive cooling schedules.  The pseudo-code below
> is kept as the original design note; see § 1b and RESULTS § 4b for the
> tuned results.

The drop-in change from `hc5` is the acceptance rule.

```python
def metropolis_accept(cur_fit, cand_fit, T, rng):
    """Accept cand if it strictly improves cur, OR with probability
    exp(-deltaE / T).  deltaE measured on the soft cost so the rule
    works across plateaus."""
    if cand_fit < cur_fit:
        return True
    delta_w = cand_fit[0] - cur_fit[0]
    delta_t = cand_fit[1] - cur_fit[1]
    # Composite delta: width dominates, t as tie-breaker.
    delta = delta_w * 1000 + delta_t
    if delta <= 0:
        return True
    return rng.random() < math.exp(-delta / T)

def cooling_schedule(t0, t_min, n_iters, kind="geometric"):
    """Standard cooling schedules.  Use 'geometric' with alpha=0.995
    for short budgets, 'linear' for paper baseline runs."""
    if kind == "geometric":
        alpha = (t_min / t0) ** (1.0 / n_iters)
        return lambda i: t0 * (alpha ** i)
    elif kind == "linear":
        return lambda i: t0 - (t0 - t_min) * i / n_iters
    else:
        raise ValueError(kind)
```

**Risks.**  Hyperparameter sweep on `(T_initial, T_min, schedule
kind, alpha)`.  Plan: pick five reasonable settings, run all five
on each instance × 30 seeds, report the best per cell *and* the
range across the hyperparameter choices.  Avoid the "we tuned T
post-hoc" reviewer complaint.

---

## 8. Definition of success per gap

Each of the three failure modes (§ 2) needs a concrete, measurable
success criterion before work starts.  Without these the chapter
will not have a clean "what we set out to do vs what we achieved"
narrative.

| Gap | Success criterion | Stretch goal |
|---|---|---|
| **Sparse plateau** | Find a permutation with `max_degree < 20` on small-graph for at least one (perm, t) pair, or *prove* (via lower bound) that none exists. | Beat the leaderboard top on small-graph. |
| **Compute wall** | Close ≥ 50 % of the medium-graph gap (down from +136 k HV to < +68 k) at matched seed budget. | Match the leaderboard top on medium-graph. |
| **Feasibility wall** | All 20 synthetic instances + large-graph admit a feasible warm start; the three previously-failing synthetic cells produce non-trivial archives. | All 20 synthetic instances + large-graph improve their best HV by ≥ 5 %. |

These are paper-defensible regardless of whether the headline
leaderboard numbers move, because they describe **what the methods
are designed to do**, not just **whether they happen to win**.

---

## 9. Compute-budget estimates for the next chapter

Realistic per-direction estimates assuming an 8-core x86_64 laptop
(no GPU) for the local runs and a single A100 GPU for the
CUDA-Torso experiment.

| Direction | One full matrix run | 30-seed run | GPU time |
|---|---:|---:|---:|
| MCS-M warm start + hc5 family | 25 min | 12.5 h | — |
| Multi-seed parallel HC (100 seeds × 25 s) | 5 min/instance | 2.5 h total | — |
| Simulated Annealing (single-thread) | 25 min | 12.5 h | — |
| NSGA-II / SMS-EMOA (PyGMO defaults) | 1 h | 30 h | — |
| GBDT surrogate pre-filter (training included) | 1.5 h | — | — |
| CUDA-Torso parallel B&B | — | — | ~1 GPU-week |

Total estimated wall time to produce the **complete next chapter
matrix + 30-seed reporting + ablation**: about three weeks of laptop
compute, plus the CUDA-Torso GPU-week if that direction is included.

---

## 10. Literature snapshot — what the next chapter must engage with

A short list of the references the paper's related-work section
needs to cite.  Listed in rough topical order.

**Chordal completion and treewidth heuristics**
- Tarjan & Yannakakis (1984), *Simple linear-time algorithms to test
  chordality of graphs* — MCS, the building block of MCS-M and
  LEX-M.
- Rose, Tarjan & Lueker (1976), *Algorithmic aspects of vertex
  elimination on graphs* — LEX-M.
- Berry, Blair, Heggernes & Peyton (2004), *Maximum cardinality
  search for computing minimal triangulations of graphs* — MCS-M.
- Bodlaender, Koster & van der Hoeven (2006), *Treewidth:
  Computational experiments*.
- Bodlaender & Koster (2011), *Treewidth computations II: Lower
  bounds*  — MMD+, MMD++.

**Multi-objective EA / hypervolume-based selection**
- Zitzler & Thiele (1999), *Multiobjective evolutionary algorithms:
  a comparative case study and the Strength Pareto approach*.
- Beume, Naujoks & Emmerich (2007), *SMS-EMOA: Multiobjective
  selection based on dominated hypervolume* — directly cited by
  `hc9`.
- Zitzler & Künzli (2004), *Indicator-Based Selection in
  Multiobjective Search* — IBEA, the principle behind `hc9`.

**Local search and metaheuristics**
- Glover (1989, 1990), *Tabu Search Parts I & II* — cited by
  `hc13`.
- Lourenço, Martin & Stützle (2003), *Iterated Local Search* —
  cited by `hc11`.
- Burke & Bykov (2017), *The late acceptance Hill-Climbing heuristic*,
  EJOR 258(1), 70–78 (canonical journal version of the 2008 PATAT
  paper) — cited by `hc8`.

**Methodology for empirical comparison**
- García, Fernández, Luengo & Herrera (2010), *Advanced
  nonparametric tests for multiple comparisons in the design of
  experiments in computational intelligence and data mining* —
  Friedman + Nemenyi.

**CUDA-accelerated treewidth**
- Pan & Zhao (2023), *Parallel treewidth computation on GPUs*.
- Yamaguchi, Yoshida & Kawarabayashi (2022), *Computing treewidth
  on the GPU*.

**Neural heuristics for permutation problems (background; not
adopted)**
- Vinyals, Fortunato & Jaitly (2015), *Pointer Networks*.
- Kool, van Hoof & Welling (2019), *Attention, Learn to Solve
  Routing Problems!*.
- Schuetz, Brubaker & Katzgraber (2022), *Combinatorial
  Optimization with Physics-Inspired Graph Neural Networks* — the
  paper whose results motivate our scepticism in § 4.3.

---

## 11. Threats to validity for the next chapter

Pre-registered risks, so the paper's discussion section can address
them honestly.

1. **Seed selection bias.**  The headline scoreboard reports seed 42,
   but the small/medium hc15-over-hc9 claim is now backed by a
   twelve-seed sweep (seeds 1–11 + 42; see RESULTS § 4.3): hc15 carries
   the best mean and wins 7/11 and 8/11 head-to-head seeds.  The margin
   is within the seed-to-seed spread, so the honest framing is
   "best-by-a-whisker, multiseed-confirmed in direction."  Extending to
   30-seed runs with a Wilcoxon signed-rank test (§ 5.1) would convert
   the directional result into a significance claim and is the remaining
   step for the journal version.
2. **Hyperparameter post-hoc tuning.**  SA's `T_initial`, LAHC's
   `L`, ILS's stagnation `K` are all sensitive to choice.  The paper
   must either fix them in advance and report unsweeted results, or
   run a sweep and report the *range* of outcomes across the
   sweep.
3. **Instance over-fitting.**  We have 23 instances total (3
   official + 20 synthetic).  Any "method-X wins on N / 23"
   statement is sensitive to which 23 we chose.  Reporting on PISA
   / DTLZ benchmarks would broaden the claim; staying with SpOC-3
   keeps the claim narrow but honest.
4. **Evaluator correctness drift.**  `test_correctness.py` pins
   `core.evaluate` and `core.hypervolume_2d` against independent
   references.  This must stay in CI for the rest of the thesis;
   any silent change to the evaluator invalidates all prior runs.
5. **Compute disparity vs the leaderboard.**  The leaderboard was
   built over an unknown wall-clock budget.  Direct
   leaderboard-comparison claims should include "at our budget"
   caveats; otherwise the paper risks claiming a methodological
   improvement when the gap is really compute-related.
6. **Reference point sensitivity for HV.**  We use ref = (n, n).
   PISA and most multi-objective EA papers use other choices.  Any
   external HV comparison must explicitly state which reference
   point is used and translate between them when the literature uses
   a different one.
7. **Statistical-test power.**  30 seeds is on the low end for
   Mann–Whitney U with multiple-comparison correction across 16
   algorithms × 23 instances.  Either increase to 100 seeds or use
   one-vs-all post-hoc tests with Bonferroni.

---

## 12. Target paper outline (this chapter and the two that follow)

To keep direction clear, the planned thesis structure:

**Chapter 1 — Hill Climbing for SpOC-3 Torso Decompositions** *(this
work; in preparation)*.  Contributions: (a) the 14-variant taxonomy,
(b) the SMS-EMOA-style HV-acceptance win on this problem, (c) the
Pareto-archive-subsumes-classical-escape observation, (d) the
incremental bitset evaluator, (e) the 20-instance synthetic
benchmark and the feasibility-frontier finding.

**Chapter 2 — Simulated Annealing & population-based MOEAs**
*(planned)*.  Contributions: (a) SA vs `hc8` head-to-head, (b)
NSGA-II / SMS-EMOA external baselines, (c) HV-acceptance vs HV-based
*selection* (SMS-EMOA proper), (d) the per-t torso-aware warm start
generalised across families.

**Chapter 3 — Width-respecting construction & exact methods**
*(planned)*.  Contributions: (a) MCS-M / LEX-M / width-budgeted
min-fill on SpOC-3, (b) SAT-based optimal sub-problem solving with
splice-back, (c) treewidth lower bounds (MMD+) as
gap-to-optimum certificates, (d) CUDA-Torso style parallel B&B as
the "exact upper bound" experiment.

**Chapter 4 (open) — Generalisation beyond SpOC-3**.  Possible
directions: (a) the chapter-1 HV-acceptance result on PISA / DTLZ
benchmarks, (b) neural heuristics revisited if pointer-network
methods mature for treewidth problems, (c) cross-instance transfer
of warm-start policies.

---

Beyond the immediate engineering, the chapter has surfaced three
research questions that are worth posing for follow-up work:

- **Is there a width-respecting elimination heuristic with provable
  performance vs. the optimum?**  Min-fill, MCS-M, LEX-M, k-core
  peeling all have empirical reputations but no tight competitive
  ratio for the SpOC-3 cost function.  A paper that proves a
  *constant-factor approximation* on the synthetic Erdős-Rényi grid
  would be a publication on its own.
- **Does HV-acceptance generalise?**  `hc9` won on 14 / 23
  instances; does the result hold on PISA test problems, NSGA-II
  test suite, or other multi-objective combinatorial benchmarks?
  This question takes the chapter's findings beyond the SpOC-3
  problem.
- **What is the optimal trade-off between archive size and
  per-iteration cost in HV-driven HC?**  We use top-K-by-HV-DP
  selection (`core.ParetoArchive.top_k_by_hv_contribution`) and a
  size cap of 20.  Whether the cap should be larger and HV-pruned
  on each iteration vs. larger and re-pruned periodically is
  unexplored — and the answer might be problem-dependent.

---

## Glossary of techniques mentioned

| Term | Meaning | Cited in |
|---|---|---|
| **Min-degree elimination** | Greedy heuristic that repeatedly eliminates the current-lowest-degree vertex.  Building block of `hc4`-`hc13` warm starts. | § 2.3, ALGORITHMS.md |
| **Min-fill elimination** | Repeatedly eliminate the vertex whose removal adds the fewest fill-in edges.  Empirically slightly better than min-degree on many instances. | § 2.3 |
| **MCS** (Maximum Cardinality Search) | Tarjan & Yannakakis 1984.  At each step pick the unprocessed vertex with the most already-processed neighbours. | § 7.1, § 10 |
| **MCS-M** | Berry et al. 2004.  MCS with a label-propagation rule that produces minimum-fill orderings. | § 2.3, § 7.1 |
| **LEX-M** | Rose, Tarjan & Lueker 1976.  Lexicographic-BFS-based elimination producing perfect orderings on chordal graphs. | § 2.3, § 10 |
| **k-core peeling** | Iteratively remove vertices with degree < k; the k-core is the maximal subgraph where every vertex has degree ≥ k.  Used as a partition heuristic in `hc10`. | § 2.3, ALGORITHMS.md |
| **Treewidth** | The minimum, over all elimination orders, of the maximum fill-in degree.  Lower-bounds `max_degree` for any solution. | § 2.1 |
| **MMD+ / MMD++** | Minor-min-width treewidth lower bounds (Bodlaender & Koster 2011). | § 5.4, § 10 |
| **Hypervolume (HV)** | The 2-D Lebesgue measure of the union of axis-aligned rectangles dominated by a Pareto front against a reference point.  The SpOC-3 score is `−HV` with ref `(n, n)`. | PROBLEM.md |
| **HSSP** (Hypervolume Subset Selection) | Given m non-dominated points, pick the K-subset with the highest HV.  Optimal in O(K · m²) for 2D via DP.  Used by `core.ParetoArchive.top_k_by_hv_contribution`. | ALGORITHMS.md |
| **Lex acceptance** | "Accept candidate if `(max_w, soft_cost)` is lexicographically smaller."  Used by `hc3`-`hc7`, `hc10`, `hc11`, `hc13`. | ALGORITHMS.md, § 1 |
| **HV-acceptance** | "Accept candidate iff the Pareto archive's HV would strictly increase."  Used by `hc9`. | § 1 |
| **Late Acceptance HC (LAHC)** | Burke & Bykov 2017 (EJOR; canonical journal version of the 2008 PATAT paper).  Accept iff the candidate is better than the *current* state OR better than the state from L iterations ago. | ALGORITHMS.md |
| **Iterated Local Search (ILS)** | Lourenço et al. 2003.  On stagnation, apply a strong perturbation to best-so-far without acceptance, resume local search. | ALGORITHMS.md |
| **Tabu Search** | Glover 1989/1990.  Record recent move signatures; forbid them for L iterations unless they trigger an *aspiration* (e.g. produce a new archive entry). | ALGORITHMS.md |
| **Pareto archive** | A set of non-dominated `(max_degree, t)` points seen during search; persists across iterations and accumulates the candidate front. | PROBLEM.md, ALGORITHMS.md |
| **Over-width** | A solution whose elimination at some step produces a fill-in degree > 500.  Evaluator returns `max_degree = 501`. | PROBLEM.md |
| **Width-respecting construction** | A warm-start heuristic that *proactively* refuses moves that would exceed 500-width.  Needed to break the feasibility wall on dense graphs. | § 2.3, § 7.1 |
| **Incremental bitset evaluator** | Re-walks only the suffix of the permutation that an operator actually changed; reuses cached chordal-completion state for the prefix.  `core.IncrementalEvaluator`, used by `hc12`. | ALGORITHMS.md |
| **Multi-seed parallelism** | Running N independent search seeds concurrently and merging the resulting Pareto archives via HSSP DP. | § 7.2 |
| **Simulated Annealing (SA)** | Probabilistic local search; accept with probability `exp(−ΔE / T)` where T cools over time.  Direct contrast to `hc8` LAHC. | § 3, § 7.3 |
| **NSGA-II** | Deb et al. 2002.  Multi-objective EA with non-dominated sorting and crowding distance.  Canonical baseline. | § 3 |
| **SMS-EMOA** | Beume, Naujoks & Emmerich 2007.  Population-based EA whose selection criterion is HV-contribution.  Population variant of `hc9`. | § 3, § 10 |
| **IBEA** | Zitzler & Künzli 2004.  Indicator-Based EA; family that contains SMS-EMOA. | § 1, § 10 |
| **CUDA-Torso** | Pan & Zhao 2023.  GPU-parallel branch-and-bound for treewidth.  The likely technique behind the leaderboard top on large-graph. | § 2.3, § 10 |
| **GBDT** | Gradient-Boosted Decision Trees.  Tabular regressor family (LightGBM, XGBoost, CatBoost are three implementations).  Considered as a surrogate fitness model in § 4.1. | § 4.1 |
| **Neuro-evolution / NEAT / CMA-ES** | Evolutionary algorithms applied to neural-network weights.  Considered and skipped in § 4.2. | § 4.2 |

---

## See also

- [EXPLAINER.md](EXPLAINER.md) — the problem from scratch in plain language
- [PROBLEM.md](PROBLEM.md) — formal problem statement
- [ALGORITHMS.md](ALGORITHMS.md) — the 15 HC variants + SA / GRASP / VNS
- [RESULTS.md](RESULTS.md) — the full scoreboard (HC + metaheuristics § 4b)
