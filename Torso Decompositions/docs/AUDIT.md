# Project Audit — SpOC-3 Torso Decompositions

_Rigorous correctness and methodology audit. Author: Adrian Ymeri.
Status: **in progress** (Phase 1 algorithms — near complete; Phase 2 parameters —
pending; conducted while a 16-seed cmaes sweep runs, so no file in `submissions/`
is touched)._

Every claim below is either (i) derived from the source and stated as a proof
sketch, or (ii) verified empirically with a reproduction snippet whose output is
quoted. Findings are labelled **BUG-n** (defects), **OBS-n** (latent risks /
hardening), or **OK-n** (verified-correct components worth recording). Severity
is rated by *effect on the official score* first, then performance and
maintainability.

---

## 0. Scope and method

The audit proceeds bottom-up: the objective evaluator and hypervolume are the
foundation on which every algorithm and every reported number rests, so they are
verified first and most stringently. Only once the scoring core is trusted do
the search algorithms and their parameters become meaningful to audit.

Artifacts examined in Phase 1: `core.py` (`evaluate`, `evaluate_full`,
`hypervolume_2d`, `score`, `ParetoArchive`, `top_k_by_hv_contribution`,
`IncrementalEvaluator`, warm starts) and `algorithms/continuous/cmaes_torso.py`
(the continuous-encoding evaluator). Reproductions were run against the live
`core.py`; the official scorer path (`tools/verify_submission.py`) is the
ground-truth oracle throughout.

---

## 1. Phase 1 — Algorithm correctness

### OK-1 — The fill-in evaluator is correct, and `t`-independent

`evaluate_full` (core.py:216) plays the elimination game: at step `i` it forms
`succ = temp[perm[i]] & suffix_mask[i]` (the later neighbours of the eliminated
vertex), records its degree, and adds the fill-in clique
`temp[v] |= succ ^ vbit` for every `v ∈ succ`. Two properties matter and both
hold:

1. **The fill-in graph is independent of `t`.** The threshold `t` enters only
   through (a) which steps contribute to `max_width` (`if i >= t`) and (b) the
   soft cost — never through the elimination itself. Hence the per-step degree
   sequence `deg[i]` is a function of the permutation alone.
2. **The cap is global.** `if deg > MAX_TW: return MAX_TW+1` fires at *any*
   step, head or torso (core.py:252). So a permutation with any step exceeding
   `MAX_TW = 500` is infeasible for *every* `t`.

These two facts are the formal justification for the continuous-encoding
evaluator's single-pass trick (see **OK-5**).

### OK-2 — `hypervolume_2d` computes the exact union area

`hypervolume_2d` (core.py:509) filters points to `x<n ∧ y<n`, builds the
lower-left non-dominated staircase (x ascending ⇒ y strictly descending), and
sums vertical strips `(x_{i+1} − x_i)(n − y_i)` with the right edge clamped to
`n`. This is exactly the area of the union of the dominated rectangles
`[x,n]×[y,n]` against reference `(n,n)`, i.e. the official score's `−HV`. The
staircase construction makes the result invariant to dominated points in the
input (verified in **BUG-1**).

### BUG-1 — `ParetoArchive.try_add` never prunes dominated points (medium)

**Defect.** `try_add` (core.py:572) correctly *rejects* a new point dominated by
an incumbent (first loop, lines 585–587). But the second loop, intended to
*evict incumbents the new point dominates*, tests

```python
if w >= w2 and t >= t2 and (w > w2 or t > t2):   # core.py:590
    continue   # drop incumbent
```

`w >= w2 ∧ t >= t2` means *the new point is dominated by the incumbent* — the
opposite of the intended direction, and a case already short-circuited by the
first loop. The eviction therefore never fires, and incumbents that the new
point dominates are retained.

**Evidence.**

```text
add (20,100): True
add (15,100) [dominates (20,100)]: True
archive points: [(15, 100), (20, 100)]      # (20,100) should have been pruned
...
front: [(10,500),(20,300),(30,100)]
after adding (5,50) which dominates all: [(5,50),(10,500),(20,300),(30,100)]
```

**Impact — no effect on any reported score.** Because `hypervolume_2d`
recomputes the non-dominated staircase, the archive's HV is identical with or
without the stale points:

```text
archive HV (dirty): 1767064.0  vs clean front HV: 1767064.0   EQUAL: True
```

Every published number is therefore correct. The real consequences are:

- **Submission cosmetics.** `top_k_by_hv_contribution` receives a dirty front;
  when the true non-dominated front has fewer than 20 points it pads the
  remaining submission slots with dominated, zero-contribution vectors. This is
  the origin of the "1 dominated vector / 19 non-dominated front" note seen on
  `small-graph`. The *submitted HV is still optimal* — padding adds zero — but
  slots are wasted.
- **Performance.** The archive is monotone-growing in distinct `(w,t)` pairs;
  on long runs (the 900 s cmaes seeds) this inflates `try_add` (O(m) per call)
  and the finalisation DP (O(k·m²)). A latent slowdown, not a failure.

**Fix (safe; apply after the sweep).** Flip the eviction test to prune
incumbents the new point dominates:

```python
if w2 >= w and t2 >= t and (w2 > w or t2 > t):
    continue
```

This restores the archive to a proper Pareto set and re-establishes the
precondition `top_k_by_hv_contribution` assumes.

**Resolved (post-sweep, 4 June 2026).** The fix is applied (core.py:589–600) and
verified. Score is invariant, as predicted, and the submission is now strictly
cleaner:

```text
small-graph portfolio after fix: -1,828,237  (identical score; value at time of
                                              the fix — final portfolio later -1,828,451)
  Dominated vectors: 0  (non-dominated front size = 15)   # was 20 incl. 1 dominated
```

i.e. the archive now submits only the 15 genuinely non-dominated points instead
of padding to 20 with zero-contribution duplicates. Verified unit behaviour:
dominated incumbents are pruned, the true front is preserved, and a dominated
newcomer is still rejected.

### OK-3 — HSSP top-k selection is HV-optimal on a clean front

`top_k_by_hv_contribution` (core.py:603) solves the 2-D Hypervolume Subset
Selection Problem with the standard O(k·m²) DP
`f(i,j) = max_{i'>i} (x_{i'} − x_i)(n − y_i) + f(i', j−1)`, reconstructing the
subset via successor pointers. The recurrence is exact *provided the input is a
non-dominated staircase*. Under **BUG-1** the input can be dirty; the DP then
maximises a quantity that under-counts configurations containing dominated
points, but since such configurations never beat the clean front it still
returns a subset whose *true* HV equals the full front's HV. Net: optimal score,
fragile precondition — closed by the **BUG-1** fix.

### OBS-1 — `hypervolume_2d` / `score` do not enforce the width cap (low)

`hypervolume_2d` admits any point with `x < n`; a capped width
`MAX_TW+1 = 501 < n` therefore *contributes* unless the caller pre-filters:

```text
HV with capped point (501,0): 1161592.0   (> 0)
```

On the production path this is harmless — `ParetoArchive.try_add` rejects
`w > MAX_TW`, and `tools/portfolio.py` / `verify_submission.py` filter to
`w ≤ MAX_TW` — which is why all submissions verify 0-capped. But
`core.score` (core.py:532) calls `hypervolume_2d` over raw `evaluate` outputs
*without* filtering, so it would over-count a capped vector. **Recommendation:**
have `hypervolume_2d` (or a thin wrapper used by `score`) drop `x > MAX_TW`
points, making the cap contract explicit at the scoring boundary rather than
implicit in every caller.

### OK-5 — The continuous-encoding evaluator is consistent with the oracle

`cmaes_torso.eval_fitness` computes the degree sequence in one elimination pass
and returns, for a feasible permutation, the front
`{(suffix-max(deg[t:]), t) : t ∈ grid}`. By **OK-1** the per-step degrees are
`t`-independent and `suffix-max(deg[t:]) = max_{i≥t} deg[i] =` the `max_width`
that `evaluate_full` returns at threshold `t`. Hence each archived `(w,t)` equals
`core.evaluate(perm, t)`, and the official re-score reproduces the in-run number
exactly — confirmed end-to-end: the 8/16-seed portfolios re-scored by
`verify_submission` are 0-capped and bit-identical to the in-run values
(small −1,828,063, medium −1,704,738, large −5,369,684).

Infeasible permutations return a graded positive penalty
`501 + Σ(deg − 500)` with early-stop at penalty > 1000 — a direct port of
cuda-torso's `libeval.cu` — supplying the feasibility gradient that the dense
instance requires. This is a *design* property, not a correctness deviation: the
penalty branch is only ever reached by permutations the oracle also rejects
(any step > 500 ⇒ capped), so no feasible point is ever mis-scored.

---

### OK-6 — Warm-start construction heuristics are correct

`min_degree_perm` (core.py:703) is a faithful greedy minimum-degree elimination:
each step eliminates a remaining vertex of least induced degree and adds the
fill-in clique `g[u] |= nbrs ^ ubit` among its remaining neighbours. The
`d == 0 ∧ rng is None` early-break is a valid optimality shortcut (a
degree-0 vertex incurs no fill). `min_fill_in_perm` (core.py:739) scores each
candidate by `fill = C(d,2) − existing`, where `existing` counts edges already
present among the neighbourhood (each counted twice, hence `// 2`) — i.e. the
exact number of fill edges its elimination would add. **Caveat (by design):** a
`sample_size = 64` guard restricts the fill scan to the 64 lowest-degree
candidates, so this is a *degree-pruned approximate* min-fill, not exact. This
is a deliberate cost bound, documented in the source, and consistent with the
project's empirical finding that min-fill's advantage over min-degree is a few
width units on sparse cells and nil on the dense one; it is recorded here only
so the "min-fill" label is not over-read as an exact triangulation.

### OK-7 — The GPU batch evaluator is bit-for-bit consistent with the oracle

The GPU scale-up (THESIS.md §9) introduces a parallel evaluator
(`algorithms/continuous/gpu_eval.py`) that must reproduce `core.evaluate`
exactly, or any GPU score is inadmissible. This is enforced, not assumed.
`tools/validate_gpu.py` builds a 512-ordering mix (random, min-degree,
feature-decoded), evaluates it on the GPU and with a numpy reference, and checks
the per-step degree sequences and feasibility status match; the reference is in
turn cross-checked against `core.evaluate`. On `large-graph` (n = 2426, W = 76)
the run reports **0 status mismatches and 0 degree-sequence mismatches**, and the
reference-vs-`core.evaluate` cross-check is OK. The kernel mirrors `eval_fitness`
semantics (graded cap penalty, bail at penalty > 1000, per-permutation
feasibility, suffix-max staircase).

One **disclosed discrepancy (OBS-2-like, low):** the *search's* live archive
bookkeeping in `gpu_search.py` (built from the 40-point t-grid) ran ≈ 1,650 HV
**optimistic** versus the authoritative full-staircase re-score by
`tools/portfolio.py`. The per-ordering evaluator is exact (above); the gap is in
the in-loop HV estimate, not the scorer. Accordingly the thesis reports the
**official re-scored** large-graph figure (−5,405,118, +6,046 HV), never the
search's internal −5,406,768. Any GPU result is banked only after an independent
`portfolio.py` / `verify_submission.py` re-score on the official evaluator.

### OK-8 — The GAPS result and its GBDT ablation are official and controlled

The novel method (THESIS.md §10, `tools/gaps_search.py`) is reported only on
**official** numbers and a **controlled** ablation:

- *Official scoring.* The headline −5,431,595 is the `tools/portfolio.py` /
  `refine_thresholds.py` re-score of the pooled submissions (full-staircase
  top-20), and the per-method figures (−5,404,348 / −5,406,555 / −5,431,321) are
  each file's top-20-by-HV-contribution submission scored by `core.evaluate`.
  The search's *internal* GAPS estimate (−5,437,116) ran ~5.8k optimistic and is
  **not** reported as the result — same discipline as OK-7.
- *Controlled ablation.* "with-GBDT" and "without-GBDT" differ in exactly one
  factor: whether `gaps_search.py` retrains the GBDT column (`--gbdt-every 8` vs
  `--gbdt-every 0`). Warm start, seed, population, polynomial basis, evaluator and
  budget are identical. The +24,766 HV margin is therefore attributable to the
  GBDT column alone. It is corroborated *causally* by the trajectory (fig11): the
  score is flat until the generation the GBDT first trains, and *statistically* by
  a multi-seed repeat (`tools/gaps_ablation_stats.py`): +18,474 HV at 4.6× the
  combined run-to-run std (with-GBDT −5,419,856 ± 497 over 3 seeds; without
  −5,401,382 ± 3,959 over 5), with GBDT also cutting variance ~8×. So the effect
  is robust, not a single-seed artefact.
- *Negative control recorded.* The earlier random-interaction variant
  (`--poly-extra 128`) stalled at the warm start; reported in §10.1 so the
  "nonlinearity helps" claim is not over-read — only the *learned* nonlinearity
  helps.

### OK-9 — The GBFC gains are official, additive, and lower-bound-bounded

GBFC (THESIS.md §11, `tools/gbfc.py`), the thesis's primary novel method, is held
to the same discipline as OK-7/OK-8:

- *Official scoring.* The reported gains (small +688 → −1,828,994; medium +734 →
  −1,712,688; large +329 → −5,431,924) are each the `tools/portfolio.py` top-20
  re-score of the pooled submission, **minus** the same re-score of the banked
  front *without* GBFC's orderings — i.e. GBFC's own *marginal* contribution to
  the verified portfolio, not the search's internal estimate. GBFC is checkpoint-
  safe (`save_sub()` writes a re-scorable submission every round), so any quoted
  number is independently re-checkable with `tools/verify_submission.py`.
- *Additive-only.* GBFC writes its own submission stems and is pooled into the
  portfolio additively; it never overwrites a canonical file, so a GBFC regression
  could only fail to improve the front, never damage it (the pool keeps the best
  per `(width, t)`). This is why every reported gain is non-negative.
- *Bounded by the certificate.* The gains are small in absolute terms because the
  banked front is already near-optimal: the §5.2 / §10.5 treewidth lower bound
  (MMD minor-min-width) certifies the dense core is *provably* optimal (width =
  499 = LB at t = 0) and the front sits a small, decelerating distance above the
  bound. GBFC's deceleration (small: +554 then +40 across two iterations) is the
  expected signature of consuming that bounded residual, not a convergence bug.
- *Scale-up estimates quarantined.* The in-progress `tools/qnegbfc.py` hybrid
  prints an `official`-labelled running score, but per project discipline those
  remain **internal** until re-scored by `portfolio.py`; no hybrid number has
  entered THESIS.md / RESULTS.md. (Observed live: small flat at the banked
  −1,828,994; medium moving — both pending re-score.)

## 2. Phase 1 — remaining algorithm families _(spot-check pending)_

Lower-yield items deferred to a second pass: the incremental evaluator's
"bit-for-bit" equality claim (asserted in-source with a regression hook), the
`ensure_seeded` over-width fallback, and one representative acceptance rule per
search family (HC HV-acceptance, SA Metropolis energy, GRASP RCL, VNS ladder,
NSGA-II/SMS selection). The built-in separable CMA-ES update equations
(`SepCMAES`) will be checked against Ros & Hansen (2008); note its empirical
validity is already strong (it beats full-covariance fcmaes head-to-head, § Phase 2).

## 3. Phase 2 — parameter / tuning audit

Method: for every metaheuristic family, the winning full-grid cell per instance
was extracted from `extra_instances/tuning_*.csv` and each winning parameter was
flagged `@MIN`/`@MAX` if it sits on a grid boundary. A boundary optimum has two
possible readings — *untested headroom* (the true optimum lies beyond the grid)
or *reduction to the simplest setting* (the family's distinctive machinery is
not paying, and tuning collapses it to a degenerate baseline). Disambiguating
the two is the core of this phase.

### PARAM-1 — Permutation-space optima collapse to the *simplest* setting (key)

Across families the winning cells sit overwhelmingly at the boundary that
*removes* each method's distinctive mechanism, not at the boundary that would
ask for more of it:

| Family | Winning cell pattern | Interpretation |
|---|---|---|
| GRASP | `alpha = 0` @MIN (small, medium) | pure greedy — the RCL randomisation does not help |
| GRASP | `restarts = 4` @MIN (small, large) | depth per restart beats many restarts |
| VNS | `k_max = 2` @MIN (all three) | the neighbourhood ladder collapses to plain ILS |
| NSGA-II / SMS | `pop = 20` @MIN (≈ everywhere) | iteration depth beats population breadth |
| ACO | `beta = 4` @MAX, large `= −0` | pheromone ignored (heavy heuristic) / fails outright |

This is **not** untested headroom — none of these boundaries can be pushed
"further" in a way that adds capability (there is no `alpha < 0`, `k_max < 2`,
or useful `pop < 20`). It is the signature of **landscape saturation**: at the
canonical budgets the problem rewards greedy depth, and each family's extra
apparatus (randomised construction, variable neighbourhoods, large populations,
pheromone memory) is dead weight that tuning correctly switches off. This both
*corroborates* the "permutation families are at their limit" conclusion and
*explains its mechanism* — they are not mis-tuned, they are structurally
reduced to the same greedy core.

### PARAM-2 — Two genuine boundary-headroom cases, both in frozen families

The only winning cells at a *capability-adding* boundary are `SA/small`
(`schedule = linear` @MAX, `steps_per_t = 400` @MAX) and `VNS/small`
(`ls_steps = 600` @MAX) — i.e. "spend more of the budget on local search."
These are real untested headroom, but both families are **frozen** (beaten by
cmaes by >10⁴ HV), so the headroom is moot for the score. Recorded for
completeness; not worth pursuing.

### PARAM-3 — The cmaes eigenvector count has an empirical optimum at K ≈ 32

The active method's one structural hyperparameter is the spectral feature count
K. The sweeps bracket it directly on `large-graph` (best single seed):

| K (eigenvectors) | search dim | large best single | note |
|---:|---:|---:|---|
| 16 | 21 | ≈ −5.29 M (1 seed, 22 s) | under-resolved features |
| **32** | **37** | **−5,376,007** | **empirical sweet spot** |
| 48 | 53 | −5,353,725 (2 seeds stuck at the warm-start floor −4,784,461) | over-dimensioned: CMA-ES cannot converge in budget |

The trade-off is principled: more eigenvectors enrich the policy's expressive
power but raise the search dimension, and at a *fixed wall budget* the
higher-dimensional CMA-ES converges more slowly — at K = 48 two of eight seeds
never escaped the warm-start seed. K = 32 is therefore not arbitrary; it is the
knee of the features-vs-convergence trade-off at the 600–900 s budget. (On a GPU
with 10³–10⁴× more evaluations the optimum K would shift upward, consistent with
cuda-torso's use of 32 eigenvectors *plus* a polynomial expansion.)

### PARAM-4 — The "builtin ≥ fcmaes" conclusion is fair, with one caveat

The engine comparison (Phase-2 of `FUTURE.md` §1g) held popsize matched at
`4 + 3·ln(dim) ≈ 14` for both engines, so it cleanly answers "does
full-covariance beat diagonal *at equal sampling*?" — and the answer is no
(builtin wins on all three instances). **Caveat:** fcmaes's *native* default
popsize is 31; a larger population could let full-covariance amortise its
O(dim²) adaptation cost differently. This was not swept. The conclusion as
stated — *full-covariance is not worth its cost at matched popsize on this
≈37-dim landscape* — is sound; a complete claim would add an fcmaes popsize
sweep. Low priority: even a favourable fcmaes result would be an optimiser
refinement, not the throughput lever the large-graph gap actually needs.

### Findings — Phase 2

| ID | Subject | Verdict |
|---|---|---|
| PARAM-1 | permutation grids collapse to simplest setting | saturation, not mis-tuning — corroborates "at limit" |
| PARAM-2 | SA/VNS small-graph @MAX headroom | real but moot (frozen families) |
| PARAM-3 | cmaes K = 32 eigenvectors | empirically optimal at the knee of the features/convergence trade-off |
| PARAM-4 | builtin ≥ fcmaes | fair at matched popsize; fcmaes-native popsize unswept (minor) |

---

## Findings summary

| ID | Component | Severity | Score impact | Status |
|---|---|---|---|---|
| OK-1 | `evaluate_full` (fill-in, cap, `t`-independence) | — | correct | verified |
| OK-2 | `hypervolume_2d` (union area) | — | correct | verified |
| BUG-1 | `ParetoArchive.try_add` dominated-pruning | medium | **none** | **FIXED** + verified (score-invariant, cleaner submissions) |
| OK-3 | `top_k_by_hv_contribution` HSSP DP | — | optimal | verified (clean input) |
| OBS-1 | cap not enforced in `hypervolume_2d`/`score` | low | none on prod path | hardening proposed |
| OK-5 | `cmaes_torso.eval_fitness` vs oracle | — | consistent | verified end-to-end |
| OK-6 | `min_degree_perm` / `min_fill_in_perm` warm starts | — | correct (min-fill degree-pruned) | verified |
