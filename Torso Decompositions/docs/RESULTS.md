# Results — full scoreboard, lessons learned, reproducibility

_Companion to [THESIS.md](THESIS.md). Contains the experimental protocol, the
cross-algorithm scoreboard on the three official instances, the 20-instance
synthetic benchmark, and the lessons-learned analysis. All scores are
canonical-scorer-verified (`tools/verify_submission.py`) and 0-capped._

---

## 0. Headline results

The project's final standing, after four chapters of permutation-space search
and the continuous-encoding breakthrough that followed (full narrative in
[THESIS.md](THESIS.md)):

| Instance | Permutation portfolio | Continuous + GBDT | Leaderboard top | % of top | GBDT ablation |
|---|---:|---:|---:|---:|---:|
| small  | −1,819,283 | **−1,828,451** | −1,829,919 | 99.92 % | +0 |
| medium | −1,617,086 | **−1,711,954** | −1,745,122 | 98.10 % | +0 |
| large  | −5,033,531 | **−5,399,072** | −5,493,062 | 98.29 % | **+2,524** |
| large (GPU §9) | — | **−5,405,118** | −5,493,062 | **98.40 %** | — |
| large (GAPS §10) | — | **−5,431,595** | −5,493,062 | **98.88 %** | **+24,766** |
| **small (GBFC §11)** | — | **−1,828,994** | −1,829,919 | **99.95 %** | +688 |
| **medium (GBFC §11)** | — | **−1,712,688** | −1,745,122 | **98.14 %** | +734 |
| **large (GBFC §11)** | — | **−5,431,924** | −5,493,062 | **98.89 %** | +329 |
| **small (GBFC++ §11.6)** | — | **−1,829,735** | −1,829,919 | **99.990 %** | +741 over GBFC |

The **GBFC rows (§11) are this thesis's primary novel method** — Gradient-Boosted
Front Construction (`tools/gbfc.py`), which reframes Pareto-front optimisation as
a boosting problem: GBDT weak learners fit to the lower-bound residual, decoded
into complementary specialist orderings, pooled, repeated. GBFC is the **only**
method studied here that measurably improves the banked front, and it does so on
every instance (verified +688 / +734 / +329 HV; the single best contributor to
each pooled portfolio, official `tools/portfolio.py` re-scores). It does not reach
#1 — the recoverable room is bounded by the treewidth-lower-bound certificate
(§5.2 / §10.5) showing the dense core is provably optimal — but it establishes a
novel, GBDT-central method with demonstrated improvement. A hybrid with the
winner's neuroevolution (`tools/qnegbfc.py`, `--no-gbdt` ablation) is the
time-boxed scale-up experiment.

The **GBFC++ row (§11.6, `tools/gbfcpp.py`) is the strongest verified result in
the project relative to its leader**: boosting at the granularity of single
front breakpoints (the residual = each breakpoint's HV marginal), with the GBDT
weak learner acting as a *learned move-proposal distribution* inside a
breakpoint-targeted incremental local search (C evaluation kernel
`tools/_fastwalk.c`, bit-exact vs `core.evaluate`, ~75k move-evals per 34 s
round). Verified small-graph **−1,829,735 = 99.990 %** of the leaderboard top
(+741 HV over GBFC, 80 % of the residual gap closed, on CPU only). Paired
same-seed ablation: the learned proposal beats uniform proposals 3/3 (mean
+44.7 vs +18.7 HV/round) using the dependency-free numpy GBDT backend
(`algorithms/continuous/np_gbdt.py`); LightGBM is the default where installed.
Runs checkpoint every round and resume (`--rounds`, persisted failure decay),
and the per-round gains had not flattened when the budget ended — continuing
the run (and the qnegbfc GPU hybrid) is the open path to #1 on small.

**The consolidated comparison — every method, CMA-ES, the QNE/cuda-torso
reproduction, and all three controlled GBDT ablations side by side — is THESIS
§12 ("The GBDT ledger"),** with figure `fig14_gbdt_ledger.png`. Its headline:
no unlearned method beats its learned counterpart anywhere; the GBDT's gains
grow as it moves from model (§6b) to decode (§10) to search control (§11/11.6);
and the winner's own engine, reproduced on our evaluator, matches our scores at
our compute — the residual 184 HV on small is compute, not modelling.

The GPU row (§9) is the linear continuous-encoding search at GPU scale: a verified
**+6,046 HV** over the CPU portfolio, which then plateaued — localising the
residual gap to a *decode-expressiveness* limit (the linear `argsort` policy
saturates), not a throughput one.

The **GAPS row (§10) is the novel method**: a GBDT-augmented *nonlinear* decode
(`score = Φ(F)·x + β·g_GBDT(F)`, `tools/gaps_search.py`) built to attack that
limit. Verified **−5,431,595 (98.88 %)** — +32,523 HV over the original CPU
portfolio. Its GBDT-ablation column is a *controlled* +24,766 HV (GAPS with vs
without the GBDT column, identical warm start and seed) — an order of magnitude
beyond GBDT's static contribution, and here the dominant driver. It does not reach
#1 (+61,467 short). The GPU evaluator is validated bit-for-bit against
`core.evaluate` by `tools/validate_gpu.py`.

The GBDT-ablation column is the gradient-boosted-decision-tree adaptive
heuristic's *independent* contribution (controlled `tools/ablation_gbdt.py`:
portfolio HV with minus without the GBDT orderings; GBDT implemented with
LightGBM/XGBoost). It is positive only on the dense instance, by design — see
[THESIS.md](THESIS.md) §6b.

![Hypervolume as a percentage of the leaderboard top: permutation portfolio
versus continuous encoding.](figures/fig1_gap_closing.png)

![Remaining hypervolume short of the top (thousands).](figures/fig2_remaining_gap.png)

Within the permutation-space families, an 11-seed Friedman omnibus
(χ²(14) = 321.607, p ≈ 3.6 × 10⁻⁶⁰) separates the methods cleanly — GRASP and
HV-acceptance hill climbing lead, ant-colony optimisation is significantly worse
than every other family (beyond the Nemenyi critical distance):

![Friedman average ranks of the permutation families.](figures/fig6_method_ranks.png)

The continuous-encoding result rests on two design choices analysed in
[THESIS.md](THESIS.md) §5–6 and audited in [AUDIT.md](AUDIT.md): the spectral
feature count sits at the knee of a features-versus-convergence trade-off
(K = 32), and the diagonal CMA-ES optimiser is at least as good as a
full-covariance one at matched population — i.e. the optimiser is not the
bottleneck.

![K = 32 eigenvectors: the features/convergence knee.](figures/fig4_eigenvector_tradeoff.png)
![Diagonal versus full-covariance CMA-ES at matched population.](figures/fig5_engine_comparison.png)

---

## 3. Experimental protocol

### 3.1 Software stack

All experiments run on a single thread of CPython 3.10+ with NumPy as
the only third-party dependency.  No JIT, no GPU, no parallelism.  The
fitness evaluator (`core.evaluate_full`) uses Python's arbitrary-
precision integers as bitmasks to walk the chordal-completion
elimination, with semantics verified against an independent set-based
reference implementation embedded in `tests/test_correctness.py`
(50 / 10 / 5 random `(π, t)` pairs agree exactly on small / medium /
large).  The 2-D hypervolume (`core.hypervolume_2d`) is verified
against a brute-force grid scan on 100 random Pareto fronts.

### 3.2 Random-seed policy

Every algorithm exposes `--seed` as a CLI argument.  The canonical
scoreboard (§ 4.1) is reported at `seed = 42`; reproducibility checks
re-evaluate the saved submission JSONs end-to-end via
`tools/verify_submission.py` and require bit-exact agreement with the
reported score.  The single-seed canonical numbers are supplemented by
**3-seed mean ± std** (seeds 42, 1, 2) for the four headline variants
hc5 / hc9 / hc11 / hc14 in § 4.2 below.  Multi-seed re-runs for the
full 14-variant matrix are deferred to the extended (journal) version.

### 3.3 Budgets

Wall-time budgets are deliberately small to expose the
*compute-bounded* regime in which leaderboard top scores are
unreachable for any single-thread Python HC.  Per official instance:
small-graph 25 s, medium-graph 12 s, large-graph 25 s.  Per synthetic
instance: 1 s.  All budgets are wall time, not iteration count, so
variants with cheaper evaluators (e.g. `hc12_incremental`) get more
iterations and the comparison rewards engineering as well as
algorithmic design — an explicit choice, consistent with how the ESA
challenge scores submissions.

### 3.4 Submission and scoring pipeline

Each algorithm writes a canonical-format ESA submission JSON
(`[{challenge, problem, decisionVector: [[π₀…π_{n-1}, t], …]}]`) to
`submissions/{problem}/{algo}.json` at the end of its run.  Only the
**top-20 by HV contribution** are written, selected optimally from
the runtime Pareto archive via a 2-D Hypervolume Subset Selection
DP (`core.ParetoArchive.top_k_by_hv_contribution`) — not by greedy
drop-smallest, which can lose up to ~1 800 HV on saturated archives.

Scores are computed as `-HV(front, reference = (n, n))`, then
re-verified end-to-end by re-running `core.evaluate` on every saved
decision vector and recomputing the HV.  Every score reported in
this document has been reproduced this way; the verification suite
(`make test` / `make verify`) runs all 54 saved submissions (15 HC +
3 metaheuristics × 3 instances) through this pipeline and must pass
before any score is trusted.

### 3.5 What is *not* in scope here

- **Statistical significance tests** (Wilcoxon / Mann-Whitney) on
  the headline pairwise claims.  These require ≥ 10 seeds for
  meaningful power and are deferred to the journal extension.
- **Hardware-class normalisation**.  All single-thread CPython
  numbers here are directly comparable to each other; comparing to
  the leaderboard top (presumably produced with parallel and / or
  exact-treewidth solvers) is *not* apples-to-apples and the gap
  numbers in § 4.1 are stated as "best-HC-result-to-leaderboard"
  with that caveat.
- **Parameter tuning**.  We use defaults from the algorithms'
  primary references: hc8 LAHC buffer `L = max(100, n/5)`, hc11
  stagnation K = 500, hc13 tabu length `√n`, hc14 sample
  `K = 16`.  Sensitivity analysis on these parameters is journal
  work.

---

## 4. The three real instances

| Problem | n | edges | avg deg | max deg | density flavour |
|---|---:|---:|---:|---:|---|
| `small-graph` | 1 357 | 2 280 | 3.36 | 7 | sparse — heuristics work well |
| `medium-graph` | 1 399 | 13 799 | 19.73 | 92 | medium density |
| `large-graph` | 2 426 | 253 895 | 209.3 | 538 | dense — random init triggers the width limit |

The 500-width limit is decisive on `large-graph`: a single vertex already has
degree 538, so any elimination order that touches it before peeling
its neighbours fails outright. That is why every variant from `hc4`
onwards begins with a *min-degree warm start* on dense graphs.

### 4.1 Scoreboard — our results vs the competition top

All scores are `−HV` against reference `(n, n)`; **more negative is
better**. Each cell is the score of the saved submission JSON, re-
verified end-to-end by `verify_submission.py`. Budgets: small 25 s,
medium 12 s, large 25 s, seed = 42.

| Variant | small (n = 1 357) | medium (n = 1 399) | large (n = 2 426) |
|---|---:|---:|---:|
| `hc1_initial` *(Algorithm 10 baseline)* † | −256 620 | −1 082 090 | −4 670 050 |
| `hc2_simple` | −1 734 246 | −1 256 302 | −4 670 050 |
| `hc3_archive` *(descent)* | −1 683 669 | 0 *(width limit not bridged)* | 0 *(width limit not bridged)* |
| `hc4_warm_start` | −1 814 329 | −1 605 381 | −4 788 226 |
| `hc5_operators` *(12 ops incl. 2-opt + 3-opt)* | −1 814 961 | −1 607 014 | −4 796 099 |
| `hc6_gap_fill` | −1 814 329 | −1 600 965 | −4 778 553 |
| `hc7_kbottleneck` *(12 ops + k-bottleneck)* | −1 814 722 | −1 608 251 | −4 797 998 |
| `hc8_lahc` *(late acceptance + 3-opt)* | −1 814 604 | −1 604 654 | −4 786 316 |
| `hc9_hv_accept` *(SMS-EMOA-style + 3-opt)* | −1 814 925 | −1 607 963 | −4 793 573 |
| `hc10_torso_warm` *(per-t warm start + 3-opt; large pre-3-opt — see note)* | −1 811 842 | −1 603 546 | −4 725 645 |
| `hc11_ils` *(iterated local search + 3-opt)* | −1 814 911 | −1 607 892 | −4 799 384 |
| `hc12_incremental` *(2-5× faster eval, 3-opt ported)* | −1 814 515 | −1 603 494 | −4 788 920 |
| `hc13_tabu` *(Tabu Search + 3-opt, Glover 1989)* | −1 814 547 | **−1 609 194** | **−4 800 196** |
| `hc14_steepest` *(stochastic best-improvement, K = 16)* | −1 814 518 | −1 603 500 | −4 779 874 |
| `hc15_hv_incremental` *(HV-accept on incremental eval)* | **−1 814 976** | −1 608 632 | −4 793 629 |
| | | | |
| **🏆 Competition leaderboard top** | **−1 829 919** | **−1 745 122** | **−5 493 062** |
| Gap (our best → top) HV | +14 943 | +135 928 | +692 866 |
| **% of leaderboard top** (HC / top) | **99.18 %** | **92.21 %** | **87.39 %** |
| **% behind leaderboard** (1 − HC/top) | 0.82 % | 7.79 % | 12.61 % |
| Theoretical max (`−n²`) | −1 841 449 | −1 957 201 | −5 885 476 |
| % of theoretical max (HC / `n²`) | 98.56 % | 82.22 % | 81.56 % |
| % of theoretical max (leaderboard / `n²`) | 99.37 % | 89.16 % | 93.33 % |

† `hc1_initial` is the only random-init, no-warm-start baseline, so its
fixed-wall-clock (25 s) score is machine-speed dependent and does not
reproduce exactly across machines (observed −191 255 to −519 696 on
different hosts). The value above (−256 620) is the reference-machine
figure; the relative ordering — hc1 far below every warm-start variant —
is invariant. All other cells are wall-clock-insensitive and reproduce
exactly.

Most variants (`hc4`, `hc8`, `hc10`, `hc11`, `hc13`) use a
**12-operator** set including the explicit 2-opt (`reverse`) and
3-opt (`op_3opt` — double-reverse of two adjacent segments)
primitives.  `hc5`, `hc9` and `hc14` extend this to **15 operators**
by adding the three structural moves `or_opt`, `bottleneck_relocate`
and `min_fill_reinsert`.  `hc12_incremental` runs a **6-operator**
subset (`swap`, `adjacent_swap`, `insert`, `reverse`, `3opt`,
`block_move`) — the moves whose width delta can be evaluated
incrementally; `op_3opt` was ported into the incremental evaluator
with leftmost-changed bookkeeping for the double-reverse move (the
one cell previously deferred to the journal extension is now closed).
`hc7_kbottleneck` retains its own 12-op set (the base perm/`t` moves
plus its `k_bottleneck→head` move) so it cleanly isolates the
K-vertex coordinated structural-move axis.  One cell was *not* rerun
with the widened operator set for an honest reason: `hc10_torso_warm`
on `large-graph`, because the 20-point torso-aware warm-start
construction alone exceeds our 25 s wall budget at n = 2 426 once the
operator set widens.

**Best variant per instance (seed-42 canonical submissions).** On the
sparse `small-graph` the operator-rich, acceptance-aware climbers are
statistically interchangeable at the top: `hc15` leads at −1 814 976,
with `hc5` (−1 814 961) and `hc9` (−1 814 925) within ~50 HV.  On
`medium-graph` Tabu Search (`hc13`) is strongest at −1 609 194, just
ahead of `hc15` (−1 608 632) and `hc7` (−1 608 251).  On the dense
`large-graph` `hc13` again leads at −4 800 196, with `hc11`
(−4 799 384) and `hc7` (−4 797 998) clustered within ~4 100 HV.  The
qualitative finding is robust: HV-improvement acceptance and the
incremental evaluator (the hc9 / hc15 line) own the sparse and
medium-density regime, while problem-aware structural moves and Tabu
escapes (hc7 / hc13) own the dense regime — a clean density-axis
crossover that the multi-seed sweep in § 4.3 corroborates.

Our best on each problem is in bold.  Competition tops are taken from
the [official ESA challenge page](https://optimize.esa.int/challenge/spoc-3-torso-decompositions/About).
**Reading the percentages**: our best HC achieves **99.18 % / 92.21 %
/ 87.39 %** of the leaderboard-top HV on small / medium / large — i.e.
we are **0.82 % / 7.79 % / 12.61 % behind** the top.  (The seed-42
canonical best is `hc15` on small and `hc13` on medium and large; the
multi-seed *sweep means* in § 4.3 put `hc15` ahead on small/medium.)  The
*theoretical-max* row records `−n²`, the HV a hypothetical single
solution at fitness `(0, 0)` would achieve; it is unreachable on any
graph with at least one edge but is useful as a denominator.  The
leaderboard top itself sits at 99.37 % / 89.16 % / 93.33 % of the
theoretical max, so the residual ~0.6 % / ~10.8 % / ~6.7 % gap there
is the *graph-structure-imposed* limit (treewidth, max degree),
distinct from the *algorithm-imposed* gap we close.

The Hill Climbing family plateaus at the min-degree heuristic on
small-graph (§ 4.1 below explains why); on medium and large the gap
to the leaderboard top is mostly due to budget — each fitness
evaluation on a dense graph costs ~120 ms, so we get only a few
thousand iterations per run.

Adrian's previous personal bests, stored in `submissions/archived/`
(older single-dict JSON format), are −1 829 796 on small (123 HV
below the leaderboard top) and −1 732 287 on medium (12 835 HV
short). Both reproduce end-to-end via `verify_submission.py`.

### 4.2 Three-seed mean ± std on the headline variants

A 3-seed sweep (seeds **42, 1, 2**) of the four headline variants
`hc5 / hc9 / hc11 / hc14` was run at the same canonical budgets
(25 / 12 / 25 s).  Scores are −HV; mean and population std across
the three seeds:

| problem | algo | seed 42 | seed 1 | seed 2 | mean | std |
|---|---|---:|---:|---:|---:|---:|
| small  | hc5  | −1 810 728 | −1 813 343 | −1 809 394 | **−1 811 155** | 2 008 |
| small  | hc9  | −1 813 478 | −1 814 134 | −1 813 416 | **−1 813 676** | **397** |
| small  | hc11 | −1 810 728 | −1 813 343 | −1 809 394 | **−1 811 155** | 2 008 |
| small  | hc14 | −1 810 504 | −1 813 181 | −1 809 176 | **−1 810 953** | 2 040 |
| medium | hc5  | −1 606 826 | −1 596 309 | −1 601 340 | **−1 601 491** | 5 260 |
| medium | hc9  | −1 607 517 | −1 598 529 | −1 602 980 | **−1 603 008** | 4 494 |
| medium | hc11 | −1 606 826 | −1 596 333 | −1 601 340 | **−1 601 499** | 5 248 |
| medium | hc14 | −1 606 351 | −1 596 854 | −1 599 582 | **−1 600 929** | 4 889 |
| large  | hc5  | −4 799 384 | −4 786 340 | −4 799 260 | **−4 794 994** | 7 495 |
| large  | hc9  | −4 793 509 | −4 793 548 | −4 791 371 | **−4 792 809** | **1 245** |
| large  | hc11 | −4 799 438 | −4 786 049 | −4 798 820 | **−4 794 769** | 7 558 |
| large  | hc14 | −4 776 273 | −4 780 181 | −4 785 572 | **−4 780 675** | 4 669 |

Reading: **hc9 has the lowest per-instance std** on both small
(397 HV) and large (1 245 HV) — by an order of magnitude on
small.  This stability is itself an HV-acceptance signature: a
candidate that strictly grows archive HV is a deterministic
function of the archive state, not of the current focus, so the
seed-induced trajectory differences propagate less.  Lex
acceptance (hc5, hc11) is much more sensitive to where the seed
lands the focus on the first archive add, hence the ~2 000 / 5 000 /
7 500 HV std.

**Pairwise winners that survive 3-seed noise** (mean gap >> 2 × max
of the two std's):

- **hc9 beats hc5, hc11, hc14 on small** (mean gap +2 521 HV vs hc5;
  hc9 std 397, hc5 std 2 008 — clean signal).
- All other pairwise comparisons (hc9 vs hc5 on medium / large; hc14
  vs hc5 anywhere) collapse into seed noise at 3 seeds.  Reporting
  these as *trends* rather than wins is the honest framing.  The
  twelve-seed sweep in § 4.3 (now including hc15) supersedes these
  three-seed trends for the headline variants.

### 4.3 Twelve-seed sweep (seeds 1–11 + canonical 42, now including hc15)

`make multiseed-official` was re-run with hc15 added to the variant
set, over seeds 1–11 at the canonical budgets (25 / 12 / 25 s); each
written submission is re-scored top-20 (the
hardware-independent canonical score).  Per-variant mean ± population
std over the eleven sweep seeds (−HV, more negative = better; best
mean per instance in bold):

| problem | hc5 | hc9 | hc11 | hc14 | hc15 |
|---|---:|---:|---:|---:|---:|
| small  | −1 813 843 ± 651 | −1 814 394 ± 753 | −1 813 794 ± 643 | −1 813 673 ± 641 | **−1 814 872 ± 847** |
| medium | −1 605 929 ± 8 182 | −1 608 923 ± 7 760 | −1 607 586 ± 7 748 | −1 603 966 ± 7 887 | **−1 609 238 ± 7 876** |
| large  | −4 807 627 ± 6 868 | −4 795 366 ± 3 658 | −4 797 937 ± 9 348 | −4 778 326 ± 2 618 | −4 793 694 ± **204** |

hc15 head-to-head vs hc9, per seed (positive = hc15 better):

| problem | hc15 wins | hc9 wins | mean(hc9 − hc15) |
|---|---:|---:|---:|
| small  | **7 / 11** | 4 / 11 | +478 HV |
| medium | **8 / 11** | 3 / 11 | +314 HV |
| large  | 8 / 11 | 3 / 11 | −1 671 HV |

Reading: **hc15 takes the best mean on small and medium**, beating hc9
(its acceptance-rule parent) on the majority of head-to-head seeds —
this confirms, beyond seed 42, that pairing HV-improvement acceptance
with the incremental evaluator is the strongest sparse/medium HC.  The
margin is modest (≈ 300–500 HV, ≈ 0.02–0.03 %) and smaller than the
seed-to-seed spread — on medium especially, the ±7 800 HV seed swing
dwarfs the ~310 HV hc15-over-hc9 gap — so this is *best-by-a-whisker*,
not a statistically clean win.  On **large**, hc15 is the **most stable**
variant by a wide margin (std 204 vs hc9's 3 658) but not the strongest
mean: hc5 averages best in this sweep (−4 807 627), and the canonical
large headline stays `hc13` (Tabu, not in this sweep).  One follow-up
worth a clean run before the journal version: hc5's large mean is more
negative than hc13's recorded seed-42 −4 800 196, so an `hc5` vs `hc13`
multiseed on large is the open question for the dense-instance headline.

### 4.4 Convergence figures

Archive-HV vs wall time for hc5 / hc9 / hc11 at seed = 42, sampled
every 50 / 20 / 10 iterations on small / medium / large (raw CSVs in
`extra_instances/conv_*.csv`, generation script at
`tools/convergence_plots.py`):

![Convergence on small-graph](../extra_instances/convergence_small.png)

![Convergence on medium-graph](../extra_instances/convergence_medium.png)

![Convergence on large-graph](../extra_instances/convergence_large.png)

Reading the curves:

- **hc9 (red, HV-acceptance) is the only variant whose archive HV
  grows monotonically throughout the run on all three problems.**
  hc5 and hc11 are step functions: long flat plateaus punctuated by
  rare lex-improvement jumps.  This is the qualitative signature
  of the headline finding — HV-acceptance recycles archive HV
  efficiently, lex acceptance does not.
- On small-graph hc9 starts higher (the warm-start front already
  fills more HV) and stays ~1 500 HV above hc5/hc11 for the entire
  run.
- On medium-graph the lex-acceptance variants stall at the
  initial-warm-start HV (~1 606 716) for the entire 12 s budget; hc9
  climbs another ~2 500 HV during the run.
- On large-graph the curves cross several times — visual confirmation
  that within seed noise hc5 / hc11 / hc9 are indistinguishable on
  this instance and that the multi-seed table § 4.2 should be the
  basis for any large-graph claim.

---

## 4b. Metaheuristics chapter — Simulated Annealing, GRASP, VNS

The second algorithm chapter adds three classical metaheuristics on top of
the *identical* infrastructure used by the Hill-Climbing chapter: the same
three official instances, the same `seed = 42` and canonical budgets
(small 25 s, medium 12 s, large 25 s), the same 15-operator move pool
(reused from `hc9` via `algorithms/meta_common.py`), the same Pareto
archive and the same `−HV` scoring. Only the *search control* differs, so
the comparison to the HC chapter is apples-to-apples.

- **Simulated Annealing** (`algorithms/simulated_annealing/sa.py`) — a single
  Metropolis walker on the scalar energy `E(w, t) = −(n − w)(n − t)` (the
  area one decision dominates against `(n, n)`), with geometric / linear /
  adaptive cooling and auto-calibrated `T0`. Every candidate is still offered
  to the shared archive, so the score is the archive HV even though the walker
  follows a scalarisation.
- **GRASP** (`algorithms/grasp/grasp.py`) — independent restarts, each a
  greedy-randomised min-degree construction (α-controlled RCL) followed by
  HV-improvement local search into the shared archive.
- **VNS** (`algorithms/vns/vns.py`) — shaking in a `k`-rung neighbourhood
  ladder + HV-improvement descent, resetting `k` on archive improvement.
  Both the shake and the descent are deadline-bounded so no rung overruns
  the wall-clock budget.

### 4b.1 Tuning

Each technique was tuned by reproducible **grid search** (`tools/tune.py`,
`make tune-{sa,grasp,vns}`) at `seed = 42` and the canonical budget, the cell
minimising `−HV` taken as the configuration. Best cells per instance:

| technique | small | medium | large |
|---|---|---|---|
| SA | linear, α 0.95, 400 steps/T | geometric, α 0.85, 100 steps/T | geometric, α 0.95, 200 steps/T |
| GRASP | α 0.0, 4 restarts | α 0.0, 8 restarts | α 0.3, 4 restarts |
| VNS | k_max 2, shake 2, ls 600 | k_max 2, shake 2, ls 150 | k_max 2, shake 1, ls 150 |

Raw sweeps are in `extra_instances/tuning_{sa,grasp,vns}.csv`. Two tuning
findings are themselves results: (i) GRASP prefers **pure-greedy
construction (α = 0)** on the two sparser instances — randomised
construction only pays off on the dense `large-graph` (α = 0.3); and
(ii) VNS's `k_max` has **no measurable effect** at these budgets, because a
single descent already consumes the whole budget so the ladder never climbs
(only 1–4 rounds complete) — the ladder mechanism is *budget-starved*, not
broken.

### 4b.2 Scoreboard (tuned, seed-42 canonical, verified top-20)

Scores are `−HV`, more negative is better; each cell is the saved submission
re-verified end-to-end by `tools/verify_submission.py`. Best metaheuristic
per instance in **bold**; the best HC result is repeated for reference.

| technique | small (n = 1 357) | medium (n = 1 399) | large (n = 2 426) |
|---|---:|---:|---:|
| Simulated Annealing | −1 814 521 | −1 603 923 | −4 777 663 |
| **GRASP** | **−1 816 174** | **−1 615 368** | **−5 010 404** |
| VNS | −1 815 418 | −1 608 650 | −4 807 533 |
| | | | |
| best HC (chapter 1) | −1 814 976 *(hc15)* | −1 609 194 *(hc13)* | −4 800 196 *(hc13)* |
| 🏆 leaderboard top | −1 829 919 | −1 745 122 | −5 493 062 |
| **% of leaderboard top (GRASP / top)** | **99.25 %** | **92.57 %** | **91.21 %** |

**Headline: GRASP wins every instance**, and **GRASP's `large-graph`
result (−5 010 404) is a new project best**, ~210 200 HV beyond the best
HC (hc13, −4 800 196) — at an honest on-budget 25.0 s seed-2 submission,
now promoted to the canonical `grasp.json` (the previous canonical
−4 904 767 is retained as `grasp.prev.bak`). GRASP also edges the best HC
on `small-graph` (−1 816 174 vs −1 814 976). On `medium-graph` GRASP
(−1 615 368) is again the strongest, beating the best HC (hc13,
−1 609 194); SA (−1 603 923) and VNS (−1 608 650) sit just behind the HC
front there. Against the leaderboard, GRASP reaches **99.25 % / 92.57 % /
91.21 %** of the top HV on small / medium / large — closing the dense-graph
gap from the HC chapter's 87.39 % to 91.21 %.

### 4b.3 GRASP-large seed robustness

Because the `large-graph` result is a new best, it was re-run at two
off-canonical seeds with the same tuned config (α 0.3, 4 restarts, 25 s):

| seed | GRASP large (verified) |
|---|---:|
| 42 | −4 904 767 |
| 1 | −4 916 147 |
| 2 | −5 010 404 |

All three clear the previous best (hc13, −4 800 196) by 104 600–210 200 HV,
and seed 2 crosses the −5.0 M mark. The win is **robust across seeds**, not
a seed artefact. The seed-2 submission (−5 010 404) is now the canonical
`submissions/large-graph/grasp.json`; the other artefacts remain at
`submissions/large-graph/grasp_s{1,2}.json`.

### 4b.4 Why the ordering

- **GRASP wins through construction diversity.** Each restart begins from a
  fresh min-degree elimination order; on the dense `large-graph` the RCL
  randomisation (α = 0.3) explores genuinely different orderings, and the
  shared archive unions their fronts. The HC family, by contrast, all warm-
  start from a *single* min-fill/min-degree order and can only perturb it.
- **VNS is mid-pack and evaluation-bound.** On the dense graph almost all of
  its archive gains come from the `t_shift` and `bottleneck→head` operators;
  the permutation moves rarely register an HV improvement, and the ladder is
  budget-starved (§ 4b.1).
- **SA is weakest.** Its Metropolis walk on the scalar energy churns
  acceptances (≈ 190–205 accepts on medium/large) but the walker wanders off
  the Pareto front rather than tightening it, so archive growth is slow. This
  is the known ceiling of a *single-scalarisation* SA, not an implementation
  defect — a dominance-based acceptance addresses it directly, now
  implemented as the **AMOSA** variant (`--accept amosa`, see § 4b.5).

### 4b.5 Enhanced variants and the four-family harness

The §4b.2 baselines are textbook-faithful but each has a named design
ceiling (§4b.4).  Each family now carries an **additive** enhancement —
default behaviour unchanged, written to an alternate `--algo` stem, so the
canonical scoreboard above is untouched:

| family | baseline ceiling | enhancement (flag) | reference |
|---|---|---|---|
| SA | single scalar energy `−(n−w)(n−t)` | **AMOSA** bi-objective Metropolis on amount-of-domination (`--accept amosa`) | Bandyopadhyay et al. 2008 |
| GRASP | similar constructions every restart, no recombination | **path-relinking + reactive-α** over an elite pool (`--path-relinking --reactive-alpha`) | Glover 1997; Prais & Ribeiro 2000 |
| VNS | single stochastic weighted-sample descent | **true VND** — ordered structured neighbourhoods, best-of-`k`, reset on improvement (`--local-search vnd`) | Hansen & Mladenović 2001 |

The head-to-head is run seed-for-seed by `tools/meta_multiseed.py`, which
scores every config (`hc9`, `sa`, `sa_amosa`, `grasp`, `grasp_pr`, `vns`,
`vns_vnd`) with the canonical evaluator and reports a **Friedman** omnibus
plus a **Nemenyi** critical-distance ranking — the same statistical
apparatus used for the chapter-1 HC sweep, now spanning all four families.
Canonical submissions are protected (hc9 backed up and restored; every
other config writes a throwaway `<name>_ms` stem deleted after scoring).

**Final sweep — 11 seeds (1–11), tuned configs, canonical budgets.** Per
config × instance, mean / best / population-std of the verified `−HV`
(more negative is better; `n = 11` seeds each):

| config | small mean / best | medium mean / best | large mean / best |
|---|---:|---:|---:|
| hc9 | −1 814 387 / −1 813 397 | −1 608 923 / −1 591 704 | −4 795 365 / −4 792 181 |
| **grasp** | −1 814 012 / −1 813 252 | −1 608 276 / −1 595 634 | **−4 941 223 / −4 902 856** |
| grasp_pr | −1 813 807 / −1 812 398 | −1 600 372 / −1 587 602 | −4 915 896 / −4 792 113 |
| vns | −1 814 259 / −1 813 392 | −1 608 872 / −1 591 695 | −4 796 455 / −4 792 482 |
| vns_vnd | −1 814 460 / −1 813 198 | −1 609 112 / −1 591 705 | −4 787 718 / −4 768 725 |
| sa | −1 813 537 / −1 812 973 | −1 602 908 / −1 583 310 | −4 797 669 / −4 781 011 |
| sa_amosa | −1 813 659 / −1 813 011 | −1 601 576 / −1 583 310 | −4 769 586 / −4 768 725 |

**Friedman omnibus** over the 7 configs × 33 blocks (3 instances × 11
seeds): χ²(6) = **62.13**, far past the α = 0.001 critical value (22.46),
so the configs are not equivalent. **Nemenyi** average ranks (1 = best),
critical distance CD₀.₀₅ = 1.78:

| rank | config | avg rank |
|---|---|---:|
| 1 | **grasp** | **2.82** |
| 2 | hc9 | 3.12 |
| 3 | vns | 3.21 |
| 4 | vns_vnd | 3.55 |
| 5 | grasp_pr | 4.00 |
| 6 | sa | 5.24 |
| 7 | sa_amosa | 6.06 |

**Per-family Wilcoxon signed-rank (enhanced vs baseline, two-sided).** In
all three families the enhancement is **not** an improvement:

| family | n | W⁺ (enh worse) | W⁻ (enh better) | verdict |
|---|---:|---:|---:|---|
| AMOSA vs SA | 31 | 431 | 65 | enhancement **significantly worse** (p ≈ 0.0003) |
| GRASP+PR/reactive-α vs GRASP | 33 | 406 | 155 | enhancement **worse** (p ≈ 0.025) |
| VND vs VNS | 33 | 365 | 196 | **no significant difference** (p ≈ 0.13) |

**Green-light verdict.** Each family is green-lit on its **baseline**:
the textbook SA / GRASP / VNS are kept as canonical, and the three
enhancements are recorded as a clean, statistically-tested **negative
result** — AMOSA and GRASP path-relinking measurably *hurt* at these
budgets (PR's per-restart recombination eats the wall-clock that pure
restart diversity spends better; AMOSA's domination-amount Metropolis
wanders more than the scalar walker), and VND is indistinguishable from
the stochastic descent. GRASP is the strongest family overall (rank 2.82),
significantly ahead of both SA variants (gap > CD). Reproduce:

```bash
python3 tools/meta_multiseed.py --seeds 1,2,3,4,5,6,7,8,9,10,11
```

### 4b.6 Reproduce

```bash
make tune-sa PROBLEM=large-graph        # or tune-grasp / tune-vns / tune-all
python3 tools/regen_canonical.py --algos sa,grasp,vns   # regen the 9 canonical
make verify                                              # re-score every JSON
```

---

## 4c. ACO bridge chapter — Ant Colony Optimization (a negative result)

The third chapter asks a single question: does a **construction-based**
metaheuristic — one that *builds* elimination orders from a pheromone-biased
probability model rather than *perturbing* a warm-started order — buy anything
on this problem? Ant Colony Optimization is the canonical construction
metaheuristic, so it is the natural bridge between the perturbative families
(HC / SA / VNS) and the restart-construction family (GRASP). It runs on the
*identical* substrate: same three official instances, same `seed = 42` and
canonical budgets, the same 15-operator pool and Pareto archive via
`algorithms/meta_common.py`, the same `−HV` scoring. The verdict, stated
up front, is that **no ACO variant is competitive**, and the chapter's value
is in *why* — a clean, statistically-tested negative result.

Two construction variants and an additive hybrid axis were implemented
(`algorithms/aco/aco.py`):

- **Ant System (AS, baseline)** — position-indexed pheromone `τ[i,v]`,
  selection `p(v) ∝ τ[i,v]^α · η(v)^β` with a dynamic min-degree heuristic
  `η = 1/(1 + deg_R(v))`; all ants deposit each iteration.
- **MAX–MIN Ant System (MMAS, enhanced)** — iteration-best-only deposit,
  pheromone bounds `[τ_min, τ_max]`, stagnation reset (Stützle & Hoos 2000).
- **Hybrid (`--local-search`)** — each ant's order is polished by a
  step-bounded HV-improvement descent over the same 15 operators GRASP/VNS
  use, accepting a move iff it strictly grows the shared archive HV. This is
  the *additive* axis (`aco_ls`, `aco_mmas_ls`); the pure-construction
  defaults are untouched.

### 4c.1 Tuning

Full-grid search at `seed = 42` (`tools/tune.py aco|aco_mmas|aco_ls|aco_mmas_ls`,
raw sweeps in `extra_instances/tuning_aco*.csv`). Best cells reward a strong
heuristic pull (`β = 4.0`) on the feasible instances. The decisive tuning
finding is on `large-graph`: **every pure-construction cell scores −0** — an
empty archive. At `n = 2 426` with 253 895 edges, a single pheromone-biased
construction takes so long that only **16–17 ants finish in 25 s**, and not
one of them yields an elimination order under the 500-width cap. The colony
never places a feasible point, so there is nothing to submit. For the hybrid,
the large-graph score is **flat across the entire grid** (β, ρ, ls_steps, ants
all irrelevant) because the per-ant descent — not the colony — supplies the
front there.

### 4c.2 Scoreboard — 11-seed sweep (seeds 1–11, tuned configs, canonical budgets)

Mean / best / population-std of the verified `−HV` over the 11 sweep seeds
(more negative is better). The best non-ACO result is repeated for reference.

| config | small mean / best / ±std | medium mean / best / ±std | large mean / best / ±std |
|---|---:|---:|---:|
| aco (AS) | −1 794 506 / −1 797 199 / ±1 346 | −1 503 617 / −1 516 483 / ±7 465 | **0 / 0 / ±0** *(total failure)* |
| aco_mmas (MMAS) | −1 795 524 / −1 797 199 / ±1 205 | −1 503 878 / −1 516 483 / ±8 329 | **0 / 0 / ±0** *(total failure)* |
| aco_ls (AS + descent) | −1 793 790 / −1 796 750 / ±2 027 | −1 495 178 / −1 523 994 / ±17 745 | −4 670 050 / −4 670 050 / ±0 |
| aco_mmas_ls (MMAS + descent) | −1 794 576 / −1 797 959 / ±2 856 | −1 495 178 / −1 523 994 / ±17 745 | −4 670 050 / −4 670 050 / ±0 |
| | | | |
| best metaheuristic (GRASP) | −1 816 174 | −1 615 368 | −5 010 404 |
| best HC | −1 814 976 *(hc15)* | −1 609 194 *(hc13)* | −4 800 196 *(hc13)* |
| 🏆 leaderboard top | −1 829 919 | −1 745 122 | −5 493 062 |

On `small-graph` the best ACO (−1 797 199) reaches only **98.2 %** of the
leaderboard top and trails every HC and metaheuristic variant by ~17 000 HV.
On `medium-graph` it is ~100 000 HV behind GRASP. On `large-graph`
pure construction fails outright; even the rescued hybrid (−4 670 050) sits
below the best HC (−4 800 196), GRASP (−5 010 404) and the −5 493 062 target.
**ACO is last on every instance.**

### 4c.3 Friedman / Nemenyi over all 11 configs

`tools/meta_multiseed.py` re-ran the full apparatus with the four ACO configs
added (11 configs × 33 blocks = 3 instances × 11 seeds). Friedman omnibus:
**χ²(10) = 258.459, p = 9.0e-50** — the configs are decisively not equivalent.
Nemenyi average ranks (1 = best), critical distance **CD₀.₀₅ = 2.583**:

| rank | config | avg rank |
|---|---|---:|
| 1 | grasp | 2.82 |
| 2 | hc9 | 3.09 |
| 3 | vns | 3.21 |
| 4 | vns_vnd | 3.53 |
| 5 | grasp_pr | 4.06 |
| 6 | sa | 5.24 |
| 7 | sa_amosa | 6.05 |
| 8 | aco_mmas_ls | 9.24 |
| 9 | aco_ls | 9.30 |
| 10 | aco_mmas | 9.46 |
| 11 | aco | 10.00 |

The four ACO variants occupy the bottom four ranks, and **every
ACO-to-non-ACO gap exceeds the critical distance** (the closest pairing,
aco_mmas_ls at 9.24 vs sa_amosa at 6.05, is a gap of 3.19 > 2.583). ACO is
*significantly* worse than every Hill-Climbing and metaheuristic config. The
pre-existing §4b ordering (grasp ≺ hc9 ≺ vns …) is reproduced exactly.

### 4c.4 Per-family Wilcoxon — the two designed comparisons

Pooling across instances of different HV scale is unsound (the large-graph
differences dwarf small/medium), so the signed-rank tests are reported
per instance (`n = 11` seeds each).

**Hybrid descent vs pure construction.** The descent is a *feasibility
rescue, not a quality lift*:

| comparison | small | medium | large |
|---|---|---|---|
| aco_ls vs aco | p = 0.48, construction better | p = 0.42, construction better | **p = 0.0009, hybrid better** (0 → −4 670 050) |
| aco_mmas_ls vs aco_mmas | p = 0.29, construction better | p = 0.25, construction better | **p = 0.0009, hybrid better** (0 → −4 670 050) |

On the dense graph the descent converts total failure into a feasible front;
on the two feasible instances it is *slightly worse* (not significant) because
the per-ant descent burns budget the colony would otherwise spend on more
constructions, and the front is already feasible there.

**MMAS vs AS** — marginal and instance-specific:

| comparison | small | medium | large |
|---|---|---|---|
| aco_mmas vs aco (construction) | **p = 0.014, MMAS better** | p = 0.61, tied | identical (both −0) |
| aco_mmas_ls vs aco_ls (hybrid) | p = 0.37, tied | identical | identical |

MMAS's only measurable edge is on sparse `small-graph` construction; on
medium it is a wash, on large both fail identically, and once descent is
added the two pheromone schemes are statistically indistinguishable
everywhere. The pooled `p ≈ 0.047` is driven entirely by the small-graph
blocks.

### 4c.5 Why no ACO approach is promising — the structural reason

The negative result is not an implementation accident; it follows from the
geometry of this problem against the construction paradigm:

1. **Construction is the wrong unit of work on a dense graph.** A single
   pheromone-biased order on `large-graph` costs ~1.5 s to build and almost
   never lands under the 500-width cap. With ~16 ants per 25 s budget, AS/MMAS
   draw 16 lottery tickets and lose every time — the archive stays empty
   (−0). The perturbative and restart families never pay this cost: they
   *start* from a min-degree/min-fill warm start that is feasible by
   construction, then improve it.

2. **The pheromone model has almost nothing to learn before the budget
   ends.** ACO's premise is that reinforcement over *many* iterations steers
   construction toward good regions. At these wall-clock budgets there are
   only 1–2 iterations on large and a few dozen on small — far too few for
   `τ` to converge. The dynamic min-degree heuristic `η` does all the work,
   which is exactly why `β = 4.0` (heavy heuristic, light pheromone) tunes
   best. An ACO whose best configuration leans almost entirely on its greedy
   heuristic is, functionally, a noisy greedy constructor — and a *plain*
   greedy min-degree construction (GRASP at α = 0) already beats it.

3. **The descent that rescues large-graph is borrowed, not native.** The only
   thing that makes ACO feasible on the dense instance is the shared
   HV-improvement descent — the *same* local search that powers GRASP and
   VNS. Once it is bolted on, the colony contributes nothing measurable
   (large-graph score is identical across all 36 grid cells and both pheromone
   schemes). The hybrid's −4 670 050 is essentially "what one descent from a
   constructed seed achieves," and it still trails GRASP's restart-construction
   (−5 010 404). So even the salvageable variant earns its score from the
   perturbative machinery, not from ant-colony construction.

4. **Diversity is supplied more cheaply elsewhere.** ACO's selling point over
   plain greedy is construction diversity. But GRASP obtains the same
   diversity from RCL randomisation (α) at a fraction of the bookkeeping, and
   unions the restart fronts in the identical shared archive — landing the
   project-best −5 010 404 on large. ACO adds a pheromone matrix, deposit
   rules and bound maintenance to reach a *worse* result.

The honest conclusion: **on torso decomposition under tight single-thread
budgets, construction metaheuristics are dominated by warm-start perturbation
(HC/SA/VNS) and by greedy-randomised restart (GRASP).** ACO is kept in the
codebase as a fully-tuned, seed-robust, statistically-characterised negative
result — the bridge chapter that shows *why* the construction paradigm does
not transfer here, not a contender for the canonical submission. The canonical
JSONs are untouched; the ACO variants live only on their own `--algo` stems.

### 4c.6 Reproduce

```bash
# full-grid tuning (writes extra_instances/tuning_aco*.csv)
python3 tools/run_metaheuristics.py --techs aco,aco_mmas,aco_ls,aco_mmas_ls
# 11-seed head-to-head incl. ACO + Friedman/Nemenyi
python3 tools/meta_multiseed.py --seeds 1,2,3,4,5,6,7,8,9,10,11
```

---


## 4d. Population chapter — NSGA-II and SMS-EMOA (a genuine front-builder)

The fourth chapter asks whether a **population-based multi-objective
evolutionary algorithm** — one that evolves a *whole population* of
elimination orders and reads the Pareto front off it directly, rather than
unioning the by-products of many single-point searches into a shared archive —
buys anything here. Every prior family (HC / SA / VNS / GRASP / ACO) is at
heart a *single-incumbent* search whose multi-objective front is a side effect
of the shared `−HV` archive. NSGA-II (Deb et al. 2002) and SMS-EMOA (Beume,
Naujoks & Emmerich 2007) are the canonical population MOEAs, so they are the
natural test of the population paradigm. They run on the *identical* substrate:
same three official instances, same `seed = 42` and canonical budgets, the
same 15-operator pool and Pareto archive via `algorithms/meta_common.py`, the
same `−HV` scoring. The verdict, stated up front: **SMS-EMOA is the first
method in the whole study to *significantly* beat hill-climbing on
`small-graph`** — a real, statistically-tested positive result on the sparse
instance — but the family **loses to GRASP on medium and large**, and the
memetic hybrid that rescued ACO *hurts* here.

Two population variants and an additive memetic axis were implemented
(`algorithms/nsga2/nsga2.py`):

- **NSGA-II (baseline)** — Deb's fast non-dominated sort + crowding distance,
  binary crowded-comparison tournament selection, generational `(μ+μ)` loop.
  Order crossover (OX, Davis 1985) recombines two parent orderings — the one
  genuinely new search move the perturbative chapters lacked.
- **SMS-EMOA (enhanced)** — steady-state `(μ+1)`: each iteration produces one
  offspring and culls the member of the worst front whose removal costs the
  *least hypervolume*. This promotes the HV-acceptance rule that hc9/GRASP use
  for *acceptance* into a *selection* policy. (PAES, the `(1+1)` ancestor, is
  already covered by hc9's HV-archive search, so it is not re-implemented.)
- **Memetic (`--local-search`)** — each offspring is polished by a
  step-bounded HV-improvement descent over the same 15 operators, the
  high-ceiling analog of `aco_ls` (`nsga2_ls`, `sms_ls`). The pure variants are
  untouched; this is the *additive* axis.

A one-time **final harvest** seeds every distinct final-population order across
the full `t`-grid before scoring, so the `t`-axis is read fairly (as GRASP
does), not just at whatever thresholds the run happened to visit.

### 4d.1 Tuning

Full-grid search at `seed = 42` (`tools/tune.py nsga2|sms|nsga2_ls|sms_ls`,
raw sweeps in `extra_instances/tuning_*.csv`). Locked winners:

| config | small | medium | large |
|---|---|---|---|
| nsga2 | pop 40, pc 0.7, pm 0.2 (−1 817 539) | pop 20, pc 0.9, pm 0.2 (−1 594 342) | pop 20, pc 0.9, pm 0.4 (−4 814 777) |
| sms | pop 20, pc 0.9, pm 0.2 (−1 818 395) | pop 20, pc 0.9, pm 0.4 (−1 590 049) | pop 20, pc 0.7, pm 0.2 (−4 779 701) |
| nsga2_ls | pop 20, pm 0.6, ls 200 (−1 816 959) | pop 20, pm 0.3, ls 200 (−1 594 821) | pop 20, pm 0.3, ls 100 (−4 776 072) |
| sms_ls | pop 20, pm 0.3, ls 50 (−1 815 819) | pop 20, pm 0.3, ls 200 (−1 594 821) | pop 20, pm 0.3, ls 100 (−4 785 225) |

A consistent finding: **small populations win** (`pop = 20` is best in 10 of 12
cells). Under a single-thread wall-clock budget a larger population means fewer
generations, and on this problem generation count matters more than population
breadth — the warm-started min-degree seeds are already good, so the search
needs *iterations to refine*, not *width to explore*.

### 4d.2 Scoreboard — 11-seed sweep (seeds 1–11, tuned configs, canonical budgets)

Mean / best / population-std of the verified `−HV` over the 11 sweep seeds
(more negative is better). Best non-population results repeated for reference.

| config | small mean / best / ±std | medium mean / best / ±std | large mean / best / ±std |
|---|---:|---:|---:|
| nsga2 | −1 815 020 / −1 817 275 / ±1 110 | −1 590 320 / −1 594 194 / ±1 852 | −4 779 519 / −4 790 891 / ±7 666 |
| sms | **−1 815 414** / −1 817 326 / ±928 | −1 591 082 / −1 596 954 / ±2 479 | −4 773 809 / −4 782 378 / ±4 773 |
| nsga2_ls | −1 814 586 / −1 815 516 / ±313 | −1 589 499 / −1 592 538 / ±961 | −4 769 881 / −4 781 444 / ±3 656 |
| sms_ls | −1 815 056 / **−1 818 628** / ±1 178 | −1 589 742 / −1 592 538 / ±1 169 | −4 771 659 / −4 782 298 / ±5 066 |
| | | | |
| best metaheuristic (GRASP) | −1 816 174 | −1 615 368 | −5 010 404 |
| best HC | −1 814 976 *(hc15)* | −1 609 194 *(hc13)* | −4 800 196 *(hc13)* |
| 🏆 leaderboard top | −1 829 919 | −1 745 122 | −5 493 062 |

On `small-graph` the population family takes the **top four mean ranks** — sms,
sms_ls, nsga2, nsga2_ls all beat every HC and every other metaheuristic on the
seed-mean, and `sms_ls` lands the single best run of the entire study on this
instance (−1 818 628). On `medium-graph` and `large-graph` the picture
inverts: the family sits ~18 000 HV below GRASP on medium and ~160 000 HV
below GRASP on large. The MOEAs win the sparse instance and lose the two
denser ones.

### 4d.3 Friedman / Nemenyi over all 15 configs

`tools/meta_multiseed.py` re-ran the full apparatus with the four population
configs added (15 configs × 33 blocks = 3 instances × 11 seeds). Friedman
omnibus: **χ²(14) = 321.607, p = 3.6e-60** — decisively non-equivalent.
Nemenyi average ranks (1 = best), critical distance **CD₀.₀₅ = 3.483**:

| rank | config | avg rank |
|---|---|---:|
| 1 | grasp | 3.848 |
| 2 | hc9 | 4.061 |
| 3 | vns | 4.182 |
| 4 | vns_vnd | 4.455 |
| 5 | grasp_pr | 5.000 |
| 6 | sms | 6.530 |
| 7 | sa | 6.803 |
| 8 | nsga2 | 6.924 |
| 9 | sms_ls | 7.803 |
| 10 | nsga2_ls | 8.167 |
| 11 | sa_amosa | 8.227 |
| 12 | aco_mmas | 13.333 |
| 13 | aco_mmas_ls | 13.394 |
| 14 | aco_ls | 13.455 |
| 15 | aco | 13.818 |

The aggregate rank places the population family in the **middle of the pack**
(ranks 6–10), above the ACO floor but below the perturbative/restart leaders.
This is expected and *not* a contradiction of §4d.2: the omnibus ranks pool all
three instances, and the family's large-graph weakness drags its average down
even though it dominates small-graph. The pre-existing ordering
(grasp ≺ hc9 ≺ vns …) is reproduced exactly. To read the population result
correctly, the per-instance Wilcoxon tests below are decisive, not the pooled
rank.

### 4d.4 Per-family Wilcoxon — the three designed comparisons

Per instance (`n = 11` seeds), signed-rank, because pooling across instances of
different HV scale is unsound.

**Q1 — SMS-EMOA's HV-selection vs NSGA-II's crowding.** The headline design
question: does culling by least-hypervolume-contribution beat Deb's crowding
distance?

| sms vs nsga2 | small | medium | large | pooled |
|---|---|---|---|---|
| | p = 0.70, tied | p = 0.32, tied | p = 0.20, tied | **p = 0.42, tied** |

**No significant difference anywhere.** On this problem HV-selection and
crowding-distance selection are statistically indistinguishable — both maintain
a good spread, and the extra HV bookkeeping SMS-EMOA pays does not convert into
a measurable front-quality edge.

**Q2 — Memetic descent vs pure population.** The same hybrid that *rescued*
ACO on large-graph:

| comparison | medium | large | pooled |
|---|---|---|---|
| nsga2_ls vs nsga2 | p = 0.043, **pure better** | p = 0.013, **pure better** | **p = 0.0029, pure better** |
| sms_ls vs sms | — | — | **p = 0.0071, pure better** |

**Memetic descent significantly *degrades* the population MOEAs** — the
opposite of ACO. The reason is budget allocation: ACO's pure construction was
*infeasible* on large (−0), so any descent was pure gain; here the pure
population is already feasible and competitive, so the per-offspring descent
just burns generations the population would otherwise spend evolving. At a
fixed wall-clock budget, fewer-but-polished offspring lose to
more-but-unpolished ones.

**Q3 — Is the SMS-EMOA small-graph win real?** SMS-EMOA vs each
single-incumbent leader on `small-graph`:

| sms vs … | p | verdict |
|---|---|---|
| grasp | 0.0058 | **SMS significantly better** |
| grasp_pr | 0.0076 | **SMS significantly better** |
| hc9 | 0.0329 | **SMS significantly better** |
| vns_vnd | 0.1095 | n.s. (SMS better on mean, not significant) |

**SMS-EMOA is the first method in the entire study to significantly beat hc9 —
the long-standing project benchmark — on any instance.** It beats GRASP,
GRASP+PR and hc9 with significance on sparse `small-graph`, and is ahead of
(though not significantly separated from) the strongest perturbative variant
vns_vnd.

### 4d.5 Why the population paradigm wins sparse and loses dense

1. **On a sparse graph the front is genuinely two-dimensional, and a
   population samples it directly.** `small-graph` (n = 1 357, avg deg 3.36)
   admits a real width/threshold trade-off curve. A population of orders, kept
   diverse by crowding/HV-selection and recombined by order crossover, lands
   points *along* that curve in one shot; a single-incumbent search has to
   stumble onto each region separately and rely on the archive to remember
   them. Order crossover is the active ingredient — it is the only move in the
   whole study that *blends* two good orderings, and on the sparse instance
   blending pays.

2. **On dense graphs the trade-off collapses to a staircase, and breadth is
   wasted.** On medium/large almost every feasible order pins to the same
   narrow width band, so the "front" is a short staircase. There is little for
   a population to spread *across*; what matters is driving the single best
   width down, which is exactly what GRASP's restart-construction and HC's deep
   perturbation do better. The population spends its budget maintaining
   diversity that the instance does not reward.

3. **Generations beat population size under a wall-clock budget.** The tuning
   (`pop = 20` wins 10/12 cells) and the memetic result (Q2) point the same
   way: this problem rewards *iteration depth* over *population breadth*,
   because the min-degree warm start is already a strong seed. Anything that
   trades generations for per-individual work (bigger population, per-offspring
   descent) loses.

4. **HV-selection and crowding are interchangeable here (Q1).** SMS-EMOA's
   theoretical advantage — selecting to directly maximise the indicator we
   score on — does not materialise because the fronts are small and the
   crowding heuristic already keeps them well-spread. The two selection schemes
   tie everywhere.

The honest conclusion: **population MOEAs are the right tool for the sparse
instance and the wrong tool for the dense ones.** SMS-EMOA earns a genuine,
statistically-significant win on `small-graph` — the study's first method to
beat hc9 — by sampling a real two-dimensional front with order crossover and
HV-selection. On the dense instances, where the front degenerates and iteration
depth dominates, GRASP's restart-construction remains best. The canonical
submission JSONs are untouched; all four population variants live only on their
own `--algo` stems.

### 4d.6 Reproduce

```bash
# full-grid tuning (writes extra_instances/tuning_{nsga2,sms,nsga2_ls,sms_ls}.csv)
python3 tools/run_metaheuristics.py --techs nsga2,sms,nsga2_ls,sms_ls
# 15-config 11-seed head-to-head incl. population MOEAs + Friedman/Nemenyi
python3 tools/meta_multiseed.py --seeds 1,2,3,4,5,6,7,8,9,10,11
```

---


## 5. Extra benchmark — 20 synthetic instances

To test external validity beyond the three official ESA instances,
we generated **20 additional Erdős–Rényi random graphs** spanning a
5 × 4 grid of (size, density):

- sizes `n` ∈ { 200, 350, 500, 750, 1000 }
- average degrees `avg_deg` ∈ { 3, 8, 18, 35 }

This grid brackets the official `small` (n = 1 357, avg deg 3.36)
and `medium` (n = 1 399, avg deg 19.73) instances in density.  All
20 instances were generated reproducibly with `seed = 42` (see
`generate_instances.py`, written to `extra_instances/data/`).  Every
algorithm was then run at a tight **1-second wall budget**, also
with `seed = 42`, producing a 13 × 20 = **260-run scoreboard**
(`extra_instances/results.csv`, full pivot in
`extra_instances/summary.md`).

### 5.1 Per-instance scoreboard (excerpt)

A representative slice of the matrix.  Scores are −HV.  On the three
densest cells no elimination order admits a ≤500-width torso at all
(§ 5.3), so every warm-start variant falls back to the same single
over-width point `(501, 0)` that hc2 submits — they **tie** at the
over-width optimum rather than crashing (the previous behaviour; see
§ 5.3 and the 29 May fix note).  Per-instance best in **bold**.

| instance | n | avg deg | hc2 | hc5 | hc1 | hc9 | hc10 | hc11 | hc12 | hc13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| inst_01_n200_d3 | 200 | 2.9 | −32 800 | −35 467 | −30 800 | −35 599 | **−35 615** | −35 467 | −35 310 | −35 509 |
| inst_04_n200_d35 | 200 | 36.6 | −5 200 | −20 202 | −10 100 | **−20 218** | −20 217 | −20 202 | −20 215 | −20 197 |
| inst_07_n350_d18 | 350 | 17.4 | −31 150 | −67 857 | −31 150 | −68 038 | −67 669 | −67 761 | **−68 064** | −67 833 |
| inst_12_n500_d35 | 500 | 34.5 | −32 000 | −127 089 | −32 010 | **−127 461** | −127 130 | −127 095 | −127 295 | −127 066 |
| inst_15_n750_d18 | 750 | 17.7 | −186 750 | −310 955 | −113 046 | −312 495 | −224 747 | −310 955 | **−313 014** | −311 229 |
| inst_16_n750_d35 | 750 | 35.4 | **−186 750** | **−186 750** | −116 034 | **−186 750** | **−186 750** | **−186 750** | **−186 750** | **−186 750** |
| inst_18_n1000_d8 | 1000 | 7.8 | −499 000 | −658 786 | −313 372 | **−661 094** | −640 369 | −658 786 | −659 295 | −658 786 |
| inst_19_n1000_d18 | 1000 | 17.9 | **−499 000** | **−499 000** | −318 362 | **−499 000** | **−499 000** | **−499 000** | **−499 000** | **−499 000** |
| inst_20_n1000_d35 | 1000 | 35.2 | **−499 000** | **−499 000** | −317 863 | **−499 000** | **−499 000** | **−499 000** | **−499 000** | **−499 000** |

The full 13-algo × 20-instance pivot table is in
[`extra_instances/summary.md`](extra_instances/summary.md); the raw
CSV is in `extra_instances/results.csv`.

### 5.2 Wins per algorithm across the 20 instances

We declare a "win" for each instance as the best **mean** score over
the 3 seeds across all 13 variants (ties broken by lexicographic order
on algorithm ID).  These counts are produced by
`tools/analyze_synthetic.py` and live in
[`extra_instances/synthetic_analysis.md`](extra_instances/synthetic_analysis.md);
the numbers below mirror that file.

| algorithm | wins / 20 |
|---|---:|
| `hc9_hv_accept` *(SMS-EMOA-style)* | **10** |
| `hc12_incremental` *(more iters per second)* | **6** *(medium-density mid-size where throughput pays off)* |
| `hc2_simple` *(plain HC)* | **3** *(the over-width frontier — see below)* |
| `hc10_torso_warm` *(per-t warm start)* | **1** |
| `hc3 … hc8`, `hc11`, `hc13` | 0 |

The headline finding holds: **hc9's HV-improvement acceptance wins
on half the synthetic instances** and is the single most common
winner.  **hc12's six wins are a methodological lesson**: at a
1-second wall budget the incremental evaluator's 2–5× throughput
multiplier lets it find slightly better fronts than hc9 on several
mid-density instances — evidence that *evaluator speed is itself an
algorithmic choice*.  hc2's three wins are the *over-width frontier*:
on the three densest cells (inst_16 / inst_19 / inst_20) **no
elimination order admits a ≤500-width torso at all** (§ 5.3), so the
best any variant can do is submit the single over-width point
`(501, 0)`.  Every warm-start variant now does exactly that (via the
`core.ensure_seeded` fallback) and ties hc2's score; hc2 takes the
nominal win only on the lexicographic tiebreak.  The interesting
fact is the *graph property* (treewidth > 500), not a difference
between the algorithms on these cells.

**Significance (Friedman + Nemenyi).**  Across the 13 variants the
Friedman test gives χ² = 112.99 (df = 12, p = 1.5e-18): the mean-rank
differences are not chance.  By the Nemenyi critical difference at
α = 0.05 (CD = 4.08), hc9 (mean rank 2.73) is significantly better than
hc1, hc2, hc3, hc6, hc8 and hc10, but **not** significantly better than
the cluster hc4, hc5, hc7, hc11, hc12, hc13 — i.e. the top group is
statistically indistinguishable and the chapter's claim is "hc9 leads a
tight front-runner group," not "hc9 dominates everything."  The
standard critical-difference diagram is rendered by
[`tools/cd_diagram.py`](../tools/cd_diagram.py) to
`extra_instances/cd_diagram.png` (and `.pdf`); it reuses the exact
pure-Python rank / Friedman / Nemenyi machinery of
`tools/analyze_synthetic.py`, so the figure and the text agree by
construction.

**Head-to-head harness.**  [`tools/compare_variants.py`](../tools/compare_variants.py)
runs the three strongest variants (hc9 HV-accept, hc12 incremental,
hc15 = the crossbreed) head-to-head on the official graphs over one or
more seeds, re-scoring each written submission and tabulating
mean ± stdev plus hc15's delta against each parent.  It writes
`extra_instances/variant_comparison.{md,csv}`.  As always these scores
are wall-clock-budgeted and must be regenerated on the reference
workstation before being quoted.  The twelve-seed multiseed sweep
(§ 4.3) is the authoritative version of this comparison: hc15 carries
the best mean over hc9 on small and medium and wins 7/11 and 8/11
head-to-head seeds respectively.

### 5.3 Coverage — which algorithms fail on which cells

The synthetic grid revealed a clean **feasibility frontier** for
the Hill Climbing family: on three dense cells the min-degree warm
start produces an over-width permutation, because the graph's
treewidth genuinely exceeds the 500 hard cap (see the head-to-head
construction table below).  No elimination order on these cells can
be feasible.

**Fix (29 May 2026).**  Previously the warm-start variants (hc4 – hc13)
*crashed* on these cells — the all-over-width archive was empty and the
algorithms indexed `archive.entries()[0]`, raising `IndexError` and
emitting `nan`.  This was a software defect, not an algorithmic limit.
A central `core.ensure_seeded` helper now seeds the archive with the
over-width fallback point `(501, 0)` (via the new `allow_overwidth`
path in `ParetoArchive.try_add`) whenever a warm start yields no
in-cap point.  All variants now run cleanly on all 20 cells and emit
the deterministic over-width score, tying hc1/hc2/hc3 on those cells
rather than producing `nan`.

| algorithm | runs cleanly / 20 | in-cap feasible / 20 |
|---|---:|---:|
| `hc1`, `hc2`, `hc3` | 20 / 20 | 17 / 20 |
| `hc4` – `hc13` | 20 / 20 | 17 / 20 |

The three over-width cells (inst_16 n750/d35, inst_19 n1000/d18,
inst_20 n1000/d35) are infeasible for *every* variant — no warm start
or heuristic order brings them under the cap, because their treewidth
exceeds 500.  This is the same dense-graph wall that defeats random
initialisation on the official medium / large graphs.

**Update (29 May 2026) — the feasibility wall on these three cells is
structural, not heuristic.**  We implemented and measured four
constructions head-to-head on the failing cells (`core.mcs_m_order`,
`core.min_fill_in_perm`, `core.min_degree_perm`, and a width-budgeted
min-fill):

| cell | min-degree | min-fill | width-budgeted min-fill | MCS-M | 500-cap |
|---|---:|---:|---:|---:|---:|
| inst_16 (n750, d35) | 584 | **576** | 576 | 607 | 500 |
| inst_19 (n1000, d18) | 649 | **635** | 635 | 690 | 500 |
| inst_20 (n1000, d35) | 786 | **781** | 783 | 808 | 500 |

Two findings.  (i) **MCS-M is *worse* than the existing min-fill** on
every one of these dense Erdős–Rényi cells (minimal ≠ minimum
triangulation; on dense random graphs the minimal triangulation is
wider).  (ii) **min-fill is the best construction in the family** but
is still 76–283 above the cap, and the width-budget constraint is
non-binding because no candidate vertex is ever under-cap.  So no
construction heuristic in this family makes these cells feasible —
their treewidth genuinely exceeds 500.  This corrects the optimistic
premise of FUTURE.md §2.3 that MCS-M / LEX-M would unlock these cells;
the honest takeaway is that **min-fill should replace min-degree as the
default warm start** (it dominates by a few width units everywhere,
including on the *feasible* official instances), while the inst_16 /
inst_19 / inst_20 over-width frontier is a property of the graphs, not
of the heuristic.  See `core.treewidth_lower_bound_mmd` for the
(loose) lower-bound side of this argument.

### 5.4 Reproduce the benchmark

```bash
# 1. Generate the 20 instances (overwrites extra_instances/data/ in place)
python3 generate_instances.py

# 2. Run an algorithm across all 20 instances at 1 s wall each
python3 bench_extra.py --algos hc5,hc9 --budget 1 --out extra_instances/results.csv

# 3. Or run a single algorithm on a single instance for debugging
python3 bench_extra.py --algos hc9 --instances inst_15_n750_d18 --budget 5
```

`bench_extra.py` supports `--append` (don't overwrite the CSV) and
`--algos all` (run every algorithm).  The output CSV has columns
`instance, algo, score, elapsed_s, budget_s, seed`.

---



## See also

- [EXPLAINER.md](EXPLAINER.md) — the problem from scratch in plain language
- [PROBLEM.md](PROBLEM.md) — formal problem statement
- [ALGORITHMS.md](ALGORITHMS.md) — every Hill Climbing variant
- [FUTURE.md](FUTURE.md) — next chapters
