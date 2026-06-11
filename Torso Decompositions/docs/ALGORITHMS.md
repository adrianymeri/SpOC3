# Algorithms — the permutation-space families

_This document catalogues the four chapters of **permutation-space** search:
the 15 hill-climbing variants and shared operator set, the Simulated
Annealing / GRASP / VNS metaheuristics chapter (§ 7), the Ant Colony
Optimisation bridge chapter (§ 8, a negative result), and the NSGA-II / SMS-EMOA
population chapter. Each is presented with its design rationale and what it
taught me about the problem._

> **Where this sits in the thesis.** These families are the work that
> established the *permutation-space ceiling* (91.6–99.4 % of the leaderboard).
> They are no longer the state of the art in this project — the
> continuous-encoding method of [THESIS.md](THESIS.md) supersedes them by
> 8,954 / 91,736 / 350,454 hypervolume — but they remain the rigorously-tuned,
> statistically-separated baseline against which that breakthrough is measured,
> and they contribute their non-dominated points to the final portfolio. The
> parameter audit ([AUDIT.md](AUDIT.md) §3) shows *why* they plateau: their
> tuned optima collapse to the simplest possible settings, the signature of a
> landscape that rewards greedy depth over every elaboration.

An 11-seed Friedman omnibus (χ²(14) = 321.607, p ≈ 3.6 × 10⁻⁶⁰) separates the
families; GRASP and HV-acceptance hill climbing lead, ACO is significantly worse
than all others (beyond the Nemenyi critical distance):

![Friedman average ranks across the permutation-space families.](figures/fig6_method_ranks.png)

---

## 2. How we solve it

The decision space is **all permutations of n vertices times all
thresholds**, which is `n! × n` — enormous. Exhaustive search is
hopeless even on the smallest official instance (n = 1 357). We
therefore use **local search**, and specifically the **Hill Climbing
family**: start from a candidate, apply a small random change, accept
if the change improves the score, repeat until time runs out.

Plain hill climbing falls into local optima — that is the whole point
of the variants. Each `hcN` file in this folder layers one
classical-literature technique on top of its predecessor to escape
those optima:

- A **Pareto archive** so the search can keep all non-dominated
  (max_degree, t) points seen, instead of optimising a single
  scalar.
- A **min-degree warm start** so the initial permutation is already
  near a good local optimum — essential on the dense graphs where a
  random start triggers the 500-width limit on the first step.
- A **rich operator set** with adaptive selection so the search has
  several neighbourhoods to escalate through.
- **Per-t local search**, **K-vertex coordinated moves**, and
  **bottleneck-aware moves** to attack the specific failure modes
  of single-step swap (the bottleneck plateau).
- **Late acceptance**, **HV-improvement acceptance**, **iterated
  local search**, and **tabu memory** as alternative escape
  mechanisms inspired by Burke & Bykov, Beume et al., Lourenço et
  al., and Glover respectively.
- **Random restarts** (the textbook Algorithm 10, `hc1`) as the
  conceptual baseline.
- An **incremental bitset evaluator** that re-walks only the
  suffix the operator actually changed, multiplying iteration
  throughput by 2–5× at zero algorithmic cost.

The full taxonomy and per-variant findings live in
[§ 6](#6-the-hill-climbing-series).  The same `core.py` is the
shared substrate for the **metaheuristics chapter** ([§ 7](#7-the-metaheuristics-chapter-sa-grasp-vns):
Simulated Annealing, GRASP, VNS — now implemented) and the planned
NSGA-II / HV-based EA chapters — the acceptance rule moves, the
evaluator, archive, and operators do not.

---


## 6. The Hill Climbing series

### 6.1 Which variant of Hill Climbing?

The HC family has many variants — first-improvement vs steepest
ascent, single-objective vs Pareto, deterministic vs stochastic.
Across `hc1` … `hc13` we use **stochastic first-improvement Hill
Climbing**.  `hc14_steepest` swaps in stochastic best-improvement
(K = 16 sampled candidates per step) so the first- vs best-
improvement axis is covered by an explicit ablation.  The shared
template otherwise is:

- *Stochastic*: at each step we pick **one** random neighbour, not
  all neighbours, and decide.
- *First-improvement* (hc1–hc13): as soon as we see a better
  neighbour we move, no further search at the current point.
- *Best-improvement* (hc14): sample K candidates, take the lex-best.
- *Deterministic acceptance*: only strictly-better neighbours are
  accepted (no probabilistic uphill moves — that is Simulated
  Annealing's job).  hc9 replaces the lex acceptance rule with HV-
  improvement acceptance.

`hc1_initial.py` is the **textbook Algorithm 10 starting point** with
random restarts.  `hc3` … `hc7` keep a **Pareto archive** alongside
the single working solution so improvements that are non-dominated
but not strictly better are still stored.

### 6.2 The variants at a glance

`hc1_initial.py` is the conceptual entry point — read it first.  Its
`Solution` class has explicit `perm`, `torso_index`, `off_torso`,
`in_torso`, `fitness` fields, nothing hidden.  The numbered series
`hc2` … `hc14` then incrementally adds capabilities on top.

| File | Approach | Operators | Initial state | Acceptance |
|---|---|---|---|---|
| `hc1_initial.py` | **Algorithm 10**: stochastic HC + random restarts. Single working solution. Submission = the final Best (1 point). | 4 (uniform): `shift_left`, `shift_right`, `shuffle`, `t_shift` | random π, random t | strict lex (max_degree, t) |
| `hc2_simple.py` | Simplest HC: fix t = 0, only minimise max_degree. Submission = 1 point. | 1: `swap` (random pairwise swap) | random π | strict improvement in max_degree |
| `hc3_archive.py` | Free t + **Pareto archive** of all non-dominated points seen. Falls back to a descent metric on over-width states so dense graphs are navigable. | 2: `swap` (90 %), `random t` (10 %) | random π, random t | archive-extending + lex on (over-width steps, max_step_d, soft) |
| `hc4_warm_start.py` | hc3 + **min-degree warm start** seeded at 20 t-values. The big win on every graph. | 2: `swap` (90 %), `random t` (10 %) | min-degree π × 20-point t-grid | archive-extending + lex (max_w, soft) |
| `hc5_operators.py` | hc4 + **15 operators** (swap, adjacent_swap, insert, reverse [= 2-opt], **3opt** [double-segment reversal], block_move, long_block_move, bottleneck→head, swap_head_tail, 2opt_torso, **or_opt** [segment-shift], **bottleneck_relocate**, **min_fill_reinsert**, t_shift, t_random) with UCB-style adaptive selection. | 15 (weighted by recent accept rate) | min-degree π × 20-point t-grid | archive-extending + lex (max_w, soft) |
| `hc6_gap_fill.py` | hc5 + **per-t local search**: round-robin through a finer 40-point t-grid, run a budget of HC iterations at each fixed t. Tries to fill the front in the middle. | 8 perm-only (no t-mutators in inner loop) | min-degree π × 40-point t-grid | archive-extending + lex (max_w, soft) |
| `hc7_kbottleneck.py` | hc5 + **K-vertex coordinated bottleneck-relocate**: when several torso vertices tie at max_degree, move K of them into the head in one move. | 12 (hc5's 11 + `k_bottleneck→head`) | min-degree π × 20-point t-grid | archive-extending + lex (max_w, soft) |
| `hc8_lahc.py` | hc5 + **Late Acceptance Hill Climbing**: keep a circular buffer of past fitness values of length L; accept candidate iff it is better than the current state OR better than the fitness from L iterations ago. Burke & Bykov 2017. | 12 (hc5's set incl. 3-opt) | min-degree π × 20-point t-grid | LAHC (lex on a sliding window of past fitness) |
| `hc9_hv_accept.py` | hc5 + **HV-improvement acceptance** (SMS-EMOA / IBEA style): replace lex acceptance with "accept iff the candidate would strictly increase archive HV". The acceptance criterion is now the same indicator we are scored on. Beume et al. 2007. | 15 (hc5's set incl. 3-opt, or_opt, bottleneck_relocate, min_fill_reinsert) | min-degree π × 20-point t-grid | strict archive-HV increase |
| `hc10_torso_warm.py` | **Per-t torso-aware warm start**: for each t in the grid, build a different π by partitioning vertices by *original degree* (high-degree → head, low-degree → torso), then min-degree-ordering each half. Bodlaender et al. 2006. | 12 (hc5's set incl. 3-opt) | one torso-aware π **per t-grid point** | archive-extending + lex (max_w, soft) |
| `hc11_ils.py` | hc5 + **Iterated Local Search**: on K consecutive iterations without an archive add, apply `strength` random long-block-move / reverse perturbations to best-so-far (no acceptance), then resume HC. Lourenço, Martin & Stützle 2003. | 12 (hc5's set incl. 3-opt) + 2 perturbation moves | min-degree π × 20-point t-grid | archive-extending + lex (max_w, soft), with perturbation kick |
| `hc12_incremental.py` | hc5 + **incremental bitset evaluator**: each operator declares the leftmost position it changed, the evaluator reuses the cached chordal-completion state for `[0, leftmost)` and re-walks only `[leftmost, n)`. t-only moves skip the walk entirely.  Currently uses the pre-3-opt operator set; 3-opt port is journal-extension work. | 7 (swap, adjacent_swap, insert, reverse, block_move, t_shift, t_random) | min-degree π × 20-point t-grid | archive-extending + strict max_degree drop |
| `hc13_tabu.py` | hc5 + **Tabu Search**: each accepted move's signature (e.g. the pair of swapped vertices) is recorded in a length-L = √n tabu list and forbidden for the next L iterations. Aspiration criterion overrides the tabu if the candidate would extend the archive. Glover 1989. | 12 (hc5's set incl. 3-opt) + tabu memory | min-degree π × 20-point t-grid | archive-extending + lex (max_w, soft), with tabu filter |
| `hc14_steepest.py` | hc5 + **stochastic best-improvement (steepest-ascent) acceptance**: at each outer step sample K = 16 candidates from the 15-operator set, pick the lex-best one, accept iff it strictly improves on the working solution. Hoos & Stützle 2004 §1.5 (first- vs best-improvement axis). | 15 (hc5's set, including 3-opt) | min-degree π × 20-point t-grid | archive-extending + lex (max_w, soft), best-of-K acceptance |
| `hc15_hv_incremental.py` | **hc9's acceptance rule on hc12's evaluator** — the crossbreed of the two strongest variants. Candidate width comes from the incremental bitset evaluator (cheap per move, hc12); the working solution moves iff adding the candidate strictly increases archive HV (SMS-EMOA / IBEA, hc9). On acceptance the working solution is committed to the incremental cache via `ev.accept(...)`. Tests whether HV-aligned acceptance and cheap evaluation compose, or whether hc9's more frequent working-solution moves defeat hc12's cache reuse. Additive: introduces no new operators and does not touch `core.py`, so it cannot perturb the hc1..hc14 canonical scoreboard. | 8 (hc12's incremental subset: swap, adjacent_swap, insert, reverse, 3opt, block_move, t_shift, t_random) | auto warm start (min-fill on small/medium, min-degree on dense) × 20-point t-grid | **HV-improvement** (SMS-EMOA / IBEA), with incremental-cache commit on accept |

Measured speed-up of `hc12` over `hc5` at matched wall-time budget
and seed 42: **5.4× on small-graph, 3.6× on medium-graph, 1.7× on
large-graph**.  Speed-up tapers on dense graphs because each partial
walk still touches a large fraction of the fill-in edges; on sparse
graphs the saved prefix is essentially free.  Final HV is within
seed noise of `hc5` on all three problems despite `hc12` carrying
one fewer operator (the `bottleneck→head` move is dropped because it
needs the bottleneck-vertex position, which would re-introduce a
full walk on every accept — recovering it incrementally is straight-
forward future work).

A few terms used above:

- **min-degree warm start** — greedy heuristic that repeatedly
  eliminates the vertex with the smallest current degree. Classical
  chordal-completion heuristic; gives a strong starting permutation
  on every instance.
- **Pareto archive** — keeps all non-dominated (max_degree, t)
  points seen during the search. The final submission is the top-20
  by hypervolume contribution (selected optimally via a small
  dynamic program in `core.py`).
- **Lex (max_w, soft)** — when two candidates have the same
  max_degree we break the tie on the smooth secondary objective
  `soft = Σ (torso-step-degree)²`. That gives the search a gradient
  to follow even when max_degree plateaus.
- **Width-limit descent** — on dense graphs a random perm has
  hundreds of elimination steps with degree > 500 (every such step
  is an **over-width step**). `hc3` uses an extra metric
  `(over-width-step-count, max_step_d, soft)` that strictly
  decreases as the perm walks toward feasibility, so it can
  navigate even when no candidate qualifies for the archive yet.

### 6.3 Operator quick reference

These are the building-block move types used across the series:

| Operator | What it does |
|---|---|
| `swap` | Swap two random positions of π. |
| `adjacent_swap`, `shift_left`, `shift_right` | Swap perm[i] with perm[i±1]. The smallest possible move. |
| `insert` | Pop perm[i], reinsert at a random position (= or-opt with segment size 1). |
| `reverse` | **2-opt**: reverse a random contiguous segment perm[i..j].  The canonical 2-opt move for permutation problems. |
| `3opt` *(added for hc5 / hc14)* | **3-opt double-reverse**: pick i < j < k and simultaneously reverse perm[i..j) and perm[j..k).  Provides a third structurally distinct mutation primitive (alongside swap-family and block-move/or-opt) that cannot be reached from a single 2-opt or single block_move. |
| `block_move`, `long_block_move` | Cut a contiguous block out and paste it at a random position (= or-opt for segment sizes > 1; a sub-case of 3-opt). |
| `or_opt` *(added for hc5 / hc9)* | **Or-opt / segment-shift**: relocate a short contiguous block of length **1–3** to a different position **without reversing it** (Or 1976).  Distinct from `reverse`/`3opt` (which only reverse) and from `block_move` (which uses larger blocks): the short, order-preserving shift is the classic Or-opt neighbourhood and is cheap to apply. |
| `bottleneck_relocate` *(added for hc5 / hc9)* | **Bottleneck-targeted relocation**: take the vertex eliminated at the step achieving the max torso width (`bn_idx` from `evaluate_with_bottlenecks`) and move it elsewhere, biased 60% toward an *earlier* slot so it is eliminated before accumulating fill-in.  Unlike `bottleneck→head` (which only moves it into the head `[0, t)`) it can move the vertex anywhere, letting the search push the bottleneck either earlier or deeper.  Ties the move *causally* to the objective instead of picking positions blindly. |
| `min_fill_reinsert` *(added for hc5 / hc9)* | **Min-fill-guided reinsertion**: remove one vertex (the bottleneck if known, else random) and reinsert it at the position — among a small sampled candidate set — that minimises the fill it would induce, estimated as `C(k,2) − existing_edges` where `k` is the number of its neighbours that end up *after* it in the order (those must become a clique when it is eliminated).  Cost is bounded by the small candidate set, so it is affordable even on dense graphs. |
| `shuffle` | Random.shuffle the whole permutation (the largest move; used only by `hc1_initial`). |
| `2opt_torso` | 2-opt restricted to a segment entirely inside the torso. |
| `swap_head_tail` | Swap a head vertex (i < t) with a torso vertex (j ≥ t). |
| `bottleneck→head` | Move the current bottleneck vertex (or one of its successors) into the head. |
| `k_bottleneck→head` | Same, but for K bottlenecks at once. |
| `t_shift` | Move the threshold t by ± a small step. |
| `t_random` | Resample t uniformly over [0, n−1]. |

The pattern is "start simple, escalate when you plateau". hc1's
four operators are deliberately the most basic ones; hc5 onwards
mixes in informed neighbourhoods (`bottleneck→head`, `2opt_torso`,
`swap_head_tail`) that bias the search toward moves that have a
chance of reducing max_degree.  The three newest operators (`or_opt`,
`bottleneck_relocate`, `min_fill_reinsert`) are defined once in
`core.py` and wired into `hc5_operators` and `hc9_hv_accept` (taking
those variants to **15 operators**); the first is a structurally
distinct order-preserving shift, the latter two are objective-aware
neighbourhoods that target the width bottleneck directly.

### 6.4 What each new variant taught us (hc8 – hc13)

The six paper-defensible extensions described in earlier drafts are
all implemented now.  Each one isolates one identifiable failure mode
of the hc2 – hc7 design and matches a classical idea from the
metaheuristics or graph-decomposition literature.  Empirical effect
at canonical budgets (seed 42, see § 3.1 for the full scoreboard):

| Slot | Idea | Reference | Effect on the scoreboard |
|---|---|---|---|
| `hc8_lahc` | **Late Acceptance Hill Climbing** — accept a candidate iff it is better than the current state OR better than the fitness from L iterations ago. | Burke & Bykov (2017), *The late acceptance Hill-Climbing heuristic*, EJOR 258(1), 70–78 (canonical journal version of the 2008 PATAT paper). | Within seed noise of hc5 on all three problems at our budget.  `late-accepts` counter stays at 0 — the buffer is filled with frequent archive-driven accepts, so the *late* clause never fires.  Either L needs to be tuned smaller or the budget extended.  Paper-honest finding. |
| `hc9_hv_accept` | **HV-improvement acceptance** — replace lex acceptance with "accept iff the candidate would strictly increase archive HV". | Beume, Naujoks & Emmerich (2007), *SMS-EMOA: Multiobjective selection based on dominated hypervolume*; Zitzler & Künzli (2004), *IBEA*. | **Best on small (+1 362 HV vs hc5) and best on medium (+2 636 HV vs hc5).**  Confirms the headline claim of § A.3 of the review: aligning the acceptance criterion with the scoring indicator pays off. |
| `hc10_torso_warm` | **Per-t torso-aware warm start** via degree-based partitioning (high-deg → head, low-deg → torso, min-degree within each half). | Bodlaender, Koster & van der Hoeven (2006), *On the maximum cardinality search lower bound for treewidth*; Seidman (1983), *Network structure and minimum degree*. | Strong on small (+1 293 HV vs hc5), regresses on large by ~22 700 HV vs hc5 because the warm-start build itself eats the whole 25 s budget (20 torso-aware permutations, ~0.8 s each on n = 2 426).  Solution for the paper: use fewer t-grid points on large, or budget the warm-start separately. |
| `hc11_ils` | **Iterated Local Search** — on stagnation, perturb best-so-far with K random `long_block_move` / `reverse` moves (no acceptance), resume HC. | Lourenço, Martin & Stützle (2003), *Iterated Local Search*. | After the 3-opt port the perturbation kick rarely fires inside our budgets: the wider operator search keeps the Pareto archive growing, so the `idle` counter rarely reaches the K = 500 threshold.  hc11 reduces to hc5 within seed noise on small (within 100 HV) and on large (within 54 HV).  *Honest result: ILS does not help in this regime once 3-opt is in the operator mix.*  Pre-3-opt, hc11 won on large by 16 546 HV vs the 11-op hc5 — see § 6.5. |
| `hc12_incremental` | **Incremental bitset evaluator** — re-walk only the suffix that the operator changed; t-only moves skip the chordal walk entirely. | Bodlaender & Koster (2010), *Treewidth computations I: Upper bounds*, §4. | Same algorithm as hc5 at **5.4× / 3.6× / 1.7× more iterations per second** on small / medium / large.  At matched wall time the final HV is within seed noise of hc5.  The win is *per-iteration cost*, which compounds for every other variant. |
| `hc13_tabu` | **Tabu Search** — record each accepted move's signature (e.g. the swapped-vertex pair) in a length-√n tabu list and forbid those moves for the next √n iterations.  Aspiration overrides the tabu if the candidate extends the archive. | Glover (1989, 1990), *Tabu Search Parts I & II*. | Within seed noise of hc5 on all three problems.  Crucial finding: the `tabu-rejected` counter stays at 0 because the *Pareto archive already supplies the diversity Tabu Search is designed to provide*.  Each archive add forces the focus to a new (w, t) region; cycling cannot occur the way it does in single-objective HC.  This is a publishable observation in its own right: tabu memory and the Pareto archive serve overlapping roles. |
| `hc14_steepest` | **Stochastic best-improvement (steepest-ascent) acceptance** — sample K = 16 candidate moves per step from the 12-operator set, take the lex-best, accept iff strictly better than the working solution. | Hoos & Stützle (2004), *Stochastic Local Search: Foundations and Applications*, §1.5; Whitley, Mathias & Pyeatt (1996), *Hyperplane ranking in simple genetic algorithms*. | Loses to hc5 (stochastic first-improvement) by 224 / 537 / 23 111 HV — the steepest-ascent variant pays a ~K× throughput penalty that first-improvement does not, and on dense graphs (~120 ms per evaluator call) the K = 16 cost is dominant.  Publishable finding: **on time-budget-limited stochastic local search over permutations, first-improvement beats best-improvement when the evaluator is the bottleneck.** Combining with the incremental evaluator (`hc12`-style) would close most of the gap. |

Headline take-aways for the Hill Climbing chapter:

1. **`hc9` validates the SMS-EMOA / IBEA prescription** for this
   problem: when the scoring function is HV, the acceptance rule
   should also be HV.  This is the paper's strongest algorithmic
   contribution.
2. **`hc12` makes everything cheaper**.  In the next chapter (SA /
   NSGA-II) it should be the default evaluator everywhere.
3. **`hc8`, `hc11`, and `hc13` reach parity-but-not-better with
   `hc5` once 3-opt is in the operator mix.**  Sliding-window
   acceptance (LAHC), tabu memory, and ILS perturbation are all
   designed to fight *cycling* and *plateau lock-in* in single-
   objective HC; in our multi-objective archive setup the archive
   already supplies the diversity these mechanisms manufacture, and
   they reduce to hc5 within seed noise.  **`hc7_kbottleneck` is
   the genuine winner on dense graphs**: its K-vertex coordinated
   bottleneck-relocate move beats hc5 by +3 147 HV on large because
   dense fill-in cascades produce multi-way bottleneck ties that
   no single-vertex operator can break.  Finding for the paper:
   *the Pareto archive subsumes classical escape machinery;
   problem-structure-aware operators (like K-vertex bottleneck-
   relocate) are the remaining lever on dense graphs.*
4. **`hc10` shows that the per-t coupling is a real failure mode of
   hc4–hc7**: a different warm start per t buys a measurable HV on
   small.  Reducing the warm-start cost (or running it asynchronously)
   would unlock the same gain on large.

### 6.5 Per-variant rationale — why each variant scored where it did

This section walks the scoreboard from hc2 to hc13 and explains, for
each variant, **what its score reveals**.  Numbers refer to the
canonical 25 / 12 / 25 s budgets at seed 42 (§ 3.1).

**`hc2_simple` (−1 734 246 / cap / cap).**  The minimum viable HC:
random π, fixed t = 0, swap-only, strict improvement.  Score on small
is decent because the sparse graph (avg deg 3.36) lets even random
swaps drop max_degree.  On medium and large the random initial
permutation triggers the 500-width limit on the very first
elimination step, max_degree is hard-clamped to 501, and the
single-point HV is whatever rectangle survives.  The two collapses
together motivate every later upgrade: free t, archive, warm start.

**`hc3_archive` (−1 735 516 / 0 / 0).**  Adds a Pareto archive and
makes t mutable.  On small this buys ~+1 270 HV (the archive picks up
non-dominated points with smaller t).  On medium and large the
archive correctly *rejects* over-width solutions, so it remains empty
because random init is too far from feasibility — hence score = 0.
This is exactly the dense-graph failure mode the descent metric
(`evaluate_descent`) tries to fix; it can navigate `cap_violations`
downward but not fast enough to reach feasibility inside 12 / 25 s.

**`hc4_warm_start` (−1 810 555 / −1 604 342 / −4 790 548).**  The
single biggest jump in the table.  Replacing random init with a
min-degree elimination order seeded across 20 t-values lifts every
score by hundreds of thousands of HV.  On small the +75 039 HV gain
vs hc3 is almost entirely the warm start's gift (random init is so
far from a width-20 plateau that local search struggles to reach it
in budget).  On medium and large the warm start is the difference
between "0" and "real score".

**`hc5_operators` (−1 810 728 / −1 606 826 / −4 799 384).**  +12
operators with UCB-style adaptive selection.  Operators now include an
explicit **2-opt** (`reverse`, segment reversal) and **3-opt**
(`op_3opt`, simultaneous reversal of two adjacent non-overlapping
segments).  Adding 3-opt on top of hc5's 11-operator predecessor lifts
the score by +173 HV on small, +306 HV on medium, and +12 447 HV on
large — confirming that the third structurally distinct mutation
primitive (alongside swap-family and block-move/or-opt) is genuinely
useful on dense graphs.  Gain over hc4 is now +173 HV on small (was a
−6 HV regression without 3-opt), +2 484 HV on medium, and +8 836 HV
on large.  The K-vertex bottleneck move (hc7, +3 147 HV over hc5) and
the ILS perturbation kick (hc11, +4 099 HV over hc5) still beat hc5
on large, but by margins much smaller than the pre-3-opt analysis
suggested.

**`hc6_gap_fill` (−1 810 261 / −1 604 628 / −4 798 388).**  Adds
per-t round-robin local search over a finer 40-point t-grid.
Regresses slightly vs hc5 on all three problems.  The diagnosis is
in `results.md §6`: the round-robin spends budget on intermediate-t
regions that are dominated by the (w, t) = (20, 0) corner, so most
candidate moves cannot enter the archive.  hc6 demonstrates that
gap-filling *needs HV-aware acceptance* (which hc9 provides) to
work; lex acceptance alone defeats it.

**`hc7_kbottleneck` (−1 810 606 / −1 607 115 / −4 802 531).**  Adds
the K-vertex coordinated bottleneck-relocate operator.  Wins on
medium by +595 HV vs hc5 and on large by +15 594 HV vs hc5 (the dense
fill-in cascades produce more multi-way bottleneck ties, which the
K-vertex move can simultaneously relocate).  On small the (w, t)
plateau is structural — the bottleneck vertices are mutually
adjacent in the fill-in graph, so moving K of them does not help —
and the operator only matches hc5 within seed noise (+57 HV).
The large-graph win makes hc7 the second-best lex-acceptance variant
on dense graphs, behind only hc11.

**`hc1_initial` (−1 396 010 / over-width / over-width).**  The
Algorithm 10 baseline: random init, four trivial operators (shift_left,
shift_right, shuffle, t_shift), random restarts, no archive, no warm
start.  Lowest score in the table on small (−338 236 HV below hc2's
single-objective HC).  This is intentional — hc1 is the conceptual
*starting point*, not a contender.  Its operator stats reveal the
headline didactic finding: `shift_left` and `shift_right` get
0 % accept rate, all progress comes from `shuffle` (large jumps)
and `t_shift` (free t reductions).  On medium and large random init
triggers the width limit on step 1, so hc1 collapses to the same
over-width point as hc2.

**`hc8_lahc` (−1 810 626 / −1 602 540 / −4 785 230).**  Within seed
noise of hc5 on all three problems.  The `late-accepts` counter is 0
across the matrix — the LAHC buffer is filled with frequent
archive-driven accepts (every iteration the current fitness is
re-written into `buf[i % L]`), so the late-acceptance clause never
fires.  Either L must shrink dramatically or the budget must extend
to expose stagnation.  Paper-honest finding: LAHC is designed for
single-objective HC stuck on a flat plateau, but our multi-objective
archive prevents the plateau from forming in the first place.

**`hc9_hv_accept` (−1 811 911 / −1 609 156 / −4 792 651).**  Best on
small and best on medium.  HV-improvement acceptance moves the working
solution whenever the candidate would extend the Pareto front — even
when it is lex-worse than the current focus.  This is the exact
mechanism missing in lex acceptance: a candidate at (w = 12, t = 800)
is lex-worse than focus (w = 20, t = 0) but adds a strip of HV that
lex acceptance discards.  hc9 keeps that HV.  On large the budget
runs out before HV-accept gets enough archive saturation to overtake
hc5's lex acceptance, so it slightly trails there.

**`hc10_torso_warm` (−1 811 842 / −1 602 335 / −4 764 249).**  Strong
on small (+1 293 HV vs hc5); near hc5 on medium (−4 185 HV);
regresses badly on large (−22 688 HV vs hc5, −39 234 HV vs the best
on large hc11) because building 20 torso-aware permutations costs
~15 s on n = 2 426, leaving almost no time for HC.  This is a real
finding: the *idea* of a different warm start per t is correct on
sparse graphs but the construction cost dominates on dense ones.  For
the paper, reducing the t-grid to 5 points on large (or running the
warm-start asynchronously) would recover most of the small-graph
gains.

**`hc11_ils` (−1 810 645 / −1 606 520 / −4 803 483).**  Within ~100 HV
of hc5 on small (+96 HV better) and identical to hc5 on medium
(both at −1 606 520).  On **large hc11 wins by +16 546 HV vs hc5**
and is the single best variant of the entire HC family on that
instance.  The perturbation kick fires only on large, where the
combination of denser fill-in and the 25 s budget allows the archive
to saturate and `idle` to reach the K = 500 threshold — at which
point the long_block_move / reverse perturbation jolts the working
solution out of the lex-acceptance basin.  Paper-honest caveat:
stagnation detection on a Pareto-archive HC could be sharper if it
used *archive-HV* idle instead of "no archive adds", because the
archive can add HV-neutral points indefinitely on sparse graphs.

**`hc12_incremental` (−1 810 510 / −1 603 830 / −4 780 244).**  Same
algorithm as hc5 with the incremental evaluator from `core.py`.  Final
HV is within seed noise of hc5 because hc12 spends its budget at 5.4×
/ 3.6× / 1.7× more iterations per second, and at this saturation
point we are essentially seed-noise-limited.  The win is the
*throughput*, not the per-run HV: every other variant — and every
future SA / NSGA-II run — benefits from the same evaluator and gets
the same multiplier.

**`hc13_tabu` (−1 810 508 / −1 606 136 / −4 798 265).**  Within seed
noise of hc5 on all three problems.  The `tabu-rejected` counter
stays at 0: the pre-sampling loop (try 8 candidates before falling
back) successfully avoids the recently-used move signatures.  More
importantly, the Pareto archive itself already supplies the
diversity that Tabu Search is designed to manufacture — each archive
add forces the working solution onto a new (w, t) point, so the
classical "cycle on a flat plateau" failure mode that motivates
tabu memory simply does not occur here.  hc13's score is a
publishable null result: *multi-objective Pareto archive HC does not
benefit from tabu memory in budgets where the archive remains
active*.

**`hc14_steepest` (−1 810 504 / −1 606 351 / −4 776 273).**  The
stochastic best-improvement counterpart of hc5.  Same 15-operator set
(including 3-opt), same warm start, same archive, same lex
acceptance — the only change is that each outer step samples
K = 16 candidate moves and adopts the lex-best.  hc14 *loses* to
hc5 on every problem (−224 HV on small, −537 HV on medium,
−23 111 HV on large).  Diagnosis: at K = 16 each outer iteration
costs 16 evaluator calls, so iteration throughput drops by ~K and
the gain from "pick the best" cannot recover it on a budget-limited
matrix.  On large the ~120 ms evaluator cost per call means hc14
manages only ~12 outer iterations in 25 s, vs hc5's several hundred.
This is the classical first-improvement-beats-best-improvement
finding (Hoos & Stützle 2004 §1.5) reproduced on a multi-objective
Pareto-archive HC: **on time-budget-limited stochastic local search
over permutations, first-improvement dominates best-improvement when
the evaluator is the bottleneck**.  The 2 976 candidate
evaluations per run still drive 2 898 archive adds, so the wider
sampling is not wasted — the Pareto front is fully populated — but
the *working-solution* trajectory underperforms hc5.  Combining hc14
with the incremental evaluator (`hc12`-style) would close most of
the throughput gap and is in scope for the journal extension.

### 6.6 Hill Climbing variants considered but not implemented

A quick map of the broader HC family with notes on what we left
out and why:

| Variant | Status here | Reasoning |
|---|---|---|
| **Variable Neighborhood Search** (Mladenović & Hansen 1997) | Not implemented as a separate file. | Approximated by hc5's adaptive operator selection — each operator IS a different neighbourhood, and the UCB-style weights cycle attention between them on stagnation.  A literal VNS protocol (escalate from `swap` to `block_move` to `shuffle` on rejection) was prototyped during development and gave results within seed noise of hc5. |
| **GRASP** (Feo & Resende 1995) | Construction phase covered by min-degree warm start (hc4+); randomised greedy construction not separately tested. | GRASP's construction phase is essentially "randomised min-degree" which is exactly what `min_degree_perm(rng=...)` does in hc4-hc7.  A separate GRASP file would be cosmetic. |
| **Guided Local Search** (Voudouris & Tsang 1999) | Not implemented. | GLS adds *penalty terms* to the fitness for features that have appeared in many local optima.  Adapting it to multi-objective HV is non-standard and would be a paper of its own. |
| **Best-improvement / Steepest-ascent HC** | Implemented as `hc14_steepest` (stochastic best-improvement with K = 16 sampled candidates per step). | A literal "all neighbours" steepest ascent is `O(n²)` per step (every pairwise swap), intractable at n ≥ 1 357.  The stochastic K = 16 variant is the standard substitute (Hoos & Stützle 2004) and produces the expected result: first-improvement (hc5) beats best-improvement (hc14) by 224 / 537 / 23 111 HV on small / medium / large, because the K-fold evaluator cost is dominant on a fixed wall budget. |
| **Random search baseline** | Not implemented. | Trivial baseline (random π, random t, return).  Listed in the methodology checklist for the paper but not as an `hc*` slot since it is not really hill climbing. |
| **Parallel / island-model HC** | Not implemented. | Out of scope for the single-thread Python codebase; covered in the planned CUDA / multi-processing chapter. |

The HC family is now covered to a depth that supports a publishable
paper: **14 implemented variants** spanning random restart (hc1),
pure HC (hc2), single-operator + Pareto archive (hc3), min-degree
warm start (hc4), operator diversity with 2-opt + 3-opt (hc5), per-t
local search (hc6), K-vertex structural moves (hc7), late acceptance
(hc8), HV-aware acceptance (hc9), torso-aware warm start (hc10),
iterated local search (hc11), fast evaluation (hc12), tabu memory
(hc13), and best-improvement / steepest ascent (hc14).  The
conclusions are:

1. **The Pareto archive is the centerpiece.** Variants that add
   memory designed to fight plateau lock-in (LAHC, tabu, ILS-
   perturbation) all get subsumed by the archive's natural
   diversity in our budgets and reduce to hc5 within seed noise.
2. **Warm start beats every operator innovation.** The hc3 → hc4
   jump (+75 039 HV on small; from 0 to feasibility on medium /
   large) dwarfs every other delta in the table.
3. **HV-aware acceptance beats lex acceptance on sparse graphs.**
   `hc9` wins on small (+2 750 HV vs hc5) and on medium (+691 HV
   vs hc5), and on 12 of the 20 synthetic instances.  On large the
   leader is `hc7_kbottleneck` (K-vertex bottleneck-relocate),
   which beats hc9 by 9 022 HV — structural operators win on
   dense graphs, HV-acceptance wins on sparse.
4. **Faster evaluation compounds.** hc12 buys 1.7–5.4× more
   iterations for every variant; in the next chapter (SA /
   NSGA-II) it should be the default evaluator everywhere.
5. **First-improvement beats stochastic best-improvement** in
   this regime: hc14 with K = 16 sampled candidates loses to hc5
   by 224 / 537 / 23 111 HV because the K-fold evaluator cost
   eats the iteration budget.  Combining hc14 with the incremental
   evaluator (hc12-style) is the obvious journal-extension fix.

### 6.7 Reasoning ladder — *why* each variant moves the score

This section is the "if a reviewer asks why hcN scored Y on instance
Z, look here first" reference.  For each variant we give a four-line
record: (i) the **capability** it adds on top of its predecessor,
(ii) the **failure mode** that capability addresses, (iii) the
**empirical Δ** vs the predecessor (mean over the three official
instances at seed = 42 unless noted), (iv) the **mechanism** that
explains the sign of the delta.

#### Capability ladder (cumulative on small-graph, seed = 42)

| Step | Variant | Δ vs predecessor (small) | Cumulative score (small) | What the lift buys |
|---|---|---:|---:|---|
| 0 | `hc1_initial` (baseline) | — | −1 396 010 | random init, four trivial operators, no archive — establishes the floor |
| 1 | `hc2_simple` (drop t-mutation, focus max_degree) | **+338 236** | −1 734 246 | most of hc1's budget is wasted on `t_shift`; pinning t=0 lets the same 25 s drive max_degree down hard |
| 2 | `hc3_archive` (Pareto archive + descent metric) | +1 270 | −1 735 516 | tiny on sparse because hc2 already at the plateau; the gain materialises on medium/large where descent is the only thing that walks out of over-width |
| 3 | `hc4_warm_start` (min-degree warm) | **+75 039** | −1 810 555 | random init is far from the (20, 0) plateau; warm start places the initial state inside the basin so local search has something to refine |
| 4 | `hc5_operators` (12 ops incl. 2-opt + 3-opt) | +173 | −1 810 728 | small lift on small because the plateau is structural — operator diversity helps mostly on medium (+2 484 HV) and large (+8 836 HV) where richer moves cross intermediate fill-in basins |
| 5 | `hc6_gap_fill` (per-t round-robin LS) | −467 | −1 810 261 | regresses *under lex acceptance* because the t-grid wastes budget on dominated intermediate regions; this same mechanism becomes a win under HV-acceptance (see hc9) |
| 6 | `hc7_kbottleneck` (K-vertex coordinated move) | +345 | −1 810 606 | breaks multi-vertex bottleneck plateaus by moving K tied vertices in one step; effect ~zero on sparse, +595 HV on medium, +3 147 HV on large |
| 7 | `hc9_hv_accept` (HV-acceptance) | **+2 750** vs hc5 | −1 813 478 | aligns the working-solution acceptance rule with the scoring indicator; intermediate-t candidates that lex acceptance discards as "lex-worse" are kept because they grow archive HV |

The two biggest jumps on small are **(step 1)** dropping t-mutation
(+338 k HV — hc1's budget was 50% wasted on resampling t) and
**(step 3)** the warm start (+75 k HV — local search needed somewhere
sensible to start from).  Everything else is small refinement at the
hundreds-of-HV level.

#### Per-variant cause → effect → mechanism

| Variant | Capability added | Failure mode addressed | Empirical effect | Mechanism |
|---|---|---|---|---|
| `hc1_initial` | random restarts (Algorithm 10) | "stuck in one basin forever" of pure HC | floor of the table | each restart is a uniform random π — too far from any basin to recover with adjacent swaps; useful as a baseline, not a contender |
| `hc2_simple` | fix t = 0; minimise only max_degree | hc1 wastes 25% of accepts on `t_shift` that doesn't drive max_degree | +338 236 HV on small | with t pinned, every accepted swap reduces actual width; the algorithm is now apples-to-apples vs the literature single-objective HC |
| `hc3_archive` | Pareto archive + descent metric | hc2 only reports one (w, t); ignores the front | +1 270 HV on small | archive captures the trade-off; descent metric (over-width step count) walks toward feasibility on dense graphs |
| `hc4_warm_start` | min-degree elimination order as initial π | random init has no chance to find the (20, 0) plateau in 25 s on n = 1 357 | +75 039 HV on small; from 0 to feasibility on medium/large | min-degree is a classical chordal-completion heuristic; it places the start state inside the high-quality basin so local search has a meaningful neighbourhood |
| `hc5_operators` | 15 operators (swap, adjacent_swap, insert, 2-opt = reverse, **3-opt**, block_move, long_block_move, bottleneck→head, swap_head_tail, 2opt_torso, **or_opt**, **bottleneck_relocate**, **min_fill_reinsert**, t_shift, t_random) with UCB-style weights | hc4's single-swap operator cannot peel the bottleneck on the (20, 0) plateau | +173 HV small, +2 484 HV medium, +8 836 HV large | each operator is a different neighbourhood; UCB biases sampling toward operators with non-zero accept rate; 3-opt in particular reaches double-reversal neighbourhoods unreachable by single 2-opt or block_move |
| `hc6_gap_fill` | per-t round-robin local search over 40-point t-grid | hc5's archive is dense at high t but sparse at intermediate t | −467 HV on small (regression) | round-robin spends budget in regions whose candidates are lex-dominated by the (20, 0) corner; lex acceptance rejects them so the budget evaporates.  *Same idea works under HV-acceptance (hc9).* |
| `hc7_kbottleneck` | K-vertex coordinated bottleneck-relocate | hc5 cannot break ties when ≥ 2 torso vertices share max_degree (one-at-a-time moves leave the others) | +57 HV small, +595 HV medium, **+3 147 HV large**, **best on large overall** | move K tied bottleneck vertices into the head in one step; effective only when the bottleneck has multi-way ties, which happens on denser fill-in cascades |
| `hc8_lahc` | Late-acceptance sliding window | hc5 cycles on flat plateaus (or so the LAHC paper expects) | parity with hc5 | the Pareto archive already supplies the diversity LAHC manufactures — every archive add forces a new (w, t) focus, so the late-acceptance clause never fires.  *Honest null result.* |
| `hc9_hv_accept` | HV-improvement acceptance (SMS-EMOA / IBEA) | hc5's lex acceptance discards candidates that grow archive HV but are lex-worse than current focus | **+2 750 HV on small**, **+691 HV on medium** — best on both | acceptance rule = scoring indicator.  Intermediate-t candidates with (w = 12, t = 800) are kept even when current focus is (w = 20, t = 0), because they add a strip of HV the focus doesn't dominate |
| `hc10_torso_warm` | per-t torso-aware warm start (high-degree → head, low-degree → torso) | hc5 reuses one global min-degree order across the whole t-grid, so intermediate t inherits irrelevant head structure | +1 114 HV small; trails on medium/large | each t gets a tailored construction; the win is real on sparse but the construction cost (20 × min-degree elimination at ~0.8 s on n = 2 426) eats the whole 25 s budget on large |
| `hc11_ils` | Iterated Local Search perturbation kick | hc5 gets stuck on flat plateaus where every accepted move is HV-neutral | +0 small/medium; **trailed hc7 by 2 335 HV on large after 3-opt port** | perturbation fires only when archive `idle` reaches K = 500; 3-opt-driven archive growth makes `idle` rarely hit that threshold, so the kick fires less often |
| `hc12_incremental` | incremental bitset evaluator (re-walk only the changed suffix) | the chordal-completion walk is the hot loop, ~120 ms / call on n = 2 426 | parity HV at 1.7–5.4× more iterations / sec | per-iteration cost drops; total search budget is now bounded by *number of useful moves found*, not *number of moves evaluated* |
| `hc13_tabu` | length-√n move-attribute tabu list with aspiration override | hc5 revisits move signatures that previously failed | parity with hc5; `tabu-rejected` counter is 0 | Pareto archive already moves the focus away from sampled-and-rejected regions; tabu has no signatures to forbid that the archive hasn't already vacated |
| `hc14_steepest` | sample K = 16 candidates per outer step, take the lex-best | reviewer asks "did you try best-improvement instead of first-improvement?" | −224 / −537 / −23 111 HV vs hc5 | each outer step costs K eval calls; on large that is ~2 s/iter and the budget allows only ~12 outer iters.  First-improvement gets ~hundreds of iters in the same wall time |

#### Decision tree — "given my instance, which variant should I reach for?"

```
Is the instance sparse (avg deg < ~5) and small (n < ~500)?
  YES → hc9_hv_accept     (HV-acceptance is the dominant lift)
  NO  → continue

Is the instance dense (avg deg > ~20) with bottlenecks that tie at max_degree?
  YES → hc7_kbottleneck   (K-vertex move on bottleneck ties)
  NO  → continue

Is the budget extremely tight (< 1 s / instance)?
  YES → hc12_incremental  (every iteration counts)
  NO  → continue

Does the warm-start permutation itself trigger the 500-width limit?
  YES → hc2_simple        (no warm-start dependency; survives the over-width front)
  NO  → hc9_hv_accept     (the safe default)
```

This decision tree is the headline takeaway for a practitioner who
wants to pick *one* variant without running the full matrix.

---


## 7. The metaheuristics chapter (SA, GRASP, VNS)

The second chapter adds three classical metaheuristics on the **same
substrate** as the HC family: the identical 15-operator pool
(`meta_common.OPERATORS`), the same `ParetoArchive`, the same
HV-improvement acceptance criterion (accept iff the candidate strictly
increases archive HV — the SMS-EMOA / IBEA rule introduced by `hc9`), and
the same warm starts and t-grid seeding.  Only the **search-control
strategy** differs, which is what makes the four families directly
comparable.  Each lives in its own package and feeds **one shared
archive**, so its score is the HV of the union over the whole run.

| algorithm | file | strategy | key hyperparameters | reference |
|---|---|---|---|---|
| **Simulated Annealing** | `algorithms/simulated_annealing/sa.py` | A single walker over the 15-operator neighbourhood using the **Metropolis criterion**: a worsening move of energy increase ΔE is accepted with probability `exp(−ΔE / T)`, where `E(w, t) = −(n − w)(n − t)` (the dominated-area energy) and `T` cools over the run.  Every candidate is still offered to the archive regardless of acceptance. | `--schedule {geometric,linear,adaptive}`, `--alpha`, `--steps-per-T`; `T0` auto-calibrated to ≈ 0.4 acceptance of a typical worsening move | Kirkpatrick et al. 1983; Černý 1985 |
| **GRASP** | `algorithms/grasp/grasp.py` | Independent restarts, each a **greedy-randomized min-degree construction** (Restricted Candidate List of all vertices with degree ≤ `d_min + α·(d_max − d_min)`, pick uniformly) followed by **HV-improvement local search**.  α = 0 is pure greedy, α = 1 fully random.  Budget split equally across restarts. | `--alpha` (RCL greediness), `--restarts` | Feo & Resende 1995; Resende & Ribeiro 2016 |
| **VNS** | `algorithms/vns/vns.py` | Alternates **shaking** (k random operator moves — the k-th neighbourhood) with **HV-improvement descent**; on archive-HV improvement accept and reset the ladder `k = 1`, else intensify `k = k + 1` up to `k_max`.  Both shake and descent are wall-clock deadline-bounded so a violent shake never overshoots the budget. | `--k-max` (ladder depth), `--shake-strength`, `--ls-steps` | Mladenović & Hansen 1997; Hansen et al. 2010 |

### 7.1 Tuned results (seed 42, canonical budgets)

Grid-tuned per problem (`tools/tune.py`); canonical budgets are
small/large 25 s, medium 12 s.  Submitted top-20 (verify) scores, more
negative is better:

| technique | small | medium | large |
|---|---:|---:|---:|
| Simulated Annealing | −1 814 521 | −1 603 923 | −4 777 663 |
| **GRASP** | **−1 816 174** | **−1 615 368** | **−5 010 404** |
| VNS | −1 815 418 | −1 608 650 | −4 807 533 |
| best HC (chapter 1) | −1 814 976 *(hc15)* | −1 609 194 *(hc13)* | −4 800 196 *(hc13)* |
| % of leaderboard top (GRASP) | 99.25 % | 92.57 % | 91.21 % |

**GRASP wins every instance**, beating the best HC on all three — its
`large-graph` −5 010 404 is a new project best (~210 200 HV beyond hc13),
robust across seeds and now promoted to canonical.  The full tuning
tables, 11-seed sweep, and Friedman/Nemenyi/Wilcoxon analysis are in
[RESULTS.md](RESULTS.md) § 4b.

**Why GRASP wins.**  On the dense large instance, GRASP's repeated
greedy-randomized *constructions* explore qualitatively different
elimination orders that a single-trajectory method (SA, VNS) cannot
reach by local moves — consistent with the HC chapter's finding that on
dense graphs the *construction heuristic* dominates the local-search
policy.  SA is the weakest of the three because it collapses the
bi-objective into one scalar energy; this is a design property of the
method, not a tuning artefact.

### 7.2 Enhanced variants (lifting each family's design ceiling)

The §7.1 baselines are faithful textbook implementations, but each has a
named ceiling that the baseline cannot escape by tuning alone.  Each
family therefore gained an **additive** enhancement — selectable by a flag
whose default preserves the canonical behaviour, and which writes to an
alternate submission stem via `--algo`, so no canonical file is touched.

| baseline ceiling | enhancement | flag | what changes |
|---|---|---|---|
| **SA** collapses the bi-objective into one scalar energy `E = −(n−w)(n−t)` | **AMOSA acceptance** — bi-objective Metropolis on the *amount of domination* against the live archive, with a normalised temperature `T/T0` | `--accept amosa` (default `scalar`) | The walker accepts on multi-objective dominance pressure instead of a single scalar, so it tracks the *front* rather than one rectangle.  Bandyopadhyay, Saha, Maulik & Deb 2008 (lineage Suppapitnarm et al. 2000). |
| **GRASP** at fixed α rebuilds *similar* constructions every restart and never recombines good solutions | **Path-relinking + reactive-α** — an elite pool of local optima; each fresh local optimum is relinked toward a random elite (one transposition per step, every intermediate offered to the archive); α is resampled by historical front quality | `--path-relinking`, `--reactive-alpha` (both off by default) | Path-relinking captures non-dominated decisions lying *between* two good orderings; reactive-α stops the sweep wasting restarts on an unproductive greediness level.  Glover 1997; Prais & Ribeiro 2000; Resende & Ribeiro 2016. |
| **VNS** uses a single stochastic, weighted-sample HV-descent — not a true structured local search | **True VND** — a fixed ordered list of neighbourhoods `N₁…N_L` (cheapest/most-local first), best-of-`k` within each, reset to `N₁` on improvement, advance on failure, stop at a genuine VND local optimum | `--local-search vnd` (default `hv-descent`) | Replaces the stochastic descent with deterministic Variable Neighbourhood Descent, so each shake is refined to a *proper* local optimum before the ladder moves.  Hansen & Mladenović 2001. |

### 7.3 Four-family head-to-head harness

`tools/meta_multiseed.py` runs every family — `hc9` (HC reference), `sa`
and `sa_amosa`, `grasp` and `grasp_pr`, `vns` and `vns_vnd` — at the same
seeds on the same instances, re-scores each freshly written submission
with the canonical evaluator (so the recorded value equals the verify
score), and reports a **Friedman** omnibus test with a **Nemenyi**
critical-distance ranking.  Canonical submissions are never disturbed:
`hc9` is backed up and restored, every other config writes a throwaway
`<name>_ms` stem that is deleted after scoring.  This is the seed-symmetric,
significance-tested apparatus for the HC-vs-SA-vs-GRASP-vs-VNS comparison
the PhD chapter needs (the chapter-1 HC sweep already has its own 12-seed
Friedman/Nemenyi in [RESULTS.md](RESULTS.md)).

```
python3 tools/meta_multiseed.py --seeds 1,2,3,4,5,6,7,8,9,10,11
```

**Result of the 11-seed sweep (final).** Friedman χ²(6) = 62.13 (≫ the
α = 0.001 critical 22.46), so the seven configs are not equivalent.
Nemenyi average ranks (1 = best, CD₀.₀₅ = 1.78): **grasp 2.82** < hc9 3.12
< vns 3.21 < vns_vnd 3.55 < grasp_pr 4.00 < sa 5.24 < sa_amosa 6.06.
GRASP is the strongest family and sits significantly ahead of both SA
variants.  Per-family Wilcoxon (enhanced vs baseline) finds **every
enhancement no better than its baseline**: AMOSA significantly worse than
SA (p ≈ 0.0003), GRASP path-relinking worse than plain GRASP (p ≈ 0.025),
and VND statistically tied with stochastic-descent VNS (p ≈ 0.13).  Each
family is therefore **green-lit on its baseline**, and the three
enhancements stand as a clean, tested **negative result** — kept in-tree
behind their default-off flags for reproducibility, not promoted.  Full
tables in [RESULTS.md](RESULTS.md) § 4b.5.

---

## 8. The ACO bridge chapter (Ant Colony Optimization) — a negative result

The third chapter tests whether a **construction** metaheuristic — one that
*builds* elimination orders from a pheromone-biased probability model instead
of *perturbing* a warm-started order — adds anything on this problem. It is
the bridge between the perturbative families (HC / SA / VNS) and the
restart-construction family (GRASP). It runs on the same substrate
(`algorithms/meta_common.py`: 15-operator pool, shared archive, `−HV`
scoring, canonical budgets). Implementation in `algorithms/aco/aco.py`:

| variant | file / flag | what it does | params | reference |
|---|---|---|---|---|
| **Ant System (AS)** | `aco.py` (`variant=as`, baseline) | Position-indexed pheromone `τ[i,v]`; each ant builds an order picking the next vertex with `p(v) ∝ τ[i,v]^α · η(v)^β`, dynamic min-degree heuristic `η = 1/(1+deg_R(v))`; all ants deposit. | `--ants`, `--alpha`, `--beta`, `--rho` | Dorigo 1992 |
| **MAX–MIN AS (MMAS)** | `aco.py` (`--variant mmas`) | Only the iteration-best ant deposits; pheromone clamped to `[τ_min, τ_max]`; stagnation reset. | + bounds, stall reset | Stützle & Hoos 2000 |
| **Hybrid** | `--local-search` (`aco_ls`, `aco_mmas_ls`) | Each ant's order is polished by a step-bounded HV-improvement descent over the same 15 operators GRASP/VNS use, accepting iff archive HV strictly grows. Additive: defaults untouched. | + `--ls-steps` | — |

### 8.1 Result: ACO is last on every instance

Tuned full-grid at seed 42, then an 11-seed head-to-head. Across the 11-config
Friedman omnibus (χ²(10) = 258.459, p = 9e-50, CD₀.₀₅ = 2.583) the four ACO
variants take the **bottom four ranks** (9.24–10.00), each separated from
every HC/SA/GRASP/VNS config by more than the critical distance — i.e. ACO is
*significantly* worse than everything else. Pure construction **fails outright
on `large-graph`** (every cell scores −0: only ~16 ants build in 25 s on
n = 2 426, none under the 500-width cap). The hybrid descent rescues
feasibility there (−4 670 050) but still trails GRASP (−5 010 404).

### 8.2 Why construction does not transfer here

Four structural reasons (full Wilcoxon tables in [RESULTS.md](RESULTS.md)
§ 4c):

1. **Construction is the wrong unit of work on a dense graph.** One
   pheromone-biased order on large costs ~1.5 s and rarely lands under the
   width cap; ~16 ants draw 16 losing tickets. The other families start from a
   warm start that is feasible by construction.
2. **The pheromone has nothing to learn in budget.** With 1–2 iterations on
   large, `τ` never converges; the heuristic `η` does all the work — which is
   why `β = 4.0` (heavy heuristic, light pheromone) tunes best. A best-config
   ACO that leans on its greedy heuristic *is* a noisy greedy constructor, and
   plain greedy min-degree (GRASP at α = 0) already beats it.
3. **The rescue is borrowed, not native.** Only the shared HV-descent makes
   ACO feasible on large; once added, the colony contributes nothing
   measurable (identical −4 670 050 across all 36 cells and both pheromone
   schemes, AS ≡ MMAS). The score is "what one descent from a constructed
   seed achieves," not an ant-colony result.
4. **Diversity is cheaper elsewhere.** GRASP gets the same construction
   diversity from RCL randomisation and unions restart fronts in the same
   archive, reaching the project-best −5 010 404. ACO adds a pheromone matrix
   and deposit rules to reach a worse number.

**MMAS vs AS** is marginal and sparse-instance-only (MMAS better on
small-graph construction at p = 0.014; tied or identical elsewhere).
**Hybrid vs construction** is a feasibility rescue, not a quality lift
(large p = 0.0009 in favour of the hybrid; small/medium slightly worse, not
significant). ACO is kept as a fully-tuned, seed-robust, statistically
characterised **negative result** — never promoted to a canonical submission.

---


## 9. The population chapter (NSGA-II, SMS-EMOA) — a sparse-instance win

The fourth chapter tests the **population** paradigm: instead of one incumbent
whose front is a by-product of the shared archive, evolve a *whole population*
of elimination orders and read the Pareto front off it directly. NSGA-II and
SMS-EMOA are the canonical population MOEAs. Same substrate
(`algorithms/meta_common.py`: 15-operator pool, shared archive, `−HV` scoring,
canonical budgets). Implementation in `algorithms/nsga2/nsga2.py`:

| variant | file / flag | what it does | params | reference |
|---|---|---|---|---|
| **NSGA-II** | `nsga2.py` (`variant=nsga2`, baseline) | Fast non-dominated sort + crowding distance, binary crowded-comparison tournament, generational `(μ+μ)`. Order crossover (OX) blends two parent orderings — the one recombination move the perturbative chapters lacked. | `--pop`, `--pc`, `--pm` | Deb et al. 2002 |
| **SMS-EMOA** | `--variant sms` | Steady-state `(μ+1)`: each step makes one offspring and culls the worst-front member whose removal costs the least hypervolume — promotes hc9/GRASP's HV *acceptance* rule to a *selection* policy. | `--pop`, `--pc`, `--pm` | Beume, Naujoks & Emmerich 2007 |
| **Memetic** | `--local-search` (`nsga2_ls`, `sms_ls`) | Each offspring polished by a step-bounded HV-improvement descent over the same 15 operators. Additive: pure variants untouched. | + `--ls-steps` | — |

A one-time **final harvest** re-seeds every distinct final-population order
across the full `t`-grid before scoring, so the threshold axis is read fairly
(as GRASP does). PAES — the `(1+1)` MOEA — is not re-implemented because hc9's
HV-archive search already covers that design point.

### 9.1 Result: SMS-EMOA is the first method to beat hc9 (on small-graph)

Tuned full-grid at seed 42, then a 15-config 11-seed head-to-head. Friedman
omnibus χ²(14) = 321.607, p = 3.6e-60, CD₀.₀₅ = 3.483. The population family
sits mid-pack on the *pooled* rank (sms 6.530, nsga2 6.924, sms_ls 7.803,
nsga2_ls 8.167) because the pooled rank averages over all three instances and
the family is weak on the dense ones. The per-instance Wilcoxon tests are
decisive instead: on **`small-graph` the four population variants take the top
four seed-means**, and SMS-EMOA significantly beats GRASP (p = 0.0058),
GRASP+PR (p = 0.0076) and **hc9 (p = 0.0329)** — the first method in the entire
study to beat the long-standing hc9 benchmark on any instance. `sms_ls` posts
the single best small-graph run of the study (−1 818 628). On medium and large,
GRASP wins by ~18 000 and ~160 000 HV.

### 9.2 Why population wins sparse and loses dense

Full Wilcoxon tables in [RESULTS.md](RESULTS.md) § 4d. Three designed
comparisons:

1. **SMS-selection ≈ NSGA-crowding (Q1).** HV-contribution culling vs crowding
   distance is *tied everywhere* (pooled p = 0.42). SMS-EMOA's theoretical edge
   does not materialise — the fronts are small and crowding already keeps them
   well-spread.
2. **Memetic descent *hurts* (Q2).** `nsga2_ls` < `nsga2` (p = 0.0029),
   `sms_ls` < `sms` (p = 0.0071), both favouring the *pure* variant — the
   opposite of ACO. ACO's pure construction was infeasible on large so descent
   was pure gain; here the pure population is already feasible, so
   per-offspring descent just burns generations.
3. **Sparse front is two-dimensional, dense front is a staircase.**
   `small-graph` admits a real width/threshold trade-off curve that a diverse,
   order-crossover-recombined population samples directly; on dense graphs the
   feasible orders pin to one narrow width band, so breadth is wasted and
   GRASP's restart-construction (drives the single best width down) wins.

A consistent thread: small populations win the tuning (`pop = 20` in 10/12
cells) and the memetic axis loses — this problem rewards **iteration depth over
population breadth**, because the min-degree warm start is already strong.
Order crossover is the active ingredient: it is the only move in the study that
*blends* two good orderings, and that is precisely what pays on the sparse
instance. All four variants stay on their own `--algo` stems; canonical
submissions are untouched.

---


## See also

- [EXPLAINER.md](EXPLAINER.md) — the problem from scratch in plain language
- [PROBLEM.md](PROBLEM.md) — formal problem statement and toy walkthrough
- [RESULTS.md](RESULTS.md) — the scoreboard (HC § 4, metaheuristics § 4b, ACO § 4c, population § 4d)
- [FUTURE.md](FUTURE.md) — where this is going
