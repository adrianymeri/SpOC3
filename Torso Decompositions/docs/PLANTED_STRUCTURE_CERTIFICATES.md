# The winning puzzle: planted-structure certificates for large-graph

*Analysis date: 2026-07-11. All numbers verified against the live submission pool
(Mac snapshot 17:23: best-20 −5,476,521, gap +16,541, envelope −5,491,465, cap cost 14,944).*

## 1. The large-graph instance, fully reverse-engineered

Bannach (p.c., 2026-07-10): instances = disjoint union of low-treewidth graphs, glued by a
dense graph. For large-graph this is now **completely recovered**:

- **Glue = exactly three vertex-disjoint cliques**: K₅₀₀ ⊇ 375-twin-class (deg 499),
  K₄₀₀ ⊇ 102-class (deg 399), K₃₀₀ ⊇ 62-class (deg 299). Verified: each class's closed
  neighbourhood is a clique; the three are pairwise disjoint; 500+400+300 = 1200 = the
  entire dense core (core number ≥ 50).
- **Planted components**: removing the 1200 glue vertices leaves **25 components,
  1226 vertices, min-fill width ≤ 6 each** (19 of them ≤ 2). Component sizes
  161, 145, 126, 124, 120, 104, 65, 54, 50, 46, 5×35, 21, 9, 6, 2×5, 4, 3, 3×1;
  per-vertex external degree to glue ≤ 25 (mostly ≤ 4).

## 2. The certificate (disjoint-clique-packing lower bound)

**Theorem.** Let K₁,…,K_m be vertex-disjoint cliques of G. Any solution with torso width
≤ w satisfies t ≥ Σᵢ max(0, |Kᵢ| − w − 1).

*Proof.* Fill edges are only added, never removed, so the first vertex of Kᵢ eliminated at
a torso step still has all later Kᵢ-members as neighbours: if the head contains jᵢ vertices
of Kᵢ, that step has width ≥ |Kᵢ| − jᵢ − 1 ≤ w, so jᵢ ≥ |Kᵢ| − w − 1. Cliques disjoint ⇒ sum. ∎

With {500, 400, 300}: **t(w) ≥ (499−w)₊ + (399−w)₊ + (299−w)₊.**

## 3. What the certificate proves about the current front

| w | env t | bound | slack |
|---:|---:|---:|---:|
| 195 | 702 | 612 | 90 |
| 219 | 597 | 540 | 57 |
| 244 | 496 | 465 | 31 |
| 274 | 390 | 375 | 15 |
| **299** | **300** | **300** | **0** |
| **332** | **234** | **234** | **0** |
| **365** | **168** | **168** | **0** |
| **399** | **100** | **100** | **0** |
| **449** | **50** | **50** | **0** |
| **499** | **0** | **0** | **0** |

**Six of the 20 submitted points are provably Pareto-optimal.** The two-sided
characterisation achieved on small now extends to the entire top third of large's front.
Every arm currently spending accepts at w ≥ 299 (including twin_construct — its 0 accepts
are now *explained*: it was attacking a certified-optimal region) is burning compute on a
mathematically closed region. The optimal front in [195, 499] is the piecewise diagonal
t = (499−w)₊+(399−w)₊+(299−w)₊ with slopes 3/2/1 — so the optimal capped-20 spacing there
is a closed-form geometry problem (zone loss on slope s over gap g is ≈ s·g²/2; equalise
s·g²), not a search problem.

## 4. Where the missing 16.5k actually lives (levers, quantified)

Baseline cap-20: −5,476,521 (gap +16,541). Cap cost 14,944, envelope residual only +1,597
— the winner's 20 points ≈ our 500-point envelope. Their front has *big clean steps*; ours
smears the same area across widths the cap truncates.

- **Lever A — attain the clique bound on w ∈ [180, 299).** Certified slack 15–90 vertices
  per width. Recomputed cap-20 if attained: **−5,482,832 (+6,311 HV)**.
  Construction: head = interleaved clique prefixes — (499−w) from K₅₀₀, (399−w) from K₄₀₀,
  (299−w) from K₃₀₀. Twins first (their elimination inside a clique is literally
  **zero-fill**); beyond the 102 twins of K₄₀₀ / 62 of K₃₀₀ the head must take clique
  *externals*, whose outside edges cause fill — that is exactly where the 15–90 slack sits
  and exactly the ranking problem GBDT should own (§6).
- **Lever B — shift the mid-range unlock widths left (w ∈ [82, 180)).** The envelope's big
  steps (torso +179 at w=82, +487 at w=99, +113 at 130, +129 at 174) are glue-interface
  unlocks of the 1226-vertex planted torso. A 15-width left-shift of this region alone is
  worth **+18,259 HV** (cap-20 −5,494,780 → beats leaderboard by 1,718). Even partial
  shifts dominate everything else available. The structural question per width: which
  glue-external / attachment vertices to admit into the torso so the planted components'
  fill stays under w — again a per-vertex ranking problem.
- **A+B combined: −5,499,793, i.e. leaderboard +6,731.**

## 5. Immediate zero-compute actions

1. **Merge Mac ↔ server pools.** Snapshots differ (large −5,476,521 vs −5,476,340; medium
   −1,739,391 vs −1,739,237). Rsync `submissions/*/` both ways, rerun cap_submit: free HV.
2. **Kill/retarget all arms proposing moves at w ≥ 299 on large.** Certified closed.
3. **Relabel gbfcpp's log metric.** `best = -arch.hypervolume(n)` is the *uncapped
   envelope*; its "gap +3,116" is the unlimited-points residual, not the submittable gap
   (+5,731). Two different quantities — the campaign log should not mix them.
4. **Feed `gbfcpp --only-widths` from the slack table, not from HSSP-of-current-front:**
   attack 180–298 (lever A) and the unlock shoulders 80–175 (lever B); never 299+.

## 6. The GBDT-centric thesis narrative (goals 1 & 2)

The GBDT story is no longer "GBDT approximates what LNS does": it is **GBDT solves the
one subproblem the certificate leaves open**.

- The certificate reduces w ∈ [180, 299) to: *choose which (399−w)−102 externals of K₄₀₀
  and (299−w)−62 of K₃₀₀ go into the head, and in what order*. Features per external:
  degree outside the clique, #attachments to each planted component, overlap with other
  chosen heads, fill-potential (pairs of non-adjacent outside-neighbours). Label: realised
  torso width / accept. This is a **learning-to-rank problem tailor-made for LightGBM**,
  trained on the archive you already have — and it directly measures "GBDT elevates the
  solution" because random/degree-descending insertion is the natural ablation baseline.
- In lever B, GBDT ranks glue-attachment vertices for admission order into the 1226-vertex
  planted torso — same features, same ablation design.
- Speed claim: the certificate deletes ≥ 40% of the width range from the search space
  (proven-optimal region + closed-form selection). GBDT-guided construction at slack
  widths starts from the clique-prefix construction, not from random orderings — report
  time-to-envelope-parity vs cuda-torso GPU-hours for the "faster than leaderboard" claim.

Novelty stack for the thesis: (i) exact reverse-engineering of a planted competition
instance, confirmed by the designer; (ii) a clique-packing optimality certificate proving
6/20 submitted points optimal — leaderboard entries never proved anything; (iii) exact
quantification of cap-20 geometry (14,944 cap cost decomposed into zone losses);
(iv) GBDT-as-ranker inside a certificate-reduced search space, with clean ablations.

## 7. Medium: same methodology, different structure (open)

Medium's twin classes are K₆'s (bound trivial beyond w≈5) — the clique certificate does
not transfer. Its glue is sparse (max core 38; whole graph eliminable at w=240) and the
planted components hide inside a 1064-vertex block (823 vertices of core exactly 9 — the
planted tw is likely 9, matching the front's first selected width w=9). The medium
analogue of §2 is a per-width lower bound on the 882-supervertex twin quotient via your
existing branch-and-bound/SAT tools at the 20 selected widths (all ≤ 240, small
subproblems). Cap cost there is only 2,560, residual 3,171 — medium is a torso-quality
problem at low widths, so certificates that *close* widths are worth as much as search.
