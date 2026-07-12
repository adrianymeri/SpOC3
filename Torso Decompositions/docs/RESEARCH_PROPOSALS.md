# Research memo — three proposals beyond the current fleet (2026-07-12)

*Written while the compute runs. Context: THESIS §13 (set-space attack, small),
§15 (certificates), §15.6/15.8 (GBDT boundary, validity). Ranked by expected
value / effort. Each is novel relative to the 2024 leaderboard methods (HRI
permutation-LNS; cuda-torso continuous-encoding neuroevolution) and to our own
running arms.*

## P1. CQS: Certificate-Quota Set-space search (the mid-band weapon)

**Observation (classical, but unexploited here).** Eliminating a vertex set X
yields the same fill on V∖X in every order: the torso graph G★(S) — S plus
edges between S-vertices connected through the head — depends only on the SET
S. Torso width = width of eliminating G★(S), head validity = a property of
V∖S. The permutation encoding used by every 2024 entrant (and all our arms
except §13's torso_deletion) is a redundant n!-sized cover of a 2^n space.

**What makes it tractable now: the certificate fixes the skeleton.** At open
width w, |S ∩ K_i| ≤ w+1 per glue clique (packing bound) — and at the bound
these are equalities. So S is parametrized ONLY by (a) WHICH R_i members of
each clique survive, (b) WHICH component vertices are sacrificed. The five
construction failures (§15.6 follow-up) all came from committing to an
ORDER; CQS never orders anything during search:

  - state: S (as three clique-subsets + a component-sacrifice set);
  - move: swap external <-> external within a clique; sacrifice/rescue a
    component vertex; k-swaps via LNS;
  - evaluation: build G★(S) by contracting the head (BFS through V∖S,
    O(m) with bitsets), then min-fill width on G★(S); exact staircase via
    the existing C evaluator on [head-any-order + min-fill-suffix] only for
    candidates that pass;
  - acceptance: |S| at fixed w (independent per width), pooled into the
    capped-20 archive as usual.

**Where GBDT enters (and why it can win here after losing in §15.6).** The
§15.6 ranker failed on an order-sensitive target with one instance of labels.
CQS's target is order-free and every evaluation is a labeled (features(S),
width) pair generated at ~100/s: train a GBDT surrogate of width(G★(S)),
screen thousands of candidate swaps per exact evaluation (batch predict,
verify top-k exactly). GBDT as *screening oracle over a set-function* — not a
selector, not a decoder — is a genuinely different integration from all four
in §6b and, to our knowledge, from the neuroevolution/LNS state of the art on
this problem family.

**Expected value.** The mid-band residual (~3.4k) + its share of cap cost is
exactly where permutation moves stall (accept rates of 1e-4). Set moves with
quota-fixed skeletons search the *right* neighborhood. Effort: ~2 days
(contraction evaluator + minfill + LNS loop + surrogate hook). Risk: min-fill
on G★(S) may under-estimate achievable width — mitigated by exact re-scoring
of accepted points (already standard in the pipeline).

## P2. Bandit-scheduled width portfolio (cheap, immediate, publishable)

With 6/20 points certified, the score is a weighted sum of ~14 independent
max-torso problems with KNOWN marginal values (zone widths from the HSSP
geometry) and MEASURABLE per-width accept rates (already in the gbfcpp/LNS
logs). Current allocation is uniform — provably suboptimal. Proposal: a UCB
scheduler that allocates each arm-round to width w maximizing
(zone_value_w × estimated_accept_rate_w + exploration bonus), re-estimated
from the live logs. GBDT variant: predict accept probability from width
features (slack-to-bound, zone width, recent-accept recency) — "learning
where to search," the same paradigm as GBFC++ one level up. Effort: half a
day (a wrapper that rewrites --only-widths weights between gbfcpp rounds).
This also yields a clean thesis figure: HV-per-cpu-hour, uniform vs bandit.

## P3. Family transfer: learn the CONSTRUCTION, not the instance

We hold the generator recipe (Bannach p.c.): disjoint low-tw graphs + dense
glue. Manufacture hundreds of family instances small enough to solve near-
optimally (our own exact/BB tools + overnight LNS), then train GBDT policies
(GBFC++ features + §15 structure features: clique membership, quota residual,
attachment counts) ACROSS instances. Deploy on the competition instances:
(a) as gbfcpp's proposal model (replacing per-round self-training);
(b) as an INFORMED basin generator for the small-graph lottery — sample
initial orderings from the family policy + temperature noise instead of
uniform random. §6b.5 already showed cross-density generalization is the
GBDT integration that transfers; this closes the loop scientifically: if the
policy transfers within the family, GBDT has learned the generator's
structure — the strongest possible form of the thesis claim. Effort: 1-2 days
(generator exists: tools/generate_instances.py; training harness exists:
gbdt_torso/gbfcpp). Risk: family mismatch (unknown generator parameters) —
mitigated by fitting parameters to the three known instances' statistics
(clique sizes, component tw profile, attachment distributions — all measured
in §15.1/§6d).

## Recommendation

P2 tonight (near-zero cost, immediate optimality of compute allocation).
P1 as the flagship build — it is the §13 method that already produced the
small-graph breakthrough, generalized by §15's certificates, with a GBDT
integration that answers §15.6's negative. P3 as the thesis's closing
chapter: the model that learned the construction. All three are additive to
the running fleet; none requires stopping anything.
