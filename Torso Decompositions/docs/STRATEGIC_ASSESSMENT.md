# Strategic assessment — external review vs. measured evidence

A second reviewer did a structural pass of the repo and proposed an architectural
overhaul (learned sequential/beam decoder, LNS, GFlowNets, diffusion-over-permutations,
and "optimize the front/archive directly instead of single orderings"). The review is
high-quality and its *diagnosis of the field* is correct: standard metaheuristics are
saturated, small is solved, the opportunity is medium/large. This note records, point by
point, how those proposals hold up against what we have actually **measured** — because
several of its highest-confidence claims are already done, or are empirically ruled out
as the bottleneck.

## The decisive measurement: the gap is torso quality, not representation

The reviewer's central thesis is "you optimize orderings individually and discard the
union; the real object is HV(front), so change the representation." We tested this
directly (`tools/cap_submit.py`):

- Our **full per-width envelope** for medium pools the per-band best across *all*
  orderings — 249 non-dominated points — and scores **≈ −1,733,000** with **no cap**.
- The leaderboard target is **−1,745,122**.
- So **even an unlimited-size submission of our entire corpus is ≈ +12,100 HV short.**

This is conclusive. The residual gap is **not** packing, **not** point-selection, **not**
the 20-point cap (which costs a further ≈ 3,600 HV on top), and **not** "optimizing
orderings individually." We already pool the union optimally (see next section). The
leaderboard fronts simply contain **larger torsos at the decisive widths** — their 20
points dominate our 249. The bottleneck is the NP-hard core: *finding bigger
decompositions*, i.e. search power / compute, not the representation or the objective.

## Where the reviewer's premises are already satisfied

- **"Optimize the front/archive directly, not single orderings" / "represent an
  individual as N orderings, fitness = HV(union)."** This is exactly what
  `core.ParetoArchive` does: it keeps the per-(width) best torso across *every* ordering
  ever evaluated (the union), and GBFC++ targets individual breakpoints. The reviewer's
  A∪B example is the archive's everyday behavior.
- **Submission is HV-optimal already.** Selecting the ≤20 points that maximize HV is the
  Hypervolume Subset Selection Problem; we solve it with the **exact 2-D DP**
  (`top_k_by_hv_contribution`, O(K·m²)), not greedily. There is no HV to recover by
  smarter selection (verified: greedy local search improves it by +1 HV).
- **"GBDT as a proposal policy, not a score."** GBFC++ and `gbdt_torso` construct-mode
  already train the GBDT to *propose moves / positions*; the dynamic features
  (`elim_nbr`, `cur_deg`, `fill`) make it a semi-sequential policy on the live graph.
- **The objective is reverse-engineered.** HV = area of the union of bottom-right
  rectangles vs reference (n,n), capped at 20 points (THESIS §1, §13.8a). Torso width is
  **order-independent**, so the front decomposes into *independent* max-torso-per-width
  problems (THESIS §13). These are the exploitable invariants — and they are already used.

## Per-proposal verdict (evidence-based)

| Proposal | Verdict |
|---|---|
| Optimize the front / archive directly (★★★★★ for the reviewer) | **Already done** (ParetoArchive union + exact HSSP). Cannot close the gap — proven above. |
| GBDT predicts move, not score | **Already done** (GBFC++/construct). Extendable, but not where the 12k lives. |
| Beam-search / learned sequential decoder | **Built and tested in BOTH forms** (`tools/beam_decode.py`): min-degree proposer *and* a GBDT policy trained on the HV-weighted corpus elites (`--no-policy` ablates it). Neither beats the trained corpus on medium (no HV gain across multiple decodes at B=10–12). The beam reproduces corpus-quality orderings but cannot exceed them — consistent with the torso-quality proof: beating the corpus needs *fundamentally bigger torsos*, not better orderings of the same quality. The decoder is a legitimate method (and an honest negative result) for the thesis; it is not a leaderboard lever. |

## Engine/config verification (the "are we even running the winner right?" check)

The winning `leaderboard_reference/cuda-torso-main/run.py` defaults are: 32 eigenvectors,
init/mutation stdev 0.3, mutation_proba 0.5, cosyne_proba 0.2, batch 1024. Our running
GPU arm (`run_gbdt.py`) uses these same defaults **plus** the additive GBDT booster
(converged engine-level Δ ≈ +1,816). So we are running the **proven winning engine at its
proven configuration**, not a handicapped variant. The only remaining difference from the
leaderboard run is **generations / wall-clock compute** — i.e. time on the GPU.

## Final verdict (four architectural levers + two proofs)

Spectral seeding (+17 HV), cap-aware selection (already HSSP-optimal), min-degree beam (no
gain) and policy beam (no gain) have all been built and **measured** — none breaks the
wall. The full-envelope proof (+12 k short with unlimited points) and the
engine/config check together establish, not assert, that the medium/large gap is
**torso quality reachable only by more GPU compute on the winning engine.** There is no
architectural or representational fix left to find; continuing to look for one would be
ignoring the evidence. The thesis contribution here is the *rigorous exclusion* itself —
a map of where the gap is **not** — plus the cap-aware and landscape-diagnosis methods.
| LNS (destroy/repair) | Partially present (set-space `torso_deletion`, `crossover_relinking`); worth extending per-width. |
| GFlowNet / diffusion-over-permutations | Research moonshots: months of work, uncertain payoff, GPU-heavy. Only as a thesis research swing, not a leaderboard play. |

## What the evidence says to actually do

The gap is torso quality at the ~20 widths that form the optimal submission
(`cap_submit.py` prints them + their marginal HV value). Order-independence (§13) means
each is an **independent** "largest torso of treewidth ≤ w" problem. So the first-
principles direction is **not** a new whole-front representation (we have the optimal one)
— it is:

1. **Cap-aware search** (done: `gbfcpp --cap20`) so 100% of compute targets leaderboard-
   visible widths instead of ~90% landing on truncated bands.
2. **Per-width max-torso attack** exploiting order-independence: dedicated effort per
   visible width, exact where tractable (low w), heuristic/GPU where not (high w).
3. **Beam-search decoder** as the one architectural bet — GBDT proposes candidates, the
   incremental C kernel prunes — to escape `argsort` brittleness and find bigger torsos.
4. **Scale** the cuda-torso GPU arm (the winning engine) — the honest, boring lever the
   measurement keeps pointing back to.

## Honest bottom line

The reviewer hoped a deep audit would find a "missing trick" in the objective's
mathematics. We did that audit: the tricks (order-independence, the HV decomposition, the
exact HSSP submission) are **found and already exploited**. The math now *proves* the
remaining medium/large gap is bigger torsos, not a representational or objective-level
oversight — a gap that persists even with unlimited submission points. That is a
search-power/compute problem. Beam-search decoding is the one architectural idea with a
real (if uncertain) shot at it; everything else is either already in the codebase or a
multi-month moonshot. The thesis is novel and honest as it stands; a leaderboard win, if
it comes, comes from bigger decompositions at ~20 widths — nothing else.

*(Companion tools: `tools/cap_submit.py`, `tools/landscape_gbdt.py`; THESIS §13.7, §13.8,
§13.8a.)*


---

## Update 2026-07-01 — the large-graph campaign (reverse-engineering payoff)

Three findings that redirected the endgame:

1. **cap_submit on large (never measured before): gap +28,574, of which 16,013 HV
   is cap cost — 4x medium's.** Large is the least-squeezed instance and its
   torso-quality residual (~12.5k) is ~0.5% per visible width vs medium's ~5%:
   proportionally 10x closer to target.
2. **int32 overflow in the reference kernel (libeval.cu, `int adj_offset = idx*N*N`):
   crashes large-graph at batch > 364. Medium fit inside int32 by 7% luck.
   Fixed (size_t) — the reference engine now runs large at batch 1024 (3x the
   evaluation width the winner's public engine was capable of).**
3. **Uniform elite breeding (`elite_range = ones(N)/B`): the winner's engine puts
   0.8% of its selection pressure on the ~20 sizes the capped submission keeps.
   run_capfocus concentrates 90% of breeding mass on the HSSP-optimal sizes —
   cap-aware optimisation applied inside the generator for the first time.**

Deployment: server = run_gbdt (warm) + run_capfocus (warm) + gbfcpp_cap20 on large,
medium cuda hedge; Mac = 2x archive_evolve + gbfcpp_cap20 on large, medium hedge.
Warm-start via front_to_checkpoint (claimed fitnesses = corpus per-position widths →
elite gate = "beat the corpus"; engine's own early submissions are decode-degraded
and are filtered by per-band-max pooling). First hours: gap +28,533 → +28,249 with
envelope growth (crossover finding genuinely new torso points).

---

## Update 2026-07-02 (evening) — HRI recipe implemented; audit

**Limmer (p.c., cited with permission) disclosed the winning MO-LNS:** vertex-level
destroy/repair; set B = neighbor destroy (vertex + original-graph neighbours, large
size) + balanced repair (MEDIAN PLACEMENT, Biedl et al. DAM 148 (2005) Sec. 5 —
paper obtained and read; our implementation is faithful incl. the odd-k side rule
of Lemma 16); set A = small random destroy + random repair; B until stall, then A.
Also: Spacekangaroos' large-graph result exploited planted graph structure —
independently confirming our twin-class discovery.

**Ours differs deliberately in one place:** acceptance is the exact capped-20 HSSP
hypervolume (the scored objective), not a full-front criterion.

**First-hours evidence:** hri_lns large 5 accepts/3k iters (all set B); medium 19
accepts/6.5k iters (-1,733,855 -> -1,734,129). Twins ablation holds (~1.7x accept
rate). Quota law converted (+3 accepts, owner-8 375-class moves). Live: large
best-20 -5,467,796 (gap +25,266, from +28,574 at pivot); medium gap ~+11,0xx and
falling fast under LNS.

**Known implementation freedoms vs. the p.c. (acceptable, monitored):** single-seed
neighbor destroy (theirs possibly multi-seed); stall-toggle B<->A (theirs possibly
one-way); insertion ordering = degree-descending (unspecified in p.c.).
