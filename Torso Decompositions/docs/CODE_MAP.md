# Code map — logical organization

*Physical paths are frozen while the campaign runs (10 live arms import from
them, and THESIS.md's reproducibility section cites them). This map organizes
the codebase logically into three tiers; the physical move to matching folders
is scheduled for post-freeze (see note at bottom).*

## 1. Permutation-space chapter: hill-climbing variants & classical metaheuristics

Historical evidence base (§2, §12 ledger). Not part of the live solving path.

- `algorithms/hillclimbing/` — the 15 HC variants (hc1–hc15)
- `algorithms/metaheuristics/` — SA, GRASP, VNS, Tabu, ILS, ACO, AMOSA-style MO-SA
- `algorithms/population/` — NSGA-II / SMS-EMOA experiments
- `extra_instances/` — convergence CSVs and plots for this chapter
- `submissions/*/hc*.json, sa.json, vns.json, grasp.json` — their banked fronts

## 2. Other algorithms & attempts (continuous paradigm, exact methods, probes)

The middle chapters: the paradigm shift, exact certification, and honest
negatives.

- `algorithms/continuous/cmaes_torso.py` — sep-CMA-ES spectral-policy decode (§4–5)
- `algorithms/continuous/gpu_eval.py` — GPU batch evaluator (§9)
- `leaderboard_reference/` — cuda-torso study copy (+ our int32 fix,
  `run_capfocus.py` cap-focused breeding)
- `tools/front_to_checkpoint.py` — warm-start builder for cuda-torso
- Exact/certification suite (§13): `tools/tw_le`-family B&B, `tools/sat_torso.py`,
  `tools/pid_torso.py` (Tamaki), `tools/band_climb.py`, `tools/clique_break.py`,
  `tools/torso_del.py`
- Probes & negatives: `tools/landscape_gbdt.py` (§13.7–13.8),
  `tools/spectral_crack.py`, `tools/spectral_seed.py`, `tools/beam_decode.py`,
  `tools/gbdt_moves.py`, `tools/twin_construct.py` (§14.2 negative)

## 3. GBDT-focused & the live solving path (the thesis spine)

- `core.py` — evaluator, ParetoArchive, exact 2-D HSSP DP, HV (test-pinned, §7)
- `tools/fastwalk.py` — C incremental evaluator (IncEvalC)
- `algorithms/continuous/gbdt_torso.py` — all four GBDT integrations incl. the
  adaptive self-improving policy (`--cap-aware --rank`, §6b/§14)
- `tools/gaps_search.py` — GAPS nonlinear GBDT decode (§10)
- `tools/gbfc.py`, `tools/gbfcpp.py` — GBFC/GBFC++ boosted front construction
  (§11; `--cap20` cap-aware mode)
- Cap-aware suite (§13.8a–14.4): `tools/cap_submit.py` (exact-DP submission +
  platform format), `tools/archive_evolve.py` (capped-archive evolution;
  `--twins` symmetry operator), `tools/quota_repair.py` (class-quota law),
  `tools/hri_lns.py` (winners' MO-LNS under the capped objective)
- `tools/verify_submission.py`, `tools/portfolio.py`, `tests/` — scoring truth

## Post-freeze physical reorganization (planned)

Target layout: `algorithms/{1_hillclimbing_variants, 2_other_attempts,
3_gbdt_core}/` with import-path updates, a compatibility shim, and a full
test-suite pass — executed on a branch after the campaign freezes, never on
live arms.
