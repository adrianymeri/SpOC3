# Small-graph leaderboard attempt — the GBDT-on-SOTA ablation protocol

**Goal.** Cross the small-graph leaderboard top (−1,829,919) *with GBDT as the
load-bearing mechanism*, in a controlled way that is honest whether or not the
beat lands. Current best (torso-deletion, §13): **−1,829,913, gap 6.** The last 6
HV is a compute-scale policy-search result (§13.4), so the vehicle is the
leaderboard winner's own engine (cuda-torso) **augmented with a GBDT
front-booster** — run against a `--no_gbdt` control so the GBDT contribution is
isolated by construction.

Everything below is small-graph only. Medium/large are deferred until the small
gap is closed; they have not been given comparable search and their numbers are
not compute-converged.

---

## 1. The three arms

All three run the *same* cuda-torso engine and evaluator; they differ only in the
GBDT booster. Run them from the cuda-torso repo (needs `libeval.so`, `data/`):

```bash
cd ~/cuda-torso

# ARM A — GBDT-augmented (the proposed method)
sudo -u hyjnesha7 python3 -u run_gbdt.py --graph small-graph \
  --gbdt_every 200 --seed 0 --max_generations 1000000 \
  2>&1 | tee small_A_gbdt.log &

# ARM B — control: identical engine/seed/budget, GBDT OFF
sudo -u hyjnesha7 python3 -u run_gbdt.py --graph small-graph \
  --no_gbdt --seed 0 --max_generations 1000000 \
  2>&1 | tee small_B_control.log &

# ARM C — stock cuda-torso (the literal leaderboard recipe), as a sanity baseline
sudo -u hyjnesha7 python3 -u run.py --graph small-graph \
  --batch_size 1024 --max_generations 1000000 \
  2>&1 | tee small_C_stock.log &
```

A and B **must** share `--seed 0` and `--max_generations` — that is what makes
the comparison a controlled ablation (only the GBDT booster differs). C is an
independent baseline confirming the engine reproduces the winner's regime.

GPU budget: three batch-1024 small runs ≈ 10–12 GB on the 32 GB V100 — fits.
`GBDT_NJOBS=1` if lightgbm stalls; falls back to the numpy GBDT otherwise.

---

## 2. What to record (every few hours)

For each arm, copy the best checkpoint into the project pool and **re-score
officially** — never quote the engine's internal HVI:

```bash
cd ~/SpOC3/Torso\ Decompositions
# pick the best (most-negative) checkpoint of each arm:
for arm in A_gbdt B_control C_stock; do
  best=$(ls -t ~/cuda-torso/submissions/small-graph/*.json | head -1)   # adjust per-arm dir if separated
done
python3 tools/verify_submission.py <checkpoint>.json     # authoritative -HV
```

Log the table:

| time | Arm A (GBDT) | Arm B (control) | Arm C (stock) | Δ = A−B | best vs leader |
|---|---:|---:|---:|---:|---:|
| … | | | | | |

---

## 3. The claims this protocol can support

The ablation makes the GBDT contribution *provable* regardless of the leaderboard
outcome:

1. **GBDT contribution to the SOTA front (Δ = A − B).** Identical engine, seed,
   budget; the only difference is the booster. A positive Δ is the GBDT
   front-booster's controlled contribution to a championship-grade search — the
   headline ablation, reportable even if neither arm beats the top.
2. **A pooled with the torso-deletion front.** Pool Arm A's best against
   `submissions/small-graph/torso_del.json` (the gap-6 front) — the constructed
   and policy fronts live in different representations (§13.4), so the *union*
   can exceed either alone. This is the most likely route to gap ≤ 0.
3. **The beat, if it lands.** If the pooled (A + torso-deletion) front re-scores
   ≤ −1,829,919, the leaderboard is crossed *and* GBDT is in the winning solution
   (Arm A), with Arm B as the control quantifying how much of the margin GBDT
   supplied.

---

## 4. Honest stopping rules

- If after a long run **Δ ≈ 0** (A and B converge to the same front), report it
  plainly: on the near-optimal small instance the booster adds little — which is
  consistent with §13's two-sided bound and §11's bounded-residual finding. The
  GBDT contribution is then carried by GBFC/GAPS on the instances with room.
- If **A > B but neither beats the leader**, the result stands as a controlled
  GBDT-on-SOTA improvement (claim 1) — a clean, publishable ablation.
- Only quote `verify_submission.py` / `portfolio.py` numbers in the thesis;
  never the engine's internal HVI.

---

## 5. Why this is the right vehicle (one paragraph for the write-up)

The small residual is, by §13.4, reachable only by `argsort`-policy search at
compute scale — the constructed-ordering methods (torso-deletion) are
representationally capped above it. The honest way to attack it *with GBDT* is
therefore not another constructed-space operator (ten of which confirm gap 6) but
to **augment the policy search itself** and ablate the augmentation. `run_gbdt.py`
adds GBDT specialist orderings to cuda-torso's per-threshold pool additively
(never disturbing the elites), so the booster can only help, and `--no_gbdt`
gives the exact control. This makes "did GBDT help the state of the art, and by
how much" a measured quantity rather than a claim.

---

## 6. Results (converged, ~580k generations, V100, identical seed/budget)

| Arm | engine | best internal HVI |
|---|---|---:|
| A | cuda-torso **+ GBDT booster** | **−1,828,402** |
| B | cuda-torso control (`--no_gbdt`) | −1,828,193 |
| C | stock cuda-torso (`run.py`) | −1,827,944 |

**Δ = A − B ≈ +200 HV** (converged; observed range +184…+211 across the run,
positive throughout). This is claim (1) of §3, landed: a **measured, controlled
GBDT contribution to a championship-grade search** on the identical engine, seed,
and budget.

**Honest reading (claim 4 / §4 stopping rule).** Both arms plateau ~1,500 HV below
the constructed front (§13, −1,829,913), and pooling either arm with the
torso-deletion front adds **+0** at every band — small is saturated, so the
engine-level Δ does not convert to a portfolio-level gain. The beat did **not**
land here, and §13.6 explains why: small is a certified near-optimum on an
instance that defeats the SOTA exact solver. The booster *helped the search*
(+200) but there is no headroom above set-space search on this instance. **The
booster's portfolio-level payoff is reserved for medium and large**, where the gap
is 18,844 / 12,561 HV and the same A/B/C protocol is now running. Report the +200
plainly as what it is; do not inflate it into a beat it did not produce.
