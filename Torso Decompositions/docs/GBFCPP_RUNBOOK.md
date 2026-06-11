# GBFC++ runbook — continuing the small-graph push to #1

State as of 2026-06-10: verified small-graph **−1,829,735** (99.990 % of the
leaderboard top −1,829,919; **184 HV short**). The run below resumes from the
checkpoint in `submissions/small-graph/gbfcpp.json` automatically.

## 1. One-line continuation (CPU, overnight)

**Preferred: the parallel swarm** — one GBFC++ worker per core, sharing
progress through `portfolio.json` between waves:

```bash
cd "Torso Decompositions"
pip3 install lightgbm            # stronger weak learner than the numpy fallback
python3 tools/fastwalk.py small-graph   # builds + self-tests the C kernel
caffeinate -i python3 tools/gbfcpp_swarm.py --problem small-graph \
    --workers 6 --waves 60 --rounds 6 --round-budget 60
```

(~6 h wall, ~36 core-hours. Ctrl-C safe; rerun resumes. `caffeinate -i`
prevents macOS sleep — locking the screen is fine.)

Single-worker alternative:

```bash
python3 tools/gbfcpp.py --problem small-graph --rounds 500 --round-budget 60 --seed 1
```

The phase-B search uses annealed (Metropolis) acceptance with kick
perturbations on stall and block-relocation moves — added after the strict
descent variant plateaued at gap +355.

**GPU note:** GBFC++ is CPU-only by design (C walk kernel + LightGBM); running
it on a GPU machine uses the CPU. The GPU experiment is §3 (qnegbfc).

- Checkpoints the submission **every round** and persists the breakpoint
  failure decay (`submissions/small-graph/.gbfcpp_state.json`) — safe to kill
  and restart at any time, and safe to run several times with different seeds.
- Expect strongly diminishing but nonzero per-round gains; in the sandbox runs
  the gains had not flattened (≈ +8 HV/35 s round near the end, CPU-only,
  numpy backend). LightGBM + 60 s rounds should do meaningfully better.
- Re-score and fold into the canonical bank when done:

```bash
python3 tools/portfolio.py --problems small-graph
python3 tools/verify_submission.py submissions/small-graph/portfolio.json
```

## 2. The rigorous ablation (for the thesis)

Paired same-seed rounds from a frozen pool. **Two critical details learned in
the sandbox (THESIS §12.3):** (i) `portfolio.json` now contains GBFC++ points,
so it must be excluded too, or the "frozen plateau" start is contaminated;
(ii) the pairs only discriminate in the *productive* regime — run them with a
cool temperature (`--t0 0.1`, ≈ the strict-descent decoder the 3/3 pilot used)
and use MULTI-ROUND arms (3 × 60 s) so most seeds land at least one productive
zone; single rounds from a saturated pool tie at +0 in both arms.

```bash
for s in 1 2 3 4 5 6 7 8 9 10; do
  python3 tools/gbfcpp.py --problem small-graph --rounds 3 --round-budget 60 \
      --seed $s --t0 0.1 --algo abl_gbdt --exclude-stems gbfcpp,abl,portfolio
  rm -f submissions/small-graph/.abl_gbdt_state.json submissions/small-graph/abl_gbdt.json
  python3 tools/gbfcpp.py --problem small-graph --rounds 3 --round-budget 60 \
      --seed $s --t0 0.1 --no-gbdt --algo abl_nogbdt --exclude-stems gbfcpp,abl,portfolio
  rm -f submissions/small-graph/.abl_nogbdt_state.json submissions/small-graph/abl_nogbdt.json
done
```

Record the total HV delta per arm per seed; report mean ± std and the paired
sign test (the pilot was 3/3 for the GBDT arm, +44.7 vs +18.7 per round).
Delete the `abl_*` files afterwards so they do not pollute `portfolio.py`.

## 3. GPU week: the two scale-ups

1. **GBFC++ as-is benefits from nothing GPU-specific** — it is breakpoint LS —
   but medium/large rounds get the C kernel speedup too:
   `python3 tools/gbfcpp.py --problem large-graph --rounds 100 --round-budget 120`.
   (Gaps there are 32k/61k HV; GBFC++ targets staircase cells, so expect
   smaller relative progress than on small.)
2. **qnegbfc hybrid (THESIS §11.4)** — the designed path to match the winner's
   compute on small/medium: GBFC/GBFC++ GBDT specialists injected into the
   per-threshold neuroevolution at GPU scale, with the `--no-gbdt` control:
   `python3 tools/qnegbfc.py --problem small-graph ...` on the T4/Colab setup
   of THESIS §9.

## 4. What was added in this iteration (code map)

| File | What |
|---|---|
| `tools/gbfcpp.py` | GBFC++: breakpoint-residual boosting, GBDT move-proposal policy, boundary scan, compound repair, path relinking, annealed acceptance + ILS kicks + block moves; resumable |
| `tools/gbfcpp_swarm.py` | parallel driver: N workers/wave, merged via portfolio.py between waves |
| `tools/_fastwalk.c` + `tools/fastwalk.py` | C elimination-walk kernel (ctypes), bit-exact vs `core.evaluate`, ~25× the Python walk; auto-built, python fallback |
| `algorithms/continuous/np_gbdt.py` | dependency-free numpy histogram GBDT; in `make_gbdt`'s auto chain before ridge |
| `docs/THESIS.md` §11.6 | method, verified result, ablation |
| `submissions/small-graph/gbfcpp.json` | the verified −1,829,735 front (16 vectors) |
