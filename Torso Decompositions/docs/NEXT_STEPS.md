# Next Steps — the plan from here

*Status snapshot and the concrete path forward. Written after small-graph was
certified at gap 6 and the medium-graph campaign began showing a large GBDT
contribution.*

---

## 1. Where we stand

| Instance | Best (−HV) | Target | Gap | Status |
|---|---:|---:|---:|---|
| **small** | **−1,829,913** | −1,829,919 | **6** (0.0003 %) | **Certified & written up** |
| medium | climbing (~−1,709k live) | −1,745,122 | ~36k (early) | **Active GPU campaign** |
| large | ~−5,480k pooled | −5,493,062 | ~12.5k | Queued after medium |

**Small is done.** −1,829,913 is verified against ESA's own UDP (byte-identical
instance, evaluator matching `_perm2fitness` exactly, HV matching `combine_scores`
to the unit). Bands 0–7 are *proven optimal* by exact branch-and-bound; bands 8–14
were exhausted by ~2.4 M exact set-space restructures and 6.5 M exact ordering-space
moves; and the instance **defeats Tamaki's PID** (PACE-2017 exact-treewidth
champion did not terminate in 10.8 h). This is a *certified near-optimum on an
exact-intractable instance* — a stronger result than a bare leaderboard number.
The write-up (§13, §13.6, §12.1a) and the rebuilt 40-page PDF reflect this.

---

## 2. The two novel contributions (the thesis spine)

1. **GBDT front-boosting (GBFC §11, GAPS §10).** Gradient-boosted weak learners
   fit to the front residual, decoded into complementary specialist orderings.
   *This is where the headroom instances pay off* — see §3.
2. **Set-space torso-deletion (§13).** Exploiting torso order-independence to
   search vertex *sets* under an exact width check; drove small to gap 6 and
   yielded the closed-form HV decomposition + two-sided certificate.

The honest GBDT story by instance, which medium/large are now sharpening:
- small Δ ≈ **+200** (engine-level, §12.1a) — small because the instance is
  saturated; portfolio-level +0.
- medium Δ = **+8,747 and climbing** (live) — the headroom payoff.
- large GAPS Δ = **+24,766** (4.6σ, §10.2) — the dense-instance showcase.

---

## 3. MEDIUM — the active front (do this now)

The three cuda-torso arms are running on the server (V100), warm with real
headroom. Playbook:

**3a. Let the GPU arms climb.** Arms A (GBDT) / B (control) / C (stock) on
`medium-graph`. Watch with:
```bash
cd ~/SpOC3/Torso\ Decompositions && ./tools/ablation_watch.sh medium
```
Track two things: **Δ = A − B** (the novelty metric — already +8,747) and **Arm
A's `official`** (climbing toward −1,745,122).

**3b. Trigger Layer 2 when Arm A passes ~−1,726,000** (the old pooled front).
Then pool the GPU checkpoints and fire the constructed refiners on top:
```bash
cp ~/cuda-torso/submissions/medium-graph/*.json submissions/medium-graph/
python3 tools/gbfcpp.py        --problem medium-graph --budget 28800 &
python3 tools/gaps_search.py   --problem medium-graph --budget 28800 &
python3 tools/torso_deletion.py --problem medium-graph --budget 28800 &
```

**3c. Pool + official score** (every few hours):
```bash
python3 tools/portfolio.py --problem medium-graph
python3 tools/verify_submission.py submissions/medium-graph/portfolio.json
```

**Done-criteria for medium:** pooled official score crosses −1,745,122 (beat), or
the arms plateau and the refiners stop improving (report the best, with the
A-vs-B Δ as the headline GBDT result regardless).

---

## 4. LARGE — same playbook, two adjustments

After medium is closing, set the arms to `large-graph` (n=2426, 500 width levels):
- **Memory:** three batch-1024 arms may not fit on one 32 GB GPU at this size —
  run the **A + B ablation pair** first, add C/stock later or drop `--batch_size`.
- **Time:** slower per generation; budget more wall-clock.
- Everything else identical, swapping `medium` → `large`. Target −5,493,062.

GAPS (§10, the +24,766 method) is the strongest single contributor on large —
make sure it runs in Layer 2 there.

---

## 5. Small — optional tail, no priority

Small is certified; nothing here changes the thesis. Two harmless background
tickets may still be running and can be left to finish or stopped:
- the Kaggle warm-started continuation of `13183.pt` (resumes a converged run;
  expected to hold near the plateau, pool only if a band beats ours);
- any `qne_search` / refiners on the Mac.
Pool anything they produce; it can only help, never hurt (additive).

---

## 6. Honest stopping rules & what "done" looks like

- **Beat lands** (pooled official ≤ target) → bank it, note GBDT's share via the
  A−B ablation, update the results table + PDF.
- **Plateau** → report the best pooled number plainly with the measured Δ. A
  positive, controlled GBDT contribution on a championship engine is a publishable
  result *whether or not* the leaderboard is crossed.
- Only ever quote `verify_submission.py` / `portfolio.py` numbers. Never an
  engine's internal HVI.

**Thesis is complete when:** medium and large each have (a) a best pooled official
number, and (b) a clean A−B GBDT Δ. Small already has both, plus the certificate.

---

## 7. Rigorous future directions (note, don't block on)

- **Exact certification of bands 8–14** would need a treewidth solver that scales
  past ~700 vertices — PID timed out, so this is open; a positive-instance or
  SAT-encoded approach is the natural attempt.
- **Decomposition-guided ASP/MaxSAT** encoding of the *width*-bounded max-torso
  problem (cf. TorsoMaxSAT's clingo torso encoding, which optimises edges not
  width) — a different exact paradigm, same hardness wall expected; future work.
- **min-fill / treewidth transfer chapter** — generalise torso-deletion and the
  HV decomposition to the wider elimination-ordering family.

These are thesis *future-work* paragraphs, not blockers for the current campaign.
The path to a finished result runs entirely through §3 (medium) and §4 (large).
