# Workstation TODO — two deferred paper items

Everything else in the paper (EN + SQ) is final. Two results are deliberately
left for you to run on your **reference workstation**, because they depend on
controlled wall-clock timing and/or an external binary that should not come
from the sandbox. Both have ready-to-run harnesses; neither blocks submission,
because the paper already states the conservative form of each claim.

---

## #3 — Anytime HV-vs-wall-clock curves (Figure: convergence)

**Status in paper:** the convergence discussion currently leans on the existing
`extra_instances/conv_*.csv` + `convergence_{small,medium,large}.png`, which were
collected under sandbox timing. For the camera-ready you want these regenerated
on the reference machine so the wall-clock axis is publication-honest and the
per-instance budgets are identical across variants.

**Harness:** `tools/convergence_plots.py` (already in the repo).

```bash
# 1. collect — run once per (algo, problem) on the workstation, fixed seed 42
for algo in hc5 hc7 hc9 hc11; do
  for prob in small-graph medium-graph large-graph; do
    python3 tools/convergence_plots.py collect --algo $algo --problem $prob --seed 42
  done
done

# 2. plot — after all CSVs exist
python3 tools/convergence_plots.py plot
```

Outputs: `extra_instances/conv_{problem}_{algo}.csv` and
`extra_instances/convergence_{small,medium,large}.png`.

**Where it goes in the paper:** replace the three convergence PNGs referenced by
Figure 1, and re-check that the prose "three distinct convergence shapes"
sentence still matches the regenerated curves (it should — the shapes are a
function of graph density, not of timing). No number in Table 1 changes.

---

## #2-exact — Independent exact treewidth confirmation (large graph)

**Status in paper:** the width claim is currently stated as a *tight bracket*:
the minor-min-width (MMD) lower bound (Bodlaender & Koster, 2011) equals the best
achieved full-graph elimination width, both = 499, so the large-graph treewidth
is provably exactly 499. This is already airtight — an external solver only
upgrades the wording from "bracketed" to "independently verified".

**Harness:** `tools/exact_treewidth.py` (new).

```bash
# bracket check + PACE .gr export (no solver needed; reproduces 499 = 499)
python3 tools/exact_treewidth.py --problem large-graph

# optional independent confirmation with a PACE-2017 exact/anytime solver
python3 tools/exact_treewidth.py --problem large-graph \
    --solver /path/to/tw-exact          # tamaki / flow-cutter / htd, etc.
```

The script writes `extra_instances/large-graph.gr` in PACE `.gr` format
(`p tw n m` header, 1-indexed edges), runs the solver if given, parses the
`s td <bags> <maxbag> <n>` line, and reports `tw = maxbag − 1`. Compatible
binaries: Tamaki tw-exact (PACE 2017), FlowCutter, htd (`htd_main --opt width`).

**Sandbox result (already verified here):**
```
MMD lower bound            = 499
best full-graph width (t=0)= 499
==> bracket is TIGHT: exact treewidth = 499 (proved, no solver needed)
```

**Where it goes in the paper:** the existing sentence that reports the
"499 = 499" structural confirmation. If the external solver also returns 499,
change "bracketed" / "structural confirmation" wording to "independently
verified by an exact solver". If for any reason it disagrees, do **not** edit
the paper — tell me and we investigate (it would mean a bug in the export or
the solver's bag convention).

---

## Recap

| Item       | Blocks submission? | Harness                       | Paper edit if rerun |
|------------|--------------------|-------------------------------|---------------------|
| #3 curves  | No                 | tools/convergence_plots.py    | swap 3 PNGs, recheck prose |
| #2-exact   | No                 | tools/exact_treewidth.py      | one word: "verified" |

The submitted EN/SQ papers stand on their own as-is. These two items are
optional camera-ready upgrades, not corrections.

---

## RESOLVED (2 Jun 2026): hc1 small-graph regenerated on reference laptop

Regenerated on the reference machine (`python3 algorithms/hill_climbing/hc1_initial.py
--problem small-graph --budget 25 --seed 42`): the run scored **−256,620**,
confirming hc1's baseline is machine-speed dependent (observed −191,255 to
−519,696 across hosts). Per the contingency below, the single hc1 small cell
in Table 1 (EN+SQ docx) and `docs/RESULTS.md §4.1` were updated to **−256,620**
with a footnote noting the machine-dependence. All other 44 cells verified
identical to the canonical reference. No further action required.

---

## (historical) REQUIRED: regenerate hc1 small-graph on the reference workstation

During the metaheuristics restructure (1 Jun 2026), the canonical
`submissions/small-graph/hc1.json` was inadvertently overwritten by a
smoke-test run on a faster machine. Because `submissions/` is gitignored
(reference artifacts, not version-controlled) it could not be git-restored,
and hc1 is the only **random-init, no-warm-start** baseline, so its 25 s
wall-clock result is highly machine-speed dependent. The current file is a
valid, re-verifiable run scoring **−519,696** on this sandbox vs the paper's
reference value **−191,255**.

`hc14` small-graph was also re-run but reproduced its canonical score
(−1,814,518) exactly, so no action is needed there.

To restore the paper number, on the reference workstation run:

```bash
python3 algorithms/hill_climbing/hc1_initial.py \
    --problem small-graph --budget 25 --seed 42
make verify        # confirm small/hc1 reads -191,255 again
```

If your workstation also produces a different hc1 value, that simply
confirms hc1's baseline is machine-dependent; in that case update the single
hc1 small cell in Table 1 (EN+SQ) and docs/RESULTS.md §4.1 to the regenerated
number. No other variant is affected (hc4 onward all warm-start and were
untouched).
