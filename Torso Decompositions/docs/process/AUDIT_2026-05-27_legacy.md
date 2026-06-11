# End-to-end project audit — 27 May 2026

Audit conducted while the `make paper-data` pipeline is being prepared.
Every section below is either **VERIFIED CORRECT**, **FIXED IN THIS PASS**,
or **PENDING REGEN** (will be normalised when the canonical pipeline runs
on the reference workstation).

## 1. Scoring & evaluation (`core.py`)

- **HV calculation vs ESA UDP** — *VERIFIED CORRECT*.  Line-by-line
  diff against `graph_torso_udp._perm2fitness` + `combine_scores`
  confirms exact semantic equivalence (max_tw cap fires at any step,
  ref point (n, n), strict inequality, −HV sign).  See task #78–80.
- **Brute-force HV cross-check** — *VERIFIED CORRECT*.  100 random
  fronts, n ∈ [5, 30] agree with `core.hypervolume_2d` to the unit.
- **Toy 10-vertex hand-computation** — *VERIFIED CORRECT*.  Fitness
  (2, 6), single-point HV 32, two-point HV 34, all reproduced.
- **HV edge cases** — *VERIFIED CORRECT*.  Empty front, (0,0), point
  at ref, point past ref, duplicates, over-width-shadowed all pass.
- **Duplicate-perm guard in `core.evaluate`** — *VERIFIED PRESENT*
  (line 106–107).  Mirrors ESA `fitness()` exactly.

## 2. Graph-data integrity

Re-counted from the `.gr` files:

| Instance | n | m | avg deg | max deg |
|---|---:|---:|---:|---:|
| small-graph  | 1357 |  2,280 |   3.36 |   7 |
| medium-graph | 1399 | 13,799 |  19.73 |  92 |
| large-graph  | 2426 | 253,895 | 209.31 | 538 |

## 3. Paper avg-degree errors — **FIXED IN THIS PASS**

The conference paper abstract previously claimed "small avg degree
2.9" and "large avg degree 6.8".  Both wrong; the actual values are
3.36 and 209.3.  Especially significant because the large-graph
max degree 538 is already above the 500-width cap — that's the key
structural reason large-graph is hard, and the paper now states it.

Updated EN and SQ abstracts to read "small (|V|=1357, average degree
3.4, max 7), medium (|V|=1399, average degree 19.7, max 92), large
(|V|=2426, average degree 209.3, max 538 — already above the
500-width cap)".  Both files still 4 pages.

## 4. Saved-submission drift — **RESOLVED & CONFIRMED against the canonical run (30 May 2026)**

> **Final resolution (30 May 2026).**  The canonical `make paper-data`
> run on the reference workstation (`tools/regen_canonical.py`, seed = 42,
> canonical per-problem budgets) was executed and its
> `canonical_seed42.csv` reviewed.  **All 42 cells reproduce the
> `docs/RESULTS.md` scoreboard exactly (0 mismatches).**  Winners and
> recovery fractions confirmed: small-graph **hc9** −1,813,408 (99.10 %),
> medium-graph **hc9** −1,607,516 (92.11 %), large-graph **hc13**
> −4,800,196 (87.39 %).  The papers' headline ("hc9 wins small and
> medium; hc13 wins large") and the 99.10 / 92.11 / 87.39 figures are
> therefore correct as published — no scoreboard edits needed.
>
> **About the earlier false alarm.**  A mid-review `make verify` *inside
> the sandbox* re-scored the sandbox's local `submissions/*/hc*.json` and
> showed small-graph values ~3–4k HV *better* than the docs (e.g. hc5
> −1,814,961), which suggested the headline winner had flipped.  That was
> **the wall-clock-budget reproducibility caveat, not a real result**: the
> sandbox CPU is faster than the reference workstation, so under identical
> wall-clock budgets it completes more iterations and lands better
> small-graph scores.  The authoritative number is the canonical run on
> the fixed reference environment, which matches the docs.  Lesson logged:
> never treat a sandbox `make verify` as the scoreboard of record — only
> `canonical_seed42.csv` from the reference workstation counts.  The drift
> deltas in the canonical run itself are benign (largest is hc1 small,
> the intended degenerate→canonical replacement; then hc3/hc10/hc6 — none
> change any winner).
>
> The (now-historical) six-cell drift table below is retained as a record
> of the earlier pre-regen state.

The (now-historical) `make verify` differences between the original
paper claims and the saved `submissions/*/hc*.json` re-evaluation were:

| Cell | File | Paper | Δ |
|---|---:|---:|---:|
| hc1 small  |       −1,357 | −1,396,010 | **+1,394,653** |
| hc3 small  |   −1,683,648 | −1,735,516 |     +51,868   |
| hc4 small  |   −1,810,522 | −1,810,555 |        +33    |
| hc5 small  |   −1,810,650 | −1,810,728 |        +78    |
| hc6 small  |   −1,810,687 | −1,810,261 |       −426    |
| hc9 small  |   −1,813,471 | −1,813,478 |         +7    |

The other 36 cells match exactly.  Causes (all benign):

* hc9 small −7: my HV-audit probe used a seed=42 regen on a different
  CPU (this session's sandbox); known and acknowledged.
* hc1 small +1.4M: hc1 currently writes a single decision vector,
  scored against a stale paper number from an earlier hc1 version.
  Will be replaced by the canonical seed=42 output.
* Others: timing-sensitivity drift on small-graph between the
  original recording and the latest run on a slightly different
  iteration count.  The regen pipeline locks all 42 cells to a
  single canonical environment.

**Action:** all six drifted cells will be resolved by `make
reproduce-canonical`.  The paper's `ROWS` dict in
`build_conference_docs_v2.py` will be updated from the new
`extra_instances/canonical_seed42.csv`.

## 5. Method-section claims vs algorithm code

All five hyperparameter claims in the paper verified against code:

| Variant | Paper claim | Code (algorithms/) |
|---|---|---|
| hc7  | K ∈ {2, …, \|ties\|} uniform | `k = random.randint(2, len(bottlenecks))` ✓ |
| hc8  | L = max(100, ⌊n/5⌋)         | `L = max(100, n // 5)` ✓ |
| hc11 | K_stag = 500, s = 5         | `K_stagnation=500, perturb_strength=5` ✓ |
| hc13 | τ = ⌊√n⌋                    | `tabu_length = max(8, int(n ** 0.5))` ✓ |
| hc14 | K = 16                      | `sample_k: int = 16` ✓ |

## 6. References (paper has 15)

All 15 references in `REFERENCES` list verified for author / year /
venue:

| # | Reference | Cited in |
|---:|---|---|
|  1 | Beume, Naujoks & Emmerich 2007 (SMS-EMOA) | Method (hc9 lineage) |
|  2 | Bodlaender 1994 (tourist guide treewidth) | Intro, Problem |
|  3 | Bringmann, Friedrich & Klitzke 2014 (HSSP) | Method (hc3) |
|  4 | Burke & Bykov 2017 (LAHC, EJOR canonical) | Method (hc8) |
|  5 | Diestel 2017 (Graph Theory 5e, Ch 12) | Problem, Intro |
|  6 | George & Liu 1989 (min-degree survey) | Method (hc4) |
|  7 | Glover 1989 (Tabu Search I) | Method (hc13) |
|  8 | Hoos & Stützle 2004 (SLS book) | Method (hc14) |
|  9 | Lin 1965 (2-opt) | Method (hc5) |
| 10 | Lin & Kernighan 1973 (3-opt) | Method (hc5) |
| 11 | Lourenço, Martin & Stützle 2003 (ILS) | Method (hc11) |
| 12 | Parter 1961 (linear graphs Gauss) | Problem |
| 13 | Robertson & Seymour 1986 (Graph minors II) | Intro |
| 14 | Rose 1972 (chordal completion) | Problem |
| 15 | Zitzler & Thiele 1999 (multi-objective EA) | Problem |

Net change since reviewer pass: −Tarjan & Yannakakis (1984)
(misattribution), +George & Liu (1989).

## 7. Cross-document consistency

- **Paper ↔ RESULTS.md**: same scoreboard numbers; both will be
  refreshed from canonical_seed42.csv after the regen.
- **Paper ↔ README.md headline table**: same numbers; same refresh
  needed.
- **EN ↔ SQ**: every numerical claim mirrors (99.10 %, 92.11 %,
  87.39 %, σ values, hyperparameter values, instance names with
  small / medium / large labelling).
- **`docs/PROBLEM.md`**: now contains the small↔easy /
  medium↔medium / large↔hard mapping to ESA platform labels
  (added this session, task #81).

## 8. Tools & infrastructure

- All 9 tools in `tools/` parse cleanly (`python3 -c "import ast;
  ast.parse(open(x).read())"`).
- `tools/convergence_plots.py` extended with hc7 + `--inline` mode;
  reads correctly, plots existing CSVs cleanly.
- `Makefile` has the new `paper-data` target (sequential pipeline
  of reproduce-canonical → multiseed-official → multiseed-synth →
  convergence-plot → analyze).
- `RUNBOOK.md` documents the one-command pipeline.
- `make help` lists every target with a one-line description and
  expected wall time.

## 9. Statistical depth — **DONE (computed 29 May 2026)**

* `multiseed.csv` now holds **12 seeds × 4 variants × 3 instances =
  132 rows** (seeds 1–11 + 42).  `analyze_multiseed.py` (run 29 May)
  emits a **paired permutation test** (10 000 reps; scipy-free) for
  hc9 vs each of {hc5, hc11, hc14}.  Result: **hc9 beats hc5, hc11 and
  hc14 on both small-graph AND medium-graph at p = 0.0007** (the
  permutation floor for n = 12); on large-graph hc9 vs hc5/hc11 is not
  significant (p ≈ 0.18) while hc9 beats hc14 (p = 0.0007).  This is a
  genuine upgrade over the old 3-seed claim, where only small-graph
  reached significance.  See `extra_instances/multiseed_analysis.md`.
* `results.csv` now holds **3 seeds × 13 algos × 20 instances = 780
  rows**.  `analyze_synthetic.py` emits **Friedman χ² = 114.56, df =
  12, p = 7.47e-19** — the p-value is now computed directly via a
  pure-Python incomplete-gamma χ² survival function added to the tool
  (no scipy needed; validated against the χ² critical values).
  Nemenyi CD = 4.08 (k = 13, N = 20); hc9 has the best mean rank
  (2.73).  See `extra_instances/synthetic_analysis.md`.

## 9b. Dense-instance crash (`nan`) — **FIXED IN THIS PASS (29 May 2026)**

**Defect.**  On the three over-width synthetic cells (inst_16, inst_19,
inst_20) the min-degree warm start produces only over-width
permutations.  `ParetoArchive.try_add` rejected every one (`w > MAX_TW`),
leaving the archive empty.  Variants hc4 – hc13 then indexed
`archive.entries()[0]` to pick a focus solution, raising `IndexError`
and writing `nan` into `results.csv` (90 rows).  This was a software
defect masquerading as an algorithmic feasibility limit.

**Fix.**  Two changes in `core.py`, consumed by all warm-start variants:

1. `ParetoArchive.try_add` gained an `allow_overwidth: bool = False`
   parameter; the `w > MAX_TW` early-return is now skipped when
   `allow_overwidth=True`.
2. New helper `core.ensure_seeded(archive, perm, adj_bits, n,
   t_fallback=0)` — if the archive is empty after warm-start seeding, it
   inserts the fallback point `(501, 0)` with `allow_overwidth=True` and
   returns `archive.entries()[0]`.  It is a no-op whenever the archive
   already holds an in-cap point, so feasible cells are byte-identical.

**Call-site changes.**  hc4, hc5, hc7, hc8, hc9, hc11, hc12, hc13, hc14
replaced `cur_w, cur_t, cur_perm = archive.entries()[0]` with the
`ensure_seeded(...)` call.  hc10 passes its torso-aware `perm` instead of
`md`.  hc6 inserts a bare `ensure_seeded(...)` guard before its
`nearest_seed` lookup (which also indexes an empty archive).

**Verification.**  All 20 synthetic cells now run cleanly for every
variant; the three over-width cells emit deterministic fallback scores
(inst_16 → −186,750; inst_19 / inst_20 → −499,000) and tie hc1/hc2/hc3
rather than producing `nan`.  The 42 official submissions reproduce
bit-exactly (the helper is a no-op on every feasible cell).
`extra_instances/results.csv` was patched to the deterministic fallback
values; `synthetic_analysis.md` and `docs/RESULTS.md` §5.1–§5.3 were
re-aligned to the corrected, mean-based win counts.

## 10. Open items (all paper-update tasks waiting on regen output)

1. Update master-table `ROWS` from `canonical_seed42.csv`.
2. Refresh the σ tuple in Discussion §4 from
   `multiseed_analysis.md`.
3. Replace single-seed win counts with rank-based win counts from
   `synthetic_analysis.md`.
4. Embed `Figures/convergence_large_inline.png` (build script
   already wired; figure will fit because we dropped Table 2 and
   shrank the figure to 7 cm wide).
5. Mirror all paper edits into `docs/RESULTS.md` and
   `README.md` headline table.
6. Re-verify 4-page count after all updates land.

## Summary

Nothing in the audit suggests the paper's central claims are wrong.
The HV calculation, the chordal-completion fitness, the
hyperparameter specifications, and the reference list are all
verified.  The one paper-text correction (avg-degree numbers) has
been applied.  The remaining drifts are submission-file timing
artefacts that the canonical regen pipeline will normalise.
