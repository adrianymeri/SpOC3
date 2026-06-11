# World-class reproducibility runbook

Single source of truth for regenerating every artefact the conference paper
depends on, in the right order.  All commands are run from the project root.

## What you have right now

| Artefact | State |
|---|---|
| `tools/regen_canonical.py` | built, dry-run tested |
| `tools/multiseed.py` | already existed (extended seed list works) |
| `tools/bench_extra.py` | already existed (extended via Makefile loop) |
| `tools/convergence_plots.py` | extended with hc7 + `--inline` mode |
| `tools/analyze_multiseed.py` | built, runs on existing data |
| `tools/analyze_synthetic.py` | built, runs on existing data |
| `Makefile` | new `paper-data`, `analyze` targets |
| `build_conference_docs_v2.py` | wired to optional convergence figure with graceful skip |
| `Konferenca Alb-Shkenca-Shtator-2026-{EN,SQ}.docx` | 4 pages, figure not yet embedded |

`core.py` is **unchanged** and still verified against the ESA UDP.

## The single command that produces every paper artefact

```sh
make paper-data
```

Runs four phases sequentially.  Total wall time on your workstation:
**≈ 2 hours**.  Safe to leave running unattended.

### Phase 1 — `make reproduce-canonical` (≈ 50 min)

Re-runs all 14 variants × 3 official instances at seed = 42 with the
canonical 25 / 12 / 25-s budgets.  Overwrites `submissions/*/hc*.json`
with byte-fresh JSON.  Writes `extra_instances/canonical_seed42.csv`
showing the score, elapsed time, and *delta* against the previous
submission.  Anything non-zero in the `delta` column is silent drift
that the paper update step will pick up.

This phase fixes the hc9 small-graph −7 unit drift introduced during
the HV-audit probe.

### Phase 2 — `make multiseed-official` (≈ 50 min)

11 extra seeds (1 … 11) on {hc5, hc9, hc11, hc14} × 3 official
instances.  Appends to `extra_instances/multiseed.csv`.  Together with
seed = 42 from Phase 1's canonical submission this gives **12 seeds
per (variant, instance) cell**, enough for Wilcoxon signed-rank tests.

### Phase 3 — `make multiseed-synth` (≈ 13 min)

3 seeds × 20 synthetic instances × 13 algos = 780 runs at the 1-s
budget.  Writes `extra_instances/results.csv`.  Enables Friedman test
+ post-hoc Nemenyi multiple-comparison across the 13 variants.

### Phase 4 — `make convergence-plot` (≈ 2 min)

Collects archive-HV-vs-wall-time traces for hc5, hc7, hc9, hc11 on
the large-graph instance, then renders a compact inline figure at
`Figures/convergence_large_inline.png` for the paper.

### Phase 5 — `make analyze` (≈ 5 s; no compute)

Reads the three CSVs from Phases 1-3 and emits two markdown reports
ready to paste into the paper:

* `extra_instances/multiseed_analysis.md` — per-cell mean ± σ table +
  Wilcoxon signed-rank p-values + a one-paragraph prose synthesis
  for Discussion §4.
* `extra_instances/synthetic_analysis.md` — Friedman χ² test, mean
  per-instance rank table, Nemenyi critical difference,
  and a new wins-by-rank-mean column for the master table.

## After the pipeline finishes

Send me back the four CSVs:
```
extra_instances/canonical_seed42.csv
extra_instances/multiseed.csv
extra_instances/results.csv
Figures/convergence_large_inline.png
```
plus the two analysis markdowns.  I will then:

1. Update `build_conference_docs_v2.py` `ROWS` with any canonical
   numbers that drifted, plus the Wilcoxon-derived σ values for
   Discussion §4 and the rank-based win counts for the master table.
2. Rebuild both EN and SQ; reverify 4 pages.
3. Mirror the new numbers in `docs/RESULTS.md`.

## Sanity checks you can run at any time

```sh
make test            # full correctness suite (~30 s)
make test-quick      # toy + edge cases only (~1 s)
make verify          # re-score every saved submission
make analyze         # regenerate markdown reports from current CSVs
```

## What still has to happen on your machine, not the sandbox

* Phase 1-4 above.  All four require canonical hardware so the
  iteration counts under the wall-clock budget match the paper's
  claims (the sandbox is a different CPU, and `core.py` is timing-
  sensitive at fixed budget).
* Final 4-page PDF render via LibreOffice or Word.  The sandbox
  build verifies layout but the publication-quality PDF should come
  from your machine.

## Sanity invariant: the HV scorer is unchanged

The audit against the ESA UDP source confirmed `core.py:hypervolume_2d`
and `core.py:evaluate` are line-for-line equivalent to
`graph_torso_udp._perm2fitness` + `combine_scores`.  Nothing in the
paper-data pipeline modifies `core.py`.  Re-running the canonical
regen on a different machine will produce slightly different scores
because of hardware-dependent iteration counts under a fixed wall
budget, **not** because the scorer changed.
