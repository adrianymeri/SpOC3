# Project restructure plan (staged)

The repo grew organically across four chapters and two folders became dumping
grounds: `submissions/` (canonical JSONs mixed with 200+ seed-provenance files)
and `extra_instances/` (graph data + tuning CSVs + figures + analysis docs all
together).

## DONE (4 June 2026)

- **`submissions/` decluttered.** Per-seed sweep outputs (`*_s<N>.json`) now
  live in `submissions/<instance>/seeds/`; the ~20 canonical method files +
  `portfolio.json` stay flat. Routing is automatic in `core.submission_path`
  (`_is_seed_stem`), `tools/portfolio.py` scans both, and the cmaes/ceiling
  sweeps resume from `seeds/`. Verified: portfolio reproduces small −1,828,063
  exactly through the new layout. See `submissions/README.md`.
- **Root decluttered.** Process/status docs (`AUDIT.md`, `RUNBOOK.md`,
  `WORKSTATION_TODO.md`, this file) moved to `docs/process/`. Root now holds
  only `README.md`, `Makefile`, `core.py` + the package dirs.
- **Navigability.** `submissions/README.md` and `extra_instances/README.md`
  document each folder's layout.
- **New code already well-placed.** `algorithms/continuous/` (the cmaes
  chapter) and `leaderboard_reference/` are in their final homes.

## DEFERRED (needs a path-centralization refactor first)

The two remaining moves below each couple to 15+ tools that hard-code their
paths, so doing them safely requires first introducing a single source of truth
for directories (a `paths.py` / `core` constants) and refactoring the tools to
import from it. Only then are the physical moves a one-file change. Until that
refactor, these stay put (and are documented in place):

- **`extra_instances/` → `results/{benchmarks,tuning,figures,analysis}/` +
  `data/synthetic/`.** Referenced by ~15 tools (`tune.py`, `bench_extra.py`,
  `meta_multiseed.py`, `convergence_*`, `analyze_*`, ...).
- **`algorithms/{hill_climbing,grasp,...}/` → `algorithms/permutation/`.**
  ~17 import sites across `meta_common.py`, `tune.py`, `meta_multiseed.py`, and
  the family runners. (Low priority — `algorithms/` is already family-organized;
  `continuous/` is already separate.)

The original full-tree target and coupling notes are retained below for when
the path-centralization refactor is scheduled.

## Target layout

```
Torso Decompositions/
  README.md  Makefile  core.py
  algorithms/
    permutation/                 # the four permutation-space chapters
      hill_climbing/  simulated_annealing/  grasp/  vns/  aco/  nsga2/
      meta_common.py
    continuous/                  # the new paradigm  (cmaes_torso.py)
  tools/                         # drivers — unchanged
  data/
    official/                    # small/medium/large .gr
    synthetic/                   # the 20 generated instances
  results/
    submissions/
      canonical/<instance>/      # headline per-instance JSONs (hc*, grasp, ...)
      seeds/<instance>/          # grasp_s*, sms_ls_s* provenance
      portfolio/<instance>/      # portfolio.json
    tuning/                      # tuning_*.csv, retune_longrun.csv
    benchmarks/                  # results.csv, *multiseed*.csv, conv_*.csv, convergence_study.csv
    figures/                     # *.png, *.pdf
    analysis/                    # *_analysis.md, summary.md
  docs/
  leaderboard_reference/         # read-only study copies (done)
```

## Coupling that must change atomically with the move

These are the only places that hard-code the current paths; update them in the
same commit as the move, then run the verification step:

1. **`core.py`** — `graph_path()`, `submission_path()`, and any `repo_root()`
   joins. Point `graph_path` at `data/official/`; `submission_path` at
   `results/submissions/canonical/<problem>/`. Add a small resolver so
   `portfolio` and seed stems land in their subfolders.
2. **`algorithms/meta_common.py`** and **`tools/tune.py`** — imports like
   `algorithms.hill_climbing.hc9_hv_accept` and `algorithms.grasp.grasp`
   become `algorithms.permutation.hill_climbing...` etc.
3. **`tools/portfolio.py`** — its `glob` over `submissions/<problem>/*.json`
   becomes a glob over `results/submissions/{canonical,seeds}/<problem>/*.json`.
4. **`tools/*` path constants** — `ceiling_validation.py`, `retune_longrun.py`,
   `convergence_study.py`, `bench_extra.py`, `generate_instances.py` build
   paths under `submissions/` and `extra_instances/`; repoint to the new tree.
5. **`Makefile`** targets that reference the old paths.

## Migration steps (one sitting, no jobs running)

1. `git switch -c restructure` (or snapshot the folder).
2. Create the new directories; `git mv` files in (keeps history).
3. Apply the path/import edits above.
4. **Verify:** `python3 test_correctness.py` (pins the evaluator),
   then `python3 tools/portfolio.py --dry-run` (must reproduce
   small −1 819 187 / medium −1 617 086 / large −5 033 531+),
   then `python3 tools/verify_submission.py results/submissions/portfolio/large-graph/portfolio.json`.
5. If all green, commit. If anything fails, the snapshot/branch reverts cleanly.

## What stays put

`core.py`, `docs/`, `tools/` (the directory — only its internal path strings
change), and `leaderboard_reference/` keep their locations. The new
`algorithms/continuous/` is already in its final home.
