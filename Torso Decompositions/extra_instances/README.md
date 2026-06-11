# extra_instances/

Synthetic-benchmark data **and** all experiment-output artifacts. This folder
is intentionally flat for now because ~15 tools hard-code paths into it
(`tools/tune.py`, `bench_extra.py`, `meta_multiseed.py`, `convergence_*`, the
`analyze_*` scripts, ...). Splitting it physically is deferred behind a
path-centralization refactor — see `docs/process/RESTRUCTURE_PLAN.md`. Until
then, the contents fall into these categories:

| Category | Files | Produced/consumed by |
|---|---|---|
| Synthetic instances | `data/`, `instances.csv` | `generate_instances.py`, `bench_extra.py` |
| Benchmark results | `results.csv`, `multiseed.csv`, `meta_multiseed.csv`, `conv_*.csv` | `bench_extra.py`, `multiseed.py`, `convergence_*` |
| Tuning sweeps | `tuning_*.csv` | `tune.py`, `retune_longrun.py`, `analyze_*` |
| Figures | `*.png`, `*.pdf`, `cd_diagram.*` | `convergence_plots.py`, `cd_diagram.py` |
| Analysis writeups | `*_analysis.md`, `summary.md`, `*_comparison.md` | `analyze_synthetic.py`, `analyze_multiseed.py` |
| Misc/backup | `canonical_seed42.csv`, `*.prefix_bak` | one-off scripts |

When the path-centralization refactor lands, these become
`results/{benchmarks,tuning,figures,analysis}/` and `data/synthetic/`.
