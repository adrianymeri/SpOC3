# SpOC-3 Torso Decompositions -- canonical task runner.
#
# All targets are POSIX shell + python3 only; no extra dependencies.
# Run `make help` for a one-line summary of each target.

PY            := python3
SEED          := 42
SEEDS         := 42

# Default budgets per instance (seconds wall time).
BUDGET_SMALL  := 25
BUDGET_MEDIUM := 12
BUDGET_LARGE  := 25
BUDGET_EXTRA  := 1

# Algorithm list -- adjust if you want a subset matrix run.
ALGOS         := hc1_initial hc2_simple hc3_archive hc4_warm_start \
                 hc5_operators hc6_gap_fill hc7_kbottleneck hc8_lahc \
                 hc9_hv_accept hc10_torso_warm hc11_ils hc12_incremental \
                 hc13_tabu hc14_steepest hc15_hv_incremental

# ---------------------------------------------------------------------------
.PHONY: help test test-quick verify bench gen matrix matrix-small \
        matrix-medium matrix-large clean \
        reproduce-canonical multiseed-official multiseed-synth \
        convergence-plot analyze cd-diagram compare paper-data \
        tune-sa tune-grasp tune-vns tune-all

# ---------------------------------------------------------------------------
# Metaheuristic hyperparameter tuning (grid search, reproducible).
# Override the instance or add --full on the command line, e.g.:
#   make tune-sa PROBLEM=medium-graph
#   python3 tools/tune.py vns --problem large-graph --full
# ---------------------------------------------------------------------------
PROBLEM := small-graph

tune-sa:
	$(PY) tools/tune.py sa --problem $(PROBLEM) --seed $(SEED)

tune-grasp:
	$(PY) tools/tune.py grasp --problem $(PROBLEM) --seed $(SEED)

tune-vns:
	$(PY) tools/tune.py vns --problem $(PROBLEM) --seed $(SEED)

tune-all: tune-sa tune-grasp tune-vns
	@echo "=== tuning sweep complete: extra_instances/tuning_{sa,grasp,vns}.csv ==="

help:
	@echo "Correctness:"
	@echo "  make test               full correctness suite (~30 s)"
	@echo "  make test-quick         toy + edge-case correctness only (~1 s)"
	@echo "  make verify             re-verify every saved submission JSON"
	@echo ""
	@echo "Generate canonical artefacts (run on the reference workstation):"
	@echo "  make reproduce-canonical  re-run all 14 × 3 at seed=42 (~50 min)"
	@echo "  make multiseed-official   add seeds {1..11} on {hc5,hc9,hc11,hc14} (~50 min)"
	@echo "  make multiseed-synth      3 seeds × 20 instances × 13 algos (~13 min)"
	@echo "  make convergence-plot     generate Figures/convergence_large_inline.png"
	@echo "  make paper-data           run all four sequentially (~2 h)"
	@echo ""
	@echo "Analyze (no compute; cheap):"
	@echo "  make analyze              run analyze_multiseed.py + analyze_synthetic.py"
	@echo "  make cd-diagram           render the Nemenyi critical-difference figure"
	@echo "  make compare              hc9 vs hc12 vs hc15 head-to-head (official graphs)"
	@echo ""
	@echo "Synthetic benchmark + matrix (legacy):"
	@echo "  make bench              run hc4 + hc9 on the 20 extra instances (~1 min)"
	@echo "  make gen                regenerate the 20 extra Erdős-Rényi instances"
	@echo "  make matrix             run every hc on every official instance"
	@echo "  make matrix-small       run every hc on small-graph only"
	@echo "  make clean              remove __pycache__ / scratch files"

# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

test:
	$(PY) tests/test_correctness.py

test-quick:
	$(PY) tests/test_correctness.py --quick

# ---------------------------------------------------------------------------
# Verification of saved submissions
# ---------------------------------------------------------------------------

verify:
	@for prob in small-graph medium-graph large-graph; do \
	  for f in submissions/$$prob/*.json; do \
	    [ -f "$$f" ] || continue ; \
	    score=$$($(PY) tools/verify_submission.py "$$f" --quiet | grep "Official score" | awk '{print $$4}'); \
	    printf "  %-14s %-14s %s\n" "$$prob" "$$(basename $$f)" "$$score"; \
	  done; \
	done

# ---------------------------------------------------------------------------
# Synthetic-instance benchmark
# ---------------------------------------------------------------------------

gen:
	$(PY) tools/generate_instances.py

bench:
	$(PY) tools/bench_extra.py --algos hc4,hc9 \
	  --budget $(BUDGET_EXTRA) --out extra_instances/results.csv

# ---------------------------------------------------------------------------
# Official-instance matrix (every algo × every problem)
# ---------------------------------------------------------------------------

matrix: matrix-small matrix-medium matrix-large

matrix-small:
	@for algo in $(ALGOS); do \
	  echo "--- $$algo small-graph ---"; \
	  $(PY) algorithms/hill_climbing/$$algo.py --problem small-graph \
	    --budget $(BUDGET_SMALL) --seed $(SEED) 2>&1 | grep -E "Official score" | head -1; \
	done

matrix-medium:
	@for algo in $(ALGOS); do \
	  echo "--- $$algo medium-graph ---"; \
	  $(PY) algorithms/hill_climbing/$$algo.py --problem medium-graph \
	    --budget $(BUDGET_MEDIUM) --seed $(SEED) 2>&1 | grep -E "Official score" | head -1; \
	done

matrix-large:
	@for algo in $(ALGOS); do \
	  echo "--- $$algo large-graph ---"; \
	  $(PY) algorithms/hill_climbing/$$algo.py --problem large-graph \
	    --budget $(BUDGET_LARGE) --seed $(SEED) 2>&1 | grep -E "Official score" | head -1; \
	done

# ---------------------------------------------------------------------------
# World-class reproducibility pipeline
# ---------------------------------------------------------------------------
# These targets are the canonical paper-data pipeline:
#   1. reproduce-canonical  -- regen all 14 × 3 cells at seed=42
#   2. multiseed-official   -- 11 extra seeds on {hc5,hc9,hc11,hc14} × 3
#   3. multiseed-synth      -- 3 seeds on 13 algos × 20 synthetic instances
#   4. convergence-plot     -- regen trace data + inline figure for the paper
#   5. analyze              -- run statistical analysis on the CSVs
#
# After paper-data finishes, send the resulting CSVs back so the
# conference paper can be updated with the new numbers + figure.

reproduce-canonical:
	@echo "=== Regenerating all 14 × 3 = 42 canonical submissions ==="
	$(PY) tools/regen_canonical.py

multiseed-official:
	@echo "=== Running 11 extra seeds × 5 variants × 3 instances ==="
	$(PY) tools/multiseed.py \
	  --algos hc5,hc9,hc11,hc14,hc15 \
	  --seeds 1,2,3,4,5,6,7,8,9,10,11 \
	  --out extra_instances/multiseed.csv

multiseed-synth:
	@echo "=== Running 3 seeds × 20 instances × 13 algos ==="
	@for seed in 42 1 2; do \
	  echo "--- seed $$seed ---"; \
	  $(PY) tools/bench_extra.py \
	    --algos hc1,hc2,hc3,hc4,hc5,hc6,hc7,hc8,hc9,hc10,hc11,hc12,hc13 \
	    --budget $(BUDGET_EXTRA) \
	    --seed $$seed \
	    --out extra_instances/results.csv \
	    $$( [ $$seed = 42 ] && echo "" || echo "--append" ); \
	done

convergence-plot:
	@echo "=== Collecting hc5/hc7/hc9/hc11 traces on large-graph ==="
	@for algo in hc5 hc7 hc9 hc11; do \
	  echo "--- $$algo ---"; \
	  $(PY) tools/convergence_plots.py collect \
	    --algo $$algo --problem large-graph --seed $(SEED); \
	done
	@echo "=== Plotting inline figure ==="
	$(PY) tools/convergence_plots.py plot \
	  --problem large-graph --inline \
	  --algos hc7,hc9,hc11 \
	  --seed $(SEED)

analyze:
	@echo "=== Analyzing multi-seed official ==="
	$(PY) tools/analyze_multiseed.py
	@echo ""
	@echo "=== Analyzing synthetic benchmark ==="
	$(PY) tools/analyze_synthetic.py
	@echo ""
	@echo "=== Rendering critical-difference diagram ==="
	$(PY) tools/cd_diagram.py

cd-diagram:
	$(PY) tools/cd_diagram.py

# Head-to-head of the three strongest variants on the official graphs.
# Override seeds/budget on the command line, e.g.:
#   make compare SEEDS=42,1,2
compare:
	$(PY) tools/compare_variants.py --variants hc9,hc12,hc15 \
	    --problems small-graph,medium-graph,large-graph --seeds $(SEEDS)

paper-data: reproduce-canonical multiseed-official multiseed-synth convergence-plot analyze
	@echo ""
	@echo "=== paper-data pipeline complete ==="
	@echo "Artefacts produced:"
	@echo "  submissions/*/hc*.json                       (re-regenerated)"
	@echo "  extra_instances/canonical_seed42.csv         (14×3 scoreboard with drift)"
	@echo "  extra_instances/multiseed.csv                (12 seeds × 4 × 3)"
	@echo "  extra_instances/results.csv                  (3 seeds × 13 × 20)"
	@echo "  Figures/convergence_large_inline.png         (paper figure)"
	@echo "  extra_instances/multiseed_analysis.md        (stats + prose)"
	@echo "  extra_instances/synthetic_analysis.md        (Friedman + Nemenyi)"

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc'       -delete             2>/dev/null || true
	rm -f extra_instances/_smoke.csv
