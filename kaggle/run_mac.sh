#!/bin/bash
# run_mac.sh -- the three CPU solvers plus the baseline, in parallel, one core each.
#
#     ./run_mac.sh                 # 1200s per run, 3 seeds  (~10 h)
#     ./run_mac.sh 600 2           # shorter, for a trial
#
# Spacekangaroos is NOT here -- it needs a GPU and runs on Kaggle.
#
# Each solver gets its own process pinned to a single thread. Hill climbing is
# pure Python and can only use one core, so capping the others matches them to
# it: every CPU method then gets identical compute, which is the whole point.
# Four single-threaded processes on a multi-core Mac do not contend.

set -euo pipefail

SECONDS_PER_RUN="${1:-1200}"
SEEDS="${2:-3}"
DATA_DIR="${3:-data/v2}"      # twin-preserving instances
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "?")
echo "cores available: $CORES   solvers to launch: 4 (1 core each)"
echo "budget: ${SECONDS_PER_RUN}s x ${SEEDS} seeds x 10 instances per solver"
echo "instances: $DATA_DIR"
echo "estimated wall-clock: $(python3 -c "print(f'{10*$SEEDS*$SECONDS_PER_RUN/3600:.1f}')") h"
echo

for solver in hill_climbing hri fast_cma_es min_degree; do
    nohup python3 -u "$HERE/bench_one.py" \
        --solver "$solver" \
        --seconds "$SECONDS_PER_RUN" \
        --seeds "$SEEDS" \
        --data-dir "$DATA_DIR" \
        > "$LOGS/$solver.log" 2>&1 &
    echo "launched $solver  (pid $!)  -> logs/$solver.log"
done

echo
echo "Keep the Mac awake in another terminal:"
echo "    caffeinate -is"
echo
echo "Check progress:"
echo "    tail -n 3 $LOGS/*.log"
echo "    grep -c , $HERE/benchmark-*.csv"
echo
echo "When all four say DONE, merge with the Kaggle GPU result:"
echo "    python3 $HERE/merge_results.py $HERE"

wait
echo "ALL SOLVERS FINISHED"
