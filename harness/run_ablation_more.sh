#!/bin/bash
# run_ablation_more.sh -- the operator arms, in parallel, one core each.
#
#     ./run_ablation_more.sh                 # 1200s, 3 seeds, all arms
#     ./run_ablation_more.sh 300 1           # short trial
#     ./run_ablation_more.sh 1200 3 4        # at most 4 arms at a time
#
# Same reasoning as run_mac.sh: hill climbing is pure Python and uses one
# core, so capping every process to a single thread gives each arm identical
# compute, and several single-threaded processes on a multi-core Mac do not
# contend. That is what makes the new operators comparable to the four
# already measured -- same seconds, same seeds, same instances, same core.
#
# Sequentially this would be 14 arms x 10 instances x 3 seeds x 1200s = 140 h.
# One core per arm and the wall-clock is 140 h / arms-in-parallel.
#
# Every arm resumes: ablation_more.py reads its own CSV first and skips
# (instance, seed) pairs already finished, so stopping and relaunching costs
# nothing. Kill everything with: pkill -f ablation_more.py

set -euo pipefail

SECONDS_PER_RUN="${1:-1200}"
SEEDS="${2:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
# Leave two cores for the OS so the timed runs are not disturbed.
PARALLEL="${3:-$(( CORES > 3 ? CORES - 2 : 1 ))}"
LOGS="$HERE/logs_ablation_more"
mkdir -p "$LOGS"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# "all" first: it is the attribution run, where every operator competes for
# one budget and per-operator accept counts are recorded. That single arm is
# what updates the sheet's attribution block, so it should finish even if the
# rest is interrupted.
ARMS=(all
      or_opt three_opt k_opt_rotate
      or_opt_aimed three_opt_aimed k_opt_aimed
      corner_shuffle corner_earlier
      min_fill_window min_fill_aimed
      degeneracy_window degeneracy_aimed
      simplicial_pull)

PER_ARM=$(python3 -c "print(f'{10*$SEEDS*$SECONDS_PER_RUN/3600:.1f}')")
TOTAL=$(python3 -c "
import math
print(f'{math.ceil(${#ARMS[@]}/$PARALLEL)*10*$SEEDS*$SECONDS_PER_RUN/3600:.1f}')")

echo "cores available: $CORES   arms: ${#ARMS[@]}   in parallel: $PARALLEL"
echo "budget per arm: ${SECONDS_PER_RUN}s x ${SEEDS} seeds x 10 instances"
echo "per-arm wall-clock: ${PER_ARM} h"
echo "total wall-clock:   ${TOTAL} h"
echo "logs: $LOGS"
echo

running=0
for arm in "${ARMS[@]}"; do
    nohup python3 -u "$HERE/ablation_more.py" \
        --operator "$arm" \
        --seconds "$SECONDS_PER_RUN" \
        --seeds "$SEEDS" \
        > "$LOGS/$arm.log" 2>&1 &
    echo "launched $arm  (pid $!)  -> logs_ablation_more/$arm.log"
    running=$(( running + 1 ))
    if [ "$running" -ge "$PARALLEL" ]; then
        wait -n 2>/dev/null || wait
        running=$(( running - 1 ))
    fi
done

echo
echo "Keep the Mac awake in another terminal:"
echo "    caffeinate -is"
echo
echo "Progress:"
echo "    tail -n 2 $LOGS/*.log"
echo "    grep -c , $HERE/ablation_more-*.csv"

wait
echo "ALL ARMS FINISHED"
