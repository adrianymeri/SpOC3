#!/bin/bash
# run_budget_sweep.sh -- does more budget let search catch construction?
#
# The one question the Limits note leaves open: "A much longer run is the
# obvious way for search to catch up, and nothing here rules it out."
#
# THE DESIGN
#   Four solvers at three budgets, all ten instances, three seeds, one core
#   each, on ONE machine so budget is the only thing that varies:
#
#     hill_climbing   the published four-operator baseline (bench_one.py)
#     all             the seventeen-operator pool      (ablation_more.py)
#     grasp_front     the construction lever -- what search has to catch
#     hc_bottleneck   the move-pool lever
#
#   Three budgets rather than two, because two endpoints cannot tell you the
#   SHAPE. If search gains grow logarithmically with budget -- the usual
#   behaviour on a flat landscape -- then 10x budget buys almost nothing and
#   construction stays ahead forever. If they grow linearly, 10x buys 10x and
#   the gap closes. 1200 / 3600 / 12000 separates those two cases; a single
#   long run cannot.
#
#   Server results are NOT comparable with the Mac's at the same wall-clock,
#   because a different CPU buys a different number of evaluations per
#   second. That is why 1200s is re-run here: it is the within-machine
#   baseline this sweep is measured against.
#
# WHY ONE JOB PER INSTANCE
#   bench_one.py and ablation_more.py both loop over all ten instances
#   internally, so one job lasts 10 x seeds x budget -- a 12000s job would
#   run 100 h by itself and leave 60 cores idle. Pointing --data-dir at a
#   directory holding a single instance makes each job one instance (the
#   other nine are reported MISSING and skipped), which turns 12 long jobs
#   into 120 short ones and fills the machine.
#
#   Every job writes its own CSV and resumes from it, so interruptions cost
#   only the runs that were in flight. Merge at the end with:
#       python3 merge_budget_sweep.py
#
#     ./run_budget_sweep.sh                # 1200,3600,12000  3 seeds
#     ./run_budget_sweep.sh "1200 3600" 2  # cheaper trial
#     ./run_budget_sweep.sh "1200" 1 8     # smoke: 8 parallel
#
# Kill everything:  pkill -f bench_one.py; pkill -f ablation_more.py

set -euo pipefail

BUDGETS="${1:-1200 3600 12000}"
SEEDS="${2:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
CORES=$(nproc 2>/dev/null || echo 8)
PARALLEL="${3:-$(( CORES > 8 ? CORES - 4 : 1 ))}"
LOGS="$HERE/logs_budget"
OUT="$HERE/budget"
mkdir -p "$LOGS" "$OUT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

INSTANCES="small-graph medium-graph large-graph synth-1 synth-2 synth-3 \
synth-4 synth-5 synth-6 synth-7"
SOLVERS="hill_climbing all grasp_front hc_bottleneck"

# One directory per instance, holding just that instance.
SPLIT="$ROOT/data_one"
for inst in $INSTANCES; do
    mkdir -p "$SPLIT/$inst"
    [ -e "$SPLIT/$inst/$inst.gr" ] || ln -s "$ROOT/data/$inst.gr" "$SPLIT/$inst/$inst.gr"
done

n_jobs=0
for b in $BUDGETS; do for s in $SOLVERS; do for i in $INSTANCES; do
    n_jobs=$(( n_jobs + 1 ))
done; done; done

cpu_h=$(python3 - <<EOF
budgets = "$BUDGETS".split()
solvers = "$SOLVERS".split()
print(f"{sum(int(b) for b in budgets) * len(solvers) * 10 * $SEEDS / 3600:.0f}")
EOF
)
longest=$(python3 - <<EOF
print(f"{max(int(b) for b in '$BUDGETS'.split()) * $SEEDS / 3600:.1f}")
EOF
)

echo "cores $CORES, running $PARALLEL at a time"
echo "budgets: $BUDGETS   seeds: $SEEDS"
echo "solvers: $SOLVERS"
echo "jobs: $n_jobs   total compute: ~${cpu_h} CPU-hours"
echo "longest single job: ${longest} h"
echo "logs: $LOGS"
echo "csv:  $OUT"
echo

running=0
# Longest budget first, so the critical path starts immediately and the
# short jobs backfill around it.
for b in $(echo "$BUDGETS" | tr ' ' '\n' | sort -rn); do
  for s in $SOLVERS; do
    for i in $INSTANCES; do
      tag="${s}-b${b}-${i}"
      csv="$OUT/$tag.csv"
      if [ "$s" = "all" ]; then
          cmd=("$HERE/ablation_more.py" --operator all)
      else
          cmd=("$HERE/bench_one.py" --solver "$s")
      fi
      nohup python3 -u "${cmd[@]}" \
          --seconds "$b" --seeds "$SEEDS" \
          --data-dir "data_one/$i" --out "$csv" \
          > "$LOGS/$tag.log" 2>&1 &
      running=$(( running + 1 ))
      if [ "$running" -ge "$PARALLEL" ]; then
          wait -n 2>/dev/null || wait
          running=$(( running - 1 ))
      fi
    done
  done
done

echo "all $n_jobs jobs queued"
echo
echo "Progress:"
echo "    ls $OUT | wc -l"
echo "    grep -h -c , $OUT/*.csv | paste -sd+ | bc"
echo "    tail -n 1 $LOGS/*b12000*.log"
echo
wait
echo "BUDGET SWEEP FINISHED"
