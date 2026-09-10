#!/usr/bin/env bash
# ============================================================
# ablation_endpoint.sh -- the controlled GBDT experiment for the thesis.
#
# Runs PAIRED arms of tools/endpoint_gbdt.py: for each seed, one GBDT-ranked
# arm and one uniform-random control arm, drawing candidates from the SAME
# pool with the SAME features and the SAME wall-clock budget.  The only
# difference between the arms is the selection rule, so any gap is
# attributable to the ranker.
#
# Metrics recorded per arm (final RESULT line of each log):
#   best_width      -- endpoint elimination width reached (primary; lower better)
#   best_bottleneck -- #vertices sitting at that width (secondary; the
#                      lexicographic gradient toward the next width drop)
#   ranker_r        -- correlation between what the model predicted for the
#                      move it chose and what that move actually achieved.
#                      This is the number that says whether GBDT learned
#                      anything; ~0 means it did not.
#
# BUDGET = NUMBER OF EVALUATIONS, not seconds.  A wall-clock budget would hand
# the two arms different amounts of work whenever machine load changes between
# them (this box has run at load 300+), which silently confounds the very
# comparison the thesis rests on.  Evaluation parity is also the standard
# budgeting convention for comparing search heuristics.
#
#   usage:  bash ablation_endpoint.sh [problem] [evals_per_arm] [n_seeds]
#   e.g.    bash ablation_endpoint.sh small-graph 150000 6
# ============================================================
cd "$(dirname "$0")" || exit 1
PROB="${1:-small-graph}"
EV="${2:-150000}"
SEEDS="${3:-6}"
OUT="ablation_endpoint_${PROB}.txt"

# keep every arm single-threaded so throughput per arm is stable and
# comparable, and so this script does not add to the load storm
export GBDT_NJOBS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== paired endpoint ablation | $PROB | ${EV} evals per arm | $SEEDS seeds ==="
echo "    equal EVALUATION budget => result independent of machine load"
echo "    arms run sequentially; single-threaded"
: > "$OUT"

for s in $(seq 1 "$SEEDS"); do
  for mode in gbdt random; do
    log="abl_${PROB}_${mode}_s${s}.log"
    echo "  [$(date '+%H:%M')] seed $s / $mode ..."
    python3 -u tools/endpoint_gbdt.py --problem "$PROB" --seed "$s" \
        --algo-mode "$mode" --max-evals "$EV" > "$log" 2>&1
    grep "^RESULT" "$log" >> "$OUT"
  done
done

echo
echo "=== RESULTS ($OUT) ==="
cat "$OUT"
echo
echo "=== PAIRED SUMMARY ==="
python3 - "$OUT" << 'PYEOF'
import sys, re
rows={}
for line in open(sys.argv[1]):
    if not line.startswith("RESULT"): continue
    d=dict(kv.split("=",1) for kv in line.split()[1:])
    rows.setdefault(int(d["seed"]),{})[d["mode"]]=d
wins=ties=losses=0
print(f"{'seed':>4} | {'gbdt (w,btl)':>16} | {'random (w,btl)':>16} | winner | ranker_r")
for s in sorted(rows):
    p=rows[s]
    if "gbdt" not in p or "random" not in p: continue
    g=(int(p['gbdt']['best_width']), int(p['gbdt']['best_bottleneck']))
    r=(int(p['random']['best_width']), int(p['random']['best_bottleneck']))
    if g<r: w="GBDT"; wins+=1
    elif g>r: w="random"; losses+=1
    else: w="tie"; ties+=1
    print(f"{s:>4} | {str(g):>16} | {str(r):>16} | {w:>6} | {p['gbdt']['ranker_r']}")
n=wins+losses
print(f"\nGBDT wins {wins}, losses {losses}, ties {ties}")
if n:
    # exact two-sided sign test on the non-tied pairs
    from math import comb
    k=min(wins,losses)
    pv=sum(comb(n,i) for i in range(0,k+1))*2/(2**n)
    print(f"exact two-sided sign test over {n} non-tied pairs: p = {min(pv,1.0):.4f}")
    print("(p < 0.05 => the ranker, not chance, explains the difference)")
else:
    print("all pairs tied -- no signal either way; report as inconclusive")
PYEOF
