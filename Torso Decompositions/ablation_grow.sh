#!/usr/bin/env bash
# ============================================================
# ablation_grow.sh -- the ablation that tests the GBDT role that WORKS.
#
# The endpoint move-ranker (endpoint_gbdt) was tested and lost: 0/8 on small,
# p = 0.0078 AGAINST it.  But that is only one of GBDT's roles.  Attribution
# says the role carrying the campaign is gbdt_grow / gbfcpp -- learned
# set-space construction -- which owns the endpoint on small and large and
# supplies 12/16, 8/20, 8/20 of the three submitted fronts.
#
# That role has never had a clean controlled test.  gbdt_grow.py has always had
# a --no-gbdt control (identical loop, classical boundary-count ranking instead
# of the learned ranker), but the earlier attempt was confounded: both arms read
# and wrote the SHARED pool, so each silently inherited the other's discoveries.
#
# This script fixes that:
#   * each arm gets an ISOLATED pool directory (--pool-dir), seeded from one
#     common snapshot, so neither can see the other;
#   * both arms get an EQUAL PASS BUDGET (--max-passes), not wall-clock, so the
#     result does not depend on machine load;
#   * arms run sequentially, single-threaded, paired by seed.
#
# Metric: delta_hv, the envelope hypervolume the arm gained from the identical
# starting snapshot.  Higher = better.
#
# PARALLELISM: because the budget is PASSES and not wall-clock, running several
# seeds at once cannot bias the comparison -- each arm still performs exactly
# the same amount of work however busy the machine is.  The two arms OF A PAIR
# still run sequentially inside their own job.  Sequential-everything was
# costing ~8 h per pair; use JOBS to fit the machine (server 64 cores: 6-8).
#
#   usage:  bash ablation_grow.sh [problem] [passes_per_arm] [n_seeds] [jobs]
#   e.g.    bash ablation_grow.sh large-graph 250 8 6
# ============================================================
cd "$(dirname "$0")" || exit 1
PROB="${1:-small-graph}"
PASSES="${2:-300}"
SEEDS="${3:-8}"
JOBS="${4:-1}"
OUT="ablation_grow_${PROB}.txt"
BASE="abl_grow_pools/${PROB}"

export GBDT_NJOBS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== paired gbdt_grow ablation | $PROB | $PASSES passes/arm | $SEEDS seeds ==="
echo "    isolated pools per arm; equal pass budget; sequential"
: > "$OUT"
rm -rf "$BASE"; mkdir -p "$BASE/snapshot"

# ONE common snapshot both arms start from.  It must contain more than
# cap20.json: gbdt_grow trains its per-width membership model on the pool's
# elite ORDERINGS, so a one-file snapshot would leave the learned ranker with
# almost nothing to fit while the classical control (which needs no training
# data at all) is unaffected -- i.e. it would handicap the treatment arm and
# bias the test toward the control.  Both arms receive the identical snapshot,
# so including real pool files keeps the comparison fair.
cp "submissions/${PROB}/cap20.json" "$BASE/snapshot/" 2>/dev/null
ls -t "submissions/${PROB}"/*.json 2>/dev/null \
  | grep -v -E "cap20_platform|/cap20\.json$" | head -30 \
  | while read -r f; do cp "$f" "$BASE/snapshot/" 2>/dev/null; done
n=$(ls "$BASE/snapshot" | wc -l | tr -d ' ')
if [ "$n" -eq 0 ]; then
  echo "  ERROR: no submission files for $PROB -- run cap_submit first"; exit 1
fi
echo "    snapshot: $n file(s) (identical for both arms)"

run_pair() {   # $1 = seed ; both arms of one pair, sequentially
  local s="$1" mode dir flag log
  for mode in gbdt nogbdt; do
    dir="$BASE/${mode}_s${s}"
    rm -rf "$dir"; mkdir -p "$dir"; cp "$BASE/snapshot/"*.json "$dir/"
    flag=""; [ "$mode" = "nogbdt" ] && flag="--no-gbdt"
    log="ablg_${PROB}_${mode}_s${s}.log"
    python3 -u tools/gbdt_grow.py --problem "$PROB" --seed "$s" \
        --algo "abl_${mode}_s${s}" --pool-dir "$dir" \
        --max-passes "$PASSES" --budget 999999 $flag > "$log" 2>&1
    # one writer at a time so concurrent pairs cannot interleave a line
    flock "$OUT.lock" -c "grep '^RESULT' '$log' >> '$OUT'" 2>/dev/null \
      || grep "^RESULT" "$log" >> "$OUT"
  done
  echo "  [$(date '+%H:%M')] seed $s done"
}

touch "$OUT.lock"
echo "  running $SEEDS pairs, $JOBS at a time"
for s in $(seq 1 "$SEEDS"); do
  run_pair "$s" &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
done
wait

echo
echo "=== RESULTS ($OUT) ==="
cat "$OUT"
echo
echo "=== PAIRED SUMMARY ==="
python3 tools/paired_stats.py "$OUT" --metric delta_hv
