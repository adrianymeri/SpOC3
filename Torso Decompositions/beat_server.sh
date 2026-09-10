#!/usr/bin/env bash
# ============================================================
# beat_server.sh -- SERVER fleet, aimed by the 2026-08-08 measurements.
#
# THE METHOD THAT BEAT MEDIUM (endpoint ILS: minimise the t=0 elimination
# width, seeded from the pool's own best ordering) applied per instance
# ACCORDING TO WHAT IS ACTUALLY REACHABLE:
#
#   medium  endpoint 228, clique bound 17  -> huge headroom, KEEP PUSHING
#           (already BEAT by 1,804 HV; more margin = insurance)
#   small   endpoint  15, max degree 7, max clique 2 -> NO obstruction at all.
#           Needs only +5 HV. Two independent routes: endpoint 15->14 (+119 HV)
#           or +5 torso vertices anywhere across widths 0..14 (+1 HV each).
#   large   endpoint 499 == K500 clique bound 499 -> PROVABLY OPTIMAL.
#           endpoint_ils can NEVER help large; it is not run here.
#           Widths 299/332/365/399/449/499 are all saturated against the
#           K500+K400+K300 packing bound. The whole +8,753 gap must come from
#           the LOW/MID widths, so large gets width-targeted growth there.
#
# Idempotent: starts only what is missing. Safe to re-run anytime.
#   usage:  cd ~/SpOC3/"Torso Decompositions" && bash beat_server.sh
# ============================================================
cd "$(dirname "$0")" || exit 1
echo "=== BEAT FLEET / SERVER  $(date '+%d.%m %H:%M') ==="

# One BLAS/OpenMP thread per arm. Without this each arm spawns a thread per
# core (dense Laplacian eigendecomposition, LightGBM), which drove this box to
# load 309 and made every arm slower than running them one at a time.
export GBDT_NJOBS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ---- large endpoint arms are provably useless: stop them ----
pkill -f "endpoint_ils.py --problem large-graph" 2>/dev/null \
  && echo "  stopped endpoint_ils on large (K500 bound => 499 is optimal)"

alive(){ pgrep -f "$1" >/dev/null 2>&1; }
launch(){
  if alive "$1"; then
    echo "  [alive] $2"
  else
    echo "  [START] $2"
    nohup bash -lc "$3" >/dev/null 2>&1 &
    sleep 1
  fi
}

# ---------- MEDIUM: keep extending the win ----------
for s in 1 2 3; do
  launch "endpoint_ils.py --problem medium-graph --seed $s" "endpoint medium s$s" \
    "python3 -u tools/endpoint_ils.py --problem medium-graph --seed $s > endpoint_med_s$s.log 2>&1"
done
launch "gbfcpp.py --problem medium-graph" "gbfcpp medium (scoring widths)" \
  "python3 -u tools/gbfcpp.py --problem medium-graph --algo gbfcpp_srv --cap20 --rounds 100000 --round-budget 120 > medium_gbfcpp_srv.log 2>&1"

# ---------- SMALL: the +5 -- both routes at once ----------
for s in 1 2 3; do
  launch "endpoint_ils.py --problem small-graph --seed $s" "endpoint small s$s" \
    "python3 -u tools/endpoint_ils.py --problem small-graph --seed $s > endpoint_small_s$s.log 2>&1"
done
launch "gbfcpp.py --problem small-graph" "gbfcpp small (cap20 widths)" \
  "python3 -u tools/gbfcpp.py --problem small-graph --algo gbfcpp_small --cap20 --rounds 100000 --round-budget 120 > small_gbfcpp.log 2>&1"
launch "gbdt_grow.py --problem small-graph" "gbdt_grow small (set-space)" \
  "python3 -u tools/gbdt_grow.py --problem small-graph --algo gbdt_grow_small --budget 2592000 > small_grow.log 2>&1"

# ---------- LARGE: only the UNSATURATED widths ----------
launch "gbfcpp_lowW" "gbfcpp large low widths" \
  "python3 -u tools/gbfcpp.py --problem large-graph --algo gbfcpp_lowW --only-widths 11,38,58,79,99,111,123 --rounds 100000 --round-budget 120 > large_loww.log 2>&1"
launch "gbfcpp_midW" "gbfcpp large mid widths" \
  "python3 -u tools/gbfcpp.py --problem large-graph --algo gbfcpp_midW --only-widths 148,170,195,213,232,254,276 --rounds 100000 --round-budget 120 > large_midw.log 2>&1"

echo "--- live now ---"
ps aux | grep -E "endpoint_ils|gbfcpp|gbdt_grow|run_gbdt" | grep -v grep \
       | awk '{print "  "$2, $12, $13, $14, $15}'
echo
echo "watch:  bash watch.sh"
