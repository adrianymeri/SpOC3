#!/usr/bin/env bash
# ============================================================
# beat_mac.sh -- MAC fleet. Deliberately LEAN (7 arms, not 16) so the
# laptop stays cool; every arm is wrapped in `caffeinate -i` so the Mac
# will not sleep them, and seeds are disjoint from the server's (11+)
# so the two machines explore different basins.
#
# Aimed the same way as beat_server.sh:
#   small  -- the +5 target, no structural obstruction (max degree 7,
#             max clique 2). Endpoint route AND set-space growth route.
#   medium -- already BEAT by 1,804 HV; extra seeds widen the margin.
#   large  -- endpoint is PROVABLY OPTIMAL (K500 clique => width >= 499,
#             we are at 499), so NO endpoint arm here. One growth arm on
#             its unsaturated low widths only.
#
# Idempotent. Stop everything again with:  bash stop_mac.sh
#   usage:  cd ~/Desktop/SpOC3/"Torso Decompositions" && bash beat_mac.sh
# ============================================================
cd "$(dirname "$0")" || exit 1
echo "=== BEAT FLEET / MAC  $(date '+%d.%m %H:%M') ==="

# one BLAS/OpenMP thread per arm (see beat_server.sh) -- keeps the laptop
# cooler and stops arms from fighting each other for cores
export GBDT_NJOBS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

alive(){ pgrep -f "$1" >/dev/null 2>&1; }
launch(){
  if alive "$1"; then
    echo "  [alive] $2"
  else
    echo "  [START] $2"
    caffeinate -i nohup bash -lc "$3" >/dev/null 2>&1 &
    sleep 1
  fi
}

# ---------- SMALL: the +5 (highest-value target on this machine) ----------
for s in 11 12 13; do
  launch "endpoint_ils.py --problem small-graph --seed $s" "endpoint small s$s" \
    "python3 -u tools/endpoint_ils.py --problem small-graph --seed $s > endpoint_small_s$s.log 2>&1"
done
launch "gbdt_grow.py --problem small-graph" "gbdt_grow small (set-space, +1 HV/vertex)" \
  "python3 -u tools/gbdt_grow.py --problem small-graph --algo gbdt_grow_small_mac --budget 2592000 > small_grow_mac.log 2>&1"

# ---------- MEDIUM: widen the margin ----------
for s in 11 12; do
  launch "endpoint_ils.py --problem medium-graph --seed $s" "endpoint medium s$s" \
    "python3 -u tools/endpoint_ils.py --problem medium-graph --seed $s > endpoint_med_s$s.log 2>&1"
done

# ---------- LARGE: unsaturated widths only ----------
launch "gbfcpp_lowW_mac" "gbfcpp large low widths" \
  "python3 -u tools/gbfcpp.py --problem large-graph --algo gbfcpp_lowW_mac --only-widths 11,38,58,79,99,111,123 --rounds 100000 --round-budget 120 > large_loww_mac.log 2>&1"

echo "--- live now ---"
ps aux | grep -E "endpoint_ils|gbfcpp|gbdt_grow" | grep -v grep | grep -v caffeinate \
       | awk '{print "  "$2, $12, $13, $14, $15}'
echo "  load:$(uptime | sed 's/.*load average://')"
echo
echo "watch:  bash watch.sh      stop everything:  bash stop_mac.sh"
