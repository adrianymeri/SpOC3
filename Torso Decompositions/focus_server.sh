#!/usr/bin/env bash
# ============================================================
# focus_server.sh -- SERVER. Concentrate the whole campaign on the ONE
# reachable lever: the t=0 endpoint width (each -1 width = +n HV).
#   * STOPS the scattered / frozen / from-scratch arms.
#   * KEEPS the GPU lottery (valid CUDA context -- never restart blindly).
#   * KEEPS one gbfcpp --cap20 medium (complementary scoring-width attack).
#   * LAUNCHES the endpoint_ils fleet (medium x3, small x2, large x2).
# Idempotent: only starts what is missing. Safe to re-run.
#   usage:  cd ~/SpOC3/"Torso Decompositions" && bash focus_server.sh
# ============================================================
cd "$(dirname "$0")" || exit 1
echo "=== FOCUS SERVER  $(date '+%d.%m %H:%M') ==="

# ---- 1. stop the frozen / scattered / from-scratch arms (endpoint is the bet now) ----
for pat in hri_plain.py hri_lns.py hri_lns_gbdt.py archive_evolve.py bandit_widths.py \
           boundary_lns.py quotient_lns.py cqs.py gbdt_grow.py grow_forever.sh \
           torso_deletion.py; do
  pkill -f "$pat" 2>/dev/null && echo "  stopped: $pat"
done
sleep 2

alive(){ pgrep -f "$1" >/dev/null 2>&1; }
launch(){  # $1 pattern  $2 label  $3 command
  if alive "$1"; then
    echo "  [alive] $2"
  else
    echo "  [START] $2"
    nohup bash -lc "$3" >/dev/null 2>&1 &
    sleep 1
  fi
}

# ---- 2. keep ONE complementary scoring-width attack on medium ----
launch "gbfcpp.py --problem medium-graph" "gbfcpp medium (scoring widths)" \
  "python3 -u tools/gbfcpp.py --problem medium-graph --algo gbfcpp_srv --cap20 --rounds 100000 --round-budget 120 > medium_gbfcpp_srv.log 2>&1"

# ---- 3. the endpoint ILS fleet (the concentrated bet) ----
launch "endpoint_ils.py --problem medium-graph --seed 1" "endpoint medium s1" \
  "python3 -u tools/endpoint_ils.py --problem medium-graph --seed 1 > endpoint_med_s1.log 2>&1"
launch "endpoint_ils.py --problem medium-graph --seed 2" "endpoint medium s2" \
  "python3 -u tools/endpoint_ils.py --problem medium-graph --seed 2 > endpoint_med_s2.log 2>&1"
launch "endpoint_ils.py --problem medium-graph --seed 3" "endpoint medium s3" \
  "python3 -u tools/endpoint_ils.py --problem medium-graph --seed 3 > endpoint_med_s3.log 2>&1"
launch "endpoint_ils.py --problem small-graph --seed 1" "endpoint small s1" \
  "python3 -u tools/endpoint_ils.py --problem small-graph --seed 1 > endpoint_small_s1.log 2>&1"
launch "endpoint_ils.py --problem small-graph --seed 2" "endpoint small s2" \
  "python3 -u tools/endpoint_ils.py --problem small-graph --seed 2 > endpoint_small_s2.log 2>&1"
launch "endpoint_ils.py --problem large-graph --seed 1" "endpoint large s1" \
  "python3 -u tools/endpoint_ils.py --problem large-graph --seed 1 > endpoint_large_s1.log 2>&1"
launch "endpoint_ils.py --problem large-graph --seed 2" "endpoint large s2" \
  "python3 -u tools/endpoint_ils.py --problem large-graph --seed 2 > endpoint_large_s2.log 2>&1"

echo "--- live now ---"
ps aux | grep -E "endpoint_ils|gbfcpp|run_gbdt|run\.py" | grep -v grep \
       | awk '{print "  "$2, $12, $13, $14, $15}'
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  GPU: /'
echo
echo "watch for a beat:   grep ENDPOINT endpoint_*.log"
echo "then score it:      python3 tools/cap_submit.py --problem medium-graph | grep -E 'best-20|residual'"
