#!/usr/bin/env bash
# ============================================================
# resume_all.sh  --  MAC
# Idempotent: starts any arm that is NOT already running.
# Safe to run any time (after a reboot, after an arm retires, etc.)
#   usage:  bash resume_all.sh
# ============================================================
cd ~/Desktop/SpOC3/"Torso Decompositions" || exit 1

ensure () {   # $1 = unique pgrep pattern   $2 = label   $3 = full command (with its own > log)
  if pgrep -f "$1" >/dev/null 2>&1; then
    echo "  [alive] $2"
  else
    echo "  [START] $2"
    caffeinate -i nohup bash -lc "$3" >/dev/null 2>&1 &
    sleep 1
  fi
}

echo "=== MAC resume $(date '+%d.%m %H:%M') ==="

# ---------- LARGE : 2x archive_evolve + 1x hri_lns (kept lean — large is near its floor) ----------
ensure "archive_evolve.py --problem large-graph --iters 2000000 --pool 40 --seed 1" \
  "ae large s1" \
  "python3 tools/archive_evolve.py --problem large-graph --iters 2000000 --pool 40 --seed 1 > ae_large1.log 2>&1"

ensure "archive_evolve.py --problem large-graph --iters 2000000 --pool 40 --seed 9" \
  "ae large s9" \
  "python3 tools/archive_evolve.py --problem large-graph --iters 2000000 --pool 40 --seed 9 > ae_large_tw9.log 2>&1"

ensure "hri_lns.py --problem large-graph" \
  "hri_lns large" \
  "python3 tools/hri_lns.py --problem large-graph --iters 2000000 --seed 2 > hri_lns_large2.log 2>&1"

# ---------- MEDIUM : 1x archive_evolve + 1x hri_lns + 1x gbfcpp generator ----------
ensure "archive_evolve.py --problem medium-graph" \
  "ae medium" \
  "python3 tools/archive_evolve.py --problem medium-graph --iters 2000000 --pool 40 --seed 1 > ae_medium_twins.log 2>&1"

ensure "hri_lns.py --problem medium-graph" \
  "hri_lns medium" \
  "python3 tools/hri_lns.py --problem medium-graph --iters 2000000 --seed 3 > hri_lns_medium3.log 2>&1"

ensure "gbfcpp.py --problem medium-graph" \
  "gbfcpp medium (generator, retires ~33h)" \
  "python3 tools/gbfcpp.py --problem medium-graph --algo gbfcpp_capm --cap20 --rounds 1000 --round-budget 120 > medium_gbfcpp.log 2>&1"

echo "--- live arms now: ---"
ps aux | grep -E "archive_evolve|gbfcpp|hri_lns" | grep -v grep | grep -v caffeinate \
       | awk '{print "  "$2, $3"%cpu", $12, $13, $14}'
echo "  load:$(uptime | sed 's/.*load average://')"
