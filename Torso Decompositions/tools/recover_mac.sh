#!/bin/bash
# recover_mac.sh -- relaunch the full Mac fleet after a crash/reboot.
# Safe to run any time: every arm reseeds from the saved corpus (submissions/*.json),
# so nothing is lost except in-memory work since the last accept.
# Usage:  cd ~/Desktop/SpOC3/"Torso Decompositions" && bash tools/recover_mac.sh
set -e
cd "$(dirname "$0")/.."

echo "== killing any stragglers =="
pkill -f archive_evolve.py 2>/dev/null; pkill -f hri_lns.py 2>/dev/null
pkill -f "gbfcpp.py" 2>/dev/null; sleep 2

echo "== relaunching 6 arms =="
caffeinate -i nohup python3 tools/archive_evolve.py --problem large-graph  --iters 2000000 --pool 40 --seed 1  --twins > ae_large1.log 2>&1 &
caffeinate -i nohup python3 tools/archive_evolve.py --problem large-graph  --iters 2000000 --pool 40 --seed 9  --twins > ae_large_tw9.log 2>&1 &
caffeinate -i nohup python3 tools/archive_evolve.py --problem medium-graph --iters 2000000 --pool 40 --seed 7  --twins > ae_medium_twins.log 2>&1 &
caffeinate -i nohup python3 tools/gbfcpp.py --problem large-graph --algo gbfcpp_cap20 --cap20 --rounds 1000 --round-budget 120 > large_gbfcpp_cap20_2.log 2>&1 &
caffeinate -i nohup python3 tools/hri_lns.py --problem large-graph  --iters 500000 --seed $RANDOM > hri_lns_large2.log 2>&1 &
caffeinate -i nohup python3 tools/hri_lns.py --problem medium-graph --iters 500000 --seed $RANDOM > hri_lns_medium2.log 2>&1 &

sleep 60
echo "== verify (want 6) =="
ps aux | grep -E "archive_evolve|gbfcpp|hri_lns" | grep -v grep | grep -v caffeinate | awk '{print "  "$2, $12, $13, $14, $15}'
python3 tools/cap_submit.py --problem large-graph  | grep -i best-20
python3 tools/cap_submit.py --problem medium-graph | grep -i best-20
echo "== done. Optional: litterbox-sync with the server if it was ahead. =="
