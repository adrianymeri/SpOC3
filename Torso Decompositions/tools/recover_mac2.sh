#!/bin/bash
# recover_mac2.sh -- relaunch the certificate-era Mac fleet (canonical since 2026-07-11).
# Safe any time: every arm reseeds from submissions/*.json.
# Usage: cd ~/Desktop/SpOC3/"Torso Decompositions" && bash tools/recover_mac2.sh
set -e
cd "$(dirname "$0")/.."
W=18,46,65,82,99,112,133,154,170,195,218,233,252,273

echo "== killing stragglers =="
pkill -f archive_evolve.py 2>/dev/null; pkill -f hri_lns.py 2>/dev/null
pkill -f gbfcpp.py 2>/dev/null; pkill -f boundary_lns.py 2>/dev/null; sleep 2

echo "== relaunching 6 arms =="
caffeinate -i nohup python3 -u tools/bandit_widths.py --problem large-graph --stint 1800 > bandit_large.log 2>&1 &
caffeinate -i nohup python3 -u tools/cqs.py --width sweep --iters 5000000 --seed $RANDOM > cqs.log 2>&1 &
caffeinate -i nohup python3 tools/hri_lns.py --problem large-graph --iters 2000000 --seed $RANDOM > hri_lns_large4.log 2>&1 &
caffeinate -i nohup python3 tools/archive_evolve.py --problem large-graph --iters 2000000 --pool 40 --seed 1 --twins > ae_large1.log 2>&1 &
caffeinate -i nohup python3 tools/gbfcpp.py --problem medium-graph --algo gbfcpp_capm --cap20 --rounds 1000 --round-budget 120 > medium_gbfcpp.log 2>&1 &
caffeinate -i nohup python3 tools/hri_lns.py --problem medium-graph --iters 2000000 --seed $RANDOM > hri_lns_medium3.log 2>&1 &
caffeinate -i nohup python3 -u tools/boundary_lns.py --iters 1000000 --seed $RANDOM > boundary_lns.log 2>&1 &

sleep 45
echo "== verify (want 6) =="
ps aux | grep -E "gbfcpp|hri_lns|archive_evolve|boundary_lns" | grep -v grep | grep -v caffeinate | awk '{print "  "$2, $12, $13, $14}'
python3 tools/cap_submit.py --problem large-graph  | grep -i best-20
python3 tools/cap_submit.py --problem medium-graph | grep -i best-20
