#!/bin/bash
# keepalive_mac.sh -- idempotent self-heal for the Mac fleet.
# Reports each of the 9 arms ALIVE, and relaunches ONLY the DEAD ones.
# Nothing is lost: every accept is already banked to submissions/; dead arms reseed from there.
# Usage: cd ~/Desktop/SpOC3/"Torso Decompositions" && bash tools/keepalive_mac.sh
cd "$(dirname "$0")/.."
echo "=== keepalive MAC $(date '+%d.%m %H:%M') ==="

a(){ pgrep -f "$1" >/dev/null 2>&1; }

a "bandit_widths.py --problem large-graph"  && echo "  ALIVE  bandit_large"  || { echo "  START  bandit_large";  caffeinate -i nohup python3 -u tools/bandit_widths.py --problem large-graph  --stint 1800 > bandit_large.log 2>&1 & }
a "cqs.py"                                   && echo "  ALIVE  cqs"           || { echo "  START  cqs";           caffeinate -i nohup python3 -u tools/cqs.py --width sweep --iters 500000000 --seed $RANDOM > cqs.log 2>&1 & }
a "quotient_lns.py"                          && echo "  ALIVE  quotient_lns"  || { echo "  START  quotient_lns";  caffeinate -i nohup python3 -u tools/quotient_lns.py --iters 500000000 --seed $RANDOM > quotient_lns.log 2>&1 & }
a "bandit_widths.py --problem medium-graph" && echo "  ALIVE  bandit_medium" || { echo "  START  bandit_medium"; caffeinate -i nohup python3 -u tools/bandit_widths.py --problem medium-graph --stint 1800 > bandit_medium.log 2>&1 & }
a "hri_lns.py --problem large-graph"         && echo "  ALIVE  hri_lns_large" || { echo "  START  hri_lns_large"; caffeinate -i nohup python3 tools/hri_lns.py --problem large-graph --iters 2000000 --seed $RANDOM > hri_lns_large4.log 2>&1 & }
a "archive_evolve.py --problem large-graph"  && echo "  ALIVE  ae_large"      || { echo "  START  ae_large";      caffeinate -i nohup python3 tools/archive_evolve.py --problem large-graph --iters 2000000 --pool 40 --seed 1 --twins > ae_large1.log 2>&1 & }
a "gbfcpp.py --problem medium-graph"         && echo "  ALIVE  gbfcpp_medium" || { echo "  START  gbfcpp_medium"; caffeinate -i nohup python3 tools/gbfcpp.py --problem medium-graph --algo gbfcpp_capm --cap20 --rounds 1000 --round-budget 120 > medium_gbfcpp.log 2>&1 & }
a "hri_lns.py --problem medium-graph"        && echo "  ALIVE  hri_lns_medium"|| { echo "  START  hri_lns_medium";caffeinate -i nohup python3 tools/hri_lns.py --problem medium-graph --iters 2000000 --seed $RANDOM > hri_lns_medium3.log 2>&1 & }
a "boundary_lns.py"                          && echo "  ALIVE  boundary_lns"  || { echo "  START  boundary_lns";  caffeinate -i nohup python3 -u tools/boundary_lns.py --iters 100000000 --seed $RANDOM > boundary_lns.log 2>&1 & }

echo "-- scores --"
for p in small-graph medium-graph large-graph; do echo "  $p: $(python3 tools/cap_submit.py --problem $p 2>/dev/null | grep -i best-20)"; done
