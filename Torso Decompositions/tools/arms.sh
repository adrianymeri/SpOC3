#!/bin/bash
# arms.sh -- READ-ONLY. Shows every arm ALIVE/DEAD + the score. Restarts nothing.
# Auto-detects Mac (9 arms) vs server (8 arms, has nvidia-smi). Run on either:
#   bash tools/arms.sh
cd "$(dirname "$0")/.."
echo "=== ARMS + SCORE  $(hostname -s)  $(date '+%d.%m %H:%M') ==="
a(){ pgrep -f "$1" >/dev/null 2>&1 && echo "  ALIVE  $2" || echo "  DEAD   $2  <-- relaunch"; }

if command -v nvidia-smi >/dev/null 2>&1; then
  # ---- SERVER roster (8) ----
  a "run_gbdt.py --graph small-graph"          "gpu_lottery_gbdt"
  a "run.py --graph small-graph"               "gpu_lottery2"
  a "bandit_widths.py --problem large-graph"   "bandit_large"
  a "cqs.py"                                    "cqs"
  a "boundary_lns.py"                           "boundary_lns"
  a "hri_lns.py --problem large-graph"          "hri_lns_large"
  a "archive_evolve.py --problem medium-graph"  "ae_medium"
  a "hri_lns.py --problem medium-graph"         "hri_lns_medium"
  nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  GPU: /'
else
  # ---- MAC roster (9) ----
  a "bandit_widths.py --problem large-graph"   "bandit_large"
  a "cqs.py"                                    "cqs"
  a "quotient_lns.py"                           "quotient_lns"
  a "bandit_widths.py --problem medium-graph"  "bandit_medium"
  a "hri_lns.py --problem large-graph"          "hri_lns_large"
  a "archive_evolve.py --problem large-graph"   "ae_large"
  a "gbfcpp.py --problem medium-graph"          "gbfcpp_medium"
  a "hri_lns.py --problem medium-graph"         "hri_lns_medium"
  a "boundary_lns.py"                           "boundary_lns"
fi

echo "-- scores (the number that matters) --"
for p in small-graph medium-graph large-graph; do echo "  $p: $(python3 tools/cap_submit.py --problem $p 2>/dev/null | grep -i best-20)"; done
