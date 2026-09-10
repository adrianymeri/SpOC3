#!/bin/bash
# recover_server2.sh -- relaunch the certificate-era server fleet (canonical since 2026-07-11).
# Safe any time: fronts live in submissions/*.json; GPU arms warm-start from large_warm.pt.
# Usage: cd ~/SpOC3/"Torso Decompositions" && bash tools/recover_server2.sh
set -e
cd "$(dirname "$0")/.."
HERE="$(pwd)"
W=18,46,65,82,99,112,133,154,170,195,218,233,252,273

echo "== ingest GPU dumps (SANITIZED -- raw dumps have over-width heads) =="
cp ~/cuda-torso/submissions/large-graph/*.json submissions/large-graph/ 2>/dev/null || true
python3 tools/sanitize_pool.py --problem large-graph

echo "== killing stragglers =="
pkill -f run_gbdt.py 2>/dev/null; pkill -f run_capfocus.py 2>/dev/null
pkill -f gbfcpp.py 2>/dev/null; pkill -f hri_lns.py 2>/dev/null
pkill -f archive_evolve.py 2>/dev/null; pkill -f cqs.py 2>/dev/null
pkill -f boundary_lns.py 2>/dev/null; pkill -f bandit_widths.py 2>/dev/null; sleep 2

echo "== GPU arms: SMALL fresh-basin lottery (2026-07-12 pivot) =="
# Rationale: small orderings are ALWAYS ESA-valid (maxdeg 7 << 500) and the
# lottery is the only mechanism that ever moved small's wall (gap 6 -> 5,
# torso 887 -> 888). The large GPU arms' raw dumps are 100% invalid (over-
# width heads) and contributed nothing -- GPU compute belongs on small.
cd ~/cuda-torso
nohup python3 run_gbdt.py --graph small-graph --batch_size 1024 --max_generations 100000000 > "$HERE/small_gpu_lottery_gbdt.log" 2>&1 &
nohup python3 run.py      --graph small-graph --batch_size 1024 --max_generations 100000000 > "$HERE/small_gpu_lottery2.log" 2>&1 &
cd "$HERE"

echo "== CPU arms =="
nohup python3 -u tools/bandit_widths.py --problem large-graph --stint 1800 > bandit_large.log 2>&1 &
nohup python3 -u tools/cqs.py --width sweep --iters 500000000 --seed $RANDOM > cqs.log 2>&1 &
nohup python3 -u tools/boundary_lns.py --iters 100000000 --seed $RANDOM > boundary_lns2.log 2>&1 &
nohup python3 tools/hri_lns.py --problem large-graph --iters 2000000 --seed $RANDOM > hri_lns_large5.log 2>&1 &
nohup python3 tools/archive_evolve.py --problem medium-graph --iters 2000000 --pool 40 --seed 11 --twins > ae_medium_tw11.log 2>&1 &
nohup python3 tools/hri_lns.py --problem medium-graph --iters 2000000 --seed $RANDOM > hri_lns_medium_s5.log 2>&1 &

sleep 45
echo "== verify (want 6) =="
ps aux | grep -E "gbfcpp|hri_lns|archive_evolve|run_capfocus|run_gbdt" | grep -v grep | awk '{print "  "$2, $12, $13, $14}'
nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader
python3 tools/cap_submit.py --problem large-graph  | grep -i best-20
python3 tools/cap_submit.py --problem medium-graph | grep -i best-20
