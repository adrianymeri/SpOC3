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
pkill -f archive_evolve.py 2>/dev/null; sleep 2

echo "== GPU arms (warm-start) =="
FS=$(python3 tools/cap_submit.py --problem large-graph | awk -F'|' 'NF>=3 && $2+0>0 {gsub(/ /,"",$2); print $2}' | paste -sd, -)
echo "   capfocus sizes: $FS"
cd ~/cuda-torso
nohup python3 run_gbdt.py    --graph large-graph --batch_size 1024 --warmstart_pt "$HERE/large_warm.pt" --max_generations 100000000 > "$HERE/large_cuda_3.log" 2>&1 &
nohup python3 run_capfocus.py --graph large-graph --batch_size 1024 --warmstart_pt "$HERE/large_warm.pt" --focus-sizes "$FS" --max_generations 100000000 > "$HERE/large_capfocus.log" 2>&1 &
cd "$HERE"

echo "== CPU arms =="
nohup python3 tools/gbfcpp.py --problem large-graph --algo gbfcpp_slack --cap20 --only-widths $W --rounds 1000 --round-budget 120 > large_gbfcpp_slack2.log 2>&1 &
nohup python3 tools/hri_lns.py --problem large-graph --iters 2000000 --seed $RANDOM > hri_lns_large5.log 2>&1 &
nohup python3 tools/archive_evolve.py --problem medium-graph --iters 2000000 --pool 40 --seed 11 --twins > ae_medium_tw11.log 2>&1 &
nohup python3 tools/hri_lns.py --problem medium-graph --iters 2000000 --seed $RANDOM > hri_lns_medium_s5.log 2>&1 &

sleep 45
echo "== verify (want 6) =="
ps aux | grep -E "gbfcpp|hri_lns|archive_evolve|run_capfocus|run_gbdt" | grep -v grep | awk '{print "  "$2, $12, $13, $14}'
nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader
python3 tools/cap_submit.py --problem large-graph  | grep -i best-20
python3 tools/cap_submit.py --problem medium-graph | grep -i best-20
