#!/bin/bash
# recover_server.sh -- relaunch the full server fleet after a power outage.
# Safe to run any time: fronts live in submissions/*.json (saved on every accept),
# GPU arms warm-start from large_warm.pt, gbfcpp resumes its own state files.
# Usage:  cd ~/SpOC3/"Torso Decompositions" && bash tools/recover_server.sh
# Optional first: litterbox-sync the Mac's cap20.json in if the Mac was ahead.
set -e
cd "$(dirname "$0")/.."
HERE="$(pwd)"

echo "== disk survived? =="
python3 tools/cap_submit.py --problem large-graph | grep -i best-20

echo "== killing any stragglers =="
pkill -f run_gbdt.py 2>/dev/null; pkill -f run_capfocus.py 2>/dev/null
pkill -f gbfcpp.py 2>/dev/null; pkill -f hri_lns.py 2>/dev/null
pkill -f archive_evolve.py 2>/dev/null; sleep 2

echo "== GPU arms (warm-start) =="
FS=$(python3 tools/cap_submit.py --problem large-graph | awk -F'|' 'NF>=3 && $2+0>0 {gsub(/ /,"",$2); print $2}' | paste -sd, -)
echo "   capfocus sizes: $FS"
cd ~/cuda-torso
nohup python3 run_gbdt.py    --graph large-graph --batch_size 1024 --warmstart_pt "$HERE/large_warm.pt" --max_generations 100000000 > "$HERE/large_cuda_2.log" 2>&1 &
nohup python3 run_capfocus.py --graph large-graph --batch_size 1024 --warmstart_pt "$HERE/large_warm.pt" --focus-sizes "$FS" --max_generations 100000000 > "$HERE/large_capfocus.log" 2>&1 &
cd "$HERE"

echo "== CPU arms =="
nohup python3 tools/gbfcpp.py --problem large-graph --algo gbfcpp_cap20 --cap20 --rounds 1000 --round-budget 120 > large_gbfcpp_cap20.log 2>&1 &
nohup python3 tools/hri_lns.py --problem large-graph  --iters 500000 --seed $RANDOM > hri_lns_large_s2.log 2>&1 &
nohup python3 tools/hri_lns.py --problem medium-graph --iters 500000 --seed $RANDOM > hri_lns_medium_s5.log 2>&1 &
nohup python3 tools/archive_evolve.py --problem medium-graph --iters 2000000 --pool 40 --seed 11 --twins > ae_medium_tw11.log 2>&1 &
nohup python3 tools/hri_lns.py --problem small-graph --iters 300000 --seed $RANDOM --destroy-cap 60 > hri_lns_small.log 2>&1 &

sleep 60
echo "== verify (want 7 processes + 2 GPU apps) =="
ps aux | grep -E "run_gbdt|run_capfocus|gbfcpp|hri_lns|archive_evolve" | grep -v grep | awk '{print "  "$2, $12, $13, $14, $15}'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
grep -E "RESUMED|cap-focus ON" "$HERE/large_cuda_2.log" "$HERE/large_capfocus.log" 2>/dev/null | head -n 3
echo "== done =="
