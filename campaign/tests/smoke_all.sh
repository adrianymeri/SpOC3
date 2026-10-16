#!/usr/bin/env bash
# Every CPU method on every instance: a short slice, then a resumed slice, and
# the submission re-scored. Parallel over the machine's cores.
#   campaign/tests/smoke_all.sh [seconds per slice, default 8] [parallel jobs]
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$DIR")"
S=${1:-8}
J=${2:-$(nproc)}
T=$(mktemp -d)
cd "$REPO"
PY="${CAMPAIGN_PYTHON:-python3}"
export PY S T DIR
jobs=$("$PY" -c "
import sys; sys.path.insert(0, '$DIR')
import config as C
for i in C.INSTANCES:
    for m in C.CPU_METHODS:
        print(i, m)")
echo "$jobs" | xargs -P "$J" -L 1 bash -c '
  inst=$0; m=$1; rd="$T/$inst/$m"; mkdir -p "$rd"
  "$PY" "$DIR/run_method.py" --method $m --instance data/$inst.gr --run-dir "$rd" --until-budget $S --quiet > "$rd/out1.txt" 2>&1 || { echo "FAIL first slice $inst/$m"; exit 0; }
  "$PY" "$DIR/run_method.py" --method $m --instance data/$inst.gr --run-dir "$rd" --until-budget $((S * 3 / 2)) --quiet > "$rd/out2.txt" 2>&1 || { echo "FAIL resume $inst/$m"; exit 0; }
  grep -q "resumed from" "$rd/log.txt" || echo "FAIL no resume $inst/$m"
  echo "ok $inst/$m $(tail -1 "$rd/out2.txt")"' > "$T/report.txt"
"$PY" - "$T" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ["DIR"])
import aggregate
T = sys.argv[1]
bad = [l for l in open(f"{T}/report.txt") if not l.startswith("ok")]
checked = mism = 0
for inst in sorted(os.listdir(T)):
    p = os.path.join(T, inst)
    if not os.path.isdir(p):
        continue
    for m in sorted(os.listdir(p)):
        b = os.path.join(p, m, "best.json")
        if not os.path.exists(b):
            continue
        best = json.load(open(b))
        s = aggregate.engine_check(inst, best)
        checked += 1
        if s != best["score"]:
            mism += 1
            bad.append(f"re-score mismatch {inst}/{m}: {s} vs {best['score']}\n")
lines = open(f"{T}/report.txt").read().splitlines()
print(f"{len(lines)} runs, {len(bad)} problems, {checked} submissions re-scored, {mism} mismatches")
for l in bad:
    print(l.rstrip())
sys.exit(1 if bad else 0)
PYEOF
echo "details: $T/report.txt"
