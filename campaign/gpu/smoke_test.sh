#!/usr/bin/env bash
# cuda-torso on the V100: 60 s on small-graph, then a resume for 30 s more, then
# 60 s on large-graph (the batch that crashed the unpatched kernel).
set -eu
DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$DIR")"
GPY="${CAMPAIGN_GPU_PYTHON:-python3}"
B=$(cat "$DIR/state/cuda_backend" 2>/dev/null || echo nvcc)
T=$(mktemp -d)
cd "$REPO"
"$GPY" "$DIR/gpu/cuda_torso.py" --instance data/small-graph.gr --run-dir "$T/small" --until-budget 60 --backend "$B"
"$GPY" "$DIR/gpu/cuda_torso.py" --instance data/small-graph.gr --run-dir "$T/small" --until-budget 90 --backend "$B"
grep -q "resumed at generation" "$T/small/log.txt"
"$GPY" "$DIR/gpu/cuda_torso.py" --instance data/large-graph.gr --run-dir "$T/large" --until-budget 60 --backend "$B"
"${CAMPAIGN_PYTHON:-python3}" - "$T" <<'PYEOF'
import json, sys
t = sys.argv[1]
for k in ("small", "large"):
    d = json.load(open(f"{t}/{k}/status.json"))
    print(k, "generations", d["generation"], "budget", d["budget_s"], "score", d["score"], "own selection", d["own_selection_score"])
    assert d["generation"] > 0 and d["score"] < 0
PYEOF
rm -rf "$T"
echo "GPU smoke test passed"
