#!/usr/bin/env bash
# Power cut: every process of a running miniature campaign is killed at once
# (SIGKILL, no chance to checkpoint), one latest checkpoint is additionally
# torn (truncated, as if the power went during the write), then the campaign
# is started again. Every run must resume from its last good checkpoint, the
# torn one from its previous checkpoint, budgets must continue from there, and
# the results must re-score exactly.
#   campaign/tests/test_power_loss.sh
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$DIR")"
T=$(mktemp -d)
export CAMPAIGN_TEST="$T" CAMPAIGN_FAKE_CORES=6 CAMPAIGN_CKPT_EVERY=6
export CAMPAIGN_TEST_OFFICIAL=small-graph,large-graph CAMPAIGN_TEST_SYNTHETIC=synth-4
export CAMPAIGN_TEST_METHODS=ne2,hc CAMPAIGN_TEST_LEVELS_S=15,30
export CAMPAIGN_END=$(date -d '+30 min' +%Y-%m-%dT%H:%M:%S)
PY="${CAMPAIGN_PYTHON:-python3}"
cd "$REPO"
fail() { echo "FAIL: $*"; exit 1; }
echo "test dir $T"
"$PY" "$DIR/supervisor.py" > "$T/sup1.out" 2>&1 &
sleep 50
echo "-- power cut: SIGKILL every campaign process at once"
pkill -9 -f "supervisor.py" ; pkill -9 -f "run_method.py.*$T"; sleep 2
pgrep -f "run_method.py.*$T" >/dev/null && fail "processes survived"
# what each run had checkpointed when the power went
"$PY" - <<'PYEOF' > "$T/before.json"
import json, os, pickle
T = os.environ["CAMPAIGN_TEST"]
out = {}
for inst in ("small-graph", "large-graph", "synth-4"):
    for m in ("ne2", "hc"):
        p = f"{T}/runs/{inst}/{m}/ckpt.pkl"
        if os.path.exists(p):
            d = pickle.load(open(p, "rb"))
            out[f"{inst}/{m}"] = {"budget": d["budget"], "reason": d.get("reason")}
print(json.dumps(out))
PYEOF
echo "-- tear the latest checkpoint of small-graph/hc (half written)"
f="$T/runs/small-graph/hc/ckpt.pkl"
prev_budget=$("$PY" -c "import pickle; print(pickle.load(open('$T/runs/small-graph/hc/ckpt.prev.pkl','rb'))['budget'])")
sz=$(stat -c %s "$f"); head -c $((sz / 2)) "$f" > "$f.half" && mv "$f.half" "$f"
echo "-- power back: start again"
"$PY" "$DIR/supervisor.py" > "$T/sup2.out" 2>&1 &
SUP=$!
sleep 40
echo stop > "$T/state/STOP"
for i in $(seq 1 120); do kill -0 $SUP 2>/dev/null || break; sleep 2; done
PREV_BUDGET=$prev_budget "$PY" - <<'PYEOF' || exit 1
import json, os, re, sys
sys.path.insert(0, os.path.join(os.environ.get("REPO_DIR", "."), "campaign"))
T = os.environ["CAMPAIGN_TEST"]
before = json.load(open(f"{T}/before.json"))
bad = []
for key, b in before.items():
    inst, m = key.split("/")
    if b.get("reason") == "slice-done":
        print(f"{key}: had finished its round before the cut (nothing to resume)")
        continue
    log = open(f"{T}/runs/{inst}/{m}/log.txt").read()
    res = re.findall(r"resumed from (\S+) at budget (\d+)s", log)
    if not res:
        bad.append(f"{key}: did not resume"); continue
    src, bud = res[-1 if key != "small-graph/hc" else 0]
    if key == "small-graph/hc":
        if src != "ckpt.prev.pkl" or abs(float(bud) - float(os.environ["PREV_BUDGET"])) > 1:
            bad.append(f"{key}: torn checkpoint not handled ({src}, {bud})")
    elif abs(float(bud) - b["budget"]) > 1:
        bad.append(f"{key}: resumed at {bud}s, checkpoint had {b['budget']:.0f}s")
    st = json.load(open(f"{T}/runs/{inst}/{m}/status.json"))
    if st["budget_s"] <= float(bud):
        bad.append(f"{key}: budget did not continue ({st['budget_s']} <= {bud})")
    v = json.load(open(f"{T}/results/verify/{inst}__{m}.json"))
    if not (v.get("engine_ok") and v.get("esa_ok")):
        bad.append(f"{key}: final check {v}")
    print(f"{key}: resumed from {src} at {bud}s, now {st['budget_s']}s, score {st['score']:,}, esa_eval ok")
if bad:
    print("FAIL:", *bad, sep="\n  "); sys.exit(1)
PYEOF
echo "power-loss test passed ($T)"
