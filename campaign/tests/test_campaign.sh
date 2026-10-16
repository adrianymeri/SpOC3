#!/usr/bin/env bash
# A miniature campaign, end to end: supervisor, pinning, rounds, a supervisor
# crash (orphans adopted), a run crash (restart from checkpoint), STOP, final
# aggregation. Runs in a temporary directory; the real campaign is untouched.
#   campaign/tests/test_campaign.sh
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$DIR")"
T=$(mktemp -d)
export CAMPAIGN_TEST="$T" CAMPAIGN_FAKE_CORES=7 CAMPAIGN_CKPT_EVERY=5
export CAMPAIGN_TEST_OFFICIAL=small-graph CAMPAIGN_TEST_SYNTHETIC=synth-1,synth-4
export CAMPAIGN_TEST_METHODS=hri,ne1,ea2 CAMPAIGN_TEST_LEVELS_S=10,20
export CAMPAIGN_END=$(date -d '+20 min' +%Y-%m-%dT%H:%M:%S)
PY="${CAMPAIGN_PYTHON:-python3}"
cd "$REPO"
fail() { echo "FAIL: $*"; tail -30 "$T/state/supervisor.log"; exit 1; }

echo "test dir $T"
"$PY" "$DIR/supervisor.py" > "$T/sup1.out" 2>&1 &
SUP=$!
sleep 45
echo "-- supervisor crash (SIGKILL); its runs keep going as orphans"
kill -9 $SUP; sleep 3
pgrep -f "run_method.py.*$T" >/dev/null || fail "no orphan runs found after the supervisor died"
"$PY" "$DIR/supervisor.py" > "$T/sup2.out" 2>&1 &
SUP=$!
sleep 30
grep -q "stopped .* orphan runs" "$T/state/supervisor.log" || fail "new supervisor did not adopt the orphans"
echo "-- kill -9 one official run; it must come back from its checkpoint"
P=$(cat "$T/runs/small-graph/hri/lock")
kill -9 "$P"; sleep 25
P2=$(cat "$T/runs/small-graph/hri/lock")
[ "$P2" != "$P" ] || fail "official run was not restarted"
grep -q "resumed from" "$T/runs/small-graph/hri/log.txt" || fail "official run did not resume from a checkpoint"
echo "-- wait for the 20 s round to complete on every synthetic run"
for i in $(seq 1 120); do
  done=$("$PY" - <<'PYEOF'
import json, os
T = os.environ["CAMPAIGN_TEST"]
ok = 0
for inst in ("synth-1", "synth-4"):
    for m in ("hri", "ne1", "ea2"):
        try:
            ok += json.load(open(f"{T}/runs/{inst}/{m}/status.json"))["budget_s"] >= 15
        except Exception:
            pass
print(ok)
PYEOF
)
  [ "$done" = "6" ] && break
  sleep 5
done
[ "$done" = "6" ] || fail "synthetic rounds did not complete"
echo "-- STOP"
echo test > "$T/state/STOP"
for i in $(seq 1 120); do kill -0 $SUP 2>/dev/null || break; sleep 2; done
kill -0 $SUP 2>/dev/null && fail "supervisor did not stop"
pgrep -f "run_method.py.*$T" >/dev/null && fail "runs still alive after STOP"
"$PY" - <<'PYEOF' || exit 1
import csv, json, os, sys
T = os.environ["CAMPAIGN_TEST"]
bad = []
for inst in ("small-graph", "synth-1", "synth-4"):
    for m in ("hri", "ne1", "ea2"):
        st = json.load(open(f"{T}/runs/{inst}/{m}/status.json"))
        best = json.load(open(f"{T}/runs/{inst}/{m}/best.json"))
        if st["score"] != best["score"] or st["budget_s"] <= 0:
            bad.append(f"{inst}/{m}")
        v = json.load(open(f"{T}/results/verify/{inst}__{m}.json"))
        if not v.get("engine_ok") or not v.get("esa_ok"):
            bad.append(f"{inst}/{m} check {v}")
rows = list(csv.reader(open(f"{T}/results/fair_synthetic.csv")))
print("fair table:", rows[0][:5], "...", len(rows) - 1, "rows at level", rows[1][2])
print(open(f"{T}/results/STATUS.txt").read())
if bad:
    print("FAIL:", bad); sys.exit(1)
PYEOF
echo "campaign test passed ($T)"
