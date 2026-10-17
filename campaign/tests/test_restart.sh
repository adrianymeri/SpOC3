#!/usr/bin/env bash
# restart.sh on a running miniature campaign: the supervisor is replaced, its
# runs checkpoint and continue from their checkpoints under the new one.
#   campaign/tests/test_restart.sh
set -u
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
T=$(mktemp -d)
REPO="$T/repo"
fail() { echo "FAIL: $*"; tail -20 "$REPO/campaign/state/supervisor.log" 2>/dev/null; exit 1; }
mkdir -p "$REPO"
cp -r "$SRC/campaign" "$SRC/data" "$SRC/leaderboard-references" "$REPO/"
cp "$SRC"/*.py "$REPO/" 2>/dev/null
rm -rf "$REPO/campaign/runs" "$REPO/campaign/results" "$REPO/campaign/state" "$REPO/campaign/snapshots"
mkdir -p "$REPO/campaign/state"
make -s -C "$REPO/campaign/engine" >/dev/null || fail "engine build"
export CAMPAIGN_TEST="$REPO/campaign" CAMPAIGN_FAKE_CORES=4 CAMPAIGN_CKPT_EVERY=5
export CAMPAIGN_TEST_OFFICIAL=small-graph CAMPAIGN_TEST_SYNTHETIC=synth-1
export CAMPAIGN_TEST_METHODS=hri,ne1 CAMPAIGN_TEST_LEVELS_S=30,60
CAMPAIGN_END=$(date -d '+20 min' +%Y-%m-%dT%H:%M:%S)
export CAMPAIGN_END
echo "export CAMPAIGN_PYTHON=$(command -v python3)" > "$REPO/campaign/state/env.sh"
cd "$REPO" || exit 1
bash campaign/start.sh
sleep 30
OLD=$(cat campaign/state/supervisor.lock)
grep -q "start small-graph/hri" campaign/state/supervisor.log || fail "runs did not start"
echo "-- restart.sh"
bash campaign/restart.sh || fail "restart.sh exit $?"
NEW=$(cat campaign/state/supervisor.lock)
[ -n "$NEW" ] && [ "$NEW" != "$OLD" ] || fail "supervisor not replaced"
for _ in $(seq 1 10); do          # gone, or a zombie waiting to be reaped
  st=$(ps -o stat= -p "$OLD" 2>/dev/null | tr -d ' ')
  { [ -z "$st" ] || [ "${st#Z}" != "$st" ]; } && break
  sleep 1
done
st=$(ps -o stat= -p "$OLD" 2>/dev/null | tr -d ' ')
{ [ -z "$st" ] || [ "${st#Z}" != "$st" ]; } || fail "old supervisor still alive ($st)"
for _ in $(seq 1 60); do
  grep -q "resumed from" campaign/runs/small-graph/hri/log.txt 2>/dev/null && \
    grep -q "resumed from" campaign/runs/small-graph/ne1/log.txt 2>/dev/null && break
  sleep 2
done
grep -q "stopped .* orphan runs" campaign/state/supervisor.log || fail "orphans not stopped"
grep -q "resumed from" campaign/runs/small-graph/hri/log.txt || fail "hri did not resume"
grep -q "resumed from" campaign/runs/small-graph/ne1/log.txt || fail "ne1 did not resume"
grep -q "exited with code" campaign/state/supervisor.log && fail "a run crashed"
bash campaign/stop.sh > "$T/stop.out" 2>&1
grep -q "ALERTS: none" campaign/results/STATUS.txt || fail "alerts after stop"
echo "restart test passed ($T)"
