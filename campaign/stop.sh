#!/usr/bin/env bash
# Stop the campaign cleanly: every run writes a final checkpoint, the results
# are aggregated one last time (with esa_eval checks). Nothing is lost:
# 'campaign/start.sh --resume' continues every run from where it stopped.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/state"
echo "stop requested $(date '+%F %T')" > "$DIR/state/STOP"
echo "stopping (runs checkpoint first; this can take up to a few minutes)..."
for i in $(seq 1 600); do
  if flock -n "$DIR/state/supervisor.lock" true 2>/dev/null; then break; fi
  sleep 1
done
if ! flock -n "$DIR/state/supervisor.lock" true 2>/dev/null; then
  echo "supervisor still busy after 10 minutes; check campaign/state/supervisor.log"; exit 1
fi
cd "$(dirname "$DIR")" && "${CAMPAIGN_PYTHON:-python3}" "$DIR/aggregate.py" --final --print
echo "stopped. Results: campaign/results/ (make a copy with campaign/snapshot.sh)"
