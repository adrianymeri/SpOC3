#!/usr/bin/env bash
# Print the campaign status (refreshed every 10 minutes by the supervisor).
#   campaign/status.sh            the latest STATUS.txt
#   campaign/status.sh --refresh  re-aggregate first
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = "--refresh" ] || [ ! -f "$DIR/results/STATUS.txt" ]; then
  (cd "$(dirname "$DIR")" && "${CAMPAIGN_PYTHON:-python3}" "$DIR/aggregate.py") || true
fi
if flock -n "$DIR/state/supervisor.lock" true 2>/dev/null; then
  if [ -f "$DIR/state/STOP" ]; then echo "SUPERVISOR: stopped (STOP file present)"; else echo "SUPERVISOR: NOT RUNNING -- run campaign/start.sh"; fi
else
  echo "SUPERVISOR: running (pid $(cat "$DIR/state/supervisor.lock" 2>/dev/null))"
fi
cat "$DIR/results/STATUS.txt"
