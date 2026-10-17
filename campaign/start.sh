#!/usr/bin/env bash
# Start the campaign supervisor if it is not running.
#   campaign/start.sh            used by cron at boot and every 5 minutes; does
#                                nothing while campaign/state/STOP exists
#   campaign/start.sh --resume   remove STOP first (restart a stopped campaign)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$DIR")"
mkdir -p "$DIR/state"
[ -f "$DIR/state/env.sh" ] && . "$DIR/state/env.sh"
if [ "${1:-}" = "--resume" ]; then rm -f "$DIR/state/STOP"; fi
if [ -f "$DIR/state/STOP" ]; then
  [ -t 1 ] && echo "campaign is stopped (campaign/state/STOP); 'campaign/start.sh --resume' restarts it"
  exit 0
fi
if flock -n "$DIR/state/supervisor.lock" true 2>/dev/null; then
  # exec: the supervisor replaces the background subshell, so no shell is
  # left waiting on it (and none reports "Killed" when it is replaced)
  (cd "$REPO" && exec nohup setsid "${CAMPAIGN_PYTHON:-python3}" "$DIR/supervisor.py" >> "$DIR/state/supervisor.out" 2>&1 < /dev/null) &
  [ -t 1 ] && echo "supervisor started (log: campaign/state/supervisor.log)"
else
  [ -t 1 ] && echo "supervisor already running (pid $(cat "$DIR/state/supervisor.lock" 2>/dev/null))"
fi
exit 0
