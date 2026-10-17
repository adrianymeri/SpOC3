#!/usr/bin/env bash
# Restart the supervisor so that every run picks up the current code (use it
# after `git pull`). The old supervisor is removed without touching its runs;
# the new one has them checkpoint and starts them again from their
# checkpoints. Each run loses a few seconds, nothing more.
#     campaign/restart.sh
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$DIR/state/STOP" ]; then
  echo "the campaign is stopped (campaign/state/STOP); campaign/start.sh --resume restarts it"
  exit 1
fi
OLD=$(cat "$DIR/state/supervisor.lock" 2>/dev/null)
if ! flock -n "$DIR/state/supervisor.lock" true 2>/dev/null; then
  [ -n "$OLD" ] && kill -9 "$OLD" 2>/dev/null
  for _ in $(seq 1 30); do flock -n "$DIR/state/supervisor.lock" true 2>/dev/null && break; sleep 1; done
fi
bash "$DIR/start.sh" > /dev/null
NEW=""
for _ in $(seq 1 30); do
  NEW=$(cat "$DIR/state/supervisor.lock" 2>/dev/null)
  [ -n "$NEW" ] && [ "$NEW" != "$OLD" ] && break
  sleep 1
done
echo "supervisor restarted (pid $NEW): every run resumes from its checkpoint with the current code"
