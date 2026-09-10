#!/bin/bash
cd "$(dirname "$0")/.."
ALGO="${1:-gbdt_grow}"
MODE="${2:-}"
STINT=10800
while true; do
  SEED=$RANDOM
  echo "=== [$(date '+%d.%m %H:%M')] stint start | algo=$ALGO seed=$SEED $MODE ==="
  python3 -u tools/gbdt_grow.py --problem medium-graph \
      --algo "$ALGO" --seed "$SEED" --budget $STINT $MODE
  echo "=== [$(date '+%d.%m %H:%M')] stint end (restarting on fresh pool) ==="
  sleep 5
done
