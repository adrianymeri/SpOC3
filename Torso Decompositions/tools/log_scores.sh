#!/bin/bash
# log_scores.sh -- append the current pooled scores to campaign_scores.csv.
# Run after every repool (and whenever curious). Powers the gap-vs-time
# thesis figure. CSV: timestamp,host,problem,best20,gap,envelope_residual
set -u
cd "$(dirname "$0")/.."
TS=$(date '+%Y-%m-%d %H:%M:%S'); H=$(hostname -s)
CSV=campaign_scores.csv
[ -f "$CSV" ] || echo "timestamp,host,problem,best20,gap,envelope_residual" > "$CSV"
for p in small-graph medium-graph large-graph; do
  OUT=$(python3 tools/cap_submit.py --problem "$p" 2>/dev/null)
  B=$(echo "$OUT" | grep -i "best-20" | tr -d ',' | grep -oE '[+-][0-9]+' | sed -n 1p)
  G=$(echo "$OUT" | grep -i "best-20" | tr -d ',' | grep -oE '[+-][0-9]+' | sed -n 2p)
  R=$(echo "$OUT" | grep -i "residual" | tr -d ',' | grep -oE '[+-][0-9]+' | sed -n 1p)
  echo "$TS,$H,$p,${B:-},${G:-},${R:-}" >> "$CSV"
done
tail -n 3 "$CSV"
