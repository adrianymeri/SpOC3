#!/usr/bin/env bash
# Pack the current results into one small archive to copy to the Mac:
#   campaign/snapshots/ola2027-YYYYmmdd-HHMM.tgz
# Copy it into SpOC3/campaign-results/ on the Mac.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$DIR")"
(cd "$REPO" && "${CAMPAIGN_PYTHON:-python3}" "$DIR/aggregate.py") || true
mkdir -p "$DIR/snapshots"
OUT="$DIR/snapshots/ola2027-$(date +%Y%m%d-%H%M).tgz"
cd "$DIR" || exit 1
LIST=$(mktemp)
find results -maxdepth 1 -type f 2>/dev/null >> "$LIST"
find results/verify -type f 2>/dev/null >> "$LIST"
find runs -maxdepth 3 -type f \( -name status.json -o -name best.json -o -name anytime.csv -o -name log.txt -o -name own_submission.json \) 2>/dev/null >> "$LIST"
for f in state/supervisor_status.json state/supervisor.log state/gpu_setup.log state/gpu_smoke.log; do [ -f "$f" ] && echo "$f" >> "$LIST"; done
tar czf "$OUT" -T "$LIST"
rm -f "$LIST"
echo "$OUT ($(du -h "$OUT" | cut -f1))"
