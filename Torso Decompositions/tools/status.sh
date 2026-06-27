#!/usr/bin/env bash
# status.sh [graph]  -- one-shot health + score + ablation snapshot.
#   ./tools/status.sh medium
G="${1:-medium}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

echo "============================================================"
echo " STATUS  ${G}-graph   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo "-- engines running --"
ps aux | grep -E "gbfcpp|gaps_search|torso_deletion|run_gbdt|run\.py|gbdt_mapelites|band_climb" \
       | grep -v grep | awk '{printf "  PID %-7s %s %s %s\n",$2,$11,$12,$13}'
[ -z "$(ps aux | grep -E 'gbfcpp|gaps_search|torso_deletion|run_gbdt|run\.py' | grep -v grep)" ] \
  && echo "  (NONE running)"

echo "-- best fronts (official re-score) --"
for f in gbfcpp gaps_search torso_del portfolio; do
  p="submissions/${G}-graph/${f}.json"
  [ -f "$p" ] || continue
  s=$(python3 tools/verify_submission.py "$p" 2>/dev/null | grep -oE 'Official score \(-HV\):\s*-[0-9,]+' | grep -oE '\-[0-9,]+')
  m=$(date -r "$p" '+%H:%M' 2>/dev/null)
  [ -n "$s" ] && printf "  %-12s %s   (updated %s)\n" "$f" "$s" "$m"
done

echo "-- GBDT ablation (cuda-torso arms) --"
./tools/ablation_watch.sh "$G" 2>/dev/null | sed -n '2,6p'

echo "-- pool everything + score --"
cp ~/cuda-torso/submissions/${G}-graph/*.json submissions/${G}-graph/ 2>/dev/null
python3 tools/portfolio.py --problem ${G}-graph 2>/dev/null | grep -E "best single|UNION|leaderboard target|gap"
echo "============================================================"
