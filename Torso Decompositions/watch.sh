#!/usr/bin/env bash
# ============================================================
# watch.sh -- ONE command, either machine, everything that matters.
#   * the three scores, each flagged BEAT / short by how much
#   * best endpoint width reached per instance (the winning lever)
#   * how many arms are alive, by type
# Read-only: starts and stops nothing.
#   usage:  bash watch.sh
# ============================================================
cd "$(dirname "$0")" || exit 1

echo "================ $(hostname -s)  $(date '+%d.%m %H:%M') ================"
echo "TARGETS:  small -1,829,919 | medium -1,745,122 | large -5,493,062"
echo "(score more negative than target = BEATING it)"
echo
if [ "$1" = "--quick" ]; then
  echo "-- SCORES: skipped (--quick).  Run 'bash watch.sh' for the real scores. --"
else
echo "-- SCORES --  (recomputed from the whole pool; takes 1-2 min)"
for p in small-graph medium-graph large-graph; do
  out=$(python3 tools/cap_submit.py --problem "$p" 2>/dev/null)   # ONE run per graph
  line=$(echo "$out" | grep -E "best-20")
  res=$(echo "$out"  | grep -E "residual" | sed 's/.*: *//')
  if echo "$line" | grep -q "gap *-"; then flag="*** BEAT ***"; else flag="still short"; fi
  printf "  %-13s %s   %s\n" "$p" "$(echo "$line" | sed 's/best-20 submission (VALID): *//')" "$flag"
  printf "  %-13s   envelope residual: %s\n" "" "$res"
done
fi

echo
echo "-- ENDPOINT WIDTH (the lever that beat medium; lower = better) --"
for p in small med large; do
  best=$(grep -ho "ENDPOINT WIDTH [0-9]*" endpoint_${p}*.log 2>/dev/null \
         | awk '{print $3}' | sort -n | head -1)
  if [ -n "$best" ]; then
    echo "  $p: best endpoint width reached = $best"
  else
    echo "  $p: (no endpoint arm log yet)"
  fi
done
echo "  NOTE large is PROVABLY OPTIMAL at 499 (K500 clique bound) -- do not run endpoint arms on it."

echo
echo "-- ARMS ALIVE --"
for pat in "endpoint_ils.py --problem small-graph:endpoint small" \
           "endpoint_ils.py --problem medium-graph:endpoint medium" \
           "gbfcpp.py --problem small-graph:gbfcpp small" \
           "gbfcpp.py --problem medium-graph:gbfcpp medium" \
           "gbfcpp_lowW:gbfcpp large lowW" \
           "gbfcpp_midW:gbfcpp large midW" \
           "gbdt_grow.py --problem small-graph:gbdt_grow small" \
           "run_gbdt.py:GPU lottery"; do
  key="${pat%%:*}"; label="${pat##*:}"
  c=$(pgrep -f "$key" 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-22s %s\n" "$label" "$c"
done
total=$(pgrep -f "endpoint_ils.py|gbfcpp.py|gbdt_grow.py|run_gbdt.py" 2>/dev/null | wc -l | tr -d ' ')
echo "  ---------------------- total: $total"
echo "  load:$(uptime | sed 's/.*load average://')"
echo
echo "if any score flips to BEAT:  python3 tools/verify_submission.py submissions/<problem>/cap20.json"
