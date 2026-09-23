#!/bin/bash
# status.sh -- is the Mac run healthy? Run it any time.
#
#     ./status.sh
#
# Expected when finished: hill_climbing 30, hri 30, fast_cma_es 30,
# min_degree 10  (min_degree does one seed -- it has no randomness to average).

HERE="$(cd "$(dirname "$0")" && pwd)"
NOW=$(date +%s)

echo "=============================================================="
echo " MAC RUN STATUS          $(date '+%a %d %b %H:%M')"
echo "=============================================================="

RUNNING=$(pgrep -f "bench_one.py" | wc -l | tr -d ' ')
MD=$(( $( [ -f "$HERE/benchmark-min_degree.csv" ] && wc -l < "$HERE/benchmark-min_degree.csv" || echo 1 ) - 1 ))
EXPECT=4; [ "$MD" -ge 10 ] && EXPECT=3      # min_degree finishes in minutes
echo
echo "processes alive: $RUNNING of $EXPECT"
[ "$MD" -ge 10 ] && echo "  (min_degree already finished -- 3 is correct)"
if [ "$RUNNING" -gt 0 ]; then
    ps -o pid,etime,%cpu,comm -p $(pgrep -f bench_one.py | tr '\n' ',' | sed 's/,$//') 2>/dev/null | sed 's/^/  /'
fi

echo
echo "progress:"
TOTAL=0
for s in hill_climbing hri fast_cma_es min_degree; do
    f="$HERE/benchmark-$s.csv"
    want=30; [ "$s" = "min_degree" ] && want=10
    if [ -f "$f" ]; then
        n=$(($(wc -l < "$f") - 1))
        TOTAL=$((TOTAL + n))
        age=$(( (NOW - $(stat -f %m "$f")) / 60 ))
        bad=$(grep -c ",False" "$f" 2>/dev/null)   # grep -c prints 0 on no match
        bad=${bad:-0}
        note=""
        [ "$n" -ge "$want" ] && note="  <- complete"
        [ "$bad" -gt 0 ] && note="$note  ** $bad INVALID **"
        printf "  %-15s %3d/%-3d  last row %3d min ago%s\n" "$s" "$n" "$want" "$age" "$note"
    else
        printf "  %-15s   no CSV yet\n" "$s"
    fi
done
echo "  ----------------------------------------"
printf "  %-15s %3d/100\n" "TOTAL" "$TOTAL"

echo
echo "errors in logs (want none):"
if grep -l -iE "traceback|error" "$HERE"/logs/*.log >/dev/null 2>&1; then
    grep -l -iE "traceback|error" "$HERE"/logs/*.log | sed 's/^/  PROBLEM: /'
else
    echo "  none"
fi

echo
echo "latest line per solver:"
for s in hill_climbing hri fast_cma_es min_degree; do
    line=$(tail -n 1 "$HERE/logs/$s.log" 2>/dev/null)
    printf "  %s\n" "${line:-  ($s: no log yet)}"
done

echo
if [ "$TOTAL" -ge 100 ]; then
    echo "MAC SIDE COMPLETE. Once Kaggle finishes, drop"
    echo "benchmark-spacekangaroos.csv in this folder and run:"
    echo "    python3 $HERE/merge_results.py $HERE"
elif [ "$RUNNING" -eq 0 ]; then
    echo "NOTHING IS RUNNING and the run is incomplete."
    echo "Restart with ./run_mac.sh -- it skips whatever is already done."
else
    LEFT=$(( (100 - TOTAL) * 20 / (RUNNING > 0 ? RUNNING : 1) ))
    echo "healthy. roughly $((LEFT / 60))h $((LEFT % 60))m of solving left."
fi
echo
