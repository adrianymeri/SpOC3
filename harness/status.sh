#!/bin/bash
# status.sh -- is the Mac run healthy? Run it any time.
#
#     ./status.sh
#
# Expected when finished: 30 rows for each searching solver (10 instances x
# 3 seeds), 10 for min_degree -- it has no randomness, so one seed is all
# there is to run.
#
# The solver list and every total below are derived from SOLVERS, so adding
# an arm to run_mac.sh and to this list is enough. Nothing is hardcoded; the
# old version counted out of 100 long after the real total had become 310.

HERE="$(cd "$(dirname "$0")" && pwd)"
NOW=$(date +%s)

# Same list as run_mac.sh, plus min_degree last because it wants fewer rows.
SOLVERS=(hill_climbing hri fast_cma_es
         sa_front vns_front grasp_front hc_bottleneck
         simulated_annealing vns grasp min_degree)

rows_wanted() { [ "$1" = "min_degree" ] && echo 10 || echo 30; }

# mtime in seconds. macOS wants `stat -f %m`, GNU/Linux `stat -c %Y`; this
# script is run on both (the Mac for real runs, Linux when checking it).
# Branch on uname rather than trying one and falling back -- GNU `stat -f`
# SUCCEEDS on Linux and prints filesystem info, so `||` never fires.
case "$(uname)" in
    Darwin) mtime() { stat -f %m "$1" 2>/dev/null || echo "$NOW"; } ;;
    *)      mtime() { stat -c %Y "$1" 2>/dev/null || echo "$NOW"; } ;;
esac

WANT=0
for s in "${SOLVERS[@]}"; do WANT=$((WANT + $(rows_wanted "$s"))); done

echo "=============================================================="
echo " MAC RUN STATUS          $(date '+%a %d %b %H:%M')"
echo "=============================================================="

RUNNING=$(pgrep -f "bench_one.py" | wc -l | tr -d ' ')

echo
echo "progress:"
TOTAL=0
INCOMPLETE=0
for s in "${SOLVERS[@]}"; do
    f="$HERE/benchmark-$s.csv"
    want=$(rows_wanted "$s")
    if [ -f "$f" ]; then
        n=$(($(wc -l < "$f") - 1))
        TOTAL=$((TOTAL + n))
        age=$(( (NOW - $(mtime "$f")) / 60 ))
        bad=$(grep -c ",False" "$f" 2>/dev/null); bad=${bad:-0}
        note=""
        if [ "$n" -ge "$want" ]; then note="  <- complete"
        else INCOMPLETE=$((INCOMPLETE + 1)); fi
        [ "$bad" -gt 0 ] && note="$note  ** $bad INVALID **"
        printf "  %-20s %3d/%-3d  last row %3d min ago%s\n" "$s" "$n" "$want" "$age" "$note"
    else
        INCOMPLETE=$((INCOMPLETE + 1))
        printf "  %-20s   no CSV yet\n" "$s"
    fi
done
echo "  ----------------------------------------"
printf "  %-20s %3d/%d\n" "TOTAL" "$TOTAL" "$WANT"

echo
echo "processes alive: $RUNNING  (solvers still incomplete: $INCOMPLETE)"
if [ "$RUNNING" -gt 0 ]; then
    ps -o pid,etime,%cpu,comm -p "$(pgrep -f bench_one.py | tr '\n' ',' | sed 's/,$//')" 2>/dev/null | sed 's/^/  /'
fi

# A run whose wall-clock falls outside this window was interrupted by
# something outside the solver -- a sleep, a shutdown, another process
# stealing the core -- and its score is not comparable. Re-run it rather
# than reporting it. This has caught five runs so far.
echo
echo "off-budget runs (wall-clock outside 1,150-1,400 s):"
found=0
for s in "${SOLVERS[@]}"; do
    f="$HERE/benchmark-$s.csv"
    [ -f "$f" ] || continue
    [ "$s" = "min_degree" ] && continue
    out=$(awk -F, -v S="$s" 'NR>1 && ($6+0 < 1150 || $6+0 > 1400){
            printf "  %-20s %-14s seed %s  %8.0f s\n", S, $1, $4, $6 }' "$f")
    [ -n "$out" ] && { echo "$out"; found=1; }
done
[ "$found" -eq 0 ] && echo "  none"

echo
echo "errors in logs (want none):"
if grep -l -iE "traceback|error" "$HERE"/logs/*.log >/dev/null 2>&1; then
    grep -l -iE "traceback|error" "$HERE"/logs/*.log | sed 's/^/  PROBLEM: /'
else
    echo "  none"
fi

echo
echo "latest line per solver:"
for s in "${SOLVERS[@]}"; do
    line=$(tail -n 1 "$HERE/logs/$s.log" 2>/dev/null)
    printf "  %s\n" "${line:-  ($s: no log yet)}"
done

echo
if [ "$TOTAL" -ge "$WANT" ]; then
    echo "MAC SIDE COMPLETE ($TOTAL/$WANT rows). Regenerate the tables with:"
    echo "    python3 $HERE/report.py"
    echo "Once Kaggle finishes, drop benchmark-spacekangaroos.csv here and run:"
    echo "    python3 $HERE/merge_results.py $HERE"
elif [ "$RUNNING" -eq 0 ]; then
    echo "NOTHING IS RUNNING and the run is incomplete ($TOTAL/$WANT rows)."
    echo "Restart with ./run_mac.sh -- it skips whatever is already done."
else
    LEFT=$(( (WANT - TOTAL) * 20 / RUNNING ))
    echo "healthy. roughly $((LEFT / 60))h $((LEFT % 60))m of solving left"
    echo "($((WANT - TOTAL)) rows x 20 min, spread over $RUNNING process(es))."
fi
echo
