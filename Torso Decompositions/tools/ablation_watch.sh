#!/usr/bin/env bash
# ablation_watch.sh -- track the small-graph GBDT-on-SOTA ablation from the live
# logs of the three arms (run on the server, where the logs live).
#
#   Arm A = run_gbdt.py            (GBDT-augmented)
#   Arm B = run_gbdt.py --no_gbdt  (control, identical seed/budget)
#   Arm C = run.py                 (stock cuda-torso baseline)
#
# Reports each arm's best internal HVI and Δ = (A − B), the GBDT booster's
# controlled contribution.  Internal HVI is the engine's own estimate -- use it
# for the A-vs-B TREND; re-score the actual checkpoint with verify_submission.py
# before quoting a final number.
#
#   ./ablation_watch.sh                 # uses ~/cuda-torso
#   watch -n 60 ./ablation_watch.sh     # live, refreshing every minute

LOGDIR="${1:-$HOME/cuda-torso}"

# best (most-negative) "official -X" value in a run_gbdt log
bestlog() { grep -oE 'official -[0-9,]+' "$1" 2>/dev/null | sed 's/official //; s/,//g' \
            | sort -n | head -1; }
# best HVI (last pipe-field) in a stock run.py log
beststock() { awk -F'|' 'NF>=8{print $NF}' "$1" 2>/dev/null | sort -n | head -1; }
gen() { tail -1 "$1" 2>/dev/null; }

A=$(bestlog   "$LOGDIR/small_A_gbdt.log")
B=$(bestlog   "$LOGDIR/small_B_control.log")
C=$(beststock "$LOGDIR/small_C_stock.log")

fmt() { [ -n "$1" ] && printf "%'d" "$1" || echo "n/a"; }

echo "=== small-graph GBDT ablation ($(date '+%H:%M:%S')) ==="
echo "  Arm A  GBDT      best HVI : $(fmt "$A")"
echo "  Arm B  control   best HVI : $(fmt "$B")"
echo "  Arm C  stock     best HVI : $(fmt "$C")"
if [ -n "$A" ] && [ -n "$B" ]; then
  D=$(( B - A ))                       # both negative; A more-negative ⇒ Δ>0 ⇒ GBDT ahead
  echo "  Δ (A − B)  GBDT contribution : $(printf "%+d" "$D")   (positive = GBDT ahead of control)"
fi
echo "  reference  banked best (torso-deletion §13): -1,829,913   leader: -1,829,919"
echo "  --- latest gen lines ---"
echo "  A: $(gen "$LOGDIR/small_A_gbdt.log")"
echo "  B: $(gen "$LOGDIR/small_B_control.log")"
echo "  C: $(gen "$LOGDIR/small_C_stock.log")"
