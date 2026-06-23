#!/usr/bin/env bash
# ablation_watch.sh [graph] [logdir] -- track the cuda-torso GBDT ablation for any
# graph from the live arm logs.
#
#   Arm A = run_gbdt.py            (GBDT-augmented, the method)
#   Arm B = run_gbdt.py --no_gbdt  (control, identical seed/budget)
#   Arm C = run.py                 (stock cuda-torso baseline)
#
# Reports each arm's best internal HVI and Δ = (A − B), the GBDT contribution.
# Internal HVI is the engine's own estimate -- use it for the A-vs-B TREND;
# re-score the actual checkpoint with verify_submission.py before quoting a final
# number.
#
#   ./ablation_watch.sh small            # default
#   ./ablation_watch.sh medium
#   ./ablation_watch.sh large ~/cuda-torso
#   watch -n 60 ./ablation_watch.sh medium

G="${1:-small}"
LOGDIR="${2:-$HOME/cuda-torso}"

case "$G" in
  small)  TGT="-1,829,919"; REF="-1,829,913 (torso-deletion §13)";;
  medium) TGT="-1,745,122"; REF="building";;
  large)  TGT="-5,493,062"; REF="building";;
  *)      TGT="?";          REF="?";;
esac

# best (most-negative) "official -X" value in a run_gbdt log
bestlog() { grep -oE 'official -[0-9,]+' "$1" 2>/dev/null | sed 's/official //; s/,//g' \
            | sort -n | head -1; }
# best HVI (last pipe-field) in a stock run.py log
beststock() { awk -F'|' 'NF>=8{print $NF}' "$1" 2>/dev/null | sort -n | head -1; }
gen() { tail -1 "$1" 2>/dev/null; }

A=$(bestlog   "$LOGDIR/${G}_A_gbdt.log")
B=$(bestlog   "$LOGDIR/${G}_B_control.log")
C=$(beststock "$LOGDIR/${G}_C_stock.log")

fmt() { [ -n "$1" ] && printf "%'d" "$1" || echo "n/a"; }

echo "=== ${G}-graph GBDT ablation ($(date '+%H:%M:%S')) ==="
echo "  Arm A  GBDT      best HVI : $(fmt "$A")"
echo "  Arm B  control   best HVI : $(fmt "$B")"
echo "  Arm C  stock     best HVI : $(fmt "$C")"
if [ -n "$A" ] && [ -n "$B" ]; then
  D=$(( B - A ))                       # both negative; A more-negative ⇒ Δ>0 ⇒ GBDT ahead
  echo "  Δ (A − B)  GBDT contribution : $(printf "%+d" "$D")   (positive = GBDT ahead of control)"
fi
echo "  target ${TGT}   reference ${REF}"
echo "  --- latest gen lines ---"
echo "  A: $(gen "$LOGDIR/${G}_A_gbdt.log")"
echo "  B: $(gen "$LOGDIR/${G}_B_control.log")"
echo "  C: $(gen "$LOGDIR/${G}_C_stock.log")"
