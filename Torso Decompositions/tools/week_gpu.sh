#!/usr/bin/env bash
# week_gpu.sh -- unattended week-long GPU campaign: small -> medium -> large.
#
# Crash-proof by design: every stage is a retry loop; qnegbfc re-seeds itself
# from the banked submissions on every (re)start, and checkpoints its own
# submission every ~2 minutes, so a crash loses almost nothing. Each restart
# uses a fresh random seed (extra basin diversity for free).
#
#   chmod +x tools/week_gpu.sh
#   nohup bash tools/week_gpu.sh > logs/week_nohup.log 2>&1 &
#
# Default allocation (edit below): small 3 days, medium 2 days, large 2 days.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

log () { echo "[$(date '+%F %T')] $*" | tee -a logs/week.log; }

run_stage () {           # $1=problem  $2=seconds  $3=batch
    local prob=$1 secs=$2 batch=$3
    local end=$(( $(date +%s) + secs ))
    log "=== stage $prob: $((secs/3600))h, batch=$batch ==="
    while :; do
        local left=$(( end - $(date +%s) ))
        [ "$left" -lt 600 ] && break
        log "$prob: (re)starting qnegbfc, ${left}s remaining"
        timeout "$left" python3 tools/qnegbfc.py --problem "$prob" \
            --batch "$batch" --budget "$left" --seed "$((RANDOM + 1))" \
            >> "logs/qnegbfc_${prob}.log" 2>&1
        rc=$?
        log "$prob: qnegbfc exited rc=$rc"
        [ "$rc" -eq 124 ] && break          # timeout = stage time used up
        sleep 15                             # crash backoff, then resume
    done
    log "$prob: merging + verifying"
    python3 tools/portfolio.py --problems "$prob" >> logs/week.log 2>&1
    python3 tools/verify_submission.py "submissions/${prob}/portfolio.json" \
        >> logs/week.log 2>&1
    log "=== stage $prob done ==="
}

# ---- gate: never burn a week on an unvalidated (or absent!) GPU ------------
# 1. HARD requirement: numba must see the CUDA device. validate_gpu.py alone
#    is not sufficient -- it validates the numpy reference and exits 0 even
#    when the GPU path cannot run.
if ! python3 -c "from numba import cuda; import sys; sys.exit(0 if cuda.is_available() else 1)" \
        >> logs/week.log 2>&1; then
    log "FATAL: numba CUDA not available for THIS python3."
    log "  fix:  python3 -m pip install numba   (same interpreter!)"
    log "  test: python3 -c 'from numba import cuda; print(cuda.is_available())'"
    exit 1
fi
# 2. the validation must actually EXECUTE a GPU kernel (validate_gpu exits 0
#    even when it only validated the numpy reference, e.g. on kernel-compile
#    failures like nvJitLink/NVVM version mismatches)
python3 tools/fastwalk.py small-graph >> logs/week.log 2>&1
python3 tools/validate_gpu.py --problem small-graph --batch 512 \
        > logs/validate_last.log 2>&1
cat logs/validate_last.log >> logs/week.log
if ! grep -q "GPU vs CPU status mismatches: 0" logs/validate_last.log; then
    log "FATAL: GPU kernel did not run bit-exact (see logs/validate_last.log)."
    log "  If it shows an nvJitLink/NVVM error:"
    log "    python3 -m pip install -U nvidia-nvjitlink-cu12 numba-cuda"
    exit 1
fi
log "GPU kernel validated bit-exact; campaign begins"

# ---- the week ---------------------------------------------------------------
run_stage small-graph  $(( 3*86400 )) 8192
run_stage medium-graph $(( 2*86400 )) 8192
run_stage large-graph  $(( 2*86400 )) 4096

log "campaign complete"
python3 tools/portfolio.py >> logs/week.log 2>&1
log "final scores:"
for p in small-graph medium-graph large-graph; do
    python3 tools/verify_submission.py "submissions/${p}/portfolio.json" \
        2>/dev/null | grep -E "Official|gap" | sed "s/^/  ${p}: /" \
        | tee -a logs/week.log
done
