#!/usr/bin/env bash
# One-time setup on the server, from the repository root:
#     campaign/install.sh            check, build, test, set up the GPU, install the cron entries
#     campaign/install.sh --no-gpu   CPU methods only (cuda-torso is skipped)
# Safe to run again; it does not touch existing runs.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$DIR")"
PY="$(command -v "${CAMPAIGN_PYTHON:-python3}")" || { echo "python3 not found"; exit 1; }
cd "$REPO"
mkdir -p "$DIR/state" "$DIR/runs" "$DIR/results"
ok()   { echo "  [ok]   $*"; }
warn() { echo "  [warn] $*"; }
die()  { echo "  [FAIL] $*"; exit 1; }

echo "== 1. machine"
nproc_all=$(nproc --all 2>/dev/null || echo "?")
cores=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
echo "  logical CPUs $nproc_all, physical cores $cores, RAM $(free -g | awk '/Mem:/{print $2}') GB, free disk $(df -h "$REPO" | awk 'NR==2{print $4}')"
[ "$cores" -ge 26 ] || warn "fewer than 26 physical cores: official runs will share cores with the synthetic pool"

echo "== 2. toolchain"
command -v make >/dev/null || die "make missing (sudo apt install build-essential)"
command -v cc >/dev/null || die "a C compiler is missing (sudo apt install build-essential)"
$PY -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null || die "python3 >= 3.9 needed"
$PY -c 'import numpy' 2>/dev/null || die "numpy missing for $PY (pip install --user numpy)"
command -v flock >/dev/null || die "flock missing (util-linux)"
ok "python $($PY -c 'import sys; print(sys.version.split()[0])'), numpy $($PY -c 'import numpy; print(numpy.__version__)')"

echo "== 3. engine"
make -s -C "$DIR/engine" clean >/dev/null; make -s -C "$DIR/engine" || die "engine build failed"
ok "engine built (-march=native)"
$PY "$DIR/tests/test_engine.py" > "$DIR/state/test_engine.log" 2>&1 && ok "engine matches esa_eval (state/test_engine.log)" \
  || die "engine test failed: see campaign/state/test_engine.log"

echo "== 4. instance structure caches"
$PY - <<'PYEOF' || die "structure precomputation failed"
import sys, os
sys.path.insert(0, "campaign")
import config as C, torso_engine as te, structure
for inst in C.INSTANCES:
    structure.load(te.Engine(C.instance_path(inst)))
print("  [ok]   structure cached for", len(C.INSTANCES), "instances")
PYEOF

echo "== 5. GPU (cuda-torso)"
rm -f "$DIR/state/NO_GPU" "$DIR/state/cuda_backend"
: > "$DIR/state/env.sh"
if [ "${1:-}" = "--no-gpu" ]; then
  echo "skipped by --no-gpu" > "$DIR/state/NO_GPU"; warn "GPU skipped: cuda-torso will not run"
elif ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  echo "no working nvidia-smi" > "$DIR/state/NO_GPU"; warn "no GPU visible: cuda-torso will not run"
else
  GPY=""
  if "$PY" -c 'import torch, scipy; assert torch.cuda.is_available()' 2>/dev/null; then
    GPY="$PY"; ok "system python has torch with CUDA"
  else
    if [ ! -x "$DIR/venv-gpu/bin/python" ]; then
      echo "  creating campaign/venv-gpu (torch, scipy, numpy; a few GB download)..."
      $PY -m venv "$DIR/venv-gpu" || die "python3 -m venv failed (sudo apt install python3-venv)"
    fi
    "$DIR/venv-gpu/bin/pip" install -q --upgrade pip >/dev/null 2>&1
    "$DIR/venv-gpu/bin/pip" install -q numpy scipy torch || { echo "pip install torch failed" > "$DIR/state/NO_GPU"; warn "torch install failed: cuda-torso will not run"; }
    "$DIR/venv-gpu/bin/python" -c 'import torch, scipy; assert torch.cuda.is_available()' 2>/dev/null && GPY="$DIR/venv-gpu/bin/python"
  fi
  if [ -z "$GPY" ]; then
    [ -f "$DIR/state/NO_GPU" ] || echo "torch cannot see the GPU" > "$DIR/state/NO_GPU"
    warn "torch with CUDA not available: cuda-torso will not run (see campaign/README.md)"
  else
    echo "export CAMPAIGN_GPU_PYTHON=$GPY" >> "$DIR/state/env.sh"
    if command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]; then
      export PATH="$PATH:/usr/local/cuda/bin"
      (cd "$DIR/gpu" && nvcc -Xcompiler -fPIC -shared -o libeval_fixed.so libeval_fixed.cu) && echo nvcc > "$DIR/state/cuda_backend" && ok "kernel built with nvcc"
    fi
    if [ ! -s "$DIR/state/cuda_backend" ]; then
      "$GPY" -m pip install -q cupy-cuda12x >/dev/null 2>&1 && "$GPY" -c 'import cupy' 2>/dev/null \
        && echo nvrtc > "$DIR/state/cuda_backend" && ok "no nvcc: kernel compiled at run time through CuPy/NVRTC"
    fi
    if [ -s "$DIR/state/cuda_backend" ]; then
      echo "  GPU smoke test (about 3 minutes)..."
      if CAMPAIGN_GPU_PYTHON="$GPY" bash "$DIR/gpu/smoke_test.sh" > "$DIR/state/gpu_smoke.log" 2>&1; then
        ok "cuda-torso runs and resumes on the V100 (state/gpu_smoke.log)"
      else
        echo "smoke test failed" > "$DIR/state/NO_GPU"; warn "cuda-torso smoke test failed: see campaign/state/gpu_smoke.log"
      fi
    else
      echo "no nvcc and no cupy" > "$DIR/state/NO_GPU"; warn "could not build the kernel: cuda-torso will not run"
    fi
  fi
fi
echo "export CAMPAIGN_PYTHON=$PY" >> "$DIR/state/env.sh"

echo "== 6. start at boot + keepalive"
LINE1="@reboot $DIR/start.sh >> $DIR/state/cron.log 2>&1"
LINE2="*/5 * * * * $DIR/start.sh >> $DIR/state/cron.log 2>&1"
if command -v crontab >/dev/null && (crontab -l 2>/dev/null; true) >/dev/null; then
  ( crontab -l 2>/dev/null | grep -v "$DIR/start.sh"; echo "$LINE1"; echo "$LINE2" ) | crontab - \
    && ok "cron: start at boot and check every 5 minutes" || warn "crontab refused the entries"
else
  warn "no crontab: use the systemd user unit in campaign/README.md, or run campaign/start.sh after a reboot"
fi

echo
echo "Ready. Start the campaign with:   campaign/start.sh"
echo "Daily:  campaign/status.sh   |   copy to the Mac: campaign/snapshot.sh   |   stop: campaign/stop.sh"
