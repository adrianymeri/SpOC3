#!/usr/bin/env bash
# GPU part of the setup (cuda-torso). Safe to run while the campaign is
# running; the CPU runs are not affected. install.sh calls it too.
#     campaign/gpu/setup.sh
#
# 1. A Python whose torch runs kernels on this GPU: the campaign's Python
#    first; else campaign/venv-gpu, inheriting that Python's packages and
#    adding scipy; else campaign/venv-gpu with PyTorch's CUDA 12.6 build (the
#    last build with code for Volta GPUs such as the V100).
# 2. The kernel, built for this GPU's architecture: nvcc on the PATH, then
#    /usr/local/cuda/bin/nvcc, then CuPy/NVRTC. The first one whose results
#    equal the exact CPU emulation on small-graph and large-graph is kept.
# 3. The smoke test: 60 s on small-graph, a resume, 60 s on large-graph.
# 4. On success: state/cuda_backend, CAMPAIGN_GPU_PYTHON in state/env.sh, and
#    state/NO_GPU removed. A running supervisor is restarted so that it reads
#    the new settings; its runs checkpoint and continue.
# On failure state/NO_GPU holds the reason and nothing else changes.
# Details: campaign/state/gpu_setup.log and campaign/state/gpu_smoke.log.
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO" || exit 1
mkdir -p "$DIR/state"
LOG="$DIR/state/gpu_setup.log"
: > "$LOG"
ok()   { echo "  [ok]   $*"; }
warn() { echo "  [warn] $*"; }
fail() { echo "$*" > "$DIR/state/NO_GPU"; echo "  [FAIL] $*"; echo "         cuda-torso will not run; the CPU campaign is not affected"; exit 1; }

# setup work stays on the system core (core 0 and its hyper-thread sibling),
# away from the pinned campaign runs, and single-threaded
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SYS_CPUS=$(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list 2>/dev/null || echo 0)
PIN=""
command -v taskset >/dev/null && PIN="taskset -c $SYS_CPUS"

[ -f "$DIR/state/env.sh" ] && . "$DIR/state/env.sh"
PY="${CAMPAIGN_PYTHON:-$(command -v python3)}"
# no new GPU session starts while this runs; success lifts it, failure keeps it
echo "campaign/gpu/setup.sh did not finish" > "$DIR/state/NO_GPU"

# ------------------------------------------------------------------ the GPU
{ command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; } || fail "no working nvidia-smi"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
[ -n "$CC" ] || fail "nvidia-smi does not report the compute capability"
ARCH="sm_${CC/./}"
ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute capability $CC ($ARCH)"

# ----------------------------------------------------------- 1. the Python
torch_runs() {      # $1 python, $2 "scipy" to require scipy too
  "$1" - "${2:-}" >> "$LOG" 2>&1 <<'EOF'
import sys
import torch
if sys.argv[1:] == ["scipy"]:
    import scipy  # noqa: F401
x = torch.ones(8, device="cuda")
assert int((x * 2).sum().item()) == 16, "wrong result from the GPU"
print(sys.executable, "torch", torch.__version__, "CUDA", torch.version.cuda, "runs on", torch.cuda.get_device_name(0))
EOF
}
VENV="$DIR/venv-gpu"
GPY=""
if torch_runs "$PY" scipy; then
  GPY="$PY"
  ok "$PY: torch $("$PY" -c 'import torch; print(torch.__version__)') runs on the GPU"
else
  if torch_runs "$PY"; then
    # its torch works but scipy is missing: a venv that inherits its packages
    if [ ! -x "$VENV/bin/python" ]; then
      "$PY" -m venv --system-site-packages "$VENV" >> "$LOG" 2>&1 || fail "python3 -m venv failed (sudo apt install python3-venv)"
    fi
    "$VENV/bin/python" -m pip install -q scipy >> "$LOG" 2>&1
    torch_runs "$VENV/bin/python" scipy && GPY="$VENV/bin/python" && ok "campaign/venv-gpu: $PY's torch plus scipy"
  fi
  if [ -z "$GPY" ]; then
    echo "  installing PyTorch's CUDA 12.6 build into campaign/venv-gpu (a few GB)..."
    if [ ! -x "$VENV/bin/python" ]; then
      "$PY" -m venv "$VENV" >> "$LOG" 2>&1 || fail "python3 -m venv failed (sudo apt install python3-venv)"
    fi
    "$VENV/bin/python" -m pip install -q --upgrade pip >> "$LOG" 2>&1
    "$VENV/bin/python" -m pip install -q numpy scipy >> "$LOG" 2>&1
    "$VENV/bin/python" -m pip install -q "torch<2.15" --index-url https://download.pytorch.org/whl/cu126 >> "$LOG" 2>&1
    torch_runs "$VENV/bin/python" scipy && GPY="$VENV/bin/python" \
      && ok "campaign/venv-gpu: torch $("$GPY" -c 'import torch; print(torch.__version__)') runs on the GPU"
  fi
fi
[ -n "$GPY" ] || fail "no Python whose torch runs on this GPU (campaign/state/gpu_setup.log)"

# ----------------------------------------------------------- 2. the kernel
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
check() {           # $1 backend: the kernel against the CPU emulation
  local inst rc
  for inst in small-graph large-graph; do
    echo "--- self-check, backend $1, $inst" >> "$LOG"
    $PIN "$GPY" "$DIR/gpu/cuda_torso.py" --instance "data/$inst.gr" --run-dir "$T/check-$1-$inst" \
      --backend "$1" --self-check-only >> "$LOG" 2>&1
    rc=$?
    cat "$T/check-$1-$inst/log.txt" >> "$LOG" 2>/dev/null
    [ $rc -eq 0 ] || return 1
  done
}
BACKEND=""
NVCCS="$(command -v nvcc 2>/dev/null || true)"
[ -x /usr/local/cuda/bin/nvcc ] && [ "/usr/local/cuda/bin/nvcc" != "$NVCCS" ] && NVCCS="$NVCCS /usr/local/cuda/bin/nvcc"
for NVCC in $NVCCS; do
  echo "--- build with $NVCC -arch=$ARCH" >> "$LOG"
  rm -f "$DIR/gpu/libeval_fixed.so"
  if (cd "$DIR/gpu" && $PIN "$NVCC" -arch="$ARCH" -Xcompiler -fPIC -shared -o libeval_fixed.so libeval_fixed.cu) >> "$LOG" 2>&1; then
    if check nvcc; then
      BACKEND=nvcc
      ok "kernel built with $NVCC ($("$NVCC" --version | grep -o 'release [0-9.]*' | head -1)) for $ARCH; it matches the CPU emulation on small-graph and large-graph"
      break
    fi
    warn "the kernel built with $NVCC does not match the CPU emulation"
  else
    warn "$NVCC cannot build for $ARCH"
  fi
done
if [ -z "$BACKEND" ]; then
  rm -f "$DIR/gpu/libeval_fixed.so"
  # CuPy only goes into the campaign's own venv, never into a system Python
  if ! "$GPY" -c 'import cupy' 2>/dev/null && [ "$GPY" = "$VENV/bin/python" ]; then
    "$GPY" -m pip install -q cupy-cuda12x >> "$LOG" 2>&1
  fi
  if "$GPY" -c 'import cupy' 2>/dev/null && check nvrtc; then
    BACKEND=nvrtc
    ok "kernel compiled at run time (CuPy/NVRTC); it matches the CPU emulation"
  fi
fi
if [ -z "$BACKEND" ]; then
  tail -n 15 "$LOG" | sed 's/^/         | /'
  fail "no build of the kernel matches the CPU emulation (the end of campaign/state/gpu_setup.log is above)"
fi

# ------------------------------------------------------- 3. the smoke test
echo "$BACKEND" > "$DIR/state/cuda_backend"
echo "  GPU smoke test (about 4 minutes)..."
if CAMPAIGN_GPU_PYTHON="$GPY" CAMPAIGN_PYTHON="$PY" $PIN bash "$DIR/gpu/smoke_test.sh" > "$DIR/state/gpu_smoke.log" 2>&1; then
  ok "cuda-torso runs, checkpoints and resumes on the GPU (campaign/state/gpu_smoke.log)"
else
  rm -f "$DIR/state/cuda_backend"
  tail -n 15 "$DIR/state/gpu_smoke.log" | sed 's/^/         | /'
  fail "GPU smoke test failed (the end of campaign/state/gpu_smoke.log is above)"
fi

# --------------------------------------------------------------- 4. enable
touch "$DIR/state/env.sh"
{ grep -v '^export CAMPAIGN_GPU_PYTHON=' "$DIR/state/env.sh"; echo "export CAMPAIGN_GPU_PYTHON=$GPY"; } > "$DIR/state/env.sh.tmp"
mv "$DIR/state/env.sh.tmp" "$DIR/state/env.sh"
if ! flock -n "$DIR/state/supervisor.lock" true 2>/dev/null; then
  # a supervisor is running with the settings it read at its start: replace it.
  # Its runs keep going; the new one checkpoints and restarts them at once.
  SUP=$(cat "$DIR/state/supervisor.lock" 2>/dev/null)
  [ -n "$SUP" ] && kill -9 "$SUP" 2>/dev/null
  for _ in $(seq 1 30); do flock -n "$DIR/state/supervisor.lock" true 2>/dev/null && break; sleep 1; done
  rm -f "$DIR/state/NO_GPU"
  bash "$DIR/start.sh" > /dev/null
  ok "supervisor restarted with the GPU enabled (its runs resume from their checkpoints)"
else
  rm -f "$DIR/state/NO_GPU"
  ok "GPU enabled; cuda-torso runs once the supervisor starts"
fi
