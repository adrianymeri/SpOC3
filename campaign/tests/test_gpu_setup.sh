#!/usr/bin/env bash
# gpu/setup.sh and the supervisor's GPU rounds, end to end, on a machine
# without a GPU: a copy of the repository, a fake nvidia-smi and nvcc, and a
# Python wrapper that runs cuda_torso.py on the exact CPU emulation of the
# kernel (--device cpu, batch 8). A miniature campaign is running when
# setup.sh enables the GPU, as on the server. Checked:
#   - setup.sh picks the Python and the nvcc build, and the self-check passes
#     on small-graph and large-graph;
#   - it restarts the running supervisor; the CPU runs resume from checkpoints;
#   - the new supervisor runs cuda-torso slices that self-check, resume and
#     report a score that the engine re-scores identically.
# Needs torch (CPU is enough).   campaign/tests/test_gpu_setup.sh
set -u
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
T=$(mktemp -d)
REPO="$T/repo"
fail() { echo "FAIL: $*"; tail -30 "$REPO/campaign/state/supervisor.log" 2>/dev/null; exit 1; }
REAL_PY="$(command -v python3)"
"$REAL_PY" -c 'import torch' 2>/dev/null || fail "torch is needed (pip install torch)"

echo "test dir $T"
mkdir -p "$REPO"
cp -r "$SRC/campaign" "$SRC/data" "$SRC/leaderboard-references" "$REPO/"
cp "$SRC"/*.py "$REPO/" 2>/dev/null
rm -rf "$REPO/campaign/runs" "$REPO/campaign/results" "$REPO/campaign/state" "$REPO/campaign/venv-gpu" "$REPO/campaign/snapshots"
mkdir -p "$REPO/campaign/state"
make -s -C "$REPO/campaign/engine" >/dev/null || fail "engine build"

# fakes: GPU tools, and a Python whose cuda_torso.py runs on the CPU emulation
BIN="$T/bin"
mkdir -p "$BIN"
cat > "$BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *compute_cap*) echo "7.0" ;;
  *name*) echo "Fake V100" ;;
  *) echo "fake nvidia-smi" ;;
esac
EOF
cat > "$BIN/nvcc" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "--version" ]; then echo "Cuda compilation tools, release 11.5, V11.5.119"; exit 0; fi
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
[ -n "$out" ] && : > "$out"
EOF
cat > "$BIN/fakegpy" <<EOF
#!/usr/bin/env bash
if [ "\${1:-}" = "-" ]; then cat > /dev/null; echo "fake: torch runs on the GPU"; exit 0; fi
if [ "\${1:-}" = "-c" ]; then
  case "\$2" in *cupy*) exit 1 ;; *torch.__version__*) echo "0.0+fake"; exit 0 ;; esac
fi
case "\${1:-}" in *cuda_torso.py) exec "$REAL_PY" "\$@" --device cpu --batch_size 8 --eigenvectors 4 ;; esac
exec "$REAL_PY" "\$@"
EOF
chmod +x "$BIN"/*
# the smoke test runs on a real GPU only (its large-graph slice is too slow on
# the CPU emulation); the supervisor's slices below exercise the same code
printf '#!/usr/bin/env bash\necho "smoke test skipped in this test"\n' > "$REPO/campaign/gpu/smoke_test.sh"

export PATH="$BIN:$PATH"
export CAMPAIGN_TEST="$REPO/campaign" CAMPAIGN_FAKE_CORES=4 CAMPAIGN_CKPT_EVERY=5
export CAMPAIGN_TEST_OFFICIAL=small-graph CAMPAIGN_TEST_SYNTHETIC=synth-1
export CAMPAIGN_TEST_METHODS=hri CAMPAIGN_TEST_LEVELS_S=20,40
export CAMPAIGN_TEST_GPU=1 CAMPAIGN_TEST_GPU_LEVELS_S=10,20
export CAMPAIGN_END=$(date -d '+20 min' +%Y-%m-%dT%H:%M:%S)
echo "export CAMPAIGN_PYTHON=$BIN/fakegpy" > "$REPO/campaign/state/env.sh"
echo "skipped by --no-gpu" > "$REPO/campaign/state/NO_GPU"
cd "$REPO"

echo "-- a miniature campaign without the GPU"
bash campaign/start.sh
sleep 25
grep -q "start small-graph/hri" campaign/state/supervisor.log || fail "CPU runs did not start"
grep -q "cuda_torso" campaign/state/supervisor.log && fail "a GPU run started while NO_GPU was set"
OLD=$(cat campaign/state/supervisor.lock)

echo "-- campaign/gpu/setup.sh"
bash campaign/gpu/setup.sh || fail "setup.sh exit $?"
[ ! -e campaign/state/NO_GPU ] || fail "NO_GPU still there"
[ "$(cat campaign/state/cuda_backend)" = nvcc ] || fail "backend not nvcc"
grep -q "^export CAMPAIGN_GPU_PYTHON=$BIN/fakegpy$" campaign/state/env.sh || fail "env.sh: $(cat campaign/state/env.sh)"
grep -q "^export CAMPAIGN_PYTHON=$BIN/fakegpy$" campaign/state/env.sh || fail "env.sh lost CAMPAIGN_PYTHON"
[ "$(grep -c 'kernel self-check passed' campaign/state/gpu_setup.log)" -ge 2 ] || fail "self-check lines missing in gpu_setup.log"
for i in $(seq 1 30); do
  NEW=$(cat campaign/state/supervisor.lock 2>/dev/null)
  [ -n "$NEW" ] && [ "$NEW" != "$OLD" ] && break
  sleep 1
done
[ "$NEW" != "$OLD" ] || fail "supervisor was not restarted"

echo "-- the restarted supervisor runs cuda-torso slices"
for i in $(seq 1 120); do
  n=$(ls campaign/runs/*/cuda_torso/status.json 2>/dev/null | wc -l)
  [ "$n" -ge 2 ] && grep -q "resumed at generation" campaign/runs/*/cuda_torso/log.txt 2>/dev/null && break
  sleep 5
done
grep -q "stopped .* orphan runs" campaign/state/supervisor.log || fail "orphans not handled"
grep -q "resumed from" campaign/runs/small-graph/hri/log.txt || fail "CPU run did not resume after the restart"
grep -q "start .*/cuda_torso on cpu" campaign/state/supervisor.log || fail "no GPU run started"
grep -q "resumed at generation" campaign/runs/*/cuda_torso/log.txt || fail "no GPU slice resumed"
grep -h "kernel self-check passed" campaign/runs/*/cuda_torso/log.txt | head -2
grep -q "exited with code" campaign/state/supervisor.log && fail "a run crashed"

echo "-- stop, final aggregation"
bash campaign/stop.sh > "$T/stop.out" 2>&1
"$REAL_PY" - "$REPO" <<'PYEOF' || fail "result check"
import json, os, sys
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "campaign"))
os.chdir(repo)
import aggregate
for inst in ("small-graph", "synth-1"):
    rd = os.path.join(repo, "campaign", "runs", inst, "cuda_torso")
    st = json.load(open(os.path.join(rd, "status.json")))
    best = json.load(open(os.path.join(rd, "best.json")))
    s = aggregate.engine_check(inst, best)
    print(f"{inst}/cuda_torso: budget {st['budget_s']} s, generation {st['generation']}, score {st['score']:,}, "
          f"own selection {st['own_selection_score']:,}, engine re-score {s:,}")
    assert st["score"] < 0 and s == best["score"]
PYEOF
grep -A3 "OFFICIAL INSTANCES" "$REPO/campaign/results/STATUS.txt"
echo "gpu setup test passed ($T)"
