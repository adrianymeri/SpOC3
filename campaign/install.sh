#!/usr/bin/env bash
# One-time setup on the server, from the repository root:
#     campaign/install.sh            check, build, test, set up the GPU, install the cron entries
#     campaign/install.sh --no-gpu   CPU methods only (cuda-torso is skipped; campaign/gpu/setup.sh adds it later)
# Safe to run again; it does not touch existing runs.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$DIR")"
PY="$(command -v "${CAMPAIGN_PYTHON:-python3}")" || { echo "python3 not found"; exit 1; }
cd "$REPO" || exit 1
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
echo "export CAMPAIGN_PYTHON=$PY" > "$DIR/state/env.sh"
if [ "${1:-}" = "--no-gpu" ]; then
  rm -f "$DIR/state/cuda_backend"
  echo "skipped by --no-gpu" > "$DIR/state/NO_GPU"
  warn "GPU skipped: cuda-torso will not run (campaign/gpu/setup.sh adds it later)"
else
  bash "$DIR/gpu/setup.sh" || warn "campaign/gpu/setup.sh can be run again later"
fi

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
