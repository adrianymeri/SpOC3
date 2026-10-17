"""
config.py -- what the campaign runs, where, and for how long.

Edit here, nowhere else. The supervisor re-reads this file when it starts.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
STATE = os.path.join(HERE, "state")

# ---------------------------------------------------------------- instances
OFFICIAL = ["small-graph", "medium-graph", "large-graph"]
SYNTH_FAMILY = {}
for i in (1, 2, 3, 8, 9, 10, 11, 12, 13, 14):
    SYNTH_FAMILY[f"synth-{i}"] = "small"
for i in (4, 5, 15, 16, 17, 18, 19, 20, 21, 22):
    SYNTH_FAMILY[f"synth-{i}"] = "medium"
for i in (6, 7, 23, 24, 25, 26, 27, 28, 29, 30):
    SYNTH_FAMILY[f"synth-{i}"] = "large"
SYNTHETIC = [f"synth-{i}" for i in range(1, 31)]
INSTANCES = OFFICIAL + SYNTHETIC
FAMILY = {"small-graph": "small", "medium-graph": "medium", "large-graph": "large", **SYNTH_FAMILY}


def instance_path(name):
    return os.path.join(DATA, name + ".gr")


# ------------------------------------------------------------------ methods
# order = column order in every table
CPU_METHODS = ["ea1", "ea2", "nsga2", "ne1", "ne2", "hri", "hc", "fcmaes"]
GPU_METHODS = ["cuda_torso"]
METHODS = CPU_METHODS + GPU_METHODS
LABEL = {
    "ea1": "EA-1 MEMETIC-PERM",
    "ea2": "EA-2 TORSO-QD",
    "nsga2": "EA-3 NSGA-II",
    "ne1": "NE-1 ANE",
    "ne2": "NE-2 AQD-NE",
    "hri": "Team HRI",
    "hc": "Hill Climbing",
    "fcmaes": "fcmaes",
    "cuda_torso": "Spacekangaroos",
}
FAMILY_OF_METHOD = {"ea1": "ours", "ea2": "ours", "nsga2": "ours", "ne1": "ours", "ne2": "ours",
                    "hri": "baseline", "hc": "baseline", "fcmaes": "baseline", "cuda_torso": "baseline"}
SEED = 1

# ------------------------------------------------------------------- budgets
# Campaign end (server local time). Every run is stopped cleanly at this time.
END = os.environ.get("CAMPAIGN_END", "2026-11-15T23:59:00")

# Synthetic CPU runs advance in rounds: nobody gets more CPU time until every
# synthetic run has reached the current level. Hours of CPU per run.
SYNTH_LEVELS_H = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 96]

# GPU (cuda-torso) rounds, hours of GPU wall time per synthetic instance;
# official instances get OFFICIAL_GPU_WEIGHT times as much in every round.
GPU_LEVELS_H = [0.5, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
OFFICIAL_GPU_WEIGHT = 3

# ------------------------------------------------------------------- layout
# One run per physical core; the hyper-thread siblings stay idle so timing is
# clean. The first physical core is kept for the system: the supervisor, the
# aggregator and cuda-torso's host process (on its sibling thread).
OFFICIAL_SLOTS = len(OFFICIAL) * len(CPU_METHODS)     # 24
RESERVED_CORES = 1

HEARTBEAT_STALE_S = 15 * 60       # a run silent this long is restarted
AGGREGATE_EVERY_S = 10 * 60
MAX_CRASHES_PER_HOUR = 6          # beyond this a run is parked and flagged

# Python interpreters
PYTHON = os.environ.get("CAMPAIGN_PYTHON", "python3")
GPU_PYTHON = os.environ.get("CAMPAIGN_GPU_PYTHON",
                            os.path.join(HERE, "venv-gpu", "bin", "python")
                            if os.path.exists(os.path.join(HERE, "venv-gpu", "bin", "python")) else "python3")
CUDA_TORSO_BACKEND = os.environ.get("CUDA_TORSO_BACKEND", "")   # "" = decided by install.sh


# ------------------------------------------------------------- test mode
# CAMPAIGN_TEST=<dir> runs a miniature campaign in <dir> (tests/test_campaign.sh)
if os.environ.get("CAMPAIGN_TEST"):
    _T = os.environ["CAMPAIGN_TEST"]
    RUNS, RESULTS, STATE = (os.path.join(_T, x) for x in ("runs", "results", "state"))
    OFFICIAL = os.environ.get("CAMPAIGN_TEST_OFFICIAL", "small-graph").split(",")
    SYNTHETIC = os.environ.get("CAMPAIGN_TEST_SYNTHETIC", "synth-1,synth-4").split(",")
    INSTANCES = OFFICIAL + SYNTHETIC
    CPU_METHODS = os.environ.get("CAMPAIGN_TEST_METHODS", "hri,ne1").split(",")
    # CAMPAIGN_TEST_GPU=1 adds cuda-torso (tests/test_gpu_setup.sh runs it on the CPU emulation)
    GPU_METHODS = ["cuda_torso"] if os.environ.get("CAMPAIGN_TEST_GPU") else []
    GPU_LEVELS_H = [float(x) / 3600 for x in os.environ.get("CAMPAIGN_TEST_GPU_LEVELS_S", "15,30").split(",")]
    METHODS = CPU_METHODS + GPU_METHODS
    OFFICIAL_SLOTS = len(OFFICIAL) * len(CPU_METHODS)
    SYNTH_LEVELS_H = [float(x) / 3600 for x in os.environ.get("CAMPAIGN_TEST_LEVELS_S", "10,20").split(",")]
    AGGREGATE_EVERY_S = 30
    HEARTBEAT_STALE_S = 120


def run_dir(instance, method):
    return os.path.join(RUNS, instance, method)
