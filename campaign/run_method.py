#!/usr/bin/env python3
"""
run_method.py -- run (or resume) one method on one instance.

    python3 campaign/run_method.py --method hri --instance data/small-graph.gr \
        --run-dir campaign/runs/small-graph/hri [--until-budget 3600] [--deadline 2026-11-15T23:59]

Resumes automatically from <run-dir>/ckpt.pkl. --until-budget stops the run
(after a checkpoint) once its accumulated CPU budget reaches that many seconds;
the supervisor uses it for the rounds. Without it the run goes on until it is
signalled (SIGTERM/SIGINT) or the deadline passes.
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import Runner  # noqa: E402


def registry():
    from methods.baselines import HRI, HC, FCMAES
    from methods.nsga2 import NSGA2
    reg = {c.name: c for c in (HRI, HC, FCMAES, NSGA2)}
    try:
        from methods.memetic import MemeticPerm
        reg[MemeticPerm.name] = MemeticPerm
    except ImportError:
        pass
    try:
        from methods.torso_qd import TorsoQD
        reg[TorsoQD.name] = TorsoQD
    except ImportError:
        pass
    try:
        from methods.neuro import ANE, AQDNE
        reg[ANE.name] = ANE
        reg[AQDNE.name] = AQDNE
    except ImportError:
        pass
    return reg


def parse_deadline(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(s, fmt))       # server local time
        except ValueError:
            pass
    raise SystemExit(f"bad deadline {s!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--until-budget", type=float, default=None)
    ap.add_argument("--deadline", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--param", action="append", default=[],
                    help="override a method parameter, key=value (value parsed as JSON); tests only")
    a = ap.parse_args()
    import json
    params = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        try:
            params[k] = json.loads(v)
        except ValueError:
            params[k] = v
    reg = registry()
    if a.method not in reg:
        raise SystemExit(f"unknown method {a.method}; known: {', '.join(sorted(reg))}")
    r = Runner(reg[a.method], a.run_dir, a.instance, seed=a.seed, params=params,
               until_budget=a.until_budget, deadline_epoch=parse_deadline(a.deadline), quiet=a.quiet)
    reason = r.run()
    print(f"{a.method} on {os.path.basename(a.instance)}: {reason}, budget {r.budget():.0f}s, "
          f"score {r.ctx.arch.score():,}", flush=True)


if __name__ == "__main__":
    main()
