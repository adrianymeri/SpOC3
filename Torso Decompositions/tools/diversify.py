#!/usr/bin/env python3
"""
diversify.py -- prefix diversification: attack the certified bottleneck.

THESIS §12.3b certifies that the small-graph tail breakpoints are OPTIMAL for
every banked prefix (complete w<=2 reduction), i.e. the residual ~180 HV is
locked behind prefixes (global fill patterns) the pool does not contain. This
driver hunts new prefix basins:

  wave = 1. K parallel FRESH CMA-ES runs (never-used seeds, min-degree warm
            start only -- each lands in its own basin), banked to
            submissions/<p>/seeds/cmaesdiv_s<seed>.json;
         2. EXACT tail push on every new prefix (tools/tail_exact reduction,
            w=1 and w=2): if a fresh prefix's fill pattern admits a longer
            width-1/2 tail, this finds it exactly and banks it;
         3. portfolio merge (folds anything new into portfolio.json);
         4. a short GBFC++ repair burst (pools the merged portfolio).

Ctrl-C safe at wave boundaries; everything is banked through submission files.

    caffeinate -i python3 tools/diversify.py --problem small-graph \
        --workers 4 --waves 40 --cma-budget 300
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, subprocess, time
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, MAX_TW, submission_path, write_submission,
                  LEADERBOARD_TARGETS)
from tools.gbfcpp import staircase, breakpoints, IncEval
from tools.tail_exact import fill_suffix_adj, reduce_order

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable or "python3"


def load_perms(fp, n):
    out = []
    try:
        for dv in json.load(open(fp))[0]["decisionVector"]:
            if isinstance(dv, list) and len(dv) == n + 1:
                p = [int(x) for x in dv[:-1]]
                if sorted(p) == list(range(n)):
                    out.append(p)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--waves", type=int, default=40)
    ap.add_argument("--cma-budget", type=float, default=300.0)
    ap.add_argument("--repair-budget", type=float, default=60.0,
                    help="GBFC++ repair seconds per wave (0 = skip)")
    ap.add_argument("--base-seed", type=int, default=20000)
    args = ap.parse_args()
    prob = args.problem
    here = repo_root()
    n, adj = load_graph(graph_path(here, prob)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(prob)
    try:
        from tools.fastwalk import IncEvalC
        ev = IncEvalC(ab, n)
    except Exception:
        ev = IncEval(ab, n)

    # persistent archive of everything diversification discovers
    arch = ParetoArchive()
    div_out = submission_path(here, prob, "divers")
    for fp in [os.path.join(here, "submissions", prob, "portfolio.json"), div_out]:
        if os.path.exists(fp):
            for p in load_perms(fp, n):
                s = staircase(ev.full(p))
                for wt, t in breakpoints(s, n):
                    if wt <= MAX_TW:
                        arch.try_add(wt, t, list(p))
    print(f"diversify -- {prob}: starting archive {-arch.hypervolume(n):,.0f}",
          flush=True)

    for wave in range(args.waves):
        t0 = time.time()
        # ---- 1. fresh CMA-ES basins, in parallel -----------------------
        seeds = [args.base_seed + wave * args.workers + i
                 for i in range(args.workers)]
        procs = [subprocess.Popen(
            [PY, os.path.join(HERE, "..", "algorithms", "continuous", "cmaes_torso.py"),
             "--problem", prob, "--budget", str(args.cma_budget),
             "--seed", str(s), "--algo", f"cmaesdiv_s{s}"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            for s in seeds]
        try:
            for p_ in procs:
                p_.wait()
        except KeyboardInterrupt:
            for p_ in procs:
                p_.terminate()
            print("\ninterrupted -- banked work is safe; rerun to resume")
            return

        # ---- 2. exact tail push on each fresh prefix --------------------
        Wpool = np.full(n, n, dtype=np.int64)
        for (_, _, pp) in arch.entries():
            Wpool = np.minimum(Wpool, staircase(ev.full(pp)))
        bps = dict(breakpoints(Wpool, n))
        pushed = []
        for s in seeds:
            fp = submission_path(here, prob, f"cmaesdiv_s{s}")
            for p in load_perms(fp, n)[:6]:
                st = staircase(ev.full(p))
                for wt, t in breakpoints(st, n):
                    if wt <= MAX_TW:
                        arch.try_add(wt, t, list(p))
                for w in (1, 2):
                    t_cur = bps.get(w)
                    if t_cur is None:
                        continue
                    t2 = t_cur - 1
                    q = list(p)
                    while t2 >= 0:
                        H, _ = fill_suffix_adj(q, t2, ab, n)
                        ro = reduce_order(H, w)
                        if ro is None:
                            break
                        q = q[:t2] + ro
                        s2 = staircase(ev.full(q))
                        for wt, t in breakpoints(s2, n):
                            if wt <= MAX_TW:
                                arch.try_add(wt, t, list(q))
                        pushed.append((w, t2))
                        bps[w] = t2
                        t2 -= 1
        top = arch.top_k_by_hv_contribution(20, n)
        write_submission([list(p)+[int(t)] for (_, t, p) in top], prob, div_out)

        # ---- 3. merge + 4. GBFC++ repair --------------------------------
        subprocess.run([PY, os.path.join(HERE, "portfolio.py"), "--problems", prob],
                       cwd=ROOT, capture_output=True)
        if args.repair_budget > 0:
            subprocess.run([PY, os.path.join(HERE, "gbfcpp.py"), "--problem", prob,
                            "--rounds", "1", "--round-budget", str(args.repair_budget),
                            "--seed", str(3000 + wave), "--t0", "0.4"],
                           cwd=ROOT, capture_output=True)
            subprocess.run([PY, os.path.join(HERE, "portfolio.py"), "--problems", prob],
                           cwd=ROOT, capture_output=True)
            # refresh archive with whatever repair found
            for p in load_perms(os.path.join(here, "submissions", prob,
                                             "portfolio.json"), n):
                s2 = staircase(ev.full(p))
                for wt, t in breakpoints(s2, n):
                    if wt <= MAX_TW:
                        arch.try_add(wt, t, list(p))

        score = -arch.hypervolume(n)
        msg = (f"== wave {wave+1}/{args.waves} | {len(seeds)} fresh basins | "
               f"exact pushes: {len(pushed)} {sorted(set(pushed)) if pushed else ''} | "
               f"score {score:,.0f}")
        if target is not None:
            msg += f" | gap {score-target:+,.0f}" + (" BEAT!" if score < target else "")
        msg += f" | {time.time()-t0:,.0f}s =="
        print(msg, flush=True)

    print("\ndone; best banked in portfolio.json / divers.json")


if __name__ == "__main__":
    main()
