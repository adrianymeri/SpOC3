#!/usr/bin/env python3
"""
boundary_lns.py -- seam-window LNS for large-graph, w in [130,220] by default.

Motivation (THESIS §15.6): the rank_externals ablation showed the open
bottleneck at the slack widths is the HEAD-TAIL INTERACTION -- which glue
vertices sit just before the threshold t and which torso vertices sit just
after it, jointly. Neither static selection (GBDT ranker) nor single swaps
(perturbation sampling) move it. This arm therefore destroys and repairs a
WINDOW AROUND THE SEAM of the per-width achiever orderings:

  destroy: remove D vertices from positions [t-D/2, t+D/2) of the achiever
           for a target width (the seam window), plus optionally their
           original-graph neighbours inside the window band;
  repair:  reinsert in one of three orders (out-degree ascending, original
           relative order, shuffled) at the seam, i.e. the head part first,
           then the torso part -- the split point (how many of the removed
           go back into the head) is itself searched over;
  accept:  exact capped-20 HSSP hypervolume of the pooled archive improves,
           OR the achiever's t at its width improves (both monotone gains).

Resume-safe: pools submissions/large-graph/*.json at start, checkpoints
top-60 to submissions/large-graph/boundary_lns.json every accept.

    python3 tools/boundary_lns.py --iters 500000 --wmin 130 --wmax 221
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

PROBLEM = "large-graph"


def staircase_add(arc, perm, df, n):
    if int(max(df)) > MAX_TW:       # 2026-07-12: dirty head => void at ESA
        return
    r = 0
    for t in range(n - 1, -1, -1):
        c = int(df[t]); r = c if c > r else r
        if r <= MAX_TW:
            arc.try_add(r, t, perm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=PROBLEM,
                    choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--iters", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wmin", type=int, default=130)
    ap.add_argument("--wmax", type=int, default=221)
    ap.add_argument("--dmax", type=int, default=80, help="max seam window size")
    ap.add_argument("--algo", default="boundary_lns")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    prob = a.problem

    n, adj_l = load_graph(graph_path(HERE, prob))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    target = LEADERBOARD_TARGETS[prob]

    # pool everything (resume-safe union)
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(HERE, "submissions", prob, "*.json")):
        if fp.endswith("_platform.json"):
            continue
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e.get("decisionVector", []):
                if isinstance(dv, list) and len(dv) == n + 1:
                    p = [int(x) for x in dv[:-1]]
                    if sorted(p) == list(range(n)):
                        staircase_add(arc, p, ev.full(p), n)
        except Exception:
            pass

    def cap20():
        top = arc.top_k_by_hv_contribution(20, n)
        return -hypervolume_2d([(w, t) for w, t, _ in top], n)

    best = cap20()
    out = os.path.join(HERE, "submissions", prob, f"{a.algo}.json")

    def save():
        top = arc.top_k_by_hv_contribution(60, n)
        write_submission([list(p) + [int(t)] for (_, t, p) in top], prob, out)

    save()
    print(f"=== boundary-LNS {prob} | pooled capped-20 {best:,.0f} "
          f"gap {best - target:+,.0f} | seam widths [{a.wmin},{a.wmax}) ===", flush=True)

    # per-width achievers inside the attack band
    def achiever(w):
        cand = [(wt, t, p) for (wt, t, p) in arc.entries() if a.wmin <= wt < a.wmax]
        if not cand:
            return None
        wt, t, p = min(cand, key=lambda z: (abs(z[0] - w), z[1]))
        return wt, t, list(p)

    accepts, t0 = 0, time.time()
    for it in range(1, a.iters + 1):
        w = rng.randrange(a.wmin, a.wmax)
        got = achiever(w)
        if got is None:
            continue
        wt, t, perm = got
        D = rng.randint(8, a.dmax)
        lo = max(0, t - D // 2)
        hi = min(n, lo + D)
        window = perm[lo:hi]
        rest = perm[:lo] + perm[hi:]
        mode = rng.random()
        if mode < 0.4:      # out-degree ascending (low-degree vertices first back)
            window = sorted(window, key=lambda v: len(adj[v]))
        elif mode < 0.7:    # keep relative order (pure relocation search)
            pass
        else:               # shuffle
            rng.shuffle(window)
        # two move families across the seam:
        if rng.random() < 0.5:
            # (a) in-place reorder of the seam window
            cand = perm[:lo] + window + perm[hi:]
        else:
            # (b) relocate a chunk of the window deeper into the head
            split = rng.randint(1, len(window))
            chunk, keep = window[:split], window[split:]
            ins = rng.randint(0, lo)
            cand = perm[:ins] + chunk + perm[ins:lo] + keep + perm[hi:]
        if len(cand) != n:
            continue
        df = ev.full(cand)
        if max(int(x) for x in df) > MAX_TW:
            continue
        before = best
        staircase_add(arc, cand, df, n)
        cur = cap20()
        if cur < before - 1e-9:
            best = cur; accepts += 1
            save()
            print(f"  *** it {it} accept #{accepts}: capped-20 {best:,.0f} "
                  f"gap {best - target:+,.0f} (seam w={w}) ***", flush=True)
        if it % 2000 == 0:
            print(f"  [it {it}/{a.iters} capped-20 {best:,.0f} {accepts} accepts "
                  f"{time.time()-t0:.0f}s]", flush=True)

    print(f"final capped-20 {best:,.0f} gap {best - target:+,.0f} | {accepts} accepts")
    save()


if __name__ == "__main__":
    main()
