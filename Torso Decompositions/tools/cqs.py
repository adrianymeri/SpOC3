#!/usr/bin/env python3
"""
cqs.py -- P1: Certificate-Quota Set-space search (large-graph).

Torso width is a property of the head SET (fill from eliminating a set is
order-independent), and the packing certificate pins |S ∩ K_i| at every open
width. CQS therefore searches SET exchanges directly, which no permutation
arm (ours or the 2024 winners') proposes at range:

  moves (all exact-evaluated with the C kernel, all banked):
    swap-ext   : kept external <-> evicted external of the SAME clique
                 (positions exchanged in the inherited ordering);
    swap-twin  : kept twin <-> evicted twin of the same clique;
    sacrifice  : move a torso component-vertex to the head boundary (t+1);
    rescue     : move a head component-vertex into the torso (t-1), tried at
                 three insertion slots (torso start / matched clique block /
                 torso end);
  acceptance  : lexicographic (over-width excess, t) at the target width;
                every intermediate staircase point banks into the shared
                archive -> submissions/large-graph/cqs.json.

    python3 tools/cqs.py --width 130 --iters 200000
    python3 tools/cqs.py --width auto            # top bandit pick
"""
from __future__ import annotations
import argparse, collections, glob, json, os, random, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

PROBLEM = "large-graph"


def structure(n, adj):
    h = {}
    for v in range(n):
        h.setdefault(frozenset(adj[v] | {v}), []).append(v)
    Ks, twins = [], []
    for vs in h.values():
        if len(vs) >= 50:
            Ks.append(sorted(set(vs) | set(adj[vs[0]]))); twins.append(sorted(vs))
    order = sorted(range(len(Ks)), key=lambda i: -len(Ks[i]))
    Ks = [set(Ks[i]) for i in order]; twins = [set(twins[i]) for i in order]
    glue = set().union(*Ks)
    who = {}
    for i, K in enumerate(Ks):
        for v in K:
            who[v] = i
    return Ks, twins, glue, who


def best_achiever(n, ev, w):
    best = None
    for fp in glob.glob(os.path.join(HERE, "submissions", PROBLEM, "*.json")):
        if fp.endswith("_platform.json"):
            continue
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e.get("decisionVector", []):
                if not (isinstance(dv, list) and len(dv) == n + 1):
                    continue
                p = [int(x) for x in dv[:-1]]
                if sorted(p) != list(range(n)):
                    continue
                df = ev.full(p)
                if int(max(df)) > MAX_TW:
                    continue
                r = 0; cand = None
                for t in range(n - 1, -1, -1):
                    c = int(df[t]); r = c if c > r else r
                    if r <= w:
                        cand = t          # r is monotone: keep the smallest t
                    elif r > w:
                        break
                if cand is not None and (best is None or cand < best[0]):
                    best = (cand, list(p))
        except Exception:
            pass
    return best


def evaluate(perm, t, w, n, ev, arc):
    if not (0 <= t < n):
        return None
    df = ev.full(perm)
    if int(max(df)) > MAX_TW:
        return None
    r = 0
    for tt in range(n - 1, -1, -1):
        c = int(df[tt]); r = c if c > r else r
        if r <= MAX_TW:
            arc.try_add(r, tt, perm)
    wt = max(int(df[i]) for i in range(t, n))
    return (max(0, wt - w), t)          # lexicographic score, lower better


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", default="130")
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algo", default="cqs")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    target = LEADERBOARD_TARGETS[PROBLEM]
    Ks, twins, glue, who = structure(n, adj)
    # creator-informed bias (2026-07-12 attachment-overlap measurement): the
    # head should hold LOW-attachment externals (their torso attachments get
    # clique-ified through the connected head glue); the torso should keep
    # HIGH-attachment externals. Tournament selection below uses this.
    attc = {v: len(adj[v] - glue) for v in glue}

    if a.width == "auto":
        st_fp = os.path.join(HERE, "submissions", PROBLEM, ".bandit_state.json")
        w = 130
        if os.path.exists(st_fp):
            st = json.load(open(st_fp))
            arms = sorted(st.get("arms", {}).items(),
                          key=lambda kv: -kv[1]["r"])
            if arms:
                w = int(arms[0][0])
    else:
        w = int(a.width)

    arc = ParetoArchive()
    got = best_achiever(n, ev, w)
    if got is None:
        print(f"no valid achiever at width {w}; aborting"); return
    t, perm = got
    print(f"CQS width {w}: warm start t={t} (torso {n-t})", flush=True)
    cur = evaluate(perm, t, w, n, ev, arc)
    best = cur
    out = os.path.join(HERE, "submissions", PROBLEM, f"{a.algo}.json")

    def save():
        top = arc.top_k_by_hv_contribution(60, n)
        write_submission([list(p) + [int(tt)] for (_, tt, p) in top], PROBLEM, out)
    save()

    pos = {v: i for i, v in enumerate(perm)}
    accepts, t0 = 0, time.time()
    for it in range(1, a.iters + 1):
        p2 = list(perm); t2 = t
        m = rng.random()
        if m < 0.45:                       # swap-ext / swap-twin within a clique
            ci = rng.randrange(3)
            headK = [v for v in Ks[ci] if pos[v] < t]
            torsK = [v for v in Ks[ci] if pos[v] >= t]
            if not headK or not torsK:
                continue
            # tournament-3: evict-from-head the attachment-heavy, rescue-to-
            # head the attachment-light (creator-informed; still stochastic)
            hsel = max(rng.sample(headK, min(3, len(headK))),
                       key=lambda v: attc[v])
            tsel = min(rng.sample(torsK, min(3, len(torsK))),
                       key=lambda v: attc[v])
            i, j = pos[hsel], pos[tsel]
            p2[i], p2[j] = p2[j], p2[i]
        elif m < 0.75:                     # rescue: head comp-vertex -> torso
            headC = [v for v in p2[:t] if v not in glue]
            if not headC:
                continue
            v = rng.choice(headC)
            p2.remove(v)
            slot = rng.choice([t2 - 1, min(n - 1, t2 + rng.randrange(1, 200)),
                               n - 1])
            p2.insert(slot, v)
            t2 = t2 - 1
        else:                              # sacrifice: torso vertex -> head end
            v = p2[rng.randrange(t, n)]
            p2.remove(v)
            p2.insert(t2, v)
            t2 = t2 + 1
        sc = evaluate(p2, t2, w, n, ev, arc)
        if sc is None:
            continue
        if sc < cur or (sc == cur and rng.random() < 0.02):
            perm, t, cur = p2, t2, sc
            pos = {v: i for i, v in enumerate(perm)}
            if sc < best:
                best = sc; accepts += 1
                save()
                print(f"  *** it {it}: excess={sc[0]} t={sc[1]} "
                      f"(torso {n-sc[1]}) ***", flush=True)
        if it % 2000 == 0:
            print(f"  [it {it}/{a.iters} w={w} best excess={best[0]} t={best[1]} "
                  f"{accepts} accepts {time.time()-t0:.0f}s]", flush=True)
    save()
    top = arc.top_k_by_hv_contribution(20, n)
    cap = -hypervolume_2d([(x, tt) for x, tt, _ in top], n)
    print(f"final: best t={best[1]} at width {w} | archive cap20 {cap:,.0f} "
          f"(target {target:,.0f})")


if __name__ == "__main__":
    main()
