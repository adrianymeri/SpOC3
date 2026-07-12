#!/usr/bin/env python3
"""
quotient_lns.py -- Twin-Normal-Form LNS for medium-graph (THESIS §15 follow-up).

Medium is a twin blow-up: 882 true-twin supervertices (weights <= 4). Exchange
lemma (Twin Normal Form): reordering any elimination ordering so that each
true-twin class is consecutive never increases any step width -- so searching
QUOTIENT orderings (882! instead of 1399!) loses nothing. This arm:

  - twin-normalizes the pooled best orderings into quotient seeds (keeping a
    normalized seed only if the exact evaluator confirms it is not worse);
  - runs destroy/repair LNS directly on the quotient permutation (remove k
    supervertices, reinsert at random positions), expanding to the full
    ordering for exact evaluation via the C kernel;
  - accepts on the exact capped-20 HSSP hypervolume; checkpoints top-60 to
    submissions/medium-graph/quotient_lns.json.

    python3 tools/quotient_lns.py --iters 500000
"""
from __future__ import annotations
import argparse, collections, glob, json, os, random, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

PROBLEM = "medium-graph"


def classes_of(n, adj):
    h = {}
    for v in range(n):
        h.setdefault(frozenset(adj[v] | {v}), []).append(v)
    cls = [sorted(c) for c in h.values()]
    cls.sort(key=lambda c: c[0])
    cid = {}
    for i, c in enumerate(cls):
        for v in c:
            cid[v] = i
    return cls, cid


def expand(qperm, cls):
    return [v for i in qperm for v in cls[i]]


def normalize(perm, cid, ncls):
    """quotient order = classes by first occurrence."""
    seen, q = set(), []
    for v in perm:
        i = cid[v]
        if i not in seen:
            seen.add(i); q.append(i)
    return q


def bank(arc, perm, df, n):
    """returns (valid, archive_improved)"""
    if int(max(df)) > MAX_TW:
        return False, False
    r = 0
    improved = False
    for t in range(n - 1, -1, -1):
        c = int(df[t]); r = c if c > r else r
        if r <= MAX_TW:
            if arc.try_add(r, t, perm):
                improved = True
    return True, improved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--algo", default="quotient_lns")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    target = LEADERBOARD_TARGETS[PROBLEM]
    cls, cid = classes_of(n, adj)
    ncls = len(cls)
    print(f"quotient: {ncls} supervertices (from n={n})", flush=True)

    # pool + twin-normalized seeds
    arc = ParetoArchive()
    seeds = []
    for fp in glob.glob(os.path.join(HERE, "submissions", PROBLEM, "*.json")):
        if fp.endswith("_platform.json"):
            continue
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e.get("decisionVector", []):
                if isinstance(dv, list) and len(dv) == n + 1:
                    p = [int(x) for x in dv[:-1]]
                    if sorted(p) != list(range(n)):
                        continue
                    df = ev.full(p)
                    ok, _ = bank(arc, p, df, n)
                    if not ok:
                        continue
                    q = normalize(p, cid, ncls)
                    p2 = expand(q, cls)
                    ok2, _ = bank(arc, p2, df2 := ev.full(p2), n)
                    if ok2:
                        seeds.append(q)
        except Exception:
            pass

    def cap20():
        top = arc.top_k_by_hv_contribution(20, n)
        return -hypervolume_2d([(w, t) for w, t, _ in top], n)

    best = cap20()
    out = os.path.join(HERE, "submissions", PROBLEM, f"{a.algo}.json")

    def save():
        top = arc.top_k_by_hv_contribution(60, n)
        write_submission([list(p) + [int(t)] for (_, t, p) in top], PROBLEM, out)

    save()
    # dedupe seeds, keep a working pool
    pool = []
    seen = set()
    for q in seeds:
        k = tuple(q)
        if k not in seen:
            seen.add(k); pool.append(q)
    print(f"pooled capped-20 {best:,.0f} gap {best-target:+,.0f} | "
          f"{len(pool)} normalized quotient seeds", flush=True)

    accepts, t0 = 0, time.time()
    for it in range(1, a.iters + 1):
        q = list(rng.choice(pool))
        k = rng.randint(4, a.kmax)
        # destroy: remove k supervertices (random or a contiguous window)
        if rng.random() < 0.5:
            idxs = sorted(rng.sample(range(ncls), k), reverse=True)
        else:
            s = rng.randrange(0, ncls - k)
            idxs = list(range(s + k - 1, s - 1, -1))
        removed = [(q.pop(i), i) for i in idxs]
        rng.shuffle(removed)
        for sv, oi in removed:
            if rng.random() < 0.7:              # balanced-ish: near original slot
                pos = max(0, min(len(q), oi + rng.randint(-30, 30)))
            else:                               # exploratory: anywhere
                pos = rng.randrange(0, len(q) + 1)
            q.insert(pos, sv)
        p = expand(q, cls)
        df = ev.full(p)
        before = best
        ok, improved = bank(arc, p, df, n)
        if ok and improved:                     # HSSP DP only when archive moved
            cur = cap20()
            if cur < before - 1e-9:
                best = cur; accepts += 1
                pool.append(q)
                if len(pool) > 60:
                    pool.pop(0)
                save()
                print(f"  *** it {it} accept #{accepts}: capped-20 {best:,.0f} "
                      f"gap {best-target:+,.0f} ***", flush=True)
        if it % 2000 == 0:
            print(f"  [it {it}/{a.iters} capped-20 {best:,.0f} {accepts} accepts "
                  f"{time.time()-t0:.0f}s]", flush=True)
    save()
    print(f"final capped-20 {best:,.0f} gap {best-target:+,.0f} | {accepts} accepts")


if __name__ == "__main__":
    main()
