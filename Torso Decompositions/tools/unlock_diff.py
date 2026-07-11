#!/usr/bin/env python3
"""
unlock_diff.py -- Lever B reconnaissance: what structurally "unlocks" between
two envelope points on large-graph (e.g. the torso jump 526 -> 1029 at w 82->99).

Takes the two achiever orderings nearest the given widths from a submission
file (default cap20.json), diffs their HEAD sets, and classifies the difference
by planted structure: K500/K400/K300 twins vs externals, and planted components.

    python3 tools/unlock_diff.py --w1 82 --w2 99
"""
from __future__ import annotations
import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import load_graph, build_adj_bitsets, graph_path, MAX_TW
from tools.fastwalk import IncEvalC

PROBLEM = "large-graph"


def classify(n, adj):
    """vertex -> label from the planted construction."""
    h = {}
    for v in range(n):
        h.setdefault(frozenset(adj[v] | {v}), []).append(v)
    lab = {}
    cliques = []
    for vs in h.values():
        if len(vs) >= 50:
            K = set(vs) | set(adj[vs[0]])
            cliques.append((sorted(vs), K))
    cliques.sort(key=lambda ck: -len(ck[1]))
    names = ["K500", "K400", "K300"]
    glue = set()
    for (tw, K), nm in zip(cliques, names):
        for v in K:
            lab[v] = f"{nm}-twin" if v in set(tw) else f"{nm}-ext"
        glue |= K
    # components of the rest
    rest = [v for v in range(n) if v not in glue]
    seen, cid = set(), 0
    for s in rest:
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop(); comp.append(u)
            for w in adj[u]:
                if w not in glue and w not in seen:
                    seen.add(w); stack.append(w)
        cid += 1
        for v in comp:
            lab[v] = f"comp{cid}(n={len(comp)})"
    return lab


def point_at(fp, n, ev, wtarget):
    """achiever (w,t,perm) whose staircase point is nearest wtarget."""
    d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
    best = None
    for dv in e["decisionVector"]:
        if not (isinstance(dv, list) and len(dv) == n + 1):
            continue
        perm = [int(x) for x in dv[:-1]]
        if sorted(perm) != list(range(n)):
            continue
        df = ev.full(perm); r = 0
        stair = {}
        for t in range(n - 1, -1, -1):
            c = int(df[t]); r = c if c > r else r
            if r <= MAX_TW and (r not in stair or t < stair[r]):
                stair[r] = t
        for w, t in stair.items():
            score = (abs(w - wtarget), t)
            if best is None or score < best[0]:
                best = (score, w, t, perm)
    _, w, t, perm = best
    return w, t, perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="cap20.json")
    ap.add_argument("--w1", type=int, default=82)
    ap.add_argument("--w2", type=int, default=99)
    a = ap.parse_args()
    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    lab = classify(n, adj)
    fp = os.path.join(HERE, "submissions", PROBLEM, a.file)

    w1, t1, p1 = point_at(fp, n, ev, a.w1)
    w2, t2, p2 = point_at(fp, n, ev, a.w2)
    H1, H2 = set(p1[:t1]), set(p2[:t2])
    print(f"A: (w={w1}, t={t1}, torso={n-t1})   B: (w={w2}, t={t2}, torso={n-t2})")

    def table(title, S):
        c = collections.Counter(lab[v] for v in S)
        print(f"\n{title} ({len(S)} vertices):")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"   {v:5d}  {k}")

    table(f"head A ONLY (in torso at w={w2} but not at w={w1})", H1 - H2)
    table(f"head B ONLY", H2 - H1)
    table("head shared", H1 & H2)


if __name__ == "__main__":
    main()
