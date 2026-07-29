#!/usr/bin/env python3
"""
medium_struct.py -- recover medium-graph's planted construction (Bannach:
low-treewidth components + glue) and print the CREATOR'S SLACK TABLE:
for each scoring width w, the idealised torso size if every component of
planted width <= w is eliminated, versus the pooled front's actual t(w).

Positive slack at a scoring width  =>  exactly where the remaining gap lives
(feed those widths to gbfcpp --only-widths / the HRI arms).
Zero/negative slack at all scoring widths  =>  a creator-level certificate
that the front matches the planted construction: the medium analogue of the
large clique certificate (thesis: §7 of PLANTED_STRUCTURE_CERTIFICATES.md).

    python3 tools/medium_struct.py                  # peel at core>9 (planted tw 9)
    python3 tools/medium_struct.py --kcut 12
"""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import load_graph, build_adj_bitsets, graph_path

PROBLEM = "medium-graph"


def core_numbers(n, adj):
    deg = {v: len(adj[v]) for v in range(n)}
    order = sorted(range(n), key=lambda v: deg[v])
    core = [0] * n
    removed = [False] * n
    import heapq
    h = [(deg[v], v) for v in range(n)]
    heapq.heapify(h)
    cur = 0
    d = dict(deg)
    while h:
        dv, v = heapq.heappop(h)
        if removed[v] or dv != d[v]:
            continue
        cur = max(cur, dv)
        core[v] = cur
        removed[v] = True
        for u in adj[v]:
            if not removed[u]:
                d[u] -= 1
                heapq.heappush(h, (d[u], u))
    return core


def components(n, adj, keep):
    comp = [-1] * n
    out = []
    for s in range(n):
        if not keep[s] or comp[s] != -1:
            continue
        cid = len(out); stack = [s]; comp[s] = cid; verts = []
        while stack:
            u = stack.pop(); verts.append(u)
            for w in adj[u]:
                if keep[w] and comp[w] == -1:
                    comp[w] = cid; stack.append(w)
        out.append(verts)
    return out


def minfill_width(verts, adj):
    """Min-fill elimination width of the induced subgraph on verts."""
    vs = set(verts)
    nbr = {v: set(adj[v]) & vs for v in verts}
    remaining = set(verts); maxw = 0
    while remaining:
        best = None
        for v in remaining:
            nb = nbr[v] & remaining
            d = len(nb)
            e = sum(len(nbr[u] & nb) for u in nb) // 2
            f = d * (d - 1) // 2 - e
            if best is None or (f, d) < (best[0], best[1]):
                best = (f, d, v, nb)
        f, d, v, nb = best
        maxw = max(maxw, d)
        remaining.discard(v)
        for u in nb:
            nbr[u] |= (nb - {u})
            nbr[u].discard(v)
    return maxw


def front_t_of_w(n, ab, widths):
    """Pooled front's t*(w) for each requested width, from cap20.json."""
    fp = os.path.join(HERE, "submissions", PROBLEM, "cap20.json")
    d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
    best = {}
    for dv in e["decisionVector"]:
        perm = [int(x) for x in dv[:-1]]
        sm = [0] * n; cur = 0
        for i in range(n - 1, -1, -1):
            sm[i] = cur; cur |= 1 << perm[i]
        tmp = list(ab); deg = [0] * n
        for i in range(n):
            s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
            while x:
                b = x & -x; x ^= b; u = b.bit_length() - 1; tmp[u] |= s ^ b
        run = 0
        for t in range(n - 1, -1, -1):
            run = max(run, deg[t])
            for w in widths:
                if run <= w and (w not in best or t < best[w]):
                    best[w] = t
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcut", type=int, default=9,
                    help="glue = vertices with core number > kcut")
    a = ap.parse_args()

    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ab = build_adj_bitsets(n, adj_l)
    core = core_numbers(n, adj)

    from collections import Counter
    cc = Counter(core)
    print(f"n={n}  max core={max(core)}  core histogram (top): "
          f"{sorted(cc.items(), key=lambda x: -x[1])[:6]}")

    glue = [v for v in range(n) if core[v] > a.kcut]
    keep = [core[v] <= a.kcut for v in range(n)]
    comps = components(n, adj, keep)
    comps.sort(key=len, reverse=True)
    print(f"\nglue (core>{a.kcut}): {len(glue)} vertices | "
          f"planted components: {len(comps)} covering {sum(map(len, comps))} vertices")

    rows = []
    for verts in comps:
        if len(verts) < 2:
            rows.append((len(verts), 0, max(len(adj[v]) - len(adj[v] & set(verts)) for v in verts)))
            continue
        wint = minfill_width(verts, adj)
        vs = set(verts)
        ext = max(len(adj[v] - vs) for v in verts)
        rows.append((len(verts), wint, ext))
    print("\ncomponent |  size | internal-width | max-ext-degree")
    for i, (sz, wi, ex) in enumerate(rows[:30]):
        print(f"   C{i:<3}   | {sz:5d} | {wi:14d} | {ex:14d}")

    scoring = [8, 17, 29, 41, 51, 60, 73, 88, 98, 108, 121, 140, 150, 162,
               178, 194, 208, 217, 230, 238]
    print("\ncreator slack table (scoring widths):")
    print("   w | ideal t (opt) | ideal t (cons) | front t | slack(opt)")
    tf = front_t_of_w(n, ab, scoring)
    for w in scoring:
        elim_o = sum(sz for sz, wi, ex in rows if wi <= w)
        elim_c = sum(sz for sz, wi, ex in rows if wi + ex <= w)
        t_o, t_c = n - (n - len(glue) - 0) + 0, 0  # placeholder clarity below
        ideal_o = n - elim_o          # torso keeps glue + oversized comps
        ideal_c = n - elim_c
        ft = tf.get(w, n)
        print(f" {w:3d} | {ideal_o:13d} | {ideal_c:14d} | {ft:7d} | {ft - ideal_o:+d}")
    print("\nslack(opt) > 0  => the front's torso at w is SMALLER than the planted"
          "\nconstruction allows -- attack those widths. slack <= 0 everywhere =>"
          "\ncreator-certificate: the front saturates the planted structure.")


if __name__ == "__main__":
    main()
