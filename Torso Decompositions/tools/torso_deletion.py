#!/usr/bin/env python3
r"""
torso_deletion.py -- the set-space attack on the torso front.

KEY INSIGHT (why this is different from every permutation method):
eliminating a prefix set X in ANY order produces, among the remaining vertices
S = V\X, exactly the *torso* edges: u-v whenever u,v in S are joined by a path
whose interior lies in X.  This is order-independent.  Therefore the best
achievable width(t) for a suffix-set S is just the elimination width of
torso(S) -- a function of the SET S alone, not the ordering.

So the Pareto front decomposes into independent set problems, one per width w:

    maximise |S|   subject to   elim_width(torso(S)) <= w        (t*(w) = n-|S|)

Permutation search (GBFC++, crossover, neuroevolution) cannot move a breakpoint
because that needs a DIFFERENT vertex set, not a local swap.  This tool searches
the right space: it hill-climbs the deletion set X per width -- repeatedly trying
to move a prefix vertex into the torso while keeping torso(S) width <= w -- and
every accepted move pushes t*(w) one step earlier (+1 HV).  Each candidate is
validated by the official full-ordering staircase, so the torso reasoning only
*proposes*; core.evaluate-equivalent scoring *disposes*.

    python3 tools/torso_deletion.py --problem small-graph --budget 3600
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)


def load_vectors(path):
    p = json.load(open(path)); e = p[0] if isinstance(p, list) else p
    return e["decisionVector"]


def front_deg(perm, ab, n):
    """Per-step fill-in degree of a full ordering (the staircase generator)."""
    sm = [0] * n; cur = 0
    for i in range(n - 1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    tmp = list(ab); deg = [0] * n
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
    return deg


def add_staircase(perm, ab, n, arc):
    deg = front_deg(perm, ab, n); run = 0
    for t in range(n - 1, -1, -1):
        if deg[t] > run: run = deg[t]
        if run <= MAX_TW: arc.try_add(int(run), t, list(perm))


def torso_adj(Slist, ab, n):
    """Bitset adjacency of torso(S): G[S] plus, for each connected component C
    of G[X] (X = V\S), a clique on C's S-boundary.  Order-independent."""
    Smask = 0
    for s in Slist: Smask |= 1 << s
    full = (1 << n) - 1
    Xmask = full ^ Smask
    tor = {s: (ab[s] & Smask) for s in Slist}           # original G[S] edges
    seen = 0
    x = Xmask
    while x:                                             # iterate components of G[X]
        b = x & -x; start = b.bit_length() - 1
        if (seen >> start) & 1:
            x &= x - 1; continue
        comp = 0; stack = [start]; boundary = 0
        while stack:
            u = stack.pop()
            if (comp >> u) & 1: continue
            comp |= 1 << u
            nb = ab[u]
            boundary |= nb & Smask                       # S-neighbours of this comp
            xn = nb & Xmask & ~comp
            while xn:
                bb = xn & -xn; xn &= xn - 1
                stack.append(bb.bit_length() - 1)
        seen |= comp
        # clique on the boundary
        bb = boundary
        while bb:
            c = bb & -bb; s = c.bit_length() - 1; bb &= bb - 1
            tor[s] |= boundary ^ (1 << s)
        x &= ~comp
    return tor, Smask


def minfill_width(tor, Slist):
    """Min-FILL elimination of the torso graph (re-optimises the whole suffix
    for the current torso set).  Matches our searched orders on easy widths and
    lets a GROWN torso be re-eliminated from scratch -- which insertion into the
    old order cannot.  Returns (width, order)."""
    tmp = {s: tor[s] for s in Slist}
    remaining = set(Slist); rem = 0
    for s in Slist: rem |= 1 << s
    order = []; maxw = 0
    while remaining:
        best = None
        for v in remaining:
            nb = tmp[v] & rem; d = nb.bit_count()
            e = 0; x = nb
            while x:
                b = x & -x; x &= x - 1; u = b.bit_length() - 1
                e += (tmp[u] & nb).bit_count()
            f = d * (d - 1) // 2 - e // 2
            if best is None or f < best[0] or (f == best[0] and d < best[1]):
                best = (f, d, v, nb)
        f, d, v, nb = best
        if d > maxw: maxw = d
        remaining.discard(v); rem &= ~(1 << v); order.append(v)
        x = nb; bits = []
        while x: b = x & -x; x &= x - 1; bits.append(b.bit_length() - 1)
        for u in bits: tmp[u] = (tmp[u] | nb) & ~(1 << u) & ~(1 << v)
    return maxw, order


def grow_minfill(w, X, S, ab, n, arc, max_cand, deadline):
    """Greedily grow the width-w torso: pull prefix vertices in (lowest torso
    cost first), re-eliminate the new torso with min-fill, accept while the
    re-optimised suffix keeps width <= w and the pooled front improves."""
    import time as _t
    X = list(X); S = list(S); moves = 0
    Smask = 0
    for s in S: Smask |= 1 << s
    # guard: if min-fill can't even re-derive width<=w on the CURRENT torso,
    # it won't keep a larger one at w either -- skip (saves wasted min-fills).
    tor0, _ = torso_adj(S, ab, n)
    w0, _ = minfill_width(tor0, S)
    if w0 > w:
        return
    while _t.time() < deadline and X:
        cand = sorted(X, key=lambda v: (ab[v] & Smask).bit_count())[:max_cand]
        progressed = False
        for v in cand:
            S2 = S + [v]
            tor, _ = torso_adj(S2, ab, n)
            w2, order2 = minfill_width(tor, S2)
            if w2 > w:
                continue
            Xrest = [u for u in X if u != v]
            full = Xrest + order2
            before = -hypervolume_2d(arc.points(), n)
            add_staircase(full, ab, n, arc)
            after = -hypervolume_2d(arc.points(), n)
            if after < before:
                X = Xrest; S = S2; Smask |= 1 << v
                moves += 1; progressed = True
                yield (w, len(X), after)
                break
        if not progressed:
            break


def mindeg_width(tor, Slist):
    """Min-degree elimination width of the torso graph (upper bound on its tw)
    and the elimination order achieving it."""
    tmp = dict(tor); remaining = set(Slist); order = []; maxw = 0
    deg = {s: tmp[s].bit_count() for s in Slist}
    while remaining:
        v = min(remaining, key=lambda s: deg[s])
        nb = tmp[v]; w = nb.bit_count()
        if w > maxw: maxw = w
        order.append(v); remaining.discard(v)
        bits = []
        x = nb
        while x:
            b = x & -x; x &= x - 1; bits.append(b.bit_length() - 1)
        for u in bits:                                   # fill clique, drop v
            tmp[u] = (tmp[u] | nb) & ~(1 << u) & ~(1 << v)
            deg[u] = tmp[u].bit_count()
        del tmp[v]
    return maxw, order


def run(problem, budget_s, here, max_cand=40, minfill=False):
    n, adj = load_graph(graph_path(here, problem))
    ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem)

    # seed the pooled archive from current submissions (full staircase)
    arc = ParetoArchive()
    files = sorted(glob.glob(os.path.join(here, "submissions", problem, "*.json")))
    for fp in files:
        if os.path.basename(fp) in ("portfolio.json",):  # avoid double counting
            pass
        try: dvs = load_vectors(fp)
        except Exception: continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1 and \
                    sorted(int(x) for x in dv[:-1]) == list(range(n)):
                add_staircase([int(x) for x in dv[:-1]], ab, n, arc)
    base = -hypervolume_2d(arc.points(), n)
    print(f"=== torso-deletion -- {problem} (n={n}) ===")
    print(f"pooled front HV = {base:,.0f}" +
          (f"  gap {base - target:+,.0f}" if target else ""))

    # seed deletion set per breakpoint from the best ordering achieving each w
    by_w = {}
    for w, t, perm in arc.entries():
        if w not in by_w or t < by_w[w][0]:
            by_w[w] = (t, perm)

    t0 = time.time(); rng_order = sorted(by_w)
    improved_total = 0
    while time.time() - t0 < budget_s:
        any_improved = False
        for w in rng_order:
            if time.time() - t0 >= budget_s: break
            t_star, perm = by_w[w]
            if t_star == 0: continue                     # already whole graph
            X = perm[:t_star]; S = perm[t_star:]
            if minfill:
                # strong move: re-eliminate the grown torso with min-fill
                last = None
                for (ww, xlen, hv) in grow_minfill(w, X, S, ab, n, arc,
                                                    max_cand, t0 + budget_s):
                    last = (xlen, hv); improved_total += 1; any_improved = True
                if last is not None:
                    # refresh this breakpoint's seed from the improved archive
                    for w2, t2, p2 in arc.entries():
                        if w2 == w and (w not in by_w or t2 < by_w[w][0]):
                            by_w[w] = (t2, p2)
                    msg = f"  w={w:2d} t*: {t_star}->{last[0]}  HV {last[1]:,.0f}"
                    if target: msg += f"  gap {last[1] - target:+,.0f}"
                    print(msg, flush=True)
                continue
            Smask = 0
            for s in S: Smask |= 1 << s
            # Rank prefix vertices by torso-clique cost: a vertex with FEW
            # S-neighbours adds a small boundary clique when pulled into the
            # torso, so it is least likely to push the width past w.  (Torso
            # theory ranks; the official staircase validates.)
            cand = [v for v in X if ab[v] & Smask]
            cand.sort(key=lambda v: (ab[v] & Smask).bit_count())
            cand = cand[:max_cand]
            moved = False
            # try deferring a BATCH of the lowest-torso-cost vertices together,
            # so a breakpoint can drop by more than one step at once (single-
            # vertex deferral alone floors out).  Larger batches first.
            for bsz in (8, 4, 2, 1):
                if moved: break
                for i in range(0, len(cand), bsz):
                    B = cand[i:i + bsz]
                    if not B: continue
                    Bset = set(B)
                    Xrest = [u for u in X if u not in Bset]
                    before = -hypervolume_2d(arc.points(), n)
                    for depth in (0, 1, 4, 16):
                        full = Xrest + S[:depth] + B + S[depth:]
                        add_staircase(full, ab, n, arc)
                    after = -hypervolume_2d(arc.points(), n)
                    if after < before:
                        by_w[w] = (len(Xrest), Xrest + B + S)
                        improved_total += 1; moved = any_improved = True
                        msg = f"  w={w:2d} t*: {t_star}->{len(Xrest)} (batch {len(B)})  HV {after:,.0f}"
                        if target: msg += f"  gap {after - target:+,.0f}"
                        print(msg, flush=True); break
            if not moved:
                continue
        if not any_improved:
            print("  no improving deletion move across any breakpoint -- converged")
            break

    final = -hypervolume_2d(arc.points(), n)
    print(f"\nFinished {time.time()-t0:.0f}s, {improved_total} accepted moves.")
    print(f"Final pooled HV = {final:,.0f}" +
          (f"  gap {final - target:+,.0f}" if target else ""))
    if improved_total:
        top = arc.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = os.path.join(here, "submissions", problem, "torso_del.json")
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                   "decisionVector": dvs}, open(out, "w"))
        print(f"saved -> {out}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--budget", type=float, default=3600.0)
    ap.add_argument("--max-cand", type=int, default=40)
    ap.add_argument("--minfill", action="store_true",
                    help="strong move: re-eliminate grown torsos with min-fill")
    a = ap.parse_args()
    run(a.problem, a.budget, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        max_cand=a.max_cand, minfill=a.minfill)


if __name__ == "__main__":
    main()
