#!/usr/bin/env python3
"""endpoint_ils.py -- attack the ONE lever nobody aimed at: the t=0 endpoint.

FINDING (2026-08-05, measured from the live pool):
  * The 20-point HSSP selection is provably OPTIMAL for the pooled envelope, so
    no re-selection can move the score -- cap_submit already extracts the best
    possible 20 points.
  * For every instance the score is dominated by the ENDPOINT point (threshold
    t=0, the whole graph is the torso), which sits at the far right of the
    staircase with a very wide hypervolume strip.
  * Reducing the endpoint width by one unit frees the widths just below it to
    take t=0.  The realised HV gain is therefore the OLD threshold at those
    widths, sum_{w=w_new}^{w_old-1} t_old(w) -- NOT n per unit (n per unit is
    only the ceiling, reached when those widths had no coverage at all).
    Always re-score with cap_submit; the per-accept figures logged below are
    upper bounds, not the banked gain.
  * CONFIRMED 2026-08-08: medium's endpoint fell 234 -> 228 under this search
    and the capped score went -1,744,477 (gap +645) -> -1,746,926 (gap -1,804),
    i.e. PAST the leaderboard target.  verify_submission: 20/20 valid, 0 capped,
    0 dominated.  small remains one unit away (endpoint 15, gap +5).
  * The endpoint width is the graph's ELIMINATION (treewidth) width.  Standard
    heuristics are far worse than the pool (min-degree/min-fill: small 20-22,
    medium 285-288) -- the pool's 15/236 came from the specialised search, so a
    from-scratch heuristic cannot help.  The only way forward is LOCAL SEARCH
    SEEDED FROM THE POOL'S OWN BEST ENDPOINT ORDERING, minimising the induced
    width with a (width, #bottleneck-vertices) lexicographic objective -- the
    #bottleneck secondary gives a gradient across the flat integer width.

NOVELTY: framing the SpOC-3 capped-hypervolume score as "the endpoint is a
treewidth sub-problem worth n HV per unit" and driving it with a bottleneck-
targeted iterated local search seeded from the pooled elimination ordering.
Any width improvement is banked as a full staircase so cap_submit pools it.

    python3 tools/endpoint_ils.py --problem small-graph  --seed 1
    python3 tools/endpoint_ils.py --problem medium-graph --seed 1
"""
from __future__ import annotations
import argparse, json, os, random, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
# single-threaded BLAS (see gbdt_grow.py) -- set before core imports numpy
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from core import (load_graph, graph_path, build_adj_bitsets, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)


def wprofile(n, ab, order, want_bottleneck=False):
    """Induced width of an elimination ordering (bitset front-degree walk).
    Returns (width, #vertices hitting the max width) and, if requested, the
    list of order-positions that hit the max (the bottleneck to attack)."""
    nbr = list(ab); rem = (1 << n) - 1; w = 0; cnt = 0; hits = []
    for idx, v in enumerate(order):
        rem &= ~(1 << v)
        hi = nbr[v] & rem; d = hi.bit_count()
        if d > w:
            w = d; cnt = 1
            if want_bottleneck: hits = [idx]
        elif d == w:
            cnt += 1
            if want_bottleneck: hits.append(idx)
        x = hi
        while x:
            b = x & -x; x ^= b; u = b.bit_length() - 1
            nbr[u] |= hi & ~(1 << u)
    return (w, cnt, hits) if want_bottleneck else (w, cnt)


def full_staircase(n, ab, order):
    """All (width, t) breakpoints of one ordering; None if any step > MAX_TW."""
    nbr = list(ab); rem = (1 << n) - 1
    deg = [0] * n
    for i, v in enumerate(order):
        rem &= ~(1 << v)
        hi = nbr[v] & rem; deg[i] = hi.bit_count()
        x = hi
        while x:
            b = x & -x; x ^= b; u = b.bit_length() - 1
            nbr[u] |= hi & ~(1 << u)
    if max(deg) > MAX_TW:
        return None
    pts = []; run = 0
    for t in range(n - 1, -1, -1):
        if deg[t] > run: run = deg[t]
        pts.append((run, t))
    return pts


def seed_from_pool(n, ab, problem):
    """Best (lowest-width) t=0 ordering already in the pool.  Reads only the
    per-problem cap20.json (fast), falling back to the deepest vector there."""
    fp = os.path.join(HERE, "submissions", problem, "cap20.json")
    d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
    best = None
    for dv in e["decisionVector"]:
        if len(dv) != n + 1:
            continue
        p = [int(x) for x in dv[:-1]]
        if sorted(p) != list(range(n)):
            continue
        w, _ = wprofile(n, ab, p)
        if best is None or w < best[0]:
            best = (w, p)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10**12)
    ap.add_argument("--restart", type=int, default=40000,
                    help="non-improving evals before a soft restart from best")
    a = ap.parse_args()

    n, adj = load_graph(graph_path(HERE, a.problem))
    ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(a.problem)
    rng = random.Random(a.seed)

    w0, order = seed_from_pool(n, ab, a.problem)
    curw, curc, hits = wprofile(n, ab, order, want_bottleneck=True)
    best = (curw, curc, order[:])
    print(f"=== endpoint_ils {a.problem} (n={n}, seed {a.seed}) | pool endpoint "
          f"width {w0} | each -1 width frees the widths below it to t=0 "
          f"(re-score with cap_submit for the banked gain) ===", flush=True)
    print(f"    start (width {curw}, #bottleneck {curc})", flush=True)

    out = os.path.join(HERE, "submissions", a.problem, f"endpoint_ils_s{a.seed}.json")

    def bank(order):
        pts = full_staircase(n, ab, order)
        if pts is None:
            return
        arc = ParetoArchive()
        for w, t in pts:
            arc.try_add(w, t, order)
        top = arc.top_k_by_hv_contribution(60, n)
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": a.problem,
                   "decisionVector": [list(p) + [int(t)] for (_, t, p) in top]},
                  open(out, "w"))

    t0 = time.time(); since = 0; evals = 0; improves = 0
    for it in range(a.iters):
        cand = order[:]
        # targeted move: relocate a bottleneck vertex (or a random one) elsewhere
        if hits and rng.random() < 0.7:
            src = rng.choice(hits)
        else:
            src = rng.randrange(n)
        moves = rng.choice([1, 1, 1, 2, 2, 3])
        for _ in range(moves):
            v = cand.pop(min(src, len(cand) - 1))
            cand.insert(rng.randrange(len(cand) + 1), v)
            src = rng.randrange(n)
        w, c, h = wprofile(n, ab, cand, want_bottleneck=True)
        evals += 1; since += 1
        if (w, c) <= (curw, curc):
            if (w, c) < (curw, curc):
                improves += 1
            order, curw, curc, hits = cand, w, c, h
            if w < best[0]:
                best = (w, c, cand[:]); since = 0
                bank(cand)
                print(f"  it {it}: *** ENDPOINT WIDTH {w}  (was {w0}) banked "
                      f"-> re-score: python3 tools/cap_submit.py "
                      f"--problem {a.problem} ***", flush=True)
        if since >= a.restart:
            order = best[2][:]
            curw, curc, hits = wprofile(n, ab, order, want_bottleneck=True)
            since = 0
            # kick: a few random relocations off the incumbent best
            for _ in range(rng.randrange(3, 9)):
                v = order.pop(rng.randrange(n)); order.insert(rng.randrange(n + 1), v)
            curw, curc, hits = wprofile(n, ab, order, want_bottleneck=True)
        if it % 3000 == 0 and it:
            el = time.time() - t0
            print(f"  [it {it} width {best[0]} cur({curw},{curc}) "
                  f"{improves} improves {evals/max(el,1e-9):.0f} eval/s {el:.0f}s]",
                  flush=True)


if __name__ == "__main__":
    main()
