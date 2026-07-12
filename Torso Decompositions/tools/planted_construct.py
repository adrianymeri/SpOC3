#!/usr/bin/env python3
"""
planted_construct.py -- structured mid-band constructions (Lever B attack).

STATUS 2026-07-11: NEGATIVE RESULT, kept as evidence (THESIS §15 companion).
Three principled orderings all fail with characteristic fill explosions:
  1. components-first tail: eliminating a whole component clique-ifies its
     ENTIRE glue neighbourhood (component-level coupling >> per-vertex);
  2. remainders-first with coupling-ranked evictions: evicting an external
     wires its attachments into the remaining clique (fill = out x remainder);
  3. eviction+attachment-closure policy: closures are large (~300-900) and
     cross-clique direct edges re-merge the cliques anyway.
Conclusion: the mid-band envelope is genuinely hard-earned; the reduced
combinatorial core is "choose evictions + attachment closure subject to fill
propagation" -- a candidate for an exact ILP/BB formulation on the quotient,
not for a fixed ordering rule. The LNS arms discover these trade-offs
incrementally, which is why they hold the envelope there.

UPDATE (12 July, post-sanitization, two more attempts): (4) closure-in-head
with cheapest-attachment evictions -- closure comps eliminated late wire the
torso components into the glue (max step ~1500); (5) closure-first +
cross-partner co-eviction -- the co-eviction CASCADES (E grows to ~2000,
closure ~1000) and fill chains through the closure comps still poison the
remainder. Five principled constructions, five distinct fill channels. This
is strong empirical evidence that the planted instance's mid-band is
construction-resistant by design: certificates close its top, incremental
search (LNS/GBFC++) is the only thing that holds its middle. Reported as a
structural-hardness result in THESIS 15.6/15.8 follow-up.

Design derived from the coupling measurement (2026-07-11): the three glue
cliques are only weakly coupled per-vertex (max ~24 direct + ~20 fill-partner
edges into other cliques). Therefore build, for each target width w:

  HEAD  = per-clique quota k_i = |K_i| - R_i, taking twins first (zero fill),
          then externals by coupling score DESCENDING (evict the vertices that
          wire the cliques together);
  TAIL  = all 25 planted components first, each in min-fill order (their width
          is <= ~6 + <=25 attachment edges, far under w), then the clique
          remainders, smallest clique first, each ordered by coupling
          ASCENDING (high-coupling members eliminated late, when their own
          remainder term is small);
  R_i   = w + 1 - margin, margin swept over a grid (the coupling allowance).

Every construction is exactly evaluated; every breakpoint is banked; output
submissions/large-graph/planted_construct.json (pool it with cap_submit).

    python3 tools/planted_construct.py --wmin 104 --wmax 299
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

PROBLEM = "large-graph"


def setup():
    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    adj = {v: set(adj_l[v]) for v in range(n)}
    h = {}
    for v in range(n):
        h.setdefault(frozenset(adj[v] | {v}), []).append(v)
    Ks, twins = [], []
    for vs in h.values():
        if len(vs) >= 50:
            Ks.append(set(vs) | set(adj[vs[0]])); twins.append(sorted(vs))
    order = sorted(range(len(Ks)), key=lambda i: -len(Ks[i]))
    Ks = [Ks[i] for i in order]; twins = [twins[i] for i in order]
    glue = set().union(*Ks)
    who = {}
    for i, K in enumerate(Ks):
        for v in K: who[v] = i
    # coupling score per glue vertex: direct other-clique edges + fill partners
    partner = collections.defaultdict(set)
    comps_v = [v for v in range(n) if v not in glue]
    for v in comps_v:
        gl = [u for u in adj[v] if u in glue]
        for a in gl:
            for b in gl:
                if a != b and who[a] != who[b]:
                    partner[a].add(b)
    coup = {}
    for v in glue:
        direct = sum(1 for u in adj[v] if u in glue and who[u] != who[v])
        coup[v] = direct + len(partner.get(v, ()))
    # components + per-component min-fill order
    comp_of, comps = {}, []
    seen = set()
    for s in comps_v:
        if s in seen: continue
        stack, c = [s], []
        seen.add(s)
        while stack:
            u = stack.pop(); c.append(u)
            for wv in adj[u]:
                if wv not in glue and wv not in seen:
                    seen.add(wv); stack.append(wv)
        comps.append(c)
    def minfill_order(c):
        nodes = set(c); nb = {v: adj[v] & nodes for v in c}; out = []
        while nodes:
            best, bk = None, None
            for v in nodes:
                nv = nb[v]; nl = list(nv)
                f = sum(1 for i in range(len(nl)) for j in range(i+1, len(nl))
                        if nl[j] not in nb[nl[i]])
                k = (f, len(nv))
                if bk is None or k < bk: bk, best = k, v
            v = best; nv = list(nb[v])
            for i in range(len(nv)):
                for j in range(i+1, len(nv)):
                    nb[nv[i]].add(nv[j]); nb[nv[j]].add(nv[i])
            for u in nv: nb[u].discard(v)
            nodes.discard(v); out.append(v)
        return out
    tail_comps = []
    for c in sorted(comps, key=len):
        tail_comps += minfill_order(c)
    return n, adj, Ks, twins, coup, tail_comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmin", type=int, default=104)
    ap.add_argument("--wmax", type=int, default=299)
    ap.add_argument("--margins", default="0,10,20,30,45,60,80")
    ap.add_argument("--algo", default="planted_construct")
    a = ap.parse_args()
    n, adj, Ks, twins, coup, tail_comps = setup()
    ab = build_adj_bitsets(n, {v: sorted(adj[v]) for v in range(n)}) if False else None
    n2, adj_l = load_graph(graph_path(HERE, PROBLEM))
    ev = IncEvalC(build_adj_bitsets(n2, adj_l), n2)
    target = LEADERBOARD_TARGETS[PROBLEM]
    sizes = [len(K) for K in Ks]
    print(f"cliques {sizes} | twins {[len(t) for t in twins]} | tail comps {len(tail_comps)}")

    arc = ParetoArchive()
    margins = [int(x) for x in a.margins.split(",")]
    t0 = time.time(); results = {}
    for w in range(a.wmin, a.wmax):
        bestrow = None
        for m in margins:
            head, rem = [], []
            ok = True
            for i, K in enumerate(Ks):
                R = w + 1 - m
                if R < 1: ok = False; break
                k = max(0, sizes[i] - R)
                tw = twins[i]
                ext = sorted((v for v in K if v not in set(tw)),
                             key=lambda v: -coup[v])
                pick = (list(tw) + ext)[:k] if k > len(tw) else list(tw[:k])
                head.append(pick)
                rem.append(sorted((v for v in K if v not in set(pick)),
                                  key=lambda v: coup[v]))
            if not ok: continue
            H = [v for part in head for v in part]
            # KEY ORDER (learned from the failed comps-first variant): glue
            # remainders FIRST (each member sees own remainder + <=39 out),
            # planted components LAST (they only receive fill from externals;
            # twins have zero outside edges).
            tail = rem[2] + rem[1] + rem[0] + tail_comps
            perm = H + tail
            if len(perm) != n or len(set(perm)) != n: continue
            df = ev.full(perm)
            if max(int(x) for x in df) > MAX_TW: continue
            t = len(H); r = 0
            for tt in range(n - 1, -1, -1):
                c = int(df[tt]); r = c if c > r else r
                if r <= MAX_TW: arc.try_add(r, tt, perm)
            wt = max(int(df[i]) for i in range(t, n))
            if bestrow is None or (wt, t) < bestrow[:2]:
                bestrow = (wt, t, m)
        if bestrow: results[w] = bestrow
        if w % 20 == 0 and bestrow:
            bound = sum(max(0, s - w - 1) for s in sizes)
            print(f"  w={w:3d} -> achieved width {bestrow[0]} at t={bestrow[1]} "
                  f"(margin {bestrow[2]}; bound t>={bound}) [{time.time()-t0:.0f}s]",
                  flush=True)

    print("\n  w | best (achieved_w, t, margin) | packing bound on t")
    for w in sorted(results):
        if w % 8 == 0 or w in (a.wmin, a.wmax - 1):
            bound = sum(max(0, s - w - 1) for s in sizes)
            print(f" {w:4d} | {results[w]} | {bound}")
    hv = -arc.hypervolume(n)
    top = arc.top_k_by_hv_contribution(20, n)
    cap = -hypervolume_2d([(x, t) for x, t, _ in top], n)
    print(f"\nconstruction archive alone: envelope {hv:,.0f} | cap20 {cap:,.0f} | target {target:,.0f}")
    out = os.path.join(HERE, "submissions", PROBLEM, f"{a.algo}.json")
    top60 = arc.top_k_by_hv_contribution(60, n)
    write_submission([list(p) + [int(t)] for (_, t, p) in top60], PROBLEM, out)
    print(f"wrote {out} -- run cap_submit to pool")


if __name__ == "__main__":
    main()
