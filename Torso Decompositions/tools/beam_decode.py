#!/usr/bin/env python3
r"""
beam_decode.py -- beam-search elimination decoder (the one architectural idea the
external reviews converge on, see docs/STRATEGIC_ASSESSMENT.md).

Every production decoder is argsort(policy . features): one continuous vector ->
one ordering.  It is brittle -- nearby vectors decode identically -- and commits
to a single greedy path.  This builds the elimination ordering INCREMENTALLY and
keeps the best `B` partial sequences at every step (a beam) instead of one:

    step i:  for each of the B live prefixes, propose the K lowest-fill next
             vertices, simulate eliminating each (bitset fill update), score the
             extended prefix by a width-aware proxy, keep the best B.

The proxy only has to keep *promising* prefixes alive; the FINAL B orderings are
scored by the exact C kernel (IncEvalC.full), so correctness never depends on the
proxy.  Multiple beams from different random tie-breaks give diverse orderings;
per-band bests are pooled into the archive.  This can find torsos the single-path
argsort decoders cannot reach -- the only mechanism with a real shot at the
torso-quality gap (THESIS s13.8a).

ADDITIVE / SAFE: writes submissions/<problem>/beam_decode.json only on a verified
front improvement.

    python3 tools/beam_decode.py --problem medium-graph --beam 8 --cand 5 \
        --decodes 6 --focus-cap20
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import heapq
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from tools.fastwalk import IncEvalC
from algorithms.continuous.cmaes_torso import get_features
from algorithms.continuous.gbdt_torso import make_gbdt, training_set


def load_front(here, problem, n, ev):
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    perm = [int(x) for x in dv[:-1]]; df = ev.full(perm); r = 0
                    for t in range(n - 1, -1, -1):
                        c = int(df[t]); r = c if c > r else r
                        if r <= MAX_TW: arc.try_add(r, t, perm)
        except Exception:
            continue
    by_w = {}; wins = {}
    for w, t, p in arc.entries():
        if w not in by_w or t < by_w[w][0]: by_w[w] = (t, list(p))
    # weight each ordering by how many front bands it wins (cheap, meaningful)
    for w, (t, p) in by_w.items():
        k = tuple(p); wins[k] = wins.get(k, 0) + 1
    el = sorted(wins.items(), key=lambda x: -x[1])[:24]
    elites = [(float(c), list(k)) for k, c in el]
    return by_w, elites


def front_hv(by_w, n):
    a = ParetoArchive()
    for w in by_w: a.try_add(w, by_w[w][0], None)
    return -hypervolume_2d(a.points(), n)


def one_beam(ab, n, B, K, rng, pred_rank=None):
    """Return the best (lowest cumulative-width) complete elimination ordering
    from a single beam search.  States carry their own bitset adjacency.

    Candidates at each step are the union of the lowest-current-degree live
    vertices (the dynamic min-degree signal) and the lowest learned-policy-rank
    live vertices (the corpus-trained GBDT policy) -- so the beam explores both
    the live-state heuristic AND the trained policy the production engines use,
    with multi-path lookahead neither argsort nor greedy min-degree has."""
    full = list(ab)
    # state = [cum_width, elim_list, adj(list of int), alive_mask]
    beam = [[0, [], list(full), (1 << n) - 1]]
    half = max(1, K // 2)
    for _ in range(n):
        children = []
        for cw, elim, adj, alive in beam:
            degs = []
            x = alive
            while x:
                b = x & -x; x &= x - 1; v = b.bit_length() - 1
                degs.append(((adj[v] & alive).bit_count(), v))
            degs.sort()
            live = [v for _, v in degs]
            cands = [v for _, v in degs[:K]]
            if pred_rank is not None:               # add policy-suggested candidates
                pol = heapq.nsmallest(half, live, key=lambda v: pred_rank[v])
                cands = list(dict.fromkeys(cands[:half] + pol + cands))[:K + half]
            if rng.random() < 0.3 and len(degs) > len(cands):   # exploration pick
                cands.append(degs[len(cands)][1])
            for v in cands:
                nbrs = adj[v] & alive & ~(1 << v)
                w = nbrs.bit_count()
                adj2 = list(adj)
                x2 = nbrs
                while x2:
                    b = x2 & -x2; x2 &= x2 - 1; u = b.bit_length() - 1
                    adj2[u] = (adj2[u] | nbrs) & ~(1 << u)
                children.append([cw + w, elim + [v], adj2, alive & ~(1 << v)])
        children.sort(key=lambda s: s[0])
        beam = children[:B]
    return beam[0][1]


def fit_policy(here, problem, n, adj, elites, seed):
    """Train a GBDT on HV-weighted corpus elites (static features -> elimination
    position); return a per-vertex predicted-rank array (0 = eliminate first)."""
    if not elites:
        return None
    F = np.asarray(get_features(here, problem, n, adj, 32)[0])
    X, y, w = training_set(elites, F, n)
    name, gb = make_gbdt("auto", seed)
    gb.fit(X, y, sample_weight=w)
    pred = np.asarray(gb.predict(F), dtype=np.float64)
    rank = pred.argsort().argsort().astype(np.float64)
    print(f"policy: trained {name} on {len(elites)} corpus elites", flush=True)
    return rank


def run(problem, here, B, K, decodes, seed, use_policy=True):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n); target = LEADERBOARD_TARGETS.get(problem)
    by_w, elites = load_front(here, problem, n, ev)
    start_hv = front_hv(by_w, n)
    print(f"=== beam-decode {problem} | n={n} | front {start_hv:,.0f}"
          f"{f'  gap {start_hv-target:+,.0f}' if target else ''} "
          f"| beam B={B} K={K} policy={use_policy} ===", flush=True)
    pred_rank = fit_policy(here, problem, n, adj, elites, seed) if use_policy else None
    rng = np.random.default_rng(seed)
    import random as _r
    raised = 0; t0 = time.time()
    for d in range(decodes):
        perm = one_beam(ab, n, B, K, _r.Random(seed + d), pred_rank)
        df = ev.full(perm); r = 0; gains = 0
        for t in range(n - 1, -1, -1):
            c = int(df[t]); r = c if c > r else r
            if r <= MAX_TW:
                cur = by_w[r][0] if r in by_w else n
                if t < cur:
                    by_w[r] = (t, list(perm)); gains += 1
        raised += gains
        c = front_hv(by_w, n)
        print(f"  decode {d+1}/{decodes}: {gains} band(s) raised | front {c:,.0f}"
              f"{f'  gap {c-target:+,.0f}' if target else ''}  [{time.time()-t0:.0f}s]",
              flush=True)
    fin = front_hv(by_w, n)
    print(f"\nfinal front {fin:,.0f}"
          f"{f'  gap {fin-target:+,.0f}' if target else ''} | "
          f"{'IMPROVED +%.0f HV' % (start_hv-fin) if fin < start_hv-0.5 else 'no gain'}",
          flush=True)
    if fin < start_hv - 0.5:
        a = ParetoArchive()
        for w in by_w:
            if by_w[w][1] is not None: a.try_add(w, by_w[w][0], by_w[w][1])
        top = a.top_k_by_hv_contribution(40, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = os.path.join(here, "submissions", problem, "beam_decode.json")
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                   "decisionVector": dvs}, open(out, "w"))
        print(f"saved -> {out}  (VERIFY with tools/verify_submission.py)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="medium-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--cand", type=int, default=5)
    ap.add_argument("--decodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-policy", action="store_true",
                    help="ablation: pure min-degree beam, no learned proposer")
    a = ap.parse_args()
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        a.beam, a.cand, a.decodes, a.seed, use_policy=not a.no_policy)


if __name__ == "__main__":
    main()
