#!/usr/bin/env python3
"""
manifold_analysis.py -- EXPLORATORY probe. NOT USED IN THE THESIS.

STATUS: this analysis did NOT yield a clean result and is deliberately excluded
from THESIS.md. It is kept only as scaffolding for a future, valid version.

Idea: test whether an ordering pi is realizable by a linear decode argsort(G.x),
i.e. whether some x satisfies G[pi[i]].x > G[pi[i+1]].x for all consecutive pairs
(fit the best x by logistic descent on the unit-normalised difference vectors;
report the fraction of order constraints satisfied).

WHY IT DID NOT LAND (two confounds, both fatal as run here):
  1. The orderings in the submission files are the HV-filtered top-20, which are
     dominated by WARM-START (banked cmaes/gbdt) orderings, NOT the raw argsort
     outputs of the GAPS decode -- so it does not measure the decode at all.
  2. The adjacency metric tests *consecutive* pairs in a 2426-long order, which
     are near-ties dominated by noise; a linear policy can get the global order
     right yet "fail" most adjacent pairs. Observed values (~53-56% for every
     arm) sit near the 50% random baseline and do not separate.
A valid version requires dumping the RAW decode orderings (pre-HV-filter,
pre-warm-start) from a GPU run and a global rank metric -- future work.

    python3 tools/manifold_analysis.py --problem large-graph   # exploratory only
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json
import numpy as np
from core import load_graph, build_adj_bitsets, graph_path, repo_root
from algorithms.continuous.gbdt_torso import build_rich_features
from tools.gaps_search import poly_expand  # noqa: E402  (same Phi as the search)


def best_linear_fraction(G, perm, iters=1500, seed=0):
    """Max fraction of consecutive order-constraints satisfiable by argsort(G.x).
    Fits x by descent on the logistic surrogate of the (n-1) constraints."""
    idx = np.asarray(perm)
    D = G[idx[:-1]] - G[idx[1:]]                 # (n-1, k): want D x > 0
    nrm = np.linalg.norm(D, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    D = D / nrm
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(G.shape[1]); x /= np.linalg.norm(x)
    lr = 0.5
    best = 0.0
    for it in range(iters):
        m = D @ x
        s = 1.0 / (1.0 + np.exp(m))              # sigmoid(-m)
        grad = -(s[:, None] * D).mean(0)         # d/dx mean softplus(-Dx)
        x -= lr * grad
        x /= (np.linalg.norm(x) + 1e-12)
        if it % 50 == 0 or it == iters - 1:
            best = max(best, float((D @ x > 0).mean()))
    return best


def file_orderings(fp, n):
    try: dvs = json.load(open(fp))[0]["decisionVector"]
    except Exception: return []
    out = []
    for dv in dvs:
        if isinstance(dv, list) and len(dv) == n + 1 and \
                sorted(int(x) for x in dv[:-1]) == list(range(n)):
            out.append([int(x) for x in dv[:-1]])
    # de-dup
    uniq = []; seen = set()
    for p in out:
        k = tuple(p)
        if k not in seen: seen.add(k); uniq.append(p)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph")
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--max-orderings", type=int, default=20)
    args = ap.parse_args()
    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))
    ab = build_adj_bitsets(n, adj)
    F = build_rich_features(here, args.problem, n, adj, ab, args.k_eig)   # 41-d
    Phi = poly_expand(F, 0, 42)                                           # 82-d (squares)
    print(f"{args.problem}: n={n}, F={F.shape[1]}-d (linear), Phi={Phi.shape[1]}-d (poly)\n")

    bases = [("F  (linear, 41-d)", F), ("Phi (poly, 82-d)", Phi)]
    stems = ["gaps_nogbdt", "gaps"]
    print(f"{'ordering source':<16}{'decode basis':<20}{'mean % order-constraints linearly realizable':>10}")
    results = {}
    for stem in stems:
        perms = file_orderings(os.path.join(here, "submissions", args.problem, f"{stem}.json"), n)[:args.max_orderings]
        if not perms:
            print(f"{stem:<16} (no file)"); continue
        for bname, G in bases:
            fr = np.array([best_linear_fraction(G, p) for p in perms])
            results[(stem, bname)] = fr
            print(f"{stem:<16}{bname:<20}{100*fr.mean():>8.2f}%  (min {100*fr.min():.2f}%, over {len(perms)} orderings)")
        print()

    print("="*78)
    print("EXPLORATORY ONLY — these numbers are confounded (see module docstring):")
    print("  the scored orderings are warm-start-dominated, not raw decode outputs,")
    print("  and the adjacency metric sits near the 50% random baseline. This probe")
    print("  is NOT used in the thesis; a valid version needs raw decode dumps.")
    print("="*78)


if __name__ == "__main__":
    main()
