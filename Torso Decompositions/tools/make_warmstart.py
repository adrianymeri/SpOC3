#!/usr/bin/env python3
"""
make_warmstart.py -- distill the banked best orderings of an instance into a
compact warm-start file for tools/gpu_search.py.

The GPU search should *continue from our current best*, not rediscover it from a
min-degree start. This collects the elimination orderings already banked in
submissions/<problem>/ (portfolio first, then every seed), de-duplicates, caps
at K, and writes them as a small JSON list of permutations to
data/warmstart_<problem>.json. gpu_search.py --warmstart loads it, seeds the
archive with their fronts (reconstructing the banked portfolio score on the GPU)
and starts CMA-ES from the strongest one.

We collect perms straight from the submission decisionVectors (no re-evaluation),
prioritising portfolio.json since it holds the best-of-union front.

Usage:
    python3 tools/make_warmstart.py --problem large-graph --k 300
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json
from core import load_graph, build_adj_bitsets, graph_path, repo_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph")
    ap.add_argument("--k", type=int, default=300, help="max orderings to keep")
    args = ap.parse_args()
    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))

    sub = os.path.join(here, "submissions", args.problem)
    # portfolio first (best-of-union), then flat stems, then seeds
    files = ([os.path.join(sub, "portfolio.json")]
             + sorted(glob.glob(os.path.join(sub, "*.json")))
             + sorted(glob.glob(os.path.join(sub, "seeds", "*.json"))))
    seen = set(); perms = []
    for fp in files:
        if not os.path.exists(fp):
            continue
        try:
            dvs = json.load(open(fp))[0]["decisionVector"]
        except Exception:
            continue
        for dv in dvs:
            if not isinstance(dv, list) or len(dv) != n + 1:
                continue
            perm = tuple(int(x) for x in dv[:-1])
            if len(perm) != n or sorted(perm) != list(range(n)):
                continue
            if perm in seen:
                continue
            seen.add(perm); perms.append(list(perm))
            if len(perms) >= args.k:
                break
        if len(perms) >= args.k:
            break

    out = os.path.join(here, "data", f"warmstart_{args.problem}.json")
    json.dump({"problem": args.problem, "n": n, "perms": perms}, open(out, "w"))
    sz = os.path.getsize(out) / 1e6
    print(f"{args.problem}: collected {len(perms)} unique orderings from "
          f"{len(files)} files -> {out} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
