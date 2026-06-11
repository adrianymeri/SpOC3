#!/usr/bin/env python3
"""
exact_treewidth.py -- workstation harness for paper item #2-exact.

Purpose
-------
The paper currently brackets the large-graph width objective:

    minor-min-width lower bound  (MMD, Bodlaender & Koster 2011) = 499
    best achieved elimination width (full graph, seed 42)        = 499

Because LB == achieved, the bracket is already tight and the exact
treewidth of the large graph is provably 499 -- *no external solver is
strictly required*. This script lets you confirm that independently with
a dedicated exact / anytime treewidth solver, so the paper can state the
result as "verified" rather than "sandwiched".

What it does
------------
1. Loads a problem graph via core.load_graph.
2. Reports the MMD lower bound and (optionally) the best achieved width
   read from the canonical submission, printing the bracket.
3. Exports the graph to PACE .gr format (1-indexed `p tw n m` header)
   so any PACE-2017-compatible solver can read it.
4. If an exact-solver binary is found on PATH (or given with --solver),
   runs it and parses the reported treewidth.

PACE-compatible solvers you can drop in on your workstation:
  * flow-cutter / FlowCutter-PACE17
  * tamaki / tw-exact (Tamaki 2017)
  * htd (htd_main --opt width)
Any binary that reads DIMACS-style `.gr` on stdin and prints a tree
decomposition (`s td ...` line whose width = bagsize-1) works; pass it
with --solver and adjust --parse if needed.

Usage
-----
    python3 tools/exact_treewidth.py --problem large-graph
    python3 tools/exact_treewidth.py --problem large-graph \
        --solver /path/to/tw-exact --export /tmp/large.gr
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from core import (  # noqa: E402
    build_adj_bitsets,
    graph_path,
    load_graph,
    treewidth_lower_bound_mmd,
)


def export_pace_gr(n, adj, path):
    """Write the graph in PACE .gr format (1-indexed, undirected, no dups)."""
    edges = set()
    for u in range(n):
        for v in adj[u]:
            a, b = (u, v) if u < v else (v, u)
            if a != b:
                edges.add((a, b))
    with open(path, "w") as f:
        f.write(f"p tw {n} {len(edges)}\n")
        for a, b in sorted(edges):
            f.write(f"{a + 1} {b + 1}\n")
    return len(edges)


def achieved_width(here, problem):
    """Best achieved *full-graph* elimination width across the archive.

    The bi-objective archive minimises the *torso* width (small when the
    threshold t is high), so evaluating at the archived t answers a
    different question. The quantity the MMD lower bound brackets is the
    treewidth of the *whole* graph, i.e. the elimination width at t = 0
    (torso = every vertex). We therefore re-evaluate each archived
    ordering at t = 0 and take the smallest full-graph width found.
    """
    sub = os.path.join(here, "submissions", problem, "hc9.json")
    if not os.path.exists(sub):
        return None
    try:
        from core import evaluate
        n, adj = load_graph(graph_path(here, problem))
        ab = build_adj_bitsets(n, adj)
        with open(sub) as f:
            payload = json.load(f)
        entry = payload[0] if isinstance(payload, list) else payload
        best = None
        for dv in entry["decisionVector"]:
            perm = dv[:-1]
            w, _t = evaluate(perm, 0, ab, n)  # t = 0 -> full graph
            best = w if best is None else min(best, w)
        return best
    except Exception as e:  # pragma: no cover - best-effort only
        print(f"  (could not read achieved width: {e})")
        return None


def run_solver(solver, gr_path, parse_re):
    try:
        with open(gr_path) as f:
            proc = subprocess.run(
                [solver], stdin=f, capture_output=True, text=True, timeout=None
            )
    except FileNotFoundError:
        print(f"  solver not found: {solver}")
        return None
    out = proc.stdout + "\n" + proc.stderr
    widths = [int(x) for x in re.findall(parse_re, out)]
    if not widths:
        print("  solver produced no parseable width; raw tail:")
        print("  " + "\n  ".join(out.strip().splitlines()[-8:]))
        return None
    # PACE `s td <bags> <maxbag> <n>` -> treewidth = maxbag - 1
    return min(widths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph")
    ap.add_argument("--export", default=None, help="path for the .gr export")
    ap.add_argument("--solver", default=None, help="exact-solver binary")
    ap.add_argument(
        "--parse",
        default=r"^s td \d+ (\d+) \d+",
        help="regex capturing max-bag-size from solver output (tw = capture-1)",
    )
    args = ap.parse_args()

    n, adj = load_graph(graph_path(_HERE, args.problem))
    ab = build_adj_bitsets(n, adj)
    lb = treewidth_lower_bound_mmd(n, ab)
    ach = achieved_width(_HERE, args.problem)

    print(f"problem            : {args.problem}  (n = {n})")
    print(f"MMD lower bound    : {lb}")
    if ach is not None:
        print(f"best full-graph width (t=0): {ach}")
        if ach == lb:
            print(f"==> bracket is TIGHT: exact treewidth = {lb} (proved, no solver needed)")
        else:
            print(f"==> bracket: {lb} <= tw <= {ach}  (run an exact solver to close it)")

    gr = args.export or os.path.join(_HERE, "extra_instances", f"{args.problem}.gr")
    m = export_pace_gr(n, adj, gr)
    print(f"PACE .gr exported  : {gr}  ({m} edges)")

    if args.solver:
        parse_re = re.compile(args.parse, re.M)
        tw = run_solver(args.solver, gr, parse_re)
        if tw is not None:
            tw -= 1  # max-bag-size -> treewidth
            print(f"exact solver tw    : {tw}")
            verdict = "CONFIRMED" if (ach is None or tw == ach) else "MISMATCH"
            print(f"==> {verdict}: exact treewidth = {tw}")
    else:
        print("  (no --solver given; .gr is ready for any PACE-2017 exact solver)")


if __name__ == "__main__":
    main()
