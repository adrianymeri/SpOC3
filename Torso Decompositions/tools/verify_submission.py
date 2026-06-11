#!/usr/bin/env python3
"""
verify_submission.py — re-evaluate a saved submission JSON end-to-end.

Loads a submission file in canonical ESA format
    [{"challenge": ..., "problem": ..., "decisionVector": [[...π, t], ...]}]
re-runs `core.evaluate` on every decision vector, and prints:

    - per-vector fitness (max_degree, t)
    - the union-of-rectangles 2-D hypervolume vs ref (n, n)
    - the official score = -HV
    - any inconsistencies (capped vectors, dominated vectors,
      duplicate fitness points, wrong-length perms, missing vertices)

Use it on every output of the HC series before submitting and as a
sanity-check after refactors:

    python3 verify_submission.py submissions/small-graph/hc6.json
    python3 verify_submission.py submissions/medium-graph/hc4.json --quiet

Exit code 0 on success, 2 if any vector is malformed, 3 if any vector
is capped (score still computable but flagged).
"""

from __future__ import annotations

# --- sys.path bootstrap (added by restructure) ---
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import sys
from typing import List, Tuple

from core import (
    LEADERBOARD_TARGETS,
    MAX_TW,
    build_adj_bitsets,
    evaluate,
    graph_path,
    hypervolume_2d,
    load_graph,
)


def _load_submission(path: str):
    """Accept both the canonical list-of-dict form `[{...}]` (current ESA
    format) and the older single-dict form `{...}` (some archived
    files).  Both must have keys challenge / problem / decisionVector."""
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                f"{path}: expected a list with exactly one entry, got "
                f"length {len(payload)}"
            )
        entry = payload[0]
    elif isinstance(payload, dict):
        entry = payload  # legacy single-dict form
    else:
        raise ValueError(
            f"{path}: expected a list or dict at top level, got "
            f"{type(payload).__name__}"
        )
    for k in ("challenge", "problem", "decisionVector"):
        if k not in entry:
            raise ValueError(f"{path}: missing key '{k}' in submission entry")
    return entry["challenge"], entry["problem"], entry["decisionVector"]


def verify(path: str, here: str, quiet: bool = False) -> int:
    challenge, problem, decision_vectors = _load_submission(path)

    if problem not in LEADERBOARD_TARGETS:
        print(f"WARNING: unknown problem '{problem}' (no leaderboard target)")

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== verify_submission — {os.path.basename(path)} ===")
    print(f"challenge    = {challenge}")
    print(f"problem      = {problem}  (n = {n})")
    print(f"vectors      = {len(decision_vectors)}")
    print()

    issues: List[str] = []
    capped = 0
    fits: List[Tuple[int, int]] = []

    for idx, dv in enumerate(decision_vectors):
        if not isinstance(dv, list) or len(dv) != n + 1:
            issues.append(
                f"vec {idx}: wrong length {len(dv) if hasattr(dv, '__len__') else '?'} "
                f"(expected {n + 1} = perm[{n}] + t)"
            )
            continue
        perm = [int(x) for x in dv[:-1]]
        t = int(dv[-1])

        if not (0 <= t < n):
            issues.append(f"vec {idx}: t = {t} out of [0, {n - 1}]")
        if sorted(perm) != list(range(n)):
            missing = sorted(set(range(n)) - set(perm))[:5]
            extra = sorted(set(perm) - set(range(n)))[:5]
            issues.append(
                f"vec {idx}: perm not a permutation of 0..{n - 1} "
                f"(missing {missing}{'...' if len(missing) == 5 else ''}, "
                f"extra {extra}{'...' if len(extra) == 5 else ''})"
            )
            continue

        max_d, t_ret = evaluate(perm, t, adj_bits, n)
        if max_d > MAX_TW:
            capped += 1
        fits.append((max_d, t_ret))

    # Detect dominated points and duplicates.
    seen = set()
    dups = 0
    for f in fits:
        if f in seen:
            dups += 1
        seen.add(f)
    nondominated = []
    for x, y in sorted(fits, key=lambda p: (p[0], p[1])):
        if not nondominated or y < nondominated[-1][1]:
            nondominated.append((x, y))
    dominated = len(fits) - len(set(nondominated))

    # Score.
    hv = hypervolume_2d(fits, n)
    score = -hv
    target = LEADERBOARD_TARGETS.get(problem)

    if not quiet:
        print("Per-vector fitness (max_degree, t):")
        for i, (w, t) in enumerate(fits):
            cap = "  CAPPED" if w > MAX_TW else ""
            print(f"  [{i:2d}] (w = {w:4d}, t = {t:5d})  size = {n - t}{cap}")
        print()

    print(f"Capped vectors:        {capped} / {len(fits)}")
    print(f"Duplicate fitnesses:   {dups}")
    print(f"Dominated vectors:     {dominated}  "
          f"(non-dominated front size = {len(nondominated)})")
    print(f"Hypervolume:           {hv:>14,.0f}  / max possible (n*n) = {n*n:,}")
    print(f"Official score (-HV):  {score:>14,.0f}")
    if target is not None:
        gap = score - target
        verdict = "BEAT" if gap < 0 else f"{abs(gap):,} short"
        print(f"Leaderboard target:    {target:>14,}    gap {gap:>+14,.0f}  ({verdict})")
    print()

    if issues:
        print("ISSUES FOUND:")
        for msg in issues:
            print(f"  - {msg}")
        return 2

    if capped:
        return 3
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission",
                    help="path to a submission JSON in canonical ESA format")
    ap.add_argument("--quiet", action="store_true",
                    help="skip per-vector fitness printout")
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rc = verify(args.submission, here, quiet=args.quiet)
    sys.exit(rc)


if __name__ == "__main__":
    main()
