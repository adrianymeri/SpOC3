#!/usr/bin/env python3
"""
test_correctness.py -- correctness gauntlet for the evaluator and HV.

The whole HC series depends on TWO functions being computed correctly:

  core.evaluate(perm, t, adj_bits, n)  -> (max_degree, t)
  core.hypervolume_2d(points, n)       -> HV against reference (n, n)

If either is off by even one HV unit, every score in the project is
wrong and the paper falls over.  This file pins both against
independent ground truth.

Ground truth for the evaluator: a slow, set-based reference
implementation in this file (no shared code with core.evaluate).
Mirrors the official graph_torso_udp._perm2fitness semantics literally.

Ground truth for the hypervolume: a brute-force O(n^2 * |front|) grid
scan in this file (no shared code with core.hypervolume_2d).  Tractable
only for small n, but unambiguous.

Tests checked:

  1. Toy 10-vertex example matches the hand-computed README result.
  2. Evaluator agrees with reference on 50/10/5 random (perm, t)
     pairs across small / medium / large.
  3. Hypervolume agrees with brute-force on 100 random fronts.
  4. Hypervolume edge cases (empty, single at origin, point on
     reference, over-width point dominated, duplicates).
  5. Every saved submission reproduces its claimed score when fed
     through verify_submission (this is the most important test --
     it catches regressions in the WHOLE pipeline, not just one
     function).

Run:
    python3 test_correctness.py            # full suite
    python3 test_correctness.py --quick    # toy + edge cases only
"""

from __future__ import annotations

# --- sys.path bootstrap (added by restructure) ---
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import random
import re
import subprocess
import sys
from typing import List, Tuple

from core import (
    MAX_TW,
    build_adj_bitsets,
    evaluate,
    hypervolume_2d,
    load_graph,
)


# ---------------------------------------------------------------------------
# Independent reference implementations (slow but obviously correct)
# ---------------------------------------------------------------------------

def reference_evaluate(perm, t, adj_sets, n):
    """Set-based chordal-completion evaluator.  Mirrors the official
    graph_torso_udp._perm2fitness semantics:
      - At step i, take perm[i]'s currently-existing neighbours that
        come LATER in perm.  Call this its "successors".
      - If i >= t, the successor count is a torso-width candidate.
      - Width-limit check fires at ANY step if successor count > MAX_TW.
      - Fill in: connect every pair of successors with a new edge.
    """
    pos = [0] * n
    for i, v in enumerate(perm):
        pos[v] = i
    g = [set(s) for s in adj_sets]

    max_width = 0
    for i in range(n):
        u = perm[i]
        successors = {w for w in g[u] if pos[w] > i}
        deg = len(successors)
        if i >= t and deg > max_width:
            max_width = deg
        if deg > MAX_TW:
            return (MAX_TW + 1, t)
        succ_list = list(successors)
        for a in range(len(succ_list)):
            for b in range(a + 1, len(succ_list)):
                x, y = succ_list[a], succ_list[b]
                g[x].add(y)
                g[y].add(x)
    return (max_width, t)


def reference_hv(points, n):
    """O(n^2 * |front|) union-of-rectangles via grid scan.  Only feasible
    for small n; unambiguously correct."""
    if n > 60:
        raise ValueError("reference_hv only feasible for n <= 60")
    valid = [(x, y) for x, y in points if x < n and y < n]
    count = 0
    for x in range(n):
        for y in range(n):
            for px, py in valid:
                if px <= x and py <= y:
                    count += 1
                    break
    return float(count)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

TOY_EDGES = [(0, 1), (0, 2), (1, 6), (2, 8), (3, 8), (3, 9),
             (4, 5), (4, 6), (4, 7)]
TOY_PERM = [0, 1, 2, 4, 5, 3, 6, 8, 9, 7]
TOY_T = 6
TOY_N = 10


def test_toy_example() -> bool:
    """The README hand-computation: fitness (2, 6), single-point HV 32,
    two-point HV 34."""
    adj = [set() for _ in range(TOY_N)]
    for u, v in TOY_EDGES:
        adj[u].add(v); adj[v].add(u)
    ab = build_adj_bitsets(TOY_N, adj)

    fit_core = evaluate(TOY_PERM, TOY_T, ab, TOY_N)
    fit_ref = reference_evaluate(TOY_PERM, TOY_T, adj, TOY_N)
    ok_fit = fit_core == (2, 6) and fit_ref == (2, 6)

    hv1 = hypervolume_2d([(2, 6)], TOY_N)
    hv2 = hypervolume_2d([(2, 6), (0, 9)], TOY_N)
    ok_hv = hv1 == 32 and hv2 == 34

    print(f"  toy fitness:  core={fit_core}, ref={fit_ref}, "
          f"expected=(2, 6)  --> {'OK' if ok_fit else 'FAIL'}")
    print(f"  toy HV:       single={hv1} (expect 32), "
          f"two-pt={hv2} (expect 34)  --> {'OK' if ok_hv else 'FAIL'}")
    return ok_fit and ok_hv


def test_random_evaluator(seed: int = 0) -> bool:
    """Cross-check evaluate() against the set-based reference on random
    (perm, t) pairs.  Fewer trials on dense graphs (the reference is slow)."""
    ok = True
    for problem, n_trials in [('small-graph', 50),
                              ('medium-graph', 10),
                              ('large-graph', 5)]:
        random.seed(seed)
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', f'{problem}.gr')
        n, adj = load_graph(path)
        ab = build_adj_bitsets(n, adj)
        mismatches = 0
        for _ in range(n_trials):
            perm = list(range(n))
            random.shuffle(perm)
            t = random.randrange(0, n)
            r = reference_evaluate(perm, t, adj, n)
            c = evaluate(perm, t, ab, n)
            if r != c:
                mismatches += 1
                print(f"    MISMATCH {problem}: ref={r} core={c}")
        verdict = 'OK' if mismatches == 0 else f'{mismatches} FAIL'
        print(f"  {problem:13s}  {n_trials:3d} random (perm, t) trials  "
              f"--> {verdict}")
        if mismatches:
            ok = False
    return ok


def test_random_hv(n_trials: int = 100, seed: int = 0) -> bool:
    """Cross-check hypervolume_2d() against brute-force on random small fronts."""
    random.seed(seed)
    mismatches = 0
    for trial in range(n_trials):
        n = random.randint(5, 30)
        k = random.randint(0, 8)
        pts = [(random.randrange(0, n + 2), random.randrange(0, n + 2))
               for _ in range(k)]
        fast = hypervolume_2d(pts, n)
        slow = reference_hv(pts, n)
        if fast != slow:
            mismatches += 1
            print(f"    MISMATCH trial {trial}: n={n} pts={pts} "
                  f"fast={fast} slow={slow}")
    verdict = 'OK' if mismatches == 0 else f'{mismatches} FAIL'
    print(f"  {n_trials} random HV fronts (n in [5, 30])  --> {verdict}")
    return mismatches == 0


def test_hv_edges() -> bool:
    """Hypervolume edge cases."""
    cases = [
        ("empty front",                    hypervolume_2d([], 10), 0.0),
        ("single (0, 0) vs ref (10, 10)",  hypervolume_2d([(0, 0)], 10), 100.0),
        ("point exactly at ref",           hypervolume_2d([(10, 10)], 10), 0.0),
        ("point past ref",                 hypervolume_2d([(11, 11)], 10), 0.0),
        ("duplicates (5, 5)+(5, 5)",       hypervolume_2d([(5, 5), (5, 5)], 10), 25.0),
        ("over-width point shadowed",
         hypervolume_2d([(501, 5), (20, 5)], 1357),
         hypervolume_2d([(20, 5)], 1357)),
    ]
    ok = True
    for label, got, expect in cases:
        match = got == expect
        if not match:
            ok = False
        print(f"  {label:42s} got={got}, expect={expect}  "
              f"--> {'OK' if match else 'FAIL'}")
    return ok


def test_saved_submissions() -> bool:
    """Every saved submission must reproduce its score via verify_submission."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = failed = 0
    for problem in ['small-graph', 'medium-graph', 'large-graph']:
        sub_dir = os.path.join(here, 'submissions', problem)
        if not os.path.isdir(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(sub_dir, fname)
            found += 1
            r = subprocess.run(
                [sys.executable, 'tools/verify_submission.py',
                 path, '--quiet'],
                capture_output=True, text=True, cwd=here, timeout=60,
            )
            ok = r.returncode in (0, 3)   # 3 = capped but reproducible
            m = re.search(r'Official score \(-HV\):\s+(-?[\d,]+)', r.stdout)
            score = m.group(1) if m else '?'
            verdict = 'OK' if ok else 'FAIL'
            print(f"  {problem:14s} {fname:14s} score={score:>16s}  "
                  f"--> {verdict}")
            if not ok:
                failed += 1
    print(f"  {found} submissions, {failed} failure(s)")
    return failed == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the heavier per-instance random-eval and "
                         "submission-roundtrip tests")
    args = ap.parse_args()

    results = []

    print("\n[1/5] Toy 10-vertex example (anchored to README hand-computation)")
    results.append(("toy", test_toy_example()))

    print("\n[2/5] Hypervolume edge cases")
    results.append(("hv_edges", test_hv_edges()))

    print("\n[3/5] Random hypervolume cross-check (100 random small fronts)")
    results.append(("random_hv", test_random_hv()))

    if not args.quick:
        print("\n[4/5] Random evaluator cross-check (small/medium/large)")
        results.append(("random_eval", test_random_evaluator()))

        print("\n[5/5] All saved submissions reproduce their scores")
        results.append(("saved_subs", test_saved_submissions()))
    else:
        print("\n[4/5] skipped (--quick)")
        print("[5/5] skipped (--quick)")

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {name:14s}  {'OK' if ok else 'FAIL'}")
    print("=" * 60)
    print("ALL TESTS PASSED" if all_ok else "TEST FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
