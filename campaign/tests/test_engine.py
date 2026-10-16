#!/usr/bin/env python3
"""
test_engine.py -- the C engine against the authoritative evaluator.

esa_eval.py (repository root) is the evaluator cross-checked against ESA's own
code. Everything the engine computes must match it exactly:

  1. graph loading (n, edges) on all 33 instances
  2. step degrees and the cap for random and min-degree orderings
  3. the official fitness (width, t) at many thresholds
  4. staircases and archive scores (exact HSSP-20) against torso.Front
  5. HSSP on random fronts against esa_eval's dynamic program
  6. windowed incremental evaluation against full evaluation
  7. greedy constructions: valid permutations with exact degrees
  8. Team HRI's repair against hri_lns.balanced_repair's logic

    python3 campaign/tests/test_engine.py            # quick (small/medium + 1 large)
    python3 campaign/tests/test_engine.py --full     # every instance
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(CAMPAIGN)
sys.path.insert(0, ROOT)
sys.path.insert(0, CAMPAIGN)

import esa_eval                       # noqa: E402
from torso import Graph, Solution, Front   # noqa: E402
import torso_engine as te             # noqa: E402

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  FAIL:", msg)


def instances(full):
    names = ["small-graph", "medium-graph", "large-graph"] + [f"synth-{i}" for i in range(1, 31)]
    if not full:
        names = ["small-graph", "medium-graph", "synth-1", "synth-4", "synth-15", "large-graph"]
    return [(nm, os.path.join(ROOT, "data", nm + ".gr")) for nm in names]


def py_min_degree(graph, rng):
    import hill_climbing
    return hill_climbing.Starts.min_degree(graph, rng)


def balanced_repair_ordered(perm, order, adj):
    """hri_lns.balanced_repair with the hole iterated in a given order.
    (The original iterates sorted(set, key=-degree): ties follow set order.)"""
    destroyed = set(order)
    base = [u for u in perm if u not in destroyed]
    pos = {u: i for i, u in enumerate(base)}
    for u in order:
        placed = sorted(pos[x] for x in adj[u] if x in pos)
        k = len(placed)
        if k == 0:
            return None                      # random slot: not comparable
        elif k % 2 == 0:
            j = placed[k // 2]
        else:
            mid = placed[k // 2]
            w = base[mid]
            wn = [pos[x] for x in adj[w] if x in pos]
            before = sum(1 for x in wn if x < mid)
            after = len(wn) - before
            j = mid if before + 1 - after <= after + 1 - before else mid + 1
        base.insert(min(j, len(base)), u)
        pos = {x: i for i, x in enumerate(base)}
    return base


def test_instance(name, path, heavy):
    t0 = time.time()
    g = Graph.load(path)
    eng = te.Engine(path)
    check(eng.n == g.n, f"{name}: n {eng.n} != {g.n}")
    check(eng.m == g.edge_count, f"{name}: edges {eng.m} != {g.edge_count}")
    check(all(eng.deg0[v] == len(g.adj[v]) for v in range(g.n)), f"{name}: degrees differ")
    n = g.n
    rng = random.Random(7)
    crng = te.Rng(11)

    perms = []
    for s in range(2):
        perms.append(py_min_degree(g, random.Random(s)))
    p, d, bad = eng.greedy(crng, 0)
    perms.append(p.tolist())
    if heavy:
        r = list(range(n)); rng.shuffle(r); perms.append(r)
        p2, _, _ = eng.greedy(crng, 2)
        perms.append(p2.tolist())

    for k, perm in enumerate(perms):
        ref = esa_eval.step_degrees(perm, g.bits, n)
        deg, bad = eng.eval(np.array(perm, dtype=np.int32), abort_cap=False)
        check(list(deg) == ref, f"{name}: step degrees differ on ordering {k}")
        ref_bad = next((i for i, x in enumerate(ref) if x > esa_eval.MAX_TW), -1)
        check(bad == ref_bad, f"{name}: first cap breach {bad} != {ref_bad} on ordering {k}")
        # staircase
        sol = Solution(g, perm)
        stairs = sol.staircase()
        w, t = eng.staircase(deg)
        check(list(zip(w.tolist(), t.tolist())) == stairs, f"{name}: staircase differs on ordering {k}")
        # official fitness at several thresholds
        for tt in [0, 1, n // 3, n // 2, n - 2, n - 1] + [rng.randrange(n) for _ in range(4)]:
            fw, ft = esa_eval.evaluate(perm, tt, g.bits, n)
            check(eng.fitness(np.array(perm, dtype=np.int32), tt) == fw,
                  f"{name}: fitness at t={tt} differs ({eng.fitness(np.array(perm, dtype=np.int32), tt)} vs {fw})")
        # best_t_for_width
        for target in sorted({int(x) for x in w[:: max(1, len(w) // 6)]} | {0, 3}):
            check(eng.best_t_for_width(deg, target) == (sol.best_t_for_width(target) if sol.best_t_for_width(target) is not None else n),
                  f"{name}: best_t_for_width({target}) differs")

    # greedy degrees are the exact degrees of the ordering it returns
    for slack in (0, 1, 3):
        p, d, bad = eng.greedy(crng, slack)
        check(eng.is_perm(p), f"{name}: greedy slack {slack} not a permutation")
        d2, bad2 = eng.eval(p, abort_cap=False)
        check(np.array_equal(d, d2), f"{name}: greedy slack {slack} degrees not exact")
        check(bad == bad2, f"{name}: greedy slack {slack} cap flag differs")
    # head-first greedy
    head = np.zeros(n, dtype=np.uint8)
    hv = rng.sample(range(n), n // 10)
    head[hv] = 1
    p, d, bad = eng.greedy(crng, 0, head=head)
    check(set(p[: n // 10].tolist()) == set(hv), f"{name}: head-first greedy does not put the head first")
    d2, _ = eng.eval(p, abort_cap=False)
    check(np.array_equal(d, d2), f"{name}: head-first greedy degrees not exact")

    # archive vs torso.Front, offering every ordering above plus perturbations
    front = Front(n)
    arch = te.Archive(n)
    offers = list(perms)
    for _ in range(6 if heavy else 3):
        base = list(rng.choice(perms))
        for _ in range(rng.randint(1, 30)):
            i, j = rng.randrange(n), rng.randrange(n)
            base[i], base[j] = base[j], base[i]
        offers.append(base)
    for perm in offers:
        front.add_solution(Solution(g, perm))
        arch.eval_add(eng, np.array(perm, dtype=np.int32))
        check(arch.score() == front.score(), f"{name}: archive score {arch.score()} != Front {front.score()}")
    ws, ts = arch.selected()
    check(sorted(zip(ws.tolist(), ts.tolist())) == sorted(front.best_k(20)) or
          arch.hv() == int(esa_eval.hypervolume_2d(front.best_k(20), n)),
          f"{name}: selected points differ in HV")
    # decision vectors validate through esa_eval and reproduce the score
    dvs = arch.decision_vectors()
    pts = []
    for dv in dvs:
        fw, ft = esa_eval.evaluate(dv[:-1], dv[-1], g.bits, n)
        pts.append((fw, ft))
    check(-int(esa_eval.hypervolume_2d(pts, n)) == arch.score(), f"{name}: decision vectors do not reproduce the score")
    # export / restore
    a2 = te.Archive.restore(arch.export())
    check(a2.score() == arch.score(), f"{name}: restored archive score differs")
    check(np.array_equal(a2.best_t_all(), arch.best_t_all()), f"{name}: restored archive widths differ")

    # incremental evaluation against full evaluation
    inc = te.Inc(eng, snapshots=16)
    cur = np.array(perms[0], dtype=np.int32)
    inc.set(cur)
    for trial in range(40 if heavy else 15):
        new = cur.copy()
        kind = trial % 4
        if kind == 0:                                   # swap
            i, j = sorted(rng.sample(range(n), 2))
            new[i], new[j] = new[j], new[i]; lo, hi = i, j
        elif kind == 1:                                 # move one vertex
            i, j = rng.randrange(n), rng.randrange(n)
            v = new[i]; lst = new.tolist(); lst.pop(i); lst.insert(j, v); new = np.array(lst, dtype=np.int32)
            lo, hi = min(i, j), max(i, j)
        elif kind == 2:                                 # reverse a segment
            i = rng.randrange(n - 2); j = min(n - 1, i + rng.randint(2, 60))
            new[i:j + 1] = new[i:j + 1][::-1]; lo, hi = i, j
        else:                                           # shuffle a window
            i = rng.randrange(n - 2); j = min(n - 1, i + rng.randint(2, 200))
            seg = new[i:j + 1].copy(); np.random.default_rng(trial).shuffle(seg); new[i:j + 1] = seg; lo, hi = i, j
        d_inc = inc.eval(new, lo, hi)
        d_full, _ = eng.eval(new, abort_cap=False)
        check(np.array_equal(d_inc, d_full), f"{name}: incremental eval differs (move {kind}, window {lo}-{hi})")
        if trial % 3 == 0:
            inc.accept(new, lo, hi)
            cur = new
            p_now, d_now = inc.get()
            check(np.array_equal(p_now, cur) and np.array_equal(d_now, d_full), f"{name}: inc accept state wrong")

    # HRI balanced repair against the reference logic
    adj = [sorted(a) for a in g.adj]
    agree = tested = 0
    for trial in range(30 if heavy else 12):
        perm = np.array(rng.choice(perms), dtype=np.int32)
        hole = te.hri_neighbour_destroy(eng, perm, 220, crng)
        check(len(set(hole.tolist())) == len(hole), f"{name}: destroy produced duplicates")
        check(len(hole) <= 220, f"{name}: destroy exceeded the cap")
        degs = [len(adj[u]) for u in hole]
        check(all(degs[i] >= degs[i + 1] for i in range(len(degs) - 1)), f"{name}: hole not sorted by degree")
        ref = balanced_repair_ordered(perm.tolist(), hole.tolist(), adj)
        out = te.hri_balanced_repair(eng, perm, hole, crng)
        check(eng.is_perm(out), f"{name}: repair output is not a permutation")
        if ref is not None:
            tested += 1
            agree += int(out.tolist() == ref)
        rd = te.hri_random_destroy_repair(eng, perm, 6, crng)
        check(eng.is_perm(rd), f"{name}: random destroy/repair output is not a permutation")
    check(agree == tested, f"{name}: balanced repair differs from reference on {tested - agree}/{tested}")
    print(f"  {name}: n={n} m={eng.m}  ok ({time.time() - t0:.1f}s, repair compared {tested})")


def test_hssp():
    rng = random.Random(3)
    for trial in range(60):
        n = rng.randint(30, 3000)
        m = rng.randint(1, 120)
        xs = sorted(rng.sample(range(0, min(n - 1, 501)), min(m, min(n - 1, 501))))
        ys = sorted(rng.sample(range(0, n - 1), len(xs)), reverse=True)
        arch = esa_eval.ParetoArchive()
        for x, y in zip(xs, ys):
            arch.try_add(x, y, None)
        k = rng.choice([1, 2, 5, 20, 20, 40])
        ref_pts = arch.top_k_by_hv_contribution(k, n)
        ref = esa_eval.hypervolume_2d([(w, t) for (w, t, _) in ref_pts], n)
        hv, idx = te.hssp(xs, ys, n, k)
        check(hv == int(ref), f"hssp trial {trial}: {hv} != {ref}")
        chosen = [(xs[i], ys[i]) for i in idx]
        check(int(esa_eval.hypervolume_2d(chosen, n)) == hv and len(chosen) == min(k, len(xs)),
              f"hssp trial {trial}: chosen subset inconsistent")
    print(f"  hssp: 60 random fronts checked ({len(FAIL)} failures so far)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    test_hssp()
    for name, path in instances(a.full):
        heavy = not name.startswith(("large", "synth-6", "synth-7")) and int(name.split("-")[1]) not in range(23, 31) if name.startswith("synth") else not name.startswith("large")
        test_instance(name, path, heavy)
    print(f"\n{len(FAIL)} failures, {time.time() - t0:.0f}s")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
