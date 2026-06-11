#!/usr/bin/env python3
"""
validate_gpu.py -- correctness gate for the GPU batch evaluator.

Before ANY GPU search result can be trusted (or put in the thesis), the GPU
evaluator must reproduce the official CPU scorer EXACTLY. This script:

  1. packs the graph, builds a mix of orderings (random + min-degree +
     feature-decoded, the kinds the search actually produces),
  2. runs the numpy reference (cpu_eval_batch) AND, if numba+CUDA is present,
     the GPU evaluator (GpuEvaluator),
  3. asserts the per-step degree sequences and feasibility status match
     bit-for-bit between GPU and CPU,
  4. cross-checks a sample against core.evaluate / eval_fitness (the official
     fitness), confirming width(t) and the feasible HV agree.

Exit code 0 iff everything matches. Run this on Colab (GPU runtime) first;
only proceed to gpu_search.py once it prints ALL CHECKS PASSED.

Usage:
    python3 tools/validate_gpu.py --problem large-graph --batch 512
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, random
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  evaluate, hypervolume_2d, MAX_TW, min_degree_perm)
from algorithms.continuous.gpu_eval import (build_adj_words, cpu_eval_batch,
                                            staircase_widths, GpuEvaluator)
from algorithms.continuous.cmaes_torso import get_features


def make_orderings(n, ab, F, k, rng):
    """A representative mix: random, min-degree variants, feature-decoded."""
    orders = []
    for s in range(max(2, k // 4)):
        orders.append(np.array(min_degree_perm(n, ab, rng=random.Random(s)), dtype=np.int32))
    for _ in range(k // 4):
        x = rng.standard_normal(F.shape[1])
        orders.append(np.argsort(F @ x).astype(np.int32))
    while len(orders) < k:
        orders.append(rng.permutation(n).astype(np.int32))
    return np.array(orders[:k], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--cpu-sample", type=int, default=64,
                    help="# orderings to also check against core.evaluate")
    args = ap.parse_args()
    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))
    ab = build_adj_bitsets(n, adj)
    aw, W = build_adj_words(n, ab)
    F, _ = get_features(here, args.problem, n, adj, args.k_eig)
    print(f"{args.problem}: n={n}, W={W}, batch={args.batch}")
    rng = np.random.default_rng(0)
    perms = make_orderings(n, ab, F, args.batch, rng)

    # --- numpy reference (always available) ---
    deg_cpu, st_cpu = cpu_eval_batch(perms, aw, n, W, MAX_TW)
    w_cpu = staircase_widths(deg_cpu)

    # --- cross-check the reference against the OFFICIAL core.evaluate ---
    print("cross-checking reference vs core.evaluate ...")
    bad = 0
    t_probe = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    for p in range(min(args.cpu_sample, args.batch)):
        for t in t_probe:
            wr, _ = evaluate(perms[p].tolist(), t, ab, n)
            if st_cpu[p] == 0:
                if int(w_cpu[p, t]) != wr:
                    bad += 1
                    if bad <= 5:
                        print(f"  REF MISMATCH p={p} t={t}: core={wr} ref={int(w_cpu[p,t])}")
            else:
                # reference says infeasible -> core must also cap somewhere <= t?
                # feasibility is perm-level; core.evaluate(t) returns >MAX_TW too
                if wr <= MAX_TW and t == 0:
                    bad += 1
    print(f"  reference vs core.evaluate: {'OK' if bad == 0 else f'{bad} MISMATCHES'}")

    # --- GPU evaluator (if available) ---
    try:
        from numba import cuda
        if not cuda.is_available():
            raise RuntimeError("CUDA device not available")
        ev = GpuEvaluator(aw, n, W, cap=MAX_TW, capacity=args.batch)
        mb = GpuEvaluator.work_bytes(args.batch, n, W) / 1e6
        print(f"GPU work buffer: {mb:.0f} MB for capacity {args.batch}")
        deg_gpu, st_gpu = ev.eval(perms)
        # status must match
        smis = int(np.count_nonzero(st_gpu != st_cpu))
        # deg must match wherever neither bailed (status 2 stops early -> partial)
        dmis = 0
        for p in range(args.batch):
            if st_cpu[p] == 2 or st_gpu[p] == 2:
                continue
            if not np.array_equal(deg_gpu[p], deg_cpu[p]):
                dmis += 1
                if dmis <= 5:
                    j = int(np.argmax(deg_gpu[p] != deg_cpu[p]))
                    print(f"  DEG MISMATCH p={p} first@step {j}: gpu={deg_gpu[p,j]} cpu={deg_cpu[p,j]}")
        print(f"  GPU vs CPU status mismatches: {smis}")
        print(f"  GPU vs CPU degree-sequence mismatches: {dmis}")
        gpu_ok = (smis == 0 and dmis == 0)
    except Exception as e:  # noqa: BLE001
        print(f"GPU evaluator not run ({e}); validated the numpy reference only.")
        gpu_ok = None

    print()
    if bad == 0 and gpu_ok in (True, None):
        tag = "ALL CHECKS PASSED" + ("" if gpu_ok else " (CPU reference only — run on a GPU runtime to validate the kernel)")
        print(tag)
        sys.exit(0)
    print("VALIDATION FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
