#!/usr/bin/env python3
"""
gaps_search.py -- GBDT-Augmented Polynomial Spectral search (GAPS).

THE NOVEL METHOD. THESIS.md §9 establishes that the residual gap to the
leaderboard is NOT a throughput limit but a *decode-expressiveness* limit: every
ordering the continuous encoding can produce is argsort(F·x) for a weight vector
x over the linear spectral features F, a low-dimensional manifold the leader's
solutions lie off. GAPS is the method that directly attacks that limit, with
gradient-boosted decision trees as the central mechanism:

  decode score  =  Phi(F) · x   +   beta · g_GBDT(F)

where
  * Phi(F) is a POLYNOMIAL expansion of the spectral features (squares +
    random-projected pairwise interactions) -- a richer *linear-in-Phi* basis
    that already reaches orderings the bare-F argsort cannot, and
  * g_GBDT(F) is a NONLINEAR score map, a gradient-boosted tree trained
    DAgger-style on the elite orderings discovered so far (node features ->
    elimination position), retrained every `--gbdt-every` generations. This is
    the off-manifold direction: a learned, nonlinear function of the features
    that no linear policy can express.

The whole decode is searched by GPU-scaled separable CMA-ES (population scored in
parallel by algorithms.continuous.gpu_eval), warm-started from the banked best.

WHY THIS IS THE RIGHT EXPERIMENT FOR ALL THREE GOALS:
  * novelty   -- a learned nonlinear decode motivated by, and tested against, the
                 project's own decode-expressiveness finding (not a reproduction).
  * GBDT      -- boosted trees are the central off-manifold mechanism; the
                 `--gbdt-every 0` switch runs the controlled poly-only vs
                 poly+GBDT ablation that *measures* their contribution in GAPS.
  * leaderboard -- the principled route past the linear-decode plateau.

ADDITIVE / SAFE: writes only the `gaps` stem (override with --algo). The GPU
evaluator is validated bit-for-bit by tools/validate_gpu.py; run that first.

Usage (Colab GPU runtime, after validate_gpu.py passes):
  python3 tools/gaps_search.py --problem large-graph --pop 4096 --budget 1800
  python3 tools/gaps_search.py --problem large-graph --pop 4096 --budget 1800 \
          --gbdt-every 0 --algo gaps_nogbdt        # ablation: poly-only
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, math, time, random, json
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, hypervolume_2d, MAX_TW, min_degree_perm,
                  submission_path, write_submission, LEADERBOARD_TARGETS)
from algorithms.continuous.cmaes_torso import get_features, SepCMAES
from algorithms.continuous.gbdt_torso import make_gbdt, training_set, build_rich_features
from algorithms.continuous.gpu_eval import (build_adj_words, cpu_eval_batch,
                                            staircase_widths)

INFEAS = 1e9


def standardize(M):
    mu = M.mean(0); sd = M.std(0); sd[sd == 0] = 1.0
    return ((M - mu) / sd).astype(np.float64)


def poly_expand(F, extra_dim, seed):
    """[F | F^2 | random-projected pairwise interactions] -> standardized."""
    n, d = F.shape
    rng = np.random.default_rng(seed)
    parts = [F, F * F]
    if extra_dim > 0:
        ai = rng.integers(0, d, size=extra_dim)
        bi = rng.integers(0, d, size=extra_dim)
        parts.append(F[:, ai] * F[:, bi])
    return standardize(np.hstack(parts))


def ridge_policy(Phi, perm, lam=1.0):
    pos = np.empty(Phi.shape[0]); pos[np.asarray(perm)] = np.arange(Phi.shape[0])
    Pb = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
    A = Pb.T @ Pb + lam * np.eye(Pb.shape[1])
    return np.linalg.solve(A, Pb.T @ pos)[:-1]


def row_penalties(deg, status, cap):
    pen = np.zeros(deg.shape[0])
    for p in np.where(status == 1)[0]:
        over = deg[p][deg[p] > cap] - cap
        pen[p] = 501.0 + float(over.sum() - over[0])
    return pen


def make_evaluator(aw, n, W, cap, capacity):
    try:
        from numba import cuda
        if not cuda.is_available():
            raise RuntimeError("no CUDA device")
        from algorithms.continuous.gpu_eval import GpuEvaluator
        ev = GpuEvaluator(aw, n, W, cap=cap, capacity=capacity)
        return (lambda perms: ev.eval(perms)), "GPU"
    except Exception as e:  # noqa: BLE001
        print(f"[GPU unavailable: {e}; using slow numpy reference]", flush=True)
        return (lambda perms: cpu_eval_batch(perms, aw, n, W, cap)), "CPU-ref"


def load_warmstart(problem, here, n):
    path = os.path.join(here, "data", f"warmstart_{problem}.json")
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
        perms = [p for p in d["perms"] if len(p) == n and sorted(p) == list(range(n))]
        return np.array(perms, dtype=np.int32) if perms else None
    except Exception:  # noqa: BLE001
        return None


def run(problem, budget_s, seed, pop, here, k_eig=32, algo="gaps",
        poly_extra=128, gbdt_every=10, backend="auto", elite_k=80):
    n, adj = load_graph(graph_path(here, problem))
    ab = build_adj_bitsets(n, adj)
    aw, W = build_adj_words(n, ab)
    target = LEADERBOARD_TARGETS.get(problem)
    t_grid = sorted({int(round(i * (n - 1) / 39)) for i in range(40)})

    print(f"\n=== GAPS search -- {problem} ===")
    print(f"n = {n}, edges = {sum(map(len, adj))//2}, W = {W}, pop = {pop}")
    print(f"poly_extra = {poly_extra}, gbdt_every = {gbdt_every} "
          f"({'GBDT ON' if gbdt_every else 'poly-only ABLATION'})")
    if target is not None:
        print(f"leaderboard target = {target:,}  (more negative = better)")
    t0 = time.time()
    F = build_rich_features(here, problem, n, adj, ab, k_eig)   # spectral+structural, standardized
    Phi_base = poly_expand(F, poly_extra, seed)
    gbdt_col = np.zeros((n, 1))                                 # learned nonlinear column
    eval_fn, label = make_evaluator(aw, n, W, MAX_TW, pop)
    print(f"features F{F.shape} -> Phi{Phi_base.shape} (+1 GBDT col), "
          f"evaluator = {label}, built in {time.time()-t0:.1f}s", flush=True)

    def decode_features():
        return np.hstack([Phi_base, standardize(gbdt_col)])

    archive = ParetoArchive()
    elite_pool = []   # list of (hv, perm) for GBDT training (DAgger)

    def harvest(perms, deg, status):
        widths = staircase_widths(deg)
        fit = np.full(perms.shape[0], INFEAS)
        pen = row_penalties(deg, status, MAX_TW)
        for p in np.where(status == 0)[0]:
            front = [(int(widths[p, t]), int(t)) for t in t_grid]
            for w, t in front:
                archive.try_add(w, t, perms[p].tolist())
            hv = hypervolume_2d(front, n)
            fit[p] = -hv
            elite_pool.append((hv, perms[p].copy()))
        mild = np.where(status == 1)[0]; fit[mild] = INFEAS + pen[mild]
        fit[status == 2] = INFEAS + 2000.0
        return fit

    def retrain_gbdt():
        if not elite_pool:
            return
        elite_pool.sort(key=lambda z: -z[0])
        top = elite_pool[:elite_k]
        X, y, w = training_set([(h, list(p)) for h, p in top], F, n)
        _, model = make_gbdt(backend, seed)
        try:
            model.fit(X, y, sample_weight=w)
            pred = np.asarray(model.predict(F), dtype=np.float64)
            gbdt_col[:, 0] = pred
        except Exception as e:  # noqa: BLE001
            print(f"  [gbdt retrain skipped: {e}]", flush=True)
        del elite_pool[elite_k:]    # keep pool bounded

    # warm start from banked best
    banked = load_warmstart(problem, here, n)
    best_perm = None; best_hv = -1.0
    if banked is not None and len(banked):
        print(f"warm start: continuing from {len(banked)} banked orderings", flush=True)
        for i in range(0, len(banked), pop):
            ch = banked[i:i + pop]; dC, sC = eval_fn(ch); harvest(ch, dC, sC)
            wC = staircase_widths(dC)
            for j in range(len(ch)):
                if sC[j] == 0:
                    hv = hypervolume_2d([(int(wC[j, t]), int(t)) for t in t_grid], n)
                    if hv > best_hv:
                        best_hv = hv; best_perm = ch[j]
    if best_perm is None:
        print("warm start: min-degree", flush=True)
        sp = np.array([min_degree_perm(n, ab, rng=random.Random(seed + s)) for s in range(4)],
                      dtype=np.int32)
        d0, s0 = eval_fn(sp); harvest(sp, d0, s0); best_perm = sp[0]
    if gbdt_every:
        retrain_gbdt()                       # seed the GBDT column from banked elites

    dim = decode_features().shape[1]
    opt = SepCMAES(dim, sigma0=0.5, seed=seed)
    opt.lam = int(pop); opt.mu = opt.lam // 2
    wts = np.log(opt.mu + 0.5) - np.log(np.arange(1, opt.mu + 1))
    opt.w = wts / wts.sum(); opt.mueff = 1.0 / np.sum(opt.w ** 2)
    opt.mean = ridge_policy(decode_features(), best_perm)

    out = submission_path(here, problem, algo)

    def save():
        """Persist the current top-20 archive to the submission (crash/stop-safe)."""
        top = archive.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        write_submission(dvs, problem, out)
        return len(dvs)

    best = -archive.hypervolume(n)
    save()                                   # save the warm start immediately
    print(f"warm-start archive {len(archive)} -> score {best:,.0f}  (decode dim {dim})",
          flush=True)
    print(f"checkpointing to {out} on every new best (stop/Ctrl-C safe)", flush=True)

    gen = 0; evals = len(archive); lastlog = t0
    try:
        while time.time() - t0 < budget_s:
            Phi = decode_features()
            X = opt.ask()
            perms = np.argsort(Phi @ X.T, axis=0).T.astype(np.int32)
            deg, status = eval_fn(perms)
            fit = harvest(perms, deg, status)
            opt.tell(fit)
            gen += 1; evals += perms.shape[0]
            if gbdt_every and gen % gbdt_every == 0:
                retrain_gbdt()               # DAgger: refresh the nonlinear column
            cur = -archive.hypervolume(n)
            if cur < best:                   # new best -> checkpoint to disk
                best = cur; save()
            if time.time() - lastlog >= 5 or time.time() - t0 >= budget_s:
                feas = int(np.count_nonzero(status == 0))
                msg = (f"  gen {gen:4d} | evals {evals:8d} | feasible {feas:4d}/{pop} "
                       f"| archive {len(archive):3d} | score {best:,.0f} | t {time.time()-t0:5.0f}s")
                if target is not None:
                    g = best - target; msg += f" | gap {g:+,.0f}" + (" BEAT!" if g < 0 else "")
                print(msg, flush=True); lastlog = time.time()
    except KeyboardInterrupt:
        print("\n[interrupted — saving current best before exit]", flush=True)

    nvec = save()                            # final save (also covers normal exit)
    final = -archive.hypervolume(n)
    print(f"\nFinished in {time.time()-t0:.1f}s, {gen} gens, {evals:,} evals, archive {len(archive)}")
    print(f"Official score: {final:,.0f}")
    if target is not None:
        g = final - target
        print(f"Gap to target ({target:,}): {g:>+14,.0f}  "
              f"({'BEAT THE LEADER' if g < 0 else f'{abs(g):,.0f} short'})")
    print(f"Wrote submission: {out}  ({nvec} vectors)")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=4096)
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--poly-extra", type=int, default=128,
                    help="# random pairwise-interaction features (0 = squares only)")
    ap.add_argument("--gbdt-every", type=int, default=10,
                    help="retrain the GBDT nonlinear column every N gens (0 = poly-only ablation)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "lightgbm", "xgboost", "hist", "ridge"])
    ap.add_argument("--elite-k", type=int, default=80)
    ap.add_argument("--algo", default="gaps", help="submission stem (additive)")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, args.pop, repo_root(),
        k_eig=args.k_eig, algo=args.algo, poly_extra=args.poly_extra,
        gbdt_every=args.gbdt_every, backend=args.backend, elite_k=args.elite_k)


if __name__ == "__main__":
    main()
