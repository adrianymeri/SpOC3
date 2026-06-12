#!/usr/bin/env python3
"""
gbdt_torso.py -- GBDT-boosted continuous-encoding search ("supercharged").

The idea
--------
The continuous encoding scores each vertex by a *linear* policy over spectral
features and decodes the ordering by argsort; CMA-ES tunes the linear weights.
The leaderboard winner (cuda-torso) beat the linear policy with *polynomial*
features -- a nonlinear scoring function. We confirmed that copying that by
adding eigenvectors backfires: more features raise the CMA-ES search dimension
and *slow convergence* at a fixed budget (K = 48 stalled).

A gradient-boosted decision tree (LightGBM / XGBoost / sklearn HistGBR) is a
nonlinear function of node features with automatic feature interactions to
arbitrary order -- cuda-torso's polynomial trick and beyond -- but it is fit by
*boosting*, not by CMA-ES. So it adds policy nonlinearity at **zero CMA-ES
dimension cost**, dissolving the wall that capped K. The only thing GBDT needs
is training targets, and we have a goldmine: every elite ordering produced by
the CMA-ES sweeps labels each vertex with a good elimination *position*.

The pipeline (policy distillation + DAgger)
-------------------------------------------
  1. Build a RICH node-feature matrix F: degree profile, k-core number,
     clustering coefficient, triangle count, average-neighbour degree, and K
     Laplacian eigenvectors. GBDT does not care about dimension, so be generous.
  2. Load every elite ordering from the cmaes submissions; score each by the
     hypervolume of the front it induces (one elimination pass each); keep the
     top M. These seed the global archive (so we start no worse than the
     existing portfolio).
  3. Build a supervised set: for each top elite ordering pi and vertex v, the
     row is (F[v] -> position of v in pi), sample-weighted by pi's HV. Train a
     GBDT to predict position from features. argsort(GBDT.predict(F)) is a new,
     NONLINEAR ordering that the linear policy could never represent.
  4. Refine: fit a linear policy to the GBDT ordering (ridge), warm-start
     CMA-ES from it, and run a short refinement on the rich features. The new
     good orderings join the elite pool.
  5. DAgger: retrain the GBDT on the enlarged/improved elite pool and repeat.
     Each round the labels come from better orderings, so the distilled policy
     bootstraps beyond the original CMA-ES elites.

Backends: --backend auto tries lightgbm, then xgboost, then sklearn HistGBR,
then a numpy ridge fallback (so it always runs). On the workstation install
LightGBM for the strongest model: `pip3 install lightgbm`.

ADDITIVE / SAFE: writes only the `gbdt` submission stem (override with --algo).

Usage
-----
    python3 algorithms/continuous/gbdt_torso.py --problem small-graph --budget 600
    python3 algorithms/continuous/gbdt_torso.py --problem large-graph \
        --eigenvectors 32 --rounds 4 --backend lightgbm --budget 1800
"""
from __future__ import annotations

import sys as _sys, os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "core.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

import argparse, glob, json, time
import numpy as np

from core import (LEADERBOARD_TARGETS, MAX_TW, ParetoArchive, build_adj_bitsets,
                  graph_path, hypervolume_2d, load_graph, repo_root,
                  submission_path, write_submission)
from algorithms.continuous.cmaes_torso import (build_node_features, eval_fitness,
                                               decode, SepCMAES, get_features)


# --------------------------------------------------------------------------- #
# Rich structural node features (cheap graph statistics GBDT can exploit)
# --------------------------------------------------------------------------- #
def structural_features(n, adj_bits, adj):
    deg = np.array([len(adj[v]) for v in range(n)], dtype=np.float64)
    # k-core number via iterative min-degree peeling
    core = np.zeros(n)
    g = list(adj_bits); remaining = (1 << n) - 1; curcore = 0
    dleft = [len(adj[v]) for v in range(n)]
    order_left = sorted(range(n), key=lambda v: dleft[v])
    removed = [False] * n
    import heapq
    heap = [(dleft[v], v) for v in range(n)]
    heapq.heapify(heap)
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v]:
            continue
        if d > curcore:
            curcore = d
        core[v] = curcore
        removed[v] = True
        m = g[v]
        while m:
            b = m & -m; m ^= b; u = b.bit_length() - 1
            if not removed[u]:
                dleft[u] -= 1
                heapq.heappush(heap, (dleft[u], u))
        # detach v
        nb = g[v]
        while nb:
            b = nb & -nb; nb ^= b; u = b.bit_length() - 1
            g[u] &= ~(1 << v)
    # clustering coefficient + triangle count via bitset intersections
    tri = np.zeros(n); clus = np.zeros(n)
    for v in range(n):
        nbrs = adj_bits[v]; dv = deg[v]
        if dv < 2:
            continue
        t = 0; m = nbrs
        while m:
            b = m & -m; m ^= b; u = b.bit_length() - 1
            t += (adj_bits[u] & nbrs).bit_count()
        t //= 2
        tri[v] = t
        clus[v] = 2.0 * t / (dv * (dv - 1))
    # average neighbour degree
    avgnd = np.zeros(n)
    for v in range(n):
        if adj[v]:
            avgnd[v] = deg[list(adj[v])].mean()
    return np.vstack([core, clus, tri, avgnd]).T  # (n, 4)


def build_rich_features(here, problem, n, adj, adj_bits, k_eig):
    spec, _ = get_features(here, problem, n, adj, k_eig)     # degree profile + eigenvectors (cached)
    struct = structural_features(n, adj_bits, adj)
    F = np.hstack([spec, struct])
    mu = F.mean(0); sd = F.std(0); sd[sd == 0] = 1.0
    return ((F - mu) / sd).astype(np.float64)


# --------------------------------------------------------------------------- #
# GBDT backend abstraction
# --------------------------------------------------------------------------- #
def make_gbdt(backend, seed, rank=False, device="cpu"):
    """Return (name, model). If rank=True, return a listwise learning-to-rank
    model (LambdaMART) where supported — the principled objective for "rank the
    chosen next-vertex highest" (each elimination step is a ranking query).
    Falls back to pointwise regression where ranking is unavailable.

    device: "cpu" (default) or "gpu". GPU routing is implementation-specific:
      * XGBoost  -> device="cuda" (works with the stock CUDA pip wheel).
      * LightGBM -> device_type="gpu" (ONLY if the wheel was built with GPU
        support; the default pip wheel is CPU-only and will raise, in which
        case we fall through to the next backend).
    The GPU only accelerates tree *training*; it does not change the learned
    orderings or the hypervolume — see THESIS.md s9."""
    gpu = str(device).lower() in ("gpu", "cuda")
    # GBDT_NJOBS caps the library's thread pool (default -1 = all cores).
    # Parallel drivers (gbfcpp_swarm, diversify) set GBDT_NJOBS=1: with N
    # workers, n_jobs=-1 spawns N x cores threads and crushes the host.
    njobs = int(_os.environ.get("GBDT_NJOBS", "-1"))
    order = {"auto": ["lightgbm", "xgboost", "hist", "numpy", "ridge"]}.get(backend, [backend])
    for b in order:
        try:
            if b == "lightgbm":
                import lightgbm as lgb
                gkw = {"device_type": "gpu"} if gpu else {}
                if rank:
                    return ("lightgbm-rank" + ("-gpu" if gpu else ""), lgb.LGBMRanker(
                        objective="lambdarank", n_estimators=600,
                        learning_rate=0.05, num_leaves=63, subsample=0.8,
                        colsample_bytree=0.8, random_state=seed, n_jobs=njobs,
                        verbose=-1, **gkw))
                return ("lightgbm" + ("-gpu" if gpu else ""), lgb.LGBMRegressor(
                    n_estimators=600, learning_rate=0.05, num_leaves=63,
                    subsample=0.8, colsample_bytree=0.8, random_state=seed,
                    n_jobs=njobs, verbose=-1, **gkw))
            if b == "xgboost":
                import xgboost as xgb
                gkw = {"device": "cuda", "tree_method": "hist"} if gpu else {}
                if rank:
                    return ("xgboost-rank" + ("-gpu" if gpu else ""), xgb.XGBRanker(
                        objective="rank:pairwise", n_estimators=600,
                        learning_rate=0.05, max_depth=7, subsample=0.8,
                        colsample_bytree=0.8, random_state=seed, n_jobs=njobs,
                        verbosity=0, **gkw))
                return ("xgboost" + ("-gpu" if gpu else ""), xgb.XGBRegressor(
                    n_estimators=600, learning_rate=0.05, max_depth=7,
                    subsample=0.8, colsample_bytree=0.8, random_state=seed,
                    n_jobs=njobs, verbosity=0, **gkw))
            if b == "hist":
                from sklearn.ensemble import HistGradientBoostingRegressor
                return ("hist", HistGradientBoostingRegressor(
                    max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                    l2_regularization=1.0, random_state=seed))
            if b == "numpy":
                from algorithms.continuous.np_gbdt import NpGBDT
                return ("numpy-gbdt", NpGBDT(n_estimators=200, learning_rate=0.08,
                                             max_depth=4, subsample=0.8, seed=seed))
            if b == "ridge":
                return ("ridge", _Ridge())
        except Exception as e:  # noqa: BLE001
            print(f"  [backend {b} unavailable: {e}]", flush=True)
    return ("ridge", _Ridge())


class _Ridge:
    """numpy ridge-regression fallback (so the pipeline always runs)."""
    def __init__(self, lam=1.0): self.lam = lam; self.w = None
    def fit(self, X, y, sample_weight=None):
        Xb = np.hstack([X, np.ones((len(X), 1))])
        if sample_weight is not None:
            sw = np.sqrt(sample_weight)[:, None]; Xb = Xb * sw; yy = y * sw[:, 0]
        else:
            yy = y
        A = Xb.T @ Xb + self.lam * np.eye(Xb.shape[1]); self.w = np.linalg.solve(A, Xb.T @ yy)
        return self
    def predict(self, X): return np.hstack([X, np.ones((len(X), 1))]) @ self.w


# --------------------------------------------------------------------------- #
# Elite-ordering corpus
# --------------------------------------------------------------------------- #
def load_elites(here, problem, n, adj_bits, t_grid, archive, top_m):
    """Read every elite ordering from the cmaes submissions, seed the archive,
    and return the top-M unique perms by induced-front HV."""
    files = (glob.glob(_os.path.join(here, "submissions", problem, "seeds", "cmaes*.json"))
             + glob.glob(_os.path.join(here, "submissions", problem, "cmaes*.json"))
             + glob.glob(_os.path.join(here, "submissions", problem, "portfolio.json")))
    seen = set(); scored = []
    for fp in files:
        try:
            dvs = json.load(open(fp))[0]["decisionVector"]
        except Exception:
            continue
        for dv in dvs:
            if not isinstance(dv, list) or len(dv) != n + 1:
                continue
            perm = tuple(int(x) for x in dv[:-1])
            if perm in seen or sorted(perm) != list(range(n)):
                continue
            seen.add(perm)
            hv = -eval_fitness(list(perm), adj_bits, n, t_grid, archive)  # seeds archive + returns HV
            if hv > 0:
                scored.append((hv, list(perm)))
    scored.sort(key=lambda z: -z[0])
    return scored[:top_m]


def training_set(elites, F, n):
    """Rows: (F[v] -> position of v in elite pi), weighted by pi's HV."""
    Xs, ys, ws = [], [], []
    hmax = max(h for h, _ in elites) if elites else 1.0
    for hv, perm in elites:
        pos = np.empty(n)
        for i, v in enumerate(perm):
            pos[v] = i
        Xs.append(F); ys.append(pos); ws.append(np.full(n, hv / hmax))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# GBDT-guided ADAPTIVE elimination (learned min-degree++).  Breaks the static
# argsort ceiling: the GBDT scores remaining candidates using DYNAMIC features
# (current residual degree, #eliminated neighbours) + static structure, at every
# step. This can represent orderings no fixed per-vertex policy can.
# --------------------------------------------------------------------------- #
def _fill1(v, g, rem_big, deg_v):
    """Fill-in count if v were eliminated now: C(deg,2) - edges already present
    among v's residual neighbours. The min-fill signal -- the most informative
    dynamic feature on dense graphs, and the one min-degree ignores."""
    if deg_v < 2:
        return 0.0
    nbrs = g[v] & rem_big; existing = 0; s = nbrs
    while s:
        b = s & -s; s ^= b; u = b.bit_length() - 1
        existing += (g[u] & nbrs).bit_count()
    return float(deg_v * (deg_v - 1) // 2 - existing // 2)


def _construct(score_fn, F, adj_bits, n, rng, shortlist=64, temp=0.0):
    g = list(adj_bits); rem_big = (1 << n) - 1
    rem_arr = np.ones(n, dtype=bool)
    cur_deg = np.fromiter((b.bit_count() for b in adj_bits), dtype=np.int32, count=n)
    elimc = np.zeros(n, dtype=np.int32)
    idx = np.arange(n); perm = []; deg_seq = np.empty(n, dtype=np.int32)
    for step in range(n):
        rem_idx = idx[rem_arr]; degs = cur_deg[rem_idx]
        k = min(shortlist, len(rem_idx))
        cand = rem_idx[np.argpartition(degs, k - 1)[:k]] if k < len(rem_idx) else rem_idx
        fl = np.array([_fill1(int(v), g, rem_big, int(cur_deg[v])) for v in cand])
        feat = np.column_stack([cur_deg[cand], elimc[cand], fl, F[cand]])
        sc = np.asarray(score_fn(feat))
        if temp > 0 and len(cand) > 1:
            p = np.exp((sc - sc.max()) / temp); p /= p.sum()
            choice = int(cand[rng.choice(len(cand), p=p)])
        else:
            choice = int(cand[int(np.argmax(sc))])
        rem_big ^= (1 << choice); rem_arr[choice] = False
        succ = g[choice] & rem_big; deg_seq[step] = succ.bit_count(); perm.append(choice)
        s = succ; members = []
        while s:
            b = s & -s; s ^= b; members.append(b.bit_length() - 1)
        for u in members:
            g[u] |= succ ^ (1 << u); elimc[u] += 1
        for u in members:
            cur_deg[u] = (g[u] & rem_big).bit_count()
    return perm, deg_seq


def _replay_rows(perm, F, adj_bits, n, shortlist, neg, rng):
    """Replay an elite ordering, emitting (features -> chose-this?) rows: the
    chosen vertex is a positive, a sample of the other shortlist candidates are
    negatives. This teaches the GBDT the elite's per-step decision rule."""
    g = list(adj_bits); rem_big = (1 << n) - 1
    rem_arr = np.ones(n, dtype=bool)
    cur_deg = np.fromiter((b.bit_count() for b in adj_bits), dtype=np.int32, count=n)
    elimc = np.zeros(n, dtype=np.int32); idx = np.arange(n)
    X, y, groups = [], [], []   # groups = per-step query sizes (for ranking)
    for step in range(n):
        c = perm[step]
        rem_idx = idx[rem_arr]; degs = cur_deg[rem_idx]
        k = min(shortlist, len(rem_idx))
        cand = rem_idx[np.argpartition(degs, k - 1)[:k]] if k < len(rem_idx) else rem_idx
        gsize = 1
        X.append(np.concatenate([[cur_deg[c], elimc[c],
                                  _fill1(int(c), g, rem_big, int(cur_deg[c]))], F[c]])); y.append(1.0)
        negs = cand[cand != c]
        if len(negs):
            for u in rng.choice(negs, size=min(neg, len(negs)), replace=False):
                u = int(u)
                X.append(np.concatenate([[cur_deg[u], elimc[u],
                                          _fill1(u, g, rem_big, int(cur_deg[u]))], F[u]])); y.append(0.0)
                gsize += 1
        groups.append(gsize)
        rem_big ^= (1 << c); rem_arr[c] = False
        succ = g[c] & rem_big; s = succ; members = []
        while s:
            b = s & -s; s ^= b; members.append(b.bit_length() - 1)
        for u in members:
            g[u] |= succ ^ (1 << u); elimc[u] += 1
        for u in members:
            cur_deg[u] = (g[u] & rem_big).bit_count()
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), groups


def _feature_names(F_dim, k_eig):
    """Names for the construct feature vector [cur_deg, elim_nbr, fill, *F]."""
    base = ["cur_deg", "elim_nbr", "fill", "orig_degree",
            "nbr_deg_min", "nbr_deg_max", "nbr_deg_mean", "nbr_deg_std"]
    eig = [f"eig{i+1}" for i in range(k_eig)]
    struct = ["kcore", "clustering", "triangles", "avg_nbr_deg"]
    names = base + eig + struct
    # pad/truncate defensively to the actual width (3 dynamic + F_dim)
    return (names + [f"f{i}" for i in range(3 + F_dim)])[:3 + F_dim]


def construct_search(F, adj_bits, n, archive, elites, budget_s, seed, backend,
                     shortlist=64, neg=4, retrain_rounds=3, rank=False,
                     report_importance=False, k_eig=32, device="cpu"):
    rng = np.random.default_rng(seed)
    t0 = time.time(); built = 0; best = -archive.hypervolume(n); cap = 0

    def add_front(perm, deg_seq):
        if deg_seq.max() > MAX_TW:
            return False
        run = 0
        for t in range(n - 1, -1, -1):
            if deg_seq[t] > run:
                run = deg_seq[t]
            archive.try_add(int(run), t, perm)
        return True

    pool = elites[:]
    for rd in range(retrain_rounds):
        if time.time() - t0 >= budget_s:
            break
        # train the per-step decision GBDT on the current elite pool
        Xs, ys, grp = [], [], []
        for _, perm in pool[:60]:
            Xe, ye, ge = _replay_rows(perm, F, adj_bits, n, shortlist, neg, rng)
            Xs.append(Xe); ys.append(ye); grp.extend(ge)
        name, model = make_gbdt(backend, seed + rd, rank=rank, device=device)
        if rd == 0:
            print(f"    model = {name}", flush=True)
        Xall, yall = np.vstack(Xs), np.concatenate(ys)
        if rank and "rank" in name:
            model.fit(Xall, yall, group=grp)
        else:
            model.fit(Xall, yall)
        sf = lambda feat: model.predict(feat)
        print(f"  round {rd}: GBDT={name} trained on {len(yall):,} step-decisions", flush=True)
        if report_importance and rd == 0 and hasattr(model, "feature_importances_"):
            names = _feature_names(F.shape[1], k_eig)
            imp = np.asarray(model.feature_importances_, dtype=float)
            imp = imp / (imp.sum() + 1e-12)
            top = np.argsort(imp)[::-1][:10]
            print("    feature importances (top 10):", flush=True)
            for j in top:
                print(f"      {names[j]:<14} {imp[j]:6.1%}", flush=True)
        # greedy construction, then many stochastic ones for diversity
        rd_deadline = t0 + budget_s * (rd + 1) / retrain_rounds
        first = True
        while time.time() < rd_deadline:
            temp = 0.0 if first else float(rng.uniform(0.15, 0.9))
            first = False
            perm, ds = _construct(sf, F, adj_bits, n, rng, shortlist, temp)
            built += 1
            if not add_front(perm, ds):
                cap += 1
            sc = -archive.hypervolume(n)
            if sc < best - 1 or built % 25 == 0:
                if sc < best:
                    best = sc
                print(f"    built {built} orders ({cap} capped) | score {sc:,.0f} | t {time.time()-t0:.0f}s", flush=True)
        # DAgger: refresh elite pool from the best constructed orderings
        ents = {}
        for w, t, p in sorted(archive.entries(), key=lambda e: (e[0], e[1])):
            ents.setdefault(tuple(p), 1)
        pool = []
        for p in list(ents)[:80]:
            from algorithms.continuous.cmaes_torso import eval_fitness as _ef
            hv = -_ef(list(p), adj_bits, n, sorted({int(round(i*(n-1)/39)) for i in range(40)}), ParetoArchive())
            if hv > 0:
                pool.append((hv, list(p)))
        pool.sort(key=lambda z: -z[0])
    print(f"  construct done: {built} adaptive orderings, {cap} capped, final {-archive.hypervolume(n):,.0f}", flush=True)


def surrogate_search(F, adj_bits, n, t_grid, archive, elites, budget_s, seed,
                     backend, surrogate_factor=8, retrain_every=160):
    """Pre-selection surrogate-assisted CMA-ES with IPOP-style restarts.

    Each generation CMA-ES proposes `surrogate_factor x lambda` candidates; the
    GBDT surrogate (policy-vector -> score, trained on every real evaluation so
    far) pre-ranks them, and only the best `lambda` are truly evaluated. The
    surrogate is GLOBAL and survives restarts, so each restart benefits from all
    accumulated landscape knowledge -- combining multi-start diversity (the lever
    that works) with GBDT throughput amplification."""
    d = F.shape[1]
    rng = np.random.default_rng(seed)

    def elite_policy(perm):
        pos = np.empty(n)
        for i, v in enumerate(perm):
            pos[v] = i
        return _Ridge(lam=1.0).fit(F, pos).w[:-1].copy()

    opt = SepCMAES(d, sigma0=0.5, seed=seed); opt.mean = elite_policy(elites[0][1])
    lam = opt.lam; lam_big = lam * surrogate_factor
    Xc, fc = [], []; surrogate = None; sname = "(warmup)"
    best = -archive.hypervolume(n); stagn = 0; last = 0
    t0 = time.time(); evals = 0; screened = 0; gens = 0; restarts = 0; lastlog = 0
    while time.time() - t0 < budget_s:
        z = rng.standard_normal((lam_big, d))
        yb = z * np.sqrt(opt.C); Xb = opt.mean + opt.sigma * yb
        screened += lam_big
        if surrogate is not None:
            sel = np.argsort(np.asarray(surrogate.predict(Xb)))[:lam]   # smallest score = best
        else:
            sel = np.arange(lam)
        fsel = np.array([eval_fitness(decode(Xb[i], F), adj_bits, n, t_grid, archive)
                         for i in sel])
        evals += lam; gens += 1
        opt.z = z[sel]; opt.y = yb[sel]; opt.tell(fsel)
        for i in sel:
            Xc.append(Xb[i])
        fc.extend(fsel.tolist())
        if len(fc) - last >= retrain_every:
            sname, surrogate = make_gbdt(backend, seed)
            surrogate.fit(np.asarray(Xc), np.asarray(fc)); last = len(fc)
        sc = -archive.hypervolume(n)
        if sc < best - 1:
            best = sc; stagn = 0
        else:
            stagn += 1
        if stagn > 45:                       # IPOP restart from a random elite basin
            restarts += 1; ej = int(rng.integers(len(elites)))
            opt = SepCMAES(d, sigma0=0.5, seed=seed + 1000 + restarts)
            opt.mean = elite_policy(elites[ej][1]); stagn = 0
        if time.time() - t0 - lastlog > 20:
            lastlog = time.time() - t0
            print(f"    surrogate[{sname}] gen {gens} evals {evals} screened {screened} "
                  f"restarts {restarts} | score {sc:,.0f} | t {time.time()-t0:.0f}s", flush=True)
    print(f"  surrogate-CMA done: {gens} gens, {evals} real evals, {screened} screened "
          f"({screened/max(1,evals):.0f}x), {restarts} restarts", flush=True)


def run(problem, budget_s, seed, here, eigenvectors=32, num_t_seeds=40, mode="surrogate",
        backend="auto", rounds=3, elite_top=80, refine_frac=0.5, algo="gbdt",
        surrogate_factor=8, rank=False, report_importance=False, device="cpu"):
    rng = np.random.default_rng(seed)
    n, adj = load_graph(graph_path(here, problem)); adj_bits = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem)
    t_grid = sorted({int(round(i * (n - 1) / (num_t_seeds - 1))) for i in range(num_t_seeds)})

    print(f"\n=== gbdt_torso -- {problem} ===")
    print(f"n = {n}, target = {target:,}" if target else f"n = {n}")
    t0 = time.time()
    F = build_rich_features(here, problem, n, adj, adj_bits, eigenvectors)
    print(f"rich features: {F.shape[1]} dims (spectral + structural), built in {time.time()-t0:.1f}s", flush=True)

    archive = ParetoArchive()
    elites = load_elites(here, problem, n, adj_bits, t_grid, archive, elite_top)
    print(f"loaded {len(elites)} elite orderings; archive seeded -> score {-archive.hypervolume(n):,.0f}", flush=True)
    print(f"mode = {mode}", flush=True)

    if mode == "construct":
        if not elites:
            print("  no elites; aborting")
        else:
            construct_search(F, adj_bits, n, archive, elites, budget_s, seed, backend,
                             rank=rank, report_importance=report_importance,
                             k_eig=eigenvectors, device=device)
        rounds = 0
    elif mode == "surrogate":
        if not elites:
            print("  no elites; aborting")
        else:
            surrogate_search(F, adj_bits, n, t_grid, archive, elites, budget_s, seed,
                             backend, surrogate_factor=surrogate_factor)
        rounds = 0  # skip the feature loop below

    deadline = time.time() + budget_s
    for r in range(rounds):
        if not elites:
            print("  no elites to distill; aborting"); break
        # 1. Train the GBDT to predict good elimination position from features.
        X, y, w = training_set(elites, F, n)
        name, model = make_gbdt(backend, seed + r)
        model.fit(X, y, sample_weight=w)
        gp = np.asarray(model.predict(F), dtype=np.float64)
        gp = (gp - gp.mean()) / (gp.std() + 1e-9)
        # 2. GBDT-as-FEATURE: append the nonlinear expert prediction as a column,
        #    so a real optimiser can weight it alongside the raw features.
        Faug = np.hstack([F, gp[:, None]])
        # seed the distilled ordering (cheap, anchors the archive)
        eval_fitness([int(v) for v in np.argsort(gp, kind="stable")], adj_bits, n, t_grid, archive)
        # 3. FULL-budget CMA-ES over [spectral | structural | gbdt-expert], seeded
        #    from a ridge fit of the best elite ordering -> real HV maximisation
        #    that can EXCEED the elites (richer representation + true search).
        best_perm = elites[0][1]
        pos = np.empty(n)
        for i, v in enumerate(best_perm):
            pos[v] = i
        x0 = _Ridge(lam=1.0).fit(Faug, pos).w[:-1].copy()
        opt = SepCMAES(Faug.shape[1], sigma0=0.3, seed=seed + 200 + r)
        opt.mean = x0
        round_deadline = time.time() + (deadline - time.time()) / max(1, rounds - r)
        gens = 0
        while time.time() < round_deadline:
            Xp = opt.ask()
            opt.tell(np.array([eval_fitness(decode(Xp[i], Faug), adj_bits, n, t_grid, archive)
                               for i in range(Xp.shape[0])]))
            gens += 1
        print(f"  round {r}: GBDT={name}  +gbdt-feature CMA ({gens} gens)  "
              f"archive {len(archive)}  score {-archive.hypervolume(n):,.0f}", flush=True)

        # DAgger: rebuild the elite pool from the improved archive (labels now
        # come from BETTER orderings, so the next GBDT chases an improving front)
        fresh = {}
        for wdth, tt, p in sorted(archive.entries(), key=lambda e: (e[0], e[1])):
            fresh.setdefault(tuple(p), 1)
        elites = []
        for p in list(fresh)[:elite_top]:
            hvp = -eval_fitness(list(p), adj_bits, n, t_grid, ParetoArchive())
            if hvp > 0:
                elites.append((hvp, list(p)))
        elites.sort(key=lambda z: -z[0]); elites = elites[:elite_top]
        if time.time() >= deadline:
            break

    final = -archive.hypervolume(n)
    print(f"\nFinished in {time.time()-t0:.1f}s; archive {len(archive)}")
    print(f"Official score: {final:,.0f}")
    if target:
        gap = final - target
        print(f"Gap to target ({target:,}): {gap:>+,.0f}  ({'BEAT' if gap < 0 else f'{abs(gap):,.0f} short'})")
    top = archive.top_k_by_hv_contribution(20, n)
    dvs = [list(p) + [int(t)] for (_, t, p) in top]
    out = submission_path(here, problem, algo); write_submission(dvs, problem, out)
    print(f"Wrote submission: {out}  ({len(dvs)} vectors)")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eigenvectors", type=int, default=32)
    ap.add_argument("--num-t-seeds", type=int, default=40)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "lightgbm", "xgboost", "hist", "ridge"])
    ap.add_argument("--mode", default="construct",
                    choices=["construct", "surrogate", "feature"],
                    help="construct = GBDT-guided ADAPTIVE elimination (breaks the "
                         "static-argsort ceiling); surrogate = GBDT-assisted CMA-ES; "
                         "feature = GBDT prediction as a CMA feature")
    ap.add_argument("--surrogate-factor", type=int, default=8,
                    help="candidates screened per real evaluation")
    ap.add_argument("--rounds", type=int, default=3, help="DAgger rounds (feature mode)")
    ap.add_argument("--elite-top", type=int, default=80, help="# elite orderings to distil from")
    ap.add_argument("--refine-frac", type=float, default=0.5, help="fraction of remaining budget for CMA refine per round")
    ap.add_argument("--algo", default="gbdt")
    ap.add_argument("--rank", action="store_true",
                    help="construct mode: train a listwise learning-to-rank model "
                         "(LambdaMART) instead of pointwise regression")
    ap.add_argument("--feature-importance", action="store_true",
                    help="construct mode: print GBDT feature importances (round 0)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"],
                    help="gpu/cuda: train the boosted trees on the GPU "
                         "(XGBoost device=cuda works with the stock wheel; LightGBM "
                         "needs a GPU-compiled build). Speeds TRAINING only — does "
                         "not change orderings or hypervolume (see THESIS.md s9).")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, repo_root(),
        eigenvectors=args.eigenvectors, num_t_seeds=args.num_t_seeds, mode=args.mode,
        backend=args.backend, rounds=args.rounds, elite_top=args.elite_top,
        refine_frac=args.refine_frac, algo=args.algo,
        surrogate_factor=args.surrogate_factor, rank=args.rank,
        report_importance=args.feature_importance, device=args.device)


if __name__ == "__main__":
    main()
