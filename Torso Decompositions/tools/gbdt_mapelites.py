#!/usr/bin/env python3
r"""
gbdt_mapelites.py -- the completed puzzle: MAP-Elites over the threshold niches,
driven by fcmaes's strong CMA-ES (ACMA) emitter, on the GBDT-augmented decode.

Why this is the missing engine.  cuda-torso (the leaderboard recipe) keeps a best
policy *per threshold* and perturbs it with random mutation + COSYNE -- a
primitive quality-diversity loop, and our long run of it plateaus ~1,550 HV below
the leader.  But the leader's score IS a policy (argsort) solution, so the policy
optimum lies *above* our constructed front: the wall was our optimiser's, not the
representation's.  fcmaes supplies the real illumination engine -- CVT-MAP-Elites
with a CMA-ES emitter.  Here we keep the threshold niches (the per-threshold
specialists), but fill them with full **ACMA** drill-down instead of random
mutation, on the **GBDT-augmented decode** (argsort([Φ(F), g_GBDT(F)]·x)), with
the GBDT retrained on the illuminated archive each round (DAgger) and used to
*bias which niche to drill* (the worst-served band vs the treewidth bound).

Pieces, all proven: cuda-torso's decode + our exact C-kernel evaluator (fastwalk)
+ fcmaes's ACMA emitter + our GBDT.  Falls back to a built-in separable CMA-ES if
fcmaes is absent, so it runs anywhere.

    pip install fcmaes
    python3 tools/gbdt_mapelites.py --problem small-graph --budget 36000 \
        --emit-evals 4000 --rounds 200

Checkpoint-safe: writes submissions/<problem>/mapelites.json every round.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from algorithms.continuous.cmaes_torso import get_features


def poly_features(raw):
    n, r = raw.shape
    idx = [(i, j) for i in range(r) for j in range(i + 1, r)]
    inter = np.empty((n, len(idx)), dtype=raw.dtype)
    for k, (i, j) in enumerate(idx):
        inter[:, k] = raw[:, i] * raw[:, j]
    F = np.hstack([raw, raw ** 2, inter])
    mu = F.mean(0); sd = F.std(0); sd[sd == 0] = 1.0
    return ((F - mu) / sd).astype(np.float64)


def make_eval(ab, n):
    try:
        from tools.fastwalk import IncEvalC
        ev = IncEvalC(ab, n)
        return (lambda perm: ev.full(perm)), "C-kernel"
    except Exception:
        def _py(perm):
            sm = [0] * n; cur = 0
            for i in range(n - 1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
            tmp = list(ab); deg = [0] * n
            for i in range(n):
                s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
                while x:
                    b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
            return np.array(deg)
        return _py, "python"


def train_gbdt_col(arch_perm, raw, n, seed, backend="auto"):
    """GBDT membership column: learn, from the niche-best orderings, each vertex's
    typical elimination position -> a learned decode feature (DAgger)."""
    from algorithms.continuous.gbdt_torso import make_gbdt
    X, y = [], []
    seen = set()
    for t, perm in arch_perm.items():
        key = id(perm)
        if key in seen: continue
        seen.add(key)
        d = max(1, n - 1)
        for i, v in enumerate(perm):
            X.append(raw[v]); y.append(i / d)
    if len(X) < 200:
        return np.zeros(n), "none"
    name, gb = make_gbdt(backend, seed)
    gb.fit(np.array(X), np.array(y))
    col = gb.predict(raw).astype(np.float64)
    return (col - col.mean()) / (col.std() + 1e-9), name


# --- a minimal separable CMA-ES fallback emitter (if fcmaes is unavailable) ---
def _sepcma_emit(fun, x0, E, evals, sigma, seed):
    import math
    rng = np.random.default_rng(seed)
    mean = np.array(x0, float); lam = 4 + int(3 * math.log(E)); mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    C = np.ones(E); used = 0; best = (1e18, mean)
    while used < evals:
        Z = rng.standard_normal((lam, E)); Y = Z * np.sqrt(C)[None]
        P = mean[None] + sigma * Y
        f = np.array([fun(P[i]) for i in range(lam)]); used += lam
        idx = np.argsort(f)
        if f[idx[0]] < best[0]: best = (float(f[idx[0]]), P[idx[0]].copy())
        mean = mean + (w[:, None] * (P[idx[:mu]] - mean[None])).sum(0)
        C = 0.8 * C + 0.2 * (w[:, None] * (Y[idx[:mu]] ** 2)).sum(0)
        C = np.clip(C, 1e-6, 1e6)
    return best


def run(problem, here, budget_s, emit_evals, rounds, k_eig, backend, seed, bound=4.0):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem)
    evfn, evkind = make_eval(ab, n)
    raw, _ = get_features(here, problem, n, adj, k_eig)
    Phi = poly_features(raw)
    print(f"=== gbdt-MAP-Elites -- {problem} (n={n}, feat={Phi.shape[1]}, "
          f"eval={evkind}) ===", flush=True)

    # niche archive: best ordering + width + warm policy per threshold
    arch_w = np.full(n, 10 ** 9, dtype=np.int64)
    arch_perm = {}
    arch_pol = {}

    def staircase_update(perm, x):
        deg = evfn(perm); run = 0; improved = False
        for t in range(n - 1, -1, -1):
            d = int(deg[t]); run = d if d > run else run
            if run <= MAX_TW and run < arch_w[t]:
                arch_w[t] = run; arch_perm[t] = list(perm)
                if x is not None: arch_pol[t] = np.array(x, float)
                improved = True
        return improved, deg

    # seed the niche floor from the existing pooled front (so we only accept
    # improvements over our gap-6 result)
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            p = json.load(open(fp)); e = p[0] if isinstance(p, list) else p
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    staircase_update([int(x) for x in dv[:-1]], None)
        except Exception:
            continue

    def front_hv():
        arc = ParetoArchive()
        for t in range(n):
            if arch_w[t] < 10 ** 9: arc.try_add(int(arch_w[t]), t, None)
        return -hypervolume_2d(arc.points(), n)
    print(f"seed front HV = {front_hv():,.0f}" +
          (f"  gap {front_hv()-target:+,.0f}" if target else ""), flush=True)

    # GBDT column + emitter
    gcol, gbname = train_gbdt_col(arch_perm, raw, n, seed, backend)
    print(f"gbdt backend: {gbname}", flush=True)

    try:
        from fcmaes import cmaescpp
        from fcmaes.optimizer import Bounds
        HAVE_FCMAES = True
    except Exception as e:
        HAVE_FCMAES = False
        print(f"[fcmaes unavailable: {e}; using separable-CMA fallback emitter]", flush=True)

    t0 = time.time(); evals = 0
    for r in range(rounds):
        if time.time() - t0 >= budget_s: break
        nodesT = np.hstack([Phi, gcol[:, None]]).T.copy()       # (E, n) augmented decode
        E = nodesT.shape[0]

        # choose the niche to drill: worst-served band (largest width over a
        # local floor) -- the residual the GBDT is meant to attack
        # drill the BREAKPOINT bands, where +1 torso-size = +1 HV: the grown
        # thresholds t*(w)-1 (one step earlier than each width's current best).
        # Rotating over these aims the emitter where the front can actually move,
        # not at the rigid high-width region.
        valid = np.where(arch_w < 10 ** 9)[0]
        bps = [int(t) for t in valid if t > 0 and arch_w[t] < arch_w[t - 1]]  # width-drop steps
        targets = sorted({max(0, t - 1) for t in bps}) or [int(valid[len(valid) // 2])]
        tt = targets[r % len(targets)]

        def fitness(x):
            nonlocal evals
            perm = [int(i) for i in np.argsort(np.asarray(x) @ nodesT)]
            staircase_update(perm, x)
            evals += 1
            return float(evfn(perm)[tt:].max())                  # width at the target niche

        x0 = arch_pol.get(tt)
        if x0 is None:
            x0 = np.random.default_rng(seed + r).normal(0, 0.3, E)
        before = evals
        if HAVE_FCMAES:
            try:
                cmaescpp.minimize(fitness, bounds=Bounds([-bound]*E, [bound]*E),
                                  x0=np.asarray(x0, float), input_sigma=0.3,
                                  popsize=31, max_evaluations=emit_evals,
                                  workers=1, delayed_update=False, normalize=False)
            except Exception as ex:
                print(f"  [fcmaes emit error: {ex}; falling back]", flush=True)
        if evals == before:                      # emitter ran nothing -> tested CMA-ES
            _sepcma_emit(fitness, x0, E, emit_evals, 0.3, seed + r)

        cur = front_hv()
        msg = f"round {r+1}/{rounds}: niche t={tt} | front HV {cur:,.0f}"
        if target: msg += f"  gap {cur-target:+,.0f}"
        msg += f" | evals {evals} | {time.time()-t0:.0f}s"
        print(msg, flush=True)

        # DAgger: retrain the GBDT column on the freshly illuminated niches
        if (r + 1) % 5 == 0:
            gcol, _ = train_gbdt_col(arch_perm, raw, n, seed + r + 1, backend)

        # bank
        arc = ParetoArchive()
        for t in range(n):
            if arch_w[t] < 10 ** 9 and t in arch_perm:
                arc.try_add(int(arch_w[t]), t, arch_perm[t])
        top = arc.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                   "decisionVector": dvs},
                  open(os.path.join(here, "submissions", problem, "mapelites.json"), "w"))

    print(f"\nFinal front HV = {front_hv():,.0f}" +
          (f"  gap {front_hv()-target:+,.0f}" if target else ""))
    print(f"saved -> submissions/{problem}/mapelites.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--budget", type=float, default=36000.0)
    ap.add_argument("--emit-evals", type=int, default=4000, help="CMA-ES evals per niche drill")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        a.budget, a.emit_evals, a.rounds, a.k_eig, a.backend, a.seed)


if __name__ == "__main__":
    main()
