#!/usr/bin/env python3
r"""hri_lns_gbdt.py -- GBDT-guided MO-LNS.

The disclosed winning Large-Neighborhood Search (S. Limmer, p.c., cited with
permission), run under OUR exact capped-20 objective, with the ONE random decision
that steers it -- the destroy SEED vertex -- replaced by a learned choice.

Plain hri_lns picks the destroy seed uniformly at random (hri_lns.py:32).  Here a
GBDT scores candidate seed vertices by their predicted marginal capped-20 HV
contribution, and the search spends its expensive repair-evaluations on the
top-ranked candidates.  The model trains ONLINE on the arm's own
(seed-features -> realised capped ΔHV) history -- no external data.  This makes
gradient boosting the *operator* of the winning method itself, aimed squarely at the
scored objective, and in the coordinated multi-vertex (destroy-region) regime where
the single-vertex move-value model (tools/gbdt_moves.py) was structurally blind.

Fair ablation, built in:  --no-gbdt keeps EVERYTHING identical (same candidate
sampling, same per-iteration evaluation budget, same operators, same exact
acceptance) but selects the evaluated candidates at RANDOM instead of by the model.
So `guided` vs `--no-gbdt` isolates exactly the GBDT contribution INSIDE the winner's
own LNS -- the controlled result the thesis needs.

ADDITIVE + SAFE:  writes submissions/<problem>/hri_lns_gbdt.json (its own stem) only
on a verified capped-20 improvement; never touches other arms' files or scripts.

    python3 tools/hri_lns_gbdt.py --problem medium-graph --iters 500000 --seed 1
    python3 tools/hri_lns_gbdt.py --problem medium-graph --iters 500000 --seed 1 --no-gbdt
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import load_graph, build_adj_bitsets, graph_path, LEADERBOARD_TARGETS, MAX_TW
from tools.fastwalk import IncEvalC
from tools.archive_evolve import order_front, capped_hv, load_seed_orders, twin_classes
from tools.hri_lns import balanced_repair, random_destroy_repair
from algorithms.continuous.cmaes_torso import get_features
from algorithms.continuous.np_gbdt import NpGBDT


def width_profile(perm, ev, n):
    """Running-max width at each position -- band context for a seed's location."""
    d = ev.full(perm); r = 0
    prof = np.empty(n, dtype=np.float32)
    for t in range(n - 1, -1, -1):
        c = int(d[t]); r = c if c > r else r
        prof[t] = r
    return prof


def neighbor_destroy_seeded(perm, adj, rng, cap, v):
    """hri_lns.neighbor_destroy, but with the seed vertex v supplied by the caller."""
    d = {v} | set(adj[v])
    if len(d) > cap:
        d = {v} | set(rng.sample(list(adj[v]), cap - 1))
    return d


def seed_feats(F, v, pos_v, nb_pos, prof, twin_sz, deg, n):
    """Static per-vertex features (incl. spectral, from get_features) + dynamic
    context of the seed in the CURRENT ordering (position, neighbour spread, band)."""
    if nb_pos:
        m = sum(nb_pos) / len(nb_pos)
        sd = (sum((x - m) ** 2 for x in nb_pos) / len(nb_pos)) ** 0.5
        before = sum(1 for x in nb_pos if x < pos_v) / max(1, deg)
    else:
        m = float(pos_v); sd = 0.0; before = 0.0
    dyn = np.array([pos_v / n, m / n, sd / n, before,
                    float(prof[pos_v]) / MAX_TW, twin_sz / n], dtype=np.float64)
    return np.concatenate([F[v], dyn])


def run(problem, here, iters, seed, use_gbdt, pool_cap, destroy_cap,
        small_k, stall, ncand, neval, eps, retrain_every):
    n, adj_sets = load_graph(graph_path(here, problem))
    adj = [sorted(s) for s in adj_sets]
    ab = build_adj_bitsets(n, adj_sets)
    ev = IncEvalC(ab, n)
    target = LEADERBOARD_TARGETS.get(problem)
    rng = random.Random(seed)

    F = np.asarray(get_features(here, problem, n, adj_sets, 32)[0], dtype=np.float64)
    twin_sz = np.zeros(n)
    for c in twin_classes(n, adj_sets):
        for v in c:
            twin_sz[v] = len(c)

    members = load_seed_orders(here, problem, n, ev, pool_cap)
    cur, top = capped_hv(members, n)

    profs = {}
    def prof_of(perm):
        key = tuple(perm)
        p = profs.get(key)
        if p is None:
            if len(profs) > 256:
                profs.clear()
            p = width_profile(perm, ev, n); profs[key] = p
        return p

    tag = "GBDT-guided" if use_gbdt else "RANDOM (control / --no-gbdt)"
    print(f"=== HRI-LNS+GBDT {problem} | {tag} | seeded {len(members)} orders | "
          f"capped-20 {cur:,.0f}{f'  gap {cur-target:+,.0f}' if target else ''} | "
          f"cand={ncand} eval={neval} eps={eps} ===", flush=True)

    DX, DY = deque(maxlen=40000), deque(maxlen=40000)
    model = None
    accepts = 0; since = 0; mode = "B"; t0 = time.time()

    for it in range(iters):
        src = list(members[rng.randrange(len(members))][0])
        best_local = None                      # (h, cand, pts, top2)

        if mode == "B":
            prof = prof_of(src)
            pos = {u: i for i, u in enumerate(src)}
            cand_idx = rng.sample(range(n), min(ncand, n))
            cand_v = [src[i] for i in cand_idx]
            feats = np.array([
                seed_feats(F, v, pos[v], [pos[x] for x in adj[v]],
                           prof, twin_sz[v], len(adj[v]), n)
                for v in cand_v])
            if use_gbdt and model is not None and rng.random() > eps:
                order = list(np.argsort(-model.predict(feats)))
            else:
                order = list(range(len(cand_v))); rng.shuffle(order)
            for oi in order[:neval]:
                v = cand_v[int(oi)]
                d = neighbor_destroy_seeded(src, adj, rng, destroy_cap, v)
                cnd = balanced_repair(src, d, adj, rng)
                pts = order_front(cnd, ev, n)
                h, top2 = capped_hv(members + [(cnd, pts)], n)
                DX.append(feats[int(oi)]); DY.append(cur - h)   # >=0 marginal contrib
                if best_local is None or h < best_local[0]:
                    best_local = (h, cnd, pts, top2)
        else:
            cnd = random_destroy_repair(src, rng, small_k)
            pts = order_front(cnd, ev, n)
            h, top2 = capped_hv(members + [(cnd, pts)], n)
            best_local = (h, cnd, pts, top2)

        h, cnd, pts, top2 = best_local
        since += 1
        if h < cur - 0.5:                       # strict capped-20 improvement
            cur = h; accepts += 1; since = 0
            keep = set(tuple(p) for (_, _, p) in top2)
            trial = members + [(cnd, pts)]
            members = [(p, x) for (p, x) in trial if tuple(p) in keep]
            if (cnd, pts) not in members:
                members.append((cnd, pts))
            print(f"  it {it} [{mode}]: *** capped-20 {cur:,.0f}"
                  f"{f'  gap {cur-target:+,.0f}' if target else ''} "
                  f"(accept #{accepts}) ***", flush=True)
            dvs = [list(p) + [int(t)] for (_, t, p) in top2]
            out = os.path.join(here, "submissions", problem, "hri_lns_gbdt.json")
            json.dump({"challenge": "spoc-3-torso-decompositions",
                       "problem": problem, "decisionVector": dvs}, open(out, "w"))

        if since >= stall:
            mode = "A" if mode == "B" else "B"; since = 0
            print(f"  [stall -> operator set {mode}]", flush=True)

        if use_gbdt and len(DY) >= 400 and it % retrain_every == 0 and it:
            X = np.asarray(DX); y = np.asarray(DY)
            model = NpGBDT(n_estimators=120, learning_rate=0.1, max_depth=4,
                           subsample=0.8, seed=seed)
            model.fit(X, y)

        if it % 500 == 0 and it:
            pos_seen = int((np.asarray(DY) > 0).sum()) if DY else 0
            gh = "guided" if (use_gbdt and model is not None) else "random"
            print(f"  [it {it}/{iters} [{mode}] {gh} capped-20 {cur:,.0f}  "
                  f"{accepts} accepts  {pos_seen}/{len(DY)} improving seen  "
                  f"{time.time()-t0:.0f}s]", flush=True)

    print(f"\nfinal capped-20 {cur:,.0f}"
          f"{f'  gap {cur-target:+,.0f}' if target else ''} | {accepts} LNS accepts "
          f"({'GBDT-guided' if use_gbdt else 'random control'})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="medium-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--iters", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gbdt", dest="use_gbdt", action="store_false",
                    help="ablation control: random candidate selection, same budget")
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--destroy-cap", type=int, default=220,
                    help="max neighbor-destroy size (large avg degree ~209)")
    ap.add_argument("--small-k", type=int, default=6)
    ap.add_argument("--stall", type=int, default=3000)
    ap.add_argument("--cand", type=int, default=48,
                    help="candidate destroy-seeds scored per iteration")
    ap.add_argument("--eval", dest="neval", type=int, default=3,
                    help="top-ranked candidates actually repaired+evaluated per iteration")
    ap.add_argument("--eps", type=float, default=0.15,
                    help="exploration: fraction of iterations that pick seeds at random")
    ap.add_argument("--retrain-every", type=int, default=1500)
    a = ap.parse_args()
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        a.iters, a.seed, a.use_gbdt, a.pool, a.destroy_cap, a.small_k, a.stall,
        a.cand, a.neval, a.eps, a.retrain_every)


if __name__ == "__main__":
    main()
