#!/usr/bin/env python3
"""
pgmo_gaps.py -- supercharge the GBDT-augmented decode (GAPS) with a pagmo/pygmo
ARCHIPELAGO of state-of-the-art optimisers.

Why this is the right pairing:
  * Our novelty is the GBDT-augmented decode: an ordering is produced by
    argsort([Phi(F), g_GBDT(F)] . x), where g_GBDT is a learned column.  The
    GBDT is the load-bearing feature -- every policy the archipelago evolves
    decodes THROUGH it.
  * The only thing that ever moved the hard instances was DIVERSE search pooled
    together.  pagmo's archipelago (many islands, different algorithms, elite
    MIGRATION across a ring topology) is purpose-built to manufacture that
    diversity -- far more than a single mutation+COSYNE loop.
  * Closed loop: evolve -> pool all island champions into the front -> retrain
    the GBDT on the new pooled elites (DAgger) -> rebuild features -> evolve
    again.  GBDT sharpens every round; pagmo searches; GBFC pooling banks it.

Run (after `conda install -c conda-forge pygmo`  OR  `pip install pygmo`):
    python3 tools/pgmo_gaps.py --problem small-graph --rounds 8 --gen 300 \
        --islands 8 --pop 64 --device gpu        # gpu uses the batch evaluator

The decode/fitness work WITHOUT pygmo (importable, testable); only the
archipelago driver needs pygmo.
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from algorithms.continuous.cmaes_torso import get_features


# ---------------------------------------------------------------------------
# features: degree profile + eigenvectors, expanded to the cuda-torso poly basis
# ---------------------------------------------------------------------------
def poly_features(raw):
    n, r = raw.shape
    cols = [raw, raw ** 2]
    idx = [(i, j) for i in range(r) for j in range(i + 1, r)]
    inter = np.empty((n, len(idx)), dtype=raw.dtype)
    for k, (i, j) in enumerate(idx):
        inter[:, k] = raw[:, i] * raw[:, j]
    F = np.hstack(cols + [inter])
    mu = F.mean(0); sd = F.std(0); sd[sd == 0] = 1.0
    return ((F - mu) / sd).astype(np.float64)


def front_deg(perm, ab, n):
    sm = [0] * n; cur = 0
    for i in range(n - 1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    tmp = list(ab); deg = [0] * n
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
    return deg


def staircase_pts(perm, ab, n):
    deg = front_deg(perm, ab, n); run = 0; pts = []
    for t in range(n - 1, -1, -1):
        if deg[t] > run: run = deg[t]
        if run <= MAX_TW: pts.append((int(run), t))
    return pts


def add_staircase(perm, ab, n, arc):
    for w, t in staircase_pts(perm, ab, n):
        arc.try_add(w, t, list(perm))


# ---------------------------------------------------------------------------
# the GBDT-augmented decode + single-ordering HV objective
# ---------------------------------------------------------------------------
class Decoder:
    def __init__(self, Faug, ab, n):
        self.nodesT = Faug.T.copy()          # (E, n)
        self.ab = ab; self.n = n; self.E = Faug.shape[1]

    def perm(self, x):
        return [int(i) for i in np.argsort(np.asarray(x, float) @ self.nodesT)]

    def neg_hv(self, x):
        return -hypervolume_2d(staircase_pts(self.perm(x), self.ab, self.n), self.n)


# ---------------------------------------------------------------------------
# pygmo user-defined problem (UDP) over the policy x
# ---------------------------------------------------------------------------
def make_udp(decoder, bound=4.0):
    class GapsUDP:
        def fitness(self, x):
            return [decoder.neg_hv(x)]
        def batch_fitness(self, xs):
            E = decoder.E; m = len(xs) // E
            X = np.asarray(xs, float).reshape(m, E)
            return [decoder.neg_hv(X[i]) for i in range(m)]
        def get_bounds(self):
            return ([-bound] * decoder.E, [bound] * decoder.E)
        def get_name(self):
            return "GAPS-decode"
    return GapsUDP()


# ---------------------------------------------------------------------------
# GBDT column: imitation of the pooled elites' elimination positions (DAgger)
# ---------------------------------------------------------------------------
def train_gbdt_col(arc, raw, ab, n, seed, backend="auto"):
    from algorithms.continuous.gbdt_torso import make_gbdt
    rows_X, rows_y = [], []
    for w, t, perm in arc.entries():            # learn from the best per (w,t)
        order = perm[t:]
        if len(order) < 8: continue
        pos = {v: i / max(1, len(order) - 1) for i, v in enumerate(order)}
        for v in order:
            rows_X.append(raw[v]); rows_y.append(pos[v])
    if len(rows_X) < 50:
        return np.zeros(n)
    name, gb = make_gbdt(backend, seed)
    gb.fit(np.array(rows_X), np.array(rows_y))
    col = gb.predict(raw).astype(np.float64)
    col = (col - col.mean()) / (col.std() + 1e-9)
    return col


# ---------------------------------------------------------------------------
def run(problem, here, rounds, gen, islands, pop, k_eig, device, backend, seed, procs=1):
    try:
        import pygmo as pg
    except Exception as e:
        print(f"pygmo not importable ({e}).\n"
              "Install it:  conda install -c conda-forge pygmo   (or: pip install pygmo)")
        return
    if procs > 1:
        import multiprocessing as _mp
        try: _mp.set_start_method("spawn", force=True)
        except RuntimeError: pass
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem)
    raw, _ = get_features(here, problem, n, adj, k_eig)      # (n, 37) degree profile + eigvecs
    Phi = poly_features(raw)
    print(f"=== pgmo-GAPS -- {problem} (n={n}, poly-features={Phi.shape[1]}) ===")

    # seed pooled archive from existing submissions
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            p = json.load(open(fp)); e = p[0] if isinstance(p, list) else p
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    add_staircase([int(x) for x in dv[:-1]], ab, n, arc)
        except Exception:
            continue
    base = -hypervolume_2d(arc.points(), n)
    print(f"seed pooled HV = {base:,.0f}" + (f"  gap {base - target:+,.0f}" if target else ""))

    # the island algorithm portfolio (diverse global optimisers)
    def algos():
        return [pg.sade(gen=gen), pg.de1220(gen=gen), pg.cmaes(gen=gen, force_bounds=True),
                pg.xnes(gen=gen, force_bounds=True), pg.pso(gen=gen), pg.sade(gen=gen)]

    gcol = train_gbdt_col(arc, raw, ab, n, seed, backend)
    for r in range(rounds):
        Faug = np.hstack([Phi, gcol[:, None]]).astype(np.float64)
        dec = Decoder(Faug, ab, n)
        prob = pg.problem(make_udp(dec))
        # archipelago of diverse islands on a ring topology -> elite migration
        archi = pg.archipelago(t=pg.ring())
        alist = algos()
        # mp_island runs each island in its OWN process -> true multicore (Python
        # UDPs hold the GIL, so thread islands wouldn't parallelise).
        udi = pg.mp_island() if procs > 1 else pg.thread_island()
        for i in range(islands):
            algo = pg.algorithm(alist[i % len(alist)])
            archi.push_back(algo=algo, prob=prob, size=pop, udi=udi)
        archi.evolve(); archi.wait_check()
        # pool ALL island champions (+ their final populations would be richer;
        # champions are the cheap, reliable signal) into the front
        for x in archi.get_champions_x():
            add_staircase(dec.perm(x), ab, n, arc)
        cur = -hypervolume_2d(arc.points(), n)
        msg = f"round {r+1}/{rounds}: pooled HV {cur:,.0f}"
        if target: msg += f"  gap {cur - target:+,.0f}"
        print(msg, flush=True)
        # DAgger: retrain GBDT on the improved pool, rebuild the decode column
        gcol = train_gbdt_col(arc, raw, ab, n, seed + r + 1, backend)
        # bank
        top = arc.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = os.path.join(here, "submissions", problem, "pgmo_gaps.json")
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                   "decisionVector": dvs}, open(out, "w"))

    final = -hypervolume_2d(arc.points(), n)
    print(f"\nFinal pooled HV = {final:,.0f}" + (f"  gap {final - target:+,.0f}" if target else ""))
    print(f"saved -> {os.path.join(here, 'submissions', problem, 'pgmo_gaps.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--gen", type=int, default=300)
    ap.add_argument("--islands", type=int, default=8)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--procs", type=int, default=1, help=">1 -> mp_island, true multicore")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        a.rounds, a.gen, a.islands, a.pop, a.k_eig, a.device, a.backend, a.seed, a.procs)


if __name__ == "__main__":
    main()
