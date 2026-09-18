#!/usr/bin/env python3
"""
neuroevo_cpu.py -- a CPU reimplementation of the SpOC-3 winning method.

The original (cuda-torso/, by the team that won the challenge) needs an
NVIDIA GPU, a compiled CUDA kernel, PyTorch, and has the three official
graph sizes hardcoded. This runs on a laptop and takes any instance, so the
winner's approach can be compared against hill climbing on the seven extra
instances.

The method, from cuda-torso/run.py
----------------------------------
A candidate is not a permutation. It is a weight vector w over per-vertex
features, and the permutation is read off by sorting:

    score = w . features[v]          for each vertex v
    perm  = argsort(score)

So evolution searches the space of *scoring rules*, not orderings. The
features are fixed up front and describe each vertex's position in the
graph:

  * local degree profile (5): degree, and the min/max/mean/std of the
    degrees of its neighbours
  * Laplacian positional encoding: the first k eigenvectors of the
    normalised Laplacian, giving each vertex a spectral coordinate
  * polynomial expansion: every feature squared, plus every pairwise
    product
  * all of it standardised to mean 0, stdev 1

Evolution is a simple elite scheme with two mutation operators:

  1. evaluate the population, keep the N best ever seen as elites
  2. resample the next population from the elites
  3. Gaussian mutation on a random subset of weights
  4. Cosyne-style permutation: copy individual weights between population
     members, which mixes partial solutions without full crossover

Differences from the original, stated plainly
---------------------------------------------
  * CPU and numpy instead of CUDA and PyTorch, so the population is much
    smaller and generations are far slower. This is not competitive with
    the original on a GPU; it is the same algorithm at a smaller scale.
  * Scoring goes through ../esa_eval.py, the same evaluator hill climbing
    uses, so the numbers are directly comparable.
  * The original picks thresholds with its own hypervolume routine; here
    every candidate contributes its whole staircase to a shared front, the
    same way hill_climbing.py does, and the front is cut to 20 points by
    the exact HSSP.
  * Without numpy the spectral features are skipped and only the degree
    profile is used. That is a weaker feature set and it is reported in
    the output when it happens.

    python3 neuroevo_cpu.py --instance ../data/small-graph.gr --seconds 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esa_eval import build_adj_bitsets, hypervolume_2d, ParetoArchive, MAX_TW
from torso import Graph, Solution, Front

try:
    import numpy as np
except ImportError:
    np = None


def build_features(graph, eigenvectors=8):
    """Per-vertex features: degree profile, spectral coords, polynomials."""
    n = graph.n
    deg = [float(len(graph.adj[v])) for v in range(n)]

    raw = []
    for v in range(n):
        nb = [deg[u] for u in graph.adj[v]] or [0.0]
        mean = sum(nb) / len(nb)
        var = sum((x - mean) ** 2 for x in nb) / len(nb)
        raw.append([deg[v], min(nb), max(nb), mean, math.sqrt(var)])

    spectral = False
    if np is not None and eigenvectors > 0:
        A = np.zeros((n, n))
        for v in range(n):
            for u in graph.adj[v]:
                A[v, u] = 1.0
        d = A.sum(1)
        d[d == 0] = 1.0
        dm = 1.0 / np.sqrt(d)
        lap = np.eye(n) - (A * dm).T * dm
        vals, vecs = np.linalg.eigh(lap)
        pe = np.real(vecs[:, vals.argsort()][:, 1:eigenvectors + 1])
        for v in range(n):
            raw[v].extend(pe[v].tolist())
        spectral = True

    width = len(raw[0])
    feats = []
    for v in range(n):
        row = list(raw[v])
        row.extend(x * x for x in raw[v])
        row.extend(raw[v][i] * raw[v][j] for i, j in combinations(range(width), 2))
        feats.append(row)

    # standardise each column
    cols = len(feats[0])
    for c in range(cols):
        col = [feats[v][c] for v in range(n)]
        mean = sum(col) / n
        var = sum((x - mean) ** 2 for x in col) / n
        sd = math.sqrt(var) or 1.0
        for v in range(n):
            feats[v][c] = (feats[v][c] - mean) / sd

    if np is not None:
        return np.asarray(feats, dtype=np.float64).T, spectral
    return feats, spectral


def order_from_weights(weights, features, n):
    """perm = argsort(w . features)."""
    if np is not None:
        scores = weights @ features
        # plain ints: the bitset evaluator needs .bit_length()
        return [int(v) for v in np.argsort(scores)]
    scores = [sum(w * f for w, f in zip(weights, features[v])) for v in range(n)]
    return sorted(range(n), key=lambda v: scores[v])


def run(instance, seconds, pop_size, elite_size, mut_std, mut_prob,
        cosyne, eigenvectors, seed, out_path):
    rng = random.Random(seed)
    graph = Graph.load(instance)
    n = graph.n

    print(f"=== neuroevo (CPU port of cuda-torso) on "
          f"{os.path.basename(instance)} ===")
    print(graph.describe())

    t0 = time.time()
    features, spectral = build_features(graph, eigenvectors)
    dim = features.shape[0] if np is not None else len(features[0])
    kind = ("degree profile + spectral" if spectral else
            "degree profile only -- install numpy for the full feature set")
    print(f"features: {dim} per vertex ({kind})  [{time.time() - t0:.1f}s]")
    print(f"population {pop_size}, elites {elite_size}, seed {seed}")
    print()

    front = Front(n)

    def make_weights():
        if np is not None:
            return np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, dim)
        return [rng.gauss(0, 1) for _ in range(dim)]

    population = [make_weights() for _ in range(pop_size)]
    elites = []                      # (cost, weights)
    evaluations = 0
    generations = 0
    deadline = time.time() + seconds

    while time.time() < deadline:
        generations += 1
        scored = []
        for w in population:
            if time.time() >= deadline:
                break
            perm = order_from_weights(w, features, n)
            solution = Solution(graph, perm)
            front.add_solution(solution)
            evaluations += 1
            stairs = solution.staircase()
            # fitness: the widest point's t, i.e. how much of the graph the
            # ordering keeps at full width. Lower is better.
            cost = stairs[-1][1] if stairs else n
            scored.append((cost, w))

        elites = sorted(elites + scored, key=lambda e: e[0])[:elite_size]
        if not elites:
            break

        population = []
        for _ in range(pop_size):
            base = elites[rng.randrange(len(elites))][1]
            if np is not None:
                child = base.copy()
                mask = np.random.default_rng(rng.randrange(1 << 30)).random(dim) < mut_prob
                child[mask] += np.random.default_rng(
                    rng.randrange(1 << 30)).normal(0, mut_std, int(mask.sum()))
            else:
                child = [x + (rng.gauss(0, mut_std) if rng.random() < mut_prob else 0.0)
                         for x in base]
            population.append(child)

        # Cosyne permutation: swap individual weights between members
        swaps = int(pop_size * dim * cosyne)
        for _ in range(swaps):
            a, b = rng.randrange(pop_size), rng.randrange(pop_size)
            c = rng.randrange(dim)
            population[a][c] = population[b][c]

        if generations % 5 == 0:
            print(f"  gen {generations:4d}  evaluations {evaluations:6d}  "
                  f"best front {front.score():,}  {time.time() - t0:.0f}s")

    points = front.best_k(20)
    print()
    print(f"generations {generations}, evaluations {evaluations}, "
          f"{time.time() - t0:.1f}s")
    print(f"SCORE: {front.score():,}")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"instance": os.path.basename(instance),
                       "solver": "neuroevo_cpu",
                       "n": n, "score": front.score(),
                       "decisionVector": front.decision_vectors()}, f)
        print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="CPU port of the cuda-torso neuro-evolution method.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--elites", type=int, default=8)
    ap.add_argument("--mutation-stdev", type=float, default=0.3)
    ap.add_argument("--mutation-proba", type=float, default=0.5)
    ap.add_argument("--cosyne-proba", type=float, default=0.05)
    ap.add_argument("--eigenvectors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    run(a.instance, a.seconds, a.pop, a.elites, a.mutation_stdev,
        a.mutation_proba, a.cosyne_proba, a.eigenvectors, a.seed, a.out)


if __name__ == "__main__":
    main()
