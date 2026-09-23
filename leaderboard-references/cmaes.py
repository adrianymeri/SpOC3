#!/usr/bin/env python3
"""
cmaes.py -- the fast-cma-es approach: CMA-ES over a continuous encoding.

The fast-cma-es entry treats the problem as continuous optimisation. A
candidate is a real-valued weight vector, the ordering is read off by sorting
a score per vertex, and CMA-ES searches the weight space:

    score = features @ x        for a weight vector x
    perm  = argsort(score)

That is the same decode Spacekangaroos use. The difference is the optimiser:
CMA-ES adapts a covariance model of where good weight vectors live, instead
of mutating elites.

This is a separable (diagonal-covariance) CMA-ES, ported from the main
project's implementation. Separable means the covariance is a diagonal rather
than a full matrix -- O(dim) instead of O(dim^2) -- which is what makes it
usable at these dimensions.

Needs numpy. Scoring goes through ../esa_eval.py like everything else here.

    python3 cmaes.py --instance ../data/small-graph.gr --seconds 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torso import Graph, Solution, Front

try:
    import numpy as np
except ImportError:
    np = None

from neuroevo_cpu import build_features


class SepCMAES:
    """Separable CMA-ES with a diagonal covariance (Ros & Hansen 2008)."""

    def __init__(self, dim, sigma0=0.5, seed=42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.mean = self.rng.normal(0, 0.1, dim)
        self.sigma = sigma0
        self.lam = 4 + int(3 * math.log(dim))
        self.mu = self.lam // 2

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)

        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.ds = (1 + 2 * max(0, math.sqrt((self.mueff - 1) / (dim + 1)) - 1)
                   + self.cs)
        self.cc = (1 + 1 / dim + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff) * (dim + 2) / 3
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff)
                       / ((dim + 2) ** 2 + self.mueff) * (dim + 2) / 3)

        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.C = np.ones(dim)
        self.chiN = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.dim))
        self.y = self.z * np.sqrt(self.C)[None, :]
        return self.mean[None, :] + self.sigma * self.y

    def tell(self, fitness):
        idx = np.argsort(fitness)
        ysel, zsel = self.y[idx[:self.mu]], self.z[idx[:self.mu]]
        yw = (self.w[:, None] * ysel).sum(0)
        zw = (self.w[:, None] * zsel).sum(0)

        self.mean = self.mean + self.sigma * yw
        self.ps = ((1 - self.cs) * self.ps
                   + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw)
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** 2)
                / self.chiN) < (1.4 + 2 / (self.dim + 1))
        self.pc = ((1 - self.cc) * self.pc
                   + (hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff)) * yw)

        cmu_term = (self.w[:, None] * (ysel ** 2)).sum(0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2
                               + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * cmu_term)
        self.C = np.maximum(self.C, 1e-20)
        self.sigma *= math.exp((self.cs / self.ds)
                               * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e6))


def solve(graph, seconds, seed=1, eigenvectors=8, sigma0=0.5, verbose=False,
          engine="auto"):
    """Run CMA-ES and return (front, generations, evaluations).

    engine="auto" uses the real fast-cma-es package if it is installed and
    falls back to the separable CMA-ES below otherwise. Installing it makes
    this column the team's actual optimiser rather than a reimplementation:

        pip install fcmaes
    """
    if np is None:
        return None, 0, 0

    n = graph.n
    features, _ = build_features(graph, eigenvectors)   # (dim, n)
    features = features.T                               # (n, dim) for the decode
    dim = features.shape[1]
    front = Front(n)
    counters = {"evaluations": 0, "generations": 0}

    def fitness(x):
        """Evaluate one weight vector; the front is filled as a side effect."""
        perm = [int(v) for v in np.argsort(features @ x, kind="stable")]
        solution = Solution(graph, perm)
        front.add_solution(solution)
        counters["evaluations"] += 1
        stairs = solution.staircase()
        return float(stairs[-1][1] if stairs else n)

    if engine in ("auto", "fcmaes"):
        try:
            from fcmaes import cmaes as fcmaes_cmaes
            from fcmaes.optimizer import Bounds
        except Exception:
            if engine == "fcmaes":
                raise SystemExit("fcmaes not installed: pip install fcmaes")
        else:
            if verbose:
                print("  engine: fast-cma-es (the team's actual optimiser)")
            start = time.time()
            # CMA-ES stops itself once sigma collapses -- on these problems
            # that happens after a minute or two, long before the budget is
            # spent. Left alone it would quietly use a fraction of the time
            # every other solver gets. So restart it from a fresh point until
            # the budget really is gone, which is also how fast-cma-es is
            # meant to be used: retry is the headline feature of the toolkit.
            rng = np.random.default_rng(seed)
            while time.time() - start < seconds:
                x0 = (np.zeros(dim) if counters["generations"] == 0
                      else rng.uniform(-1.0, 1.0, dim))
                counters["generations"] += 1
                fcmaes_cmaes.minimize(
                    fitness, Bounds([-3.0] * dim, [3.0] * dim),
                    x0=x0, input_sigma=sigma0,
                    max_evaluations=10 ** 9, max_iterations=10 ** 9,
                    popsize=4 + int(3 * math.log(dim)), workers=1,
                    is_terminate=lambda runid, it, best: (
                        time.time() - start) > seconds,
                    rg=np.random.default_rng(int(rng.integers(1 << 30))))
                if verbose:
                    print(f"  restart {counters['generations']}: "
                          f"{front.score():,}  "
                          f"({time.time() - start:.0f}s of {seconds:.0f}s)")
            return front, counters["generations"], counters["evaluations"]

    if verbose:
        print("  engine: built-in separable CMA-ES "
              "(pip install fcmaes for the real thing)")
    optimiser = SepCMAES(dim, sigma0, seed)
    deadline = time.time() + seconds
    while time.time() < deadline:
        counters["generations"] += 1
        population = optimiser.ask()
        values = []
        for x in population:
            if time.time() >= deadline:
                values.extend([float(n)] * (len(population) - len(values)))
                break
            values.append(fitness(x))
        optimiser.tell(np.asarray(values))
        if verbose and counters["generations"] % 10 == 0:
            print(f"  gen {counters['generations']}: {front.score():,}")

    return front, counters["generations"], counters["evaluations"]


def main():
    ap = argparse.ArgumentParser(description="fast-cma-es style solver.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eigenvectors", type=int, default=8)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if np is None:
        raise SystemExit("this solver needs numpy")

    graph = Graph.load(a.instance)
    print(f"=== CMA-ES on {os.path.basename(a.instance)} ===")
    print(graph.describe())
    t0 = time.time()
    front, generations, evaluations = solve(graph, a.seconds, a.seed,
                                            a.eigenvectors, a.sigma0,
                                            verbose=True)
    print(f"\n{generations} generations, {evaluations} evaluations, "
          f"{time.time() - t0:.1f}s")
    print(f"SCORE: {front.score():,}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"instance": os.path.basename(a.instance),
                       "solver": "cmaes", "n": graph.n,
                       "score": front.score(),
                       "decisionVector": front.decision_vectors()}, f)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
