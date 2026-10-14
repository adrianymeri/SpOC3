#!/usr/bin/env python3
"""
esa_crosscheck.py -- run ESA's own published evaluation code next to esa_eval.py.

ESA publishes the evaluation code of every SpOC-3 torso problem on the
problem's page ("Evaluation code", optimise.esa.int/challenge/
spoc-3-torso-decompositions/p/<problem>). This script loads that file as it
is published, with a single substitution: ESA's code uses PyTorch only as a
boolean matrix (tensor, zeros, where, combinations, clone), and a numpy
stand-in supplies those five calls. ESA's elimination walk, its 500-width cap
and its hypervolume (pygmo) run as written.

    python3 harness/esa_crosscheck.py --esa-file <the downloaded .py> [--instances ...]

Three checks, on every instance asked for (default: all ten):

  1  fitness      graph_torso_udp.fitness(x) == esa_eval.evaluate(perm, t) on
                  min-degree orderings, perturbed ones, random orderings that
                  break the cap, and a permutation with a repeated vertex, each
                  at several thresholds t;
  2  hypervolume  ESA's combine_scores (pygmo) == -esa_eval.hypervolume_2d on
                  random sets of up to 20 points, 501-width points included;
  3  end to end   the min-degree x8 submission (min_degree_x8.py, seeds 1-8)
                  scored by ESA's code == harness score_vectors ==
                  benchmark-min_degree-x8.csv.

Needs numpy, pygmo, loguru and matplotlib (ESA's own imports), not torch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import random
import sys
import tempfile
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from esa_eval import build_adj_bitsets, evaluate, hypervolume_2d   # noqa: E402
from torso import Graph, Solution, Front                           # noqa: E402
import hill_climbing as hc                                          # noqa: E402
from bench_one import INSTANCES, score_vectors                      # noqa: E402

# ESA's published files, read 2026-10-14. The instance files are ESA's
# easy/medium/hard.gr; data/{small,medium,large}-graph.gr must be the same bytes.
ESA_PY_SHA256 = {
    "5d64ac5d67c003d96675ce8e85c02cb9d6aefa85ba3c51f55e4a16530b9163e6": "small-graph (…-1711380312223.py)",
    "cccfb1c19c794ce6d786b25bbee9a9830fc34ee8c4dcdffbeeec5119d6aa9b56": "medium-graph (…-1711380328535.py)",
    "5839d2e5acce5bd0c719946375acc16fb05d4ce67fcbc7f81d9a4a26dab3a782": "large-graph (…-1711380341869.py)",
}
ESA_GR = {  # ESA name -> (file in data/, sha256 of api.optimize.esa.int/data/spoc3/torso/<name>)
    "easy.gr": ("small-graph.gr", "1013552d860d1f54b2658979aa9276aab01999bd0fd74551f7ffc00030f46030"),
    "medium.gr": ("medium-graph.gr", "5efe9a480cc11ab6ccd2406423f2fcdb38913f3fc924990a0cbca2e2f7ff8345"),
    "hard.gr": ("large-graph.gr", "d657ddc5e84d6e90837233d26ccf8c9b2fce57d3f7c381341f8313f8afa0aa89"),
}


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def numpy_torch():
    """The five torch calls ESA's evaluator makes, on numpy arrays."""
    class Tensor(np.ndarray):
        def cpu(self):
            return self

        def clone(self):
            return self.copy()

    def combinations(x, r=2):
        assert r == 2
        x = np.asarray(x)
        if len(x) < 2:
            return np.zeros((0, 2), dtype=x.dtype)
        i, j = np.triu_indices(len(x), 1)
        return np.stack([x[i], x[j]], axis=1)

    pt = types.ModuleType("torch")
    pt.int32 = np.int32
    pt.bool = np.bool_
    pt.tensor = lambda data, dtype=None: np.asarray(data, dtype=dtype).view(Tensor)
    pt.zeros = lambda shape, dtype=None: np.zeros(tuple(int(s) for s in shape), dtype=dtype).view(Tensor)
    pt.where = lambda cond: np.where(np.asarray(cond))
    pt.combinations = combinations
    return pt


def load_esa(esa_file, workdir):
    """Import ESA's file unchanged from a tree where its data paths resolve."""
    data = os.path.join(workdir, "data", "spoc3", "torso")
    os.makedirs(data, exist_ok=True)
    for esa_name, (ours, digest) in ESA_GR.items():
        src = os.path.join(ROOT, "data", ours)
        if sha256(src) != digest:
            raise SystemExit(f"data/{ours} is not byte-identical to ESA's {esa_name}")
        os.symlink(src, os.path.join(data, esa_name))
    for name in INSTANCES:
        os.symlink(os.path.join(ROOT, "data", f"{name}.gr"), os.path.join(data, f"{name}.gr"))
    pkg = os.path.join(workdir, "esa")
    os.makedirs(pkg)
    target = os.path.join(pkg, "esa_udp.py")
    with open(esa_file, "rb") as fh, open(target, "wb") as out:
        out.write(fh.read())
    sys.modules["torch"] = numpy_torch()
    spec = importlib.util.spec_from_file_location("esa_udp", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vectors(graph, rng):
    """(perm, t) pairs covering the paths through ESA's walk."""
    n = graph.n
    ts = [0, 1, n // 4, n // 2, 3 * n // 4, n - 2, n - 1]
    out = []
    base = hc.Starts.min_degree(graph, random.Random(1))
    out += [(base, t) for t in ts]
    for seed in (2, 3):
        p = hc.Starts.min_degree(graph, random.Random(seed))
        out += [(p, rng.randrange(n)) for _ in range(2)]
    for swaps in (5, 50, 500):
        p = list(base)
        for _ in range(swaps):
            i, j = rng.randrange(n), rng.randrange(n)
            p[i], p[j] = p[j], p[i]
        out += [(p, rng.randrange(n)) for _ in range(2)]
    for _ in range(2):
        p = list(range(n))
        rng.shuffle(p)
        out += [(p, 0), (p, rng.randrange(n))]
    dup = list(base)
    dup[1] = dup[0]
    out.append((dup, n // 2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--esa-file", required=True,
                    help="an evaluation .py downloaded from an SpOC-3 torso problem page")
    ap.add_argument("--instances", nargs="*", default=INSTANCES)
    ap.add_argument("--hv-sets", type=int, default=200)
    a = ap.parse_args()

    digest = sha256(a.esa_file)
    print(f"ESA evaluation file  {os.path.basename(a.esa_file)}")
    print(f"  sha256 {digest}  -> "
          f"{ESA_PY_SHA256.get(digest, 'NOT one of the three files published by ESA')}")

    md_csv = {r["instance"]: int(r["score"]) for r in
              csv.DictReader(open(os.path.join(HERE, "benchmark-min_degree-x8.csv")))}

    with tempfile.TemporaryDirectory() as work:
        esa = load_esa(a.esa_file, work)
        print("  instance files: data/small-, medium-, large-graph.gr are byte-identical "
              "to ESA's easy, medium, hard.gr\n")
        print(f"  {'instance':<14}{'n':>6}{'fitness':>12}{'(at 501)':>9}{'hypervolume':>14}"
              f"{'min-degree x8: CSV':>22}{'harness':>12}{'ESA':>12}{'s':>7}")
        total_f = total_h = total_e = bad = 0
        for name in a.instances:
            t0 = time.time()
            rng = random.Random(2026)
            graph = Graph.load(os.path.join(ROOT, "data", f"{name}.gr"))
            n = graph.n
            bits = build_adj_bitsets(n, graph.adj)
            udp = esa.graph_torso_udp(f"data/spoc3/torso/{name}.gr")
            assert int(udp.n_nodes) == n
            esa.udp = udp                         # combine_scores reads udp.n_nodes

            # 1 fitness
            vecs = test_vectors(graph, rng)
            same_f = cap = 0
            for perm, t in vecs:
                ours = evaluate(perm, t, bits, n)
                cap += ours[0] == 501
                theirs = esa.udp.fitness(np.array(list(perm) + [t]))
                if (int(theirs[0]), int(theirs[1])) == (int(ours[0]), int(ours[1])):
                    same_f += 1
                else:
                    print(f"    MISMATCH fitness t={t}: ESA {theirs} vs esa_eval {ours}")

            # 2 hypervolume
            same_h = 0
            for _ in range(a.hv_sets):
                k = rng.randint(1, 20)
                pts = [[rng.choice([rng.randrange(0, 501), 501]), rng.randrange(0, n)] for _ in range(k)]
                theirs = esa.combine_scores(np.array(pts, dtype=float))
                ours = -hypervolume_2d([tuple(p) for p in pts], n)
                if theirs == ours:
                    same_h += 1
                else:
                    print(f"    MISMATCH hypervolume: ESA {theirs} vs esa_eval {ours}")

            # 3 end to end: the min-degree x8 column
            front = Front(n)
            for seed in range(1, 9):
                front.add_solution(Solution(graph, hc.Starts.min_degree(graph, random.Random(seed))))
            sub = front.decision_vectors(20)
            ok, harness = score_vectors(graph, sub)
            esa_pts = np.array([esa.udp.fitness(np.array(x)) for x in sub], dtype=float)
            esa_score = int(round(esa.combine_scores(esa_pts)))
            e2e = ok and md_csv[name] == harness == esa_score

            total_f += same_f == len(vecs)
            total_h += same_h == a.hv_sets
            total_e += e2e
            bad += (same_f != len(vecs)) + (same_h != a.hv_sets) + (not e2e)
            print(f"  {name:<14}{n:>6}{f'{same_f}/{len(vecs)}':>12}{cap:>9}{f'{same_h}/{a.hv_sets}':>14}"
                  f"{md_csv[name]:>22,}{harness:>12,}{esa_score:>12,}{time.time() - t0:>7.0f}",
                  flush=True)
        k = len(a.instances)
        print(f"\n  identical fitness on {total_f}/{k} instances, identical hypervolume on "
              f"{total_h}/{k}, min-degree x8 column reproduced by ESA's code on {total_e}/{k}")
        print("  ALL IDENTICAL" if bad == 0 else f"  {bad} check(s) differ")
        sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
