#!/usr/bin/env python3
r"""
spectral_seed.py -- diagnosis-driven seeding for the medium/large instances.

The landscape probe (THESIS s13.8, tools/landscape_gbdt.py) showed that on medium
the optimal torso membership is determined by the GLOBAL low-Laplacian
(community/separator) structure at *every* width band, while the production search
(gbfcpp/GAPS) decides mainly from local/dynamic features -- a representation
mismatch that plateaus the search ~13k HV short.  This tool injects the missing
global structure as additive seeds:

  1. spectral nested-dissection orderings -- recursively bisect the graph along the
     low eigenvectors and eliminate the SEPARATORS LAST (the low-fill ordering that
     bounded-treewidth theory prescribes), pure numpy (no scipy).  Several depth /
     separator-fraction variants.
  2. plain spectral orderings -- argsort along each low eigenvector and random
     low-eig combinations.
  3. a classical min-fill ordering as a strong non-spectral baseline.

Every ordering is scored with the exact C-kernel front evaluator, so each adopted
band is a real torso size; the per-band best across all candidates AND the current
front is pooled and written to submissions/<problem>/spectral_seed.json.  Dropping
that file in the pool means gbfcpp/portfolio (and the GBDT they train) learn from
globally-structured elites they would never have generated locally.

ADDITIVE / SAFE: writes only the `spectral_seed` stem.  Honest framing: this is a
*seeding* play for an escapable plateau, not an exact certificate; whether it beats
the production front is exactly what the local test reports.

    python3 tools/spectral_seed.py --problem medium-graph --keig 16 \
        --depth 6 --sep 0.04,0.06,0.08 --combos 40
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(400000)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, min_fill_in_perm)
from tools.fastwalk import IncEvalC
from algorithms.continuous.cmaes_torso import get_features


def load_front(here, problem, n, ev):
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            p = json.load(open(fp)); e = p[0] if isinstance(p, list) else p
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    perm = [int(x) for x in dv[:-1]]; d = ev.full(perm); r = 0
                    for t in range(n - 1, -1, -1):
                        c = int(d[t]); r = c if c > r else r
                        if r <= MAX_TW: arc.try_add(r, t, perm)
        except Exception:
            continue
    by_w = {}
    for w, t, p in arc.entries():
        if w not in by_w or t < by_w[w][0]: by_w[w] = (t, list(p))
    return by_w


def eig_low(here, problem, n, adj, keig):
    F, _ = get_features(here, problem, n, adj, max(32, keig))
    return np.asarray(F)[:, 5:5 + keig]


def front_hv(by_w, n):
    a = ParetoArchive()
    for w in by_w: a.try_add(w, by_w[w][0], None)
    return -hypervolume_2d(a.points(), n)


# --------------------------------------------------------------------------- #
# spectral nested dissection (numpy-only): recursively split along successive
# low eigenvectors, separators eliminated LAST (low-fill ordering).
# --------------------------------------------------------------------------- #
def spectral_nd(verts, E, depth, max_depth, leaf, sep_frac):
    m = len(verts)
    col = E[verts, depth % E.shape[1]]
    sv = [verts[i] for i in np.argsort(col)]
    if m <= leaf or depth >= max_depth:
        return sv
    s = max(2, int(m * sep_frac)); lo = (m - s) // 2; hi = lo + s
    A, sep, B = sv[:lo], sv[lo:hi], sv[hi:]
    return (spectral_nd(A, E, depth + 1, max_depth, leaf, sep_frac)
            + spectral_nd(B, E, depth + 1, max_depth, leaf, sep_frac)
            + sep)


def adopt(perm, ev, n, by_w, max_w, best_band):
    """Score an elimination ordering; record per-band best torso size."""
    d = ev.full(perm); r = 0
    for t in range(n - 1, -1, -1):
        c = int(d[t]); r = c if c > r else r
        if r > max_w:
            continue
        cur = best_band.get(r, n + 1)
        if t < cur:
            best_band[r] = t
            if r not in by_w or t < by_w[r][0]:
                by_w[r] = (t, list(perm))


def run(problem, here, keig, depth, seps, combos, leaf, seed):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n); target = LEADERBOARD_TARGETS.get(problem)
    by_w = load_front(here, problem, n, ev)
    start_hv = front_hv(by_w, n)
    max_w = max(by_w) if by_w else MAX_TW
    E = eig_low(here, problem, n, adj, keig)
    rng = np.random.default_rng(seed); allv = list(range(n))
    best_band = {w: by_w[w][0] for w in by_w}
    print(f"=== spectral-seed {problem} | n={n} | front {start_hv:,.0f}"
          f"{f'  gap {start_hv-target:+,.0f}' if target else ''} | keig={keig} ===", flush=True)

    t0 = time.time(); n_ord = 0
    # 1) spectral nested-dissection variants
    for sf in seps:
        perm = spectral_nd(allv, E, 0, depth, leaf, sf); adopt(perm, ev, n, by_w, max_w, best_band); n_ord += 1
    c = front_hv(by_w, n)
    print(f"  [spectral-ND] {len(seps)} variants | front {c:,.0f}"
          f"{f'  gap {c-target:+,.0f}' if target else ''}", flush=True)

    # 2) single-eigenvector + random low-eig combination orderings
    dirs = [E[:, j] for j in range(keig)]
    for _ in range(combos):
        w = rng.normal(0, 1, min(keig, 6)); w /= np.linalg.norm(w) + 1e-12
        dirs.append(E[:, :len(w)] @ w)
    for vec in dirs:
        for perm in (list(np.argsort(vec)), list(np.argsort(-vec))):
            adopt([int(x) for x in perm], ev, n, by_w, max_w, best_band); n_ord += 1
    c = front_hv(by_w, n)
    print(f"  [spectral-orders] {len(dirs)*2} orderings | front {c:,.0f}"
          f"{f'  gap {c-target:+,.0f}' if target else ''}", flush=True)

    # 3) classical min-fill baseline (non-spectral reference)
    try:
        perm = min_fill_in_perm(n, ab, sample_size=64, rng=random.Random(seed))
        adopt([int(x) for x in perm], ev, n, by_w, max_w, best_band); n_ord += 1
        c = front_hv(by_w, n)
        print(f"  [min-fill baseline] front {c:,.0f}"
              f"{f'  gap {c-target:+,.0f}' if target else ''}", flush=True)
    except Exception as e:
        print(f"  [min-fill skipped: {e}]", flush=True)

    fin = front_hv(by_w, n)
    gained = start_hv - fin
    print(f"\n{n_ord} orderings scored in {time.time()-t0:.0f}s | "
          f"final front {fin:,.0f}"
          f"{f'  gap {fin-target:+,.0f}' if target else ''} | "
          f"{'IMPROVED +%.0f HV' % gained if gained > 0.5 else 'no gain over production front'}", flush=True)

    # always write the pooled front as a seed source for gbfcpp/portfolio
    a = ParetoArchive()
    for w in by_w:
        if by_w[w][1] is not None: a.try_add(w, by_w[w][0], by_w[w][1])
    top = a.top_k_by_hv_contribution(40, n)
    dvs = [list(p) + [int(t)] for (_, t, p) in top]
    out = os.path.join(here, "submissions", problem, "spectral_seed.json")
    json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
               "decisionVector": dvs}, open(out, "w"))
    print(f"saved -> {out}  (VERIFY with tools/verify_submission.py)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="medium-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--keig", type=int, default=16)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--sep", default="0.04,0.06,0.08")
    ap.add_argument("--combos", type=int, default=40)
    ap.add_argument("--leaf", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seps = [float(x) for x in a.sep.split(",") if x.strip()]
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        a.keig, a.depth, seps, a.combos, a.leaf, a.seed)


if __name__ == "__main__":
    main()
