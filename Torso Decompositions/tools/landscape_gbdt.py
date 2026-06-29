#!/usr/bin/env python3
r"""
landscape_gbdt.py -- GBDT as a *landscape probe*, not a generator.

Every other GBDT tool in this repo uses gradient boosting to PROPOSE orderings
(gbfcpp, gaps, mapelites).  This one turns the model around and uses it to
*explain* the optimal front we already have: for each width band w it asks

    "which structural property of a vertex decides whether it belongs in the
     optimal width-w torso S(w)?"

by training a GBDT to predict membership 1[v in S(w)] from the same spectral /
degree node-features the search uses, and reading off **permutation importance**
(model-agnostic: how much cross-validated AUC collapses when a feature family is
shuffled).  The output is a band x feature-family importance map -- a picture of
*what the landscape rewards at each resolution* -- plus a ruggedness / neutrality
measurement that quantifies why the search plateaus.

This is a LANDSCAPE-UNDERSTANDING tool.  On small-graph the front is gap-6
near-optimal (proven behind a clique wall), so this does not chase a beat; it
(a) characterises the optimum for the thesis and (b) tells us which features
matter at the hard high bands -- knowledge we feed into the medium/large decode,
where the plateau is genuinely escapable.

    python3 tools/landscape_gbdt.py --problem small-graph --k-eig 32 \
        --bands 1-14 --folds 5 --walk 4000

Additive / safe: writes only docs/figures/landscape_*.{png,csv}; touches no
submission.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  MAX_TW, LEADERBOARD_TARGETS)
from algorithms.continuous.cmaes_torso import build_node_features
from algorithms.continuous.np_gbdt import NpGBDT
from tools.fastwalk import IncEvalC
import tools.torso_deletion as td

_TREES = 80          # light booster: importance is stable well below 200 trees


# --------------------------------------------------------------------------- #
# feature family layout produced by build_node_features:
#   [0] degree
#   [1..4] neighbour-degree {min,max,mean,std}
#   [5..5+k-1] Laplacian eigenvectors 1..k (ascending non-trivial)
# We group the eigenvectors into low (smoothest, global structure) and high.
# --------------------------------------------------------------------------- #
def feature_families(k_eig):
    fam = {"degree": [0],
           "nbr-degree": [1, 2, 3, 4]}
    base = 5
    half = max(1, k_eig // 2)
    fam["eig-low(1..%d)" % half] = list(range(base, base + half))
    fam["eig-high(%d..%d)" % (half + 1, k_eig)] = list(range(base + half, base + k_eig))
    return fam


def auc(y, score):
    """Mann-Whitney AUC; robust to ties.  y in {0,1}."""
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    s = np.concatenate([neg, pos]); u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); start = cum - cnt
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[len(neg):].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def cv_auc(F, y, seed, folds, fam, perm_repeats=3):
    """k-fold CV AUC + permutation importance per feature family."""
    n = len(y); rng = np.random.default_rng(seed)
    idx = rng.permutation(n); fold = np.array_split(idx, folds)
    base_aucs = []; fam_drop = {f: [] for f in fam}
    for k in range(folds):
        te = fold[k]; tr = np.concatenate([fold[j] for j in range(folds) if j != k])
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):   # degenerate band
            continue
        gb = NpGBDT(n_estimators=_TREES, learning_rate=0.1, max_depth=4,
                    subsample=0.8, seed=seed + k)
        gb.fit(F[tr], y[tr].astype(np.float64))
        base = gb.predict(F[te]); a0 = auc(y[te], base)
        if not np.isfinite(a0):
            continue
        base_aucs.append(a0)
        for fname, cols in fam.items():
            drops = []
            for r in range(perm_repeats):
                Fp = F[te].copy()
                pr = np.random.default_rng(seed * 100 + k * 10 + r)
                for c in cols:
                    Fp[:, c] = Fp[pr.permutation(len(te)), c]
                drops.append(a0 - auc(y[te], gb.predict(Fp)))
            fam_drop[fname].append(np.mean(drops))
    mean_auc = float(np.mean(base_aucs)) if base_aucs else float("nan")
    imp = {f: float(np.mean(v)) if v else 0.0 for f, v in fam_drop.items()}
    return mean_auc, imp


# --------------------------------------------------------------------------- #
# best optimal front -> S(w) per band
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# fitness-landscape ruggedness + neutrality (torso-set space, exact oracle)
# --------------------------------------------------------------------------- #
def landscape_walk(by_w, ab, n, W, steps, seed):
    """Random adjacent walk over the width-W torso set: each step swap one
    in-vertex for one out-vertex, keep width<=W (the feasible neutral set).
    Returns (autocorr_lag1, neutral_fraction) of the torso-SIZE series."""
    if W not in by_w:
        return float("nan"), float("nan")
    t, perm = by_w[W]; S = set(perm[t:]); rng = np.random.default_rng(seed)

    def width_of(Sset):
        # min-degree elimination width UB, run directly on bitsets (popcount) --
        # ~order of magnitude faster than the python-set version.
        Slist = list(Sset)
        tor, _ = td.torso_adj(Slist, ab, n)
        a = {s: tor[s] for s in Slist}          # bitset adjacency restricted to S
        Smask = 0
        for s in Slist: Smask |= 1 << s
        for s in Slist: a[s] &= Smask; a[s] &= ~(1 << s)
        w = 0; live = set(Slist)
        while live:
            v = min(live, key=lambda u: a[u].bit_count())
            ns = a[v]; d = ns.bit_count()
            if d > w: w = d
            x = ns
            while x:
                b = x & -x; x &= x - 1; u = b.bit_length() - 1
                a[u] |= ns; a[u] &= ~(1 << u); a[u] &= ~(1 << v)
            live.discard(v)
            for u in live: a[u] &= ~(1 << v)
        return w

    sizes = [len(S)]; neutral = 0; moves = 0
    allv = list(range(n))
    for _ in range(steps):
        ins = list(S); outs = [v for v in allv if v not in S]
        if not ins or not outs: break
        drop = rng.choice(ins); add = rng.choice(outs)
        cand = (S - {int(drop)}) | {int(add)}
        if width_of(cand) <= W:           # feasible neutral/▒improving move
            if len(cand) == len(S): neutral += 1
            S = cand; moves += 1
        sizes.append(len(S))
    sizes = np.array(sizes, dtype=np.float64)
    if sizes.std() < 1e-9:
        ac = 1.0                          # perfectly flat = maximally neutral
    else:
        s0 = sizes - sizes.mean()
        ac = float((s0[:-1] * s0[1:]).sum() / (s0 * s0).sum())
    neutral_frac = neutral / max(1, moves)
    return ac, neutral_frac


def parse_bands(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--bands", default="1-14")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--walk", type=int, default=3000, help="ruggedness walk steps (0=skip)")
    ap.add_argument("--walk-bands", default="", help="restrict walk to these bands (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n, adj = load_graph(graph_path(here, a.problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n)
    bands = [w for w in parse_bands(a.bands)]
    walk_bands = set(parse_bands(a.walk_bands)) if a.walk_bands else set()
    fam = feature_families(a.k_eig)

    print(f"=== GBDT landscape probe | {a.problem} n={n} k_eig={a.k_eig} ===", flush=True)
    t0 = time.time()
    F = build_node_features(n, adj, a.k_eig)
    by_w = load_front(here, a.problem, n, ev)
    print(f"features {F.shape}  front bands {sorted(by_w)}  [{time.time()-t0:.0f}s]", flush=True)

    rows = []                              # band, |S|, auc, importances..., ac, neutral
    famnames = list(fam)
    figdir = os.path.join(here, "docs", "figures"); os.makedirs(figdir, exist_ok=True)
    csvp = os.path.join(figdir, f"landscape_{a.problem}.csv")
    csv_fh = open(csvp, "w")
    csv_fh.write("band,torso_size,cv_auc," + ",".join(famnames) + ",ruggedness,neutrality\n")
    csv_fh.flush()
    print("\n band |S|   CV-AUC  " + "  ".join(f"{f:>16}" for f in famnames)
          + "   rugged  neutral", flush=True)
    for W in bands:
        if W not in by_w:
            continue
        t, perm = by_w[W]; S = set(perm[t:])
        y = np.zeros(n, dtype=np.int64)
        for v in S: y[v] = 1
        if y.sum() in (0, n):
            continue
        mauc, imp = cv_auc(F, y, a.seed, a.folds, fam)
        ac, neu = (float("nan"), float("nan"))
        do_walk = a.walk > 0 and (not walk_bands or W in walk_bands)
        if do_walk:
            ac, neu = landscape_walk(by_w, ab, n, W, a.walk, a.seed + W)
        impv = [imp[f] for f in famnames]
        row = [W, len(S), mauc] + impv + [ac, neu]
        rows.append(row)
        csv_fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row) + "\n")
        csv_fh.flush()
        print(f"  {W:>3} {len(S):>4} {mauc:>7.3f}  "
              + "  ".join(f"{v:>16.3f}" for v in impv)
              + f"   {ac:>6.3f}  {neu:>6.3f}", flush=True)

    csv_fh.close()
    print(f"\nwrote {csvp}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        M = np.array([[r[3 + j] for j in range(len(famnames))] for r in rows], dtype=float)
        bb = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.5))
        im = ax.imshow(M, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(famnames))); ax.set_xticklabels(famnames, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(bb))); ax.set_yticklabels([f"w={w}" for w in bb], fontsize=8)
        ax.set_title(f"What determines optimal torso membership ({a.problem})\n"
                     "permutation importance (CV-AUC drop) per feature family")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUC drop")
        fig.tight_layout()
        pngp = os.path.join(figdir, f"landscape_{a.problem}.png")
        fig.savefig(pngp, dpi=140); print(f"wrote {pngp}", flush=True)
    except Exception as e:
        print(f"[figure skipped: {e}]", flush=True)


if __name__ == "__main__":
    main()
