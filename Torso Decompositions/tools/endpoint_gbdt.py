#!/usr/bin/env python3
"""endpoint_gbdt.py -- GBDT-RANKED endpoint search, with a paired control.

WHY THIS EXISTS
---------------
endpoint_ils.py broke the medium-graph leaderboard wall by minimising the t=0
elimination width, but it chooses its move AT RANDOM (a bottleneck vertex is
popped and reinserted at a uniformly random position).  That choice is exactly
a learnable ranking problem, and it is the one place where a learned model can
drive the result that actually scores.

This tool keeps the identical search loop and the identical acceptance rule,
and changes ONE thing: at each iteration it generates M candidate relocations
and picks which to evaluate.

    --algo-mode gbdt   : a LightGBM regressor ranks the M candidates and the
                         top-ranked one is evaluated  (treatment)
    --algo-mode random : one of the SAME M candidates is drawn uniformly
                         (paired control)

Because both arms draw from the same candidate pool with the same features and
the same budget, the ONLY difference is the selection rule -- so any gap is
attributable to the ranker.  This is the controlled ablation the thesis needs.

LEARNING SETUP
--------------
  sample  = one evaluated relocation (v: position i -> position j)
  features= static  : degree, neighbour-degree mean/min/max/std, k Laplacian
                      eigenvector coordinates of v
            dynamic : normalised i and j, displacement, front-degree at i,
                      front-degree at j, whether i is a current bottleneck,
                      current width and bottleneck count
  label   = lexicographic improvement  (cur_w - w) * 1000 + (cur_cnt - cnt)
            i.e. how much the move reduced the width, tie-broken by how much
            it thinned the set of vertices sitting at the width.

The model is refit periodically on a bounded replay buffer; the first
--warmup evaluations are random in BOTH arms so the treatment arm has data to
learn from (and so the arms share an identical cold start).

    python3 tools/endpoint_gbdt.py --problem small-graph --seed 1 --algo-mode gbdt
    python3 tools/endpoint_gbdt.py --problem small-graph --seed 1 --algo-mode random
"""
from __future__ import annotations
import argparse, json, os, random, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
# Pin BLAS/OpenMP to one thread BEFORE numpy loads.  Without this every arm
# spawns one BLAS thread per core (the dense Laplacian eigendecomposition and
# LightGBM both do it), which on a many-core box drives load into the hundreds
# and makes every arm slower.  It also keeps per-arm throughput stable, which
# the paired ablation depends on.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from core import (load_graph, graph_path, build_adj_bitsets, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from tools.endpoint_ils import full_staircase, seed_from_pool


# ----------------------------------------------------------------- the walk

def walk(n, ab, order):
    """Front-degree of every elimination step.  deg[i] = |N(order[i]) among
    the vertices still uneliminated|, with fill-in.  width = max(deg)."""
    nbr = list(ab); rem = (1 << n) - 1
    deg = [0] * n
    for i, v in enumerate(order):
        rem &= ~(1 << v)
        hi = nbr[v] & rem; deg[i] = hi.bit_count()
        x = hi
        while x:
            b = x & -x; x ^= b; u = b.bit_length() - 1
            nbr[u] |= hi & ~(1 << u)
    return deg


def summarise(deg):
    w = max(deg)
    hits = [i for i, d in enumerate(deg) if d == w]
    return w, len(hits), hits


# ------------------------------------------------------------ static features

def static_features(n, adj_l, k_eig):
    deg = np.array([len(adj_l[v]) for v in range(n)], dtype=np.float64)
    nd = []
    for v in range(n):
        ds = [len(adj_l[u]) for u in adj_l[v]] or [0]
        nd.append((float(np.mean(ds)), float(np.min(ds)),
                   float(np.max(ds)), float(np.std(ds))))
    nd = np.array(nd, dtype=np.float64)
    F = [deg[:, None], nd]
    if k_eig > 0:
        A = np.zeros((n, n))
        for v in range(n):
            for u in adj_l[v]:
                A[v, u] = 1.0
        d = A.sum(1); d[d == 0] = 1.0
        Dm = 1.0 / np.sqrt(d)
        L = np.eye(n) - (A * Dm).T * Dm
        _, vecs = np.linalg.eigh(L)
        F.append(vecs[:, 1:k_eig + 1])
    S = np.hstack(F)
    return (S - S.mean(0)) / (S.std(0) + 1e-9)


def make_model():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=120, num_leaves=31,
                             learning_rate=0.08, verbose=-1, n_jobs=1)


# -------------------------------------------------------------------- search

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--algo-mode", default="gbdt", choices=["gbdt", "random"],
                    help="gbdt = ranker picks the move; random = paired control")
    ap.add_argument("--cands", type=int, default=48, help="candidate moves per iteration")
    ap.add_argument("--warmup", type=int, default=3000, help="random evals before ranking")
    ap.add_argument("--retrain", type=int, default=1500, help="evals between refits")
    ap.add_argument("--epsilon", type=float, default=0.2,
                    help="exploration: fraction of iterations that ignore the "
                         "ranker and pick uniformly (pure argmax over-exploits "
                         "and collapses candidate diversity)")
    ap.add_argument("--buffer", type=int, default=30000)
    ap.add_argument("--k-eig", type=int, default=8)
    ap.add_argument("--restart", type=int, default=40000)
    ap.add_argument("--iters", type=int, default=10**12)
    ap.add_argument("--budget", type=float, default=0.0,
                    help="seconds; 0 = unlimited. NOT for the ablation -- a "
                         "wall-clock budget under fluctuating machine load "
                         "gives the two arms unequal work and confounds the "
                         "comparison. Use --max-evals instead.")
    ap.add_argument("--max-evals", type=int, default=0,
                    help="stop after this many evaluated moves; 0 = unlimited. "
                         "This is the correct ablation budget: both arms get "
                         "EXACTLY the same number of objective evaluations, so "
                         "the result is independent of machine load.")
    ap.add_argument("--algo", default="", help="submission stem (default auto)")
    a = ap.parse_args()

    n, adj_l = load_graph(graph_path(HERE, a.problem))
    ab = build_adj_bitsets(n, adj_l)
    rng = random.Random(a.seed)
    nprng = np.random.default_rng(a.seed)

    w0, order = seed_from_pool(n, ab, a.problem)
    deg = walk(n, ab, order)
    curw, curc, hits = summarise(deg)
    best = (curw, curc, order[:])     # ordering banked on a WIDTH drop
    # lexicographic best ever seen -- this is the ablation's primary metric.
    # (Tracking it separately matters: width drops are rare, so a metric that
    #  only updates on a width drop reports the starting value forever and
    #  makes every paired comparison a tie.)
    bestpair = (curw, curc)

    use_gbdt = (a.algo_mode == "gbdt")
    model = None
    if use_gbdt:
        try:
            model = make_model()
        except Exception as e:
            print(f"lightgbm unavailable ({e}); falling back to random control",
                  flush=True)
            use_gbdt = False

    print(f"=== endpoint_gbdt {a.problem} (n={n}, seed {a.seed}) | "
          f"mode {'GBDT-ranked' if use_gbdt else 'RANDOM control'} | "
          f"pool endpoint width {w0} | {a.cands} candidates/iter ===", flush=True)
    S = static_features(n, adj_l, a.k_eig)
    print(f"    static features: {S.shape[1]} per vertex | start "
          f"(width {curw}, #bottleneck {curc})", flush=True)

    stem = a.algo or f"endpoint_gbdt_{a.algo_mode}_s{a.seed}"
    out = os.path.join(HERE, "submissions", a.problem, f"{stem}.json")

    def bank(o):
        pts = full_staircase(n, ab, o)
        if pts is None:
            return
        arc = ParetoArchive()
        for w, t in pts:
            arc.try_add(w, t, o)
        top = arc.top_k_by_hv_contribution(60, n)
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": a.problem,
                   "decisionVector": [list(p) + [int(t)] for (_, t, p) in top]},
                  open(out, "w"))

    def build_rows(order, deg, curw, curc, cands):
        """Feature matrix for a list of (i, j) relocations."""
        idx = np.array([order[i] for (i, _) in cands], dtype=np.int64)
        ii = np.array([i for (i, _) in cands], dtype=np.float64)
        jj = np.array([j for (_, j) in cands], dtype=np.float64)
        di = np.array([deg[i] for (i, _) in cands], dtype=np.float64)
        dj = np.array([deg[min(int(j), n - 1)] for (_, j) in cands], dtype=np.float64)
        isb = (di == curw).astype(np.float64)
        dyn = np.stack([ii / n, jj / n, (jj - ii) / n, di, dj, isb,
                        np.full(len(cands), curw, dtype=np.float64),
                        np.full(len(cands), curc, dtype=np.float64)], axis=1)
        return np.hstack([S[idx], dyn])

    X = []; Y = []
    t0 = time.time(); evals = 0; fits = 0; since = 0
    # ranker-quality diagnostic: correlation between what the model PREDICTED
    # for the move it chose and what that move actually achieved.  This is the
    # number that says whether the GBDT learned anything at all.
    pred_hist = []; act_hist = []

    for it in range(a.iters):
        # ---- generate the SAME candidate pool in both arms ----
        cands = []
        for _ in range(a.cands):
            i = rng.choice(hits) if (hits and rng.random() < 0.7) else rng.randrange(n)
            j = rng.randrange(n)
            cands.append((i, j))
        rows = build_rows(order, deg, curw, curc, cands)

        # ---- the ONE difference: how the candidate is chosen ----
        ranked = False
        if use_gbdt and model is not None and evals >= a.warmup and fits > 0 \
                and rng.random() >= a.epsilon:
            pred = model.predict(rows)
            pick = int(np.argmax(pred))
            pred_hist.append(float(pred[pick])); ranked = True
        else:
            pick = rng.randrange(len(cands))

        i, j = cands[pick]
        cand = order[:]
        v = cand.pop(i)
        cand.insert(min(j, len(cand)), v)
        d2 = walk(n, ab, cand)
        w, c, h = summarise(d2)
        evals += 1; since += 1

        # ---- label + replay buffer ----
        label = (curw - w) * 1000.0 + (curc - c)
        X.append(rows[pick]); Y.append(label)
        if ranked:
            act_hist.append(label)
        if len(X) > a.buffer:
            X = X[-a.buffer:]; Y = Y[-a.buffer:]

        if (w, c) <= (curw, curc):
            order, curw, curc, hits, deg = cand, w, c, h, d2
            if (w, c) < bestpair:
                bestpair = (w, c)
            if w < best[0]:
                best = (w, c, cand[:]); since = 0
                bank(cand)
                print(f"  it {it}: *** ENDPOINT WIDTH {w} (was {w0}) banked "
                      f"[{a.algo_mode}] -> re-score: python3 tools/cap_submit.py "
                      f"--problem {a.problem} ***", flush=True)

        # ---- refit ----
        if use_gbdt and model is not None and evals >= a.warmup and \
                evals % a.retrain == 0 and len(X) >= 200:
            Xa = np.asarray(X); Ya = np.asarray(Y)
            if np.std(Ya) > 1e-9:
                try:
                    model.fit(Xa, Ya); fits += 1
                except Exception as e:
                    print(f"  [refit failed: {e}]", flush=True)

        if since >= a.restart:
            order = best[2][:]
            for _ in range(rng.randrange(3, 9)):
                vv = order.pop(rng.randrange(n)); order.insert(rng.randrange(n + 1), vv)
            deg = walk(n, ab, order); curw, curc, hits = summarise(deg); since = 0

        if a.max_evals and evals >= a.max_evals:
            break
        if a.budget and (time.time() - t0) >= a.budget:
            break
        if it % 2000 == 0 and it:
            el = time.time() - t0
            pr = ""
            if len(pred_hist) >= 200:
                p = np.asarray(pred_hist[-4000:]); q = np.asarray(act_hist[-4000:])
                m = min(len(p), len(q)); p, q = p[-m:], q[-m:]
                if np.std(p) > 1e-9 and np.std(q) > 1e-9:
                    pr = f" ranker-r {float(np.corrcoef(p, q)[0, 1]):+.3f}"
            print(f"  [it {it} {a.algo_mode} best{bestpair} cur({curw},{curc}) "
                  f"{fits} fits {evals/max(el,1e-9):.0f} eval/s {el:.0f}s{pr}]",
                  flush=True)

    # machine-readable final line for the paired ablation readout
    r = ""
    if len(pred_hist) >= 200:
        p = np.asarray(pred_hist); q = np.asarray(act_hist)
        m = min(len(p), len(q)); p, q = p[-m:], q[-m:]
        if np.std(p) > 1e-9 and np.std(q) > 1e-9:
            r = f"{float(np.corrcoef(p, q)[0, 1]):+.4f}"
    print(f"RESULT problem={a.problem} mode={a.algo_mode} seed={a.seed} "
          f"start_width={w0} best_width={bestpair[0]} best_bottleneck={bestpair[1]} "
          f"evals={evals} fits={fits} ranker_r={r or 'na'}", flush=True)


if __name__ == "__main__":
    main()
