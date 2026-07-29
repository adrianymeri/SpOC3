#!/usr/bin/env python3
"""
gbdt_grow.py -- GBDT-guided set-space torso growth, targeted at the 20
HSSP-scoring widths (the only widths the leaderboard sees).

WHY THIS EXISTS (the corrected finding, 2026-07-29): torso_deletion's medium
run was budget-truncated at 24h while sweeping widths ascending -- its accepts
(w=44/83/87) are all BELOW the scoring widths; set-space growth has never been
applied to the widths that actually score.  Medium's envelope exceeds the
target by ~1,900 HV; the entire remaining gap is packing at those widths.
This tool is torso_deletion's fast deferral move, but:
  1. restricted to the HSSP-optimal 20 widths (auto-derived, like gbfcpp --cap20),
  2. candidates ranked by a GBDT trained per-width on the pool's own elite
     orderings (features: degree profile + Laplacian eigenvectors; label:
     elite torso-membership frequency at that width)  -- NOVELTY: GBDT as a
     learned set-space deferral ranker,
  3. --no-gbdt = identical loop with the classical boundary-count ranking
     (the paired ablation control),
  4. month-scale budget by default; saves on every accept (interruption-safe).

    python3 tools/gbdt_grow.py --problem medium-graph --budget 2592000 \
        > gbdt_grow_medium.log 2>&1 &
    grep ACCEPT gbdt_grow_medium.log            # progress
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)


# ---------- staircase helpers (torso_deletion lineage, bitset-fast) ----------

def front_deg(perm, ab, n):
    sm = [0] * n; cur = 0
    for i in range(n - 1, -1, -1):
        sm[i] = cur; cur |= 1 << perm[i]
    tmp = list(ab); deg = [0] * n
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
    return deg


def add_staircase(perm, ab, n, arc):
    deg = front_deg(perm, ab, n)
    if max(deg) > MAX_TW:          # dirty head voids the vector at ESA
        return False
    run = 0
    for t in range(n - 1, -1, -1):
        if deg[t] > run: run = deg[t]
        arc.try_add(int(run), t, list(perm))
    return True


def load_pool(problem, n, ab, arc):
    pool = []
    for fp in sorted(glob.glob(os.path.join(HERE, "submissions", problem, "*.json"))):
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            dvs = e["decisionVector"]
        except Exception:
            continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1:
                p = [int(x) for x in dv[:-1]]
                if sorted(p) == list(range(n)):
                    pool.append(p)
    seen, uniq = set(), []
    for p in pool:
        tp = tuple(p)
        if tp not in seen:
            seen.add(tp); uniq.append(p)
    kept = []
    for p in uniq:
        if add_staircase(p, ab, n, arc):
            kept.append(p)
    return kept


# ---------- features + per-width GBDT membership ranker ----------

def build_features(n, adj_l, k_eig=16):
    deg = np.array([len(adj_l[v]) for v in range(n)], dtype=np.float64)
    nd = []
    for v in range(n):
        ds = [len(adj_l[u]) for u in adj_l[v]] or [0]
        nd.append((np.mean(ds), np.min(ds), np.max(ds), np.std(ds)))
    nd = np.array(nd)
    A = np.zeros((n, n))
    for v in range(n):
        for u in adj_l[v]:
            A[v, u] = 1.0
    d = A.sum(1); d[d == 0] = 1.0
    Dm = 1.0 / np.sqrt(d)
    L = np.eye(n) - (A * Dm).T * Dm
    vals, vecs = np.linalg.eigh(L)
    E = vecs[:, 1:k_eig + 1]
    F = np.hstack([deg[:, None], nd, E])
    return (F - F.mean(0)) / (F.std(0) + 1e-9)


def make_model():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.08, verbose=-1)
    except Exception:
        return None, None


def membership_labels(w, pool, ab, n):
    """label[v] = fraction of pool orderings whose width-w torso contains v."""
    cnt = np.zeros(n); tot = 0
    for p in pool:
        deg = front_deg(p, ab, n)
        run = 0; tstar = None
        for t in range(n - 1, -1, -1):
            if deg[t] > run: run = deg[t]
            if run <= w:
                tstar = t
        if tstar is None:
            continue
        tot += 1
        for v in p[tstar:]:
            cnt[v] += 1.0
    return (cnt / max(tot, 1))


# ---------- main growth loop ----------

def run(problem, budget_s, algo, no_gbdt, max_cand, seed):
    n, adj_l = load_graph(graph_path(HERE, problem))
    ab = build_adj_bitsets(n, adj_l)
    target = LEADERBOARD_TARGETS.get(problem)
    rng = np.random.default_rng(seed)

    arc = ParetoArchive()
    pool = load_pool(problem, n, ab, arc)
    base = -hypervolume_2d(arc.points(), n)
    print(f"=== gbdt_grow -- {problem} (n={n}) | pool {len(pool)} | "
          f"HV {base:,.0f}" + (f" gap {base-target:+,.0f}" if target else "") +
          f" | mode {'CONTROL(no-gbdt)' if no_gbdt else 'GBDT'} ===", flush=True)

    top20 = arc.top_k_by_hv_contribution(20, n)
    widths = sorted({int(w) for (w, _, _) in top20})
    print(f"scoring widths: {widths}", flush=True)

    model_name, F = None, None
    if not no_gbdt:
        model_name, _ = make_model()
        F = build_features(n, adj_l)
        print(f"ranker: {model_name or 'UNAVAILABLE -> falling back to boundary'}",
              flush=True)

    def best_by_w():
        bw = {}
        for w, t, p in arc.entries():
            if w in widths and (w not in bw or t < bw[w][0]):
                bw[w] = (t, p)
        return bw

    out = os.path.join(HERE, "submissions", problem, f"{algo}.json")
    def save():
        top = arc.top_k_by_hv_contribution(60, n)
        write_submission([list(p) + [int(t)] for (_, t, p) in top], problem, out)

    fails = {}
    t0 = time.time(); accepts = 0; passes = 0
    rank_cache = {}
    while time.time() - t0 < budget_s:
        bw = best_by_w()
        cand_w = []
        ws = sorted(bw)
        for i, w in enumerate(ws):
            t_w = bw[w][0]
            nxt = bw[ws[i + 1]][0] if i + 1 < len(ws) else 0
            room = t_w - nxt
            if room <= 0 or t_w == 0:
                continue
            cand_w.append((room * (0.5 ** fails.get(w, 0)), w))
        if not cand_w:
            print("no addressable scoring widths; stopping"); break
        wts = np.array([c[0] for c in cand_w]); wts /= wts.sum()
        w = int(cand_w[int(rng.choice(len(cand_w), p=wts))][1])
        t_star, perm = bw[w]
        X, S = perm[:t_star], perm[t_star:]
        Smask = 0
        for s in S: Smask |= 1 << s

        # rank head candidates
        head = [v for v in X if ab[v] & Smask]
        if not no_gbdt and model_name:
            if w not in rank_cache or rank_cache[w][0] != len(pool):
                y = membership_labels(w, pool[:40], ab, n)
                import lightgbm as lgb
                m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31,
                                      learning_rate=0.08, verbose=-1)
                m.fit(F, y)
                rank_cache[w] = (len(pool), m.predict(F))
            score = rank_cache[w][1]
            head.sort(key=lambda v: -score[v])
        else:
            head.sort(key=lambda v: (ab[v] & Smask).bit_count())
        head = head[:max_cand]

        moved = False
        before = -hypervolume_2d(arc.points(), n)
        for bsz in (8, 4, 2, 1):
            if moved: break
            for i in range(0, len(head), bsz):
                B = head[i:i + bsz]
                if not B: continue
                Bset = set(B)
                Xrest = [u for u in X if u not in Bset]
                for depth in (0, 1, 4, 16):
                    full = Xrest + S[:depth] + B + S[depth:]
                    add_staircase(full, ab, n, arc)
                after = -hypervolume_2d(arc.points(), n)
                if after < before:
                    accepts += 1; moved = True
                    fails[w] = 0
                    msg = (f"*** ACCEPT #{accepts} w={w} t: {t_star}->"
                           f"{len(Xrest)} (batch {len(B)}) HV {after:,.0f}")
                    if target: msg += f" gap {after - target:+,.0f}"
                    print(msg, flush=True)
                    save()
                    pool.append(full)
                    break
        if not moved:
            fails[w] = fails.get(w, 0) + 1
        passes += 1
        if passes % 25 == 0:
            cur = -hypervolume_2d(arc.points(), n)
            print(f"[pass {passes} | {accepts} accepts | HV {cur:,.0f}"
                  + (f" gap {cur-target:+,.0f}" if target else "")
                  + f" | {time.time()-t0:.0f}s]", flush=True)

    final = -hypervolume_2d(arc.points(), n)
    print(f"final: {accepts} accepts | HV {final:,.0f}"
          + (f" gap {final-target:+,.0f}" if target else ""), flush=True)
    save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="medium-graph",
                    choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--budget", type=float, default=2592000.0,  # 30 days
                    help="seconds (default: one month)")
    ap.add_argument("--algo", default="gbdt_grow")
    ap.add_argument("--no-gbdt", action="store_true",
                    help="ablation control: boundary-count ranking")
    ap.add_argument("--max-cand", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.problem, a.budget, a.algo, a.no_gbdt, a.max_cand, a.seed)


if __name__ == "__main__":
    main()
