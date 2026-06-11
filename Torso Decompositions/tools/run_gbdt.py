"""
run_gbdt.py -- cuda-torso's per-threshold neuroevolution + an ADDITIVE GBDT
front-booster (this work's novelty), with a --no_gbdt control for ablation.

Engine (verbatim from cuda-torso/run.py): 740 polynomial-spectral features, a
population of E-dim policies decoded by argsort(policy @ nodes) ON THE GPU, the
custom libeval.cu evaluator, N per-threshold elites, mutation + COSYNE. This is
the proven leaderboard method and reaches its score at ~10^5 generations.

Novelty (GBFC, additive): every --gbdt_every generations we fit a gradient-boosted
weak learner to the worst-served threshold band, decode GBDT-guided specialist
orderings, evaluate them with the SAME kernel, and pool them into the front as a
separate "bonus" set. The neuroevolution elites are never disturbed, so the
booster can only help (the pool keeps the best width per threshold). --no_gbdt
disables the booster entirely == vanilla cuda-torso, giving a controlled
with/without-GBDT ablation on the identical engine, seed and budget.

Drop this file next to run.py in the cuda-torso repo (it imports libeval.so and
reads data/<graph>.gr exactly as run.py does), then:

    python3 run_gbdt.py --graph small-graph --gbdt_every 200            # boosted
    python3 run_gbdt.py --graph small-graph --no_gbdt                   # control
"""
import argparse, ctypes, json, math, os, time
from ctypes import c_size_t, c_bool, c_uint16, c_int, POINTER
import numpy as np
from numpy.linalg import eigh
from scipy.sparse.csgraph import laplacian
import torch

GRAPH_SIZES = {"small-graph": 1357, "medium-graph": 1399, "large-graph": 2426}

libeval = ctypes.CDLL('./libeval.so', mode=ctypes.RTLD_GLOBAL)
evaluate = libeval.evaluate
evaluate.argtypes = [POINTER(c_bool), POINTER(c_uint16), POINTER(c_uint16),
                     POINTER(c_int), c_size_t, c_size_t]


def _eval_perms(perms_t, N, adj_bool):
    """Evaluate a (K,N) batch of orderings with the cuda-torso kernel.
    Returns fitnesses (K0,N) int32 = width at every threshold for each ordering.
    The kernel launches <<<128, B/128>>>, so B MUST be a multiple of 128 (and
    >=128) or it silently evaluates nothing — we pad the batch and slice back."""
    K0 = perms_t.shape[0]
    K = ((K0 + 127) // 128) * 128
    if K != K0:                                   # pad up to a multiple of 128
        perms_t = torch.cat([perms_t, perms_t[-1:].repeat(K - K0, 1)], 0)
    adjs = torch.from_numpy(np.repeat(adj_bool[None], K, 0)).cuda()
    perms = perms_t.to(torch.uint16).contiguous().cuda()
    degrees = torch.empty((K, N), dtype=torch.uint32).cuda()
    fitnesses = torch.full((K, N), 999999, dtype=torch.int).cuda()
    evaluate(ctypes.cast(adjs.data_ptr(), POINTER(c_bool)),
             ctypes.cast(perms.data_ptr(), POINTER(c_uint16)),
             ctypes.cast(degrees.data_ptr(), POINTER(c_uint16)),
             ctypes.cast(fitnesses.data_ptr(), POINTER(c_int)), K, N)
    return fitnesses[:K0]


def _gbdt_candidates(raw_np, band_perms, n, n_rand, seed):
    """Fit a GBDT weak learner to the band-best orderings and emit GBDT-guided
    candidate orderings: pure GBDT ranking, wide guided-random policies, and
    perturbations. Returns (cand_perms (K,N) int64 cpu, gcol (n,) float32)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(seed)
    # training set: node -> its position, over the band's best orderings
    X = np.repeat(raw_np, len(band_perms), 0)
    y = np.empty(n * len(band_perms), dtype=np.float64)
    for k, p in enumerate(band_perms):
        pos = np.empty(n); pos[p] = np.arange(n)
        y[k*n:(k+1)*n] = pos
    gb = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08,
                                       max_depth=4, random_state=seed)
    gb.fit(X, y)
    gcol = gb.predict(raw_np).astype(np.float32)
    gcol = (gcol - gcol.mean()) / (gcol.std() + 1e-9)
    cands = [np.argsort(gcol), np.argsort(-gcol)]            # pure GBDT orderings
    feats_g = np.vstack([raw_np.T, gcol[None]])              # (37+1, n)
    for _ in range(n_rand):                                  # wide guided explorers
        x = rng.normal(0, 1, feats_g.shape[0]).astype(np.float32)
        x[-1] *= float(rng.uniform(2.0, 6.0))               # emphasise learned column
        cands.append(np.argsort(feats_g.T @ x))
    return np.asarray(cands, dtype=np.int64), gcol


def main():
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    p.add_argument("--graph", choices=set(GRAPH_SIZES), default="small-graph")
    p.add_argument("--eigenvectors", type=int, default=32)
    p.add_argument("--init_stdev", type=float, default=0.3)
    p.add_argument("--mutation_stdev", type=float, default=0.3)
    p.add_argument("--mutation_proba", type=float, default=0.5)
    p.add_argument("--cosyne_proba", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_generations", type=int, default=100_000)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--gbdt_every", type=int, default=200, help="GBDT boost period (0/none with --no_gbdt)")
    p.add_argument("--gbdt_band", type=int, default=120, help="half-width of the targeted threshold band")
    p.add_argument("--gbdt_rand", type=int, default=48, help="wide GBDT-guided explorers per boost")
    p.add_argument("--no_gbdt", action="store_true", help="ablation: vanilla cuda-torso")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    use_gbdt = not args.no_gbdt
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    run_id = int(time.time()); B = args.batch_size; N = GRAPH_SIZES[args.graph]
    E = feature_count(args.eigenvectors)
    for d in ("logs", "submissions", "checkpoints"):
        os.makedirs(f"{d}/{args.graph}", exist_ok=True)

    adj = init_adj(N, args.graph).astype(np.bool_)
    adjs = np.repeat(adj[None, :, :], B, 0)
    nodes_np = init_node_features(adj, args.eigenvectors)        # (E, N)
    raw_np = nodes_np[:args.eigenvectors + 5, :].T.copy()        # (N, 37) GBDT input
    nodes = torch.from_numpy(nodes_np).cuda()

    population = np.random.normal(0.0, args.init_stdev, (B, E)).astype(np.float32)
    elites = torch.empty((N, E), dtype=torch.float32).cuda()
    bkup_adjs = torch.from_numpy(adjs).cuda(); adjs = torch.from_numpy(adjs).cuda()
    population = torch.from_numpy(population).cuda()
    perms = torch.empty((B, N), dtype=torch.uint16).cuda()
    degrees = torch.empty((B, N), dtype=torch.uint32).cuda()
    fitnesses = torch.full((B, N), 999999, dtype=torch.int).cuda()
    elite_fitnesses = torch.full((N,), 999999, dtype=torch.int).cuda()
    elite_range = torch.ones((N,), dtype=torch.float32, device="cuda") / B
    ts = torch.arange(N).cuda()
    logits = torch.empty((B, N), dtype=torch.float32).cuda()

    # ADDITIVE bonus front from the GBDT booster (explicit orderings, not policies)
    bonus_perms = torch.zeros((N, N), dtype=torch.int64).cuda()
    bonus_fit = torch.full((N,), 999999, dtype=torch.int).cuda()
    have_bonus = torch.zeros((N,), dtype=torch.bool).cuda()

    print(f"=== cuda-torso + GBDT booster -- {args.graph} "
          f"(B={B}, gbdt={'OFF (control)' if not use_gbdt else f'every {args.gbdt_every}'}) ===", flush=True)

    for generation in range(args.max_generations):
        start = time.perf_counter()
        logits[:] = population @ nodes
        perms[:] = logits.argsort(axis=1).to(torch.uint16)
        adjs[:, :, :] = bkup_adjs[:, :, :]
        evaluate(ctypes.cast(adjs.data_ptr(), POINTER(c_bool)),
                 ctypes.cast(perms.data_ptr(), POINTER(c_uint16)),
                 ctypes.cast(degrees.data_ptr(), POINTER(c_uint16)),
                 ctypes.cast(fitnesses.data_ptr(), POINTER(c_int)), B, N)
        best = fitnesses.min(axis=0)
        better = best.values <= elite_fitnesses
        elites[better] = population[best.indices][better]
        elite_fitnesses[better] = best.values[better]

        # next population (cuda-torso, verbatim)
        idx = torch.multinomial(elite_range, B, replacement=N < B)
        population[:] = elites[idx]
        mask = torch.rand((B, E), device="cuda") > args.mutation_proba
        population += torch.normal(0.0, args.mutation_stdev, (B, E), device="cuda") * mask
        permn = int(population.shape[0] * population.shape[1] * args.cosyne_proba)
        tr = torch.randint(0, B, (permn,)); orr = torch.randint(0, B, (permn,)); cc = torch.randint(0, E, (permn,))
        population[tr, cc] = population[orr, cc]

        # ---- GBDT booster (the novelty); additive, never touches elites ----
        if use_gbdt and args.gbdt_every and generation > 0 and generation % args.gbdt_every == 0:
            pooled = torch.minimum(elite_fitnesses, bonus_fit)
            tstar = int(torch.argmax(pooled.float()).item())
            lo, hi = max(0, tstar - args.gbdt_band), min(N, tstar + args.gbdt_band)
            el_perms = (elites @ nodes).argsort(axis=1)             # (N,N) on GPU
            band_ts = sorted(set(int(round(lo + i*(hi-lo-1)/9)) for i in range(10)))
            band_perms = [el_perms[t].cpu().numpy() for t in band_ts]
            for t in band_ts:                                       # include bonus winners
                if bool(have_bonus[t].item()):
                    band_perms.append(bonus_perms[t].cpu().numpy())
            cand, _ = _gbdt_candidates(raw_np, band_perms, N, args.gbdt_rand, args.seed + generation)
            cfit = _eval_perms(torch.from_numpy(cand), N, adj)      # (K,N)
            cbest = cfit.min(axis=0)
            imp = cbest.values < bonus_fit
            if bool(imp.any().item()):
                rows = cand[cbest.indices.cpu().numpy()]            # best ordering per threshold
                bonus_perms[imp] = torch.from_numpy(rows).cuda()[imp]
                bonus_fit[imp] = cbest.values[imp]
                have_bonus[imp] = True

        if generation % args.log_every == 0:
            rt = time.perf_counter() - start
            pooled = torch.minimum(elite_fitnesses, bonus_fit)
            _, hvi = calculate_hvi(ts, pooled, N)
            extra = f" | bonus-t {int(have_bonus.sum().item())}" if use_gbdt else ""
            print(f"gen {generation:6d} | official {hvi:,} | rate {(B/rt):.0f}/s{extra}", flush=True)

        if generation % args.checkpoint_every == 0:
            pooled = torch.minimum(elite_fitnesses, bonus_fit)
            sel, hvi = calculate_hvi(ts, pooled, N)
            sub = create_submission_pooled(elites, nodes, bonus_perms, bonus_fit,
                                           elite_fitnesses, sel, args.graph, N)
            sp = f"submissions/{args.graph}/{hvi}.json"
            if not os.path.exists(sp):
                json.dump(sub, open(sp, "w"))


def create_submission_pooled(elites, nodes, bonus_perms, bonus_fit, elite_fit, ts, graph, N):
    """For each selected threshold, submit whichever ordering (elite-decoded or
    GBDT bonus) gives the lower width."""
    el = (elites @ nodes).argsort(axis=1)
    dv = []
    for t in ts:
        t = int(t)
        if int(bonus_fit[t].item()) < int(elite_fit[t].item()):
            perm = bonus_perms[t].cpu().tolist()
        else:
            perm = el[t].cpu().tolist()
        dv.append(perm + [t])
    return {"challenge": "spoc-3-torso-decompositions", "problem": graph, "decisionVector": dv}


def calculate_hvi(ts, degrees, N):
    degrees = degrees.cpu().numpy(); ts = ts.cpu().numpy()
    min_degree = np.full_like(degrees, N); min_t = np.full_like(ts, N)
    hvi = 0; selected = []
    for _ in range(20):
        area = (min_t - ts) * (min_degree - degrees)
        am = int(np.argmax(area)); selected.append(am)
        t = ts[am]; d = degrees[am]
        contribution = (min_degree[am] - d) * (min_t[am] - t)
        i = am
        while 0 <= i < N and min_degree[i] > d: min_degree[i] = d; i += 1
        i = am
        while 0 <= i < N and min_t[i] > t: min_t[i] = t; i -= 1
        hvi -= contribution
    return np.array(selected), hvi


def feature_count(e):
    r = e + 5
    return r + r + (math.factorial(r) // (2 * math.factorial(r - 2)))


def init_adj(N, graph):
    adj = np.zeros((N, N), dtype=np.bool_)
    for line in open(f"data/{graph}.gr"):
        s, d = line.split(" "); adj[int(s), int(d)] = True; adj[int(d), int(s)] = True
    return adj


def init_node_features(adj, eigenvectors):
    N = adj.shape[0]; dp = 5; raw = dp + eigenvectors; feats = feature_count(eigenvectors)
    nodes = np.zeros((N, feats), dtype=np.float32)
    for i in range(N): nodes[i, 0] = adj[i].sum()
    for i in range(N):
        nb = nodes[adj[i], 0]
        nodes[i, 1] = nb.min(); nodes[i, 2] = nb.max(); nodes[i, 3] = nb.mean(); nodes[i, 4] = nb.std()
    lap = laplacian(adj.astype(np.int8), normed=True)
    ev, evec = eigh(lap); evec = np.real(evec[:, ev.argsort()])
    nodes[:, dp:dp + eigenvectors] = evec[:, 1:eigenvectors + 1]
    from itertools import combinations
    for i in range(raw): nodes[:, raw + i] = nodes[:, i] ** 2
    for ii, (i, j) in enumerate(combinations(range(raw), 2)):
        nodes[:, raw + raw + ii] = nodes[:, i] * nodes[:, j]
    means = nodes.mean(0); stds = nodes.std(0); stds[stds == 0] = 1.0
    return ((nodes - means) / stds).T


if __name__ == "__main__":
    main()
