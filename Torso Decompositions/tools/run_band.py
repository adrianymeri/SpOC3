r"""
run_band.py -- cuda-torso's per-threshold neuroevolution, but with the search
CONCENTRATED on a chosen set of threshold bands (the "specialist front per
band" idea).

Stock cuda-torso reseeds each generation UNIFORMLY from its N per-threshold
elites, so every threshold gets 1/N of the search pressure.  When only a handful
of breakpoints are stuck, that's wasteful.  run_band.py reseeds almost entirely
from the elites of the TARGET thresholds (+ a small window), pouring ~N/|target|
times more pressure onto exactly the bands we need to crack -- while still
tracking ALL per-threshold elites so the saved submission is a complete front to
pool against the banked best.

Drop next to run.py in the cuda-torso repo (needs libeval.so + data/<graph>.gr):

    # crack the small-graph rigid breakpoints (grown thresholds t*(w)-1):
    python3 run_band.py --graph small-graph \
        --targets 678,971,923,867,1043,235 --band_radius 25 \
        --batch_size 1024 --init_stdev 0.5 --mutation_stdev 0.45 \
        --cosyne_proba 0.35 --max_generations 300000 --band_frac 0.9
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
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--targets", type=str, required=True,
                   help="comma-separated target thresholds to concentrate search on")
    p.add_argument("--band_radius", type=int, default=25,
                   help="+/- window of thresholds around each target to also weight")
    p.add_argument("--band_frac", type=float, default=0.9,
                   help="fraction of reseeding pressure put on the target bands")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    run_id = int(time.time()); B = args.batch_size; N = GRAPH_SIZES[args.graph]
    E = feature_count(args.eigenvectors)
    for d in ("logs", "submissions", "checkpoints"):
        os.makedirs(f"{d}/{args.graph}", exist_ok=True)

    adj = init_adj(N, args.graph).astype(np.bool_)
    adjs = np.repeat(adj[None, :, :], B, 0)
    nodes = torch.from_numpy(init_node_features(adj, args.eigenvectors)).cuda()
    population = np.random.normal(0.0, args.init_stdev, (B, E)).astype(np.float32)
    elites = torch.empty((N, E), dtype=torch.float32).cuda()
    bkup_adjs = torch.from_numpy(adjs).cuda(); adjs = torch.from_numpy(adjs).cuda()
    population = torch.from_numpy(population).cuda()
    perms = torch.empty((B, N), dtype=torch.uint16).cuda()
    degrees = torch.empty((B, N), dtype=torch.uint32).cuda()
    fitnesses = torch.full((B, N), 999999, dtype=torch.int).cuda()
    elite_fitnesses = torch.full((N,), 999999, dtype=torch.int).cuda()
    ts = torch.arange(N).cuda()
    logits = torch.empty((B, N), dtype=torch.float32).cuda()

    # ---- band reseeding weights: concentrate on target thresholds + window ----
    targets = [int(x) for x in args.targets.split(",") if x.strip() != ""]
    band = np.zeros(N, dtype=np.float64)
    for t in targets:
        lo, hi = max(0, t - args.band_radius), min(N, t + args.band_radius + 1)
        band[lo:hi] += 1.0
    band = band / band.sum()
    uniform = np.ones(N) / N
    w = args.band_frac * band + (1 - args.band_frac) * uniform   # mix in a little global
    band_w = torch.from_numpy((w / w.sum()).astype(np.float64)).cuda()
    print(f"=== run_band -- {args.graph} | targets={targets} radius={args.band_radius} "
          f"band_frac={args.band_frac} ===", flush=True)

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

        # reseed CONCENTRATED on the target bands (this is the only change)
        idx = torch.multinomial(band_w, B, replacement=True)
        population[:] = elites[idx]
        mask = torch.rand((B, E), device="cuda") > args.mutation_proba
        population += torch.normal(0.0, args.mutation_stdev, (B, E), device="cuda") * mask
        permn = int(B * E * args.cosyne_proba)
        tr = torch.randint(0, B, (permn,)); orr = torch.randint(0, B, (permn,)); cc = torch.randint(0, E, (permn,))
        population[tr, cc] = population[orr, cc]

        if generation % args.log_every == 0:
            _, hvi = calculate_hvi(ts, elite_fitnesses, N)
            tgt = " ".join(f"t{t}:{int(elite_fitnesses[t].item())}" for t in targets)
            print(f"gen {generation:6d} | HVI {hvi:,} | {tgt} | {(B/(time.perf_counter()-start)):.0f}/s",
                  flush=True)

        if generation % args.checkpoint_every == 0:
            sel, hvi = calculate_hvi(ts, elite_fitnesses, N)
            sub = create_submission(elites, nodes, sel, args.graph)
            sp = f"submissions/{args.graph}/{hvi}.json"
            if not os.path.exists(sp):
                json.dump(sub, open(sp, "w"))


def create_submission(elites, nodes, ts, graph):
    perms = (elites @ nodes).argsort(axis=1)
    return {"challenge": "spoc-3-torso-decompositions", "problem": graph,
            "decisionVector": [perm.tolist() + [int(t)] for perm, t in zip(perms[ts].cpu(), ts)]}


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
