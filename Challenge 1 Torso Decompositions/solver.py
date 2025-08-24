#!/usr/bin/env python3
"""
solver_targeted.py

Two-phase MOSA + focused-SA solver for SPOC-3 torso decompositions.
Single-file. Uses local data files in ./data/<problem>.gr.

Features:
 - Phase A: broad MOSA exploration with adaptive acceptance (favors width when size equal)
 - Phase B: targeted focused-SA seeds with controlled size drop (explore sweet spots)
 - Fast exact evaluator (bitset) with per-process initializer and lru_cache
 - Multiprocessing for evaluations; final exact re-evaluation in parallel
 - Output: submission_<problem>.json (top 20 decision vectors)
"""
import argparse
import json
import math
import os
import random
import sys
import time
from functools import lru_cache
from typing import List, Set, Tuple, Dict

import numpy as np
from tqdm import tqdm
import multiprocessing

try:
    import pygmo as pg
    HAVE_PYGMO = True
except Exception:
    HAVE_PYGMO = False

# -------------------------
# Default params
# -------------------------
DEFAULTS = {
    "initial_temp": 1.5,
    "cooling_rate": 0.9997,
    "steps_per_temp": 200,
    "max_steps": 200000,
    "workers": max(1, multiprocessing.cpu_count() - 1),
    "relink_interval": 5000,
    "intensify_interval": 4000,
    "intensify_iters": 200,
    "archive_cap": 5000,
    "phase2_seeds": 128,
    "phase2_max_size_drop": 120,
    "size_relax_prob": 0.10,
    "diversity_inject_frequency": 2000,
    "diversity_inject_strength": 120,
    "log_every": 500,
    "seed_file": None,
    "data_dir": "data",
}


# -------------------------
# Graph loading & bitsets
# -------------------------
def load_graph_local(problem_id: str, data_dir: str = "data") -> Tuple[int, List[Set[int]]]:
    path = os.path.join(data_dir, f"{problem_id}.gr")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    edges = []
    max_node = -1
    with open(path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(b"#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            edges.append((u, v))
            max_node = max(max_node, u, v)
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    print(f" Graph loaded: n={n}, edges={len(edges)} from {path}")
    return n, adj


def build_adj_bitsets(n: int, adj_list: List[Set[int]]) -> List[int]:
    bits = [0] * n
    for u in range(n):
        b = 0
        for v in adj_list[u]:
            b |= (1 << v)
        bits[u] = b
    return bits


# -------------------------
# Worker globals for pool
# -------------------------
WORKER_ADJ_BITS = None
WORKER_N = None


def _init_worker(adj_bits: List[int], n: int):
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = tuple(adj_bits)
    WORKER_N = int(n)


# -------------------------
# Fast exact evaluator (bitset)
# -------------------------
def bitcount(x: int) -> int:
    try:
        return int(x).bit_count()
    except Exception:
        return bin(int(x)).count("1")


@lru_cache(maxsize=300000)
def evaluate_bitset_cached(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Return (size, width). size = n - t (bigger is better). width <= 500 else set to 501.
    solution_tuple length must be n+1: perm[0..n-1], t.
    """
    global WORKER_ADJ_BITS, WORKER_N
    if WORKER_ADJ_BITS is None or WORKER_N is None:
        raise RuntimeError("Worker globals not initialized")
    n = WORKER_N
    adj_bits = list(WORKER_ADJ_BITS)

    if len(solution_tuple) != n + 1:
        return (0, 501)
    t = int(solution_tuple[-1])
    perm = list(map(int, solution_tuple[:-1]))
    # clamp t
    t = max(0, min(n - 1, t))
    size = n - t
    if size <= 0:
        return (0, 501)

    # suffix mask: nodes appearing after position i
    suffix_mask = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr
        curr |= (1 << perm[i])

    temp = adj_bits[:]  # copy
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount(succ)
        if i >= t:
            if out_deg > max_width:
                max_width = out_deg
                if max_width >= 500:
                    return (size, 501)
        if succ == 0:
            continue
        s = succ
        # close edges between successors
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= (succ ^ (1 << v))
    return (size, max_width)


def eval_wrapper_for_pool(sol_tuple):
    return sol_tuple, evaluate_bitset_cached(sol_tuple)


# -------------------------
# Pareto helpers
# -------------------------
def dominates(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    # a dominates b if a.size >= b.size & a.width <= b.width and strictly better in one
    return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])


def update_archive(archive: List[Dict], candidate: Dict, archive_cap: int) -> Tuple[List[Dict], bool]:
    new_archive = []
    cand_dom = False
    for item in archive:
        if dominates(item['score'], candidate['score']):
            cand_dom = True
            break
    if cand_dom:
        return archive, False
    # remove those dominated by candidate
    for item in archive:
        if dominates(candidate['score'], item['score']):
            continue
        new_archive.append(item)
    new_archive.append(candidate)
    # cap archive
    if len(new_archive) > archive_cap:
        new_archive.sort(key=lambda x: (-x['score'][0], x['score'][1]))
        new_archive = new_archive[:archive_cap]
    return new_archive, True


def compute_hypervolume(archive: List[Dict], n: int) -> float:
    if not archive:
        return 0.0
    pts = [[p['score'][0], p['score'][1]] for p in archive]
    try:
        if HAVE_PYGMO:
            hv = pg.hypervolume(pts)
            ref = [n + 1, n + 1]  # safe ref
            return hv.compute(ref)
    except Exception:
        pass
    # crude hv fallback
    ref_s, ref_w = n + 1, n + 1
    pts_sorted = sorted(pts, key=lambda x: (-x[0], x[1]))
    hv = 0.0
    prev_s = ref_s
    max_w = ref_w
    for s, w in pts_sorted:
        width = max_w - w
        hv += max(0, prev_s - s) * max(0, width)
        prev_s = s
    return float(hv)


# -------------------------
# Neighborhood operators
# -------------------------
def get_neighbor(solution: List[int], n: int, size_relax_prob: float = 0.0) -> List[int]:
    perm = solution[:-1][:]
    t = int(solution[-1])
    r = random.random()
    if r < 0.35:
        if n > 1:
            i = random.randint(0, n - 2)
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
    elif r < 0.70:
        if n > 2:
            a, b = sorted(random.sample(range(n), 2))
            perm[a:b + 1] = list(reversed(perm[a:b + 1]))
    elif r < 0.95:
        shift = int(max(1, n * 0.05))
        t = max(0, min(n - 1, t + random.randint(-shift, shift)))
    else:
        if n > 3:
            block_size = random.randint(2, max(3, int(n * 0.05)))
            start = random.randint(0, n - block_size)
            block = perm[start:start + block_size]
            del perm[start:start + block_size]
            insert_pos = random.randint(0, len(perm))
            perm = perm[:insert_pos] + block + perm[insert_pos:]

    # probabilistic size relaxation: sometimes encourage increasing t (reducing size)
    if random.random() < size_relax_prob:
        relax = int(max(1, n * 0.01))
        t = max(0, min(n - 1, t + random.randint(0, relax)))
    return perm + [t]


# -------------------------
# Path relink (simple greedy)
# -------------------------
def path_relink(sol1: List[int], sol2: List[int], n: int, max_steps: int = 2000) -> List[Dict]:
    perm1 = sol1[:-1][:]
    perm2 = sol2[:-1][:]
    t_new = int((sol1[-1] + sol2[-1]) / 2)
    curr = perm1[:]
    pos = {node: i for i, node in enumerate(curr)}
    path = []
    for step in range(min(n, max_steps)):
        i = 0
        while i < n and curr[i] == perm2[i]:
            i += 1
        if i >= n:
            break
        desired = perm2[i]
        cur_pos = pos.get(desired)
        if cur_pos is None:
            break
        curr[i], curr[cur_pos] = curr[cur_pos], curr[i]
        pos[curr[cur_pos]] = cur_pos
        pos[curr[i]] = i
        solution = curr[:] + [t_new]
        try:
            score = evaluate_bitset_cached(tuple(solution))
        except RuntimeError:
            # if called from main before init, skip scoring here
            score = (0, 501)
        path.append({'solution': solution[:], 'score': score})
    return path


# -------------------------
# MOSA search (Phase A)
# -------------------------
def mosa_search(n: int, adj_bits: List[int], adj_list: List[Set[int]], config: Dict, seed_solutions: List[List[int]] = None):
    # ensure main thread worker globals for direct calls
    _init_worker(adj_bits, n)

    # create pool for later final evaluations and parallel intensifications
    workers = max(1, min(config.get("workers", 1), multiprocessing.cpu_count()))
    pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=(adj_bits, n))

    try:
        # initial solution
        if seed_solutions:
            cur = random.choice(seed_solutions)[:]
            if len(cur) != n + 1:
                perm = cur[:-1] if len(cur) >= n + 1 else list(cur[:-1]) + [i for i in range(n) if i not in cur[:-1]]
                t = cur[-1] if len(cur) >= n + 1 else int(n * 0.5)
                cur = perm[:n] + [max(0, min(n - 1, int(t)))]
        else:
            cur = list(np.random.permutation(n)) + [random.randint(int(n * 0.2), int(n * 0.8))]

        cur_score = evaluate_bitset_cached(tuple(int(x) for x in cur))
        archive = []
        archive, _ = update_archive(archive, {'solution': cur[:], 'score': cur_score}, config['archive_cap'])
        best_hv = compute_hypervolume(archive, n)

        step = 0
        temp = config['initial_temp']
        last_relink = 0
        last_intensify = 0

        pbar = tqdm(total=config['max_steps'], desc="=% MOSA", unit="it")
        size_relax_prob = config.get("size_relax_prob", 0.1)
        while step < config['max_steps'] and temp > 1e-8:
            for _ in range(config['steps_per_temp']):
                if step >= config['max_steps']: break
                step += 1
                pbar.update(1)

                cand = get_neighbor(cur, n, size_relax_prob=size_relax_prob)
                try:
                    cand_score = evaluate_bitset_cached(tuple(int(x) for x in cand))
                except RuntimeError:
                    cand_score = (0, 501)

                # adaptive delta (favors width reduction when size equal)
                ds = cand_score[0] - cur_score[0]
                dw = cur_score[1] - cand_score[1]
                if ds > 0:
                    delta = 4.0 * ds + 0.5 * dw
                elif ds == 0:
                    delta = 0.0 + 10.0 * dw
                else:
                    delta = 1.0 * ds + 2.0 * dw

                accept = False
                if dominates((cand_score[0], cand_score[1]), (cur_score[0], cur_score[1])):
                    accept = True
                else:
                    if delta >= 0:
                        accept = True
                    else:
                        # stochastic acceptance governed by temperature
                        prob = math.exp(delta / max(1e-12, temp))
                        if random.random() < prob:
                            accept = True

                if accept:
                    cur = cand
                    cur_score = cand_score

                archive, added = update_archive(archive, {'solution': cand[:], 'score': cand_score}, config['archive_cap'])
                if added:
                    hv = compute_hypervolume(archive, n)
                    if hv > best_hv:
                        best_hv = hv

                # path relinking occasionally
                if step - last_relink >= config['relink_interval'] and len(archive) >= 2:
                    last_relink = step
                    arch_sorted = sorted(archive, key=lambda x: (-x['score'][0], x['score'][1]))
                    rel_path = path_relink(arch_sorted[0]['solution'], arch_sorted[min(1, len(arch_sorted)-1)]['solution'], n)
                    for s in rel_path:
                        try:
                            archive, _ = update_archive(archive, s, config['archive_cap'])
                        except Exception:
                            pass

                # intensify occasionally (short greedy local moves)
                if step - last_intensify >= config['intensify_interval']:
                    last_intensify = step
                    arch_sorted = sorted(archive, key=lambda x: (-x['score'][0], x['score'][1]))
                    if arch_sorted:
                        top = arch_sorted[0]['solution']
                        best_local = top[:]
                        best_local_score = evaluate_bitset_cached(tuple(int(x) for x in best_local))
                        for ii in range(config.get("intensify_iters", 120)):
                            cand_local = get_neighbor(best_local, n, size_relax_prob=0.02)
                            score_local = evaluate_bitset_cached(tuple(int(x) for x in cand_local))
                            if dominates((score_local[0], score_local[1]), (best_local_score[0], best_local_score[1])):
                                best_local = cand_local[:]
                                best_local_score = score_local
                        archive, _ = update_archive(archive, {'solution': best_local[:], 'score': best_local_score}, config['archive_cap'])

                # diversity injection events
                if step % config.get("diversity_inject_frequency", 10000) == 0:
                    for _inj in range(max(1, config.get("diversity_inject_strength", 50) // 50)):
                        # inject a random smaller-size seed: increase t by random amount to reduce size
                        drop = random.randint(1, max(1, config.get("diversity_inject_strength", 120)))
                        t_new = max(0, min(n - 1, cur[-1] + drop))
                        perm = list(np.random.permutation(n))
                        injected = perm + [t_new]
                        try:
                            sscore = evaluate_bitset_cached(tuple(int(x) for x in injected))
                            archive, _ = update_archive(archive, {'solution': injected, 'score': sscore}, config['archive_cap'])
                        except RuntimeError:
                            pass

                # logging
                if step % config.get("log_every", 500) == 0:
                    hv = compute_hypervolume(archive, n)
                    top_share = sum(1 for a in archive if a['score'][0] == n) / max(1, len(archive))
                    width_best = min(a['score'][1] for a in archive)
                    tqdm.write(f"[step {step}] archive={len(archive)} cur={cur_score} hv={hv:.1f} temp={temp:.6f} top_share={top_share:.2f} width_best={width_best}")

            temp *= config['cooling_rate']

        pbar.close()

        # finalize
        print(" Phase A finished. Preparing seeds for Phase B...")
        # make unique archive list
        uniq = {}
        for item in archive:
            key = tuple(map(int, item['solution']))
            if key not in uniq:
                uniq[key] = item
        uniq_list = list(uniq.values())
        pool.close()
        pool.join()
        return uniq_list

    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass


# -------------------------
# Phase B: focused SA on seeds with controlled size drops
# -------------------------
def focused_sa_on_seeds(n: int, adj_bits: List[int], seeds: List[Dict], config: Dict) -> List[Dict]:
    _init_worker(adj_bits, n)
    workers = max(1, min(config.get("workers", 1), multiprocessing.cpu_count()))
    pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=(adj_bits, n))
    try:
        chosen = sorted(seeds, key=lambda x: (-x['score'][0], x['score'][1]))[:config.get("phase2_seeds", 128)]
        # generate focused seeds by applying controlled size drops (increase t)
        focused_seeds = []
        max_drop = config.get("phase2_max_size_drop", 120)
        for s in chosen:
            base_perm = s['solution'][:-1]
            base_t = int(s['solution'][-1])
            # generate several seeds with t increased (size decreased)
            for drop in [0] + list(range(1, max(1, max_drop // 8) + 1, max(1, max_drop // 8)) )[:8]:
                tnew = min(n - 1, base_t + drop)
                cand = base_perm[:] + [tnew]
                focused_seeds.append({'solution': cand, 'score': evaluate_bitset_cached(tuple(int(x) for x in cand))})
        # unique
        uniq = {}
        for item in focused_seeds:
            key = tuple(map(int, item['solution']))
            if key not in uniq:
                uniq[key] = item
        focused = list(uniq.values())
        print(f" Phase B: running focused-SA on {len(focused)} seeds...")
        results = []
        # run each seed sequentially but using per-seed local SA (we can parallelize this but keep it simple)
        def run_seed(seed_item):
            cur = seed_item['solution'][:]
            cur_score = evaluate_bitset_cached(tuple(int(x) for x in cur))
            temp = config.get("phase2_initial_temp", 0.6)
            cooling = config.get("phase2_cooling", 0.9996)
            steps = config.get("phase2_steps", 10000)
            for step in range(steps):
                cand = get_neighbor(cur, n, size_relax_prob=0.02)
                cand_score = evaluate_bitset_cached(tuple(int(x) for x in cand))
                ds = cand_score[0] - cur_score[0]
                dw = cur_score[1] - cand_score[1]
                if ds > 0:
                    delta = 4.0 * ds + 0.5 * dw
                elif ds == 0:
                    delta = 0.0 + 10.0 * dw
                else:
                    delta = 1.0 * ds + 2.0 * dw
                accept = False
                if dominates((cand_score[0], cand_score[1]), (cur_score[0], cur_score[1])):
                    accept = True
                else:
                    if delta >= 0:
                        accept = True
                    else:
                        if random.random() < math.exp(delta / max(1e-12, temp)):
                            accept = True
                if accept:
                    cur = cand
                    cur_score = cand_score
                temp *= cooling if (step % 200 == 0) else 1.0
            return {'solution': cur[:], 'score': cur_score}

        # Parallel map seeds across pool
        seed_args = [s for s in focused]
        # Use pool.imap_unordered for better resource use
        results = []
        for r in pool.imap_unordered(run_seed, seed_args):
            # each r already evaluated
            results.append(r)
        pool.close()
        pool.join()
        # combine results
        final_archive = []
        for r in results:
            final_archive, _ = update_archive(final_archive, r, config['archive_cap'])
        return final_archive
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass


# -------------------------
# Orchestrator
# -------------------------
def write_submission(nondom: List[Dict], problem_id: str, top_k: int = 20):
    if not nondom:
        print("No solutions to write.")
        return
    nondom_sorted = sorted(nondom, key=lambda x: (-x['score'][0], x['score'][1]))
    decision_vectors = [item['solution'] for item in nondom_sorted[:top_k]]
    mapping = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
    submission = {"decisionVector": decision_vectors, "problem": mapping.get(problem_id, problem_id),
                  "challenge": "spoc-3-torso-decompositions"}
    fname = f"submission_{problem_id}.json"
    with open(fname, "w") as f:
        json.dump(submission, f, indent=2)
    print(f" Wrote {len(decision_vectors)} solutions to {fname}")


def load_seed_file(seed_file: str, n: int) -> List[List[int]]:
    if not seed_file or not os.path.exists(seed_file):
        return []
    data = json.load(open(seed_file, "r"))
    vecs = []
    if isinstance(data, dict) and "decisionVector" in data:
        vecs = data["decisionVector"]
    elif isinstance(data, list):
        vecs = data
    seeds = []
    for v in vecs:
        flat = v[0] if isinstance(v[0], (list, tuple)) else v
        if len(flat) >= n + 1:
            seeds.append(list(map(int, flat[:n+1])))
    print(f" Loaded {len(seeds)} seeds from {seed_file}")
    return seeds


def parse_args():
    p = argparse.ArgumentParser(description="Targeted MOSA solver (two-phase)")
    p.add_argument("--problem", type=str, required=True, choices=["easy", "medium", "hard"])
    p.add_argument("--data-dir", type=str, default=DEFAULTS['data_dir'])
    p.add_argument("--max_steps", type=int, default=DEFAULTS['max_steps'])
    p.add_argument("--steps_per_temp", type=int, default=DEFAULTS['steps_per_temp'])
    p.add_argument("--initial_temp", type=float, default=DEFAULTS['initial_temp'])
    p.add_argument("--cooling_rate", type=float, default=DEFAULTS['cooling_rate'])
    p.add_argument("--workers", type=int, default=DEFAULTS['workers'])
    p.add_argument("--relink_interval", type=int, default=DEFAULTS['relink_interval'])
    p.add_argument("--intensify_interval", type=int, default=DEFAULTS['intensify_interval'])
    p.add_argument("--intensify_iters", type=int, default=DEFAULTS['intensify_iters'])
    p.add_argument("--archive_cap", type=int, default=DEFAULTS['archive_cap'])
    p.add_argument("--phase2_seeds", type=int, default=DEFAULTS['phase2_seeds'])
    p.add_argument("--phase2_max_size_drop", type=int, default=DEFAULTS['phase2_max_size_drop'])
    p.add_argument("--size_relax_prob", type=float, default=DEFAULTS['size_relax_prob'])
    p.add_argument("--diversity_inject_frequency", type=int, default=DEFAULTS['diversity_inject_frequency'])
    p.add_argument("--diversity_inject_strength", type=int, default=DEFAULTS['diversity_inject_strength'])
    p.add_argument("--log_every", type=int, default=DEFAULTS['log_every'])
    p.add_argument("--seed_file", type=str, default=DEFAULTS['seed_file'])
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)

    # load graph
    n, adj_list = load_graph_local(args.problem, data_dir=args.data_dir)
    adj_bits = build_adj_bitsets(n, adj_list)

    # load seeds
    seeds = load_seed_file(args.seed_file, n)

    config = {
        "max_steps": args.max_steps,
        "steps_per_temp": args.steps_per_temp,
        "initial_temp": args.initial_temp,
        "cooling_rate": args.cooling_rate,
        "workers": args.workers,
        "relink_interval": args.relink_interval,
        "intensify_interval": args.intensify_interval,
        "intensify_iters": args.intensify_iters,
        "archive_cap": args.archive_cap,
        "phase2_seeds": args.phase2_seeds,
        "phase2_max_size_drop": args.phase2_max_size_drop,
        "size_relax_prob": args.size_relax_prob,
        "diversity_inject_frequency": args.diversity_inject_frequency,
        "diversity_inject_strength": args.diversity_inject_strength,
        "log_every": args.log_every,
        # phase2 local SA params
        "phase2_initial_temp": 0.6,
        "phase2_cooling": 0.9996,
        "phase2_steps": int(20000),
    }

    start = time.time()
    print("=== Phase A: broad MOSA exploration ===")
    archiveA = mosa_search(n, adj_bits, adj_list, config, seed_solutions=seeds)
    print(f" Phase A returned {len(archiveA)} unique archive members. Time: {time.time() - start:.1f}s")

    # Phase B focused runs
    print("=== Phase B: focused SA on seeds derived from Phase A ===")
    focused_archive = focused_sa_on_seeds(n, adj_bits, archiveA, config)
    # merge and final nondominated filter (exact re-eval)
    merged = {}
    for item in archiveA + focused_archive:
        key = tuple(map(int, item['solution']))
        if key not in merged:
            merged[key] = item
    merged_list = list(merged.values())

    # final parallel exact re-eval
    # initialize worker in main so evaluate_bitset_cached can be called here
    _init_worker(adj_bits, n)
    workers = max(1, min(args.workers, multiprocessing.cpu_count()))
    pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=(adj_bits, n))
    try:
        tasks = [tuple(int(x) for x in item['solution']) for item in merged_list]
        results = pool.map(eval_wrapper_for_pool, tasks)
    finally:
        pool.close()
        pool.join()

    final_archive = []
    for sol_tuple, score in results:
        final_archive, _ = update_archive(final_archive, {'solution': list(sol_tuple), 'score': score}, config['archive_cap'])

    print(f" Final nondominated solutions: {len(final_archive)}")
    write_submission(final_archive, args.problem)
    print(f" Total Time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
