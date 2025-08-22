#!/usr/bin/env python3
"""
solver_mosa.py

Multi-Objective Simulated Annealing (MOSA) solver for SPOC-3 torso decompositions.
Designed to run on a powerful multi-core server.

Features:
- Fast exact evaluator using integer bitsets (per-process LRU cache).
- MOSA main loop with archive (Pareto) maintenance.
- Seeding from data folder and optional seed JSON file.
- Path relinking and intensification.
- Parallel final exact re-evaluation and output submission file.
- CLI options to fully utilize the machine.
"""

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm
import multiprocessing

try:
    import pygmo as pg
    HAVE_PYGMO = True
except ImportError:
    HAVE_PYGMO = False

# -------------------------
# CLI and default params
# -------------------------
DEFAULTS = {
    "initial_temp": 1.0,
    "final_temp": 1e-5,
    "cooling_rate": 0.9995,
    "steps_per_temp": 200,
    "max_steps": 200000,
    "workers": max(1, multiprocessing.cpu_count()),
    "relink_interval": 5000,
    "intensify_interval": 4000,
    "intensify_iters": 120,
    "archive_cap": 5000,
    "seed_file": None,
    "data_dir": "data",
    "log_every": 1000,
    "restart_patience": 20000,
}

# -------------------------
# Graph loading & bitset builder
# -------------------------
def load_graph_local(problem_id: str, data_dir: str = "data") -> Tuple[int, List[Set[int]]]:
    path = os.path.join(data_dir, f"{problem_id}.gr")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}. Please create a 'data' directory and place it there.")
    
    edges = []
    max_node = -1
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            max_node = max(max_node, u, v)
    
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    print(f"✅ Graph loaded: n={n}, edges={len(edges)} from {path}")
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
# Worker globals & initializer
# -------------------------
WORKER_ADJ_BITS = None
WORKER_N = None

def _init_worker(adj_bits: List[int], n: int):
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n

# -------------------------
# Fast exact evaluator (bitset)
# -------------------------
def bitcount(x: int) -> int:
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")

@lru_cache(maxsize=200000)
def evaluate_bitset_cached(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    global WORKER_ADJ_BITS, WORKER_N
    if WORKER_ADJ_BITS is None or WORKER_N is None:
        raise RuntimeError("Worker globals not initialized")

    n = WORKER_N
    adj_bits = WORKER_ADJ_BITS
    t = int(solution_tuple[-1])
    perm = list(solution_tuple[:-1])
    
    size = n - t
    if size <= 0:
        return (0, 501)

    suffix_mask = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr
        curr |= (1 << perm[i])

    temp = adj_bits[:]
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        
        if i >= t:
            out_deg = bitcount(succ)
            if out_deg > max_width:
                max_width = out_deg
                if max_width >= 500:
                    return (size, 501)
        
        if succ == 0:
            continue
            
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= (succ ^ vbit)
            
    return (size, max_width)

def eval_wrapper_for_pool(sol_tuple):
    return sol_tuple, evaluate_bitset_cached(sol_tuple)

# -------------------------
# Pareto helpers
# -------------------------
def dominates(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return (a[0] >= b[0] and a[1] < b[1]) or (a[0] > b[0] and a[1] <= b[1])

def update_archive(archive: List[Dict], candidate: Dict, archive_cap: int) -> Tuple[List[Dict], bool]:
    cand_score = candidate['score']
    if any(dominates(s['score'], cand_score) for s in archive):
        return archive, False
        
    new_archive = [s for s in archive if not dominates(cand_score, s['score'])]
    new_archive.append(candidate)
    
    if len(new_archive) > archive_cap:
        new_archive.sort(key=lambda x: (-x['score'][0], x['score'][1]))
        new_archive = new_archive[:archive_cap]
        
    return new_archive, True

def compute_hypervolume(archive: List[Dict], n: int) -> float:
    if not archive or not HAVE_PYGMO:
        return 0.0
        
    points = np.array([[p['score'][1], -p['score'][0]] for p in archive], dtype=float)
    
    try:
        hv = pg.hypervolume(points)
        ref = [502, 0] 
        return -hv.compute(ref)
    except Exception:
        return 0.0

# -------------------------
# Neighborhood operators (MOSA)
# -------------------------
def get_neighbor(solution: List[int], n: int) -> List[int]:
    perm, t = solution[:-1], solution[-1]
    r = random.random()
    if r < 0.35:
        if n > 1: i = random.randint(0, n - 2); perm[i], perm[i+1] = perm[i+1], perm[i]
    elif r < 0.70:
        if n > 2: a, b = sorted(random.sample(range(n), 2)); perm[a:b+1] = reversed(perm[a:b+1])
    elif r < 0.95:
        shift = int(max(1, n * 0.05)); t = max(0, min(n-1, t + random.randint(-shift, shift)))
    else:
        if n > 3:
            block_size = random.randint(2, max(3, int(n * 0.05)))
            start = random.randint(0, n - block_size); block = perm[start:start+block_size]
            del perm[start:start+block_size]; insert_pos = random.randint(0, len(perm))
            perm = perm[:insert_pos] + block + perm[insert_pos:]
            
    return perm + [t]

# -------------------------
# Main MOSA Solver
# -------------------------
def mosa_search(n: int, adj_list: List[Set[int]], adj_bits: List[int], config: Dict, seed_solutions: List[List[int]]):
    _init_worker(adj_bits, n)

    if seed_solutions:
        current = random.choice(seed_solutions)[:]
    else:
        perm = list(np.random.permutation(n))
        t = random.randint(int(n * 0.2), int(n * 0.8))
        current = perm + [t]

    current_score = evaluate_bitset_cached(tuple(current))
    archive = [{'solution': current[:], 'score': current_score}]
    best_hv = compute_hypervolume(archive, n)

    temp = config['initial_temp']
    step = 0
    last_added_step = 0

    pbar = tqdm(total=config['max_steps'], desc="🔥 MOSA", unit="iter")
    while step < config['max_steps'] and temp > config['final_temp']:
        for _ in range(config['steps_per_temp']):
            if step >= config['max_steps']: break
            step += 1
            pbar.update(1)

            neighbor = get_neighbor(current, n)
            neighbor_score = evaluate_bitset_cached(tuple(neighbor))

            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                ds = neighbor_score[0] - current_score[0]
                dw = current_score[1] - neighbor_score[1]
                delta = 2.0 * ds + dw 
                if delta >= 0:
                    accept = True
                else:
                    try:
                        prob = math.exp(delta / temp)
                        if random.random() < prob:
                            accept = True
                    except OverflowError:
                        pass

            if accept:
                current, current_score = neighbor, neighbor_score

            archive, added = update_archive(archive, {'solution': neighbor, 'score': neighbor_score}, config['archive_cap'])
            if added:
                last_added_step = step
                new_hv = compute_hypervolume(archive, n)
                if new_hv < best_hv:
                    best_hv = new_hv
                    pbar.set_postfix({'best_hv': f"{best_hv:.2f}", 'arch': len(archive)})
            
            if step - last_added_step > config['restart_patience']:
                current = random.choice(archive)['solution']
                current_score = evaluate_bitset_cached(tuple(current))
                last_added_step = step
                tqdm.write("🔄 Restarting search from a random archive member.")

        temp *= config['cooling_rate']

    pbar.close()
    return archive

# -------------------------
# Final Re-evaluation
# -------------------------
def final_re_evaluate_and_filter(archive: List[Dict], n: int, adj_bits: List[int], workers: int) -> List[Dict]:
    print("🔁 Running final exact re-evaluation (parallel)...")
    
    with multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=(adj_bits, n)) as pool:
        unique_solutions = {tuple(entry['solution']) for entry in archive}
        results = pool.map(eval_wrapper_for_pool, unique_solutions)

    final_pop = [{'solution': list(sol_tuple), 'score': score} for sol_tuple, score in results]
    
    final_archive, _ = update_archive([], {'solution': [], 'score': (0, 501)}, len(final_pop) + 1)
    for sol in final_pop:
        final_archive, _ = update_archive(final_archive, sol, len(final_pop) + 1)
        
    final_archive.sort(key=lambda x: (-x['score'][0], x['score'][1]))
    return final_archive

# -------------------------
# CLI & Main Execution
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MOSA solver for SPOC-3 torso decompositions")
    parser.add_argument("--problem", type=str, required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--data-dir", type=str, default=DEFAULTS['data_dir'])
    parser.add_argument("--max_steps", type=int, default=DEFAULTS['max_steps'])
    parser.add_argument("--steps_per_temp", type=int, default=DEFAULTS['steps_per_temp'])
    parser.add_argument("--initial_temp", type=float, default=DEFAULTS['initial_temp'])
    parser.add_argument("--final_temp", type=float, default=DEFAULTS['final_temp'])
    parser.add_argument("--cooling_rate", type=float, default=DEFAULTS['cooling_rate'])
    parser.add_argument("--workers", type=int, default=DEFAULTS['workers'])
    parser.add_argument("--archive_cap", type=int, default=DEFAULTS['archive_cap'])
    parser.add_argument("--seed_file", type=str, default=DEFAULTS['seed_file'])
    parser.add_argument("--restart_patience", type=int, default=DEFAULTS['restart_patience'])
    parser.add_argument("--relink_interval", type=int, default=DEFAULTS['relink_interval'])
    parser.add_argument("--intensify_interval", type=int, default=DEFAULTS['intensify_interval'])
    parser.add_argument("--intensify_iters", type=int, default=DEFAULTS['intensify_iters'])
    parser.add_argument("--log_every", type=int, default=DEFAULTS['log_every'])
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(42); np.random.seed(42)

    try:
        n, adj_list = load_graph_local(args.problem, data_dir=args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

    adj_bits = build_adj_bitsets(n, adj_list)
    
    seeds = []
    if args.seed_file:
        try:
            with open(args.seed_file, "r") as f:
                data = json.load(f)
            seeds = data.get("decisionVector", [])
            print(f"🌱 Loaded {len(seeds)} seed vectors from {args.seed_file}")
        except Exception as e:
            print(f"⚠️ Could not load seed file: {e}")

    # Use a dictionary for config to pass to mosa_search
    config = {k: getattr(args, k) for k in DEFAULTS}
    # Override final_temp if it was passed
    if args.final_temp is not None:
        config['final_temp'] = args.final_temp

    start = time.time()
    archive = mosa_search(n, adj_list, adj_bits, config, seed_solutions=seeds)
    print(f"\n🔬 MOSA exploration finished in {time.time() - start:.1f}s. Archive size={len(archive)}")
    
    final_solutions = final_re_evaluate_and_filter(archive, n, adj_bits, workers=args.workers)
    print(f"✅ Final non-dominated solutions: {len(final_solutions)}")

    final_hv = compute_hypervolume(final_solutions, n)
    print(f"🏆 Final Hypervolume Score: {final_hv:.2f}")

    # Write submission file
    outname = f"submission_{args.problem}.json"
    decs = [sol['solution'] for sol in final_solutions][:20]
    mapping = {"easy": "torso-easy", "medium": "torso-medium", "hard": "torso-hard"}
    submission = {
        "decisionVector": [[int(x) for x in vec] for vec in decs],
        "problem": mapping.get(args.problem, args.problem),
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(outname, "w") as f: json.dump(submission, f, indent=4)
    print(f"📄 Wrote {len(decs)} solutions to {outname}")
    print("Done.")

if __name__ == "__main__":
    main()
