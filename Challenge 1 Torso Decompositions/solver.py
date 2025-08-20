#!/usr/bin/env python3
# This solver uses a compiled Cython module for high-performance evaluation.

import os
import sys
import time
import random
import pickle
import json
from typing import List, Set, Tuple, Dict
import numpy as np
from tqdm import tqdm
import multiprocessing
import urllib.request

# Import our newly compiled C code
import solver_cython

# CONFIGURATION
CONFIG = {
    "general": { "num_islands": os.cpu_count() or 4, "pop_size_per_island": 50, },
    "easy": { "generations": 500 },
    "medium": { "generations": 1000 },
    "hard": { "generations": 1500 },
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker initialization for multiprocessing ---
WORKER_ADJ_BITS = None
WORKER_N = None

def init_worker_py(n, adj_bits):
    global WORKER_N, WORKER_ADJ_BITS
    WORKER_N = n
    WORKER_ADJ_BITS = adj_bits
    # IMPORTANT: Initialize the Cython module for this worker process
    solver_cython.init_worker_cython(n, adj_bits)

# --- Wrapper to call the fast Cython function ---
def eval_wrapper(solution_np):
    return (solution_np, solver_cython.evaluate_solution_cy(solution_np))

# --- Graph loading and other utilities (simplified) ---
def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    url = PROBLEMS.get(problem_id, problem_id)
    print(f"📥 Loading graph data for '{problem_id}' from {url} ...")
    edges = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))
    max_node = max(max(u, v) for u, v in edges)
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, adj

def build_adj_bitsets_np(n: int, adj_list: List[Set[int]]) -> np.ndarray:
    adj_bits = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        val = np.uint64(0)
        for v in adj_list[u]:
            val |= (np.uint64(1) << np.uint64(v))
        adj_bits[u] = val
    return adj_bits

def score_better(a, b):
    return (a[0] > b[0]) or (a[0] == b[0] and a[1] < b[1])

def create_submission_file(solution, problem_id):
    filename = f"submission_{problem_id}.json"
    vector = [int(v) for v in solution]
    with open(filename, "w") as f:
        json.dump({"decisionVector": [vector], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
    print(f"📄 Created submission file: {filename}")

# --- Main algorithm (simplified for clarity) ---
def simple_ga(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str):
    adj_bits_np = build_adj_bitsets_np(n, adj_list)
    pop_size = config['pop_size_per_island']
    generations = config['generations']

    print(f"🧬 Initializing population of size {pop_size}...")
    population = []
    base_perm = np.arange(n, dtype=np.int64)
    for _ in range(pop_size):
        np.random.shuffle(base_perm)
        t = np.random.randint(int(n * 0.1), int(n * 0.9))
        population.append({'solution': np.append(base_perm.copy(), t)})

    with multiprocessing.Pool(initializer=init_worker_py, initargs=(n, adj_bits_np)) as pool:
        # Initial evaluation
        results = pool.map(eval_wrapper, [p['solution'] for p in population])
        for sol, score in results:
            for p in population:
                if np.array_equal(p['solution'], sol):
                    p['score'] = score
                    break
        
        global_best_score = (-1, 999)
        global_best_solution = None

        pbar = tqdm(range(generations), desc="🚀 Evolving with Cython")
        for gen in pbar:
            # Simple tournament selection and evolution
            offspring = []
            for _ in range(pop_size):
                p1 = random.choice(population)
                p2 = random.choice(population)
                parent = p1 if score_better(p1['score'], p2['score']) else p2
                
                # Mutate
                new_sol = parent['solution'].copy()
                perm = new_sol[:-1]
                i, j = np.random.choice(n, 2, replace=False)
                perm[i], perm[j] = perm[j], perm[i]
                new_sol[:-1] = perm
                offspring.append(new_sol)

            # Evaluate offspring
            results = pool.map(eval_wrapper, offspring)
            offspring_pop = [{'solution': sol, 'score': score} for sol, score in results]

            # Combine and select the best
            combined_pop = sorted(population + offspring_pop, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
            population = combined_pop[:pop_size]

            if score_better(population[0]['score'], global_best_score):
                global_best_score = population[0]['score']
                global_best_solution = population[0]['solution']
                pbar.write(f"✨ Gen {gen+1}: New global best {global_best_score}")

            pbar.set_postfix({"best_score": global_best_score})
    
    return global_best_solution

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(42)
    np.random.seed(42)
    
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS:
        sys.exit("❌ Invalid problem ID. Exiting.")
    
    run_config = CONFIG['general'].copy()
    run_config.update(CONFIG[problem_id])
    
    n, adj = load_graph(problem_id)
    
    start_time = time.time()
    best_solution = simple_ga(n, adj, run_config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")
    
    if best_solution is not None:
        create_submission_file(best_solution, problem_id)
