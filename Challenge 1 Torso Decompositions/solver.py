#!/usr/bin/env python3
# This solver uses a compiled Cython module for a high-performance memetic algorithm.

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
# ==============================================================================
CONFIG = {
    "general": {
        "mutation_rate": 0.5, "crossover_rate": 0.9, "num_islands": os.cpu_count() or 4,
    },
    "easy": {"pop_size_per_island": 40, "generations": 100, "local_search_intensity": 20},
    "medium": {"pop_size_per_island": 50, "generations": 200, "local_search_intensity": 25},
    "hard": {"pop_size_per_island": 60, "generations": 300, "local_search_intensity": 30},
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker initialization for multiprocessing ---
def init_worker_py(n, adj_bits):
    solver_cython.init_worker_cython(n, adj_bits)

# --- Wrappers to call the fast Cython functions ---
def eval_wrapper(solution_np):
    return (solution_np, solver_cython.evaluate_solution_cy(solution_np))

def local_search_wrapper(args):
    return solver_cython.local_search_cy(*args)

# --- Graph loading and other utilities ---
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

# --- Crossover and NSGA-II Selection ---
def pmx_crossover_py(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=np.int64)
    child[a:b+1] = p1[a:b+1]
    for i in range(a, b + 1):
        val = p2[i]
        if val not in child:
            pos = i
            while True:
                mapped = p1[pos]
                pos = np.where(p2 == mapped)[0][0]
                if child[pos] == -1:
                    child[pos] = val
                    break
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]
    return child

def dominates(p_score, q_score):
    return (p_score[0] >= q_score[0] and p_score[1] < q_score[1]) or \
           (p_score[0] > q_score[0] and p_score[1] <= q_score[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    for p in population: p['dominates_set'], p['dominated_by_count'] = [], 0
    fronts = [[]]
    for p in population:
        for q in population:
            if p is q: continue
            if dominates(p['score'], q['score']): p['dominates_set'].append(q)
            elif dominates(q['score'], p['score']): p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0: fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0: next_front.append(q)
        fronts.append(next_front)
        i += 1
    new_population = []
    for front in fronts:
        if not front: continue
        if len(new_population) + len(front) > pop_size:
            for p in front: p['distance'] = 0.0
            for i_obj in range(2):
                front.sort(key=lambda p: p['score'][i_obj])
                front[0]['distance'] = front[-1]['distance'] = float('inf')
                f_min, f_max = front[0]['score'][i_obj], front[-1]['score'][i_obj]
                if f_max > f_min:
                    for j in range(1, len(front) - 1):
                        front[j]['distance'] += (front[j+1]['score'][i_obj] - front[j-1]['score'][i_obj]) / (f_max - f_min)
            front.sort(key=lambda p: p['distance'], reverse=True)
            new_population.extend(front[:pop_size - len(new_population)])
            break
        new_population.extend(front)
    return new_population

def score_better(a, b):
    return (a[0] > b[0]) or (a[0] == b[0] and a[1] < b[1])

def create_submission_file(solutions, problem_id):
    filename = f"submission_{problem_id}.json"
    vectors = [[int(v) for v in sol] for sol in solutions]
    with open(filename, "w") as f:
        json.dump({"decisionVector": vectors, "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(vectors)} solutions.")

# --- Main Memetic Algorithm ---
def memetic_algorithm_islands(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str):
    adj_bits_np = build_adj_bitsets_np(n, adj_list)
    pop_size = config['pop_size_per_island']
    generations = config['generations']
    ls_intensity = config['local_search_intensity']

    print(f"🧬 Initializing population of size {pop_size}...")
    population = []
    base_perm = np.arange(n, dtype=np.int64)
    for _ in range(pop_size):
        np.random.shuffle(base_perm)
        t = np.random.randint(int(n * 0.1), int(n * 0.9))
        population.append({'solution': np.append(base_perm.copy(), t)})

    with multiprocessing.Pool(initializer=init_worker_py, initargs=(n, adj_bits_np)) as pool:
        results = pool.map(eval_wrapper, [p['solution'] for p in population])
        for sol, score in results:
            for p in population:
                if np.array_equal(p['solution'], sol): p['score'] = score; break
        
        global_best_score = (-1, 999)

        pbar = tqdm(range(generations), desc="🚀 Evolving with Cython (Memetic)")
        for gen in pbar:
            mating_pool = crowding_selection(population, len(population))
            offspring_sols = []
            while len(offspring_sols) < len(population):
                p1, p2 = random.sample(mating_pool, 2)
                perm1, perm2 = p1['solution'][:-1], p2['solution'][:-1]
                if random.random() < config['general']['crossover_rate']:
                    child_perm = pmx_crossover_py(perm1, perm2)
                else:
                    child_perm = perm1.copy()
                
                if random.random() < config['general']['mutation_rate']:
                    i, j = np.random.choice(n, 2, replace=False)
                    child_perm[i], child_perm[j] = child_perm[j], child_perm[i]
                
                t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                offspring_sols.append(np.append(child_perm, t))

            ls_args = [(sol, ls_intensity) for sol in offspring_sols]
            improved_offspring = pool.map(local_search_wrapper, ls_args)
            
            results = pool.map(eval_wrapper, improved_offspring)
            offspring_pop = [{'solution': sol, 'score': score} for sol, score in results]

            population = crowding_selection(population + offspring_pop, len(population))

            current_best_score = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)[0]['score']
            if score_better(current_best_score, global_best_score):
                global_best_score = current_best_score
                pbar.write(f"✨ Gen {gen+1}: New best score in population {global_best_score}")

            pbar.set_postfix({"best_score": global_best_score})
    
    final_front = crowding_selection(population, 20)
    return [p['solution'] for p in final_front]

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(int(time.time()))
    np.random.seed(int(time.time()) % (2**32 - 1))
    
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS:
        sys.exit("❌ Invalid problem ID. Exiting.")
    
    run_config = CONFIG['general'].copy()
    run_config.update(CONFIG[problem_id])
    
    n, adj = load_graph(problem_id)
    
    start_time = time.time()
    best_solutions = memetic_algorithm_islands(n, adj, run_config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")
    
    if best_solutions:
        create_submission_file(best_solutions, problem_id)
