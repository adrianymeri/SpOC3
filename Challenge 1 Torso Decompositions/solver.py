#!/usr/bin/env python3
"""
solver.py
This is a high-performance conversion of the robust memetic NSGA-II solver.
It uses Numba JIT compilation for core functions and multiprocessing for parallelism.

Key changes:
- Core evaluation and local search functions are JIT-compiled with Numba for C-like speed.
- Python lists and tuples are replaced with NumPy arrays in performance-critical code.
- Python's `lru_cache` is removed; raw execution speed of JIT functions is faster.
- The island model's evolution loop is parallelized across all CPU cores.
"""

import os
import re
import sys
import time
import math
import random
import pickle
import json
from typing import List, Set, Tuple, Dict
import numpy as np
from tqdm import tqdm
import multiprocessing
import numba
import urllib.request

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "general": {
        "mutation_rate": 0.5,
        "crossover_rate": 0.9,
        "checkpoint_interval": 5,
        "elite_count": 5,
        "elite_ls_multiplier": 4,
        "stagnation_limit": 15,
        "mutation_boost_factor": 1.8,
        "num_islands": os.cpu_count() or 4, # Use all available cores
        "migration_interval": 20,
        "migration_size": 4,
    },
    "easy": {"pop_size_per_island": 40, "generations": 600, "local_search_intensity": 20},
    "medium": {"pop_size_per_island": 50, "generations": 1200, "local_search_intensity": 25},
    "hard": {"pop_size_per_island": 60, "generations": 2000, "local_search_intensity": 35},
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# ==============================================================================
# WORKER GLOBALS (for multiprocessing)
# ==============================================================================
WORKER_ADJ_BITS = None
WORKER_N = None

def _init_worker(adj_bits_np: np.ndarray, n: int):
    """Initializes globals for each worker process."""
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits_np
    WORKER_N = n

# ==============================================================================
# NUMBA JIT-COMPILED CORE FUNCTIONS (THE FAST PART)
# ==============================================================================

@numba.jit(nopython=True, fastmath=True)
def bitcount_numba(x: np.uint64) -> int:
    """A Numba-compatible, fast bitcount implementation."""
    c = 0
    while x > 0:
        x &= x - 1
        c += 1
    return c

@numba.jit(nopython=True, fastmath=True)
def evaluate_solution_numba(solution: np.ndarray) -> Tuple[int, int]:
    """
    JIT-compiled evaluator. Operates on NumPy arrays. No caching needed.
    This function is the heart of the optimizer.
    """
    n = WORKER_N
    adj_bits = WORKER_ADJ_BITS
    t = solution[-1]
    perm = solution[:-1]
    size = n - t
    if size <= 0:
        return (0, 999) # Using a high width for invalid solutions

    suffix_mask = np.zeros(n, dtype=np.uint64)
    curr_mask = np.uint64(0)
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (np.uint64(1) << perm[i])

    temp = adj_bits.copy()
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount_numba(succ)
        if out_deg > max_width:
            max_width = out_deg
        
        if succ == 0:
            continue
        
        s = succ
        while s > 0:
            v_bit = s & -s
            s ^= v_bit

            # THIS IS THE CRITICAL FIX that prevents the TypingError.
            # We use an integer-only loop to find the bit index instead of log2.
            v = 0
            temp_v_bit = v_bit
            while temp_v_bit > 1:
                temp_v_bit >>= 1
                v += 1
            
            temp[v] |= (succ ^ v_bit)
            
    return (size, max_width)

@numba.jit(nopython=True, fastmath=True)
def inversion_mutation_numba(perm: np.ndarray) -> np.ndarray:
    n = len(perm)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    res = perm.copy()
    res[a:b+1] = res[a:b+1][::-1]
    return res

@numba.jit(nopython=True, fastmath=True)
def swap_mutation_numba(perm: np.ndarray) -> np.ndarray:
    n = len(perm)
    i, j = np.random.choice(n, 2, replace=False)
    res = perm.copy()
    res[i], res[j] = res[j], res[i]
    return res

@numba.jit(nopython=True, fastmath=True)
def local_search_numba(solution: np.ndarray, intensity: int) -> np.ndarray:
    """JIT-compiled Variable Neighborhood Search."""
    n = WORKER_N
    best_sol = solution.copy()
    best_score = evaluate_solution_numba(best_sol)

    for _ in range(intensity):
        # Choose a neighborhood
        r = np.random.rand()
        perm = best_sol[:-1].copy()
        
        if r < 0.5: # Inversion
            new_perm = inversion_mutation_numba(perm)
        else: # Swap
            new_perm = swap_mutation_numba(perm)
        
        # Torso shift can also be applied
        t = best_sol[-1]
        if np.random.rand() < 0.3:
            shift = max(1, int(n * 0.05))
            t = max(0, min(n - 1, t + np.random.randint(-shift, shift + 1)))

        cand_sol = np.append(new_perm, t)
        cand_score = evaluate_solution_numba(cand_sol)

        # Dominance check
        if (cand_score[0] > best_score[0]) or \
           (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
            best_sol = cand_sol
            best_score = cand_score
            
    return best_sol

# ==============================================================================
# PYTHON GLUE AND ALGORITHM LOGIC
# ==============================================================================

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

# --- Wrappers for multiprocessing ---
def eval_wrapper(solution: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    return (solution, evaluate_solution_numba(solution))

def local_search_wrapper(args: Tuple[np.ndarray, int]) -> np.ndarray:
    solution, intensity = args
    return local_search_numba(solution, intensity)

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

# --- NSGA-II Selection (standard implementation) ---
def dominates(p_score, q_score):
    return (p_score[0] >= q_score[0] and p_score[1] < q_score[1]) or \
           (p_score[0] > q_score[0] and p_score[1] <= q_score[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    # This standard NSGA-II function remains unchanged
    for p in population:
        p['dominates_set'] = []
        p['dominated_by_count'] = 0
    fronts = [[]]
    for p in population:
        for q in population:
            if p is q: continue
            if dominates(p['score'], q['score']):
                p['dominates_set'].append(q)
            elif dominates(q['score'], p['score']):
                p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0:
                    next_front.append(q)
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

# --- Persistence and Submission ---
def score_better(a, b):
    return (a[0] > b[0]) or (a[0] == b[0] and a[1] < b[1])

def persist_best(solution, score, problem_id):
    sol_list = [int(x) for x in solution]
    with open(f"best_solution_{problem_id}.pkl", "wb") as f:
        pickle.dump({'solution': sol_list, 'score': score}, f)
    with open(f"best_submission_{problem_id}.json", "w") as f:
        json.dump({
            "decisionVector": [sol_list],
            "problem": problem_id,
            "challenge": "spoc-3-torso-decompositions"
        }, f, indent=2)

def create_submission_file(solutions, problem_id):
    filename = f"submission_{problem_id}.json"
    vectors = [[int(v) for v in sol] for sol in solutions]
    with open(filename, "w") as f:
        json.dump({
            "decisionVector": vectors,
            "problem": problem_id,
            "challenge": "spoc-3-torso-decompositions"
        }, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(vectors)} solutions.")

# ==============================================================================
# MAIN ALGORITHM
# ==============================================================================
def memetic_algorithm_islands(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str) -> List[np.ndarray]:
    # --- Setup ---
    adj_bits_np = build_adj_bitsets_np(n, adj_list)
    num_islands = config['general']['num_islands']
    pop_size = config['pop_size_per_island']
    generations = config['generations']
    ls_intensity = config['local_search_intensity']

    # --- Initialization ---
    islands = []
    print(f"🏝️ Initializing {num_islands} islands of size {pop_size}...")
    base_perm = np.arange(n, dtype=np.int64)
    for _ in range(num_islands):
        population = []
        for _ in range(pop_size):
            np.random.shuffle(base_perm)
            perm = base_perm.copy()
            t = np.random.randint(int(n * 0.1), int(n * 0.9))
            population.append({'solution': np.append(perm, t)})
        islands.append(population)
    
    global_best_score = (-1, 999)
    global_best_solution = None

    # --- Main Loop with Multiprocessing Pool ---
    with multiprocessing.Pool(initializer=_init_worker, initargs=(adj_bits_np, n)) as pool:
        # Initial evaluation
        for i, island_pop in enumerate(islands):
            solutions_np = [p['solution'] for p in island_pop]
            results = pool.map(eval_wrapper, solutions_np)
            for sol, score in results:
                for p in island_pop:
                    if np.array_equal(p['solution'], sol):
                        p['score'] = score
                        break
            islands[i] = island_pop

        for p in [p for island in islands for p in island]:
            if score_better(p['score'], global_best_score):
                global_best_score = p['score']
                global_best_solution = p['solution']
        if global_best_solution is not None:
            persist_best(global_best_solution, global_best_score, problem_id)

        pbar = tqdm(range(generations), desc="🚀 Evolving")
        for gen in pbar:
            # Evolve each island
            for i in range(num_islands):
                pop = islands[i]
                mating_pool = crowding_selection(pop, len(pop))
                
                # Create offspring
                offspring_sols = []
                while len(offspring_sols) < len(pop):
                    p1, p2 = random.sample(mating_pool, 2)
                    perm1, perm2 = p1['solution'][:-1], p2['solution'][:-1]
                    if random.random() < config['general']['crossover_rate']:
                        child_perm = pmx_crossover_py(perm1, perm2)
                    else:
                        child_perm = perm1.copy()
                    
                    if random.random() < config['general']['mutation_rate']:
                        child_perm = inversion_mutation_numba(child_perm) if random.random() < 0.5 else swap_mutation_numba(child_perm)
                    
                    t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                    offspring_sols.append(np.append(child_perm, t))

                # Apply local search
                ls_args = [(sol, ls_intensity) for sol in offspring_sols]
                improved_offspring = pool.map(local_search_wrapper, ls_args)
                
                # Evaluate new solutions
                results = pool.map(eval_wrapper, improved_offspring)
                offspring_pop = [{'solution': sol, 'score': score} for sol, score in results]

                # Select next generation
                islands[i] = crowding_selection(pop + offspring_pop, len(pop))

            # Update global best
            for p in [p for island in islands for p in island]:
                if score_better(p['score'], global_best_score):
                    global_best_score = p['score']
                    global_best_solution = p['solution']
                    persist_best(global_best_solution, global_best_score, problem_id)
                    pbar.write(f"✨ Gen {gen+1}: New global best {global_best_score}")

            pbar.set_postfix({"best_score": global_best_score})

    # --- Final Result ---
    final_population = [p for island in islands for p in island]
    final_front = crowding_selection(final_population, 20)
    return [p['solution'] for p in final_front]

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(42)
    np.random.seed(42)

    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower()
    if not problem_id: problem_id = "easy"
    if problem_id not in PROBLEMS:
        print("❌ Invalid problem ID. Exiting.")
        sys.exit(1)
        
    run_config = CONFIG['general'].copy()
    run_config.update(CONFIG[problem_id])

    n, adj = load_graph(problem_id)
    
    # One-time JIT compilation warm-up
    print("🚀 Compiling JIT functions (one-time warm-up)...")
    _init_worker(build_adj_bitsets_np(n, adj), n)
    dummy_sol = np.append(np.arange(n, dtype=np.int64), n // 2)
    local_search_numba(dummy_sol, 1)
    print("✅ Compilation complete.")

    start_time = time.time()
    final_solutions = memetic_algorithm_islands(n, adj, run_config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id)
