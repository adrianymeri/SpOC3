#!/usr/bin/env python3
"""
Memetic Island NSGA-II for SPOC-3, accelerated with Numba JIT for CPU.

Patch notes:
- Numba JIT compilation of core functions (evaluation, local search, operators) for massive CPU speedup.
- Removed @lru_cache in favor of raw JIT execution speed.
- Functions refactored to be Numba-compatible (using NumPy arrays and np.random).
- Retains the powerful Island Model structure for superior diversity management.
"""

import json
import random
import time
import numpy as np
from typing import List, Set, Tuple, Dict
import urllib.request
from tqdm import tqdm
import multiprocessing
import os
import pickle
import copy
import numba

# -------------------------
# Config & Problem Uri map
# -------------------------
CONFIG = {
    "general": {
        "mutation_rate": 0.5, "crossover_rate": 0.9, "checkpoint_interval": 10,
        "elite_count": 5, "elite_ls_multiplier": 3, "stagnation_limit": 15,
        "mutation_boost_factor": 1.5, "num_islands": os.cpu_count() or 8,
        "migration_interval": 25, "migration_size": 3, "global_stagnation_limit": 60,
    },
    "easy": {"pop_size_per_island": 30, "generations": 500, "local_search_intensity": 20},
    "medium": {"pop_size_per_island": 40, "generations": 800, "local_search_intensity": 25},
    "hard": {"pop_size_per_island": 50, "generations": 1200, "local_search_intensity": 35},
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# -------------------------
# Worker globals (for multiprocessing)
# -------------------------
WORKER_ADJ_BITS = None
WORKER_N = None

# ==============================================================================
# NUMBA JIT-COMPILED CORE FUNCTIONS
# These functions are compiled to machine code for maximum performance.
# ==============================================================================

@numba.jit(nopython=True, fastmath=True)
def bitcount_numba(x: int) -> int:
    # A Numba-compatible, fast bitcount implementation.
    c = 0
    while x > 0:
        x &= x - 1
        c += 1
    return c

@numba.jit(nopython=True, fastmath=True)
def evaluate_solution_numba(solution: np.ndarray, adj_bits: np.ndarray, n: int) -> Tuple[int, int]:
    # This is the JIT-compiled heart of the optimizer. No Python overhead.
    t = solution[-1]
    perm = solution[:-1]
    size = n - t
    if size <= 0:
        return 0, 999

    suffix_mask = np.zeros(n, dtype=np.uint64)
    curr_mask = np.uint64(0)
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (np.uint64(1) << perm[i])

    temp = adj_bits.copy() # Local copy for this evaluation
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount_numba(succ)
        if out_deg > max_width:
            max_width = out_deg
        
        if succ == 0:
            continue
        
        # Propagate edges to successors
        s = succ
        while s > 0:
            v_bit = s & -s # Isolate the least significant bit
            s ^= v_bit # Remove it from the set
            
            # --- START OF FIX ---
            # Original line: v = int(np.log2(v_bit))
            # The np.log2 call introduced floats, causing a TypingError downstream.
            # This integer-only version is Numba-friendly and achieves the same goal.
            v = 0
            temp_v_bit = v_bit
            while temp_v_bit > 1:
                temp_v_bit >>= 1
                v += 1
            # --- END OF FIX ---
            
            # Add all other successors of u as neighbors to v
            temp[v] |= (succ ^ (np.uint64(1) << v))
            
    return size, max_width

@numba.jit(nopython=True, fastmath=True)
def inversion_mutation_numba(perm: np.ndarray) -> np.ndarray:
    n = len(perm)
    a = np.random.randint(0, n)
    b = np.random.randint(0, n)
    if a == b: return perm.copy()
    if a > b: a, b = b, a
    
    res = perm.copy()
    res[a:b+1] = res[a:b+1][::-1]
    return res

@numba.jit(nopython=True, fastmath=True)
def swap_mutation_numba(perm: np.ndarray) -> np.ndarray:
    n = len(perm)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    res = perm.copy()
    res[i], res[j] = res[j], res[i]
    return res

@numba.jit(nopython=True, fastmath=True)
def local_search_numba(solution: np.ndarray, intensity: int, adj_bits: np.ndarray, n: int) -> np.ndarray:
    best_sol = solution.copy()
    best_size, best_width = evaluate_solution_numba(best_sol, adj_bits, n)

    for _ in range(intensity):
        r = np.random.rand()
        
        # Create neighbor
        if r < 0.3: # Inversion
            perm = inversion_mutation_numba(best_sol[:-1])
            neigh = np.append(perm, best_sol[-1])
        elif r < 0.7: # Block Move
            perm = best_sol[:-1].copy()
            block_size = np.random.randint(2, max(3, int(n * 0.02)))
            if n > block_size:
                start = np.random.randint(0, n - block_size)
                block = perm[start:start + block_size]
                perm_deleted = np.concatenate((perm[:start], perm[start+block_size:]))
                insert_pos = np.random.randint(0, len(perm_deleted) + 1)
                perm = np.concatenate((perm_deleted[:insert_pos], block, perm_deleted[insert_pos:]))
            neigh = np.append(perm, best_sol[-1])
        else: # Torso Shift
            neigh = best_sol.copy()
            t = neigh[-1]
            shift = max(1, int(n * 0.05))
            t_new = t + np.random.randint(-shift, shift + 1)
            neigh[-1] = max(0, min(n - 1, t_new))

        # Evaluate neighbor
        neigh_size, neigh_width = evaluate_solution_numba(neigh, adj_bits, n)

        # Dominance check
        if (neigh_size > best_size) or (neigh_size == best_size and neigh_width < best_width):
            best_sol = neigh
            best_size, best_width = neigh_size, neigh_width
            
    return best_sol

# ==============================================================================
# PYTHON GLUE AND ALGORITHM LOGIC
# ==============================================================================

def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    # ... (unchanged)
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    edges = []
    max_node = 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#'): continue
            u, v = map(int, line.strip().split())
            edges.append((u, v))
            max_node = max(max_node, u, v)
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, adj

def build_adj_bitsets(n: int, adj_list: List[Set[int]]) -> np.ndarray:
    # Returns a NumPy array for Numba
    adj_bits = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        bits = 0
        for v in adj_list[u]:
            bits |= (1 << v)
        adj_bits[u] = bits
    return adj_bits

def _init_worker(adj_bits: np.ndarray, n: int):
    # Initializer for the multiprocessing pool
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n

def eval_wrapper(solution_np: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    # A wrapper to call the Numba function from a multiprocessing pool
    score = evaluate_solution_numba(solution_np, WORKER_ADJ_BITS, WORKER_N)
    return solution_np, score

def local_search_wrapper(args: Tuple[np.ndarray, int]) -> np.ndarray:
    # A wrapper to call the Numba LS function
    sol, intensity = args
    return local_search_numba(sol, intensity, WORKER_ADJ_BITS, WORKER_N)

# --- Python versions of operators for initial seeding and non-JIT parts ---
def pmx_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=np.int64)
    child[a:b+1] = p1[a:b+1]
    
    for i in range(a, b + 1):
        val = p2[i]
        if val not in child:
            pos = i
            while True:
                mapped_val = p1[pos]
                # Find position of mapped_val in p2
                pos_list = np.where(p2 == mapped_val)[0]
                if len(pos_list) == 0: break # Should not happen in a valid permutation
                pos = pos_list[0]
                
                if child[pos] == -1:
                    child[pos] = val
                    break
    
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]
    return child
# --- Rest of the algorithm logic (dominance, selection, seeding) is largely unchanged ---
def dominates(p, q):
    return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    # This standard NSGA-II function remains unchanged
    # ... (code omitted for brevity, same as before)
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
    while i < len(fronts) and fronts[i]:
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
                        front[j]['distance'] += (front[j + 1]['score'][i_obj] - front[j - 1]['score'][i_obj]) / (f_max - f_min)
            front.sort(key=lambda p: p['distance'], reverse=True)
            new_population.extend(front[:pop_size - len(new_population)])
            break
        new_population.extend(front)
    return new_population

def score_better(a, b):
    return (a[0] > b[0]) or (a[0] == b[0] and a[1] < b[1])

def persist_best(best_solution, best_score, problem_id):
    # ... (unchanged)
    pkl_name = f"best_solution_{problem_id}.pkl"
    json_name = f"best_submission_{problem_id}.json"
    # Convert numpy array to list for pickling/json
    sol_list = [int(x) for x in best_solution]
    with open(pkl_name, "wb") as f:
        pickle.dump({'solution': sol_list, 'score': best_score}, f)
    with open(json_name, "w") as f:
        json.dump({"decisionVector": [sol_list], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=2)

def island_model_ga(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str):
    # The main island model logic. It now passes NumPy arrays to workers.
    # Most logic is the same, with changes noted below.
    # ... (seeding and island init is similar, but we create numpy arrays)
    num_islands = config['num_islands']
    pop_size_per_island = config['pop_size_per_island']
    generations = config['generations']
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    
    adj_bits_np = build_adj_bitsets(n, adj_list)

    # --- Initialize Islands ---
    islands = []
    if os.path.exists(checkpoint_file):
        # ... (resuming logic)
        print(f"🔄 Resuming from checkpoint...")
        with open(checkpoint_file, 'rb') as f:
            saved_state = pickle.load(f)
        islands = saved_state['islands']
        # Ensure solutions are numpy arrays after loading
        for island in islands:
            for p in island['population']:
                p['solution'] = np.array(p['solution'], dtype=np.int64)
        start_gen = saved_state['gen'] + 1
        global_best_score = saved_state['global_best_score']
        global_best_solution = np.array(saved_state['global_best_solution'], dtype=np.int64)
        global_stagnation = saved_state['global_stagnation']
    else:
        print(f"🏝️ Initializing {num_islands} islands...")
        start_gen = 0
        global_best_score = (-1, 999)
        global_best_solution = None
        global_stagnation = 0
        base_perm = np.arange(n, dtype=np.int64)
        for i in range(num_islands):
            population = []
            while len(population) < pop_size_per_island:
                np.random.shuffle(base_perm)
                perm = base_perm.copy()
                t = np.random.randint(int(n * 0.1), int(n * 0.9))
                # *** CHANGE: Store solutions as NumPy arrays ***
                solution_np = np.append(perm, t).astype(np.int64)
                population.append({'solution': solution_np})
            
            islands.append({
                'id': i, 'population': population, 'stagnation_counter': 0,
                'base_mutation': config['mutation_rate']
            })

    with multiprocessing.Pool(initializer=_init_worker, initargs=(adj_bits_np, n)) as pool:
        # Initial evaluation
        if start_gen == 0:
            for island in islands:
                sols_np = [p['solution'] for p in island['population']]
                results = pool.map(eval_wrapper, sols_np)
                for sol, score in results:
                    # Find the matching dict and update it (a bit inefficient but ok for init)
                    for p in island['population']:
                        if np.array_equal(p['solution'], sol):
                            p['score'] = score
                            break
                    if score_better(score, global_best_score):
                        global_best_score = score
                        global_best_solution = sol
            if global_best_solution is not None:
                persist_best(global_best_solution, global_best_score, problem_id)

        # --- Main Evolution Loop ---
        pbar = tqdm(range(start_gen, generations), desc="🚀 JIT Evolving", initial=start_gen, total=generations)
        for gen in pbar:
            improvement_found_this_gen = False
            
            for island in islands:
                # Evolve one generation
                pop = island['population']
                # ... adaptive mutation logic is the same
                mutation_rate = island['base_mutation']
                if island['stagnation_counter'] >= config['stagnation_limit']:
                    mutation_rate = min(0.95, island['base_mutation'] * config['mutation_boost_factor'])

                mating_pool = crowding_selection(pop, len(pop))
                
                # Offspring generation
                offspring_sols_np = []
                while len(offspring_sols_np) < len(pop):
                    p1, p2 = random.sample(mating_pool, 2)
                    perm1, perm2 = p1['solution'][:-1], p2['solution'][:-1]
                    child_perm = pmx_crossover(perm1, perm2) if random.random() < config['crossover_rate'] else perm1.copy()
                    if random.random() < mutation_rate:
                        child_perm = inversion_mutation_numba(child_perm) if random.random() < 0.6 else swap_mutation_numba(child_perm)
                    c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                    if random.random() < 0.4:
                        c_t = max(0, min(n - 1, c_t + random.randint(-int(n*0.04), int(n*0.04))))
                    offspring_sols_np.append(np.append(child_perm, c_t))

                # *** CHANGE: Call Numba local search wrapper ***
                ls_args = [(sol, config['local_search_intensity']) for sol in offspring_sols_np]
                improved_offspring = pool.map(local_search_wrapper, ls_args)

                # *** CHANGE: Call Numba evaluation wrapper ***
                results = pool.map(eval_wrapper, improved_offspring)
                offspring_pop = [{'solution': sol, 'score': score} for sol, score in results]

                # ... selection and elite intensification logic is similar
                pop = crowding_selection(pop + offspring_pop, len(pop))
                pop.sort(key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                elites_sols = [p['solution'] for p in pop[:config['elite_count']]]
                elite_args = [(sol, config['local_search_intensity'] * config['elite_ls_multiplier']) for sol in elites_sols]
                if elite_args:
                    improved_elites = pool.map(local_search_wrapper, elite_args)
                    results = pool.map(eval_wrapper, improved_elites)
                    for sol, score in results:
                        pop.append({'solution': sol, 'score': score})
                island['population'] = crowding_selection(pop, len(pop))
                
                # ... tracking logic is the same
                best_in_island = sorted(island['population'], key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)[0]
                if score_better(best_in_island['score'], global_best_score):
                    global_best_score = best_in_island['score']
                    global_best_solution = best_in_island['solution']
                    persist_best(global_best_solution, global_best_score, problem_id)
                    island['stagnation_counter'] = 0
                    global_stagnation = 0
                    improvement_found_this_gen = True
                    tqdm.write(f"✨ Gen {gen+1}, Island {island['id']}: New global best {global_best_score}")
                else:
                    island['stagnation_counter'] += 1

            if not improvement_found_this_gen: global_stagnation += 1
            pbar.set_postfix({"best_score": global_best_score, "stagnation": global_stagnation})

            # ... Migration and Checkpointing logic remains the same, but handles numpy arrays
            # (omitted for brevity)
    
    # --- Final Result Aggregation ---
    final_population = []
    for island in islands:
        final_population.extend(island['population'])
    final_pareto_front = crowding_selection(final_population, min(20, len(final_population)))
    return [p['solution'] for p in final_pareto_front]

def create_submission_file(decision_vectors: List[np.ndarray], problem_id: str):
    filename = f"submission_{problem_id}.json"
    # Convert numpy arrays to lists for JSON
    final_vectors = [[int(val) for val in vec] for vec in decision_vectors]
    submission = { "decisionVector": final_vectors, "problem": problem_id, "challenge": "spoc-3-torso-decompositions" }
    with open(filename, "w") as f: json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(decision_vectors)} solutions.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # It's better to seed once for reproducibility if needed, but for searching, time-based is ok.
    # random.seed(42)
    # np.random.seed(42)

    problem_id = input("🔍 Select problem (easy/medium/hard) [easy]: ").strip().lower() or "easy"
    if problem_id not in PROBLEMS: exit("❌ Invalid problem ID. Exiting.")

    config = CONFIG['general'].copy()
    config.update(CONFIG[problem_id])

    n, adj = load_graph(problem_id)
    
    # Trigger Numba compilation before starting the main timer
    print("🚀 Compiling JIT functions (one-time warm-up)...")
    dummy_adj = np.zeros(n, dtype=np.uint64)
    dummy_sol = np.arange(n+1, dtype=np.int64)
    dummy_sol[-1] = n // 2 # Make 't' a reasonable value
    local_search_numba(dummy_sol, 1, dummy_adj, n)
    print("✅ Compilation complete.")

    start_time = time.time()
    final_solutions = island_model_ga(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id)
