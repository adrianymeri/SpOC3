#!/usr/bin/env python3
# This solver uses a compiled Cython module to run a high-performance memetic algorithm.

import os, sys, time, random, pickle, json, urllib.request
from typing import List, Set, Tuple, Dict
import numpy as np
from tqdm import tqdm
import multiprocessing
import solver_cython # Import our compiled C module

# CONFIGURATION
# ==============================================================================
CONFIG = {
    "general": {
        "mutation_rate": 0.5, "crossover_rate": 0.9, "checkpoint_interval": 5,
        "elite_count": 8, "elite_ls_multiplier": 5,
        "stagnation_limit": 15, "mutation_boost_factor": 1.5,
    },
    "easy": {"pop_size": 120, "generations": 800, "local_search_intensity": 20},
    "medium": {"pop_size": 150, "generations": 1200, "local_search_intensity": 25},
    "hard": {"pop_size": 200, "generations": 2000, "local_search_intensity": 35},
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker Initialization & Wrappers ---
def init_worker_py(n, adj_bits):
    solver_cython.init_worker_cython(n, adj_bits)

def eval_wrapper(solution_np):
    return (solution_np, solver_cython.evaluate_solution_cy(solution_np))

def local_search_wrapper(args):
    return solver_cython.local_search_cy(*args)

# --- Graph Loading & Bitset Building ---
def load_graph(problem_id: str):
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    edges, max_node = [], 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#'): continue
            u, v = map(int, line.strip().split())
            edges.append((u, v)); max_node = max(max_node, u, v)
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges: adj[u].add(v); adj[v].add(u)
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, adj

def build_adj_bitsets_np(n: int, adj_list: List[Set[int]]):
    adj_bits = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        for v in adj_list[u]: adj_bits[u] |= (np.uint64(1) << np.uint64(v))
    return adj_bits

# --- Genetic Operators (Python-side) ---
def pmx_crossover_py(p1: np.ndarray, p2: np.ndarray):
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
                if child[pos] == -1: child[pos] = val; break
    for i in range(n):
        if child[i] == -1: child[i] = p2[i]
    return child

# --- NSGA-II Selection ---
def dominates(p_score, q_score):
    return (p_score[0] >= q_score[0] and p_score[1] < q_score[1]) or \
           (p_score[0] > q_score[0] and p_score[1] <= q_score[1])

def crowding_selection(population: List[Dict], pop_size: int):
    # Standard NSGA-II crowding distance selection
    # (Code is standard and omitted here for brevity, but it's the same as your version)
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

# --- Main Memetic Algorithm ---
def memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str):
    adj_bits_np = build_adj_bitsets_np(n, adj_list)
    pop_size, generations = config['pop_size'], config['generations']
    base_mutation = config['mutation_rate']

    print(f"🧬 Initializing population of size {pop_size}...")
    population = []
    base_perm = np.arange(n, dtype=np.int64)
    for _ in range(pop_size):
        np.random.shuffle(base_perm)
        t = np.random.randint(int(n * 0.2), int(n * 0.8))
        population.append({'solution': np.append(base_perm.copy(), t)})

    best_score, stagnation_counter = (-1, 999), 0
    with multiprocessing.Pool(initializer=init_worker_py, initargs=(n, adj_bits_np)) as pool:
        # Initial evaluation
        results = pool.map(eval_wrapper, [p['solution'] for p in population])
        for sol, score in results:
            for p in population:
                if np.array_equal(p['solution'], sol): p['score'] = score; break

        pbar = tqdm(range(generations), desc="🚀 Evolving (Cython Memetic)")
        for gen in pbar:
            mutation_rate = base_mutation * (config['mutation_boost_factor'] if stagnation_counter >= config['stagnation_limit'] else 1.0)
            mating_pool = crowding_selection(population, pop_size)
            
            # --- Offspring Generation ---
            offspring_sols = []
            while len(offspring_sols) < pop_size:
                p1, p2 = random.sample(mating_pool, 2)
                perm1, perm2 = p1['solution'][:-1], p2['solution'][:-1]
                child_perm = pmx_crossover_py(perm1, perm2) if random.random() < config['crossover_rate'] else perm1.copy()
                if random.random() < mutation_rate:
                    i, j = np.random.choice(n, 2, replace=False) # Simple Swap
                    child_perm[i], child_perm[j] = child_perm[j], child_perm[i]
                t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                offspring_sols.append(np.append(child_perm, t))

            # --- Local Search (Memetic Step) ---
            ls_args = [(sol, config['local_search_intensity']) for sol in offspring_sols]
            improved_offspring = pool.map(local_search_wrapper, ls_args)

            # --- Evaluation & Selection ---
            results = pool.map(eval_wrapper, improved_offspring)
            offspring_pop = [{'solution': sol, 'score': score} for sol, score in results]
            population = crowding_selection(population + offspring_pop, pop_size)
            
            # --- Elite Intensification ---
            pop_sorted = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
            elite_sols = [p['solution'] for p in pop_sorted[:config['elite_count']]]
            elite_args = [(sol, config['local_search_intensity'] * config['elite_ls_multiplier']) for sol in elite_sols]
            if elite_args:
                intensified_elites = pool.map(local_search_wrapper, elite_args)
                results = pool.map(eval_wrapper, intensified_elites)
                for sol, score in results:
                    population.append({'solution': sol, 'score': score})
                population = crowding_selection(population, pop_size)
            
            # --- Tracking ---
            current_best_score = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)[0]['score']
            if (current_best_score[0] > best_score[0]) or (current_best_score[0] == best_score[0] and current_best_score[1] < best_score[1]):
                best_score = current_best_score
                stagnation_counter = 0
                pbar.write(f"✨ Gen {gen+1}: New best score in population {best_score}")
            else:
                stagnation_counter += 1
            pbar.set_postfix({"best_score": best_score, "stagn": stagnation_counter})
            
    final_front = crowding_selection(population, 20)
    return [p['solution'] for p in final_front]

# --- Entry Point ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS: sys.exit("❌ Invalid problem ID.")
    
    config = CONFIG['general'].copy(); config.update(CONFIG[problem_id])
    n, adj = load_graph(problem_id)
    
    start_time = time.time()
    best_solutions = memetic_algorithm(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")
    
    if best_solutions:
        # For submission, usually the single best solution is preferred
        final_evals = {tuple(sol): solver_cython.evaluate_solution_cy(sol) for sol in best_solutions}
        best_sol_tuple = max(final_evals.keys(), key=lambda k: (final_evals[k][0], -final_evals[k][1]))
        
        print(f"🏆 Final best solution score: {final_evals[best_sol_tuple]}")
        with open(f"submission_{problem_id}.json", "w") as f:
            json.dump({"decisionVector": [[int(v) for v in best_sol_tuple]], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
        print(f"📄 Created submission file: submission_{problem_id}.json")
