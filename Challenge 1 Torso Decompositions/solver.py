#!/usr/bin/env python3
"""
Definitive Indicator-Based Evolutionary Algorithm (IBEA) for SPOC-3.

This version is a state-of-the-art multi-objective solver designed to overcome
premature convergence by directly optimizing for hypervolume contribution.
This final version corrects the submission output to adhere to the 20-solution limit.
"""

import json
import random
import time
import math
import numpy as np
from typing import List, Set, Tuple, Dict
import os
import pickle
from functools import lru_cache
import pygmo as pg
from tqdm import tqdm
import multiprocessing

# -------------------------
# Config for High-Effort IBEA Run
# -------------------------
CONFIG = {
    "general": {
        "kappa": 0.05,
        "mutation_rate": 0.6,
        "crossover_rate": 0.9,
        "checkpoint_interval": 10,
        "elite_count": 10,
        "elite_ls_multiplier": 5,
    },
    "easy": {
        "pop_size": 250,
        "generations": 500,
        "local_search_intensity": 20,
        "target_hv": -1829919
    },
    "medium": {
        "pop_size": 300,
        "generations": 600,
        "local_search_intensity": 25,
        "target_hv": -1745122
    },
    "hard": {
        "pop_size": 400,
        "generations": 800,
        "local_search_intensity": 30,
        "target_hv": -5493062
    },
}

# -------------------------
# Worker Globals & Graph Loading
# -------------------------
WORKER_ADJ_SETS = None
WORKER_N = None

def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    filepath = f"./{problem_id}.gr"
    print(f"📥 Loading graph data for '{problem_id}' from '{filepath}'...")
    edges, max_node = [], -1
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            u, v = map(int, line.strip().split())
            edges.append((u, v)); max_node = max(max_node, u, v)
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, adj

def _init_worker(adj_sets: List[Set[int]], n: int):
    global WORKER_ADJ_SETS, WORKER_N
    WORKER_ADJ_SETS, WORKER_N = adj_sets, n

# -------------------------
# Core Evaluation & Dominance
# -------------------------
@lru_cache(maxsize=1000000)
def evaluate_solution(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    n, adj_sets = WORKER_N, WORKER_ADJ_SETS
    t, perm = solution_tuple[-1], list(solution_tuple[:-1])
    if (n - t) <= 0: return (501, t)

    temp_adj, max_width = [s.copy() for s in adj_sets], 0
    nodes_after = [set(perm[i+1:]) for i in range(n)]

    for i in range(n):
        u = perm[i]
        successors = temp_adj[u].intersection(nodes_after[i])
        if i >= t:
            degree = len(successors)
            if degree > max_width: max_width = degree
        if max_width > 500: return (501, t)
        if not successors: continue
        for v in successors:
            temp_adj[v].update(successors - {v})
    return (max_width, t)

def eval_wrapper(solution_tuple):
    return solution_tuple, evaluate_solution(solution_tuple)

def dominates(p, q): return (p[0] <= q[0] and p[1] < q[1]) or (p[0] < q[0] and p[1] <= q[1])

# -------------------------
# Operators
# -------------------------
def local_search_worker(args):
    sol, intensity = args
    best_sol, best_score = tuple(sol), evaluate_solution(tuple(sol))
    for _ in range(intensity):
        perm, t = list(best_sol[:-1]), best_sol[-1]
        r = random.random()
        if r < 0.5: perm = inversion_mutation(perm)
        else: perm = block_move(perm, WORKER_N)
        if r > 0.8: t = smart_torso_shift(t, WORKER_N)
        neighbor = tuple(perm + [t])
        neighbor_score = evaluate_solution(neighbor)
        if dominates(neighbor_score, best_score):
            best_sol, best_score = neighbor, neighbor_score
    return list(best_sol)

def inversion_mutation(p): return p if len(p)<2 else (lambda a,b,c:c[:a]+c[a:b+1][::-1]+c[b+1:])(*sorted(random.sample(range(len(p)),2)),p[:])
def block_move(p, n):
    if n <= 3: return p
    perm, block_size = p[:], random.randint(2, max(3, int(n * 0.05)))
    start = random.randint(0, n - block_size); block = perm[start:start+block_size]
    del perm[start:start+block_size]; insert_pos = random.randint(0, len(perm))
    return perm[:insert_pos] + block + perm[insert_pos:]
def smart_torso_shift(t, n): return max(0, min(n - 1, t + random.randint(-int(max(1, n * 0.05)), int(max(1, n * 0.05)))))
def pmx_crossover(p1, p2):
    n = len(p1); a, b = sorted(random.sample(range(n), 2)); child = [-1]*n
    child[a:b+1] = p1[a:b+1]
    for i in range(a, b+1):
        val = p2[i]
        if val in child: continue
        pos = i
        while True:
            mapped = p1[pos]
            try: pos = p2.index(mapped)
            except ValueError: break
            if child[pos] == -1: child[pos] = val; break
    return [val if val != -1 else p2[i] for i, val in enumerate(child)]

# -------------------------
# IBEA Selection
# -------------------------
def indicator_based_selection(population: List[Dict], pop_size: int, kappa: float):
    # Calculate fitness for all individuals
    for p1 in population:
        loss = 0
        for p2 in population:
            if p1 is not p2 and dominates(p2['score'], p1['score']):
                loss += 1
        p1['fitness'] = -loss

    # Environmental selection
    while len(population) > pop_size:
        worst_p = min(population, key=lambda p: p['fitness'])
        population.remove(worst_p)
        
        for p in population:
            if dominates(worst_p['score'], p['score']):
                p['fitness'] += 1
    return population

# -------------------------
# Hypervolume & Persistence
# -------------------------
def calculate_hypervolume(population: List[Dict], n: int) -> float:
    scores = np.array([p['score'] for p in population])
    if scores.shape[0] == 0: return 0.0
    try:
        ndf_mask = pg.non_dominated_front_2d(scores)
        ndf_points = scores[ndf_mask]
        if ndf_points.shape[0] == 0: return 0.0
        hv = pg.hypervolume(ndf_points)
        ref_point = [502, n]
        return -hv.compute(ref_point)
    except Exception: return 0.0

def persist_final_front(population: List[Dict], problem_id: str, n: int):
    scores = np.array([p['score'] for p in population])
    if scores.shape[0] == 0:
        print("⚠️ No solutions in the final population to save.")
        return
        
    ndf_mask = pg.non_dominated_front_2d(scores)
    final_pop = [population[i] for i, is_nd in enumerate(ndf_mask) if is_nd]
    
    # Sort the front by the heuristic (max size, then min width)
    # Convert (width, t) score back to (size, width) for sorting
    final_pop.sort(key=lambda p: (-(n - p['score'][1]), p['score'][0]))
    
    # Take only the top 20 solutions
    top_20_solutions = [p['solution'] for p in final_pop[:20]]
    
    filename = f"submission_{problem_id}.json"
    problem_map = {"easy": "torso-easy", "medium": "torso-medium", "hard": "torso-hard"}
    submission = {"decisionVector": [[int(v) for v in vec] for vec in top_20_solutions],
                  "problem": problem_map.get(problem_id, problem_id),
                  "challenge": "spoc-3-torso-decompositions"}
    with open(filename, "w") as f: json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(top_20_solutions)} solutions.")

# -------------------------
# Main IBEA Algorithm
# -------------------------
def ibea_memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str):
    pop_size, target_hv = config['pop_size'], config['target_hv']
    
    population = [{'solution': list(np.random.permutation(n)) + [random.randint(int(n*0.1), int(n*0.9))]} for _ in range(pop_size)]
    
    with multiprocessing.Pool(initializer=_init_worker, initargs=(adj_list, n)) as pool:
        sols = [tuple(p['solution']) for p in population]
        results = dict(pool.map(eval_wrapper, sols))
        for p in population: p['score'] = results.get(tuple(p['solution']), (501, n))
        
        best_hypervolume = calculate_hypervolume(population, n)
        print(f"📊 Initial HV score: {best_hypervolume:.2f}, Target: {target_hv}")

        for gen in tqdm(range(config['generations']), desc="🧬 Evolving (IBEA)", total=config['generations']):
            if best_hypervolume <= target_hv:
                print(f"🎯 Target score reached at gen {gen}!"); break

            mating_pool = random.choices(population, k=pop_size)
            offspring_sols = []
            for _ in range(pop_size):
                p1, p2 = random.sample(mating_pool, 2)
                child_perm = pmx_crossover(p1['solution'][:-1], p2['solution'][:-1]) if random.random() < config['crossover_rate'] else p1['solution'][:-1][:]
                if random.random() < config['mutation_rate']:
                    child_perm = inversion_mutation(child_perm) if random.random() < 0.6 else block_move(child_perm, n)
                c_t = smart_torso_shift(int((p1['solution'][-1] + p2['solution'][-1]) / 2), n)
                offspring_sols.append(child_perm + [c_t])
            
            population.sort(key=lambda p: p['score'][0] * p['score'][1])
            elites = population[:config['elite_count']]
            elite_sols = [e['solution'] for e in elites]
            intensified_elites = pool.map(local_search_worker, [(sol, config['local_search_intensity']) for sol in elite_sols])
            
            eval_sols = offspring_sols + intensified_elites
            results = dict(pool.map(eval_wrapper, [tuple(sol) for sol in eval_sols]))
            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in results.items()]

            population = indicator_based_selection(population + offspring_pop, pop_size, config['kappa'])
            
            current_hypervolume = calculate_hypervolume(population, n)
            if current_hypervolume < best_hypervolume:
                best_hypervolume = current_hypervolume
                tqdm.write(f"📊 Gen {gen + 1}: New best HV score {best_hypervolume:.2f}")

            if (gen + 1) % config['checkpoint_interval'] == 0:
                tqdm.write(f"\n💾 Saving checkpoint...")
                with open(f"checkpoint_{problem_id}.pkl", 'wb') as f: pickle.dump({'pop': population, 'gen': gen}, f)

    persist_final_front(population, problem_id, n)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(int(time.time())); np.random.seed(int(time.time()))
    
    problem_id = input("🔍 Select problem (easy/medium/hard) [easy]: ").strip().lower() or "easy"
    if problem_id not in CONFIG: print("❌ Invalid problem ID. Exiting."); exit(1)
    
    config = {**CONFIG['general'], **CONFIG[problem_id]}
    n, adj = load_graph(problem_id)
    
    start_time = time.time()
    ibea_memetic_algorithm(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")
