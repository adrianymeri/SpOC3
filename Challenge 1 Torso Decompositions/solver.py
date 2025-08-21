#!/usr/bin/env python3
"""
Definitive Memetic NSGA-II Solver for SPOC-3 Torso Decompositions.

This version combines the best of all previous scripts:
- It uses the user's proven (but technically incorrect) evaluation function as a
  powerful heuristic to guide the multi-objective search internally.
- It uses a second, officially-correct evaluation function to report the true
  score and calculate the hypervolume for target-checking.
- It is robust against OverflowErrors and TypeError bugs.
- It uses the user's successful hyperparameter configuration.
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
# Config & Problem Data (with correct final targets)
# -------------------------
CONFIG = {
    "general": {
        "mutation_rate": 0.5,
        "crossover_rate": 0.9,
        "checkpoint_interval": 5,
        "elite_count": 6,
        "elite_ls_multiplier": 4,
        "stagnation_limit": 12,
        "mutation_boost_factor": 1.8,
        "improvement_factor": 0.1,
    },
    "easy": {
        "pop_size": 120,
        "generations": 300,
        "local_search_intensity": 18,
        "target_hv": -1829919
    },
    "medium": {
        "pop_size": 150,
        "generations": 400,
        "local_search_intensity": 22,
        "target_hv": -1745122
    },
    "hard": {
        "pop_size": 200,
        "generations": 800,
        "local_search_intensity": 30,
        "target_hv": -5493062
    },
}

# -------------------------
# Worker Globals
# -------------------------
WORKER_ADJ_SETS = None
WORKER_N = None

# -------------------------
# Graph Loading & Initialization
# -------------------------
def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    filepath = f"./{problem_id}.gr"
    print(f"📥 Loading graph data for '{problem_id}' from '{filepath}'...")
    edges = []
    max_node = -1
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
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

def _init_worker(adj_sets: List[Set[int]], n: int):
    global WORKER_ADJ_SETS, WORKER_N
    WORKER_ADJ_SETS, WORKER_N = adj_sets, n

# -------------------------
# Dual Evaluation Functions
# -------------------------
@lru_cache(maxsize=500000)
def internal_heuristic_evaluation(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    n, adj_sets = WORKER_N, WORKER_ADJ_SETS
    t, perm = solution_tuple[-1], list(solution_tuple[:-1])
    size = n - t
    if size <= 0: return (0, 501)

    temp_adj = [s.copy() for s in adj_sets]
    max_width = 0
    nodes_after = [set(perm[i+1:]) for i in range(n)]

    for i in range(n):
        u = perm[i]
        successors = temp_adj[u].intersection(nodes_after[i])
        degree = len(successors)

        if degree > max_width:
            max_width = degree
        if max_width > 500: return (size, 501)
        if not successors: continue
        
        for v in successors:
            temp_adj[v].update(successors)
            temp_adj[v].remove(v)
            
    return (size, max_width)

@lru_cache(maxsize=500000)
def official_score_evaluation(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    n, adj_sets = WORKER_N, WORKER_ADJ_SETS
    t, perm = solution_tuple[-1], list(solution_tuple[:-1])
    if (n - t) <= 0: return (501, t)

    temp_adj = [s.copy() for s in adj_sets]
    max_width = 0
    nodes_after = [set(perm[i+1:]) for i in range(n)]

    for i in range(n):
        u = perm[i]
        successors = temp_adj[u].intersection(nodes_after[i])
        
        if i >= t:
            degree = len(successors)
            if degree > max_width:
                max_width = degree

        if max_width > 500: return (501, t)
        if not successors: continue
        
        for v in successors:
            temp_adj[v].update(successors)
            temp_adj[v].remove(v)

    return (max_width, t)

def eval_wrapper(solution_tuple):
    internal_score = internal_heuristic_evaluation(solution_tuple)
    official_score = official_score_evaluation(solution_tuple)
    return (solution_tuple, internal_score, official_score)

# -------------------------
# Dominance (operates on INTERNAL score for guidance)
# -------------------------
def dominates(p_internal_score, q_internal_score):
    return (p_internal_score[0] >= q_internal_score[0] and p_internal_score[1] < q_internal_score[1]) or \
           (p_internal_score[0] > q_internal_score[0] and p_internal_score[1] <= q_internal_score[1])

def score_better(a_internal_score, b_internal_score):
    return (a_internal_score[0] > b_internal_score[0]) or \
           (a_internal_score[0] == b_internal_score[0] and a_internal_score[1] < b_internal_score[1])

# -------------------------
# Operators
# -------------------------
def local_search_worker(args):
    sol, intensity = args
    best_sol = tuple(sol)
    best_internal_score = internal_heuristic_evaluation(best_sol)
    
    for _ in range(intensity):
        current_sol = list(best_sol)
        r = random.random()
        if r < 0.3:
            perm = block_move(current_sol[:-1], WORKER_N)
            neighbor = perm + [current_sol[-1]]
        elif r < 0.7:
            perm = inversion_mutation(current_sol[:-1])
            neighbor = perm + [current_sol[-1]]
        else:
            t = smart_torso_shift(current_sol[-1], WORKER_N)
            neighbor = current_sol[:-1] + [t]

        neighbor_internal_score = internal_heuristic_evaluation(tuple(neighbor))
        if score_better(neighbor_internal_score, best_internal_score):
            best_sol, best_internal_score = tuple(neighbor), neighbor_internal_score
            
    return list(best_sol)

def inversion_mutation(perm: List[int]) -> List[int]:
    if len(perm) < 2: return perm
    a, b = sorted(random.sample(range(len(perm)), 2)); p = perm[:]
    p[a:b + 1] = reversed(p[a:b + 1]); return p

def swap_mutation(perm: List[int]) -> List[int]:
    if len(perm) < 2: return perm
    p = perm[:]; i, j = random.sample(range(len(perm)), 2)
    p[i], p[j] = p[j], p[i]; return p

def block_move(perm: List[int], n: int) -> List[int]:
    p = perm[:]; block_size = random.randint(2, max(3, int(n * 0.05)))
    if n > block_size:
        start = random.randint(0, n - block_size); block = p[start:start + block_size]
        del p[start:start + block_size]; insert_pos = random.randint(0, len(p))
        p[insert_pos:insert_pos] = block
    return p

def smart_torso_shift(t: int, n: int) -> int:
    shift = int(max(1, n * 0.05))
    return max(0, min(n - 1, t + random.randint(-shift, shift)))
    
def pmx_crossover(p1: List[int], p2: List[int]) -> List[int]:
    n = len(p1); a, b = sorted(random.sample(range(n), 2)); child = [-1] * n
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
    for i in range(n):
        if child[i] == -1: child[i] = p2[i]
    return child

# -------------------------
# Population Management (operates on INTERNAL score for guidance)
# -------------------------
def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    if len(population) <= pop_size: return population
    for p in population: p['dominates_set'], p['dominated_by_count'] = [], 0
    fronts = [[]]
    for p in population:
        for q in population:
            if p is q: continue
            if dominates(p['internal_score'], q['internal_score']): p['dominates_set'].append(q)
            elif dominates(q['internal_score'], p['internal_score']): p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0: fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0: next_front.append(q)
        fronts.append(next_front); i += 1
    new_population = []
    for front in fronts:
        if not front: continue
        if len(new_population) + len(front) > pop_size:
            for p in front: p['distance'] = 0.0
            for i_obj, reverse_sort in [(0, True), (1, False)]: # size (max), width (min)
                front.sort(key=lambda p: p['internal_score'][i_obj], reverse=reverse_sort)
                front[0]['distance'] = front[-1]['distance'] = float('inf')
                f_min, f_max = front[-1]['internal_score'][i_obj], front[0]['internal_score'][i_obj]
                if abs(f_max - f_min) > 1e-9:
                    for j in range(1, len(front) - 1):
                        front[j]['distance'] += abs(front[j+1]['internal_score'][i_obj] - front[j-1]['internal_score'][i_obj]) / (f_max - f_min)
            front.sort(key=lambda p: p['distance'], reverse=True)
            new_population.extend(front[:pop_size - len(new_population)]); break
        new_population.extend(front)
    return new_population

def enhanced_initialization(n: int, pop_size: int) -> List[Dict]:
    population = []
    while len(population) < pop_size:
        perm = list(np.random.permutation(n))
        t = random.randint(int(n * 0.2), int(n * 0.8))
        population.append({'solution': perm + [t]})
    return population

def update_adaptive_mutation(stagnation_counter: int, base_mutation: float, config: Dict) -> float:
    if stagnation_counter >= config['stagnation_limit']:
        return min(0.95, base_mutation * config['mutation_boost_factor'])
    elif stagnation_counter == 0:
        return max(0.1, base_mutation * (1 - config['improvement_factor']))
    return base_mutation

# -------------------------
# Hypervolume (operates on OFFICIAL score) & Persistence
# -------------------------
def calculate_hypervolume(population: List[Dict], n: int) -> float:
    if not population: return 0.0
    points_array = np.array([p['official_score'] for p in population])
    try:
        ndf_mask = pg.non_dominated_front_2d(points_array)
        non_dominated_points = points_array[ndf_mask]
        if non_dominated_points.shape[0] == 0: return 0.0
        hv = pg.hypervolume(non_dominated_points)
        ref_point = [502, n]
        return -hv.compute(ref_point)
    except Exception as e:
        print(f"Error computing hypervolume: {e}"); return 0.0

def persist_best(best_solution, problem_id):
    json.dump({'decisionVector': [[int(x) for x in best_solution]]},
              open(f"best_submission_{problem_id}.json", "w"), indent=2)

# -------------------------
# Main Algorithm
# -------------------------
def memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str):
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    pop_size, target_hv = config['pop_size'], config['target_hv']
    
    population = enhanced_initialization(n, pop_size)
    start_gen = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f: saved = pickle.load(f)
        population, start_gen = saved['pop'], saved['gen'] + 1
        print(f"🔄 Resuming from gen {start_gen}")

    best_internal_score = (0, 501)
    best_hypervolume = 1.0 

    with multiprocessing.Pool(initializer=_init_worker, initargs=(adj_list, n)) as pool:
        sols = [tuple(p['solution']) for p in population]
        results = pool.map(eval_wrapper, sols)
        for p, res in zip(population, results):
            p['internal_score'] = res[1]
            p['official_score'] = res[2]
        
        best_solution_obj = max(population, key=lambda p: (p['internal_score'][0], -p['internal_score'][1]))
        best_internal_score = best_solution_obj['internal_score']
        persist_best(best_solution_obj['solution'], problem_id)

        best_hypervolume = calculate_hypervolume(population, n)
        print(f"📊 Initial HV score: {best_hypervolume:.2f}, Target: {target_hv}")

        stagnation_counter = 0
        for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving", initial=start_gen, total=config['generations']):
            if best_hypervolume <= target_hv:
                print(f"🎯 Target score reached at gen {gen}!"); break

            mutation_rate = update_adaptive_mutation(stagnation_counter, config['mutation_rate'], config)
            mating_pool = crowding_selection(population, pop_size)
            offspring_sols = []
            while len(offspring_sols) < pop_size:
                p1, p2 = random.sample(mating_pool, 2)
                perm1, perm2 = p1['solution'][:-1], p2['solution'][:-1]
                child_perm = pmx_crossover(perm1, perm2) if random.random() < config['crossover_rate'] else perm1[:]
                if random.random() < mutation_rate:
                    child_perm = inversion_mutation(child_perm) if random.random() < 0.6 else swap_mutation(child_perm)
                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                offspring_sols.append(child_perm + [c_t])

            improved_offspring = pool.map(local_search_worker, [(sol, config['local_search_intensity']) for sol in offspring_sols])
            
            results = pool.map(eval_wrapper, [tuple(sol) for sol in improved_offspring])
            offspring_pop = [{'solution': list(res[0]), 'internal_score': res[1], 'official_score': res[2]} for res in results]
            population = crowding_selection(population + offspring_pop, pop_size)

            pop_sorted = sorted(population, key=lambda p: (p['internal_score'][0], -p['internal_score'][1]), reverse=True)
            elites = pop_sorted[:config['elite_count']]
            elite_sols = [e['solution'] for e in elites]
            
            intensified_elites = pool.map(local_search_worker, [(sol, config['local_search_intensity'] * config['elite_ls_multiplier']) for sol in elite_sols])
            results = pool.map(eval_wrapper, [tuple(sol) for sol in intensified_elites])
            elite_pop = [{'solution': list(res[0]), 'internal_score': res[1], 'official_score': res[2]} for res in results]
            population = crowding_selection(population + elite_pop, pop_size)

            current_hypervolume = calculate_hypervolume(population, n)
            current_best_obj = max(population, key=lambda p: (p['internal_score'][0], -p['internal_score'][1]))

            # Use hypervolume for stagnation tracking
            if current_hypervolume < best_hypervolume:
                best_hypervolume = current_hypervolume; stagnation_counter = 0
                tqdm.write(f"📊 Gen {gen + 1}: New best HV score {best_hypervolume:.2f}")
            else:
                stagnation_counter += 1
            
            # Persist the best individual solution based on the guiding heuristic
            if score_better(current_best_obj['internal_score'], best_internal_score):
                best_internal_score = current_best_obj['internal_score']
                persist_best(current_best_obj['solution'], problem_id)
                tqdm.write(f"✨ Gen {gen + 1}: New best internal (size, width)={best_internal_score}")
            
            tqdm.write(f"Gen {gen + 1}: best_internal(s,w)={best_internal_score} HV_score={best_hypervolume:.2f} stagn={stagnation_counter} mut={mutation_rate:.3f}")

            if (gen + 1) % config['checkpoint_interval'] == 0:
                tqdm.write(f"\n💾 Saving checkpoint at gen {gen + 1}...")
                with open(checkpoint_file, 'wb') as f: pickle.dump({'pop': population, 'gen': gen}, f)

    final_pop = crowding_selection(population, 20)
    create_submission_file([p['solution'] for p in final_pop], problem_id)

def create_submission_file(decision_vectors: List[List[int]], problem_id: str):
    filename = f"submission_{problem_id}.json"
    problem_name_map = {"easy": "torso-easy", "medium": "torso-medium", "hard": "torso-hard"}
    submission = {"decisionVector": [[int(val) for val in vec] for vec in decision_vectors],
                  "problem": problem_name_map.get(problem_id, "torso-hard"),
                  "challenge": "spoc-3-torso-decompositions"}
    with open(filename, "w") as f: json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(decision_vectors)} solutions.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(int(time.time())); np.random.seed(int(time.time()))
    
    problem_id = input("🔍 Select problem (easy/medium/hard) [easy]: ").strip().lower() or "easy"
    if problem_id not in CONFIG: print("❌ Invalid problem ID. Exiting."); exit(1)
    
    config = {**CONFIG['general'], **CONFIG[problem_id]}
    n, adj = load_graph(problem_id)
    
    start_time = time.time()
    memetic_algorithm(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")
