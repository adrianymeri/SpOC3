#!/usr/bin/env python3
"""
Memetic NSGA-II with elite intensification & adaptive mutation for SPOC-3 torso decompositions.

Patch notes:
- Elite intensification (parallel stronger LS on top E individuals every generation)
- Adaptive mutation (increases when no improvement)
- Larger per-process LRU cache for evaluations
- Best-solution logging and persistence
- Keeps fast bitset evaluator + pool initializer from previous version
"""

import json
import random
import time
import math
import numpy as np
from typing import List, Set, Tuple, Dict
import urllib.request
from tqdm import tqdm
import multiprocessing
import os
import pickle
from functools import lru_cache

# -------------------------
# Config & Problem Uri map
# -------------------------
CONFIG = {
    "general": {
        "mutation_rate": 0.5,
        "crossover_rate": 0.9,
        "checkpoint_interval": 5,
        # elite/intensification params
        "elite_count": 6,
        "elite_ls_multiplier": 4,
        # adaptive mutation
        "stagnation_limit": 12,  # gens without improvement before raising mutation
        "mutation_boost_factor": 1.8,
    },
    "easy": {"pop_size": 120, "generations": 300, "local_search_intensity": 18},
    "medium": {"pop_size": 150, "generations": 400, "local_search_intensity": 22},
    "hard": {"pop_size": 200, "generations": 800, "local_search_intensity": 30},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# -------------------------
# Worker globals (set by initializer)
# -------------------------
WORKER_ADJ_BITS = None  # list[int] bitset adjacency
WORKER_N = None

# -------------------------
# Utility: load graph & build bitsets
# -------------------------
def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    edges = []
    max_node = 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#'):
                continue
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

def build_adj_bitsets(n: int, adj_list: List[Set[int]]) -> List[int]:
    adj_bits = [0] * n
    for u in range(n):
        bits = 0
        for v in adj_list[u]:
            bits |= (1 << v)
        adj_bits[u] = bits
    return adj_bits

# -------------------------
# Pool initializer
# -------------------------
def _init_worker(adj_bits: List[int], n: int):
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n

# -------------------------
# Fast bitset evaluator (with robust bitcount)
# -------------------------
def bitcount(x: int) -> int:
    try:
        return int(x).bit_count()
    except Exception:
        return bin(int(x)).count('1')

# Larger per-process cache for better reuse
@lru_cache(maxsize=500000)
def evaluate_solution_bitset_cached(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    global WORKER_ADJ_BITS, WORKER_N
    if WORKER_ADJ_BITS is None or WORKER_N is None:
        raise RuntimeError("Worker globals not initialized")

    n = WORKER_N
    adj_bits = WORKER_ADJ_BITS

    t = solution_tuple[-1]
    perm = list(solution_tuple[:-1])
    size = n - t
    if size <= 0:
        return (0, 501)

    suffix_mask = [0] * n
    curr_mask = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (1 << perm[i])

    temp = adj_bits[:]  # copy
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
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
            temp[v] |= (succ ^ (1 << v))
    return (size, max_width)

def eval_wrapper(solution_tuple):
    return (solution_tuple, evaluate_solution_bitset_cached(solution_tuple))

# -------------------------
# Local search operators
# -------------------------
def smart_torso_shift(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    t = neighbor[-1]
    shift = int(max(1, n * 0.05))
    neighbor[-1] = max(0, min(n - 1, t + random.randint(-shift, shift)))
    return neighbor

def block_move(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    perm = neighbor[:-1]
    block_size = random.randint(2, max(3, int(n * 0.02)))
    if n > block_size:
        start = random.randint(0, n - block_size)
        block = perm[start:start + block_size]
        del perm[start:start + block_size]
        insert_pos = random.randint(0, len(perm))
        perm[insert_pos:insert_pos] = block
        neighbor[:-1] = perm
    return neighbor

def inversion_mutation(perm: List[int]) -> List[int]:
    a, b = sorted(random.sample(range(len(perm)), 2))
    perm = perm[:]
    perm[a:b+1] = reversed(perm[a:b+1])
    return perm

def swap_mutation(perm: List[int]) -> List[int]:
    perm = perm[:]
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm

# -------------------------
# Local search in worker
# -------------------------
def local_search_worker(args):
    sol, intensity = args
    n = WORKER_N
    best = tuple(int(x) for x in sol)
    best_score = evaluate_solution_bitset_cached(best)

    for _ in range(intensity):
        r = random.random()
        if r < 0.3:
            neigh = block_move(list(best), n)
        elif r < 0.7:
            perm = list(best[:-1])
            perm = inversion_mutation(perm)
            neigh = perm + [best[-1]]
        else:
            neigh = smart_torso_shift(list(best), n)
        neigh_t = tuple(int(x) for x in neigh)
        neigh_score = evaluate_solution_bitset_cached(neigh_t)
        if (neigh_score[0] > best_score[0]) or (neigh_score[0] == best_score[0] and neigh_score[1] < best_score[1]):
            best = neigh_t
            best_score = neigh_score
    return list(best)

# -------------------------
# PMX crossover
# -------------------------
def pmx_crossover(p1: List[int], p2: List[int]) -> List[int]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b+1] = p1[a:b+1]
    for i in range(a, b+1):
        val = p2[i]
        if val in child:
            continue
        pos = i
        while True:
            mapped = p1[pos]
            try:
                pos = p2.index(mapped)
            except ValueError:
                break
            if child[pos] == -1:
                child[pos] = val
                break
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]
    return child

# -------------------------
# Dominance, selection helpers
# -------------------------
def dominates(p, q):
    return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    for p in population:
        p['dominates_set'] = []
        p['dominated_by_count'] = 0
    fronts = [[]]
    for p in population:
        for q in population:
            if p is q:
                continue
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
        if not front:
            continue
        if len(new_population) + len(front) > pop_size:
            for p in front:
                p['distance'] = 0.0
            for i_obj in range(2):
                front.sort(key=lambda p: p['score'][i_obj])
                front[0]['distance'] = front[-1]['distance'] = float('inf')
                f_min = front[0]['score'][i_obj]
                f_max = front[-1]['score'][i_obj]
                if f_max > f_min:
                    for j in range(1, len(front) - 1):
                        prev_v = front[j - 1]['score'][i_obj]
                        next_v = front[j + 1]['score'][i_obj]
                        front[j]['distance'] += (next_v - prev_v) / (f_max - f_min)
            front.sort(key=lambda p: p['distance'], reverse=True)
            new_population.extend(front[:pop_size - len(new_population)])
            break
        new_population.extend(front)
    return new_population

# -------------------------
# Seeding heuristics (min-degree & min-fill)
# -------------------------
def min_degree_order(adj_list: List[Set[int]]) -> List[int]:
    n = len(adj_list)
    degree = [len(adj_list[i]) for i in range(n)]
    neighbors = [set(s) for s in adj_list]
    import heapq
    heap = [(degree[i], i) for i in range(n)]
    heapq.heapify(heap)
    removed = [False] * n
    order = []
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v] or degree[v] != d:
            continue
        removed[v] = True
        order.append(v)
        for u in list(neighbors[v]):
            if not removed[u]:
                neighbors[u].remove(v)
                degree[u] -= 1
                heapq.heappush(heap, (degree[u], u))
        neighbors[v].clear()
    return order

def min_fill_order(adj_list: List[Set[int]]) -> List[int]:
    n = len(adj_list)
    neighbors = [set(s) for s in adj_list]
    alive = set(range(n))
    order = []
    for _ in range(n):
        best_v = None
        best_fill = None
        for v in alive:
            neigh = neighbors[v]
            k = len(neigh)
            if k <= 1:
                fill = 0
            else:
                existing = 0
                for u in neigh:
                    existing += sum(1 for w in neighbors[u] if w in neigh)
                existing = existing // 2
                total_pairs = k * (k - 1) // 2
                fill = total_pairs - existing
            if best_fill is None or fill < best_fill:
                best_fill = fill
                best_v = v
        order.append(best_v)
        neigh = neighbors[best_v]
        for a in list(neigh):
            for b in list(neigh):
                if a != b:
                    neighbors[a].add(b)
        for u in neigh:
            neighbors[u].discard(best_v)
        neighbors[best_v].clear()
        alive.remove(best_v)
    return order

# -------------------------
# Utility: compare scores and persistence
# -------------------------
def score_better(a, b):
    # Return True if a better than b (a,b are (size,width))
    return (a[0] > b[0]) or (a[0] == b[0] and a[1] < b[1])

def persist_best(best_solution, best_score, problem_id):
    pkl_name = f"best_solution_{problem_id}.pkl"
    json_name = f"best_submission_{problem_id}.json"
    with open(pkl_name, "wb") as f:
        pickle.dump({'solution': best_solution, 'score': best_score}, f)
    # also write submission-style JSON vector for convenience
    with open(json_name, "w") as f:
        json.dump({"decisionVector": [[int(x) for x in best_solution]], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=2)

# -------------------------
# Main memetic algorithm with elite intensification & adaptive mutation
# -------------------------
def memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str) -> List[List[int]]:
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    adj_bits = build_adj_bitsets(n, adj_list)
    pop_size = config['pop_size']

    # Initial population (seeded + random)
    population = []
    seed_orders = []
    try:
        seed_orders.append(min_fill_order(adj_list))
    except Exception:
        pass
    try:
        seed_orders.append(min_degree_order(adj_list))
    except Exception:
        pass
    for base in list(seed_orders):
        seed_orders.append(list(reversed(base)))
        for _ in range(3):
            p = base[:]
            for _ in range(max(1, len(p)//200)):
                i, j = random.sample(range(len(p)), 2)
                p[i], p[j] = p[j], p[i]
            seed_orders.append(p)
    while len(seed_orders) < 10:
        seed_orders.append(list(np.random.permutation(n)))
    while len(population) < pop_size:
        base = random.choice(seed_orders)
        perm = base[:]
        for _ in range(random.randint(0, 4)):
            perm = inversion_mutation(perm)
            perm = swap_mutation(perm)
        t = random.randint(int(n * 0.2), int(n * 0.8))
        population.append({'solution': perm + [t]})

    start_gen = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            saved = pickle.load(f)
        population = saved['pop']
        start_gen = saved['gen'] + 1
        print(f"🔄 Resuming from gen {start_gen}")

    # tracking best
    best_solution = None
    best_score = (-1, 1000)
    stagnation_counter = 0
    base_mutation = CONFIG['general']['mutation_rate']

    with multiprocessing.Pool(initializer=_init_worker, initargs=(adj_bits, n)) as pool:
        # initial evaluate
        if 'score' not in population[0]:
            sols = [tuple(int(x) for x in p['solution']) for p in population]
            results = list(pool.map(eval_wrapper, sols))
            sol_to_score = {sol: score for sol, score in results}
            for p in population:
                p['score'] = sol_to_score.get(tuple(p['solution']), (0, 501))

        # initialize best from initial population
        for p in population:
            if best_solution is None or score_better(p['score'], best_score):
                best_score = p['score']
                best_solution = list(p['solution'])
        persist_best(best_solution, best_score, problem_id)

        for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving", initial=start_gen, total=config['generations']):
            # adaptive mutation update
            mutation_rate = base_mutation
            if stagnation_counter >= CONFIG['general']['stagnation_limit']:
                mutation_rate = min(0.95, base_mutation * CONFIG['general']['mutation_boost_factor'])

            mating_pool = crowding_selection(population, pop_size)

            offspring_sols = []
            while len(offspring_sols) < pop_size:
                p1 = random.choice(mating_pool)
                p2 = random.choice(mating_pool)
                perm1 = list(p1['solution'][:-1])
                perm2 = list(p2['solution'][:-1])

                if random.random() < config['crossover_rate']:
                    child_perm = pmx_crossover(perm1, perm2)
                else:
                    child_perm = perm1[:]

                if random.random() < mutation_rate:
                    if random.random() < 0.6:
                        child_perm = inversion_mutation(child_perm)
                    else:
                        child_perm = swap_mutation(child_perm)

                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                if random.random() < 0.35:
                    c_t = max(0, min(n - 1, c_t + random.randint(-int(n*0.03), int(n*0.03))))
                offspring_sols.append(child_perm + [c_t])

            # run normal local search in parallel
            ls_args = [(sol, config['local_search_intensity']) for sol in offspring_sols]
            improved_offspring = pool.map(local_search_worker, ls_args)

            # evaluate offspring
            eval_tuples = [tuple(int(x) for x in sol) for sol in improved_offspring]
            eval_results = list(pool.map(eval_wrapper, eval_tuples))
            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in eval_results]

            # combine and select
            population = crowding_selection(population + offspring_pop, pop_size)

            # Elite intensification: take top E by score and intensify local search
            E = CONFIG['general']['elite_count']
            elite_ls = CONFIG['general']['elite_ls_multiplier']
            # sort current population by (size desc, width asc)
            pop_sorted = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
            elites = pop_sorted[:E]
            elite_args = []
            for e in elites:
                # run a stronger LS on their solution
                elite_args.append((e['solution'], max(1, config['local_search_intensity'] * elite_ls)))
            if elite_args:
                improved_elites = pool.map(local_search_worker, elite_args)
                # evaluate and possibly replace in population
                eval_tuples = [tuple(int(x) for x in sol) for sol in improved_elites]
                eval_results = list(pool.map(eval_wrapper, eval_tuples))
                for sol, score in eval_results:
                    # find worst dominated by this or simply insert and reselect next gen
                    population.append({'solution': list(sol), 'score': score})
                population = crowding_selection(population, pop_size)

            # track best & stagnation
            current_best = None
            for p in population:
                if current_best is None or score_better(p['score'], current_best['score']):
                    current_best = p
            if current_best and score_better(current_best['score'], best_score):
                best_score = current_best['score']
                best_solution = list(current_best['solution'])
                persist_best(best_solution, best_score, problem_id)
                stagnation_counter = 0
                tqdm.write(f"✨ Gen {gen+1}: New best {best_score}")
            else:
                stagnation_counter += 1

            # print summary per generation
            tqdm.write(f"Gen {gen+1}: best(size,width)={best_score} stagn={stagnation_counter} mut={mutation_rate:.3f}")

            # checkpoint
            if (gen + 1) % config['checkpoint_interval'] == 0:
                tqdm.write(f"\n💾 Saving checkpoint at generation {gen + 1}...")
                to_save = {'pop': population, 'gen': gen}
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(to_save, f)

    final = crowding_selection(population, min(20, len(population)))
    return [p['solution'] for p in final]

# -------------------------
# Submission / main
# -------------------------
def create_submission_file(decision_vectors: List[List[int]], problem_id: str):
    filename = f"submission_{problem_id}.json"
    problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
    final_vectors = [[int(val) for val in vec] for vec in decision_vectors]
    submission = {
        "decisionVector": final_vectors,
        "problem": problem_name_map.get(problem_id, problem_id),
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(decision_vectors)} solutions.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(42)
    np.random.seed(42)

    problem_id = input("🔍 Select problem (easy/medium/hard) [easy]: ").strip().lower() or "easy"
    if problem_id not in PROBLEMS:
        print("❌ Invalid problem ID. Exiting.")
        exit(1)

    config = CONFIG['general'].copy()
    config.update(CONFIG[problem_id])

    n, adj = load_graph(problem_id)

    start_time = time.time()
    final_solutions = memetic_algorithm(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id)
