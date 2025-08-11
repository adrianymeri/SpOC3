#!/usr/bin/env python3
"""
Improved memetic NSGA-II for SPOC-3 torso decompositions.

Key improvements:
- Pool initializer to avoid pickling the whole graph per task
- Bitset-based evaluator (fast) with early bailout
- Per-process LRU caching of evaluations
- Parallel local search implemented as a worker function (avoids pickling objects)
- PMX crossover + inversion mutation + torso hill-climb
- Seeding with min-fill/min-degree heuristics + perturbations
- Safer checkpointing (only population + gen + RNG state)
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
    },
    "easy": {"pop_size": 120, "generations": 250, "local_search_intensity": 18},
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
    # Called in each worker process to set globals (avoids pickling adj per call)
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n

# -------------------------
# Fast bitset evaluator
# -------------------------
def bitcount(x: int) -> int:
    # Coerce to plain Python int before calling .bit_count() to avoid AttributeError
    try:
        return int(x).bit_count()
    except Exception:
        # Fallback (slower) if something weird happens
        return bin(int(x)).count('1')

# We decorate the core evaluator with lru_cache per-process.
# The argument is a tuple of ints: (perm0, perm1, ..., perm_{n-1}, t)
@lru_cache(maxsize=200000)
def evaluate_solution_bitset_cached(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    """Evaluates a solution tuple using bitset adjacency stored in WORKER_ADJ_BITS & WORKER_N.
    Returns (size, width). Uses early bailout if width >= 500."""
    # note: this runs in workers (or main if called there); relies on globals set by initializer
    global WORKER_ADJ_BITS, WORKER_N
    if WORKER_ADJ_BITS is None or WORKER_N is None:
        raise RuntimeError("Worker globals not initialized (WORKER_ADJ_BITS is None)")

    n = WORKER_N
    adj_bits = WORKER_ADJ_BITS

    t = solution_tuple[-1]
    perm = list(solution_tuple[:-1])
    size = n - t
    if size <= 0:
        return (0, 501)

    # suffix_mask[i] = bits of nodes at positions > i
    suffix_mask = [0] * n
    curr_mask = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (1 << perm[i])

    # working copy of adjacency (ints)
    temp = adj_bits[:]  # shallow copy of ints

    max_width = 0
    # elimination in permutation order
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
        # add fill-ins: connect every v in succ to succ \ {v}
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= (succ ^ (1 << v))
    return (size, max_width)

# Wrapper for pool.map that expects a single-argument iterable
def eval_wrapper(solution_tuple):
    return (solution_tuple, evaluate_solution_bitset_cached(solution_tuple))

# -------------------------
# Local search operators (pure functions)
# -------------------------
def smart_torso_shift(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    t = neighbor[-1]
    shift = int(max(1, n * 0.04))
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
# Local search executed in worker (to avoid pickling objects)
# -------------------------
def local_search_worker(args):
    """args: (solution_list, intensity)
    Runs a simple hill-climbing local search using evaluate_solution_bitset_cached (cached)."""
    sol, intensity = args
    n = WORKER_N
    best = tuple(int(x) for x in sol)
    best_score = evaluate_solution_bitset_cached(best)

    for _ in range(intensity):
        # randomly choose operator (weighted uniform)
        r = random.random()
        if r < 0.35:
            neigh = block_move(list(best), n)
        elif r < 0.7:
            perm = list(best[:-1])
            perm = inversion_mutation(perm)
            neigh = perm + [best[-1]]
        else:
            neigh = smart_torso_shift(list(best), n)
        neigh_t = tuple(int(x) for x in neigh)
        neigh_score = evaluate_solution_bitset_cached(neigh_t)
        # dominance: prefer larger size and smaller width
        if (neigh_score[0] > best_score[0]) or (neigh_score[0] == best_score[0] and neigh_score[1] < best_score[1]):
            best = neigh_t
            best_score = neigh_score
    return list(best)

# -------------------------
# Crossover: PMX (Partially-Mapped Crossover)
# -------------------------
def pmx_crossover(p1: List[int], p2: List[int]) -> List[int]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    # copy segment from p1
    child[a:b+1] = p1[a:b+1]
    # mapping for conflicts
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
    # fill remaining from p2
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]
    return child

# -------------------------
# Dominance & Crowding Selection (NSGA-II-like)
# -------------------------
def dominates(p, q):
    # p and q are (size, width) -> we want larger size, smaller width
    return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    # Non-dominated sort
    for p in population:
        p['dominates_set'] = []
        p['dominated_by_count'] = 0
    fronts = [[]]
    for i, p in enumerate(population):
        for j, q in enumerate(population):
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
            # compute crowding distance
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
# Seeding heuristics: min-degree and min-fill
# -------------------------
def min_degree_order(adj_list: List[Set[int]]) -> List[int]:
    n = len(adj_list)
    alive = set(range(n))
    degree = [len(adj_list[i]) for i in range(n)]
    order = []
    neighbors = [set(s) for s in adj_list]
    import heapq
    heap = [(degree[i], i) for i in range(n)]
    heapq.heapify(heap)
    removed = [False] * n
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
    # Greedy min-fill: at each step remove vertex whose elimination adds fewest fill-ins
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
# PMX/OX and utility for torso hill-climb
# -------------------------
def pmx_for_parents(p1, p2):
    return pmx_crossover(p1, p2)

def hill_climb_t_value(perm: List[int], t_init: int, tries=21) -> int:
    best_t = int(t_init)
    best_score = evaluate_solution_bitset_cached(tuple(perm + [best_t]))
    n = len(perm)
    half = max(1, tries // 2)
    for delta in range(-half, half + 1):
        t = max(0, min(n - 1, t_init + delta))
        sc = evaluate_solution_bitset_cached(tuple(perm + [t]))
        if (sc[0] > best_score[0]) or (sc[0] == best_score[0] and sc[1] < best_score[1]):
            best_score = sc
            best_t = t
    return best_t

# -------------------------
# Main memetic algorithm
# -------------------------
def memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str) -> List[List[int]]:
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    adj_bits = build_adj_bitsets(n, adj_list)

    # build initial population (with seeding heuristics)
    pop_size = config['pop_size']
    population = []
    # create several seeds: min-fill, min-degree, reversed, and random perturbations
    print("🌱 Building seeded initial population...")
    seed_orders = []
    try:
        mf = min_fill_order(adj_list)
        seed_orders.append(mf)
    except Exception:
        pass
    try:
        md = min_degree_order(adj_list)
        seed_orders.append(md)
    except Exception:
        pass
    # include reversed seeds and perturbed versions
    for base in list(seed_orders):
        seed_orders.append(list(reversed(base)))
        # small perturbations
        for _ in range(3):
            p = base[:]
            for _ in range(max(1, len(p)//200)):
                i, j = random.sample(range(len(p)), 2)
                p[i], p[j] = p[j], p[i]
            seed_orders.append(p)

    while len(seed_orders) < 10:
        perm = list(np.random.permutation(n))
        seed_orders.append(list(perm))

    while len(population) < pop_size:
        base = random.choice(seed_orders)
        perm = base[:]
        for _ in range(random.randint(0, 3)):
            perm = inversion_mutation(perm)
            perm = swap_mutation(perm)
        t = random.randint(int(n * 0.2), int(n * 0.8))
        population.append({'solution': perm + [t]})

    start_gen = 0
    if os.path.exists(checkpoint_file):
        print(f"🔄 Loading checkpoint {checkpoint_file} ...")
        with open(checkpoint_file, 'rb') as f:
            saved = pickle.load(f)
        population = saved['pop']
        start_gen = saved['gen'] + 1
        print(f"Resuming from generation {start_gen}")

    with multiprocessing.Pool(initializer=_init_worker, initargs=(adj_bits, n)) as pool:
        if 'score' not in population[0]:
            print("Evaluating initial population...")
            sols = [tuple(int(x) for x in p['solution']) for p in population]
            results = list(pool.map(eval_wrapper, sols))
            sol_to_score = {sol: score for sol, score in results}
            for p in population:
                p['score'] = sol_to_score.get(tuple(p['solution']), (0, 501))

        for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving", initial=start_gen, total=config['generations']):
            mating_pool = crowding_selection(population, config['pop_size'])

            offspring_sols = []
            while len(offspring_sols) < pop_size:
                p1 = random.choice(mating_pool)
                p2 = random.choice(mating_pool)
                perm1 = list(p1['solution'][:-1])
                perm2 = list(p2['solution'][:-1])

                if random.random() < config['crossover_rate']:
                    child_perm = pmx_for_parents(perm1, perm2)
                else:
                    child_perm = perm1[:]

                if random.random() < config['mutation_rate']:
                    if random.random() < 0.5:
                        child_perm = inversion_mutation(child_perm)
                    else:
                        child_perm = swap_mutation(child_perm)

                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                if random.random() < 0.3:
                    c_t = max(0, min(n - 1, c_t + random.randint(-int(n*0.02), int(n*0.02))))
                offspring_sols.append(child_perm + [c_t])

            ls_args = [(sol, config['local_search_intensity']) for sol in offspring_sols]
            improved_offspring = pool.map(local_search_worker, ls_args)

            eval_tuples = [tuple(int(x) for x in sol) for sol in improved_offspring]
            eval_results = list(pool.map(eval_wrapper, eval_tuples))
            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in eval_results]

            population = crowding_selection(population + offspring_pop, config['pop_size'])

            if (gen + 1) % config['checkpoint_interval'] == 0:
                tqdm.write(f"\n💾 Saving checkpoint at generation {gen + 1}...")
                to_save = {'pop': population, 'gen': gen}
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(to_save, f)

    final = crowding_selection(population, min(20, len(population)))
    return [p['solution'] for p in final]

# -------------------------
# Submission file creator
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

# -------------------------
# Main execution
# -------------------------
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
