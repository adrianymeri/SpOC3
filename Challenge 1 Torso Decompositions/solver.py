#!/usr/bin/env python3
"""
solver.py
Robust, end-to-end memetic NSGA-II (island model) for SpOC-3 torso decompositions.

Features:
- Robust loader that tolerantly parses .gr graph files (skips stray lines and reports examples)
- Big-int bitset evaluator (fast in Python using int.bit_count())
- Island-model memetic NSGA-II with migration
- Variable Neighborhood Search (VNS) local search worker
- Elite intensification and optional SA intensification to escape plateaus
- Adaptive mutation (boosts when stagnation detected)
- Checkpointing and best-solution persistence
- Submission file creation

Usage:
    python3 solver.py
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
from functools import lru_cache
from collections import deque

import numpy as np
from tqdm import tqdm
import multiprocessing

# --------------------
# Configuration
# --------------------
CONFIG = {
    "general": {
        "mutation_rate": 0.42,
        "crossover_rate": 0.92,
        "checkpoint_interval": 5,
        "elite_count": 8,
        "elite_ls_multiplier": 6,
        "stagnation_limit": 12,
        "mutation_boost_factor": 2.0,
        "island_count": 4,
        "migration_interval": 6,
        "migration_size": 6,
        "restart_fraction": 0.25,
        # SA intensification params
        "sa_intensity": 120,
        "sa_tabu_tenure": 120,
    },
    "easy": {"pop_size": 160, "generations": 600, "local_search_intensity": 18},
    "medium": {"pop_size": 240, "generations": 1200, "local_search_intensity": 24},
    "hard": {"pop_size": 360, "generations": 2000, "local_search_intensity": 36},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --------------------
# Robust graph loader
# --------------------
def load_graph(path_or_id: str) -> Tuple[int, List[Set[int]]]:
    """
    Robust loader for .gr graphs. Accepts:
      - 'easy'/'medium'/'hard' (tries ./data/<name>.gr first, then URL)
      - a local file path
      - a URL
    Returns: n (num nodes), adjacency list (List[Set[int]])
    Skips non-edge lines and prints a few examples for debugging.
    """
    candidate_paths = []
    if "/" in path_or_id or path_or_id.endswith(".gr"):
        candidate_paths.append(path_or_id)
    candidate_paths.append(os.path.join(os.getcwd(), "data", f"{path_or_id}.gr"))
    candidate_paths.append(os.path.join(os.getcwd(), f"{path_or_id}.gr"))
    if path_or_id in PROBLEMS:
        candidate_paths.append(PROBLEMS[path_or_id])

    chosen = None
    is_url = False
    for p in candidate_paths:
        if not p:
            continue
        if str(p).startswith("http://") or str(p).startswith("https://"):
            chosen = p
            is_url = True
            break
        if os.path.exists(p):
            chosen = p
            is_url = False
            break

    if chosen is None:
        if path_or_id in PROBLEMS:
            chosen = PROBLEMS[path_or_id]
            is_url = True
        else:
            raise FileNotFoundError(f"Could not find .gr file or URL for '{path_or_id}'")

    print(f"Loading graph data for '{path_or_id}' from {chosen} ...")
    int_pair_re = re.compile(r"^\s*(\d+)\s+(\d+)\s*(?:#.*)?$")
    edges = []
    skipped_examples = []
    line_no = 0

    if is_url:
        import urllib.request
        fh = urllib.request.urlopen(chosen)
        is_bytes = True
    else:
        fh = open(chosen, "rb")
        is_bytes = True

    try:
        for raw in fh:
            line_no += 1
            try:
                s = raw.decode("utf-8", errors="ignore").strip() if is_bytes else str(raw).strip()
            except Exception:
                s = str(raw).strip()
            if not s or s.startswith("#"):
                continue
            m = int_pair_re.match(s)
            if m:
                u = int(m.group(1)); v = int(m.group(2))
            else:
                nums = re.findall(r"\d+", s)
                if len(nums) >= 2:
                    u = int(nums[0]); v = int(nums[1])
                else:
                    if len(skipped_examples) < 8:
                        skipped_examples.append((line_no, s))
                    continue
            edges.append((u, v))
    finally:
        fh.close()

    if skipped_examples:
        print("Warning: Some non-edge lines were skipped (first examples):")
        for ln, txt in skipped_examples:
            print(f"  line {ln}: {txt!r}")
        print("If you see 'import' lines at the top, the .gr file is corrupted. Re-download the file if necessary.")

    if not edges:
        raise RuntimeError("No edges parsed from the graph file - aborting.")

    max_node = max(max(u, v) for u, v in edges)
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    print(f"Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, adj

# --------------------
# Bitset build and worker init
# --------------------
def build_adj_bitsets(n: int, adj_list: List[Set[int]]) -> List[int]:
    adj_bits = [0] * n
    for u in range(n):
        bits = 0
        for v in adj_list[u]:
            bits |= (1 << v)
        adj_bits[u] = bits
    return adj_bits

WORKER_ADJ_BITS = None
WORKER_N = None

def _init_worker(adj_bits: List[int], n: int):
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n

# --------------------
# Evaluator (cached per process)
# --------------------
def bitcount(x: int) -> int:
    try:
        return int(x).bit_count()
    except Exception:
        return bin(int(x)).count("1")

# tune cache size to avoid memory blowup; per-process LRU
@lru_cache(maxsize=300000)
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

# --------------------
# Neighborhoods & Local Search
# --------------------
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

def insertion_mutation(perm: List[int]) -> List[int]:
    perm = perm[:]
    i, j = sorted(random.sample(range(len(perm)), 2))
    val = perm.pop(j)
    perm.insert(i, val)
    return perm

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

def smart_torso_shift(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    t = neighbor[-1]
    shift = int(max(1, n * 0.05))
    neighbor[-1] = max(0, min(n - 1, t + random.randint(-shift, shift)))
    return neighbor

def vns_local_search(sol: List[int], intensity: int):
    # local search executed inside a worker process (uses global WORKER_N and cache)
    n = WORKER_N
    best = tuple(int(x) for x in sol)
    best_score = evaluate_solution_bitset_cached(best)
    neighborhoods = ['block', 'inv', 'swap', 'ins', 'shift']
    trials = max(1, intensity)
    for _ in range(trials):
        nb = random.choice(neighborhoods)
        if nb == 'block':
            cand = block_move(list(best), n)
        elif nb == 'inv':
            perm = list(best[:-1])
            perm = inversion_mutation(perm)
            cand = perm + [best[-1]]
        elif nb == 'swap':
            perm = list(best[:-1])
            perm = swap_mutation(perm)
            cand = perm + [best[-1]]
        elif nb == 'ins':
            perm = list(best[:-1])
            perm = insertion_mutation(perm)
            cand = perm + [best[-1]]
        else:
            cand = smart_torso_shift(list(best), n)
        cand_t = tuple(int(x) for x in cand)
        cand_score = evaluate_solution_bitset_cached(cand_t)
        if (cand_score[0] > best_score[0]) or (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
            best = cand_t
            best_score = cand_score
    return list(best)

def local_search_worker(args):
    sol, intensity = args
    return vns_local_search(sol, intensity)

# --------------------
# Crossover
# --------------------
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

# --------------------
# Multiobjective helpers (NSGA-II style)
# --------------------
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

# --------------------
# Seeding heuristics
# --------------------
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

def greedy_degree_order(adj_list: List[Set[int]]) -> List[int]:
    n = len(adj_list)
    order = sorted(range(n), key=lambda i: len(adj_list[i]), reverse=True)
    return order

# --------------------
# Persistence
# --------------------
def score_better(a, b):
    return (a[0] > b[0]) or (a[0] == b[0] and a[1] < b[1])

def persist_best(best_solution, best_score, problem_id):
    pkl_name = f"best_solution_{problem_id}.pkl"
    json_name = f"best_submission_{problem_id}.json"
    with open(pkl_name, "wb") as f:
        pickle.dump({'solution': best_solution, 'score': best_score}, f)
    with open(json_name, "w") as f:
        json.dump({"decisionVector": [[int(x) for x in best_solution]], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=2)

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
    print(f"Created submission file: {filename} with {len(decision_vectors)} solutions.")

# --------------------
# SA intensification (applied to elites if needed)
# --------------------
def sa_intensify(solution: List[int], intensity: int, tabu: Dict[Tuple[int,...], int]):
    n = WORKER_N
    curr = tuple(int(x) for x in solution)
    curr_score = evaluate_solution_bitset_cached(curr)
    best = curr
    best_score = curr_score
    T0 = 1.0
    Tf = 0.001
    for k in range(max(1, intensity)):
        temp = T0 * ((Tf / T0) ** (k / max(1, intensity - 1)))
        perm = list(curr[:-1])
        if random.random() < 0.6:
            perm = swap_mutation(perm)
        else:
            perm = inversion_mutation(perm)
        if random.random() < 0.2:
            t = max(0, min(n - 1, curr[-1] + random.randint(-max(1, n//50), max(1, n//50))))
        else:
            t = curr[-1]
        cand = tuple(list(perm) + [t])
        if cand in tabu:
            continue
        cand_score = evaluate_solution_bitset_cached(cand)
        # composite delta: prioritize size increase, penalize width slightly
        delta = (cand_score[0] - curr_score[0]) - (cand_score[1] - curr_score[1]) * 0.0002
        if delta > 0 or math.exp(delta / max(temp, 1e-12)) > random.random():
            curr = cand
            curr_score = cand_score
            tabu[cand] = CONFIG['general']['sa_tabu_tenure']
            if (curr_score[0] > best_score[0]) or (curr_score[0] == best_score[0] and curr_score[1] < best_score[1]):
                best = curr
                best_score = curr_score
    return list(best), best_score

# --------------------
# Island-model memetic algorithm (main workhorse)
# --------------------
class Island:
    def __init__(self, pop: List[Dict]):
        self.pop = pop

def memetic_algorithm_islands(n: int, adj_list: List[Set[int]], run_config: Dict, problem_id: str) -> List[List[int]]:
    """
    run_config must be a dict with:
      - 'general' (dict)
      - 'pop_size', 'generations', 'local_search_intensity', 'crossover_rate'
    """
    checkpoint_file = f"checkpoint_islands_{problem_id}.pkl"
    adj_bits = build_adj_bitsets(n, adj_list)
    island_count = run_config['general'].get('island_count', 4)
    total_pop_per_island = max(8, int(run_config['pop_size'] // island_count))

    # build seed orders
    seed_orders = []
    try:
        seed_orders.append(min_fill_order(adj_list))
    except Exception:
        pass
    try:
        seed_orders.append(min_degree_order(adj_list))
    except Exception:
        pass
    try:
        seed_orders.append(greedy_degree_order(adj_list))
    except Exception:
        pass
    # perturb seeds
    for base in list(seed_orders):
        seed_orders.append(list(reversed(base)))
        for _ in range(4):
            p = base[:]
            for _ in range(max(1, len(p)//150)):
                i, j = random.sample(range(len(p)), 2)
                p[i], p[j] = p[j], p[i]
            seed_orders.append(p)
    while len(seed_orders) < 20:
        seed_orders.append(list(np.random.permutation(n)))

    # initialize islands
    islands = []
    for _ in range(island_count):
        pop = []
        while len(pop) < total_pop_per_island:
            base = random.choice(seed_orders)
            perm = base[:]
            for _ in range(random.randint(0, 6)):
                perm = inversion_mutation(perm)
                perm = swap_mutation(perm)
            t = random.randint(int(n * 0.18), int(n * 0.78))
            pop.append({'solution': perm + [t]})
        islands.append(Island(pop))

    start_gen = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                saved = pickle.load(f)
            islands = saved['islands']
            start_gen = saved['gen'] + 1
            print(f"Resuming from gen {start_gen}")
        except Exception as e:
            print("Warning: failed to load checkpoint:", e)

    best_solution = None
    best_score = (-1, 1000)
    stagnation_counter = 0
    base_mutation = run_config['general'].get('mutation_rate', CONFIG['general']['mutation_rate'])

    # multiprocessing pool
    with multiprocessing.Pool(initializer=_init_worker, initargs=(adj_bits, n)) as pool:
        # initial evaluation
        for isl in islands:
            sols = [tuple(int(x) for x in p['solution']) for p in isl.pop]
            results = list(pool.imap_unordered(eval_wrapper, sols))
            sol_to_score = {sol: score for sol, score in results}
            for p in isl.pop:
                p['score'] = sol_to_score.get(tuple(p['solution']), (0, 501))

        # init best
        for isl in islands:
            for p in isl.pop:
                if best_solution is None or score_better(p['score'], best_score):
                    best_score = p['score']
                    best_solution = list(p['solution'])
        persist_best(best_solution, best_score, problem_id)

        total_gens = run_config['generations']
        crossover_rate = float(run_config.get('crossover_rate', run_config['general'].get('crossover_rate', 0.9)))
        for gen in tqdm(range(start_gen, total_gens), desc="Evolving", initial=start_gen, total=total_gens):
            mutation_rate = base_mutation
            if stagnation_counter >= run_config['general'].get('stagnation_limit', CONFIG['general']['stagnation_limit']):
                mutation_rate = min(0.95, base_mutation * run_config['general'].get('mutation_boost_factor', CONFIG['general']['mutation_boost_factor']))

            # evolve islands
            for isl in islands:
                pop = isl.pop
                pop_size = len(pop)
                mating_pool = crowding_selection(pop, pop_size)

                # offspring generation
                offspring = []
                while len(offspring) < pop_size:
                    p1 = random.choice(mating_pool)
                    p2 = random.choice(mating_pool)
                    perm1 = list(p1['solution'][:-1])
                    perm2 = list(p2['solution'][:-1])
                    if random.random() < crossover_rate:
                        child_perm = pmx_crossover(perm1, perm2)
                    else:
                        child_perm = perm1[:]
                    mut_r = mutation_rate * (1.0 + random.random() * 0.5)
                    if random.random() < mut_r:
                        r2 = random.random()
                        if r2 < 0.5:
                            child_perm = inversion_mutation(child_perm)
                        elif r2 < 0.85:
                            child_perm = swap_mutation(child_perm)
                        else:
                            child_perm = insertion_mutation(child_perm)
                    c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                    if random.random() < 0.35:
                        c_t = max(0, min(n - 1, c_t + random.randint(-int(n*0.03), int(n*0.03))))
                    offspring.append(child_perm + [c_t])

                # local search (parallel)
                ls_int = run_config.get('local_search_intensity', CONFIG['easy']['local_search_intensity'])
                ls_args = [(sol, ls_int) for sol in offspring]
                improved = list(pool.map(local_search_worker, ls_args))

                # evaluate improved offspring
                eval_tuples = [tuple(int(x) for x in sol) for sol in improved]
                eval_results = list(pool.imap_unordered(eval_wrapper, eval_tuples))
                offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in eval_results]

                # combine and select
                isl.pop = crowding_selection(pop + offspring_pop, pop_size)

            # Migration
            if (gen + 1) % run_config['general'].get('migration_interval', CONFIG['general']['migration_interval']) == 0:
                migrants = []
                for isl in islands:
                    isl_sorted = sorted(isl.pop, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                    migrants.append([m['solution'] for m in isl_sorted[:run_config['general'].get('migration_size', CONFIG['general']['migration_size'])]])
                for i, isl in enumerate(islands):
                    incoming = migrants[(i - 1) % len(islands)]
                    isl_sorted = sorted(isl.pop, key=lambda p: (p['score'][0], -p['score'][1]))
                    for j, sol in enumerate(incoming):
                        isl_sorted[j]['solution'] = sol
                        isl_sorted[j]['score'] = evaluate_solution_bitset_cached(tuple(int(x) for x in sol))
                    isl.pop = isl_sorted

            # Elite intensification (VNS) and optional SA
            all_pop = [p for isl in islands for p in isl.pop]
            pop_sorted = sorted(all_pop, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
            E = run_config['general'].get('elite_count', CONFIG['general']['elite_count'])
            elites = pop_sorted[:E]
            elite_ls_mult = run_config['general'].get('elite_ls_multiplier', CONFIG['general']['elite_ls_multiplier'])
            elite_args = [(e['solution'], max(1, int(run_config['local_search_intensity'] * elite_ls_mult))) for e in elites]
            if elite_args:
                improved_elites = list(pool.map(local_search_worker, elite_args))
                # optionally apply SA intensification to the top few elites to escape plateaus
                sa_count = max(1, min(4, len(improved_elites)//4))
                tabu = {}
                sa_improved = []
                for sol in improved_elites[:sa_count]:
                    sol2, score2 = sa_intensify(sol, run_config['general'].get('sa_intensity', CONFIG['general']['sa_intensity']), tabu)
                    sa_improved.append((tuple(sol2), score2))
                # evaluate all improved elites (including SA results)
                eval_tuples = [tuple(int(x) for x in sol) for sol in improved_elites]
                eval_results = list(pool.imap_unordered(eval_wrapper, eval_tuples))
                for sol, score in eval_results:
                    # insert into worst island to maintain diversity
                    worst_isl = min(islands, key=lambda isl: min(p['score'][0] for p in isl.pop))
                    worst_isl.pop.append({'solution': list(sol), 'score': score})
                    worst_isl.pop = crowding_selection(worst_isl.pop, len(worst_isl.pop) if len(worst_isl.pop) < total_pop_per_island else total_pop_per_island)

            # update best & stagnation
            current_best = None
            for p in [item for isl in islands for item in isl.pop]:
                if current_best is None or score_better(p['score'], current_best['score']):
                    current_best = p
            if current_best and score_better(current_best['score'], best_score):
                best_score = current_best['score']
                best_solution = list(current_best['solution'])
                persist_best(best_solution, best_score, problem_id)
                stagnation_counter = 0
                tqdm.write(f"New best at gen {gen+1}: {best_score}")
            else:
                stagnation_counter += 1

            tqdm.write(f"Gen {gen+1}: best(size,width)={best_score} stagn={stagnation_counter} mut={mutation_rate:.3f}")

            # diversification / partial restart when heavily stagnated
            if stagnation_counter and stagnation_counter % (run_config['general'].get('stagnation_limit', CONFIG['general']['stagnation_limit']) * 2) == 0:
                tqdm.write("Applying diversification (partial restart) due to stagnation")
                for isl in islands:
                    k = int(len(isl.pop) * run_config['general'].get('restart_fraction', CONFIG['general']['restart_fraction']))
                    for i in range(k):
                        base = random.choice(seed_orders)
                        perm = base[:]
                        for _ in range(random.randint(1, 8)):
                            perm = inversion_mutation(perm)
                            perm = swap_mutation(perm)
                        t = random.randint(int(n * 0.18), int(n * 0.78))
                        isl.pop[-1 - i] = {'solution': perm + [t], 'score': evaluate_solution_bitset_cached(tuple(perm + [t]))}

            # checkpoint
            if (gen + 1) % run_config.get('checkpoint_interval', run_config['general'].get('checkpoint_interval', CONFIG['general']['checkpoint_interval'])) == 0:
                tqdm.write(f"Saving checkpoint at generation {gen+1}...")
                to_save = {'islands': islands, 'gen': gen}
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(to_save, f)

    # final gather and select diverse top solutions
    all_pop = [p for isl in islands for p in isl.pop]
    final = crowding_selection(all_pop, min(40, len(all_pop)))
    return [p['solution'] for p in final]

# --------------------
# Main entry
# --------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(42)
    np.random.seed(42)

    choice = input("Select problem to run (easy/medium/hard or path): ").strip()
    if not choice:
        choice = "easy"
    if choice not in PROBLEMS and not os.path.exists(choice):
        print("Invalid problem ID or path. Exiting.")
        sys.exit(1)

    # Build run_config with required structure
    general_cfg = CONFIG.get("general", {}).copy()
    per_problem_cfg = CONFIG.get(choice, {}) if choice in CONFIG else {}
    run_cfg = {
        "general": general_cfg,
        "pop_size": per_problem_cfg.get("pop_size", CONFIG["easy"]["pop_size"]),
        "generations": per_problem_cfg.get("generations", CONFIG["easy"]["generations"]),
        "local_search_intensity": per_problem_cfg.get("local_search_intensity", CONFIG["easy"]["local_search_intensity"]),
        "crossover_rate": general_cfg.get("crossover_rate", CONFIG["general"].get("crossover_rate", 0.9)),
        "checkpoint_interval": general_cfg.get("checkpoint_interval", CONFIG["general"]["checkpoint_interval"]),
    }

    # Load graph (robust)
    n, adj = load_graph(choice)

    # Run
    start_time = time.time()
    final_solutions = memetic_algorithm_islands(n, adj, run_cfg, choice)
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f} s")

    create_submission_file(final_solutions, choice)
