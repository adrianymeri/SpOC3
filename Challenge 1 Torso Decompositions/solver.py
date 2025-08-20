#!/usr/bin/env python3
"""
Full solver.py - robust loader + island memetic NSGA-II with VNS, elite intensification,
adaptive mutation, checkpointing and persistence.

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

import numpy as np
from tqdm import tqdm
import multiprocessing

# --------------------
# Config & problems
# --------------------
CONFIG = {
    "general": {
        "mutation_rate": 0.45,
        "crossover_rate": 0.92,
        "checkpoint_interval": 5,
        "elite_count": 8,
        "elite_ls_multiplier": 5,
        "stagnation_limit": 10,
        "mutation_boost_factor": 2.0,
        "island_count": 4,
        "migration_interval": 6,
        "migration_size": 6,
        "restart_fraction": 0.25,
    },
    "easy": {"pop_size": 160, "generations": 450, "local_search_intensity": 16},
    "medium": {"pop_size": 200, "generations": 700, "local_search_intensity": 22},
    "hard": {"pop_size": 260, "generations": 1100, "local_search_intensity": 30},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --------------------
# Robust loader
# --------------------
def load_graph(path_or_id: str) -> Tuple[int, List[Set[int]]]:
    """
    Robust graph loader. path_or_id may be:
      - one of 'easy','medium','hard' (then local ./data/<name>.gr preferred, else URL)
      - a local path to a .gr file
      - a URL
    Returns (n, adj_list)
    """
    # Decide source path
    candidate_paths = []
    # if user provided a path containing a slash, try it directly first
    if "/" in path_or_id or path_or_id.endswith(".gr"):
        candidate_paths.append(path_or_id)
    # try local data/<name>.gr
    candidate_paths.append(os.path.join(os.getcwd(), "data", f"{path_or_id}.gr"))
    # try local <name>.gr
    candidate_paths.append(os.path.join(os.getcwd(), f"{path_or_id}.gr"))
    # if known URL exists and no local file, use it
    if path_or_id in PROBLEMS:
        candidate_paths.append(PROBLEMS[path_or_id])

    chosen = None
    is_url = False
    for p in candidate_paths:
        if p is None:
            continue
        # URL?
        if str(p).startswith("http://") or str(p).startswith("https://"):
            chosen = p
            is_url = True
            break
        # local file exists?
        if os.path.exists(p):
            chosen = p
            is_url = False
            break

    if chosen is None:
        # fallback: try the mapping URL
        if path_or_id in PROBLEMS:
            chosen = PROBLEMS[path_or_id]
            is_url = True
        else:
            raise FileNotFoundError(f"Could not find a local .gr file or known URL for '{path_or_id}'")

    print(f"📥 Loading graph data for '{path_or_id}' from {chosen} ...")

    edges = []
    max_node = 0
    int_pair_re = re.compile(r'^\s*(\d+)\s+(\d+)\s*(?:#.*)?$')
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
                line = raw.decode("utf-8", errors="ignore") if is_bytes else str(raw)
            except Exception:
                line = raw if isinstance(raw, str) else str(raw)
            s = line.strip()
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
            if u > max_node: max_node = u
            if v > max_node: max_node = v
    finally:
        fh.close()

    if skipped_examples:
        print("⚠️  Some non-edge lines were encountered and skipped (first examples):")
        for ln, txt in skipped_examples:
            print(f"   line {ln}: {txt!r}")
        print("If you see unexpected content (e.g. 'import'), check the .gr file for corruption or wrong path.")

    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, adj

# --------------------
# Bitset builder + worker initializer
# --------------------
def build_adj_bitsets(n: int, adj_list: List[Set[int]]) -> List[int]:
    adj_bits = [0] * n
    for u in range(n):
        bits = 0
        for v in adj_list[u]:
            bits |= (1 << v)
        adj_bits[u] = bits
    return adj_bits

# Worker globals for multiprocessing
WORKER_ADJ_BITS = None
WORKER_N = None
def _init_worker(adj_bits: List[int], n: int):
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n

# --------------------
# Fast evaluator (per-process cached)
# --------------------
def bitcount(x: int) -> int:
    try:
        # Python 3.8+
        return int(x).bit_count()
    except Exception:
        return bin(int(x)).count("1")

@lru_cache(maxsize=200000)
def evaluate_solution_bitset_cached(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Evaluator using big-int bitsets, cached per process. Returns (size, width).
    solution_tuple is permutation + t at the end.
    """
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

    temp = adj_bits[:]  # copy to allow propagation
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
# Neighborhoods and local search
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
# NSGA-II helpers
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
# Persistence & helpers
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
    print(f"📄 Created submission file: {filename} with {len(decision_vectors)} solutions.")

# --------------------
# Main memetic islands algorithm
# --------------------
class Island:
    def __init__(self, pop: List[Dict]):
        self.pop = pop

def memetic_algorithm_islands(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str) -> List[List[int]]:
    checkpoint_file = f"checkpoint_islands_{problem_id}.pkl"
    adj_bits = build_adj_bitsets(n, adj_list)
    island_count = config['general']['island_count']
    total_pop_per_island = max(10, config['pop_size'] // island_count)

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
            print(f"🔄 Resuming from gen {start_gen}")
        except Exception as e:
            print("⚠️  Failed to load checkpoint (ignored):", e)

    best_solution = None
    best_score = (-1, 1000)
    stagnation_counter = 0
    base_mutation = CONFIG['general']['mutation_rate']

    # launch pool once per run
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

        total_gens = config['generations']
        for gen in tqdm(range(start_gen, total_gens), desc="🧬 Evolving", initial=start_gen, total=total_gens):
            mutation_rate = base_mutation
            if stagnation_counter >= CONFIG['general']['stagnation_limit']:
                mutation_rate = min(0.95, base_mutation * CONFIG['general']['mutation_boost_factor'])

            # evolve islands
            for isl in islands:
                pop = isl.pop
                pop_size = len(pop)
                mating_pool = crowding_selection(pop, pop_size)
                offspring = []
                while len(offspring) < pop_size:
                    p1 = random.choice(mating_pool)
                    p2 = random.choice(mating_pool)
                    perm1 = list(p1['solution'][:-1])
                    perm2 = list(p2['solution'][:-1])
                    if random.random() < config['crossover_rate']:
                        child_perm = pmx_crossover(perm1, perm2)
                    else:
                        child_perm = perm1[:]
                    mut_r = mutation_rate * (1.0 + random.random() * 0.5)
                    if random.random() < mut_r:
                        r2 = random.random()
                        if r2 < 0.5:
                            child_perm = inversion_mutation(child_perm)
                        elif r2 < 0.8:
                            child_perm = swap_mutation(child_perm)
                        else:
                            child_perm = insertion_mutation(child_perm)
                    c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                    if random.random() < 0.35:
                        c_t = max(0, min(n - 1, c_t + random.randint(-int(n*0.03), int(n*0.03))))
                    offspring.append(child_perm + [c_t])

                # local search (parallel)
                ls_int = config['local_search_intensity']
                ls_args = [(sol, ls_int) for sol in offspring]
                improved = list(pool.map(local_search_worker, ls_args))

                # evaluate improved offspring
                eval_tuples = [tuple(int(x) for x in sol) for sol in improved]
                eval_results = list(pool.imap_unordered(eval_wrapper, eval_tuples))
                offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in eval_results]

                # combine and select
                isl.pop = crowding_selection(pop + offspring_pop, pop_size)

            # Migration
            if (gen + 1) % CONFIG['general']['migration_interval'] == 0:
                migrants = []
                for isl in islands:
                    isl_sorted = sorted(isl.pop, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                    migrants.append([m['solution'] for m in isl_sorted[:CONFIG['general']['migration_size']]])
                for i, isl in enumerate(islands):
                    incoming = migrants[(i - 1) % len(islands)]
                    isl_sorted = sorted(isl.pop, key=lambda p: (p['score'][0], -p['score'][1]))
                    for j, sol in enumerate(incoming):
                        isl_sorted[j]['solution'] = sol
                        isl_sorted[j]['score'] = evaluate_solution_bitset_cached(tuple(int(x) for x in sol))
                    isl.pop = isl_sorted

            # Elite intensification
            all_pop = [p for isl in islands for p in isl.pop]
            pop_sorted = sorted(all_pop, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
            E = CONFIG['general']['elite_count']
            elites = pop_sorted[:E]
            elite_ls = CONFIG['general']['elite_ls_multiplier']
            elite_args = []
            for e in elites:
                elite_args.append((e['solution'], max(1, config['local_search_intensity'] * elite_ls)))
            if elite_args:
                improved_elites = list(pool.map(local_search_worker, elite_args))
                eval_tuples = [tuple(int(x) for x in sol) for sol in improved_elites]
                eval_results = list(pool.imap_unordered(eval_wrapper, eval_tuples))
                for sol, score in eval_results:
                    # insert into worst island to keep diversity
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
                tqdm.write(f"✨ Gen {gen+1}: New best {best_score}")
            else:
                stagnation_counter += 1

            tqdm.write(f"Gen {gen+1}: best(size,width)={best_score} stagn={stagnation_counter} mut={mutation_rate:.3f}")

            # diversification / restart
            if stagnation_counter and stagnation_counter % (CONFIG['general']['stagnation_limit'] * 2) == 0:
                tqdm.write("🔁 Applying diversification (partial restart) due to stagnation")
                for isl in islands:
                    k = int(len(isl.pop) * CONFIG['general']['restart_fraction'])
                    for i in range(k):
                        base = random.choice(seed_orders)
                        perm = base[:]
                        for _ in range(random.randint(1, 8)):
                            perm = inversion_mutation(perm)
                            perm = swap_mutation(perm)
                        t = random.randint(int(n * 0.18), int(n * 0.78))
                        isl.pop[-1 - i] = {'solution': perm + [t], 'score': evaluate_solution_bitset_cached(tuple(perm + [t]))}

            # checkpoint
            if (gen + 1) % config['checkpoint_interval'] == 0:
                tqdm.write(f"\n💾 Saving checkpoint at generation {gen + 1}...")
                to_save = {'islands': islands, 'gen': gen}
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(to_save, f)

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

    choice = input(" Select problem to run (easy/medium/hard): ").strip().lower() or "easy"
    if choice not in PROBLEMS and not os.path.exists(choice):
        print("❌ Invalid problem ID or path. Exiting.")
        sys.exit(1)

    # build config
    config = CONFIG['general'].copy()
    local_cfg = CONFIG.get(choice, {})
    cfg = config.copy()
    cfg.update(local_cfg)
    # unify naming (pop_size, generations, local_search_intensity)
    # ensure keys exist
    cfg['pop_size'] = local_cfg.get('pop_size', cfg.get('pop_size', 160))
    cfg['generations'] = local_cfg.get('generations', cfg.get('generations', 450))
    cfg['local_search_intensity'] = local_cfg.get('local_search_intensity', cfg.get('local_search_intensity', 16))

    # load graph
    n, adj = load_graph(choice)

    start_time = time.time()
    final_solutions = memetic_algorithm_islands(n, adj, cfg, choice)
    total_time = time.time() - start_time
    print(f"\n⏱️  Total Optimization Time: {total_time:.2f} seconds")

    create_submission_file(final_solutions, choice)
