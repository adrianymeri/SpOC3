#!/usr/bin/env python3
"""
solver_fixed.py

Robust single-file memetic NSGA-II solver for SPOC-3 torso decompositions.

Save as solver_fixed.py and run with:
  python solver_fixed.py --problem easy --generations 300 --pop_size 120

See --help for CLI options.
"""

from __future__ import annotations
import argparse
import json
import random
import time
import math
import numpy as np
from typing import List, Set, Tuple, Dict, Any, Optional
import urllib.request
from tqdm import tqdm
import multiprocessing
import os
import pickle
from functools import lru_cache
from copy import deepcopy

# -------------------------
# Config (tweak for compute)
# -------------------------
CONFIG = {
    "general": {
        "mutation_rate": 0.55,
        "crossover_rate": 0.95,
        "checkpoint_interval": 5,
        "elite_count": 8,
        "elite_ls_multiplier": 4,
        "stagnation_limit": 12,
        "mutation_boost_factor": 1.8,
        "archive_max_size": 500,
        "cache_size": 500_000,
        "sanitize_warn_limit": 24,
    },
    "easy": {"pop_size": 120, "generations": 300, "local_search_intensity": 18},
    "medium": {"pop_size": 150, "generations": 400, "local_search_intensity": 22},
    "hard": {"pop_size": 220, "generations": 800, "local_search_intensity": 30},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# -------------------------
# Worker globals (set by pool initializer)
# -------------------------
WORKER_ADJ_BITS: Optional[List[int]] = None
WORKER_N: Optional[int] = None

# sanitizer counter (multiprocess safe)
_SANITIZE_WARN_COUNT = multiprocessing.Value('i', 0)


# ---------- Utilities ----------
def repair_perm_list(perm: List[int], n: int) -> List[int]:
    """
    Ensure perm is a valid permutation of 0..n-1.
    Preserve order of valid unique values; append missing values deterministically.
    """
    seen = set()
    fixed = []
    for x in perm:
        try:
            xi = int(x)
        except Exception:
            continue
        if 0 <= xi < n and xi not in seen:
            seen.add(xi)
            fixed.append(xi)
    for v in range(n):
        if v not in seen:
            fixed.append(v)
    return fixed[:n]


def clamp_t(t_raw: Any, n: int) -> int:
    try:
        t = int(t_raw)
    except Exception:
        t = 0
    return max(0, min(n - 1, t))


# -------------------------
# Graph loading & bitsets
# -------------------------
def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}' from {url} ...")
    edges = []
    max_node = 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = map(int, parts[:2])
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
        b = 0
        for v in adj_list[u]:
            b |= (1 << v)
        adj_bits[u] = b
    return adj_bits


# -------------------------
# Pool initializer
# -------------------------
def _init_worker(adj_bits: List[int], n: int):
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n


# -------------------------
# Robust bitcount
# -------------------------
def bitcount(x: int) -> int:
    try:
        return int(x).bit_count()
    except Exception:
        return bin(int(x)).count('1')


# -------------------------
# Sanitizer for solution tuples
# -------------------------
def sanitize_solution_tuple(solution_tuple: Tuple[Any, ...], n: int) -> Tuple[int, ...]:
    """
    Ensure solution tuple has length n+1, permutation contains all 0..n-1 exactly once.
    Repair if necessary. Controlled logging to avoid spam.
    """
    global _SANITIZE_WARN_COUNT
    # Quick valid check
    if len(solution_tuple) == n + 1:
        try:
            perm = [int(x) for x in solution_tuple[:-1]]
            t = int(solution_tuple[-1])
            if all(0 <= x < n for x in perm) and len(set(perm)) == n:
                t = max(0, min(n - 1, t))
                return tuple(perm) + (t,)
        except Exception:
            pass
    # repair path
    perm_raw = list(solution_tuple[:-1])
    t_raw = solution_tuple[-1] if len(solution_tuple) >= 1 else 0
    perm_fixed = repair_perm_list(perm_raw, n)
    t_fixed = clamp_t(t_raw, n)
    with _SANITIZE_WARN_COUNT.get_lock():
        if _SANITIZE_WARN_COUNT.value < CONFIG['general']['sanitize_warn_limit']:
            print("⚠️  Warning: repaired malformed permutation passed to evaluator (length/dup/out-of-range).")
            _SANITIZE_WARN_COUNT.value += 1
    return tuple(int(x) for x in perm_fixed) + (t_fixed,)


# -------------------------
# Evaluator with LRU cache (per process)
# -------------------------
def make_evaluator_cache(size: int):
    """
    Return a fresh lru_cache-wrapped evaluator function so that we can set cache size via config.
    We wrap inside a factory so the decorator can use the configured `size`.
    """

    @lru_cache(maxsize=size)
    def _evaluate(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
        global WORKER_ADJ_BITS, WORKER_N
        if WORKER_ADJ_BITS is None or WORKER_N is None:
            raise RuntimeError("Worker not initialized with graph")
        n = WORKER_N
        sol = sanitize_solution_tuple(solution_tuple, n)
        t = int(sol[-1])
        perm = list(sol[:-1])
        size_nodes = n - t
        if size_nodes <= 0:
            return (0, 501)
        adj_bits = WORKER_ADJ_BITS
        # build suffix mask
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
                    return (size_nodes, 501)
            if succ == 0:
                continue
            s = succ
            while s:
                vbit = s & -s
                s ^= vbit
                v = vbit.bit_length() - 1
                # OR all other successors (reachability update)
                temp[v] |= (succ ^ (1 << v))
        return (size_nodes, max_width)

    return _evaluate


# We'll create a default evaluator factory (and call make_evaluator_cache in main when we have config)
# For multiprocessing worker usage, we'll store a pointer to the function in global variable after init.
EVALUATOR = None  # will be set in main (per-process via initializer wrapper)


def eval_wrapper(solution_tuple):
    """
    Simple wrapper used by pool.map. Works with EVALUATOR which is per-process.
    """
    try:
        return (solution_tuple, EVALUATOR(solution_tuple))
    except Exception as e:
        # try to sanitize and re-evaluate
        if WORKER_N is not None:
            st = sanitize_solution_tuple(solution_tuple, WORKER_N)
            try:
                return (st, EVALUATOR(st))
            except Exception:
                pass
        # fallback: worst score
        return (solution_tuple, (0, 501))


# -------------------------
# Genetic & local moves
# -------------------------
def pmx_crossover(p1: List[int], p2: List[int]) -> List[int]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b + 1] = p1[a:b + 1]
    for i in range(a, b + 1):
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


def inversion_mutation(perm: List[int]) -> List[int]:
    if len(perm) < 2:
        return perm[:]
    a, b = sorted(random.sample(range(len(perm)), 2))
    perm = perm[:]
    perm[a:b + 1] = list(reversed(perm[a:b + 1]))
    return perm


def swap_mutation(perm: List[int]) -> List[int]:
    if len(perm) < 2:
        return perm[:]
    perm = perm[:]
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


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


def local_search_worker(args):
    sol, intensity = args
    n = WORKER_N
    best = tuple(int(x) for x in sol)
    best_score = EVALUATOR(best)
    for _ in range(intensity):
        r = random.random()
        if r < 0.28:
            neigh = block_move(list(best), n)
        elif r < 0.7:
            perm = list(best[:-1])
            perm = inversion_mutation(perm)
            neigh = perm + [best[-1]]
        else:
            neigh = smart_torso_shift(list(best), n)
        neigh_t = tuple(int(x) for x in neigh)
        neigh_score = EVALUATOR(neigh_t)
        if (neigh_score[0] > best_score[0]) or (neigh_score[0] == best_score[0] and neigh_score[1] < best_score[1]):
            best = neigh_t
            best_score = neigh_score
    return list(best)


# -------------------------
# Dominance / crowding selection
# -------------------------
def dominates_internal(p, q):
    return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])


def crowding_selection(population: List[Dict[str, Any]], pop_size: int) -> List[Dict[str, Any]]:
    # NSGA-II style non-dominated sort + crowding distance
    for p in population:
        p.setdefault('dominates_set', [])
        p.setdefault('dominated_by_count', 0)
    fronts: List[List[Dict[str, Any]]] = [[]]
    for p in population:
        p['dominates_set'] = []
        p['dominated_by_count'] = 0
    for p in population:
        for q in population:
            if p is q:
                continue
            if dominates_internal(p['score'], q['score']):
                p['dominates_set'].append(q)
            elif dominates_internal(q['score'], p['score']):
                p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[Dict[str, Any]] = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0:
                    next_front.append(q)
        fronts.append(next_front)
        i += 1
    new_pop: List[Dict[str, Any]] = []
    for front in fronts:
        if not front:
            continue
        if len(new_pop) + len(front) > pop_size:
            # crowding distance
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
            need = pop_size - len(new_pop)
            new_pop.extend(front[:need])
            break
        new_pop.extend(front)
    return new_pop


# -------------------------
# Persistence helpers
# -------------------------
def persist_archive(archive: List[Dict[str, Any]], n_nodes: int, problem_id: str, combined_score: float):
    fn_pkl = f"best_archive_{problem_id}.pkl"
    fn_json = f"best_archive_{problem_id}.json"
    try:
        with open(fn_pkl, "wb") as f:
            pickle.dump({'archive': archive, 'hv': combined_score}, f)
        decs = [[int(x) for x in a['solution']] for a in archive]
        with open(fn_json, "w") as f:
            json.dump({"decisionVector": decs, "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=2)
        print(f"💾 Persisted archive -> {fn_pkl} / {fn_json}")
    except Exception as e:
        print("⚠️  Warning: could not persist archive:", e)


def persist_best(best_solution: List[int], best_score: Tuple[int, int], problem_id: str):
    try:
        pkl_name = f"best_solution_{problem_id}.pkl"
        json_name = f"best_submission_{problem_id}.json"
        with open(pkl_name, "wb") as f:
            pickle.dump({'solution': best_solution, 'score': best_score}, f)
        with open(json_name, "w") as f:
            json.dump({"decisionVector": [[int(x) for x in best_solution]], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=2)
    except Exception as e:
        print("⚠️  Warning: could not persist best solution:", e)


# -------------------------
# Seeding loader
# -------------------------
def load_seed_file(seed_path: str, n: int, assert_on_invalid: bool = False) -> List[List[int]]:
    """
    Load results.json or similar containing a "decisionVector" list of vectors.
    Repair and validate each entry. Return list of valid solutions (list[int]).
    """
    try:
        with open(seed_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not read seed file {seed_path}: {e}")
    decs = data.get("decisionVector") or data.get("decisionVectors") or data
    # Accept both top-level list or dict {decisionVector: [...]}
    if isinstance(decs, dict) and "decisionVector" in decs:
        decs = decs["decisionVector"]
    if not isinstance(decs, list):
        raise RuntimeError("Seed file format not recognized: expected JSON array under 'decisionVector' or top-level list.")
    repaired = []
    for idx, vec in enumerate(decs):
        if not isinstance(vec, list):
            if assert_on_invalid:
                raise RuntimeError(f"Invalid seed entry at index {idx}: not a list")
            continue
        # repair: convert all numeric-like to int and clamp
        perm = [int(x) for x in vec[:-1]] if len(vec) >= 1 else []
        tval = int(vec[-1]) if len(vec) >= 1 else 0
        perm_fixed = repair_perm_list(perm, n)
        t_fixed = clamp_t(tval, n)
        repaired.append(perm_fixed + [t_fixed])
    print(f"✅ Loaded {len(repaired)} seed vectors from {seed_path}")
    return repaired


# -------------------------
# Main memetic algorithm
# -------------------------
def memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict[str, Any], problem_id: str,
                      seed_vectors: Optional[List[List[int]]] = None, diagnostic: bool = False) -> List[List[int]]:
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    adj_bits = build_adj_bitsets(n, adj_list)
    pop_size = config['pop_size']

    # set worker globals for main thread calls and prepare evaluator factory
    global EVALUATOR
    EVALUATOR = make_evaluator_cache(CONFIG['general']['cache_size'])
    # wrap a tiny initializer that sets the EVALUATOR on each worker
    def _init_pool(adj_bits_local, n_local):
        _init_worker(adj_bits_local, n_local)
        # monkeypatch per-process EVALUATOR to the lru-cached function created earlier (closure)
        global EVALUATOR
        EVALUATOR = make_evaluator_cache(CONFIG['general']['cache_size'])
        # after setting globals, set module-level EVALUATOR to call sanitized evaluator
        # note: EVALUATOR is process-local

    # initial seeded population
    population: List[Dict[str, Any]] = []
    seed_orders: List[List[int]] = []
    try:
        mf = min_fill_order(adj_list)
        if mf:
            seed_orders.append(mf)
    except Exception:
        pass
    try:
        md = min_degree_order(adj_list)
        if md:
            seed_orders.append(md)
    except Exception:
        pass

    for base in list(seed_orders):
        seed_orders.append(list(reversed(base)))
        # small perturbations
        for _ in range(3):
            p = base[:]
            for _ in range(max(1, len(p) // 200)):
                i, j = random.sample(range(len(p)), 2)
                p[i], p[j] = p[j], p[i]
            seed_orders.append(p)
    while len(seed_orders) < max(10, pop_size // 6):
        seed_orders.append(list(np.random.permutation(n)))

    # Build population (prefer seeds if provided)
    while len(population) < pop_size:
        if seed_vectors and random.random() < 0.45:
            base = random.choice(seed_vectors)
            # seed_vectors already are full vectors (perm + [t])
            repaired = repair_perm_list(base[:-1], n) + [clamp_t(base[-1], n)]
            # small shuffle for diversity
            perm = repaired[:-1][:]
            for _ in range(random.randint(0, 3)):
                perm = inversion_mutation(perm)
                perm = swap_mutation(perm)
            tval = clamp_t(repaired[-1], n)
            population.append({'solution': perm + [tval]})
        else:
            base = random.choice(seed_orders)
            perm = base[:]
            for _ in range(random.randint(0, 4)):
                perm = inversion_mutation(perm)
                perm = swap_mutation(perm)
            t = random.randint(int(n * 0.18), int(n * 0.82))
            population.append({'solution': perm + [t]})

    start_gen = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                saved = pickle.load(f)
            population = saved.get('pop', population)
            start_gen = saved.get('gen', 0) + 1
            print(f"🔄 Resuming from gen {start_gen}")
        except Exception:
            print("⚠️ Warning: could not load checkpoint (ignoring)")

    # Archive: store nondominated solutions
    archive: List[Dict[str, Any]] = []
    best_hv = None

    # set worker globals for main thread (single-process eval)
    WORKER_ADJ = build_adj_bitsets(n, adj_list)
    _init_worker(WORKER_ADJ, n)
    # EVALUATOR for main thread
    EVALUATOR = make_evaluator_cache(CONFIG['general']['cache_size'])

    # Create multiprocessing pool (if available CPUs >1)
    cpus = max(1, multiprocessing.cpu_count() - 1)
    use_pool = cpus > 0
    pool = multiprocessing.Pool(processes=cpus, initializer=_init_pool, initargs=(WORKER_ADJ, n)) if use_pool else None

    try:
        # initial evaluation
        sols = [tuple(int(x) for x in p['solution']) for p in population]
        if pool:
            results = list(pool.map(eval_wrapper, sols))
        else:
            results = [eval_wrapper(s) for s in sols]
        sol_to_score = {sol: score for sol, score in results}
        for p in population:
            p['score'] = sol_to_score.get(tuple(p['solution']), (0, 501))

        # seed archive
        for p in population:
            # add nondominated
            archive = add_to_archive_safe(archive, p, CONFIG['general']['archive_max_size'], n)

        # main loop
        base_mutation = CONFIG['general']['mutation_rate']
        stagnation_counter = 0

        gens_total = config['generations']
        for gen in tqdm(range(start_gen, gens_total), desc="🧬 Evolving", initial=start_gen, total=gens_total):
            mutation_rate = base_mutation
            if stagnation_counter >= CONFIG['general']['stagnation_limit']:
                mutation_rate = min(0.98, base_mutation * CONFIG['general']['mutation_boost_factor'])

            mating_pool = crowding_selection(population, pop_size)

            # produce offspring
            offspring_sols: List[List[int]] = []
            while len(offspring_sols) < pop_size:
                p1 = random.choice(mating_pool)
                p2 = random.choice(mating_pool)
                perm1 = list(p1['solution'][:-1])
                perm2 = list(p2['solution'][:-1])
                if random.random() < CONFIG['general']['crossover_rate']:
                    child_perm = pmx_crossover(perm1, perm2)
                else:
                    child_perm = perm1[:]
                if random.random() < mutation_rate:
                    if random.random() < 0.7:
                        child_perm = inversion_mutation(child_perm)
                    else:
                        child_perm = swap_mutation(child_perm)
                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                if random.random() < 0.33:
                    c_t = max(0, min(n - 1, c_t + random.randint(-int(n * 0.03), int(n * 0.03))))
                offspring_sols.append(child_perm + [c_t])

            # local search (parallel)
            ls_args = [(sol, config['local_search_intensity']) for sol in offspring_sols]
            if pool:
                improved_offspring = pool.map(local_search_worker, ls_args)
            else:
                improved_offspring = [local_search_worker(arg) for arg in ls_args]

            # evaluate offspring
            eval_tuples = [tuple(int(x) for x in sol) for sol in improved_offspring]
            if pool:
                eval_results = list(pool.map(eval_wrapper, eval_tuples))
            else:
                eval_results = [eval_wrapper(t) for t in eval_tuples]
            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in eval_results]

            # combine and select next gen
            population = crowding_selection(population + offspring_pop, pop_size)

            # Elite intensification
            E = CONFIG['general']['elite_count']
            elite_ls = CONFIG['general']['elite_ls_multiplier']
            pop_sorted = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
            elites = pop_sorted[:E]
            elite_args = [(e['solution'], max(1, config['local_search_intensity'] * elite_ls)) for e in elites]
            if elite_args:
                if pool:
                    improved_elites = pool.map(local_search_worker, elite_args)
                else:
                    improved_elites = [local_search_worker(a) for a in elite_args]
                eval_tuples = [tuple(int(x) for x in sol) for sol in improved_elites]
                if pool:
                    eval_results = list(pool.map(eval_wrapper, eval_tuples))
                else:
                    eval_results = [eval_wrapper(t) for t in eval_tuples]
                for sol, score in eval_results:
                    population.append({'solution': list(sol), 'score': score})
                population = crowding_selection(population, pop_size)

            # Merge population into archive
            for p in population:
                archive = add_to_archive_safe(archive, p, CONFIG['general']['archive_max_size'], n)

            # compute HV-like combined score (we store negative hv to match leaderboard style)
            hv = compute_hv_for_archive(archive, n)
            combined_score = -hv
            improved = False
            # update best single & persistence
            best_single = None
            for p in population:
                if best_single is None or dominates_internal(p['score'], best_single['score']):
                    best_single = p
            if best_single and ((best_single['score'][0] > best_single.get('__best_seen__', (-1, 999))[0]) or True):
                # update persisted best (we persist by comparing to stored best on disk)
                # Simple policy: persist best_single when it improves global file
                persist_best(best_single['solution'], best_single['score'], problem_id)

            # stagnation update vs hypervolume
            if best_hv is None or hv > best_hv + 1e-12:
                best_hv = hv
                persist_archive(archive, n, problem_id, -hv)
                stagnation_counter = 0
                tqdm.write(f"✨ Gen {gen+1}: HV improved -> {hv:.6f} | archive_size {len(archive)}")
            else:
                stagnation_counter += 1

            if diagnostic:
                # brief diagnostic summary then exit early if requested
                tqdm.write(f"[DIAGNOSTIC] Gen {gen+1}: best_single={best_single['score'] if best_single else None}")
                break

            # logging per generation
            best_single_score = best_single['score'] if best_single else (0, 999)
            tqdm.write(f"Gen {gen+1}: best_single(size,width)={best_single_score} archive_size={len(archive)} stagn={stagnation_counter} mut={mutation_rate:.3f}")

            # checkpoint
            if (gen + 1) % CONFIG['general']['checkpoint_interval'] == 0:
                try:
                    to_save = {'pop': population, 'gen': gen, 'archive': archive, 'best_hv': best_hv}
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(to_save, f)
                    tqdm.write(f"💾 Saved checkpoint at generation {gen + 1}")
                except Exception as e:
                    print("⚠️  Warning: checkpoint save failed:", e)

    finally:
        if pool:
            pool.close()
            pool.join()

    final_archive_sorted = sorted(archive, key=lambda a: (a['score'][0], -a['score'][1]), reverse=True)
    return [a['solution'] for a in final_archive_sorted[:20]]


# -------------------------
# Archive helpers (simple wrapper to avoid mutation bugs)
# -------------------------
def add_to_archive_safe(archive: List[Dict[str, Any]], candidate: Dict[str, Any], archive_max_size: int, n_nodes: int) -> List[Dict[str, Any]]:
    # Defensive copy and call add_to_archive (which assumes immutability)
    return add_to_archive(deepcopy(archive), deepcopy(candidate), archive_max_size, n_nodes)


def add_to_archive(archive: List[Dict[str, Any]], candidate: Dict[str, Any], archive_max_size: int, n_nodes: int) -> List[Dict[str, Any]]:
    cand_score = candidate['score']
    # discard if dominated by any in archive
    for a in archive:
        if dominates_internal(a['score'], cand_score):
            return archive
    # remove archive entries dominated by candidate
    new_archive = [a for a in archive if not dominates_internal(cand_score, a['score'])]
    new_archive.append({'solution': candidate['solution'], 'score': cand_score})
    # trim by diversity if too large (use simple crowding in score-space)
    if len(new_archive) > archive_max_size:
        pts = [(a['score'][0], a['score'][1]) for a in new_archive]
        # compute simple pairwise distances
        import math
        n = len(pts)
        scores = []
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j: continue
                s += math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        keep_idx = set(idx for idx, _ in scores[:archive_max_size])
        new_archive = [new_archive[i] for i in range(len(new_archive)) if i in keep_idx]
    return new_archive


# -------------------------
# Hypervolume (2D minimization transform)
# -------------------------
def transform_for_hv(score_tuple: Tuple[int, int], n_nodes: int) -> Tuple[int, int]:
    size, width = score_tuple
    return (width, int(n_nodes - size))


def nondominated_2d_minimization(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    nondom = []
    best_y = float('inf')
    for x, y in pts:
        if y < best_y:
            nondom.append((x, y))
            best_y = y
    return nondom


def hypervolume_2d_minimization(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    rx, ry = ref
    pts = [(float(x), float(y)) for x, y in points if x <= rx and y <= ry]
    if not pts:
        return 0.0
    nd = nondominated_2d_minimization(pts)
    nd_desc = sorted(nd, key=lambda p: p[0], reverse=True)
    hv = 0.0
    best_y = ry
    for x, y in nd_desc:
        if y < best_y:
            hv += (rx - x) * (best_y - y)
            best_y = y
    return float(hv)


def compute_hv_for_archive(archive: List[Dict[str, Any]], n_nodes: int) -> float:
    if not archive:
        return 0.0
    pts = [transform_for_hv(a['score'], n_nodes) for a in archive]
    ref = (n_nodes, n_nodes)
    return hypervolume_2d_minimization(pts, ref)


# -------------------------
# CLI and runner
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=str, default="easy", choices=PROBLEMS.keys())
    ap.add_argument("--seed", type=str, default=None, help="Path to JSON seed file (results.json).")
    ap.add_argument("--no-seed", action="store_true", help="Ignore any seed file even if present.")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--pop_size", type=int, default=None)
    ap.add_argument("--intensity", type=int, default=None, help="local search intensity override")
    ap.add_argument("--assert-on-invalid", action="store_true", help="fail if seed contains invalid entries")
    ap.add_argument("--diagnostic", action="store_true", help="run one diagnostic generation and exit early")
    return ap.parse_args()


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


def main():
    args = parse_args()
    problem_id = args.problem
    config = deepcopy(CONFIG['general'])
    mode_conf = deepcopy(CONFIG[problem_id])
    config.update(mode_conf)
    if args.generations is not None:
        config['generations'] = args.generations
    if args.pop_size is not None:
        config['pop_size'] = args.pop_size
    if args.intensity is not None:
        config['local_search_intensity'] = args.intensity

    n, adj = load_graph(problem_id)
    seed_vectors = None
    if args.seed and not args.no_seed:
        seed_vectors = load_seed_file(args.seed, n, assert_on_invalid=args.assert_on_invalid)

    start_time = time.time()
    final_solutions = memetic_algorithm(n, adj, config, problem_id, seed_vectors=seed_vectors, diagnostic=args.diagnostic)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(42)
    np.random.seed(42)
    main()
