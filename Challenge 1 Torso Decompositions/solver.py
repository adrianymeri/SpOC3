#!/usr/bin/env python3
"""
solver_numba.py

Memetic NSGA-II solver for SPOC-3 torso decompositions with a Numba-jitted
packed-bitset evaluator for high-performance CPU evaluation.

Features:
 - Packed uint64 adjacency representation
 - Numba nopython evaluator core (fast)
 - Safe fallback to Python evaluator if Numba is missing
 - Multiprocessing Pool initializer to share packed adjacency
 - Basic memetic loop: PMX crossover, inversion/swap mutation, local search,
   elite intensification, adaptive mutation rate, checkpointing, submission export.
 - CLI flags for quick diagnostics and profiling.
"""

from __future__ import annotations
import argparse
import json
import random
import time
import os
import pickle
import math
from typing import List, Set, Tuple, Dict, Any
from tqdm import tqdm
import multiprocessing
import urllib.request

import numpy as np

# Try to import numba; if not available, we'll fallback
try:
    import numba as nb
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ---------- Configuration ----------
CONFIG = {
    "general": {
        "mutation_rate": 0.55,
        "crossover_rate": 0.95,
        "checkpoint_interval": 5,
        "elite_count": 10,
        "elite_ls_multiplier": 4,
        "stagnation_limit": 12,
        "mutation_boost_factor": 1.8,
    },
    "easy": {"pop_size": 220, "generations": 600, "local_search_intensity": 18},
    "medium": {"pop_size": 220, "generations": 500, "local_search_intensity": 22},
    "hard": {"pop_size": 300, "generations": 1000, "local_search_intensity": 28},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# ---------- Globals shared with workers ----------
WORKER_ADJ_PACKED = None  # np.uint64 array shape (n_nodes, W)
WORKER_W = None
WORKER_N = None

# sanitizer warning counter (multiprocess-safe)
_SANITIZE_WARN_COUNT = multiprocessing.Value('i', 0)

# ---------- Utilities: load graph and pack adjacency ----------
def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    edges = []
    max_node = -1
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#'):
                continue
            u, v = map(int, line.strip().split())
            edges.append((u, v))
            if u > max_node:
                max_node = u
            if v > max_node:
                max_node = v
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, adj

def pack_adj_bits(adj_list: List[Set[int]], n: int) -> Tuple[np.ndarray, int]:
    """
    Pack adjacency into uint64 words: shape (n, W) where W = ceil(n/64)
    """
    W = (n + 63) // 64
    arr = np.zeros((n, W), dtype=np.uint64)
    for u in range(n):
        for v in adj_list[u]:
            w = v // 64
            b = v % 64
            arr[u, w] |= (np.uint64(1) << np.uint64(b))
    return arr, W

# ---------- Sanitizer ----------
def repair_perm_list(perm: List[int], n: int) -> List[int]:
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

def sanitize_solution_tuple(solution_tuple: Tuple[int, ...], n: int) -> Tuple[int, ...]:
    """
    Ensure tuple has length n+1, perm is valid permutation 0..n-1, clamp t.
    """
    global _SANITIZE_WARN_COUNT
    if len(solution_tuple) == n + 1:
        perm = solution_tuple[:-1]
        if all(isinstance(x, (int, np.integer)) and 0 <= int(x) < n for x in perm) and len(set(perm)) == n:
            t = int(solution_tuple[-1])
            t = max(0, min(n - 1, t))
            return tuple(int(x) for x in perm) + (t,)
    # repair
    perm_raw = list(solution_tuple[:-1]) if len(solution_tuple) >= 1 else []
    t_raw = solution_tuple[-1] if len(solution_tuple) >= 1 else 0
    perm_fixed = repair_perm_list(perm_raw, n)
    try:
        t_fixed = int(t_raw)
    except Exception:
        t_fixed = 0
    t_fixed = max(0, min(n - 1, t_fixed))
    with _SANITIZE_WARN_COUNT.get_lock():
        if _SANITIZE_WARN_COUNT.value < 20:
            print("⚠️  Warning: repaired malformed permutation passed to evaluator (length/dup/out-of-range).")
            _SANITIZE_WARN_COUNT.value += 1
    return tuple(perm_fixed) + (t_fixed,)

# ---------- Numba evaluator (packed bitsets) ----------
if NUMBA_AVAILABLE:
    @nb.njit(cache=True)
    def _nb_popcount(x: np.uint64) -> int:
        cnt = 0
        # Kernighan's algorithm
        while x != 0:
            x = x & (x - np.uint64(1))
            cnt += 1
        return cnt

    @nb.njit(cache=True)
    def _nb_ctz(x: np.uint64) -> int:
        # count trailing zeros. precondition: x != 0
        idx = 0
        # see note: shifting is fine in numba
        while (x & np.uint64(1)) == np.uint64(0):
            x = x >> np.uint64(1)
            idx += 1
        return idx

    @nb.njit(cache=True)
    def _nb_eval_core(adj_packed: np.ndarray, perm: np.ndarray, t: int, n: int, W: int) -> Tuple[int, int]:
        """
        Numba core evaluator. Returns (size, max_width).
        adj_packed: uint64[n, W]
        perm: int64[n]
        """
        # Build suffix_masks: shape (n, W)
        suffix_masks = np.empty((n, W), dtype=np.uint64)
        curr = np.zeros(W, dtype=np.uint64)
        for i in range(n-1, -1, -1):
            suffix_masks[i, :] = curr
            v = perm[i]
            widx = v // 64
            b = v % 64
            curr[widx] = curr[widx] | (np.uint64(1) << np.uint64(b))

        # copy adjacency (we will mutate temp)
        temp = np.empty_like(adj_packed)
        for i in range(n):
            for w in range(W):
                temp[i, w] = adj_packed[i, w]

        max_width = 0
        size = n - t
        if size <= 0:
            return 0, 501

        for i in range(n):
            u = perm[i]
            # compute succ mask wordwise
            out_deg = 0
            # first compute succ words and count bits
            for w in range(W):
                succ_word = temp[u, w] & suffix_masks[i, w]
                if succ_word != 0:
                    out_deg += _nb_popcount(succ_word)
            if out_deg > max_width:
                max_width = out_deg
                if max_width >= 500:
                    return size, 501
            if out_deg == 0:
                continue
            # For each v in succ, propagate
            for w in range(W):
                succ_word = temp[u, w] & suffix_masks[i, w]
                while succ_word != 0:
                    # get lowest bit
                    vbit = succ_word & -succ_word
                    # remove it
                    succ_word = succ_word & (succ_word - np.uint64(1))
                    b = _nb_ctz(vbit)
                    v_global = w * 64 + b
                    # build mask (succ_mask without the v bit) and OR into temp[v_global]
                    for ww in range(W):
                        # compute succ_mask (temp[u, ww] & suffix_masks[i, ww]) with v bit removed if same word
                        mm = temp[u, ww] & suffix_masks[i, ww]
                        if ww == w:
                            mm = mm & ~(np.uint64(1) << np.uint64(b))
                        temp[v_global, ww] = temp[v_global, ww] | mm
        return size, max_width

    # wrapper to call core from Python
    def evaluate_solution_numba(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
        global WORKER_ADJ_PACKED, WORKER_W, WORKER_N
        if WORKER_ADJ_PACKED is None or WORKER_N is None:
            raise RuntimeError("Worker globals not initialized for numba evaluator")
        n = WORKER_N
        W = WORKER_W
        sol = sanitize_solution_tuple(solution_tuple, n)
        perm = np.array(sol[:-1], dtype=np.int64)
        t = int(sol[-1])
        return tuple(_nb_eval_core(WORKER_ADJ_PACKED, perm, t, n, W))
else:
    evaluate_solution_numba = None

# ---------- Fallback pure-Python evaluator (keeps original logic) ----------
def bitcount_py(x: int) -> int:
    try:
        return int(x).bit_count()
    except Exception:
        return bin(int(x)).count('1')

def evaluate_solution_python(solution_tuple: Tuple[int, ...], adj_list: List[int]=None) -> Tuple[int, int]:
    """
    Simple Python evaluator operating on bit-int adjacency (adj_list as python ints)
    """
    global WORKER_N
    if WORKER_N is None:
        raise RuntimeError("Worker globals not initialized for python evaluator")
    n = WORKER_N
    sol = sanitize_solution_tuple(solution_tuple, n)
    perm = list(sol[:-1])
    t = int(sol[-1])
    size = n - t
    if size <= 0:
        return (0, 501)
    # if adj_list provided as python ints, use that; else expect WORKER_ADJ_PACKED not None
    if adj_list is None:
        # try to build simple int bitset adjacency from packed
        if WORKER_ADJ_PACKED is None:
            raise RuntimeError("No adjacency available for python evaluator")
        # create python ints per node
        adj_ints = []
        nW = WORKER_W
        for u in range(WORKER_ADJ_PACKED.shape[0]):
            bits = 0
            for w in range(nW):
                bits |= int(WORKER_ADJ_PACKED[u,w]) << (w*64)
            adj_ints.append(bits)
        adj = adj_ints
    else:
        adj = adj_list

    # compute suffix mask ints
    suffix_mask = [0] * n
    curr_mask = 0
    for i in range(n-1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (1 << perm[i])

    temp = adj[:]  # copy list of ints
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount_py(succ)
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

# ---------- Pool initializer to set globals ----------
def _init_worker(adj_packed: np.ndarray, W: int, n: int, use_numba: bool):
    global WORKER_ADJ_PACKED, WORKER_W, WORKER_N, evaluate_solution_cached
    WORKER_ADJ_PACKED = adj_packed
    WORKER_W = W
    WORKER_N = n
    if use_numba and NUMBA_AVAILABLE:
        # bind evaluate wrapper to numba version
        def _eval_wrapper(sol_tpl):
            return (sol_tpl, evaluate_solution_numba(sol_tpl))
    else:
        def _eval_wrapper(sol_tpl):
            return (sol_tpl, evaluate_solution_python(sol_tpl))
    # assign to module-level name (so pool.map can find it)
    globals()['eval_wrapper_local'] = _eval_wrapper

def eval_wrapper(solution_tuple):
    # top-level wrapper used when pool not initialized or in main thread
    global WORKER_ADJ_PACKED, WORKER_N, WORKER_W
    if NUMBA_AVAILABLE and WORKER_ADJ_PACKED is not None:
        try:
            return (solution_tuple, evaluate_solution_numba(solution_tuple))
        except Exception:
            # fallback to python
            return (solution_tuple, evaluate_solution_python(solution_tuple))
    else:
        return (solution_tuple, evaluate_solution_python(solution_tuple))

# The worker-local eval wrapper (set in initializer); default to top-level fallback
def eval_wrapper_in_worker(sol_tpl):
    if 'eval_wrapper_local' in globals():
        return globals()['eval_wrapper_local'](sol_tpl)
    else:
        return eval_wrapper(sol_tpl)

# ---------- Genetic operators & LS ----------
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

def inversion_mutation(perm: List[int]) -> List[int]:
    if len(perm) < 2:
        return perm[:]
    a, b = sorted(random.sample(range(len(perm)), 2))
    perm = perm[:]
    perm[a:b+1] = reversed(perm[a:b+1])
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
    shift = max(1, int(n * 0.05))
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

def targeted_repair(solution: List[int], max_tries: int = 200) -> List[int]:
    # Simple targeted repair based on offenders using the Python evaluator for masks (cheap)
    global WORKER_N
    n = WORKER_N
    if n is None:
        return solution
    sol_tpl = tuple(int(x) for x in solution)
    size, maxw, offenders, succ_masks = compute_maxwidth_offenders(sol_tpl)
    if maxw <= 1 or not offenders:
        return solution
    perm = solution[:-1]
    t = solution[-1]
    best = tuple(int(x) for x in solution)
    best_score = evaluate_solution_python(best) if not NUMBA_AVAILABLE else evaluate_solution_numba(best)
    tries = 0
    random.shuffle(offenders)
    window = max(4, int(n * 0.06))
    for u in offenders:
        if tries >= max_tries:
            break
        try:
            pos_u = perm.index(u)
        except ValueError:
            continue
        for shift in [1, 2, 4, 8]:
            if tries >= max_tries or shift > window:
                break
            new_pos = max(0, pos_u - shift)
            if new_pos == pos_u:
                continue
            new_perm = perm[:]
            new_perm.pop(pos_u)
            new_perm.insert(new_pos, u)
            candidate = tuple(new_perm + [t])
            cand_score = evaluate_solution_python(candidate) if not NUMBA_AVAILABLE else evaluate_solution_numba(candidate)
            tries += 1
            if (cand_score[0] > best_score[0]) or (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
                return list(candidate)
    return solution

def compute_maxwidth_offenders(solution_tuple: Tuple[int, ...]) -> Tuple[int,int,List[int],Dict[int,int]]:
    """
    Compute offenders list and succ_masks using python-level bit-int approach for convenience.
    Returns (size, max_width, offenders, succ_masks)
    """
    global WORKER_N, WORKER_ADJ_PACKED, WORKER_W
    n = WORKER_N
    sol = sanitize_solution_tuple(solution_tuple, n)
    perm = list(sol[:-1])
    t = int(sol[-1])
    size = n - t
    if size <= 0:
        return 0, 501, [], {}
    # build python int adjacency from packed
    W = WORKER_W
    adj_ints = [0] * n
    for u in range(n):
        vbits = 0
        for w in range(W):
            vbits |= int(WORKER_ADJ_PACKED[u, w]) << (w * 64)
        adj_ints[u] = vbits

    suffix_mask = [0] * n
    curr_mask = 0
    for i in range(n-1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (1 << perm[i])

    temp = adj_ints[:]
    max_width = 0
    offenders = []
    succ_masks = {}
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount_py(succ)
        if out_deg > max_width:
            max_width = out_deg
            offenders = [u]
            succ_masks = {u: succ}
        elif out_deg == max_width and out_deg > 0:
            offenders.append(u)
            succ_masks[u] = succ
        if succ == 0:
            continue
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= (succ ^ (1 << v))
    return size, max_width, offenders, succ_masks

# ---------- Local search worker ----------
def local_search_worker(args):
    sol, intensity = args
    n = WORKER_N
    best = tuple(int(x) for x in sol)
    best_score = evaluate_solution_numba(best) if (NUMBA_AVAILABLE and WORKER_ADJ_PACKED is not None) else evaluate_solution_python(best)
    for _ in range(intensity):
        r = random.random()
        if r < 0.22:
            neigh = block_move(list(best), n)
        elif r < 0.6:
            perm = list(best[:-1])
            perm = inversion_mutation(perm)
            neigh = perm + [best[-1]]
        else:
            neigh = smart_torso_shift(list(best), n)
        neigh_t = tuple(int(x) for x in neigh)
        neigh_score = evaluate_solution_numba(neigh_t) if (NUMBA_AVAILABLE and WORKER_ADJ_PACKED is not None) else evaluate_solution_python(neigh_t)
        if (neigh_score[0] > best_score[0]) or (neigh_score[0] == best_score[0] and neigh_score[1] < best_score[1]):
            best = neigh_t
            best_score = neigh_score
    # occasional targeted repair
    if random.random() < 0.08:
        cand = targeted_repair(list(best), max_tries=32)
        cand_t = tuple(int(x) for x in cand)
        cand_score = evaluate_solution_numba(cand_t) if (NUMBA_AVAILABLE and WORKER_ADJ_PACKED is not None) else evaluate_solution_python(cand_t)
        if (cand_score[0] > best_score[0]) or (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
            return list(cand_t)
    return list(best)

# ---------- Selection & hyper-helpers ----------
def dominates_internal(p, q):
    # p and q are (size, width)
    return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

def crowding_selection(population: List[Dict[str,Any]], pop_size: int) -> List[Dict[str,Any]]:
    for p in population:
        p.setdefault('dominates_set', [])
        p.setdefault('dominated_by_count', 0)
    fronts = [[]]
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
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0:
                    next_front.append(q)
        fronts.append(next_front)
        i += 1
    new_pop = []
    for front in fronts:
        if not front:
            continue
        if len(new_pop) + len(front) > pop_size:
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

# ---------- Persistence helpers ----------
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

# ---------- Seeding heuristics ----------
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

# ---------- Main memetic algorithm ----------
def memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict[str,Any], problem_id: str,
                      seed_vectors: List[List[int]] = None, use_numba: bool = True,
                      pool_processes: int = None, profile: bool = False, generations_override: int = None,
                      pop_size_override: int = None, intensity_override: int = None) -> List[List[int]]:
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    adj_packed, W = pack_adj_bits(adj_list, n)
    pop_size = config['pop_size'] if pop_size_override is None else pop_size_override
    generations = config['generations'] if generations_override is None else generations_override
    intensity = config.get('local_search_intensity', 18) if intensity_override is None else intensity_override

    # initial population
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
    # augment seeds
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
    # if user provided seed vectors, prefer them
    if seed_vectors:
        for s in seed_vectors:
            # sanitize each
            t = tuple(int(x) for x in s)
            st = sanitize_solution_tuple(t, n)
            population.append({'solution': list(st), 'score': None})
    while len(population) < pop_size:
        base = random.choice(seed_orders)
        perm = base[:]
        for _ in range(random.randint(0, 4)):
            perm = inversion_mutation(perm)
            perm = swap_mutation(perm)
        t = random.randint(int(n * 0.2), int(n * 0.8))
        population.append({'solution': perm + [t], 'score': None})

    # optionally resume
    start_gen = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            saved = pickle.load(f)
        if saved and 'pop' in saved and 'gen' in saved:
            population = saved['pop']
            start_gen = saved['gen'] + 1
            print(f"🔄 Resuming from gen {start_gen}")

    # set up pool
    if pool_processes is None:
        pool_processes = max(1, multiprocessing.cpu_count() - 1)
    use_numba_effective = use_numba and NUMBA_AVAILABLE

    # init main-thread globals for direct eval calls
    global WORKER_ADJ_PACKED, WORKER_W, WORKER_N
    WORKER_ADJ_PACKED = adj_packed
    WORKER_W = W
    WORKER_N = n

    # evaluate initial population
    if population and population[0].get('score') is None:
        sols = [tuple(int(x) for x in p['solution']) for p in population]
        if pool_processes <= 1:
            results = [eval_wrapper(sol) for sol in sols]
        else:
            with multiprocessing.Pool(processes=pool_processes, initializer=_init_worker,
                                      initargs=(adj_packed, W, n, use_numba_effective)) as pool:
                results = pool.map(eval_wrapper_in_worker, sols)
        sol_to_score = {sol: score for sol, score in results}
        for p in population:
            p['score'] = sol_to_score.get(tuple(p['solution']), (0, 501))

    # initialize best
    best_solution = None
    best_score = (-1, 1000)
    for p in population:
        if best_solution is None or (p['score'][0] > best_score[0]) or (p['score'][0] == best_score[0] and p['score'][1] < best_score[1]):
            best_solution = p['solution']
            best_score = p['score']
    persist_best(best_solution, best_score, problem_id)

    base_mutation = CONFIG['general']['mutation_rate']
    stagnation_counter = 0

    # Main generational loop
    for gen in tqdm(range(start_gen, generations), desc="🧬 Evolving", initial=start_gen, total=generations):
        mutation_rate = base_mutation
        if stagnation_counter >= CONFIG['general']['stagnation_limit']:
            mutation_rate = min(0.99, base_mutation * CONFIG['general']['mutation_boost_factor'])

        mating_pool = crowding_selection(population, pop_size)

        # produce offspring
        offspring_sols = []
        while len(offspring_sols) < pop_size:
            p1 = random.choice(mating_pool)
            p2 = random.choice(mating_pool)
            perm1 = list(p1['solution'][:-1])
            perm2 = list(p2['solution'][:-1])
            if random.random() < config.get('crossover_rate', 0.9):
                child_perm = pmx_crossover(perm1, perm2)
            else:
                child_perm = perm1[:]
            if random.random() < mutation_rate:
                if random.random() < 0.7:
                    child_perm = inversion_mutation(child_perm)
                else:
                    child_perm = swap_mutation(child_perm)
            c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
            if random.random() < 0.35:
                c_t = max(0, min(n - 1, c_t + random.randint(-int(n*0.03), int(n*0.03))))
            offspring_sols.append(child_perm + [c_t])

        # local search (parallel)
        ls_args = [(sol, intensity) for sol in offspring_sols]
        if pool_processes <= 1:
            improved_offspring = [local_search_worker(arg) for arg in ls_args]
        else:
            with multiprocessing.Pool(processes=pool_processes, initializer=_init_worker,
                                      initargs=(adj_packed, W, n, use_numba_effective)) as pool:
                improved_offspring = pool.map(local_search_worker, ls_args)

        # evaluate offspring
        eval_tuples = [tuple(int(x) for x in sol) for sol in improved_offspring]
        if pool_processes <= 1:
            eval_results = [eval_wrapper(sol) for sol in eval_tuples]
        else:
            with multiprocessing.Pool(processes=pool_processes, initializer=_init_worker,
                                      initargs=(adj_packed, W, n, use_numba_effective)) as pool:
                eval_results = pool.map(eval_wrapper_in_worker, eval_tuples)
        offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in eval_results]

        # combine and select
        population = crowding_selection(population + offspring_pop, pop_size)

        # Elite intensification
        E = CONFIG['general']['elite_count']
        elite_ls = CONFIG['general']['elite_ls_multiplier']
        pop_sorted = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
        elites = pop_sorted[:E]
        elite_args = [(e['solution'], max(1, intensity * elite_ls)) for e in elites]
        if elite_args:
            if pool_processes <= 1:
                improved_elites = [local_search_worker(arg) for arg in elite_args]
            else:
                with multiprocessing.Pool(processes=pool_processes, initializer=_init_worker,
                                          initargs=(adj_packed, W, n, use_numba_effective)) as pool:
                    improved_elites = pool.map(local_search_worker, elite_args)
            eval_tuples = [tuple(int(x) for x in sol) for sol in improved_elites]
            if pool_processes <= 1:
                eval_results = [eval_wrapper(sol) for sol in eval_tuples]
            else:
                with multiprocessing.Pool(processes=pool_processes, initializer=_init_worker,
                                          initargs=(adj_packed, W, n, use_numba_effective)) as pool:
                    eval_results = pool.map(eval_wrapper_in_worker, eval_tuples)
            for sol, score in eval_results:
                population.append({'solution': list(sol), 'score': score})
            population = crowding_selection(population, pop_size)

        # track best
        current_best = max(population, key=lambda p: (p['score'][0], -p['score'][1]))
        if (current_best['score'][0] > best_score[0]) or (current_best['score'][0] == best_score[0] and current_best['score'][1] < best_score[1]):
            best_solution = list(current_best['solution'])
            best_score = current_best['score']
            persist_best(best_solution, best_score, problem_id)
            stagnation_counter = 0
            tqdm.write(f"✨ Gen {gen+1}: New best {best_score} | archive_size N/A")
        else:
            stagnation_counter += 1

        tqdm.write(f"Gen {gen+1}: best(size,width)={best_score} stagn={stagnation_counter} mut={mutation_rate:.3f}")

        # checkpoint
        if (gen + 1) % config.get('checkpoint_interval', 5) == 0:
            tqdm.write(f"\n💾 Saving checkpoint at generation {gen + 1}...")
            to_save = {'pop': population, 'gen': gen}
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(to_save, f)

    # final selection
    final = crowding_selection(population, min(20, len(population)))
    return [p['solution'] for p in final]

# ---------- CLI & run ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default="easy", choices=list(PROBLEMS.keys()))
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--pop_size", type=int, default=None)
    p.add_argument("--intensity", type=int, default=None)
    p.add_argument("--seed", type=str, default=None, help="JSON file with decision vectors to seed (list of vectors)")
    p.add_argument("--no-numba", action='store_true', help="Disable Numba even if available")
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--profile", action='store_true')
    p.add_argument("--diagnostic", action='store_true', help="Run tiny diagnostic generation")
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    problem_id = args.problem
    if problem_id not in PROBLEMS:
        print("Invalid problem")
        return
    config = CONFIG['general'].copy()
    config.update(CONFIG[problem_id])

    if args.pop_size:
        config['pop_size'] = args.pop_size
    if args.generations:
        config['generations'] = args.generations
    if args.intensity:
        config['local_search_intensity'] = args.intensity

    n, adj = load_graph(problem_id)

    seed_vectors = None
    if args.seed:
        try:
            with open(args.seed, "r") as f:
                j = json.load(f)
            # expected format: list of vectors
            if isinstance(j, dict) and 'decisionVector' in j:
                seed_vectors = j['decisionVector']
            elif isinstance(j, list):
                seed_vectors = j
            else:
                print("Unknown seed JSON format")
                seed_vectors = None
            if seed_vectors:
                print(f"✅ Loaded {len(seed_vectors)} seed vectors from {args.seed}")
        except Exception as e:
            print("Warning: couldn't load seed file:", e)

    use_numba = (not args.no_numba) and NUMBA_AVAILABLE
    if not NUMBA_AVAILABLE and not args.no_numba:
        print("⚠️ Numba not available; running with pure-Python evaluator (much slower). Install numba for speedup.")

    start_time = time.time()
    final_solutions = memetic_algorithm(n, adj, config, problem_id, seed_vectors=seed_vectors,
                                       use_numba=use_numba, pool_processes=args.processes,
                                       profile=args.profile, generations_override=args.generations,
                                       pop_size_override=args.pop_size, intensity_override=args.intensity)
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total Optimization Time: {elapsed:.2f} seconds")
    create_submission_file(final_solutions, problem_id)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
