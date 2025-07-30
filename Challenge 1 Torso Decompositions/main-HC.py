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

# --- Algorithm & Problem Configuration ---
CONFIG = {
    "general": {
        "mutation_rate": 0.6,
        "crossover_rate": 0.9,
        "checkpoint_interval": 5,
    },
    "easy": {"pop_size": 100, "generations": 200, "local_search_intensity": 20},
    "medium": {"pop_size": 150, "generations": 400, "local_search_intensity": 25},
    "hard": {"pop_size": 200, "generations": 800, "local_search_intensity": 30},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Core Data Loading and Correct Evaluation ---

def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    """Loads graph data and returns the number of nodes and an adjacency list."""
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

def evaluate_solution_task(args: Tuple[Tuple[int, ...], int, List[Set[int]]]) -> Tuple[Tuple[int, ...], Tuple[int, int]]:
    """Worker function for multiprocessing. Correctly evaluates a single solution."""
    solution_tuple, n, adj = args
    
    t = solution_tuple[-1]
    perm = solution_tuple[:-1]
    size = n - t
    if size <= 0: return solution_tuple, (0, 501)

    pos = {node: i for i, node in enumerate(perm)}
    
    temp_adj = [s.copy() for s in adj]
    for i in range(n):
        u = perm[i]
        successors = [v for v in temp_adj[u] if pos.get(v, -1) > i]
        for j1 in range(len(successors)):
            for j2 in range(j1 + 1, len(successors)):
                v1, v2 = successors[j1], successors[j2]
                if v2 not in temp_adj[v1]:
                    temp_adj[v1].add(v2)
                    temp_adj[v2].add(v1)
    
    max_width = 0
    for u in perm[t:]:
        out_degree = sum(1 for v in temp_adj[u] if pos.get(v, -1) > pos[u])
        max_width = max(max_width, out_degree)
        if max_width >= 500: return solution_tuple, (size, 501)
            
    return solution_tuple, (size, max_width)

# --- Your Proven Local Search Operators ---

def smart_torso_shift(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    t = neighbor[-1]
    shift = int(n * 0.05) + 1
    neighbor[-1] = max(0, min(n - 1, t + random.randint(-shift, shift)))
    return neighbor

def block_move(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    perm = neighbor[:-1]
    block_size = random.randint(3, max(4, int(n * 0.02)))
    if n > block_size:
        start = random.randint(0, n - block_size)
        block = perm[start:start + block_size]
        del perm[start:start + block_size]
        insert_pos = random.randint(0, len(perm))
        perm[insert_pos:insert_pos] = block
        neighbor[:-1] = perm
    return neighbor

# --- Memetic & NSGA-II Components ---

class AdaptiveLocalSearcher:
    """Manages and adaptively applies local search operators."""
    def __init__(self, n, adj):
        self.operators = [smart_torso_shift, block_move]
        self.weights = np.ones(len(self.operators))
        self.n, self.adj = n, adj

    def apply(self, args: Tuple[List[int], int]) -> List[int]:
        solution, intensity = args
        current_sol = solution
        _, best_score = evaluate_solution_task((tuple(current_sol), self.n, self.adj))
        
        for _ in range(intensity):
            op_idx = np.random.choice(len(self.operators), p=self.weights / self.weights.sum())
            op = self.operators[op_idx]
            
            neighbor = op(current_sol, self.n)
            _, neighbor_score = evaluate_solution_task((tuple(neighbor), self.n, self.adj))

            if dominates(neighbor_score, best_score):
                current_sol = neighbor
                best_score = neighbor_score
                self.weights[op_idx] += 0.1
        return current_sol

def dominates(p, q): return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    """Selects the new population based on non-domination rank and crowding distance."""
    # Non-Dominated Sort
    for p in population: p['dominates_set'], p['dominated_by_count'] = [], 0
    fronts = [[]]
    for i, p in enumerate(population):
        for j, q in enumerate(population[i+1:]):
            if dominates(p['score'], q['score']): p['dominates_set'].append(q); q['dominated_by_count'] += 1
            elif dominates(q['score'], p['score']): q['dominates_set'].append(p); p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0: fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0: next_front.append(q)
        fronts.append(next_front)
        i += 1
    
    # Crowding Distance and Final Selection
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

# --- Main Memetic Algorithm Loop ---

def memetic_algorithm(n: int, adj: List[Set[int]], config: Dict, problem_id: str) -> List[List[int]]:
    start_gen = 0
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    
    if os.path.exists(checkpoint_file):
        print(f"🔄 Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            saved_state = pickle.load(f)
        population, start_gen, adaptive_ls = saved_state['pop'], saved_state['gen'] + 1, saved_state['ls']
        print(f"Resuming from generation {start_gen}")
    else:
        print("🌱 Initializing fresh population...")
        population = [{'solution': list(np.random.permutation(n)) + [random.randint(int(n*0.2), int(n*0.8))]} for _ in range(config['pop_size'])]
        adaptive_ls = AdaptiveLocalSearcher(n, adj)

    with multiprocessing.Pool() as pool:
        if start_gen == 0:
            print("Evaluating initial population...")
            results = pool.map(evaluate_solution_task, [(tuple(p['solution']), n, adj) for p in population])
            sol_to_score = dict(results)
            for p in population: p['score'] = sol_to_score.get(tuple(p['solution']), (0, 501))

        for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving", initial=start_gen, total=config['generations']):
            mating_pool = crowding_selection(population, config['pop_size'])
            
            offspring_sols = []
            while len(offspring_sols) < config['pop_size']:
                p1, p2 = random.sample(mating_pool, 2)
                c_perm = list(p1['solution'][:-1])
                if random.random() < config['crossover_rate']:
                    start, end = sorted(random.sample(range(n), 2))
                    p2_slice = [item for item in p2['solution'][:-1] if item not in c_perm[start:end]]
                    c_perm = p2_slice[:start] + c_perm[start:end] + p2_slice[start:]
                if random.random() < config['mutation_rate']:
                    idx1, idx2 = random.sample(range(n), 2)
                    c_perm[idx1], c_perm[idx2] = c_perm[idx2], c_perm[idx1]
                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                offspring_sols.append(c_perm + [c_t])
            
            ls_args = [(sol, config['local_search_intensity']) for sol in offspring_sols]
            improved_offspring = pool.map(adaptive_ls.apply, ls_args)
            
            eval_results = pool.map(evaluate_solution_task, [(tuple(sol), n, adj) for sol in improved_offspring])
            
            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in eval_results]
            population = crowding_selection(population + offspring_pop, config['pop_size'])

            if (gen + 1) % config['checkpoint_interval'] == 0:
                tqdm.write(f"\n💾 Saving checkpoint at generation {gen + 1}...")
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'pop': population, 'gen': gen, 'ls': adaptive_ls}, f)

    return [p['solution'] for p in crowding_selection(population, 20)]

# --- Main Execution ---

def create_submission_file(decision_vectors: List[List[int]], problem_id: str):
    filename = f"submission_{problem_id}.json"
    problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
    
    # **FIXED HERE**: Convert all numbers to standard Python integers for JSON compatibility
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

    problem_id = input("🔍 Select problem (easy/medium/hard): ").lower()
    if problem_id not in PROBLEMS:
        print("❌ Invalid problem ID. Exiting.")
        exit()

    config = CONFIG['general'].copy()
    config.update(CONFIG[problem_id])

    n, adj = load_graph(problem_id)
    
    start_time = time.time()
    final_solutions = memetic_algorithm(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id)
