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
# These are aggressive settings for a university server.
# For your final runs, these parameters should be tuned using a tool like Optuna.
CONFIG = {
    "general": {
        "mutation_rate": 0.5,
        "checkpoint_interval": 5,
    },
    "easy": {"pop_size": 100, "generations": 200, "local_search_intensity": 25, "vns_shake_strength": 0.1},
    "medium": {"pop_size": 150, "generations": 400, "local_search_intensity": 30, "vns_shake_strength": 0.15},
    "hard": {"pop_size": 200, "generations": 800, "local_search_intensity": 35, "vns_shake_strength": 0.2},
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

# --- Advanced Genetic & Local Search Operators ---

def edge_recombination_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Edge Recombination Crossover (ERX)."""
    n = len(p1)
    adj_map = [set() for _ in range(n)]
    for p in [p1, p2]:
        for i in range(n):
            adj_map[p[i]].add(p[i-1])
            adj_map[p[i]].add(p[(i+1)%n])

    current_node = p1[0]
    child = [current_node]
    unvisited = set(p1) - {current_node}

    while len(child) < n:
        adj_map[current_node].discard(current_node)
        neighbors = list(adj_map[current_node])
        
        for neighbor in unvisited:
            if neighbor in neighbors:
                adj_map[neighbor].discard(current_node)
        
        if not neighbors or not any(n in unvisited for n in neighbors):
            next_node = random.choice(list(unvisited))
        else:
            neighbors_in_unvisited = [n for n in neighbors if n in unvisited]
            min_len = min(len(adj_map[n]) for n in neighbors_in_unvisited)
            next_node = random.choice([n for n in neighbors_in_unvisited if len(adj_map[n]) == min_len])
        
        child.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
        
    return child

def inversion_mutation(perm: List[int]) -> List[int]:
    """Inversion mutation for permutations."""
    size = len(perm)
    start, end = sorted(random.sample(range(size), 2))
    perm[start:end+1] = reversed(perm[start:end+1])
    return perm

class VariableNeighborhoodSearcher:
    """Applies VNS to a solution to find a better local optimum."""
    def __init__(self, n, adj):
        self.neighborhoods = [block_move, smart_torso_shift]
        self.n, self.adj = n, adj

    def apply(self, args: Tuple[List[int], int, float]) -> List[int]:
        solution, intensity, shake_strength = args
        best_sol = solution
        _, best_score = evaluate_solution_task((tuple(best_sol), self.n, self.adj))

        k = 0
        while k < len(self.neighborhoods):
            # Exploration within the current neighborhood
            for _ in range(intensity):
                op = self.neighborhoods[k]
                neighbor = op(best_sol, self.n)
                _, neighbor_score = evaluate_solution_task((tuple(neighbor), self.n, self.adj))
                
                if dominates(neighbor_score, best_score):
                    best_sol = neighbor
                    best_score = neighbor_score
                    k = 0 # Go back to the first neighborhood
                    continue
            k += 1
        return best_sol

# (Dominates and Crowding Selection functions remain the same as the previous correct version)
def dominates(p, q): return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])
def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
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
        with open(checkpoint_file, 'rb') as f: saved_state = pickle.load(f)
        population, start_gen, vns = saved_state['pop'], saved_state['gen'] + 1, saved_state['vns']
        print(f"Resuming from generation {start_gen}")
    else:
        print("🌱 Initializing fresh population...")
        population = [{'solution': list(np.random.permutation(n)) + [random.randint(int(n*0.2), int(n*0.8))]} for _ in range(config['pop_size'])]
        vns = VariableNeighborhoodSearcher(n, adj)

    with multiprocessing.Pool() as pool:
        if start_gen == 0:
            results = pool.map(evaluate_solution_task, [(tuple(p['solution']), n, adj) for p in population])
            for p, (sol_t, score) in zip(population, results): p['score'] = score

        for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving", initial=start_gen, total=config['generations']):
            mating_pool = crowding_selection(population, config['pop_size'])
            
            offspring_sols = []
            while len(offspring_sols) < config['pop_size']:
                p1, p2 = random.sample(mating_pool, 2)
                c_perm = edge_recombination_crossover(p1['solution'][:-1], p2['solution'][:-1])
                if random.random() < config['mutation_rate']:
                    c_perm = inversion_mutation(c_perm)
                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                offspring_sols.append(c_perm + [c_t])
            
            ls_args = [(sol, config['local_search_intensity'], config['vns_shake_strength']) for sol in offspring_sols]
            improved_offspring = pool.map(vns.apply, ls_args)
            
            eval_args = [(tuple(sol), n, adj) for sol in improved_offspring]
            results = pool.map(evaluate_solution_task, eval_args)
            
            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in results]
            population = crowding_selection(population + offspring_pop, config['pop_size'])

            if (gen + 1) % config['checkpoint_interval'] == 0:
                tqdm.write(f"\n💾 Saving checkpoint at generation {gen + 1}...")
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'pop': population, 'gen': gen, 'vns': vns}, f)

    return [p['solution'] for p in crowding_selection(population, 20)]

# --- Main Execution & Submission ---

def create_submission_file(decision_vectors: List[List[int]], problem_id: str):
    filename = f"submission_{problem_id}.json"
    problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
    final_vectors = [[int(val) for val in vec] for vec in decision_vectors]
    submission = { "decisionVector": final_vectors, "problem": problem_name_map.get(problem_id, problem_id), "challenge": "spoc-3-torso-decompositions" }
    with open(filename, "w") as f: json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(decision_vectors)} solutions.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(42)
    np.random.seed(42)

    problem_id = input("🔍 Select problem (easy/medium/hard): ").lower()
    if problem_id not in PROBLEMS: exit("❌ Invalid problem ID. Exiting.")

    config = CONFIG['general'].copy()
    config.update(CONFIG[problem_id])

    n, adj = load_graph(problem_id)
    
    start_time = time.time()
    final_solutions = memetic_algorithm(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id)
