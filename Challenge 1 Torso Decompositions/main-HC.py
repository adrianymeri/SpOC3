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
    # Increase these values for longer, more thorough runs on a server
    "easy": {"pop_size": 100, "generations": 200, "local_search_intensity": 20},
    "medium": {"pop_size": 120, "generations": 400, "local_search_intensity": 25},
    "hard": {"pop_size": 150, "generations": 600, "local_search_intensity": 30},
    "mutation_rate": 0.5,
    "crossover_rate": 0.9,
    "checkpoint_interval": 10, # Save progress every 10 generations
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# Global cache for evaluated solutions
MEMO_EVAL = {}

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

def evaluate_solution_correct(args: Tuple[Tuple[int, ...], int, List[Set[int]]]) -> Tuple[Tuple[int, ...], Tuple[int, int]]:
    """
    Correctly evaluates a single solution, including fill-in edges.
    Designed to be called by multiprocessing.
    Returns the solution tuple and its score for mapping back.
    """
    solution_tuple, n, adj = args
    if solution_tuple in MEMO_EVAL:
        return solution_tuple, MEMO_EVAL[solution_tuple]

    t = solution_tuple[-1]
    perm = solution_tuple[:-1]
    size = n - t
    if size <= 0:
        return solution_tuple, (0, 501)

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
        if max_width >= 500:
            result = (size, 501)
            MEMO_EVAL[solution_tuple] = result
            return solution_tuple, result
            
    result = (size, max_width)
    MEMO_EVAL[solution_tuple] = result
    return solution_tuple, result

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

# --- Adaptive Memetic Components & NSGA-II Framework ---

class AdaptiveLocalSearcher:
    def __init__(self, n, adj):
        self.operators = [smart_torso_shift, block_move]
        self.weights = np.ones(len(self.operators))
        self.n, self.adj = n, adj

    def apply(self, solution: List[int], intensity: int) -> List[int]:
        current_sol = solution
        _, best_score = evaluate_solution_correct((tuple(current_sol), self.n, self.adj))
        
        for _ in range(intensity):
            op_idx = np.random.choice(len(self.operators), p=self.weights / self.weights.sum())
            op = self.operators[op_idx]
            
            neighbor = op(current_sol, self.n)
            _, neighbor_score = evaluate_solution_correct((tuple(neighbor), self.n, self.adj))

            if dominates(neighbor_score, best_score):
                current_sol = neighbor
                best_score = neighbor_score
                self.weights[op_idx] += 0.2
        
        self.weights = np.clip(self.weights, 0.1, 10)
        return current_sol

def dominates(p, q): return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    # Non-Dominated Sort
    for p in population: p['dominates_set'], p['dominated_by_count'] = [], 0
    fronts = [[]]
    for i, p in enumerate(population):
        for j, q in enumerate(population[i+1:]):
            if dominates(p['score'], q['score']):
                p['dominates_set'].append(q)
                q['dominated_by_count'] += 1
            elif dominates(q['score'], p['score']):
                q['dominates_set'].append(p)
                p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0:
                    next_front.append(q)
        fronts.append(next_front)
        i += 1
    
    # Crowding Distance and Selection
    new_population = []
    for front in fronts:
        if not front: continue
        if len(new_population) + len(front) > pop_size:
            # Calculate crowding distance
            for p in front: p['distance'] = 0
            for i in range(2):
                front.sort(key=lambda p: p['score'][i])
                front[0]['distance'] = front[-1]['distance'] = float('inf')
                f_min, f_max = front[0]['score'][i], front[-1]['score'][i]
                if f_max > f_min:
                    for j in range(1, len(front) - 1):
                        front[j]['distance'] += (front[j+1]['score'][i] - front[j-1]['score'][i]) / (f_max - f_min)
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
            population = saved_state['population']
            start_gen = saved_state['generation'] + 1
            adaptive_ls = saved_state['adaptive_ls']
            global MEMO_EVAL
            MEMO_EVAL = saved_state['memo_eval']
        print(f"Resuming from generation {start_gen + 1}")
    else:
        print("🌱 Initializing fresh population...")
        population = [{'solution': list(np.random.permutation(n)) + [random.randint(0, n-1)]} for _ in range(config['pop_size'])]
        adaptive_ls = AdaptiveLocalSearcher(n, adj)

    with multiprocessing.Pool() as pool:
        # Initial evaluation
        if start_gen == 0:
            eval_args = [(tuple(p['solution']), n, adj) for p in population]
            results = pool.map(evaluate_solution_correct, eval_args)
            sol_to_score = {sol: score for sol, score in results}
            for p in population: p['score'] = sol_to_score[tuple(p['solution'])]

        # Main Loop
        for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving Population", initial=start_gen, total=config['generations']):
            mating_pool = crowding_selection(population, config['pop_size'])
            
            offspring_sols = []
            while len(offspring_sols) < config['pop_size']:
                p1, p2 = random.sample(mating_pool, 2)
                c_perm = p1['solution'][:-1]
                if random.random() < config['crossover_rate']:
                    start, end = sorted(random.sample(range(n), 2))
                    p2_slice = [item for item in p2['solution'][:-1] if item not in c_perm[start:end]]
                    c_perm = p2_slice[:start] + c_perm[start:end] + p2_slice[start:]
                if random.random() < config['mutation_rate']:
                    idx1, idx2 = random.sample(range(n), 2)
                    c_perm[idx1], c_perm[idx2] = c_perm[idx2], c_perm[idx1]
                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                offspring_sols.append(c_perm + [c_t])
            
            # Memetic Step: Apply local search to all new offspring
            improved_offspring = [adaptive_ls.apply(sol, config['local_search_intensity']) for sol in offspring_sols]
            
            # Evaluate new solutions in parallel
            eval_args = [(tuple(sol), n, adj) for sol in improved_offspring]
            results = pool.map(evaluate_solution_correct, eval_args)
            sol_to_score = {sol: score for sol, score in results}

            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in sol_to_score.items()]
            
            population = mating_pool + offspring_pop

            if (gen + 1) % config['checkpoint_interval'] == 0:
                print(f"\n💾 Saving checkpoint at generation {gen + 1}...")
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({
                        'population': crowding_selection(population, config['pop_size']),
                        'generation': gen,
                        'adaptive_ls': adaptive_ls,
                        'memo_eval': dict(list(MEMO_EVAL.items())[-50000:])
                    }, f)

    final_population = crowding_selection(population, config['pop_size'])
    return [p['solution'] for p in final_population]

# --- Main Execution ---

def create_submission_file(decision_vectors: List[List[int]], problem_id: str):
    filename = f"submission_{problem_id}.json"
    # The submission format requires a list of decision vectors
    submission = {
        "decisionVector": decision_vectors[:20], # Max 20 solutions
        "problem": f"{problem_id}-graph" if problem_id != "easy" else "small-graph", # Adjusting for submission names
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(submission['decisionVector'])} solutions.")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    problem_id = input("🔍 Select problem (easy/medium/hard): ").lower()
    if problem_id not in PROBLEMS:
        print("❌ Invalid problem ID. Exiting.")
        exit()

    n, adj = load_graph(problem_id)
    problem_config = CONFIG[problem_id]
    
    start_time = time.time()
    final_solutions = memetic_algorithm(n, adj, problem_config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id)
