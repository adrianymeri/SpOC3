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
# With the fast heuristic, we can afford much larger populations and many more generations.
CONFIG = {
    "general": {
        "mutation_rate": 0.5,
        "checkpoint_interval": 50,
        "stagnation_limit": 100, # Generations without heuristic improvement before a reset
    },
    "easy": {"pop_size": 200, "generations": 2000, "local_search_intensity": 25},
    "medium": {"pop_size": 250, "generations": 3000, "local_search_intensity": 30},
    "hard": {"pop_size": 300, "generations": 5000, "local_search_intensity": 35},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Evaluation Functions ---

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

def evaluate_heuristic_task(args: Tuple[Tuple[int, ...], int, List[Set[int]]]) -> Tuple[Tuple[int, ...], Tuple[int, int]]:
    """Your original, high-speed heuristic evaluation. This guides the main search."""
    solution_tuple, n, adj = args
    t = solution_tuple[-1]
    size = n - t
    if size <= 0: return solution_tuple, (0, 501)
    
    perm = solution_tuple[:-1]
    max_width = 0
    for i in range(t, n):
        node = perm[i]
        width = len(adj[node])
        if width > max_width:
            max_width = width
    return solution_tuple, (size, max_width if max_width < 500 else 501)

def evaluate_correct_task(args: Tuple[Tuple[int, ...], int, List[Set[int]]]) -> Tuple[Tuple[int, ...], Tuple[int, int]]:
    """The slow, but 100% correct evaluation function for final scoring."""
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

# --- Neighborhood Operators ---

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

def inversion_mutation_op(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    perm = neighbor[:-1]
    start, end = sorted(random.sample(range(n), 2))
    perm[start:end+1] = reversed(perm[start:end+1])
    neighbor[:-1] = perm
    return neighbor

# --- Advanced Genetic & Local Search Operators ---

def edge_recombination_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Edge Recombination Crossover (ERX)."""
    n = len(p1)
    adj_map = {node: set() for node in p1}
    for p in [p1, p2]:
        for i in range(n):
            adj_map[p[i]].add(p[i-1])
            adj_map[p[i]].add(p[(i+1)%n])

    current_node = p1[0]
    child = [current_node]
    unvisited = set(p1) - {current_node}

    while len(child) < n:
        for neighbor in unvisited:
            if current_node in adj_map.get(neighbor, set()):
                adj_map[neighbor].remove(current_node)
        
        neighbors_in_unvisited = [node for node in adj_map.get(current_node, set()) if node in unvisited]
        if not neighbors_in_unvisited:
            next_node = random.choice(list(unvisited))
        else:
            min_len = min(len(adj_map[node]) for node in neighbors_in_unvisited)
            next_node = random.choice([node for node in neighbors_in_unvisited if len(adj_map[node]) == min_len])
        
        child.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
    return child

class VariableNeighborhoodSearcher:
    """Applies VNS to a solution using the fast heuristic."""
    def __init__(self, n, adj):
        self.neighborhoods = [block_move, smart_torso_shift, inversion_mutation_op]
        self.n, self.adj = n, adj

    def apply(self, args: Tuple[List[int], int]) -> List[int]:
        solution, intensity = args
        best_sol = solution
        _, best_score = evaluate_heuristic_task((tuple(best_sol), self.n, self.adj))

        k = 0
        iters_since_improvement = 0
        # VNS will try each neighborhood type until no improvement is found in any of them
        while k < len(self.neighborhoods):
            op = self.neighborhoods[k]
            improved = False
            for _ in range(intensity): # Try to improve within this neighborhood
                neighbor = op(best_sol, self.n)
                _, neighbor_score = evaluate_heuristic_task((tuple(neighbor), self.n, self.adj))
                
                if dominates(neighbor_score, best_score):
                    best_sol = neighbor
                    best_score = neighbor_score
                    k = 0 # Go back to the first neighborhood upon improvement
                    improved = True
                    break 
            if not improved:
                k += 1 # Move to the next neighborhood type
                
        return best_sol

# --- NSGA-II Selection ---
def dominates(p, q): return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])
def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
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
    best_overall_score = (0, float('inf'))
    generations_since_improvement = 0
    
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
            print("Evaluating initial population with fast heuristic...")
            results = pool.map(evaluate_heuristic_task, [(tuple(p['solution']), n, adj) for p in population])
            for p, (sol_t, score) in zip(population, results): p['score'] = score

        for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving", initial=start_gen, total=config['generations']):
            current_ls_intensity = int(config['local_search_intensity'] * (1 - gen / config['generations'])**0.5) + 5
            mating_pool = crowding_selection(population, config['pop_size'])
            
            current_best_score = min([p['score'] for p in mating_pool], key=lambda x: (x[1], -x[0]))
            if dominates(current_best_score, best_overall_score):
                generations_since_improvement = 0
                best_overall_score = current_best_score
            else:
                generations_since_improvement += 1

            if generations_since_improvement >= config['stagnation_limit']:
                tqdm.write(f"\n⚠️ Stagnation detected! Resetting 30% of population.")
                num_to_reset = int(0.3 * len(population))
                reset_indices = random.sample(range(len(population)), num_to_reset)
                for i in reset_indices:
                    population[i] = {'solution': list(np.random.permutation(n)) + [random.randint(int(n*0.2), int(n*0.8))]}
                
                results = pool.map(evaluate_heuristic_task, [(tuple(p['solution']), n, adj) for p in [population[i] for i in reset_indices]])
                for p, (sol_t, score) in zip([population[i] for i in reset_indices], results): p['score'] = score
                generations_since_improvement = 0

            offspring_sols = []
            while len(offspring_sols) < config['pop_size']:
                p1, p2 = random.sample(mating_pool, 2)
                c_perm = edge_recombination_crossover(p1['solution'][:-1], p2['solution'][:-1])
                if random.random() < config['mutation_rate']:
                    c_perm = inversion_mutation_op(c_perm, n)
                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                offspring_sols.append(c_perm + [c_t])
            
            ls_args = [(sol, current_ls_intensity) for sol in offspring_sols]
            improved_offspring = pool.map(vns.apply, ls_args)
            
            eval_args = [(tuple(sol), n, adj) for sol in improved_offspring]
            results = pool.map(evaluate_heuristic_task, eval_args)
            
            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in results]
            population = crowding_selection(population + offspring_pop, config['pop_size'])

            if (gen + 1) % config['checkpoint_interval'] == 0:
                tqdm.write(f"\n💾 Saving checkpoint at generation {gen + 1}...")
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'pop': population, 'gen': gen, 'vns': vns}, f)

    # *** FINAL RE-EVALUATION STEP ***
    print(f"\n🔬 Performing final accurate evaluation of {len(population)} elite solutions...")
    final_elite_solutions = [p['solution'] for p in crowding_selection(population, 40)] # Re-evaluate more than we need
    final_results = pool.map(evaluate_correct_task, [(tuple(sol), n, adj) for sol in final_elite_solutions])
    
    final_population = [{'solution': list(sol), 'score': score} for sol, score in final_results]
    
    return [p['solution'] for p in crowding_selection(final_population, 20)]

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
