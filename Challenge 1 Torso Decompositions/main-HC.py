import json
import random
import time
import math
import numpy as np
from typing import List, Set, Tuple, Dict
import urllib.request
from tqdm import tqdm

# --- Problem Configuration ---
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Core Data Loading and Evaluation ---

def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    """Loads graph data and returns the number of nodes and an adjacency list."""
    url = problems[problem_id]
    print(f"📥 Loading graph data from: {url}")
    edges = []
    max_node = 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b"#"):
                continue
            u, v = map(int, line.strip().split())
            edges.append((u, v))
            max_node = max(max_node, u, v)
    n = max_node + 1
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges")
    return n, adj

def evaluate_solution(decision_vector: List[int], n: int, adj: List[Set[int]]) -> Tuple[int, int]:
    """
    Correctly and efficiently evaluates the solution, including fill-in edges.
    This is the accurate evaluation based on the problem description.
    """
    t = decision_vector[-1]
    size = n - t
    if size <= 0:
        return 0, 501

    perm = decision_vector[:-1]
    pos = {node: i for i, node in enumerate(perm)}

    # Create a temporary, mutable copy of the graph for this evaluation
    temp_adj = [s.copy() for s in adj]

    # Add fill-in edges
    for i in range(n):
        u = perm[i]
        # Neighbors of u that appear after it in the permutation
        successors = [v for v in temp_adj[u] if pos.get(v, -1) > i]
        for j1 in range(len(successors)):
            for j2 in range(j1 + 1, len(successors)):
                v1, v2 = successors[j1], successors[j2]
                if v2 not in temp_adj[v1]:
                    temp_adj[v1].add(v2)
                    temp_adj[v2].add(v1)
    
    # Calculate width on the new graph
    max_width = 0
    for u in perm[t:]:
        out_degree = sum(1 for v in temp_adj[u] if pos.get(v, -1) > pos[u])
        max_width = max(max_width, out_degree)
        if max_width >= 500:
            return size, 501

    return size, max_width

# --- NSGA-II Components ---

def dominates(p: Tuple[int, int], q: Tuple[int, int]) -> bool:
    """Checks if solution score p dominates score q."""
    return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

def non_dominated_sort(population: List[Dict]) -> List[List[Dict]]:
    """Sorts the population into Pareto fronts."""
    pop_size = len(population)
    for p in population:
        p['dominates_set'] = []
        p['dominated_by_count'] = 0

    fronts = [[]]
    for i in range(pop_size):
        p = population[i]
        for j in range(i + 1, pop_size):
            q = population[j]
            if dominates(p['score'], q['score']):
                p['dominates_set'].append(q)
                q['dominated_by_count'] += 1
            elif dominates(q['score'], p['score']):
                q['dominates_set'].append(p)
                p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0:
            p['rank'] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0:
                    q['rank'] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]

def crowding_distance(front: List[Dict]):
    """Calculates crowding distance for a front."""
    if not front: return
    
    for p in front: p['distance'] = 0
    
    for i in range(2): # For each objective
        front.sort(key=lambda p: p['score'][i])
        front[0]['distance'] = front[-1]['distance'] = float('inf')
        f_min, f_max = front[0]['score'][i], front[-1]['score'][i]
        if f_max == f_min: continue
        
        for j in range(1, len(front) - 1):
            front[j]['distance'] += (front[j+1]['score'][i] - front[j-1]['score'][i]) / (f_max - f_min)

# --- Genetic Operators (Permutation-Safe) ---

def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Order Crossover (OX1) for permutations."""
    size = len(parent1)
    p1, p2 = parent1[:], parent2[:]
    
    start, end = sorted(random.sample(range(size), 2))
    
    def ox_core(p_a, p_b):
        child = [None] * size
        child[start:end+1] = p_a[start:end+1]
        
        p_b_remaining = [item for item in p_b if item not in child]
        
        idx = 0
        for i in list(range(start)) + list(range(end + 1, size)):
            child[i] = p_b_remaining[idx]
            idx += 1
        return child

    return ox_core(p1, p2), ox_core(p2, p1)

def inversion_mutation(perm: List[int]) -> List[int]:
    """Inversion mutation for permutations."""
    size = len(perm)
    start, end = sorted(random.sample(range(size), 2))
    perm[start:end+1] = reversed(perm[start:end+1])
    return perm

# --- Memetic Algorithm (HGA) ---

def local_search(solution: List[int], n: int, adj: List[Set[int]], max_ls_iter: int) -> List[int]:
    """A short, aggressive local search using a simple swap."""
    best_sol = solution
    best_score = evaluate_solution(best_sol, n, adj)

    for _ in range(max_ls_iter):
        neighbor = best_sol[:]
        # Swap two random positions in the permutation
        perm = neighbor[:-1]
        idx1, idx2 = random.sample(range(n), 2)
        perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
        neighbor[:-1] = perm
        
        # Or, sometimes mutate the threshold
        if random.random() < 0.2:
            neighbor[-1] = random.randint(0, n - 1)
            
        neighbor_score = evaluate_solution(neighbor, n, adj)
        
        if dominates(neighbor_score, best_score):
            best_sol = neighbor
            best_score = neighbor_score
            
    return best_sol

def hybrid_genetic_algorithm(n: int, adj: List[Set[int]], pop_size: int, n_generations: int) -> List[List[int]]:
    """The main HGA loop."""
    # Initialization
    population = [{'solution': perm + [random.randint(0, n-1)]} for perm in [list(np.random.permutation(n)) for _ in range(pop_size)]]
    for p in population:
        p['score'] = evaluate_solution(p['solution'], n, adj)

    # Main Loop
    for gen in tqdm(range(n_generations), desc="🧬 Evolving Population"):
        # Selection
        fronts = non_dominated_sort(population)
        mating_pool = []
        for front in fronts:
            crowding_distance(front)
            mating_pool.extend(front)

        # Crossover & Mutation
        offspring = []
        for i in range(pop_size // 2):
            p1 = random.choice(mating_pool)
            p2 = random.choice(mating_pool)
            
            c1_perm, c2_perm = order_crossover(p1['solution'][:-1], p2['solution'][:-1])
            c1_perm = inversion_mutation(c1_perm) if random.random() < 0.4 else c1_perm
            c2_perm = inversion_mutation(c2_perm) if random.random() < 0.4 else c2_perm

            # Child threshold from parents
            t1 = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
            
            c1 = {'solution': c1_perm + [t1]}
            c2 = {'solution': c2_perm + [t1]} # Both children get the same initial t
            
            offspring.extend([c1, c2])

        # Local Search (Memetic Step) & Evaluation
        ls_intensity = 20 # Number of iterations for the local search
        for child in offspring:
            child['solution'] = local_search(child['solution'], n, adj, ls_intensity)
            child['score'] = evaluate_solution(child['solution'], n, adj)
            
        # Create new population
        combined_population = population + offspring
        new_fronts = non_dominated_sort(combined_population)
        
        population = []
        for front in new_fronts:
            if len(population) + len(front) > pop_size:
                crowding_distance(front)
                front.sort(key=lambda p: p['distance'], reverse=True)
                population.extend(front[:pop_size - len(population)])
                break
            population.extend(front)

    # Final result
    final_front = non_dominated_sort(population)[0]
    return [p['solution'] for p in final_front]

# --- Main Execution ---
def create_submission_file(decision_vectors, problem_id, filename="submission.json"):
    submission = {
        "decisionVector": decision_vectors,
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(decision_vectors)} solutions.")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    problem_id = input("🔍 Select problem (easy/medium/hard): ").lower()
    if problem_id not in problems:
        print("❌ Invalid problem ID. Exiting.")
        exit()

    n, adj = load_graph(problem_id)
    
    # --- Parameters for different problem sizes ---
    pop_size_map = {"easy": 80, "medium": 100, "hard": 120}
    gen_map = {"easy": 100, "medium": 150, "hard": 200}
    
    start_time = time.time()
    final_solutions = hybrid_genetic_algorithm(n, adj, pop_size=pop_size_map[problem_id], n_generations=gen_map[problem_id])
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id, f"submission_{problem_id}.json")
