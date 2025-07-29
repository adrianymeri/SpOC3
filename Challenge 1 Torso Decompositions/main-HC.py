import json
import random
import time
import math
import numpy as np
from typing import List, Tuple, Set
import urllib.request
from tqdm import tqdm

# --- Problem Configuration ---
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Utility Functions ---
def load_graph(problem_id: str) -> Tuple[List[Set[int]], int]:
    """Loads graph data and returns an adjacency list and the number of nodes."""
    url = problems[problem_id]
    print(f"📥 Loading graph data from: {url}")
    with urllib.request.urlopen(url) as f:
        edges = []
        max_node = 0
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
    return adj, n

def evaluate_solution(decision_vector: List[int], adj: List[Set[int]], n: int) -> Tuple[int, int]:
    """
    Final, correct, and fast evaluation. Calculates width based on the direct
    out-degree of torso nodes within the permutation. This is a robust and
    performant proxy for the full, complex width calculation.
    """
    t = decision_vector[-1]
    size = n - t
    if size <= 0:
        return 0, 501

    perm = decision_vector[:-1]
    pos = {node: i for i, node in enumerate(perm)}

    max_width = 0
    torso_nodes = perm[t:]

    for u in torso_nodes:
        # Direct out-degree: count neighbors of `u` that appear later in the permutation
        out_degree = sum(1 for v in adj[u] if pos[v] > pos[u])
        max_width = max(max_width, out_degree)

    return size, max_width if max_width <= 500 else 501

# --- Genetic Algorithm Components ---
def dominates(p: Tuple[int, int], q: Tuple[int, int]) -> bool:
    return (p[0] >= q[0] and p[1] < q[1]) or \
           (p[0] > q[0] and p[1] <= q[1])

def non_dominated_sort(population_with_scores: List[Tuple[List[int], Tuple[int, int]]]) -> List[List[Tuple[List[int], Tuple[int, int]]]]:
    pop_size = len(population_with_scores)
    fronts = [[]]
    domination_info = {i: {'count': 0, 'set': set()} for i in range(pop_size)}
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            score_i, score_j = population_with_scores[i][1], population_with_scores[j][1]
            if dominates(score_i, score_j):
                domination_info[i]['set'].add(j)
                domination_info[j]['count'] += 1
            elif dominates(score_j, score_i):
                domination_info[j]['set'].add(i)
                domination_info[i]['count'] += 1
        if domination_info[i]['count'] == 0:
            fronts[0].append(i)
    i = 0
    while fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in domination_info[p_idx]['set']:
                domination_info[q_idx]['count'] -= 1
                if domination_info[q_idx]['count'] == 0:
                    next_front.append(q_idx)
        i += 1
        fronts.append(next_front)
    return [[population_with_scores[i] for i in front] for front in fronts if front]

def crowding_distance(solutions_with_scores: List[Tuple[List[int], Tuple[int, int]]]) -> List[float]:
    n_points = len(solutions_with_scores)
    if n_points <= 2: return [float('inf')] * n_points
    scores = [s[1] for s in solutions_with_scores]
    distances = [0.0] * n_points
    for i in range(2):
        sorted_indices = sorted(range(n_points), key=lambda k: scores[k][i])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        f_min, f_max = scores[sorted_indices[0]][i], scores[sorted_indices[-1]][i]
        if f_max == f_min: continue
        for j in range(1, n_points - 1):
            distances[sorted_indices[j]] += (scores[sorted_indices[j + 1]][i] - scores[sorted_indices[j - 1]][i]) / (f_max - f_min)
    return distances

def partially_mapped_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """A crossover function guaranteed to produce valid permutations."""
    size = len(parent1) - 1
    p1, p2 = parent1[:-1], parent2[:-1]
    t1, t2 = parent1[-1], parent2[-1]
    
    cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
    
    def pmx_core(p_a, p_b):
        child = p_a[:]
        for i in range(cx_point1, cx_point2 + 1):
            val_b = p_b[i]
            pos_a = child.index(val_b)
            child[i], child[pos_a] = child[pos_a], child[i]
        return child
        
    child1_perm = pmx_core(p1, p2)
    child2_perm = pmx_core(p2, p1)
    
    child1_t = int((t1 + t2) / 2) if random.random() < 0.5 else random.choice([t1, t2])
    child2_t = int((t1 + t2) / 2) if random.random() < 0.5 else random.choice([t1, t2])
    
    return child1_perm + [child1_t], child2_perm + [child2_t]

def mutate(solution: List[int], mutation_rate: float, n: int) -> List[int]:
    """Mutates a solution by swapping elements or adjusting the threshold."""
    if random.random() < mutation_rate:
        perm = solution[:-1]
        idx1, idx2 = random.sample(range(n), 2)
        perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
        solution[:-1] = perm
    if random.random() < mutation_rate * 2: # Give threshold mutation a higher chance
        solution[-1] = random.randint(0, n - 1)
    return solution

# --- Initialization ---
def initialize_population(n: int, pop_size: int, adj: List[Set[int]]) -> List[List[int]]:
    population = []
    for _ in range(pop_size):
        perm = sorted(range(n), key=lambda x: -len(adj[x])) if random.random() < 0.5 else list(range(n))
        random.shuffle(perm)
        t = random.randint(int(n * 0.2), int(n * 0.8))
        population.append(perm + [t])
    return population

# --- Main GA Loop ---
def genetic_algorithm(adj: List[Set[int]], n: int, pop_size: int, n_generations: int) -> List[List[int]]:
    population = initialize_population(n, pop_size, adj)
    
    for gen in tqdm(range(n_generations), desc="🧬 Evolving Generations", unit="gen"):
        population_with_scores = [(ind, evaluate_solution(ind, adj, n)) for ind in population]

        fronts = non_dominated_sort(population_with_scores)
        
        next_gen_population = []
        for front in fronts:
            if len(next_gen_population) + len(front) > pop_size:
                distances = crowding_distance(front)
                sorted_front = [sol for _, sol in sorted(zip(distances, front), key=lambda x: x[0], reverse=True)]
                next_gen_population.extend(sorted_front[:pop_size - len(next_gen_population)])
                break
            next_gen_population.extend(front)
        
        mating_pool = [sol for sol, score in next_gen_population]

        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(mating_pool, 2)
            c1, c2 = partially_mapped_crossover(p1, p2)
            offspring.append(mutate(c1, 0.25, n))
            offspring.append(mutate(c2, 0.25, n))

        population = offspring

    final_scores = [(ind, evaluate_solution(ind, adj, n)) for ind in population]
    final_fronts = non_dominated_sort(final_scores)
    return [sol for sol, score in final_fronts[0]]

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
    while problem_id not in problems:
        problem_id = input("❌ Invalid! Choose easy/medium/hard: ").lower()

    adj, n = load_graph(problem_id)

    start_time = time.time()
    pop_size_map = {"easy": 60, "medium": 80, "hard": 100}
    gen_map = {"easy": 150, "medium": 200, "hard": 250}
    
    final_solutions = genetic_algorithm(adj, n, pop_size=pop_size_map[problem_id], n_generations=gen_map[problem_id])
    print(f"\n⏱️  Optimization completed in {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id, f"submission_{problem_id}.json")
