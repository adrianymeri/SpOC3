import json
import random
import time
import math
import numpy as np
from typing import List, Set, Tuple
import urllib.request
from collections import deque

# --- Problem Configuration ---
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Utility Functions ---
def load_graph(problem_id: str) -> Tuple[List[List[int]], int]:
    """Loads graph data and returns edges and the number of nodes."""
    url = problems[problem_id]
    print(f"📥 Loading graph data from: {url}")
    with urllib.request.urlopen(url) as f:
        edges = []
        max_node = 0
        for line in f:
            if line.startswith(b"#"):
                continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
            max_node = max(max_node, u, v)
    n = max_node + 1
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges")
    return edges, n

def evaluate_solution(decision_vector: List[int], edges: List[List[int]], n: int) -> Tuple[int, int]:
    """Evaluates the size and width of a given solution."""
    t = decision_vector[-1]
    size = n - t
    
    # Early exit for invalid solutions
    if size <= 0:
        return 0, 501

    perm = decision_vector[:-1]
    pos = {node: i for i, node in enumerate(perm)}
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    max_width = 0
    torso_nodes = perm[t:]

    for i in range(n):
        u = perm[i]
        neighbors_in_perm_after_u = [v for v in adj[u] if pos[v] > i]
        for j1 in range(len(neighbors_in_perm_after_u)):
            for j2 in range(j1 + 1, len(neighbors_in_perm_after_u)):
                v1 = neighbors_in_perm_after_u[j1]
                v2 = neighbors_in_perm_after_u[j2]
                if v2 not in adj[v1]:
                    adj[v1].add(v2)
                    adj[v2].add(v1)
    
    for node in torso_nodes:
        out_degree = sum(1 for neighbor in adj[node] if pos[neighbor] > pos[node])
        max_width = max(max_width, out_degree)

    return size, max_width

# --- Genetic Algorithm Components ---
def dominates(p: Tuple[int, int], q: Tuple[int, int]) -> bool:
    """Check if solution p dominates solution q."""
    return (p[0] >= q[0] and p[1] < q[1]) or \
           (p[0] > q[0] and p[1] <= q[1])

def non_dominated_sort(population: List[List[int]], scores: List[Tuple[int, int]]) -> List[List[int]]:
    """Sorts the population into Pareto fronts."""
    fronts = [[]]
    domination_info = {i: {'count': 0, 'set': set()} for i in range(len(population))}

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if dominates(scores[i], scores[j]):
                domination_info[i]['set'].add(j)
                domination_info[j]['count'] += 1
            elif dominates(scores[j], scores[i]):
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
        
    return [[population[i] for i in front] for front in fronts if front]

def crowding_distance(solutions: List[List[int]], scores: List[Tuple[int, int]]) -> List[float]:
    """Calculates the crowding distance for each solution."""
    n_points = len(solutions)
    if n_points <= 2:
        return [float('inf')] * n_points

    distances = [0.0] * n_points
    for i in range(2):  # For each objective
        sorted_indices = sorted(range(n_points), key=lambda k: scores[k][i])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        f_min = scores[sorted_indices[0]][i]
        f_max = scores[sorted_indices[-1]][i]
        
        if f_max == f_min:
            continue

        for j in range(1, n_points - 1):
            distances[sorted_indices[j]] += (scores[sorted_indices[j + 1]][i] - scores[sorted_indices[j - 1]][i]) / (f_max - f_min)
            
    return distances

# --- Crossover and Mutation ---
def partially_mapped_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Partially Mapped Crossover (PMX)."""
    size = len(parent1) -1
    p1, p2 = parent1[:-1], parent2[:-1]
    
    cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
    
    def pmx(p1, p2):
        child = [None] * size
        mapping = {p2[i]: p1[i] for i in range(cx_point1, cx_point2 + 1)}
        child[cx_point1:cx_point2+1] = p1[cx_point1:cx_point2+1]
        
        for i in list(range(cx_point1)) + list(range(cx_point2 + 1, size)):
            val = p2[i]
            while val in mapping:
                val = mapping[val]
            child[i] = val
        return child + [parent1[-1]]

    return pmx(p1, p2), pmx(p2, p1)

def mutate(solution: List[int], mutation_rate: float) -> List[int]:
    """Mutates a solution by swapping elements or adjusting the threshold."""
    if random.random() < mutation_rate:
        perm = solution[:-1]
        idx1, idx2 = random.sample(range(len(perm)), 2)
        perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
        solution[:-1] = perm

    if random.random() < mutation_rate:
        n = len(solution) - 1
        solution[-1] = random.randint(0, n - 1)
        
    return solution

# --- Initialization ---
def initialize_population(n: int, pop_size: int, edges: List[List[int]]) -> List[List[int]]:
    """Creates an initial population of diverse solutions."""
    population = []
    # Build adjacency list
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    for i in range(pop_size):
        if i % 4 == 0: # Degree-based
            perm = sorted(range(n), key=lambda x: -len(adj_list[x]), reverse=random.random() < 0.5)
        elif i % 4 == 1: # Community-based
            # Basic community detection (can be enhanced)
            communities = detect_communities(list(range(n)), edges)
            perm = []
            for comm in communities:
                random.shuffle(comm)
                perm.extend(comm)
        else: # Random
            perm = list(range(n))
            random.shuffle(perm)
        
        t = random.randint(int(n*0.1), int(n*0.9))
        population.append(perm + [t])
        
    return population
    
def detect_communities(nodes: List[int], edges: List[List[int]]) -> List[List[int]]:
    """Community detection using label propagation"""
    n = len(nodes)
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    labels = list(range(n))
    for _ in range(5):  # Iterate a few times for convergence
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            if not adj_list[i]: continue
            label_counts = {}
            for neighbor in adj_list[i]:
                label_counts[labels[neighbor]] = label_counts.get(labels[neighbor], 0) + 1
            if label_counts:
                max_label = max(label_counts, key=label_counts.get)
                labels[i] = max_label

    communities = {}
    for node, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    return list(communities.values())

# --- Main GA Loop ---
def genetic_algorithm(edges: List[List[int]], n: int, pop_size: int = 100, n_generations: int = 200) -> List[List[int]]:
    """The main genetic algorithm loop."""
    start_time = time.time()
    
    # Initialize population
    population = initialize_population(n, pop_size, edges)
    scores = [evaluate_solution(ind, edges, n) for ind in population]
    
    for gen in range(n_generations):
        # Selection
        fronts = non_dominated_sort(population, scores)
        next_pop = []
        for front in fronts:
            if len(next_pop) + len(front) > pop_size:
                distances = crowding_distance(front, [scores[population.index(ind)] for ind in front])
                sorted_front = [x for _, x in sorted(zip(distances, front), reverse=True)]
                next_pop.extend(sorted_front[:pop_size - len(next_pop)])
                break
            next_pop.extend(front)

        population = next_pop

        # Crossover and Mutation
        offspring = []
        for _ in range(pop_size // 2):
            p1, p2 = random.sample(population, 2)
            c1, c2 = partially_mapped_crossover(p1, p2)
            offspring.append(mutate(c1, 0.1))
            offspring.append(mutate(c2, 0.1))

        population.extend(offspring)
        scores = [evaluate_solution(ind, edges, n) for ind in population]
        
        # Report progress
        if (gen + 1) % 10 == 0:
            best_in_gen = min(scores, key=lambda x: (x[1], -x[0]))
            print(f"🧬 Gen {gen+1}/{n_generations} | Best Score: {best_in_gen} | Pop Size: {len(population)} | Time: {time.time()-start_time:.1f}s")
            
    # Final selection of the best solutions
    final_fronts = non_dominated_sort(population, scores)
    return final_fronts[0]

def create_submission_file(decision_vectors, problem_id, filename="submission.json"):
    """Creates a submission file with multiple decision vectors."""
    submission = {
        "decisionVector": decision_vectors,
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    problem_id = input("🔍 Select problem (easy/medium/hard): ").lower()
    while problem_id not in problems:
        problem_id = input("❌ Invalid! Choose easy/medium/hard: ").lower()

    edges, n = load_graph(problem_id)
    
    start_time = time.time()
    final_solutions = genetic_algorithm(edges, n)
    print(f"\n⏱️  Optimization completed in {time.time() - start_time:.2f} seconds")
    
    # Create a single submission file with the best non-dominated solutions
    create_submission_file(final_solutions, problem_id, f"submission_{problem_id}.json")
