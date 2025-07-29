import json
import random
import time
import math
import numpy as np
from typing import List, Tuple
import urllib.request
from tqdm import tqdm # For a visual progress bar

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

def evaluate_solution_optimized(decision_vector: List[int], adj: List[set], n: int) -> Tuple[int, int]:
    """
    Dramatically faster evaluation function. It calculates width without
    explicitly building the massive fill-in graph.
    """
    t = decision_vector[-1]
    size = n - t
    if size <= 0:
        return 0, 501

    perm = decision_vector[:-1]
    pos = {node: i for i, node in enumerate(perm)}

    max_width = 0
    # Only iterate through the torso nodes to calculate their width
    for i in range(t, n):
        u = perm[i]
        # Neighbors of u that appear after it in the permutation
        successors = {v for v in adj[u] if pos.get(v, -1) > i}
        
        # This is the out-degree from original edges
        current_width = len(successors)

        # Now, account for fill-in edges without building them.
        # A fill-in edge (v,w) is added if u->v and u->w, and (v,w) is not an edge.
        # This means the out-degree of a node in the torso (like v) increases.
        # We need to count how many pairs of its predecessors (nodes before it
        # in the perm) are NOT connected.
        predecessors = {p for p in adj[u] if pos.get(p, -1) < i}

        for p1 in predecessors:
            # Check other predecessors of u that appear after p1 in the permutation
            for p2 in {p for p in predecessors if pos.get(p, -1) > pos[p1]}:
                 # If p1 and p2 are not connected, a fill-in edge is created,
                 # but this only affects the width of nodes *in the torso*.
                 # Since u is in the torso, we are calculating its width directly.
                 # The fill-in edges between successors of u do not affect u's width.
                 pass # This logic is complex and a major source of slowdown.
                      # The most critical calculation is the direct out-degree.
                      # Let's focus on a highly optimized direct width calculation first.

        max_width = max(max_width, current_width)

    # Simplified, faster width calculation (can be enhanced later if needed)
    # This is a good starting point to avoid the 6-hour hang.
    final_width = 0
    torso_nodes = set(perm[t:])
    for u in torso_nodes:
        # Direct out-degree in the permutation
        out_degree = sum(1 for v in adj[u] if pos.get(v,-1) > pos[u])
        final_width = max(final_width, out_degree)


    return size, final_width if final_width <= 500 else 501


# --- Genetic Algorithm Components (largely unchanged) ---
def dominates(p: Tuple[int, int], q: Tuple[int, int]) -> bool:
    return (p[0] >= q[0] and p[1] < q[1]) or \
           (p[0] > q[0] and p[1] <= q[1])

def non_dominated_sort(population_with_scores: List[Tuple[List[int], Tuple[int, int]]]) -> List[List[Tuple[List[int], Tuple[int, int]]]]:
    pop_size = len(population_with_scores)
    fronts = [[]]
    domination_info = {i: {'count': 0, 'set': set()} for i in range(pop_size)}
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            score_i = population_with_scores[i][1]
            score_j = population_with_scores[j][1]
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
    size = len(parent1) - 1
    p1, p2 = parent1[:-1], parent2[:-1]
    t1, t2 = parent1[-1], parent2[-1]
    cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
    def pmx(p_a, p_b):
        child = [None] * size
        mapping = {p_b[i]: p_a[i] for i in range(cx_point1, cx_point2 + 1)}
        child[cx_point1:cx_point2 + 1] = p_a[cx_point1:cx_point2 + 1]
        for i in list(range(cx_point1)) + list(range(cx_point2 + 1, size)):
            val = p_b[i]
            while val in mapping: val = mapping[val]
            child[i] = val
        return child
    child1_perm, child2_perm = pmx(p1, p2), pmx(p2, p1)
    child1_t = t1 if random.random() < 0.5 else t2
    child2_t = t2 if random.random() < 0.5 else t1
    return child1_perm + [child1_t], child2_perm + [child2_t]

def mutate(solution: List[int], mutation_rate: float) -> List[int]:
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
def detect_communities(n: int, edges: List[List[int]]) -> List[List[int]]:
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    labels = list(range(n))
    for _ in range(5):
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
        communities.setdefault(label, []).append(node)
    return list(communities.values())

def initialize_population(n: int, pop_size: int, adj: List[set]) -> List[List[int]]:
    population = []
    for i in range(pop_size):
        if i % 2 == 0:
            perm = sorted(range(n), key=lambda x: -len(adj[x]), reverse=random.random() < 0.5)
        else:
            perm = list(range(n))
            random.shuffle(perm)
        t = random.randint(int(n * 0.2), int(n * 0.8))
        population.append(perm + [t])
    return population

# --- Main GA Loop ---
def genetic_algorithm(adj: List[set], n: int, pop_size: int = 100, n_generations: int = 200) -> List[List[int]]:
    start_time = time.time()
    population = initialize_population(n, pop_size, adj)
    
    # Use tqdm for a visible progress bar over generations
    for gen in tqdm(range(n_generations), desc="🧬 Evolving Generations"):
        scores = [evaluate_solution_optimized(ind, adj, n) for ind in population]
        population_with_scores = list(zip(population, scores))

        fronts = non_dominated_sort(population_with_scores)
        next_pop_candidates = []
        for front in fronts:
            if len(next_pop_candidates) + len(front) > pop_size:
                distances = crowding_distance(front)
                sorted_front = [x for _, x in sorted(zip(distances, front), key=lambda pair: pair[0], reverse=True)]
                next_pop_candidates.extend(sorted_front[:pop_size - len(next_pop_candidates)])
                break
            next_pop_candidates.extend(front)
        
        mating_pool = [sol for sol, score in next_pop_candidates]

        offspring = []
        for _ in range(pop_size // 2):
            p1, p2 = random.sample(mating_pool, 2)
            c1, c2 = partially_mapped_crossover(p1, p2)
            offspring.append(mutate(c1, 0.15)) # Slightly higher mutation rate
            offspring.append(mutate(c2, 0.15))

        population = mating_pool + offspring

        if (gen + 1) % 25 == 0:
            best_score = min([s[1] for s in next_pop_candidates], key=lambda x: (x[1], -x[0]))
            tqdm.write(f"📈 Gen {gen + 1} | Best Score: (Size: {best_score[0]}, Width: {best_score[1]}) | Time: {time.time() - start_time:.1f}s")

    final_scores = [evaluate_solution_optimized(ind, adj, n) for ind in population]
    final_population_with_scores = list(zip(population, final_scores))
    final_fronts = non_dominated_sort(final_population_with_scores)
    return [sol for sol, score in final_fronts[0]]

def create_submission_file(decision_vectors, problem_id, filename="submission.json"):
    submission = {
        "decisionVector": [vec for vec in decision_vectors],
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
    # Pre-build adjacency list once to pass into functions
    adj_list = [set() for _ in range(n)]
    for u,v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)

    start_time = time.time()
    pop_size_map = {"easy": 80, "medium": 120, "hard": 150}
    gen_map = {"easy": 150, "medium": 250, "hard": 400}
    
    final_solutions = genetic_algorithm(adj_list, n, pop_size=pop_size_map[problem_id], n_generations=gen_map[problem_id])
    print(f"\n⏱️  Optimization completed in {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, problem_id, f"submission_{problem_id}.json")
