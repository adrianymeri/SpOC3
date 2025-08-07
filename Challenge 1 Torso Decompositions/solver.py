import json
import random
import time
import math
import numpy as np
from typing import List, Set, Tuple, Dict
import urllib.request
from tqdm import tqdm
import multiprocessing

# --- Problem Configurations ---
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Evaluation Functions ---

def load_graph(problem_id: str) -> Tuple[int, List[List[int]], List[Set[int]]]:
    """Loads graph data and returns n, edges, and adjacency list."""
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    edges = []
    max_node = 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#'): continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
            max_node = max(max_node, u, v)
    n = max_node + 1
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, edges, adj_list

def evaluate_heuristic(decision_vector: List[int], n: int, adj_list: List[Set[int]]) -> List[float]:
    """Your proven, high-speed evaluation function."""
    t = decision_vector[-1]
    size = n - t
    if size <= 0: return [0, 501]
    max_width = 0
    for i in range(t, n):
        node = decision_vector[i]
        current_width = len(adj_list[node])
        if current_width > max_width:
            max_width = current_width
    return [size, max_width if max_width < 500 else 501]

def evaluate_correct_task(args: Tuple[Tuple[int, ...], int, List[List[int]]]) -> Tuple[Tuple[int, ...], Tuple[int, int]]:
    """The 100% correct evaluation function, wrapped for multiprocessing."""
    solution_tuple, n, edges = args
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
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

def dominates(score1: List[float], score2: List[float]) -> bool:
    return (score1[0] > score2[0] and score1[1] <= score2[1]) or \
           (score1[0] >= score2[0] and score1[1] < score2[1])

# --- Your Original, Proven Search Algorithm Components ---

def smart_torso_shift(current: List[int], n: int, adj_list: List[Set[int]]) -> List[int]:
    neighbor = current[:]
    t = neighbor[-1]
    score = evaluate_heuristic(neighbor, n, adj_list)
    if score[1] > 400:
        new_t = min(n - 1, t + random.randint(5, 10))
    else:
        shift = int(n * 0.04) + 1
        new_t = max(0, min(n - 1, t + random.randint(-shift, shift)))
    neighbor[-1] = new_t
    return neighbor

def block_move(current: List[int], n: int, adj_list: List[Set[int]]) -> List[int]:
    neighbor = current[:]
    perm = neighbor[:-1]
    score = evaluate_heuristic(neighbor, n, adj_list)
    base_size = 3 if score[1] < 100 else 5
    block_size = random.randint(base_size, base_size + 2)
    if n > block_size:
        start = random.randint(0, n - block_size)
        block = perm[start:start + block_size]
        del perm[start:start + block_size]
        insert_pos = random.randint(0, len(perm))
        perm[insert_pos:insert_pos] = block
        neighbor[:-1] = perm
    return neighbor

def initialize_solution(n: int, adj_list: List[Set[int]]) -> List[int]:
    perm = sorted(range(n), key=lambda x: -len(adj_list[x]))
    if random.random() < 0.5:
        perm.reverse()
    t = int(n * np.random.beta(1.5, 2.5))
    return perm + [t]

def hill_climbing_solver(
    n: int,
    adj_list: List[Set[int]],
    max_iterations: int,
    num_restarts: int,
    cooling_rate: float,
    initial_temp: float
) -> List[Dict]:
    """Your proven multi-start simulated annealing algorithm."""
    
    all_solutions = []
    neighbor_operators = [smart_torso_shift, block_move]

    for _ in tqdm(range(num_restarts), desc="🚀 Running Your Best Solver"):
        current = initialize_solution(n, adj_list)
        current_score = evaluate_heuristic(current, n, adj_list)
        best_local = current[:]
        best_local_score = current_score[:]
        T = initial_temp
        last_improvement = 0

        for iteration in range(max_iterations):
            T *= cooling_rate
            op = random.choice(neighbor_operators)
            neighbor = op(current, n, adj_list)
            
            neighbor_score = evaluate_heuristic(neighbor, n, adj_list)
            
            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                size_gain = neighbor_score[0] - current_score[0]
                width_diff = current_score[1] - neighbor_score[1]
                delta = 2 * size_gain + width_diff
                if T > 1e-6 and random.random() < math.exp(delta / T):
                    accept = True

            if accept:
                current = neighbor
                current_score = neighbor_score
                if dominates(current_score, best_local_score):
                    best_local = current[:]
                    best_local_score = current_score[:]
                    last_improvement = iteration
            
            if iteration - last_improvement > 1500:
                break
        all_solutions.append({'solution': best_local, 'score': best_local_score})
    
    return all_solutions

# --- Main Execution Block ---

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(42)
    np.random.seed(42)

    # --- Parameters from your best-performing code that scored -1,795,311 ---
    PARAMS = {
        'max_iterations': 25000,
        'num_restarts': 250,
        'cooling_rate': 0.9992,
        'initial_temp': 3000.0
    }

    problem_id = input("🔍 Select problem to run (easy/medium/hard): ").lower()
    if problem_id not in PROBLEMS: exit("❌ Invalid problem ID. Exiting.")

    n, edges, adj_list = load_graph(problem_id)
    
    print(f"\n⚙️  Running your best solver for '{problem_id}'...")
    start_time = time.time()
    
    # === STAGE 1: HIGH-SPEED GLOBAL EXPLORATION (Your Proven Code) ===
    all_solutions = hill_climbing_solver(
        n=n,
        adj_list=adj_list,
        **PARAMS
    )
    
    # Filter for the non-dominated front based on HEURISTIC scores
    heuristic_pareto_front = []
    for candidate in all_solutions:
        if not any(dominates(other['score'], candidate['score']) for other in all_solutions):
            heuristic_pareto_front.append(candidate)
    unique_solutions = {tuple(p['solution']): p for p in heuristic_pareto_front}
    
    # === STAGE 2: ACCURATE RE-EVALUATION STEP ===
    print(f"\n🔬 Performing final accurate evaluation of {len(unique_solutions)} elite solutions...")
    solutions_to_re_eval = [v['solution'] for v in unique_solutions.values()]
    with multiprocessing.Pool() as pool:
        final_results = pool.map(evaluate_correct_task, [(tuple(sol), n, edges) for sol in solutions_to_re_eval])

    final_population = [{'solution': list(sol), 'score': list(score)} for sol, score in final_results]
    
    # --- Final Selection based on CORRECT scores ---
    final_pareto_front = []
    for candidate in final_population:
        if not any(dominates(other['score'], candidate['score']) for other in final_population):
            final_pareto_front.append(candidate)
    unique_final_solutions = list({tuple(p['solution']): p for p in final_pareto_front}.values())
    
    end_time = time.time()

    # --- Create Submission File ---
    if unique_final_solutions:
        filename = f"submission_{problem_id}.json"
        problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
        decision_vectors = [p['solution'] for p in unique_final_solutions]
        final_vectors = [[int(val) for val in vec] for vec in decision_vectors]
        submission = {
            "decisionVector": final_vectors[:20],
            "problem": problem_name_map.get(problem_id, problem_id),
            "challenge": "spoc-3-torso-decompositions",
        }
        with open(filename, "w") as f:
            json.dump(submission, f, indent=4)
        print(f"\n📄 Created submission file: {filename} with {len(submission['decisionVector'])} solutions.")
        print(f"⏱️  Finished '{problem_id}' in {end_time - start_time:.2f} seconds.")
    else:
        print(f"--- No non-dominated solutions found for '{problem_id}' ---")
