import json
import random
import time
import math
import numpy as np
from typing import List, Set, Tuple, Dict
import urllib.request
from tqdm import tqdm
import multiprocessing
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eigh

# --- Problem Configurations ---
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Evaluation Functions ---

def load_graph(problem_id: str) -> Tuple[int, List[List[int]], List[Set[int]]]:
    """Loads graph data from local 'data/' folder instead of downloading."""
    local_path = os.path.join(
        os.path.dirname(__file__), "data", f"{problem_id}.gr"
    )
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"❌ Graph file not found: {local_path}")

    print(f"📥 Loading graph data for '{problem_id}' from {local_path}...")
    edges = []
    max_node = 0
    with open(local_path, "r") as f:
        for line in f:
            if line.startswith("#"): 
                continue
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

def evaluate_heuristic_v2(decision_vector: List[int], n: int, adj_list: List[Set[int]]) -> List[float]:
    """UPGRADE 1: An intelligent heuristic that penalizes the 'cut' size."""
    t = decision_vector[-1]
    perm = decision_vector[:-1]
    size = n - t
    if size <= 0: return [0, 501]

    max_width = 0
    torso_nodes = set(perm[t:])
    for node in torso_nodes:
        current_width = len(adj_list[node])
        if current_width > max_width:
            max_width = current_width
    
    # Calculate the "cut penalty" for edges crossing the boundary
    cut_penalty = 0
    boundary_start = max(0, t - int(n * 0.1)) # Check last 10% of non-torso
    non_torso_boundary = perm[boundary_start:t]
    for node in non_torso_boundary:
        for neighbor in adj_list[node]:
            if neighbor in torso_nodes:
                cut_penalty += 1
    
    # The weight (0.1) adds a small penalty for a messy "cut"
    final_width = max_width + 0.1 * cut_penalty
    
    return [size, final_width if final_width < 500 else 501]

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

# --- Your Proven Search Algorithm Components ---

def smart_torso_shift(current: List[int], n: int, adj_list: List[Set[int]]) -> List[int]:
    neighbor = current[:]
    t = neighbor[-1]
    shift = int(n * 0.05) + 1
    neighbor[-1] = max(0, min(n - 1, t + random.randint(-shift, shift)))
    return neighbor

def block_move(current: List[int], n: int, adj_list: List[Set[int]]) -> List[int]:
    neighbor = current[:]
    perm = neighbor[:-1]
    block_size = random.randint(3, max(4, int(n * 0.03)))
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
    if random.random() < 0.5: perm.reverse()
    t = int(n * np.random.beta(1.5, 2.5))
    return perm + [t]

def hill_climbing_run(
    n: int,
    adj_list: List[Set[int]],
    max_iterations: int,
    initial_temp: float,
    cooling_rate: float
) -> Dict:
    """Performs a single run of your proven Simulated Annealing algorithm."""
    neighbor_operators = [smart_torso_shift, block_move]
    current = initialize_solution(n, adj_list)
    current_score = evaluate_heuristic_v2(current, n, adj_list) # Using the upgraded heuristic
    
    best_local = current[:]
    best_local_score = current_score[:]
    
    T = initial_temp
    last_improvement = 0

    for iteration in range(max_iterations):
        T *= cooling_rate
        op = random.choice(neighbor_operators)
        neighbor = op(current, n, adj_list)
        
        neighbor_score = evaluate_heuristic_v2(neighbor, n, adj_list)
        
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
            current, current_score = neighbor, neighbor_score
            if dominates(current_score, best_local_score):
                best_local, best_local_score = current[:], current_score[:]
                last_improvement = iteration
        
        if iteration - last_improvement > 1500:
            break
            
    return {'solution': best_local, 'score': best_local_score}

def path_relinking(sol1: Dict, sol2: Dict, n: int, adj_list: List[Set[int]]) -> List[Dict]:
    """UPGRADE 2: Explores the path between two elite solutions."""
    print("🔬 Performing Path Relinking between top 2 solutions...")
    path_solutions = []
    
    start_sol = sol1['solution']
    end_perm = sol2['solution'][:-1]
    
    current_perm = start_sol[:-1][:]
    
    for i in tqdm(range(n), desc="  -> Relinking Path", leave=False):
        if current_perm[i] != end_perm[i]:
            node_to_move = end_perm[i]
            try:
                current_pos = current_perm.index(node_to_move)
                current_perm[i], current_perm[current_pos] = current_perm[current_pos], current_perm[i]
            except ValueError:
                continue # Node not found, should not happen in a valid permutation
            
            new_t = int((start_sol[-1] + sol2['solution'][-1]) / 2)
            new_solution = current_perm + [new_t]
            score = evaluate_heuristic_v2(new_solution, n, adj_list)
            path_solutions.append({'solution': new_solution, 'score': score})
            
    return path_solutions

# --- Main Execution Block ---

if __name__ == "__main__":
    multiprocessing.freeze_support()
    random.seed(42)
    np.random.seed(42)

    # --- Parameters from your best-performing code ---
    PARAMS = {
        'max_iterations': 25000,
        'num_restarts': 250,
        'cooling_rate': 0.9992,
        'initial_temp': 3000.0
    }

    problem_id = input("🔍 Select problem to run (easy/medium/hard): ").lower()
    if problem_id not in PROBLEMS: exit("❌ Invalid problem ID. Exiting.")

    n, edges, adj_list = load_graph(problem_id)
    
    print(f"\n⚙️  Running Upgraded Solver for '{problem_id}'...")
    start_time = time.time()
    
    all_solutions = []
    # Using your sequential restart loop which proved most effective
    for _ in tqdm(range(PARAMS['num_restarts']), desc="🚀 Running Restarts"):
        result = hill_climbing_run(
            n=n,
            adj_list=adj_list,
            max_iterations=PARAMS['max_iterations'],
            initial_temp=PARAMS['initial_temp'],
            cooling_rate=PARAMS['cooling_rate']
        )
        all_solutions.append(result)
    
    # Filter for the non-dominated front based on the HEURISTIC scores
    heuristic_pareto_front = []
    for candidate in all_solutions:
        if not any(dominates(other['score'], candidate['score']) for other in all_solutions):
            heuristic_pareto_front.append(candidate)
    unique_solutions = list({tuple(p['solution']): p for p in heuristic_pareto_front}.values())
    
    if len(unique_solutions) >= 2:
        unique_solutions.sort(key=lambda p: -p['score'][0] * 100 + p['score'][1])
        path_res = path_relinking(unique_solutions[0], unique_solutions[1], n, adj_list)
        all_solutions.extend(path_res)
        
        # Re-filter the front with the new solutions from path relinking
        heuristic_pareto_front = []
        for candidate in all_solutions:
            if not any(dominates(other['score'], candidate['score']) for other in all_solutions):
                heuristic_pareto_front.append(candidate)
        unique_solutions = list({tuple(p['solution']): p for p in heuristic_pareto_front}.values())

    # *** FINAL ACCURATE RE-EVALUATION STEP ***
    print(f"\n🔬 Performing final accurate evaluation of {len(unique_solutions)} elite solutions...")
    solutions_to_re_eval = [v['solution'] for v in unique_solutions]
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
