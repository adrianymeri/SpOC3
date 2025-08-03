import json
import random
import time
import math
import numpy as np
from typing import List
import urllib.request
from tqdm import tqdm

# --- Problem Configurations ---
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Helper Functions ---

def load_graph(problem_id: str) -> tuple[int, list[list[int]]]:
    """Loads graph data and returns the number of nodes and edges."""
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    edges = []
    max_node = 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b"#"): continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
            max_node = max(max_node, u, v)
    n = max_node + 1
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, edges

def evaluate_solution(decision_vector: List[int], n: int, adj_list: List[set]) -> List[float]:
    """Your proven, high-speed evaluation function."""
    t = decision_vector[-1]
    size = n - t
    if size <= 0: return [0, 501, []]

    max_width = 0
    for i in range(t, n):
        node = decision_vector[i]
        current_width = len(adj_list[node])
        if current_width > max_width:
            max_width = current_width
    
    return [size, max_width if max_width < 500 else 501, []]

def dominates(score1: List[float], score2: List[float]) -> bool:
    """Checks if score1 dominates score2."""
    return (score1[0] > score2[0] and score1[1] <= score2[1]) or \
           (score1[0] >= score2[0] and score1[1] < score2[1])

# --- Neighborhood Operators ---

def smart_torso_shift(current: List[int], n: int, adj_list: List[set]) -> List[int]:
    """Adaptive threshold adjustment."""
    neighbor = current[:]
    t = neighbor[-1]
    shift = int(n * 0.05) + 1
    neighbor[-1] = max(0, min(n - 1, t + random.randint(-shift, shift)))
    return neighbor

def block_move(current: List[int], n: int, adj_list: List[set]) -> List[int]:
    """Moves a block of nodes."""
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

def inversion_move(current: List[int], n: int, adj_list: List[set]) -> List[int]:
    """Inverts a random subsection of the permutation."""
    neighbor = current[:]
    perm = neighbor[:-1]
    start, end = sorted(random.sample(range(n), 2))
    perm[start:end+1] = reversed(perm[start:end+1])
    neighbor[:-1] = perm
    return neighbor

# --- Initialization & Main Algorithm ---

def initialize_solution(n: int, adj_list: List[set]) -> List[int]:
    """Creates a single solution to start a search run."""
    perm = sorted(range(n), key=lambda x: -len(adj_list[x]))
    if random.random() < 0.5:
        perm.reverse()
    t = int(n * np.random.beta(1.5, 2.5))
    return perm + [t]

def shake(solution: List[int], n: int, adj_list: List[set]) -> List[int]:
    """Applies several random moves to 'shake' a solution out of a local optimum."""
    shaken_sol = solution[:]
    for _ in range(5):
        op = random.choice([block_move, smart_torso_shift, inversion_move])
        shaken_sol = op(shaken_sol, n, adj_list)
    return shaken_sol

def variable_neighborhood_search(
    n: int,
    adj_list: List[set],
    max_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    best_known_solution: List[int] = None
) -> dict:
    """
    Performs a single, powerful VNS run.
    """
    neighborhoods = [block_move, smart_torso_shift, inversion_move]
    
    if best_known_solution and random.random() < 0.7:
        current = shake(best_known_solution, n, adj_list)
    else:
        current = initialize_solution(n, adj_list)

    current_score = evaluate_solution(current, n, adj_list)[:2]
    best_local = current[:]
    best_local_score = current_score[:]
    
    T = initial_temp
    k = 0
    iters_since_improvement = 0

    for _ in range(max_iterations):
        T *= cooling_rate
        
        op = neighborhoods[k]
        neighbor = op(current, n, adj_list)
        neighbor_score = evaluate_solution(neighbor, n, adj_list)[:2]
        
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
            k = 0
            iters_since_improvement = 0
            if dominates(current_score, best_local_score):
                best_local = current[:]
                best_local_score = current_score[:]
        else:
            k = (k + 1) % len(neighborhoods)
            iters_since_improvement += 1
        
        if iters_since_improvement > 2500:
            break
            
    return {'solution': best_local, 'score': best_local_score}

# --- Main Execution Block ---

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # --- Use the Optimal Parameters Found by Your Optuna Run ---
    TUNED_PARAMS = {
        'max_iterations': 45000,
        'num_restarts': 250,
        'cooling_rate': 0.999408137776452,
        'initial_temp': 4886.692842415465
    }

    problem_id = input("🔍 Select problem to run (easy/medium/hard): ").lower()
    
    if problem_id not in PROBLEMS:
        print("❌ Invalid problem ID. Exiting.")
        exit()

    n, edges = load_graph(problem_id)
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    print(f"\n⚙️  Running VNS-powered solver for '{problem_id}' with tuned parameters...")
    start_time = time.time()
    
    all_solutions = []
    best_solution_so_far = None
    best_score_so_far = (0, float('inf'))

    for _ in tqdm(range(TUNED_PARAMS['num_restarts']), desc="🚀 Running VNS Restarts"):
        result = variable_neighborhood_search(
            n=n,
            adj_list=adj_list,
            max_iterations=TUNED_PARAMS['max_iterations'],
            initial_temp=TUNED_PARAMS['initial_temp'],
            cooling_rate=TUNED_PARAMS['cooling_rate'],
            best_known_solution=best_solution_so_far
        )
        all_solutions.append(result)
        if dominates(result['score'], best_score_so_far):
            best_solution_so_far = result['solution']
            best_score_so_far = result['score']
    
    # Filter for the final non-dominated front
    final_pareto_front = []
    for candidate in all_solutions:
        if not any(dominates(other['score'], candidate['score']) for other in all_solutions):
            final_pareto_front.append(candidate)
            
    unique_solutions = {tuple(p['solution']): p for p in final_pareto_front}
    final_solutions = list(unique_solutions.values())
    
    end_time = time.time()

    # --- Create Submission File ---
    if final_solutions:
        filename = f"submission_{problem_id}.json"
        problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
        decision_vectors = [p['solution'] for p in final_solutions]
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
