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
import torch

# --- Algorithm & Problem Configuration ---
CONFIG = {
    "general": {
        "checkpoint_interval": 500,
    },
    # These settings are aggressive and designed for long runs on a server
    "easy": {"population_size": 2048, "generations": 15000, "mutation_stdev": 0.05, "feature_dim": 16},
    "medium": {"population_size": 2048, "generations": 20000, "mutation_stdev": 0.05, "feature_dim": 24},
    "hard": {"population_size": 4096, "generations": 25000, "mutation_stdev": 0.05, "feature_dim": 32},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Graph Loading & Feature Generation ---

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

def init_node_features(n: int, adj: List[Set[int]], feature_dim: int) -> torch.Tensor:
    """Creates simple, effective node features based on degree propagation."""
    print("🧬 Generating node features...")
    features = np.zeros((n, feature_dim), dtype=np.float32)
    degrees = np.array([len(neighbors) for neighbors in adj], dtype=np.float32)
    
    features[:, 0] = degrees / (degrees.max() + 1e-6)
    
    # Create features based on neighbors' degrees (degree propagation)
    for i in range(1, feature_dim):
        neighbor_degrees = np.array([np.mean(degrees[list(adj[j])]) if adj[j] else 0 for j in range(n)])
        features[:, i] = neighbor_degrees / (neighbor_degrees.max() + 1e-6)
        degrees = neighbor_degrees

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensor device: {device}")
    return torch.from_numpy(features).to(device)

# --- Evaluation Functions ---

def evaluate_heuristic(perm: List[int], t: int, n: int, adj: List[Set[int]]) -> Tuple[int, int]:
    """Your fast heuristic evaluation function."""
    size = n - t
    if size <= 0: return 0, 501
    max_width = 0
    for i in range(t, n):
        node = perm[i]
        width = len(adj[node])
        if width > max_width:
            max_width = width
    return (size, max_width if max_width < 500 else 501)

def evaluate_correct_task(args: Tuple[Tuple[int, ...], int, List[Set[int]]]) -> Tuple[Tuple[int, ...], Tuple[int, int]]:
    """The 100% correct evaluation function, wrapped for multiprocessing."""
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

def dominates(p, q): return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

# --- Neuroevolutionary Search Algorithm ---

def neuroevolution_search(n: int, adj: List[Set[int]], config: Dict, problem_id: str) -> List[Dict]:
    """Main neuroevolutionary solver based on the winning architecture."""
    B = config['population_size']
    E = config['feature_dim']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    node_features = init_node_features(n, adj, E).T # Transpose for matmul
    
    population = torch.randn((B, E), dtype=torch.float32, device=device) * 0.1
    
    # Elites store the best WEIGHT VECTOR found for each threshold t
    elites = [None] * n 
    elite_scores = [(0, 501)] * n

    for gen in tqdm(range(config['generations']), desc="🧬 Evolving"):
        # 1. Generate permutations from network weights
        logits = population @ node_features
        perms = logits.argsort(dim=1).cpu().numpy()
        
        # 2. Evaluate all permutations against all thresholds using the fast heuristic
        for i in range(B):
            perm = perms[i]
            for t in range(n):
                score = evaluate_heuristic(perm, t, n, adj)
                
                # If this is a better solution for this `t`, save the weight vector
                if dominates(score, elite_scores[t]):
                    elite_scores[t] = score
                    elites[t] = population[i].clone()
        
        # 3. Create the next generation from the best elites found so far
        non_dominated_elites = []
        for t1 in range(n):
            if elites[t1] is None: continue
            is_dominated = False
            for t2 in range(n):
                if elites[t2] is None: continue
                if dominates(elite_scores[t2], elite_scores[t1]):
                    is_dominated = True; break
            if not is_dominated:
                non_dominated_elites.append(elites[t1])

        if not non_dominated_elites:
            # If all elites are dominated (rare), pick the best of what we have
            non_dominated_elites = [e for e in elites if e is not None]
            if not non_dominated_elites: # If no solutions found yet, re-initialize
                population = torch.randn((B, E), dtype=torch.float32, device=device) * 0.1
                continue

        # Procreate from the best elite solutions
        parent_indices = torch.randint(0, len(non_dominated_elites), (B,), device=device)
        population = torch.stack([non_dominated_elites[i] for i in parent_indices])
        
        # 4. Mutate the weights
        mutation = torch.randn_like(population) * config['mutation_stdev']
        mask = torch.rand_like(population) < 0.4 # 40% mutation probability
        population[mask] += mutation[mask]

    # Return the final set of non-dominated elites (weights and heuristic scores)
    final_pareto_front = []
    for t in range(n):
        if elites[t] is None: continue
        is_dominated = False
        for t2 in range(n):
            if elites[t2] is None: continue
            if dominates(elite_scores[t2], elite_scores[t]):
                is_dominated = True; break
        if not is_dominated:
            final_pareto_front.append({'solution_weights': elites[t], 'heuristic_score': elite_scores[t]})
    
    return final_pareto_front

# --- Main Execution & Submission ---

def create_submission_file(final_solutions: List[Dict], n: int, adj: List[Set[int]], node_features: torch.Tensor, problem_id: str):
    filename = f"submission_{problem_id}.json"
    problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
    
    # *** FINAL, ACCURATE RE-EVALUATION STEP ***
    print(f"\n🔬 Performing final accurate evaluation of {len(final_solutions)} elite solutions...")
    
    # Regenerate permutations from the best weights
    decision_vectors_to_score = []
    for sol in final_solutions:
        weights = sol['solution_weights']
        t = n - sol['heuristic_score'][0]
        logits = weights @ node_features
        perm = logits.argsort().cpu().tolist()
        decision_vectors_to_score.append(perm + [t])

    # Re-score using the correct function in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(evaluate_correct_task, [(tuple(sol), n, adj) for sol in decision_vectors_to_score])

    # Create a final population with the CORRECT scores
    final_population = [{'solution': list(sol), 'score': score} for sol, score in results]

    # Select the best 20 for submission based on the CORRECT scores
    final_submission_pop = crowding_selection(final_population, 20)
    final_vectors_for_json = [[int(val) for val in p['solution']] for p in final_submission_pop]

    submission = { "decisionVector": final_vectors_for_json, "problem": problem_name_map.get(problem_id, problem_id), "challenge": "spoc-3-torso-decompositions" }
    with open(filename, "w") as f: json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(final_vectors_for_json)} solutions.")

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    """NSGA-II selection, used here to select the final top 20 solutions."""
    # (Full implementation)
    pass # This will be filled in the full script

if __name__ == "__main__":
    multiprocessing.freeze_support()
    torch.set_grad_enabled(False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    problem_id = input("🔍 Select problem (easy/medium/hard): ").lower()
    if problem_id not in PROBLEMS: exit("❌ Invalid problem ID. Exiting.")

    config = CONFIG['general'].copy()
    config.update(CONFIG[problem_id])

    n, adj = load_graph(problem_id)
    node_features = init_node_features(n, adj, config['feature_dim']).T

    start_time = time.time()
    final_solutions = neuroevolution_search(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, n, adj, node_features, problem_id)
