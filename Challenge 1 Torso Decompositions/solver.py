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
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eigh

# --- Algorithm & Problem Configuration ---
CONFIG = {
    "general": {
        "checkpoint_interval": 500,
    },
    "easy": {"population_size": 2048, "generations": 15000, "mutation_stdev": 0.05, "eigenvectors": 10},
    "medium": {"population_size": 2048, "generations": 20000, "mutation_stdev": 0.05, "eigenvectors": 16},
    "hard": {"population_size": 4096, "generations": 25000, "mutation_stdev": 0.05, "eigenvectors": 24},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Graph Loading & Advanced Feature Generation ---

def load_graph(problem_id: str) -> Tuple[int, np.ndarray]:
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
    adj_mat = np.zeros((n, n), dtype=np.bool_)
    for u, v in edges:
        adj_mat[u, v] = True
        adj_mat[v, u] = True
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges.")
    return n, adj_mat

def init_node_features(n: int, adj: np.ndarray, eigenvectors: int) -> torch.Tensor:
    """Creates advanced node features based on the winning solution's strategy."""
    print("🧬 Generating advanced node features (this may take a moment)...")
    
    degree_profile_size = 5
    nodes = np.zeros((n, degree_profile_size + eigenvectors), dtype=np.float32)
    degrees = adj.sum(axis=1)
    nodes[:, 0] = degrees
    for i in range(n):
        neighbors = adj[i]
        if neighbors.any():
            neighbor_degrees = degrees[neighbors]
            nodes[i, 1] = neighbor_degrees.min()
            nodes[i, 2] = neighbor_degrees.max()
            nodes[i, 3] = neighbor_degrees.mean()
            nodes[i, 4] = neighbor_degrees.std()

    print("   - Calculating Laplacian Eigenvectors...")
    lap = laplacian(adj.astype(np.int8), normed=True)
    eig_vals, eig_vecs = eigh(lap)
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = eig_vecs[:, 1: eigenvectors + 1]
    nodes[:, degree_profile_size:] = pe

    nodes = (nodes - nodes.mean(axis=0)) / (nodes.std(axis=0) + 1e-6)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Tensor device: {device}")
    return torch.from_numpy(nodes).to(device)

# --- Evaluation Functions ---

def evaluate_heuristic(perm: List[int], t: int, n: int, adj: np.ndarray) -> Tuple[int, int]:
    """Fast heuristic using original degrees from the adjacency matrix."""
    size = n - t
    if size <= 0: return 0, 501
    max_width = 0
    degrees = adj.sum(axis=1)
    for i in range(t, n):
        node = perm[i]
        width = degrees[node]
        if width > max_width: max_width = width
    return (size, max_width if max_width < 500 else 501)

def evaluate_correct_task(args: Tuple[Tuple[int, ...], int, np.ndarray]) -> Tuple[Tuple[int, ...], Tuple[int, int]]:
    solution_tuple, n, adj_mat = args
    adj = [set(np.where(row)[0]) for row in adj_mat]
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

def neuroevolution_search(n: int, adj: np.ndarray, node_features: torch.Tensor, config: Dict, problem_id: str) -> List[Dict]:
    """Main neuroevolutionary solver using advanced features."""
    B = config['population_size']
    E = node_features.shape[1]
    device = node_features.device
    
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    start_gen = 0

    if os.path.exists(checkpoint_file):
        print(f"🔄 Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f: saved_state = pickle.load(f)
        population, elites, elite_scores, start_gen = saved_state['pop'], saved_state['elites'], saved_state['scores'], saved_state['gen'] + 1
        print(f"Resuming from generation {start_gen}")
    else:
        print("🌱 Initializing fresh population...")
        population = torch.randn((B, E), dtype=torch.float32, device=device) * 0.1
        elites = [None] * n 
        elite_scores = [(0, 501)] * n

    for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving"):
        logits = population @ node_features
        perms = logits.argsort(dim=1, descending=True).cpu().numpy()
        
        for i in range(B):
            perm = perms[i]
            for t in range(n):
                score = evaluate_heuristic(perm, t, n, adj)
                if dominates(score, elite_scores[t]):
                    elite_scores[t] = score
                    elites[t] = population[i].clone()
        
        non_dominated_elites = []
        for t1 in range(n):
            if elites[t1] is not None:
                if not any(dominates(elite_scores[t2], elite_scores[t1]) for t2 in range(n) if elites[t2] is not None):
                    non_dominated_elites.append(elites[t1])

        if not non_dominated_elites:
            non_dominated_elites = [e for e in elites if e is not None]
            if not non_dominated_elites: continue

        parent_indices = torch.randint(0, len(non_dominated_elites), (B,), device=device)
        population = torch.stack([non_dominated_elites[i] for i in parent_indices])
        
        mutation = torch.randn_like(population) * config['mutation_stdev']
        mask = torch.rand_like(population) < 0.4
        population[mask] += mutation[mask]

        if (gen + 1) % config['checkpoint_interval'] == 0:
            tqdm.write(f"\n💾 Checkpoint Gen {gen+1}: Found {len(non_dominated_elites)} non-dominated solutions.")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'pop': population, 'elites': elites, 'scores': elite_scores, 'gen': gen}, f)

    final_pareto_front = []
    for t in range(n):
        if elites[t] is not None:
            if not any(dominates(elite_scores[t2], elite_scores[t]) for t2 in range(n) if elites[t2] is not None):
                final_pareto_front.append({'solution_weights': elites[t], 'heuristic_score': elite_scores[t]})
    
    return final_pareto_front

# --- Main Execution & Submission ---

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    """NSGA-II selection, used here to select the final top solutions."""
    if not population: return []
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
                if len(front) > 1:
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

def create_submission_file(final_solutions: List[Dict], n: int, adj: np.ndarray, node_features: torch.Tensor, problem_id: str):
    filename = f"submission_{problem_id}.json"
    problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
    
    if not final_solutions:
        print("No solutions found to create a submission file.")
        return

    print(f"\n🔬 Performing final accurate evaluation of {len(final_solutions)} elite solutions...")
    
    decision_vectors_to_score = []
    for sol in final_solutions:
        weights = sol['solution_weights']
        t = n - sol['heuristic_score'][0]
        if t < 0 or t >= n: t = random.randint(0, n - 1)
        logits = weights @ node_features
        perm = logits.argsort(descending=True).cpu().tolist()
        decision_vectors_to_score.append(perm + [t])

    with multiprocessing.Pool() as pool:
        results = pool.map(evaluate_correct_task, [(tuple(sol), n, adj) for sol in decision_vectors_to_score])

    final_population = [{'solution': list(sol), 'score': score} for sol, score in results]
    final_submission_pop = crowding_selection(final_population, 20)
    final_vectors_for_json = [[int(val) for val in p['solution']] for p in final_submission_pop]

    submission = { "decisionVector": final_vectors_for_json, "problem": problem_name_map.get(problem_id, problem_id), "challenge": "spoc-3-torso-decompositions" }
    with open(filename, "w") as f: json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(final_vectors_for_json)} solutions.")

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
    node_features_T = init_node_features(n, adj, config['eigenvectors']).T # Transposed for matmul

    start_time = time.time()
    final_solutions = neuroevolution_search(n, adj, node_features_T, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

    create_submission_file(final_solutions, n, adj, node_features_T, problem_id)
