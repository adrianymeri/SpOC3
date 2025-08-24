#!/usr/bin/env python3
import os, sys, time, json, urllib.request, subprocess, random, ctypes
from typing import List, Set, Tuple, Dict
import numpy as np
from tqdm import tqdm
import multiprocessing
import pygmo as pg

# --- Auto-compile CUDA module ---
if not os.path.exists("evaluator.so") or os.path.getmtime("evaluator.cu") > os.path.getmtime("evaluator.so"):
    subprocess.check_call([sys.executable, "setup_cuda.py"])

libeval = ctypes.CDLL('./evaluator.so')
evaluate_on_gpu = libeval.evaluate_on_gpu
evaluate_on_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t, ctypes.c_size_t,
]

# CONFIGURATION
# ==============================================================================
CONFIG = {
    "general": {
        "num_islands": 8, "pop_size_per_island": 1024, "migration_interval": 15, 
        "migration_size": 50, "stagnation_limit": 40, "mutation_boost_factor": 2.0,
        "restart_stagnation_trigger": 100, "restart_fraction": 0.5,
        "elite_count": 16, "elite_ls_multiplier": 3,
        "crossover_rate": 0.9, "mutation_rate": 0.7
    },
    "easy": {"generations": 1000, "local_search_intensity": 5},
    "medium": {"generations": 1500, "local_search_intensity": 10},
    "hard": {"generations": 2500, "local_search_intensity": 15},
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker Functions (CPU for LS) and GPU Evaluator ---
def local_search_worker(args: Tuple[np.ndarray, int, np.ndarray]) -> np.ndarray:
    # This is a simplified LS, as the main search power comes from the massive population
    solution, intensity, adj_bits_np = args
    n = len(solution) - 1
    # For multiprocessing, we can't call the GPU, so we re-implement a simple evaluator
    # This is only used for LS, the main evaluation is on the GPU
    # (A full implementation would use a separate Cython evaluator for the CPU workers)
    return solution # Placeholder: For this pure GPU model, we rely on evolutionary operators

class GpuEvaluator:
    def __init__(self, n, adj_flat):
        self.n = n
        self.adj_flat = adj_flat.astype(np.bool_)
    
    def evaluate_batch(self, perms, ts):
        batch_size = len(perms)
        perms_gpu = perms.astype(np.uint16)
        degrees_gpu = np.empty((batch_size, self.n), dtype=np.uint16)
        
        evaluate_on_gpu(
            self.adj_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            perms_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            degrees_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            batch_size, self.n
        )
        
        widths = np.max(np.where(np.arange(self.n) >= ts[:, None], degrees_gpu, 0), axis=1)
        sizes = self.n - ts
        return np.vstack([sizes, widths]).T

# --- Core Algorithm Components ---
def load_graph(problem_id: str) -> Tuple[int, np.ndarray]:
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    edges, max_node = [], 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#'): continue
            u, v = map(int, line.strip().split())
            edges.append((u, v)); max_node = max(max_node, u, v)
    n = max_node + 1
    adj = np.zeros((n, n), dtype=bool)
    for u,v in edges: adj[u, v] = adj[v, u] = True
    print(f"✅ Loaded graph with {n} nodes.")
    return n, adj

def pmx_crossover(p1, p2): # NumPy optimized PMX
    n = p1.shape[1]
    a, b = np.sort(np.random.choice(n, 2, replace=False))
    
    p1_swath, p2_swath = p1[:, a:b+1], p2[:, a:b+1]
    
    # This is a simplified crossover for demonstration; a full PMX is more complex in NumPy
    # For speed, we'll use a simple uniform crossover here.
    mask = np.random.randint(0, 2, size=p1.shape, dtype=bool)
    child = np.where(mask, p1, p2)
    # Simple repair to ensure permutation
    for i in range(len(child)):
        if len(np.unique(child[i])) < n:
            child[i] = np.random.permutation(n)
    return child

# --- Main Memetic Algorithm ---
def memetic_algorithm(n: int, adj: np.ndarray, config: Dict, problem_id: str):
    num_islands, pop_size = config['num_islands'], config['pop_size_per_island']
    print(f"🧬 Initializing {num_islands} island populations of size {pop_size}...")
    islands_perms = [np.vstack([np.random.permutation(n) for _ in range(pop_size)]) for _ in range(num_islands)]
    islands_ts = [np.random.randint(0, n // 2, size=pop_size) for _ in range(num_islands)]
    
    gpu_evaluator = GpuEvaluator(n, adj.flatten())

    best_hv = -1.0
    pbar = tqdm(range(config['generations']), desc="🚀 Evolving on GPU")
    for gen in pbar:
        all_perms = np.vstack(islands_perms)
        all_ts = np.concatenate(islands_ts)
        
        all_fitnesses = gpu_evaluator.evaluate_batch(all_perms, all_ts)

        # Update islands with new fitnesses
        islands = []
        for i in range(num_islands):
            start, end = i * pop_size, (i+1) * pop_size
            island_pop = []
            for j in range(pop_size):
                island_pop.append({'solution': np.append(all_perms[start+j], all_ts[start+j]), 'score': tuple(all_fitnesses[start+j])})
            islands.append(island_pop)

        # Checkpoint and track best hypervolume
        non_dominated_indices = pg.non_dominated_front_2d(all_fitnesses)
        best_fitnesses_pygmo = [(f[1], -f[0]) for f in all_fitnesses[non_dominated_indices]] # (width, -size)
        ref_point = [n, 0]; hv = pg.hypervolume(best_fitnesses_pygmo); current_hv = -hv.compute(ref_point)

        if current_hv > best_hv:
            best_hv = current_hv
            pbar.write(f"✨ Gen {gen+1}: New best hypervolume {best_hv:,.2f}")
            with open(f"submission_{problem_id}_checkpoint.json", "w") as f:
                sols_to_save = [np.append(all_perms[i], all_ts[i]).tolist() for i in non_dominated_indices]
                json.dump({"decisionVector": sols_to_save}, f)

        pbar.set_postfix({"best_hv": f"{best_hv:,.0f}"})

        # --- Evolution and Migration ---
        new_islands_perms, new_islands_ts = [], []
        for i in range(num_islands):
            # Simple selection and crossover
            parent_indices = np.random.randint(0, pop_size, size=(pop_size, 2))
            p1_perms, p2_perms = islands_perms[i][parent_indices[:,0]], islands_perms[i][parent_indices[:,1]]
            
            child_perms = pmx_crossover(p1_perms, p2_perms)
            child_ts = np.random.randint(0, n // 2, size=pop_size) # Re-sample torsos
            
            new_islands_perms.append(child_perms)
            new_islands_ts.append(child_ts)
        islands_perms, islands_ts = new_islands_perms, new_islands_ts

# --- Entry Point ---
if __name__ == "__main__":
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS: sys.exit("❌ Invalid problem ID.")
    
    config = CONFIG['general'].copy(); config.update(CONFIG[problem_id])
    
    n, adj = load_graph(problem_id)
    memetic_algorithm(n, adj, config, problem_id)
