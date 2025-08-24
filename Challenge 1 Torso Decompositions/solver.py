#!/usr/bin/env python3
import os, sys, time, json, urllib.request, subprocess, random, ctypes
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import multiprocessing
import pygmo as pg

# --- Auto-compile CUDA module ---
if not os.path.exists("evaluator.so") or (os.path.exists("evaluator.cu") and os.path.getmtime("evaluator.cu") > os.path.getmtime("evaluator.so")):
    subprocess.check_call([sys.executable, "setup_cuda.py"])

libeval = ctypes.CDLL('./evaluator.so')
evaluate_on_gpu = libeval.evaluate_on_gpu
evaluate_on_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t, ctypes.c_size_t,
]

# --- CONFIGURATION (Tuned for aggressive GPU search) ---
CONFIG = {
    "general": {
        "pop_size": 8192, # Massive population for the GPU
        "elite_fraction": 0.1,
        "mutation_rate": 0.5
    },
    "easy": { "generations": 2000 },
    "medium": { "generations": 4000 },
    "hard": { "generations": 8000 },
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- GPU Evaluator Class ---
class GpuEvaluator:
    def __init__(self, n, adj):
        self.n = n
        self.adj_flat = adj.astype(np.bool_).flatten()
    
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
        widths[widths > 500] = 501
        return np.vstack([sizes, widths]).T

# --- Helper Functions ---
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

# --- Main GPU-Accelerated Genetic Algorithm ---
def main():
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS: sys.exit("❌ Invalid problem ID.")
    
    config = CONFIG['general'].copy(); config.update(CONFIG[problem_id])
    n, adj = load_graph(problem_id)
    pop_size = config['pop_size']
    
    gpu_evaluator = GpuEvaluator(n, adj)
    
    print(f"🧬 Initializing population of size {pop_size} for GPU...")
    perms = np.vstack([np.random.permutation(n) for _ in range(pop_size)])
    ts = np.random.randint(0, n // 2, size=pop_size)
    
    best_hv = -1.0

    pbar = tqdm(range(config['generations']), desc="🚀 Evolving on GPU")
    for gen in pbar:
        fitnesses = gpu_evaluator.evaluate_batch(perms, ts)
        
        # --- Elitist Selection ---
        # Find the best individuals based on a simple lexicographical sort
        elite_indices = np.lexsort((-fitnesses[:, 0], fitnesses[:, 1]))[:int(pop_size * config['elite_fraction'])]
        
        # --- Create Next Generation ---
        parent_indices = np.random.choice(elite_indices, size=pop_size)
        
        # Crossover (Simple Uniform Crossover with repair)
        p1_perms = perms[np.random.choice(parent_indices, size=pop_size)]
        p2_perms = perms[np.random.choice(parent_indices, size=pop_size)]
        mask = np.random.randint(0, 2, size=p1_perms.shape, dtype=bool)
        child_perms = np.where(mask, p1_perms, p2_perms)
        
        # Mutation (Inversion)
        for i in range(pop_size):
            if random.random() < config['mutation_rate']:
                a, b = sorted(random.sample(range(n), 2))
                child_perms[i, a:b+1] = child_perms[i, a:b+1][::-1]

        # Repair permutations to ensure validity
        for i in range(pop_size):
            if len(np.unique(child_perms[i])) < n:
                child_perms[i] = np.random.permutation(n)
        
        perms = child_perms
        ts = np.random.randint(0, n // 2, size=pop_size) # Re-sample torsos for exploration

        # --- Reporting & Checkpointing ---
        if gen % 10 == 0:
            non_dominated_indices = pg.non_dominated_front_2d(fitnesses)
            best_fitnesses = fitnesses[non_dominated_indices]
            ref_point = [n, 501]; hv_fitnesses = [(f[1], -f[0]) for f in best_fitnesses]
            hv = pg.hypervolume(hv_fitnesses); current_hv = -hv.compute(ref_point)

            if current_hv > best_hv:
                best_hv = current_hv
                pbar.write(f"✨ Gen {gen+1}: New best hypervolume {best_hv:,.2f}")
                with open(f"submission_{problem_id}_checkpoint.json", "w") as f:
                    sols_to_save = [np.append(perms[i], ts[i]).astype(int).tolist() for i in non_dominated_indices]
                    json.dump({"decisionVector": sols_to_save, "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f)

            pbar.set_postfix({"best_hv": f"{best_hv:,.0f}"})

if __name__ == "__main__":
    main()
