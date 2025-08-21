#!/usr/bin/env python3
import os, sys, time, json, urllib.request, subprocess
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import pygmo as pg
from concurrent.futures import ProcessPoolExecutor

# --- Auto-compile Cython module ---
def compile_cython_module():
    module_name = "solver_cython"
    try: import importlib.util; ext_suffix = importlib.util.EXTENSIONS[0]
    except (ImportError, AttributeError): import sysconfig; ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    cython_so_file = f"{module_name}{ext_suffix}"
    if not os.path.exists(cython_so_file) or os.path.getmtime("solver_cython.pyx") > os.path.getmtime(cython_so_file):
        print(f"🚀 Building/updating Cython module '{cython_so_file}'...")
        try:
            subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
            print("✅ Cython module built successfully.")
        except Exception as e:
            print(f"❌ Failed to build Cython module. Error: {e}")
            sys.exit(1)

compile_cython_module()
import solver_cython

# CONFIGURATION
# ==============================================================================
CONFIG = {
    "general": { "num_islands": os.cpu_count() or 8, "pop_size_per_island": 100, "migration_size": 10 },
    "easy": { "migration_interval_gens": 25, "total_generations": 1500 },
    "medium": { "migration_interval_gens": 30, "total_generations": 2500 },
    "hard": { "migration_interval_gens": 40, "total_generations": 4000 },
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker Initialization & Global UDP ---
UDP_INSTANCE = None
def init_worker(n_nodes, adj_bits_np, problem_id):
    """Initializes the problem object with pre-loaded data for each worker."""
    global UDP_INSTANCE
    UDP_INSTANCE = TorsoProblem(problem_id, n_nodes, adj_bits_np)

# PYGMO PROBLEM DEFINITION
# ==============================================================================
class TorsoProblem:
    def __init__(self, problem_id: str, n_nodes=None, adj_bits_np=None):
        if n_nodes is None: # Only called once in main process
            print(f"📥 Loading graph data for '{problem_id}'...")
            self.n_nodes, self.adj_bits_np = self._load_and_prep_graph(problem_id)
        else: # Workers receive pre-loaded data
            self.n_nodes, self.adj_bits_np = n_nodes, adj_bits_np
        
        solver_cython.init_worker_cython(self.n_nodes, self.adj_bits_np)

    @staticmethod
    def _load_and_prep_graph(problem_id):
        url = PROBLEMS[problem_id]
        edges, max_node = [], 0
        with urllib.request.urlopen(url) as f:
            for line in f:
                if line.startswith(b'#'): continue
                u, v = map(int, line.strip().split())
                edges.append((u, v)); max_node = max(max_node, u, v)
        n = max_node + 1
        adj = [set() for _ in range(n)]
        for u, v in edges: adj[u].add(v); adj[v].add(u)
        print(f"✅ Loaded graph with {n} nodes.")
        adj_bits = np.zeros(n, dtype=np.uint64)
        for u in range(n):
            for v in adj[u]: adj_bits[u] |= (np.uint64(1) << np.uint64(v))
        return n, adj_bits

    def fitness(self, x):
        return list(solver_cython.evaluate_solution_cy(np.array(x, dtype=np.int64)))

    def get_bounds(self):
        lb = [0] * (self.n_nodes + 1)
        ub = [self.n_nodes - 1] * self.n_nodes + [self.n_nodes]
        return (lb, ub)

    def get_nobj(self): return 2
    def get_nix(self): return self.n_nodes + 1

# THE EVOLUTIONARY ENGINE FOR A SINGLE ISLAND
# ==============================================================================
def evolve_island(args: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Evolves a single island's population using pygmo's C++ NSGA2."""
    initial_vectors, generations = args
    prob = pg.problem(UDP_INSTANCE)
    pop = pg.population(prob=prob, size=0)
    for vec in initial_vectors:
        pop.push_back(x=vec.astype(np.float64))
    
    algo = pg.algorithm(pg.nsga2(gen=generations))
    final_pop = algo.evolve(pop)
    return final_pop.get_x(), final_pop.get_f()

# --- MAIN ORCHESTRATOR ---
if __name__ == "__main__":
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS: sys.exit("❌ Invalid problem ID.")
    
    config = CONFIG['general'].copy(); config.update(CONFIG[problem_id])
    
    main_udp = TorsoProblem(problem_id)
    num_islands = config['num_islands']
    pop_size_per_island = config['pop_size_per_island']
    
    print(f"🧬 Initializing {num_islands} island populations of size {pop_size_per_island}...")
    base_perm = np.arange(main_udp.n_nodes, dtype=np.int64)
    island_populations_x = []
    for _ in range(num_islands):
        island_pop = []
        for _ in range(pop_size_per_island):
            np.random.shuffle(base_perm)
            t = np.random.randint(0, main_udp.n_nodes // 2)
            island_pop.append(np.append(base_perm.copy(), t))
        island_populations_x.append(np.array(island_pop))

    n_evolutions = config['total_generations'] // config['migration_interval_gens']
    
    # Pass the pre-loaded graph data to the workers when they are created
    init_args = (main_udp.n_nodes, main_udp.adj_bits_np, problem_id)
    with ProcessPoolExecutor(max_workers=num_islands, initializer=init_worker, initargs=init_args) as executor:
        for evo_step in range(n_evolutions):
            print(f"\n🏝️  Evolution Step {evo_step + 1} / {n_evolutions}...")
            
            args_for_workers = [(pop, config['migration_interval_gens']) for pop in island_populations_x]
            results = list(tqdm(executor.map(evolve_island, args_for_workers), total=num_islands, desc="Evolving islands"))

            all_solutions_x = np.concatenate([res[0] for res in results])
            all_solutions_f = np.concatenate([res[1] for res in results])

            print("🔄 Performing migration...")
            non_dominated_indices = pg.non_dominated_front_2d(all_solutions_f)
            best_individuals_x = all_solutions_x[non_dominated_indices]

            for i in range(num_islands):
                current_island_x = island_populations_x[i] # Use the state before evolution
                num_to_replace = min(config['migration_size'], len(best_individuals_x), len(current_island_x))
                if num_to_replace > 0:
                    migrants_indices = np.random.choice(len(best_individuals_x), num_to_replace, replace=False)
                    # Replace the first 'n' individuals with migrants
                    current_island_x[:num_to_replace] = best_individuals_x[migrants_indices]
                # Shuffle the island to mix migrants with the existing population
                np.random.shuffle(current_island_x)
                island_populations_x[i] = current_island_x

            # --- Occasional Reporting & Submission File Creation ---
            best_fitnesses = all_solutions_f[non_dominated_indices]
            ref_point = [main_udp.n_nodes, 0]
            hv = pg.hypervolume(best_fitnesses)
            current_hv = -hv.compute(ref_point)
            
            # Find the single best point for display
            readable_scores = [(int(-f[1]), int(f[0])) for f in best_fitnesses]
            best_point = max(readable_scores, key=lambda s: (s[0], -s[1]))
            
            print(f"✨ Step {evo_step + 1} complete. Best point: (size={best_point[0]}, width={best_point[1]}). Hypervolume: {current_hv:,.2f}")
            
            # Save a checkpoint submission file
            with open(f"submission_{problem_id}_step{evo_step+1}.json", "w") as f:
                best_for_submission = [v.astype(np.int64).tolist() for v in best_individuals_x]
                json.dump({"decisionVector": best_for_submission, "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
            print(f"📄 Created checkpoint submission file for step {evo_step+1}")


    print("\n✅ Evolution complete. Finalizing results...")
    final_solutions_f = np.concatenate([res[1] for res in results])
    final_solutions_x = np.concatenate([res[0] for res in results])
    non_dominated_idx = pg.non_dominated_front_2d(final_solutions_f)
    best_fitnesses = final_solutions_f[non_dominated_idx]
    best_solutions = final_solutions_x[non_dominated_idx]
    
    readable_scores = [(int(-f[1]), int(f[0])) for f in best_fitnesses]
    best_idx = max(range(len(readable_scores)), key=lambda i: (readable_scores[i][0], -readable_scores[i][1]))
    best_solution = best_solutions[best_idx]
    best_score = readable_scores[best_idx]
    
    print(f"\n🏆 Final best individual solution (size, width): {best_score}")
    
    ref_point = [main_udp.n_nodes, 0] 
    hv = pg.hypervolume(best_fitnesses)
    final_hv = -hv.compute(ref_point)
    print(f"📈 Final Hypervolume: {final_hv:,.2f}")

    with open(f"submission_{problem_id}.json", "w") as f:
        json.dump({"decisionVector": [s.astype(np.int64).tolist() for s in best_solutions], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
    print(f"📄 Created final submission file: submission_{problem_id}.json")
