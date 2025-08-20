#!/usr/bin/env python3
import os, sys, time, json, urllib.request, subprocess
from typing import Tuple
import numpy as np
from tqdm import tqdm
import pygmo as pg
from concurrent.futures import ProcessPoolExecutor

# --- Auto-compile Cython module ---
def compile_cython_module():
    """Checks for and builds the Cython module if it's missing."""
    module_name = "solver_cython"
    try: # Modern Python
        import importlib.util
        ext_suffix = importlib.util.EXTENSIONS[0]
    except (ImportError, AttributeError): # Fallback for older Python
         import sysconfig
         ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'

    cython_so_file = f"{module_name}{ext_suffix}"
    if not os.path.exists(cython_so_file):
        print(f"🚀 Building Cython module '{cython_so_file}'...")
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
    "general": { "num_islands": os.cpu_count() or 8, "pop_size_per_island": 100, "migration_size": 8 },
    "easy": { "migration_interval_gens": 25, "total_generations": 1500 },
    "medium": { "migration_interval_gens": 30, "total_generations": 2500 },
    "hard": { "migration_interval_gens": 40, "total_generations": 4000 },
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker Initialization ---
UDP_INSTANCE = None

def init_worker(problem_id):
    """Initializes the problem object and Cython module for each worker process."""
    global UDP_INSTANCE
    UDP_INSTANCE = TorsoProblem(problem_id, initialize_cython=True)

# PYGMO PROBLEM DEFINITION
# ==============================================================================
class TorsoProblem:
    def __init__(self, problem_id: str, n_nodes=None, adj_bits_np=None, initialize_cython=False):
        if n_nodes is None:
            print(f"📥 Loading graph data for '{problem_id}'...")
            self.n_nodes, self.adj_bits_np = self._load_and_prep_graph(problem_id)
        else:
            self.n_nodes, self.adj_bits_np = n_nodes, adj_bits_np
        
        if initialize_cython:
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
        x_int = np.array(x, dtype=np.int64)
        return list(solver_cython.evaluate_solution_cy(x_int))

    def get_bounds(self):
        lb = [0] * (self.n_nodes + 1)
        ub = [self.n_nodes - 1] * self.n_nodes + [self.n_nodes]
        return (lb, ub)

    def get_nobj(self): return 2
    def get_nix(self): return self.n_nodes + 1

# THE EVOLUTIONARY ENGINE FOR A SINGLE ISLAND
# ==============================================================================
def evolve_island(args: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Takes a population, evolves it using pygmo's NSGA2, and returns the result."""
    initial_vectors, generations = args
    prob = pg.problem(UDP_INSTANCE)
    
    # THE FIX for TypeError: Create an empty population, then push back individuals.
    pop = pg.population(prob=prob, size=0)
    for vec in initial_vectors:
        # pygmo requires float vectors, so we cast here.
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
    
    init_args = (problem_id,)
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
                current_island_x = results[i][0]
                num_to_replace = min(config['migration_size'], len(best_individuals_x), len(current_island_x))
                if num_to_replace > 0:
                    migrants_indices = np.random.choice(len(best_individuals_x), num_to_replace, replace=False)
                    current_island_x[:num_to_replace] = best_individuals_x[migrants_indices]
                island_populations_x[i] = current_island_x

            current_best_f = min(all_solutions_f.tolist(), key=lambda f: (f[0], f[1]))
            current_best_score = (int(-current_best_f[1]), int(current_best_f[0]))
            print(f"✨ Best score after step {evo_step + 1}: (size={current_best_score[0]}, width={current_best_score[1]})")

    print("\n✅ Evolution complete. Finding best overall solution...")
    final_solutions_x = np.concatenate(island_populations_x)
    final_solutions_f = np.array([main_udp.fitness(x) for x in final_solutions_x])
    
    non_dominated_idx = pg.non_dominated_front_2d(final_solutions_f)
    best_solutions = final_solutions_x[non_dominated_idx]
    best_fitnesses = final_solutions_f[non_dominated_idx]

    readable_scores = [(int(-f[1]), int(f[0])) for f in best_fitnesses]
    best_idx = max(range(len(readable_scores)), key=lambda i: (readable_scores[i][0], -readable_scores[i][1]))
    best_solution = best_solutions[best_idx]
    best_score = readable_scores[best_idx]

    print(f"\n🏆 Final best solution score (size, width): {best_score}")
    
    with open(f"submission_{problem_id}.json", "w") as f:
        json.dump({"decisionVector": [[int(v) for v in best_solution]], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
    print(f"📄 Created submission file: submission_{problem_id}.json")

    ref_point = [main_udp.n_nodes, 0] 
    hv = pg.hypervolume(best_fitnesses)
    print(f"📈 Hypervolume: {-hv.compute(ref_point):,.2f}")
