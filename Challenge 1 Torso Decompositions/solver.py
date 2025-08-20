#!/usr/bin/env python3
import os, sys, time, random, json, urllib.request, subprocess
from typing import List, Set, Tuple, Dict
import numpy as np
from tqdm import tqdm
import pygmo as pg

# --- Auto-compile Cython module ---
def compile_cython_module():
    """Checks for the compiled Cython module and compiles it if missing."""
    module_name = "solver_cython"
    # Find the platform-specific shared object file extension
    try:
        import importlib.util
        ext_suffix = importlib.util.EXTENSIONS[0] # e.g., .cpython-310-x86_64-linux-gnu.so
    except (ImportError, AttributeError): # Fallback for older python
         import sysconfig
         ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    cython_so_file = f"{module_name}{ext_suffix}"

    if not os.path.exists(cython_so_file):
        print(f"🚀 Compiled Cython module '{cython_so_file}' not found. Building...")
        try:
            # Use setup.py to build the module
            subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
            print("✅ Cython module built successfully.")
        except Exception as e:
            print(f"❌ Failed to build Cython module. Error: {e}")
            sys.exit(1)

compile_cython_module()
import solver_cython # Import the now-guaranteed-to-exist module

# CONFIGURATION
# ==============================================================================
CONFIG = {
    "general": { "islands": os.cpu_count() or 8, "pop_size": 100 },
    "easy": { "generations": 200 },
    "medium": { "generations": 350 },
    "hard": { "generations": 500 },
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# PYGMO PROBLEM DEFINITION (UDP)
# ==============================================================================
class TorsoProblem:
    def __init__(self, problem_id: str):
        self.problem_id = problem_id
        self.n_nodes, self.adj_list = self._load_graph()
        self.adj_bits_np = self._build_adj_bitsets()
        # Initialize the Cython worker globals for the main process
        solver_cython.init_worker_cython(self.n_nodes, self.adj_bits_np)

    def _load_graph(self):
        url = PROBLEMS[self.problem_id]
        print(f"📥 Loading graph data for '{self.problem_id}'...")
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
        return n, adj

    def _build_adj_bitsets(self):
        adj_bits = np.zeros(self.n_nodes, dtype=np.uint64)
        for u in range(self.n_nodes):
            for v in self.adj_list[u]: adj_bits[u] |= (np.uint64(1) << np.uint64(v))
        return adj_bits

    def fitness(self, x):
        # This is the crucial link: pygmo calls this, and we call our fast C code.
        # The objective is to MINIMIZE width and MINIMIZE -size (i.e., maximize size).
        x_int = np.array(x, dtype=np.int64)
        width, neg_size = solver_cython.evaluate_solution_cy(x_int)
        return [width, neg_size]

    def get_bounds(self):
        # Permutation part: 0 to n-1. Torso part: 0 to n.
        lb = [0] * (self.n_nodes + 1)
        ub = [self.n_nodes - 1] * self.n_nodes + [self.n_nodes]
        return (lb, ub)

    def get_nobj(self):
        return 2

    def get_nix(self):
        return self.n_nodes + 1
    
    # We define a custom random population to create valid permutations
    def create_random_population(self, n_individuals):
        pop = []
        base_perm = np.arange(self.n_nodes, dtype=np.int64)
        for _ in range(n_individuals):
            np.random.shuffle(base_perm)
            t = np.random.randint(0, self.n_nodes // 2)
            pop.append(np.append(base_perm, t))
        return np.array(pop)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS: sys.exit("❌ Invalid problem ID.")
    
    config = CONFIG['general'].copy(); config.update(CONFIG[problem_id])

    # 1. Create the problem instance
    udp = TorsoProblem(problem_id)
    prob = pg.problem(udp)

    # 2. Define the algorithm (NSGA-II)
    # We can run it for many generations because our fitness function is so fast.
    algo = pg.algorithm(pg.nsga2(gen=config['generations']))

    # 3. Set up the Archipelago (parallel islands)
    # This is pygmo's powerful, built-in C++ parallelism.
    print(f"🏝️ Setting up archipelago with {config['islands']} islands...")
    archi = pg.archipelago(n=config['islands'], algo=algo, prob=prob, pop_size=config['pop_size'],
                           # This tells pygmo how to create valid random solutions
                           seed=int(time.time()))
    # We must explicitly set the initial population using our custom creator
    initial_pop = udp.create_random_population(config['islands'] * config['pop_size'])
    archi.set_champions(x=initial_pop)


    # 4. Evolve in parallel
    print("🚀 Evolving...")
    archi.evolve()
    archi.wait()
    print("✅ Evolution complete.")

    # 5. Extract results
    solutions = archi.get_champions_x()
    fitnesses = archi.get_champions_f()
    
    # Filter for non-dominated solutions
    non_dominated_idx = pg.non_dominated_front_2d(fitnesses)
    final_solutions = solutions[non_dominated_idx]
    final_fitnesses = fitnesses[non_dominated_idx]

    # Convert fitness back to (size, width) for readability
    readable_scores = [(int(-f[1]), int(f[0])) for f in final_fitnesses]
    best_idx = max(range(len(readable_scores)), key=lambda i: (readable_scores[i][0], -readable_scores[i][1]))
    best_solution = final_solutions[best_idx]
    best_score = readable_scores[best_idx]

    print(f"\n🏆 Final best solution score (size, width): {best_score}")
    
    # --- Create Submission File ---
    with open(f"submission_{problem_id}.json", "w") as f:
        json.dump({"decisionVector": [[int(v) for v in best_solution]], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
    print(f"📄 Created submission file: submission_{problem_id}.json")

    # --- Hypervolume Calculation ---
    # Reference point: worst possible score [max_width, -min_size] = [n_nodes, -0]
    ref_point = [udp.n_nodes, 0] 
    hv = pg.hypervolume(final_fitnesses)
    print(f"📈 Hypervolume: {-hv.compute(ref_point):,.2f}") # We negate it back for the leaderboard score
