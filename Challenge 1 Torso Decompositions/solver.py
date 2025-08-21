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
    "general": {
        "num_islands": os.cpu_count() or 8,
        "pop_size_per_island": 100,
        "migration_size": 10,
        "elite_fraction": 0.1,
        "elite_ls_multiplier": 3,
    },
    "easy": { "migration_interval_gens": 20, "total_generations": 1000, "local_search_intensity": 20 },
    "medium": { "migration_interval_gens": 25, "total_generations": 2000, "local_search_intensity": 25 },
    "hard": { "migration_interval_gens": 30, "total_generations": 3000, "local_search_intensity": 30 },
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker Initialization & Global UDP ---
UDP_INSTANCE = None
def init_worker(problem_id):
    global UDP_INSTANCE
    UDP_INSTANCE = TorsoProblem(problem_id)

# PYGMO PROBLEM DEFINITION
# ==============================================================================
class TorsoProblem:
    def __init__(self, problem_id: str):
        self.problem_id = problem_id
        self.n_nodes, self.adj_bits_np = self._load_and_prep_graph(problem_id)
        solver_cython.init_worker_cython(self.n_nodes, self.adj_bits_np)

    @staticmethod
    def _load_and_prep_graph(problem_id):
        url = PROBLEMS[problem_id]
        print(f"📥 Loading graph data for '{problem_id}' (worker)...")
        edges, max_node = [], 0
        with urllib.request.urlopen(url) as f:
            for line in f:
                if line.startswith(b'#'): continue
                u, v = map(int, line.strip().split())
                edges.append((u, v)); max_node = max(max_node, u, v)
        n = max_node + 1
        adj = [set() for _ in range(n)]
        for u, v in edges: adj[u].add(v); adj[v].add(u)
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
def evolve_island(args: Tuple[np.ndarray, int, int, float, int]) -> Tuple[np.ndarray, np.ndarray]:
    initial_vectors, generations, ls_intensity, elite_frac, ls_mult = args
    prob = pg.problem(UDP_INSTANCE)
    pop = pg.population(prob=prob, size=0)
    for vec in initial_vectors:
        pop.push_back(x=vec.astype(np.float64))
    
    algo = pg.algorithm(pg.nsga2(gen=generations))
    pop = algo.evolve(pop)

    elite_count = int(len(pop.get_x()) * elite_frac)
    if elite_count > 0:
        fronts = pg.sort_population_mo(points=pop.get_f())
        elite_indices = fronts[0][:elite_count]
        elite_vectors = pop.get_x()[elite_indices]
        intensified_elites = [local_search_py(v, ls_intensity * ls_mult) for v in elite_vectors]
        for elite_vec in intensified_elites:
            pop.push_back(x=elite_vec.astype(np.float64))

    return pop.get_x(), pop.get_f()

def local_search_py(solution, intensity):
    n = UDP_INSTANCE.n_nodes
    best_sol = solution.astype(np.int64)
    best_score = UDP_INSTANCE.fitness(best_sol)
    
    for _ in range(intensity):
        cand_sol = best_sol.copy()
        r = random.random()
        perm = cand_sol[:-1]

        if r < 0.5: # Inversion
            a, b = sorted(random.sample(range(n), 2))
            perm[a:b+1] = perm[a:b+1][::-1]
        else: # Torso Shift
            t = cand_sol[-1]
            shift = max(1, n // 20)
            new_t = t + random.randint(-shift, shift)
            cand_sol[-1] = max(0, min(n, int(new_t)))

        cand_score = UDP_INSTANCE.fitness(cand_sol)

        if (cand_score[0] < best_score[0]) or \
           (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
            best_sol = cand_sol
            best_score = cand_score
            
    return best_sol

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
    
    with ProcessPoolExecutor(max_workers=num_islands, initializer=init_worker, initargs=(problem_id,)) as executor:
        for evo_step in range(n_evolutions):
            print(f"\n🏝️  Evolution Step {evo_step + 1} / {n_evolutions}...")
            
            # THIS IS THE CORRECTED LINE
            ls_args = (config['local_search_intensity'], config['elite_fraction'], config['elite_ls_multiplier'])
            
            args_for_workers = [(pop, config['migration_interval_gens']) + ls_args for pop in island_populations_x]
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

            with open(f"submission_{problem_id}_step{evo_step+1}.json", "w") as f:
                best_for_submission = [v.astype(np.int64).tolist() for v in best_individuals_x]
                json.dump({"decisionVector": best_for_submission, "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
            print(f"📄 Created checkpoint submission file for step {evo_step+1}")

    print("\n✅ Evolution complete. Calculating final hypervolume...")
    final_solutions_f = np.concatenate([res[1] for res in results])
    non_dominated_idx = pg.non_dominated_front_2d(final_solutions_f)
    best_fitnesses = final_solutions_f[non_dominated_idx]
    
    ref_point = [main_udp.n_nodes, 0] 
    hv = pg.hypervolume(best_fitnesses)
    final_hv = -hv.compute(ref_point)
    print(f"🏆 Final Hypervolume: {final_hv:,.2f}")
