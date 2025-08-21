#!/usr/bin/env python3
import os, sys, time, json, urllib.request, subprocess, random
from typing import List, Set, Tuple, Dict
import numpy as np
from tqdm import tqdm
import multiprocessing
import pygmo as pg

# --- Auto-compile Cython module ---
def compile_cython_module():
    module_name = "solver_cython"
    pyx_file = "solver_cython.pyx"
    try: import importlib.util; ext_suffix = importlib.util.EXTENSIONS[0]
    except (ImportError, AttributeError): import sysconfig; ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    
    cython_so_file = f"{module_name}{ext_suffix}"
    if not os.path.exists(cython_so_file) or os.path.getmtime(pyx_file) > os.path.getmtime(cython_so_file):
        print(f"🚀 Building/updating Cython module...")
        try:
            subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
            print("✅ Cython module built successfully.")
        except Exception as e:
            print(f"❌ Failed to build Cython module: {e}"); sys.exit(1)

compile_cython_module()
import solver_cython

# CONFIGURATION
# ==============================================================================
CONFIG = {
    "general": {
        "mutation_rate": 0.6, "crossover_rate": 0.9, "num_islands": os.cpu_count() or 8,
        "elite_count": 8, "elite_ls_multiplier": 4, "stagnation_limit": 50, 
        "mutation_boost_factor": 1.5, "migration_interval": 25, "migration_size": 5,
        "restart_stagnation_trigger": 150, "restart_fraction": 0.25
    },
    "easy": {"pop_size_per_island": 80, "generations": 2000, "local_search_intensity": 25},
    "medium": {"pop_size_per_island": 100, "generations": 3000, "local_search_intensity": 30},
    "hard": {"pop_size_per_island": 120, "generations": 5000, "local_search_intensity": 35},
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker Functions (for the multiprocessing Pool) ---
# ... (These are unchanged and correct, omitted for brevity) ...

# --- Core Algorithm Components ---
# ... (These are unchanged and correct, omitted for brevity) ...

# --- Main Memetic Algorithm ---
def memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict, problem_id: str):
    adj_bits_np = build_adj_bitsets_np(n, adj_list)
    num_islands, pop_size_per_island = config['num_islands'], config['pop_size_per_island']
    
    print(f"🧬 Initializing {num_islands} island populations...")
    islands = []
    base_perm = np.arange(n, dtype=np.int64)
    for _ in range(num_islands):
        island_pop = []
        for _ in range(pop_size_per_island):
            np.random.shuffle(base_perm)
            t = np.random.randint(int(n * 0.1), int(n * 0.5))
            island_pop.append({'solution': np.append(base_perm.copy(), t)})
        islands.append(island_pop)

    best_score, stagnation_counter = (-1, 999), 0
    base_mutation = config['mutation_rate']

    with multiprocessing.Pool(initializer=init_worker, initargs=(n, adj_bits_np)) as pool:
        for i in range(num_islands):
            results = pool.map(eval_wrapper, [p['solution'] for p in islands[i]])
            for sol, score in results:
                for p in islands[i]:
                    if np.array_equal(p['solution'], sol): p['score'] = score; break
        
        pbar = tqdm(range(config['generations']), desc="🚀 Evolving")
        for gen in pbar:
            mutation_rate = base_mutation * (config['mutation_boost_factor'] if stagnation_counter >= config['stagnation_limit'] else 1.0)
            
            for i in range(num_islands):
                pop = islands[i]
                mating_pool = crowding_selection(pop, len(pop))
                offspring_sols = []
                while len(offspring_sols) < len(pop):
                    p1, p2 = random.sample(mating_pool, 2)
                    child_perm = pmx_crossover(p1['solution'][:-1], p2['solution'][:-1]) if random.random() < config['crossover_rate'] else p1['solution'][:-1].copy()
                    if random.random() < mutation_rate:
                        # NEW: Stronger mutation choices
                        if random.random() < 0.5:
                             a, b = sorted(random.sample(range(n), 2))
                             child_perm[a:b+1] = child_perm[a:b+1][::-1] # Inversion
                        else:
                             a, b = np.random.choice(n, 2, replace=False); child_perm[a], child_perm[b] = child_perm[b], child_perm[a] # Swap
                    t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                    offspring_sols.append(np.append(child_perm, t))

                improved_offspring = pool.map(local_search_worker, [(sol, config['local_search_intensity']) for sol in offspring_sols])
                results = pool.map(eval_wrapper, improved_offspring)
                offspring_pop = [{'solution': sol, 'score': score} for sol, score in results]
                islands[i] = crowding_selection(pop + offspring_pop, len(pop))
                
                pop_sorted = sorted(islands[i], key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                elite_args = [(p['solution'], config['local_search_intensity'] * config['elite_ls_multiplier']) for p in pop_sorted[:config['elite_count']]]
                if elite_args:
                    intensified_elites = pool.map(local_search_worker, elite_args)
                    results = pool.map(eval_wrapper, intensified_elites)
                    islands[i].extend([{'solution': sol, 'score': score} for sol, score in results])
                    islands[i] = crowding_selection(islands[i], len(pop))

            # --- Population Restart Logic ---
            if stagnation_counter > config['restart_stagnation_trigger']:
                pbar.write(f"⚠️ Stagnation limit {config['restart_stagnation_trigger']} reached. Shaking up population...")
                for i in range(num_islands):
                    num_to_replace = int(len(islands[i]) * config['restart_fraction'])
                    for j in range(num_to_replace):
                        np.random.shuffle(base_perm)
                        t = np.random.randint(int(n * 0.1), int(n * 0.5))
                        new_sol = np.append(base_perm.copy(), t)
                        # Replace the worst individuals
                        islands[i][-(j+1)]['solution'] = new_sol
                # Re-evaluate the new individuals
                for i in range(num_islands):
                    num_to_replace = int(len(islands[i]) * config['restart_fraction'])
                    sols_to_eval = [islands[i][-(j+1)]['solution'] for j in range(num_to_replace)]
                    results = pool.map(eval_wrapper, sols_to_eval)
                    res_dict = {tuple(sol): score for sol, score in results}
                    for j in range(num_to_replace):
                        islands[i][-(j+1)]['score'] = res_dict.get(tuple(islands[i][-(j+1)]['solution']))
                stagnation_counter = 0 # Reset counter after restart

            # --- Migration and Tracking ---
            if gen > 0 and gen % config['migration_interval'] == 0:
                all_individuals = sorted([ind for island in islands for ind in island], key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                migrants = [p['solution'] for p in all_individuals[:config['migration_size']]]
                if migrants:
                    for island in islands:
                        for j in range(len(migrants)):
                            if j < len(island): island[-(j+1)]['solution'] = migrants[j]

            found_better = False
            for island in islands:
                current_best_score = sorted(island, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)[0]['score']
                if (current_best_score[0] > best_score[0]) or (current_best_score[0] == best_score[0] and current_best_score[1] < best_score[1]):
                    best_score = current_best_score
                    found_better = True
            
            if found_better:
                stagnation_counter = 0
                pbar.write(f"✨ Gen {gen+1}: New best score {best_score}")
            else:
                stagnation_counter += 1
            pbar.set_postfix({"best_score": best_score, "stagn": stagnation_counter})
    
    # --- Final result collection ---
    # ... (Omitted for brevity, this part is correct) ...

# --- (The rest of the script, including omitted functions, is below) ---

# --- Entry Point ---
if __name__ == "__main__":
    # ... (This part is correct) ...

# --- Full Function Definitions (Omitted from above for brevity) ---
def load_graph...
# ... etc ...
