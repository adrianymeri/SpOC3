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
def init_worker(n_val, adj_bits_val):
    solver_cython.init_worker_cython(n_val, adj_bits_val)
    random.seed(); np.random.seed()

def eval_wrapper(solution_np: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    size, width = solver_cython.evaluate_solution_cy(solution_np)
    return solution_np, (int(size), int(width))

def local_search_worker(args: Tuple[np.ndarray, int]) -> np.ndarray:
    solution, intensity = args
    n = len(solution) - 1
    best_sol = solution
    _, best_score = eval_wrapper(best_sol)
    
    for _ in range(intensity):
        cand_sol = best_sol.copy()
        r = random.random()
        perm = cand_sol[:-1]

        if r < 0.4: # Inversion
            a, b = sorted(random.sample(range(n), 2))
            perm[a:b+1] = perm[a:b+1][::-1]
        elif r < 0.8: # Block Move
            block_size = random.randint(2, max(3, n // 50))
            if n > block_size:
                start = random.randint(0, n - block_size)
                block = perm[start:start+block_size]
                perm_deleted = np.delete(perm, np.s_[start:start+block_size])
                insert_pos = random.randint(0, len(perm_deleted))
                cand_sol[:-1] = np.insert(perm_deleted, insert_pos, block)
        else: # Torso Shift
            t = cand_sol[-1]
            shift = max(1, n // 20)
            new_t = t + random.randint(-shift, shift)
            cand_sol[-1] = max(0, min(n - 1, int(new_t)))

        _, cand_score = eval_wrapper(cand_sol)

        if (cand_score[0] > best_score[0]) or (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
            best_sol = cand_sol
            best_score = cand_score
            
    return best_sol

# --- Core Algorithm Components ---
def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
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

def build_adj_bitsets_np(n: int, adj_list: List[Set[int]]) -> np.ndarray:
    adj_bits = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        for v in adj_list[u]: adj_bits[u] |= (np.uint64(1) << np.uint64(v))
    return adj_bits

def pmx_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=np.int64)
    child[a:b+1] = p1[a:b+1]
    for i in range(a, b + 1):
        val = p2[i]
        if val not in child:
            pos = i
            while True:
                mapped = p1[pos]
                try: pos = np.where(p2 == mapped)[0][0]
                except IndexError: break
                if child[pos] == -1: child[pos] = val; break
    for i in range(n):
        if child[i] == -1: child[i] = p2[i]
    return child

def dominates(p_score, q_score) -> bool:
    return (p_score[0] >= q_score[0] and p_score[1] < q_score[1]) or \
           (p_score[0] > q_score[0] and p_score[1] <= q_score[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    if len(population) <= pop_size: return population
    for p in population: p['dominates_set'], p['dominated_by_count'] = [], 0
    fronts = [[]]
    for p in population:
        for q in population:
            if p is q: continue
            if dominates(p['score'], q['score']): p['dominates_set'].append(q)
            elif dominates(q['score'], p['score']): p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0: fronts[0].append(p)
    new_population = []
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0: next_front.append(q)
        if len(new_population) + len(fronts[i]) > pop_size:
            for p in fronts[i]: p['distance'] = 0.0
            for i_obj in range(2):
                fronts[i].sort(key=lambda p: p['score'][i_obj])
                f_min, f_max = fronts[i][0]['score'][i_obj], fronts[i][-1]['score'][i_obj]
                fronts[i][0]['distance'] = fronts[i][-1]['distance'] = float('inf')
                if f_max > f_min:
                    for j in range(1, len(fronts[i]) - 1):
                        fronts[i][j]['distance'] += (fronts[i][j+1]['score'][i_obj] - fronts[i][j-1]['score'][i_obj]) / (f_max - f_min)
            fronts[i].sort(key=lambda p: p['distance'], reverse=True)
            new_population.extend(fronts[i][:pop_size - len(new_population)])
            break
        new_population.extend(fronts[i])
        fronts.append(next_front)
        i += 1
    return new_population

# --- NEW CHECKPOINTING FUNCTION ---
def persist_checkpoint(problem_id: str, islands: List[List[Dict]]):
    """Saves the entire non-dominated front from all islands to a submission file."""
    filename = f"submission_{problem_id}_checkpoint.json"
    
    # Gather all individuals from all islands
    all_individuals = [ind for island in islands for ind in island]
    
    # Find the non-dominated front
    final_front = crowding_selection(all_individuals, 20) # Get up to 20 best solutions
    final_solutions = [p['solution'] for p in final_front]

    # Write to the submission file
    with open(filename, "w") as f:
        json.dump({"decisionVector": [s.tolist() for s in final_solutions], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
    tqdm.write(f"📄 Checkpoint file updated: {filename} with {len(final_solutions)} solutions.")


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
                    perm1, perm2 = p1['solution'][:-1], p2['solution'][:-1]
                    child_perm = pmx_crossover(perm1, perm2) if random.random() < config['crossover_rate'] else perm1.copy()
                    if random.random() < mutation_rate:
                        a, b = np.random.choice(n, 2, replace=False); child_perm[a], child_perm[b] = child_perm[b], child_perm[a]
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

            if gen > 0 and gen % config['migration_interval'] == 0:
                all_individuals = sorted([ind for island in islands for ind in island], key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                migrants = [p['solution'] for p in all_individuals[:config['migration_size'] * num_islands]]
                if migrants:
                    for island in islands:
                        for j in range(config['migration_size']):
                            if j < len(island):
                                island[-(j+1)]['solution'] = random.choice(migrants)

            found_better = False
            for island in islands:
                current_best_score = sorted(island, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)[0]['score']
                if (current_best_score[0] > best_score[0]) or (current_best_score[0] == best_score[0] and current_best_score[1] < best_score[1]):
                    best_score = current_best_score
                    found_better = True
            
            if found_better:
                stagnation_counter = 0
                pbar.write(f"✨ Gen {gen+1}: New best score {best_score}")
                # Save checkpoint file whenever a new best score is found
                persist_checkpoint(problem_id, islands)
            else:
                stagnation_counter += 1
            pbar.set_postfix({"best_score": best_score, "stagn": stagnation_counter})
    
    # --- Final result collection and submission ---
    final_population = [p for island in islands for p in island]
    final_front = crowding_selection(final_population, 20)
    
    final_solutions = [p['solution'] for p in final_front]
    final_fitnesses_pygmo = [(p['score'][1], -p['score'][0]) for p in final_front] # Convert to (width, -size)
    
    print(f"\n🏆 Final best individual solution (size, width): {best_score}")
    ref_point = [n, 501]
    hv = pg.hypervolume(final_fitnesses_pygmo)
    final_hv = -hv.compute(ref_point)
    print(f"📈 Final Hypervolume: {final_hv:,.2f}")

    with open(f"submission_{problem_id}.json", "w") as f:
        json.dump({"decisionVector": [s.tolist() for s in final_solutions], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
    print(f"📄 Created final submission file.")

# --- Entry Point ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS: sys.exit("❌ Invalid problem ID.")
    
    config = CONFIG['general'].copy(); config.update(CONFIG[problem_id])
    
    start_time = time.time()
    n, adj = load_graph(problem_id)
    memetic_algorithm(n, adj, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")
