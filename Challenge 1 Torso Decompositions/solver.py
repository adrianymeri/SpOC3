import argparse, ctypes, json, math, os, time, sys, urllib.request
import numpy as np
import torch
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eigh
from tqdm import tqdm

# --- CUDA Kernel Setup ---
if not os.path.exists("./evaluator.so"):
    print("❌ evaluator.so not found. Please compile it first by running: make")
    sys.exit(1)
libeval = ctypes.CDLL('./evaluator.so')
evaluate = libeval.evaluate
evaluate.argtypes = [
    ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t, ctypes.c_size_t
]

# --- Graph & Problem Info ---
GRAPH_SIZES = {
    "small-graph": 1357, "medium-graph": 4941, "large-graph": 15102,
}
PROBLEM_URLS = {
    "small-graph": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium-graph": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "large-graph": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Helper Functions ---
def load_graph(graph_name, n_nodes):
    filepath = f"data/{graph_name}.gr"
    if not os.path.exists(filepath):
        print(f"📥 Downloading graph data for '{graph_name}'...")
        urllib.request.urlretrieve(PROBLEM_URLS[graph_name], filepath)
        print("✅ Download complete.")

    print(f"Loading graph data from '{filepath}'...")
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith('#'): continue
            u, v = map(int, line.strip().split())
            adj[u, v] = adj[v, u] = True
    print(f"✅ Loaded graph with {n_nodes} nodes.")
    return adj

def get_node_features(adj, num_eigenvectors):
    print("🔬 Calculating node features (Laplacian eigenvectors)...")
    L = laplacian(adj.astype(np.float32), normed=True)
    eig_vals, eig_vecs = eigh(L)
    features = np.real(eig_vecs[:, 1:num_eigenvectors+1]).astype(np.float32)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    return features.T

def calculate_hvi_and_select_front(widths, n_nodes):
    try:
        import pygmo as pg
    except ImportError:
        print("⚠️ Pygmo not installed, cannot calculate HVI.")
        return np.array([np.argmin(widths)]), -1.0

    sizes = n_nodes - np.arange(n_nodes)
    fitnesses = np.vstack([widths, -sizes]).T
    
    non_dominated_indices = pg.non_dominated_front_2d(fitnesses)
    
    if len(non_dominated_indices) > 20:
        step = len(non_dominated_indices) // 20
        non_dominated_indices = non_dominated_indices[::step][:20]

    front_fitnesses = fitnesses[non_dominated_indices]
    ref_point = [n_nodes + 1, 0] 
    hv = pg.hypervolume(front_fitnesses)
    score = -hv.compute(ref_point)
    return non_dominated_indices, score

def create_submission(elites, node_features, selected_indices, graph_name):
    ts = torch.from_numpy(selected_indices).long().to(elites.device)
    selected_elites = elites[ts]
    logits = selected_elites @ node_features
    perms = logits.argsort(axis=1)
    return {
        "decisionVector": [perm.tolist() + [t.item()] for perm, t in zip(perms.cpu(), ts.cpu())],
        "problem": graph_name, "challenge": "spoc-3-torso-decompositions",
    }

# --- Main Neuroevolutionary Algorithm ---
def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Neuroevolutionary Solver for SPOC3")
    parser.add_argument("--graph", choices=GRAPH_SIZES.keys(), default="small-graph")
    parser.add_argument("--eigenvectors", type=int, default=32)
    parser.add_argument("--mutation_stdev", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_generations", type=int, default=30000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔩 Using device: {device}")
    if not torch.cuda.is_available(): sys.exit("❌ FATAL: This solver requires a CUDA-enabled GPU.")
        
    B, N, E = args.batch_size, GRAPH_SIZES[args.graph], args.eigenvectors
    
    adj = load_graph(args.graph, N)
    node_features = torch.from_numpy(get_node_features(adj, E)).to(device)

    population = torch.randn((B, E), dtype=torch.float32, device=device) * 0.01
    elites = torch.zeros((N, E), dtype=torch.float32, device=device)
    elite_fitnesses = torch.full((N,), 501, dtype=torch.int32, device=device)
    
    perms_gpu = torch.empty((B, N), dtype=torch.uint16, device=device)
    degrees_gpu = torch.empty((B, N), dtype=torch.uint16, device=device)
    fitnesses_gpu = torch.empty((B, N), dtype=torch.int32, device=device)
    
    adj_for_gpu = torch.from_numpy(adj).to(device)

    pbar = tqdm(range(args.max_generations), desc="🚀 Evolving on GPU")
    for generation in pbar:
        logits = population @ node_features
        perms_gpu[:] = logits.argsort(axis=1).to(torch.uint16)

        adj_batch_gpu = adj_for_gpu.unsqueeze(0).expand(B, N, N).clone()
        
        evaluate(
            ctypes.cast(adj_batch_gpu.data_ptr(), ctypes.POINTER(ctypes.c_bool)),
            ctypes.cast(perms_gpu.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            ctypes.cast(degrees_gpu.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            ctypes.cast(fitnesses_gpu.data_ptr(), ctypes.POINTER(ctypes.c_int)),
            B, N
        )
        
        best_widths_in_batch, best_indices_in_batch = fitnesses_gpu.min(axis=0)
        
        is_better = best_widths_in_batch < elite_fitnesses
        if torch.any(is_better):
            elite_fitnesses[is_better] = best_widths_in_batch[is_better]
            elite_indices_to_update = best_indices_in_batch[is_better]
            elites[is_better] = population[elite_indices_to_update]

        parent_indices = torch.randint(0, N, (B,), device=device)
        population[:] = elites[parent_indices]
        population += torch.randn_like(population) * args.mutation_stdev

        if generation % 10 == 0:
            elite_widths = elite_fitnesses.cpu().numpy()
            _, hvi = calculate_hvi_and_select_front(elite_widths, N)
            best_t0_width = elite_widths[0]
            pbar.set_postfix({"best_width_t=0": best_t0_width, "hvi": f"{hvi:,.0f}"})

        if generation > 0 and generation % args.checkpoint_every == 0:
            elite_widths = elite_fitnesses.cpu().numpy()
            selected_indices, hvi = calculate_hvi_and_select_front(elite_widths, N)
            
            submission = create_submission(elites, node_features, selected_indices, args.graph)
            with open(f"submissions/{args.graph}/gen_{generation}_{hvi:,.0f}.json", "w") as f:
                json.dump(submission, f, indent=4)
            print(f"\n📄 Saved checkpoint gen {generation}. HV: {hvi:,.2f}")

if __name__ == "__main__":
    main()
