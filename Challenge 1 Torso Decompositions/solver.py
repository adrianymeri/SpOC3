import argparse, ctypes, json, math, os, time, sys, urllib.request
import numpy as np
import torch
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eigh

# --- CUDA Kernel Setup ---
if not os.path.exists("./evaluator.so"):
    print("❌ evaluator.so not found. Please compile it first by running: make compile")
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

# --- Helper Functions ---
def load_graph(graph_name, n_nodes):
    filepath = f"data/{graph_name}.gr"
    print(f"📥 Loading graph data for '{graph_name}' from '{filepath}'...")
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
    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    return features.T # Return as (E, N) for matrix multiplication

def calculate_hvi_and_select_front(widths, n_nodes):
    """Calculates hypervolume and returns the indices of the solutions for the submission file."""
    try:
        import pygmo as pg
    except ImportError:
        print("⚠️ Warning: pygmo not installed. Cannot calculate hypervolume. Returning best single solution.")
        best_t = np.argmin(widths)
        return np.array([best_t]), -1.0

    sizes = n_nodes - np.arange(n_nodes)
    fitnesses = np.vstack([widths, -sizes]).T
    
    non_dominated_indices = pg.non_dominated_front_2d(fitnesses)
    
    # Ensure we don't select more than 20 solutions
    if len(non_dominated_indices) > 20:
        # A simple truncation, more advanced selection could be used
        non_dominated_indices = non_dominated_indices[:20]

    front_fitnesses = fitnesses[non_dominated_indices]
    ref_point = [n_nodes + 1, 0]
    hv = pg.hypervolume(front_fitnesses)
    score = -hv.compute(ref_point)
    return non_dominated_indices, score

def create_submission(elites, node_features, selected_indices, graph_name):
    """Generates the decision vectors for the submission file."""
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
    parser.add_argument("--mutation_stdev", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_generations", type=int, default=20000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔩 Using device: {device}")
    if not torch.cuda.is_available():
        print("❌ FATAL: This solver requires a CUDA-enabled GPU.")
        sys.exit(1)
        
    B, N, E = args.batch_size, GRAPH_SIZES[args.graph], args.eigenvectors
    
    adj = load_graph(args.graph, N)
    node_features = torch.from_numpy(get_node_features(adj, E)).to(device)

    # --- GPU Tensor Allocation ---
    population = torch.randn((B, E), dtype=torch.float32, device=device) * 0.1
    elites = torch.zeros((N, E), dtype=torch.float32, device=device)
    elite_fitnesses = torch.full((N,), 501, dtype=torch.int32, device=device)
    
    # These tensors are passed to the CUDA kernel
    perms_gpu = torch.empty((B, N), dtype=torch.uint16, device=device)
    degrees_gpu = torch.empty((B, N), dtype=torch.uint16, device=device)
    fitnesses_gpu = torch.empty((B, N), dtype=torch.int32, device=device)
    adjs_batch_gpu = torch.from_numpy(np.tile(adj, (B, 1, 1))).to(device)
    
    pbar = tqdm(range(args.max_generations), desc="🚀 Evolving on GPU")
    for generation in pbar:
        logits = population @ node_features
        perms_gpu[:] = logits.argsort(axis=1).to(torch.uint16)

        # The CUDA kernel modifies its `adjs` input, so we must pass a fresh copy each time
        adj_copy = adjs_batch_gpu.clone()
        evaluate(
            ctypes.cast(adj_copy.data_ptr(), ctypes.POINTER(ctypes.c_bool)),
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
            selected_indices, hvi = calculate_hvi_and_select_front(elite_widths, N)
            best_t0_width = elite_widths[0]
            pbar.set_postfix({"best_width_t0": best_t0_width, "hvi": f"{hvi:,.0f}"})

        if generation > 0 and generation % args.checkpoint_every == 0:
            elite_widths = elite_fitnesses.cpu().numpy()
            selected_indices, hvi = calculate_hvi_and_select_front(elite_widths, N)
            
            submission = create_submission(elites, node_features, selected_indices, args.graph)
            with open(f"submissions/{args.graph}/gen_{generation}_{hvi:,.0f}.json", "w") as f:
                json.dump(submission, f)
            print(f"\n📄 Saved checkpoint submission for gen {generation}")

if __name__ == "__main__":
    main()
