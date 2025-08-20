# test.py
import numpy as np
import numba
from typing import Tuple # <-- ADD THIS LINE

# These globals are needed for the function signature
WORKER_N = 100
WORKER_ADJ_BITS = np.zeros(WORKER_N, dtype=np.uint64)

@numba.jit(nopython=True)
def bitcount_numba(x: np.uint64) -> int:
    c = 0
    while x > 0:
        x &= x - 1
        c += 1
    return c

@numba.jit(nopython=True)
def evaluate_solution_numba(solution: np.ndarray) -> Tuple[int, int]:
    n = WORKER_N
    adj_bits = WORKER_ADJ_BITS
    t = solution[-1]
    perm = solution[:-1]
    size = n - t
    if size <= 0:
        return (0, 999)

    suffix_mask = np.zeros(n, dtype=np.uint64)
    curr_mask = np.uint64(0)
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (np.uint64(1) << perm[i])

    temp = adj_bits.copy()
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        # This is where the error occurs in the big script
        out_deg = bitcount_numba(succ)
        if out_deg > max_width:
            max_width = out_deg
        
        if succ == 0:
            continue
        
        s = succ
        while s > 0:
            v_bit = s & -s
            s ^= v_bit
            
            # The critical integer-only fix
            v = 0
            temp_v_bit = v_bit
            while temp_v_bit > 1:
                temp_v_bit >>= 1
                v += 1
            
            temp[v] |= (succ ^ v_bit)
            
    return (size, max_width)

if __name__ == '__main__':
    print("🚀 Compiling minimal test function...")
    # Create dummy data with the correct types
    dummy_perm = np.arange(WORKER_N, dtype=np.int64)
    dummy_solution = np.append(dummy_perm, WORKER_N // 2)
    
    # Call the function to trigger compilation
    result = evaluate_solution_numba(dummy_solution)
    
    print(f"✅ Compilation successful. Result: {result}")
