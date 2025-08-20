#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX

# --- C-level Type Definitions ---
ctypedef np.int64_t INT64_t
ctypedef np.uint64_t UINT64_t

# --- Worker Globals (set by Python) ---
cdef int N
cdef UINT64_t[:] ADJ_BITS

def init_worker_cython(int n_val, np.ndarray[UINT64_t, ndim=1] adj_bits_val):
    global N, ADJ_BITS
    N = n_val
    ADJ_BITS = adj_bits_val

# --- Core C-level Functions ---
cdef int bitcount_cy(UINT64_t x):
    cdef int c = 0
    while x > 0:
        x &= x - 1
        c += 1
    return c

cpdef tuple evaluate_solution_cy(np.ndarray[INT64_t, ndim=1] solution):
    cdef int t = solution[-1]
    cdef INT64_t[:] perm = solution[:-1]
    cdef int size = N - t
    if size <= 0: return (0, 999)

    cdef UINT64_t[:] suffix_mask = np.zeros(N, dtype=np.uint64)
    cdef UINT64_t curr_mask = 0
    cdef int i, u, v, out_deg, max_width
    cdef UINT64_t succ, s, v_bit, temp_v_bit
    
    for i in range(N - 1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (1 << perm[i])

    cdef UINT64_t[:] temp = ADJ_BITS.copy()
    max_width = 0
    
    for i in range(N):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount_cy(succ)
        if out_deg > max_width: max_width = out_deg
        if succ == 0: continue
        
        s = succ
        while s > 0:
            v_bit = s & -s
            s ^= v_bit
            v = 0
            temp_v_bit = v_bit
            while temp_v_bit > 1:
                temp_v_bit >>= 1
                v += 1
            temp[v] |= (succ ^ v_bit)
            
    return (size, max_width)

# --- C-level Local Search Operators ---
cdef np.ndarray[INT64_t, ndim=1] inversion_cy(np.ndarray[INT64_t, ndim=1] perm_arr):
    # This function now returns a new array to avoid side effects
    cdef INT64_t[:] perm = perm_arr.copy() # Work on a copy
    cdef int n = perm.shape[0], a = rand() % n, b = rand() % n
    cdef int start, end
    if a == b: return np.asarray(perm) # Return the copied array
    start, end = (a, b) if a < b else (b, a)
    while start < end:
        perm[start], perm[end] = perm[end], perm[start]
        start += 1
        end -= 1
    return np.asarray(perm)

# ==============================================================================
# MEMORY-SAFE BLOCK MOVE
# This new version is not "in-place". It creates and returns a new, correct array,
# which avoids memory corruption bugs. It uses NumPy's C-API for safety and speed.
# ==============================================================================
cdef np.ndarray[INT64_t, ndim=1] block_move_cy(np.ndarray[INT64_t, ndim=1] perm):
    cdef int n = perm.shape[0]
    cdef int block_size = 2 + (rand() % max(1, n // 50))
    if n <= block_size: return perm.copy()
    
    cdef int start_idx = rand() % (n - block_size)
    cdef int insert_pos = rand() % (n - block_size)

    # Safely extract the parts using slicing
    cdef np.ndarray[INT64_t, ndim=1] block = perm[start_idx : start_idx + block_size]
    cdef np.ndarray[INT64_t, ndim=1] perm_without_block = np.delete(perm, np.s_[start_idx : start_idx + block_size])
    
    # Safely insert the block into the new position and return the new array
    return np.insert(perm_without_block, insert_pos, block)

# --- Main Local Search Function (Memetic Step) ---
cpdef np.ndarray[INT64_t, ndim=1] local_search_cy(np.ndarray[INT64_t, ndim=1] solution, int intensity):
    cdef np.ndarray[INT64_t, ndim=1] best_sol = solution.copy()
    cdef tuple best_score = evaluate_solution_cy(best_sol)
    cdef np.ndarray[INT64_t, ndim=1] cand_sol, cand_perm
    cdef tuple cand_score
    cdef int shift, t
    cdef double r

    cdef int i
    for i in range(intensity):
        r = <double>rand() / RAND_MAX
        
        if r < 0.4:
            # Apply block move and get a new permutation array
            cand_perm = block_move_cy(best_sol[:-1])
            cand_sol = np.append(cand_perm, best_sol[-1])
        elif r < 0.8:
            # Apply inversion and get a new permutation array
            cand_perm = inversion_cy(best_sol[:-1])
            cand_sol = np.append(cand_perm, best_sol[-1])
        else:
            # Apply torso shift (this is already safe)
            cand_sol = best_sol.copy()
            t = cand_sol[-1]
            shift = max(1, N // 20)
            t = max(0, min(N - 1, t + (rand() % (2 * shift + 1)) - shift))
            cand_sol[-1] = t

        cand_score = evaluate_solution_cy(cand_sol)

        if (cand_score[0] > best_score[0]) or \
           (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
            best_sol = cand_sol
            best_score = cand_score
            
    return best_sol
