#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX

# Declare C-level types for performance
ctypedef np.int64_t INT64_t
ctypedef np.uint64_t UINT64_t

# Global variables for the worker process
cdef int N
cdef UINT64_t[:] ADJ_BITS

def init_worker_cython(int n_val, np.ndarray[UINT64_t, ndim=1] adj_bits_val):
    global N, ADJ_BITS
    N = n_val
    ADJ_BITS = adj_bits_val

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

cdef void swap_cy(INT64_t[:] perm):
    cdef int n = perm.shape[0]
    cdef int i = rand() % n
    cdef int j = rand() % n
    cdef INT64_t temp = perm[i]
    perm[i] = perm[j]
    perm[j] = temp

cdef void inversion_cy(INT64_t[:] perm):
    cdef int n = perm.shape[0]
    cdef int a = rand() % n
    cdef int b = rand() % n
    cdef int start, end
    cdef INT64_t temp
    if a == b: return
    if a < b:
        start, end = a, b
    else:
        start, end = b, a
    while start < end:
        temp = perm[start]
        perm[start] = perm[end]
        perm[end] = temp
        start += 1
        end -= 1

cpdef np.ndarray[INT64_t, ndim=1] local_search_cy(np.ndarray[INT64_t, ndim=1] solution, int intensity):
    cdef np.ndarray[INT64_t, ndim=1] best_sol = solution.copy()
    cdef tuple best_score = evaluate_solution_cy(best_sol)
    cdef np.ndarray[INT64_t, ndim=1] cand_sol
    cdef tuple cand_score
    cdef int shift, t

    cdef int i
    for i in range(intensity):
        cand_sol = best_sol.copy()
        # Choose a neighborhood
        if rand() < 0.7 * RAND_MAX: # Permutation neighborhood
            if rand() < 0.5 * RAND_MAX:
                inversion_cy(cand_sol[:-1])
            else:
                swap_cy(cand_sol[:-1])
        else: # Torso shift neighborhood
            t = cand_sol[-1]
            shift = max(1, N // 20)
            t = max(0, min(N - 1, t + (rand() % (2 * shift + 1)) - shift))
            cand_sol[-1] = t

        cand_score = evaluate_solution_cy(cand_sol)

        # Dominance check
        if (cand_score[0] > best_score[0]) or \
           (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
            best_sol = cand_sol
            best_score = cand_score
            
    return best_sol
