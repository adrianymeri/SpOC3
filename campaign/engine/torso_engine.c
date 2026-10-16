/*
 * torso_engine.c -- the shared engine of the OLA 2027 campaign.
 *
 * Every CPU method in the campaign (ours and the re-run baselines) evaluates
 * through this file, so evaluation speed is equal for everyone.
 *
 * The semantics are those of esa_eval.py (ESA's graph_torso_udp):
 *
 *   decision vector  = permutation perm[0..n-1] + threshold t
 *   step degree      d[i] = number of not-yet-eliminated neighbours of
 *                    perm[i] in the fill-in graph when perm[i] is eliminated
 *   width(t)         = max_{i >= t} d[i]
 *   cap              any d[i] > 500, at ANY step (head included), voids the
 *                    vector
 *   fitness          (width, t), both minimised
 *   score            -HV of at most 20 points, reference point (n, n)
 *
 * One elimination pass gives every d[i], hence the whole staircase of
 * (width, t) points a permutation offers (suffix maxima).
 *
 * Contents
 *   rng        xoshiro256**, state owned by the caller (checkpointable)
 *   graph      .gr loader, CSR lists, adjacency bitsets
 *   state      the fill-in elimination state (work rows + alive set)
 *   eval       full evaluation, staircase
 *   archive    best t per width with the permutation behind it, Pareto
 *              filter and the exact HSSP-20 dynamic program
 *   greedy     min-degree constructions, head-set-first decoding
 *   inc        windowed incremental re-evaluation (snapshots)
 *   hri        Team HRI's destroy / repair operators
 *   policy     the adaptive neural decoder of the NE methods
 *
 * Build: see Makefile (cc -O3 -march=native -shared -fPIC).
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TW 500

int64_t te_hssp(const int32_t *xs, const int32_t *ys, int m, int n, int k, int32_t *out_idx, int *out_cnt);

/* ------------------------------------------------------------------ rng */

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

/* xoshiro256** (Blackman & Vigna). s[4] is owned by the caller. */
static inline uint64_t rng_next(uint64_t *s) {
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl64(s[3], 45);
    return result;
}

/* uniform integer in [0, m) without modulo bias (Lemire) */
static inline uint32_t rng_below(uint64_t *s, uint32_t m) {
    if (m <= 1) return 0;
    uint64_t x = rng_next(s) >> 32;
    uint64_t prod = x * (uint64_t)m;
    uint32_t low = (uint32_t)prod;
    if (low < m) {
        uint32_t thresh = (uint32_t)(-m) % m;
        while (low < thresh) {
            x = rng_next(s) >> 32;
            prod = x * (uint64_t)m;
            low = (uint32_t)prod;
        }
    }
    return (uint32_t)(prod >> 32);
}

static inline double rng_unit(uint64_t *s) { return (rng_next(s) >> 11) * (1.0 / 9007199254740992.0); }

void te_rng_seed(uint64_t *s, uint64_t seed) {
    /* splitmix64 expansion */
    for (int i = 0; i < 4; i++) {
        uint64_t z = (seed += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s[i] = z ^ (z >> 31);
    }
}

uint32_t te_rng_below(uint64_t *s, uint32_t m) { return rng_below(s, m); }
double te_rng_unit(uint64_t *s) { return rng_unit(s); }

void te_shuffle(int32_t *a, int len, uint64_t *s) {
    for (int i = len - 1; i > 0; i--) {
        int j = (int)rng_below(s, (uint32_t)(i + 1));
        int32_t x = a[i]; a[i] = a[j]; a[j] = x;
    }
}

/* ---------------------------------------------------------------- graph */

typedef struct {
    int n;              /* vertices 0..n-1 */
    int W;              /* 64-bit words per bitset row */
    int64_t m;          /* edges */
    uint64_t *adj0;     /* n*W original adjacency bitsets */
    int32_t *off;       /* CSR offsets, n+1 */
    int32_t *nbr;       /* CSR neighbours, sorted ascending */
    int32_t *deg0;      /* original degrees */
    /* scratch owned by the graph (one thread per process) */
    uint64_t *work;     /* n*W */
    uint64_t *alive;    /* W */
    uint64_t *nb;       /* W */
    int32_t *tmp_a, *tmp_b, *tmp_c; /* n each */
} G;

static void *xmalloc(size_t sz) {
    void *p = malloc(sz ? sz : 1);
    if (!p) { fprintf(stderr, "torso_engine: out of memory (%zu bytes)\n", sz); exit(2); }
    return p;
}
static void *xcalloc(size_t cnt, size_t sz) {
    void *p = calloc(cnt ? cnt : 1, sz ? sz : 1);
    if (!p) { fprintf(stderr, "torso_engine: out of memory\n"); exit(2); }
    return p;
}

static int cmp_i32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* Load a .gr edge list: one "u v" pair per line, 0-based. Lines starting with
 * p, c, # or % are skipped. n = max id + 1. Self-loops and duplicate edges are
 * ignored, as torso.Graph.load does. Returns NULL on error. */
G *te_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    size_t cap = 1 << 16, m = 0;
    int32_t *E = xmalloc(sizeof(int32_t) * 2 * cap);
    char line[512];
    int maxid = -1;
    while (fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == 'p' || *p == 'c' || *p == '#' || *p == '%' || *p == '\n' || *p == '\r' || *p == 0) continue;
        long a, b;
        if (sscanf(p, "%ld %ld", &a, &b) != 2) continue;
        if (a < 0 || b < 0) { fclose(f); free(E); return NULL; }
        if (m == cap) { cap *= 2; E = realloc(E, sizeof(int32_t) * 2 * cap); if (!E) exit(2); }
        E[2 * m] = (int32_t)a; E[2 * m + 1] = (int32_t)b; m++;
        if (a > maxid) maxid = (int)a;
        if (b > maxid) maxid = (int)b;
    }
    fclose(f);
    G *g = xcalloc(1, sizeof(G));
    int n = maxid + 1;
    g->n = n;
    g->W = (n + 63) / 64;
    int W = g->W;
    g->adj0 = xcalloc((size_t)n * W, 8);
    for (size_t i = 0; i < m; i++) {
        int a = E[2 * i], b = E[2 * i + 1];
        if (a == b) continue;
        g->adj0[(size_t)a * W + b / 64] |= 1ULL << (b % 64);
        g->adj0[(size_t)b * W + a / 64] |= 1ULL << (a % 64);
    }
    free(E);
    g->deg0 = xmalloc(sizeof(int32_t) * n);
    g->off = xmalloc(sizeof(int32_t) * (n + 1));
    int64_t tot = 0;
    for (int v = 0; v < n; v++) {
        int d = 0;
        for (int w = 0; w < W; w++) d += __builtin_popcountll(g->adj0[(size_t)v * W + w]);
        g->deg0[v] = d;
        g->off[v] = (int32_t)tot;
        tot += d;
    }
    g->off[n] = (int32_t)tot;
    g->m = tot / 2;
    g->nbr = xmalloc(sizeof(int32_t) * (tot ? tot : 1));
    for (int v = 0; v < n; v++) {
        int k = g->off[v];
        for (int w = 0; w < W; w++) {
            uint64_t x = g->adj0[(size_t)v * W + w];
            while (x) { int b = __builtin_ctzll(x); x &= x - 1; g->nbr[k++] = w * 64 + b; }
        }
    }
    g->work = xmalloc((size_t)n * W * 8);
    g->alive = xmalloc((size_t)W * 8);
    g->nb = xmalloc((size_t)W * 8);
    g->tmp_a = xmalloc(sizeof(int32_t) * (n + 1));
    g->tmp_b = xmalloc(sizeof(int32_t) * (n + 1));
    g->tmp_c = xmalloc(sizeof(int32_t) * (n + 1));
    return g;
}

void te_free(G *g) {
    if (!g) return;
    free(g->adj0); free(g->off); free(g->nbr); free(g->deg0);
    free(g->work); free(g->alive); free(g->nb);
    free(g->tmp_a); free(g->tmp_b); free(g->tmp_c);
    free(g);
}

int te_n(const G *g) { return g->n; }
int64_t te_m(const G *g) { return g->m; }
int te_words(const G *g) { return g->W; }
int te_deg0(const G *g, int v) { return g->deg0[v]; }
/* copy CSR into caller buffers (off: n+1, nbr: 2m) */
void te_csr(const G *g, int32_t *off, int32_t *nbr) {
    memcpy(off, g->off, sizeof(int32_t) * (g->n + 1));
    memcpy(nbr, g->nbr, sizeof(int32_t) * g->off[g->n]);
}

/* -------------------------------------------------------- perm checking */

/* 1 if perm is a permutation of 0..n-1, else 0 */
int te_is_perm(const G *g, const int32_t *perm) {
    int n = g->n;
    uint8_t *seen = (uint8_t *)g->tmp_a; /* n bytes fit in n int32 */
    memset(seen, 0, n);
    for (int i = 0; i < n; i++) {
        int v = perm[i];
        if (v < 0 || v >= n || seen[v]) return 0;
        seen[v] = 1;
    }
    return 1;
}

/* ---------------------------------------------------------------- state */

/* The elimination state: work rows (the fill-in graph, rows may carry stale
 * bits of eliminated vertices -- every read is masked with alive) and the
 * alive set. */
typedef struct {
    int n, W, step;
    uint64_t *work;   /* n*W */
    uint64_t *alive;  /* W */
} S;

S *te_state_new(const G *g) {
    S *s = xcalloc(1, sizeof(S));
    s->n = g->n; s->W = g->W;
    s->work = xmalloc((size_t)g->n * g->W * 8);
    s->alive = xmalloc((size_t)g->W * 8);
    return s;
}
void te_state_free(S *s) { if (s) { free(s->work); free(s->alive); free(s); } }

static void alive_all(uint64_t *alive, int n, int W) {
    for (int w = 0; w < W; w++) alive[w] = ~0ULL;
    int r = n % 64;
    if (r) alive[W - 1] = (1ULL << r) - 1;
}

void te_state_reset(const G *g, S *s) {
    memcpy(s->work, g->adj0, (size_t)g->n * g->W * 8);
    alive_all(s->alive, g->n, g->W);
    s->step = 0;
}
void te_state_copy(S *dst, const S *src) {
    memcpy(dst->work, src->work, (size_t)src->n * src->W * 8);
    memcpy(dst->alive, src->alive, (size_t)src->W * 8);
    dst->step = src->step;
}

/* Eliminate vertex u from state s; returns its degree. nb receives the
 * neighbourhood bitset (only words [lo, hi] are meaningful). */
static inline int eliminate(S *s, int u, uint64_t *nb, int *plo, int *phi) {
    int W = s->W;
    s->alive[u >> 6] &= ~(1ULL << (u & 63));
    const uint64_t *row = s->work + (size_t)u * W;
    int d = 0, lo = W, hi = -1;
    for (int w = 0; w < W; w++) {
        uint64_t x = row[w] & s->alive[w];
        nb[w] = x;
        if (x) { d += __builtin_popcountll(x); if (lo == W) lo = w; hi = w; }
    }
    if (d > 1) {
        for (int w = lo; w <= hi; w++) {
            uint64_t x = nb[w];
            while (x) {
                int b = __builtin_ctzll(x); x &= x - 1;
                uint64_t *rv = s->work + (size_t)(w * 64 + b) * W;
                for (int k = lo; k <= hi; k++) rv[k] |= nb[k];
            }
        }
    }
    s->step++;
    *plo = lo; *phi = hi;
    return d;
}

/* current degree of alive vertex v in state s */
static inline int cur_degree(const S *s, int v) {
    const uint64_t *row = s->work + (size_t)v * s->W;
    int d = 0;
    for (int w = 0; w < s->W; w++) d += __builtin_popcountll(row[w] & s->alive[w]);
    /* row may carry v's own bit from earlier fills; v is alive, so drop it */
    if (row[v >> 6] & (1ULL << (v & 63))) d--;
    return d;
}

/* Advance state s by eliminating seq[0..len-1] (all must be alive); writes
 * their degrees to deg (may be NULL). If abort_cap, stops at the first degree
 * above MAX_TW and returns its index in seq; returns -1 when no step breaks
 * the cap (with abort_cap = 0 the walk always completes and the return value
 * is the first offending index, or -1). */
int te_state_advance(const G *g, S *s, const int32_t *seq, int len, int32_t *deg, int abort_cap) {
    int first = -1, lo, hi;
    for (int i = 0; i < len; i++) {
        int d = eliminate(s, seq[i], g->nb, &lo, &hi);
        if (deg) deg[i] = d;
        if (d > MAX_TW && first < 0) {
            first = i;
            if (abort_cap) return first;
        }
    }
    return first;
}

/* ----------------------------------------------------------------- eval */

static S *g_scratch_state = NULL;   /* reused by te_eval */
static const G *g_scratch_owner = NULL;

static S *scratch_state(const G *g) {
    if (g_scratch_owner != g || !g_scratch_state || g_scratch_state->n != g->n) {
        if (g_scratch_state) te_state_free(g_scratch_state);
        g_scratch_state = te_state_new(g);
        g_scratch_owner = g;
    }
    return g_scratch_state;
}

/* Every step degree of perm in one pass. Returns the index of the first step
 * whose degree exceeds MAX_TW, or -1 if the vector respects the cap.
 * With abort_cap the walk stops there (later deg entries are left as 0).
 * perm must be a valid permutation (see te_is_perm). */
int te_eval(const G *g, const int32_t *perm, int32_t *deg, int abort_cap) {
    S *s = scratch_state(g);
    te_state_reset(g, s);
    if (abort_cap) memset(deg, 0, sizeof(int32_t) * g->n);
    return te_state_advance(g, s, perm, g->n, deg, abort_cap);
}

/* The official fitness (width, t) of (perm, t): width = max_{i>=t} deg[i],
 * or MAX_TW+1 if any step breaks the cap. Mirrors esa_eval.evaluate.
 * Returns width; t is given. Invalid permutations give MAX_TW+1. */
int te_fitness(const G *g, const int32_t *perm, int t) {
    if (!te_is_perm(g, perm)) return MAX_TW + 1;
    if (g->n - t <= 0) return 9999;
    int32_t *deg = g->tmp_b;
    int bad = te_eval(g, perm, deg, 1);
    if (bad >= 0) return MAX_TW + 1;
    int w = 0;
    for (int i = t; i < g->n; i++) if (deg[i] > w) w = deg[i];
    return w;
}

/* Staircase of a degree vector, exactly as torso.Solution.staircase():
 * for every distinct suffix maximum w, the smallest t with max(deg[t:]) == w.
 * Points come out with w ascending (t descending). Returns the number of
 * points, or 0 if any degree exceeds MAX_TW. */
int te_staircase(const int32_t *deg, int n, int32_t *out_w, int32_t *out_t) {
    for (int i = 0; i < n; i++) if (deg[i] > MAX_TW) return 0;
    int k = 0, run = 0;
    for (int t = n - 1; t >= 0; t--) {
        if (deg[t] > run) run = deg[t];
        if (k > 0 && out_w[k - 1] == run) out_t[k - 1] = t;
        else { out_w[k] = run; out_t[k] = t; k++; }
    }
    return k;
}

/* Smallest t with max(deg[t:]) <= target, or n if unreachable / capped. */
/* HV of a permutation's own staircase cut to its best k points. */
int64_t te_stair_hv(const int32_t *deg, int n, int k) {
    int32_t *w = malloc(sizeof(int32_t) * n), *t = malloc(sizeof(int32_t) * n);
    if (!w || !t) exit(2);
    int m = te_staircase(deg, n, w, t);
    int64_t hv = m ? te_hssp(w, t, m, n, k, NULL, NULL) : 0;
    free(w); free(t);
    return hv;
}

int te_best_t_for_width(const int32_t *deg, int n, int target) {
    for (int i = 0; i < n; i++) if (deg[i] > MAX_TW) return n;
    int run = 0;
    for (int t = n - 1; t >= 0; t--) {
        if (deg[t] > run) run = deg[t];
        if (run > target) return (t + 1 < n) ? t + 1 : n;
    }
    return 0;
}

/* --------------------------------------------------------------- HSSP */

/* Exact hypervolume subset selection for a 2-D front (both minimised,
 * reference (n, n)). xs ascending, ys strictly descending, m points, all
 * x < n and y < n. Chooses min(k, m) points maximising the dominated area.
 * Writes the chosen indices (ascending) to out_idx, returns the area.
 * The O(k m^2) dynamic program of esa_eval.ParetoArchive.top_k_by_hv_contribution:
 *   f(i, 1) = (n - x_i)(n - y_i)
 *   f(i, j) = max_{i' > i} (x_i' - x_i)(n - y_i) + f(i', j-1)                */
int64_t te_hssp(const int32_t *xs, const int32_t *ys, int m, int n, int k, int32_t *out_idx, int *out_cnt) {
    if (m <= 0) { if (out_cnt) *out_cnt = 0; return 0; }
    if (m <= k) {
        int64_t hv = 0;
        for (int i = 0; i < m; i++) {
            int64_t nx = (i + 1 < m) ? xs[i + 1] : n;
            hv += (nx - xs[i]) * (int64_t)(n - ys[i]);
            if (out_idx) out_idx[i] = i;
        }
        if (out_cnt) *out_cnt = m;
        return hv;
    }
    int64_t *f = xmalloc(sizeof(int64_t) * (size_t)m * (k + 1));
    int32_t *nx = xmalloc(sizeof(int32_t) * (size_t)m * (k + 1));
#define F(j, i) f[(size_t)(j) * m + (i)]
#define NX(j, i) nx[(size_t)(j) * m + (i)]
    for (int i = 0; i < m; i++) { F(1, i) = (int64_t)(n - xs[i]) * (n - ys[i]); NX(1, i) = -1; }
    for (int j = 2; j <= k; j++) {
        for (int i = 0; i < m; i++) { F(j, i) = -1; NX(j, i) = -1; }
        for (int i = 0; i + j - 1 < m; i++) {
            int64_t strip = n - ys[i], best = -1; int bi = -1;
            for (int ip = i + 1; ip < m; ip++) {
                int64_t prev = F(j - 1, ip);
                if (prev < 0) continue;
                int64_t val = (int64_t)(xs[ip] - xs[i]) * strip + prev;
                if (val > best) { best = val; bi = ip; }
            }
            F(j, i) = best; NX(j, i) = bi;
        }
    }
    int64_t best = -1; int start = -1;
    for (int i = 0; i < m; i++) if (F(k, i) > best) { best = F(k, i); start = i; }
    int cnt = 0;
    for (int j = k, i = start; j >= 1 && i >= 0; j--) { if (out_idx) out_idx[cnt] = i; cnt++; i = NX(j, i); }
    if (out_cnt) *out_cnt = cnt;
#undef F
#undef NX
    free(f); free(nx);
    return best;
}

/* ------------------------------------------------------------- archive */

/* Best t per width (0..MAX_TW) and the permutation behind it, as torso.Front.
 * Permutations live in a refcounted pool: one evaluated permutation that
 * improves many widths is stored once. */
typedef struct {
    int n;
    int32_t best_t[MAX_TW + 1];   /* n = empty */
    int32_t slot[MAX_TW + 1];     /* pool slot of the permutation, -1 empty */
    int64_t stamp[MAX_TW + 1];    /* evaluation counter when set */
    int nslots, cap;
    int32_t *pool;                /* cap * n */
    int32_t *refc;                /* cap */
    int32_t *freelist; int nfree;
    int dirty;
    int64_t hv_cache; int k_cache;
    int32_t sel_w[64], sel_t[64]; int nsel;
    int64_t adds;                 /* staircases offered */
    int64_t improvements;         /* (width) improvements accepted */
} A;

A *te_arch_new(int n) {
    A *a = xcalloc(1, sizeof(A));
    a->n = n;
    for (int w = 0; w <= MAX_TW; w++) { a->best_t[w] = n; a->slot[w] = -1; a->stamp[w] = 0; }
    a->cap = 8;
    a->pool = xmalloc(sizeof(int32_t) * (size_t)a->cap * n);
    a->refc = xcalloc(a->cap, sizeof(int32_t));
    a->freelist = xmalloc(sizeof(int32_t) * a->cap);
    a->nfree = 0; a->nslots = 0;
    a->dirty = 1; a->k_cache = -1;
    return a;
}
void te_arch_free(A *a) { if (a) { free(a->pool); free(a->refc); free(a->freelist); free(a); } }

static int arch_alloc_slot(A *a) {
    if (a->nfree > 0) return a->freelist[--a->nfree];
    if (a->nslots == a->cap) {
        a->cap *= 2;
        a->pool = realloc(a->pool, sizeof(int32_t) * (size_t)a->cap * a->n);
        a->refc = realloc(a->refc, sizeof(int32_t) * a->cap);
        a->freelist = realloc(a->freelist, sizeof(int32_t) * a->cap);
        if (!a->pool || !a->refc || !a->freelist) exit(2);
    }
    a->refc[a->nslots] = 0;
    return a->nslots++;
}
static void arch_release(A *a, int sl) {
    if (sl < 0) return;
    if (--a->refc[sl] == 0) a->freelist[a->nfree++] = sl;
}

/* Offer one evaluated permutation's staircase. Points with width > MAX_TW,
 * width >= n or t >= n are ignored (torso.Front.add). Returns the number of
 * widths whose best t improved. */
int te_arch_add(A *a, const int32_t *perm, const int32_t *ws, const int32_t *ts, int m, int64_t stamp) {
    a->adds++;
    int improved = 0, sl = -1;
    for (int i = 0; i < m; i++) {
        int w = ws[i], t = ts[i];
        if (w < 0 || w > MAX_TW || w >= a->n || t >= a->n) continue;
        if (t < a->best_t[w]) {
            if (sl < 0) {
                sl = arch_alloc_slot(a);
                memcpy(a->pool + (size_t)sl * a->n, perm, sizeof(int32_t) * a->n);
            }
            arch_release(a, a->slot[w]);
            a->slot[w] = sl; a->refc[sl]++;
            a->best_t[w] = t; a->stamp[w] = stamp;
            improved++;
        }
    }
    if (improved) { a->dirty = 1; a->improvements += improved; }
    return improved;
}

/* Evaluate perm and offer its staircase. Returns improvements; deg (n) and
 * the staircase (sw, st: up to n entries) are written for the caller. */
int te_arch_eval_add(const G *g, A *a, const int32_t *perm, int32_t *deg, int32_t *sw, int32_t *st, int *npts, int64_t stamp) {
    int bad = te_eval(g, perm, deg, 1);
    if (bad >= 0) { *npts = 0; return 0; }
    *npts = te_staircase(deg, g->n, sw, st);
    return te_arch_add(a, perm, sw, st, *npts, stamp);
}

/* Pareto front of the archive: widths ascending, t strictly descending. */
int te_arch_pareto(const A *a, int32_t *pw, int32_t *pt) {
    int m = 0, best = a->n;
    for (int w = 0; w <= MAX_TW; w++) {
        int t = a->best_t[w];
        if (t < best && w < a->n && t < a->n) { pw[m] = w; pt[m] = t; m++; best = t; }
    }
    return m;
}

/* HV of the best k-subset (exact HSSP); cached until the archive changes. */
int64_t te_arch_hv(A *a, int k) {
    if (!a->dirty && a->k_cache == k) return a->hv_cache;
    int32_t pw[MAX_TW + 1], pt[MAX_TW + 1], idx[64];
    int m = te_arch_pareto(a, pw, pt), cnt = 0;
    if (k > 64) k = 64;
    int64_t hv = te_hssp(pw, pt, m, a->n, k, idx, &cnt);
    /* chosen indices come out in ascending order from the DP walk */
    int32_t tmp[64];
    for (int i = 0; i < cnt; i++) tmp[i] = idx[i];
    qsort(tmp, cnt, sizeof(int32_t), cmp_i32);
    a->nsel = cnt;
    for (int i = 0; i < cnt; i++) { a->sel_w[i] = pw[tmp[i]]; a->sel_t[i] = pt[tmp[i]]; }
    a->hv_cache = hv; a->k_cache = k; a->dirty = 0;
    return hv;
}

/* The selected points of the last te_arch_hv call (widths ascending). */
int te_arch_selected(A *a, int k, int32_t *ow, int32_t *ot) {
    te_arch_hv(a, k);
    for (int i = 0; i < a->nsel; i++) { ow[i] = a->sel_w[i]; ot[i] = a->sel_t[i]; }
    return a->nsel;
}

int te_arch_best_t(const A *a, int w) { return (w < 0 || w > MAX_TW) ? a->n : a->best_t[w]; }
int64_t te_arch_stamp(const A *a, int w) { return (w < 0 || w > MAX_TW) ? -1 : a->stamp[w]; }
int te_arch_get_perm(const A *a, int w, int32_t *out) {
    if (w < 0 || w > MAX_TW || a->slot[w] < 0) return 0;
    memcpy(out, a->pool + (size_t)a->slot[w] * a->n, sizeof(int32_t) * a->n);
    return 1;
}
int64_t te_arch_adds(const A *a) { return a->adds; }
int64_t te_arch_improvements(const A *a) { return a->improvements; }

/* Hypothetical HV if a staircase were added (does not modify the archive). */
int64_t te_arch_hv_with(A *a, const int32_t *ws, const int32_t *ts, int m, int k) {
    int32_t bt[MAX_TW + 1];
    memcpy(bt, a->best_t, sizeof bt);
    int changed = 0;
    for (int i = 0; i < m; i++) {
        int w = ws[i], t = ts[i];
        if (w < 0 || w > MAX_TW || w >= a->n || t >= a->n) continue;
        if (t < bt[w]) { bt[w] = t; changed = 1; }
    }
    if (!changed) return te_arch_hv(a, k);
    int32_t pw[MAX_TW + 1], pt[MAX_TW + 1];
    int mm = 0, best = a->n;
    for (int w = 0; w <= MAX_TW; w++) {
        int t = bt[w];
        if (t < best && w < a->n && t < a->n) { pw[mm] = w; pt[mm] = t; mm++; best = t; }
    }
    return te_hssp(pw, pt, mm, a->n, k, NULL, NULL);
}

/* Restore one archive entry (checkpoint load). */
void te_arch_set(A *a, int w, int t, const int32_t *perm, int64_t stamp) {
    if (w < 0 || w > MAX_TW) return;
    int sl = arch_alloc_slot(a);
    memcpy(a->pool + (size_t)sl * a->n, perm, sizeof(int32_t) * a->n);
    arch_release(a, a->slot[w]);
    a->slot[w] = sl; a->refc[sl]++;
    a->best_t[w] = t; a->stamp[w] = stamp; a->dirty = 1;
}
void te_arch_set_counters(A *a, int64_t adds, int64_t improvements) { a->adds = adds; a->improvements = improvements; }

/* --------------------------------------------------------------- greedy */

/* Bucket queue over alive vertices keyed by current degree: one vector per
 * degree, O(1) moves, uniformly random choice inside the minimum bucket. */
typedef struct {
    int n, cap;          /* degrees 0..cap-1 */
    int32_t **lst;       /* lst[d] -> vertices */
    int32_t *len, *alloc;
    int32_t *where;      /* index of v inside lst[key[v]] */
    int32_t *key;        /* current key of v, -1 if absent */
    int mink;            /* lower bound of the minimum key */
    int count;
} BQ;

static BQ *bq_new(int n, int cap) {
    BQ *q = xcalloc(1, sizeof(BQ));
    q->n = n; q->cap = cap;
    q->lst = xcalloc(cap, sizeof(int32_t *));
    q->len = xcalloc(cap, sizeof(int32_t));
    q->alloc = xcalloc(cap, sizeof(int32_t));
    q->where = xmalloc(sizeof(int32_t) * n);
    q->key = xmalloc(sizeof(int32_t) * n);
    for (int v = 0; v < n; v++) q->key[v] = -1;
    q->mink = 0; q->count = 0;
    return q;
}
static void bq_free(BQ *q) {
    for (int d = 0; d < q->cap; d++) free(q->lst[d]);
    free(q->lst); free(q->len); free(q->alloc); free(q->where); free(q->key); free(q);
}
static void bq_insert(BQ *q, int v, int k) {
    if (k >= q->cap) k = q->cap - 1;
    if (q->len[k] == q->alloc[k]) {
        q->alloc[k] = q->alloc[k] ? q->alloc[k] * 2 : 16;
        q->lst[k] = realloc(q->lst[k], sizeof(int32_t) * q->alloc[k]);
        if (!q->lst[k]) exit(2);
    }
    q->where[v] = q->len[k];
    q->lst[k][q->len[k]++] = v;
    q->key[v] = k;
    if (k < q->mink) q->mink = k;
    q->count++;
}
static void bq_remove(BQ *q, int v) {
    int k = q->key[v];
    if (k < 0) return;
    int i = q->where[v];
    int last = q->lst[k][--q->len[k]];
    q->lst[k][i] = last; q->where[last] = i;
    q->key[v] = -1;
    q->count--;
}
static void bq_update(BQ *q, int v, int k) { bq_remove(q, v); bq_insert(q, v, k); }
/* random vertex among those with key <= min + slack */
static int bq_pick(BQ *q, int slack, uint64_t *rng) {
    if (q->count == 0) return -1;
    while (q->mink < q->cap && q->len[q->mink] == 0) q->mink++;
    int lo = q->mink;
    if (slack <= 0) {
        return q->lst[lo][rng_below(rng, (uint32_t)q->len[lo])];
    }
    int hi = lo + slack; if (hi >= q->cap) hi = q->cap - 1;
    int64_t tot = 0;
    for (int d = lo; d <= hi; d++) tot += q->len[d];
    uint32_t r = rng_below(rng, (uint32_t)tot);
    for (int d = lo; d <= hi; d++) {
        if (r < (uint32_t)q->len[d]) return q->lst[d][r];
        r -= q->len[d];
    }
    return q->lst[lo][0];
}

/* Greedy elimination. Vertices with inH[v] = 1 are eliminated first (the
 * head), the rest afterwards (the torso); inH may be NULL (no head).
 * Within each phase: pick uniformly at random among alive vertices of the
 * phase whose current degree is <= minimum + slack (slack 0 = min-degree
 * with random tie-breaks, as hill_climbing.Starts.min_degree).
 * Writes perm and the exact step degrees. Returns the first step index whose
 * degree exceeds MAX_TW, or -1. */
int te_greedy(const G *g, const uint8_t *inH, int slack, uint64_t *rng, int32_t *perm, int32_t *deg) {
    int n = g->n, W = g->W;
    S *s = scratch_state(g);
    te_state_reset(g, s);
    BQ *qh = bq_new(n, n + 1), *qt = bq_new(n, n + 1);
    for (int v = 0; v < n; v++) {
        if (inH && inH[v]) bq_insert(qh, v, g->deg0[v]);
        else bq_insert(qt, v, g->deg0[v]);
    }
    uint64_t *nb = g->nb;
    int first = -1;
    for (int i = 0; i < n; i++) {
        BQ *q = qh->count ? qh : qt;
        int u = bq_pick(q, slack, rng);
        bq_remove(q, u);
        int lo, hi;
        int d = eliminate(s, u, nb, &lo, &hi);
        perm[i] = u; deg[i] = d;
        if (d > MAX_TW && first < 0) first = i;
        /* neighbours' degrees changed: recompute (masked popcount) */
        for (int w = lo; w <= hi; w++) {
            uint64_t x = nb[w];
            while (x) {
                int b = __builtin_ctzll(x); x &= x - 1;
                int v = w * 64 + b;
                int dv = cur_degree(s, v);
                if (qh->key[v] >= 0) bq_update(qh, v, dv); else bq_update(qt, v, dv);
            }
        }
        (void)W;
    }
    bq_free(qh); bq_free(qt);
    return first;
}

/* ------------------------------------------------------------------ inc */

/* Windowed incremental re-evaluation (cf. fast_degrees.py). deg[i] depends
 * only on perm[i] and the SET of vertices before i. A move that rearranges a
 * window [lo, hi] without changing which vertices occupy it leaves every
 * degree outside [lo, hi] unchanged, and leaves the elimination state after
 * hi unchanged too. So only [lo, hi] is re-walked, from the nearest snapshot. */
typedef struct {
    int n, W, stride, nsnap;
    int32_t *perm, *deg;
    uint64_t *snap_work;   /* nsnap * n*W : state before step k*stride */
    uint64_t *snap_alive;  /* nsnap * W */
    S *tmp;
} INC;

INC *te_inc_new(const G *g, int nsnap) {
    INC *c = xcalloc(1, sizeof(INC));
    c->n = g->n; c->W = g->W;
    if (nsnap < 1) nsnap = 1;
    c->stride = (g->n + nsnap - 1) / nsnap;
    c->nsnap = (g->n + c->stride - 1) / c->stride;
    c->perm = xmalloc(sizeof(int32_t) * g->n);
    c->deg = xmalloc(sizeof(int32_t) * g->n);
    c->snap_work = xmalloc((size_t)c->nsnap * g->n * g->W * 8);
    c->snap_alive = xmalloc((size_t)c->nsnap * g->W * 8);
    c->tmp = te_state_new(g);
    return c;
}
void te_inc_free(INC *c) {
    if (!c) return;
    free(c->perm); free(c->deg); free(c->snap_work); free(c->snap_alive); te_state_free(c->tmp); free(c);
}

static void inc_save(INC *c, int k, const S *s) {
    memcpy(c->snap_work + (size_t)k * c->n * c->W, s->work, (size_t)c->n * c->W * 8);
    memcpy(c->snap_alive + (size_t)k * c->W, s->alive, (size_t)c->W * 8);
}
static void inc_load(INC *c, int k, S *s) {
    memcpy(s->work, c->snap_work + (size_t)k * c->n * c->W, (size_t)c->n * c->W * 8);
    memcpy(s->alive, c->snap_alive + (size_t)k * c->W, (size_t)c->W * 8);
    s->step = k * c->stride;
}

/* full walk of perm, rebuilding every snapshot; returns first cap breach or -1 */
int te_inc_set(const G *g, INC *c, const int32_t *perm) {
    memcpy(c->perm, perm, sizeof(int32_t) * c->n);
    S *s = c->tmp;
    te_state_reset(g, s);
    int first = -1, lo, hi;
    for (int i = 0; i < c->n; i++) {
        if (i % c->stride == 0) inc_save(c, i / c->stride, s);
        int d = eliminate(s, perm[i], g->nb, &lo, &hi);
        c->deg[i] = d;
        if (d > MAX_TW && first < 0) first = i;
    }
    return first;
}

/* Degrees of newperm, which equals c->perm outside [lo, hi] and holds the
 * same SET of vertices inside it. deg_out gets all n degrees. c is unchanged. */
void te_inc_eval(const G *g, INC *c, const int32_t *newperm, int lo, int hi, int32_t *deg_out) {
    int n = c->n;
    if (lo < 0) lo = 0;
    if (hi >= n) hi = n - 1;
    if (deg_out != c->deg) memcpy(deg_out, c->deg, sizeof(int32_t) * n);
    if (lo > hi) return;
    int k = lo / c->stride;
    S *s = c->tmp;
    inc_load(c, k, s);
    int a, b;
    for (int i = k * c->stride; i < lo; i++) eliminate(s, c->perm[i], g->nb, &a, &b);
    for (int i = lo; i <= hi; i++) deg_out[i] = eliminate(s, newperm[i], g->nb, &a, &b);
}

/* Commit newperm (same window contract); refresh snapshots inside (lo, hi]. */
void te_inc_accept(const G *g, INC *c, const int32_t *newperm, int lo, int hi) {
    int n = c->n;
    if (lo < 0) lo = 0;
    if (hi >= n) hi = n - 1;
    if (lo > hi) return;
    int k = lo / c->stride;
    S *s = c->tmp;
    inc_load(c, k, s);
    int a, b;
    for (int i = k * c->stride; i < lo; i++) eliminate(s, c->perm[i], g->nb, &a, &b);
    for (int i = lo; i <= hi; i++) {
        if (i % c->stride == 0 && i > k * c->stride) inc_save(c, i / c->stride, s);
        c->deg[i] = eliminate(s, newperm[i], g->nb, &a, &b);
    }
    memcpy(c->perm + lo, newperm + lo, sizeof(int32_t) * (hi - lo + 1));
    /* snapshot at the first boundary after hi is still valid (set invariance) */
}

const int32_t *te_inc_perm(const INC *c) { return c->perm; }
const int32_t *te_inc_deg(const INC *c) { return c->deg; }
void te_inc_get(const INC *c, int32_t *perm, int32_t *deg) {
    if (perm) memcpy(perm, c->perm, sizeof(int32_t) * c->n);
    if (deg) memcpy(deg, c->deg, sizeof(int32_t) * c->n);
}

/* ------------------------------------------------------------------ hri */

/* Team HRI's operators, as in leaderboard-references/hri_lns.py.
 *
 * neighbour destroy: a seed vertex v drawn uniformly from the ordering and
 * its original-graph neighbours; on dense graphs v plus cap-1 neighbours
 * sampled without replacement. Returns the hole size; the hole is written
 * in the order the repair uses: degree descending, then vertex id. */
static const int32_t *cmp_deg_desc_deg;
static int cmp_deg_desc(const void *a, const void *b) {
    int x = *(const int32_t *)a, y = *(const int32_t *)b;
    int dx = cmp_deg_desc_deg[x], dy = cmp_deg_desc_deg[y];
    if (dx != dy) return dy - dx;
    return x - y;
}

int te_hri_neighbour_destroy(const G *g, const int32_t *perm, int cap, uint64_t *rng, int32_t *hole) {
    int n = g->n;
    int v = perm[rng_below(rng, (uint32_t)n)];
    int d = g->deg0[v];
    const int32_t *nv = g->nbr + g->off[v];
    int k = 0;
    hole[k++] = v;
    if (1 + d <= cap) {
        for (int i = 0; i < d; i++) hole[k++] = nv[i];
    } else {
        /* sample cap-1 of the d neighbours without replacement */
        int32_t *buf = g->tmp_c;
        memcpy(buf, nv, sizeof(int32_t) * d);
        for (int i = 0; i < cap - 1; i++) {
            int j = i + (int)rng_below(rng, (uint32_t)(d - i));
            int32_t x = buf[i]; buf[i] = buf[j]; buf[j] = x;
            hole[k++] = buf[i];
        }
    }
    cmp_deg_desc_deg = g->deg0;
    qsort(hole, k, sizeof(int32_t), cmp_deg_desc);
    return k;
}

/* Balanced repair (MEDIAN PLACEMENT, Biedl et al. 2005 §5), as
 * hri_lns.balanced_repair: the hole is removed from the ordering, then each
 * hole vertex (in the given order) is inserted at the median position of its
 * already-placed neighbours; for an odd count, on the side of the median
 * neighbour that leaves it better balanced; with no placed neighbour, at a
 * uniformly random slot. */
void te_hri_balanced_repair(const G *g, const int32_t *perm, const int32_t *hole, int k, uint64_t *rng, int32_t *out) {
    int n = g->n;
    int32_t *pos = g->tmp_a;      /* position in out, -1 if not placed */
    int32_t *placed = g->tmp_b;   /* scratch list */
    for (int v = 0; v < n; v++) pos[v] = 0;
    for (int i = 0; i < k; i++) pos[hole[i]] = -1;
    int len = 0;
    for (int i = 0; i < n; i++) { int u = perm[i]; if (pos[u] != -1) out[len++] = u; }
    for (int i = 0; i < len; i++) pos[out[i]] = i;
    for (int h = 0; h < k; h++) {
        int u = hole[h];
        const int32_t *nu = g->nbr + g->off[u];
        int du = g->deg0[u], cnt = 0;
        for (int i = 0; i < du; i++) { int x = nu[i]; if (pos[x] >= 0) placed[cnt++] = pos[x]; }
        int j;
        if (cnt == 0) {
            j = (int)rng_below(rng, (uint32_t)(len + 1));
        } else {
            qsort(placed, cnt, sizeof(int32_t), cmp_i32);
            if (cnt % 2 == 0) {
                j = placed[cnt / 2];
            } else {
                int mid = placed[cnt / 2];
                int w = out[mid];
                const int32_t *nw = g->nbr + g->off[w];
                int dw = g->deg0[w], before = 0, tot = 0;
                for (int i = 0; i < dw; i++) {
                    int x = nw[i];
                    if (pos[x] >= 0) { tot++; if (pos[x] < mid) before++; }
                }
                int after = tot - before;
                j = (before + 1 - after <= after + 1 - before) ? mid : mid + 1;
            }
        }
        if (j > len) j = len;
        memmove(out + j + 1, out + j, sizeof(int32_t) * (len - j));
        out[j] = u;
        len++;
        for (int i = j; i < len; i++) pos[out[i]] = i;
    }
}

/* Operator set A: remove k random positions (taken highest index first) and
 * reinsert each at a uniformly random slot, as hri_lns.random_destroy_repair. */
void te_hri_random_destroy_repair(int n, const int32_t *perm, int k, uint64_t *rng, int32_t *out, int32_t *scratch) {
    /* choose k distinct indices */
    int32_t *idx = scratch;          /* needs n */
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < k; i++) {
        int j = i + (int)rng_below(rng, (uint32_t)(n - i));
        int32_t x = idx[i]; idx[i] = idx[j]; idx[j] = x;
    }
    qsort(idx, k, sizeof(int32_t), cmp_i32);
    int32_t popped[64];
    if (k > 64) k = 64;
    memcpy(out, perm, sizeof(int32_t) * n);
    int len = n;
    for (int a = k - 1, c = 0; a >= 0; a--, c++) {   /* highest index first */
        int i = idx[a];
        popped[c] = out[i];
        memmove(out + i, out + i + 1, sizeof(int32_t) * (len - i - 1));
        len--;
    }
    for (int c = 0; c < k; c++) {
        int j = (int)rng_below(rng, (uint32_t)(len + 1));
        memmove(out + j + 1, out + j, sizeof(int32_t) * (len - j));
        out[j] = popped[c];
        len++;
    }
}

/* ------------------------------------------------------------ local loops */

/* scratch buffers for the loops below (one thread per process) */
static int32_t *L_cand = NULL, *L_deg = NULL, *L_sw = NULL, *L_st = NULL;
static int L_n = 0;
static void loop_scratch(int n) {
    if (L_n >= n) return;
    free(L_cand); free(L_deg); free(L_sw); free(L_st);
    L_cand = xmalloc(sizeof(int32_t) * (n + 1));
    L_deg = xmalloc(sizeof(int32_t) * (n + 1));
    L_sw = xmalloc(sizeof(int32_t) * (n + 1));
    L_st = xmalloc(sizeof(int32_t) * (n + 1));
    L_n = n;
}

/* move perm[i] to position j (positions of the n-1 list after removal) */
static inline void move_elem(int32_t *p, int n, int i, int j) {
    int32_t v = p[i];
    if (j == i) return;
    if (j < i) { memmove(p + j + 1, p + j, sizeof(int32_t) * (i - j)); p[j] = v; }
    else { memmove(p + i, p + i + 1, sizeof(int32_t) * (j - i)); p[j] = v; }
}

static inline void reverse_seg(int32_t *p, int lo, int hi) {
    while (lo < hi) { int32_t x = p[lo]; p[lo] = p[hi]; p[hi] = x; lo++; hi--; }
}

/* Hill Climbing's inner loop (hill_climbing.HillClimber.climb). c holds the
 * current ordering. Each iteration applies one of the four moves, drawn
 * uniformly -- swap neighbours, swap any two, move one vertex, reverse a
 * segment of 2..10 -- evaluates it incrementally, banks its staircase in a,
 * and accepts it iff the smallest t within `target` strictly drops.
 * *cost_io is the current cost (n = unreachable). Returns accepted moves. */
int te_hc_run(const G *g, INC *c, A *a, int target, int iters, uint64_t *rng,
              int32_t *cost_io, int64_t stamp0) {
    int n = g->n;
    loop_scratch(n);
    int cost = *cost_io, acc = 0;
    for (int it = 0; it < iters; it++) {
        memcpy(L_cand, c->perm, sizeof(int32_t) * n);
        int op = (int)rng_below(rng, 4), lo, hi;
        if (op == 0) {
            int i = (int)rng_below(rng, (uint32_t)(n - 1));
            int32_t x = L_cand[i]; L_cand[i] = L_cand[i + 1]; L_cand[i + 1] = x;
            lo = i; hi = i + 1;
        } else if (op == 1) {
            int i = (int)rng_below(rng, (uint32_t)n), j = (int)rng_below(rng, (uint32_t)n);
            int32_t x = L_cand[i]; L_cand[i] = L_cand[j]; L_cand[j] = x;
            lo = i < j ? i : j; hi = i < j ? j : i;
        } else if (op == 2) {
            int i = (int)rng_below(rng, (uint32_t)n), j = (int)rng_below(rng, (uint32_t)n);
            move_elem(L_cand, n, i, j);
            lo = i < j ? i : j; hi = i < j ? j : i;
        } else {
            int i = (int)rng_below(rng, (uint32_t)(n - 1));
            int L = 2 + (int)rng_below(rng, 9);
            int j = i + L; if (j > n) j = n;
            reverse_seg(L_cand, i, j - 1);
            lo = i; hi = j - 1;
        }
        te_inc_eval(g, c, L_cand, lo, hi, L_deg);
        int m = te_staircase(L_deg, n, L_sw, L_st);
        if (m) te_arch_add(a, L_cand, L_sw, L_st, m, stamp0 + it);
        int nc = te_best_t_for_width(L_deg, n, target);
        if (nc < cost) {
            te_inc_accept(g, c, L_cand, lo, hi);
            cost = nc; acc++;
        }
    }
    *cost_io = cost;
    return acc;
}

/* cost of a degree vector for a target width: smallest t with
 * max(deg[t:]) <= target (n if capped), and the degree of the blocking step
 * t-1 (0 if t == 0) as a tie-breaker. */
static inline void target_cost(const int32_t *deg, int n, int target, int *pt, int *pb) {
    int t = te_best_t_for_width(deg, n, target);
    *pt = t;
    *pb = (t > 0 && t < n) ? deg[t - 1] : 0;
    if (t >= n) *pb = 1 << 20;
}

/* Targeted local search for one width (EA-1, EA-2). c holds the current
 * ordering. Moves concentrate on the blocking step b = t_target - 1, the step
 * that keeps the threshold from dropping: a position p is drawn near b
 * (geometric bias, both sides) and perm[p] is moved to, swapped with, or a
 * short segment around it reversed within `window` positions. Each candidate
 * is evaluated incrementally and banked. A move is kept when (t, blocking
 * degree) improves lexicographically, or ties with probability p_side.
 * Returns the number of strict improvements; *t_io / *b_io carry the cost. */
int te_ls_target(const G *g, INC *c, A *a, int target, int iters, int window,
                 double p_side, uint64_t *rng, int32_t *t_io, int32_t *b_io, int64_t stamp0) {
    int n = g->n;
    loop_scratch(n);
    int t, b;
    target_cost(c->deg, n, target, &t, &b);
    int better = 0;
    if (window < 2) window = 2;
    for (int it = 0; it < iters; it++) {
        if (t == 0) break;
        int anchor = (t >= n) ? n - 1 : t - 1;
        /* position near the anchor: geometric offset, mostly before it */
        int off = 0;
        while (off < window && rng_unit(rng) < 0.85) off++;
        int p = (rng_unit(rng) < 0.75) ? anchor - off : anchor + off;
        if (p < 0) p = 0;
        if (p >= n) p = n - 1;
        int q = p + (int)rng_below(rng, (uint32_t)(2 * window + 1)) - window;
        if (q < 0) q = 0;
        if (q >= n) q = n - 1;
        if (q == p) q = (p + 1 < n) ? p + 1 : p - 1;
        memcpy(L_cand, c->perm, sizeof(int32_t) * n);
        int op = (int)rng_below(rng, 3), lo = p < q ? p : q, hi = p < q ? q : p;
        if (op == 0) move_elem(L_cand, n, p, q);
        else if (op == 1) { int32_t x = L_cand[p]; L_cand[p] = L_cand[q]; L_cand[q] = x; }
        else reverse_seg(L_cand, lo, hi);
        te_inc_eval(g, c, L_cand, lo, hi, L_deg);
        int m = te_staircase(L_deg, n, L_sw, L_st);
        if (m) te_arch_add(a, L_cand, L_sw, L_st, m, stamp0 + it);
        int nt, nb;
        target_cost(L_deg, n, target, &nt, &nb);
        if (nt < t || (nt == t && nb < b)) {
            te_inc_accept(g, c, L_cand, lo, hi);
            t = nt; b = nb; better++;
        } else if (nt == t && nb == b && rng_unit(rng) < p_side) {
            te_inc_accept(g, c, L_cand, lo, hi);
        }
    }
    *t_io = t; *b_io = b;
    return better;
}

/* ---------------------------------------------------------------- policy */

/* The adaptive decoder of the NE methods: the next vertex to eliminate is the
 * alive vertex with the highest score among those whose current degree is
 * <= MAX_TW (the cap is never broken by choice; if no vertex qualifies, the
 * lowest-degree one is taken and the ordering is void). Scores come from a
 * one-hidden-layer ReLU MLP over NF = NS + 6 features per vertex:
 *
 *   static (NS, given by the caller per vertex: original degree, core
 *           number, twin-class size, clique size, spectral coordinates ...)
 *   live   current degree /100, (current degree - target)/100, fraction of
 *          original neighbours already eliminated, fill gained /100
 *   global progress (step/n), target/500
 *
 * The static part of the first layer is computed once per decoding. A vertex
 * is rescored whenever its live features change (it was a current neighbour of
 * the eliminated vertex), and every vertex every `rescore_every` steps (the
 * progress input changes). weights: W1 (H x NF, row-major), b1 (H), w2 (H),
 * b2 (1). Writes perm and the exact step degrees; returns the first cap breach
 * or -1. */
#define NLIVE 4
#define NGLOB 2
static float *P_score = NULL, *P_zs = NULL; static int32_t *P_dcur = NULL, *P_elim = NULL;
static uint8_t *P_alive = NULL; static int P_n = 0, P_h = 0;

int te_policy_decode(const G *g, const float *stat, int NS, const float *weights, int H,
                     int target, int rescore_every, int32_t *perm, int32_t *deg) {
    int n = g->n;
    int NF = NS + NLIVE + NGLOB;
    if (H > 64) return -2;
    const float *W1 = weights, *b1 = weights + (size_t)H * NF, *w2 = b1 + H;
    float b2 = w2[H];
    if (P_n < n || P_h < H) {
        free(P_score); free(P_zs); free(P_dcur); free(P_elim); free(P_alive);
        P_score = xmalloc(sizeof(float) * n); P_zs = xmalloc(sizeof(float) * (size_t)n * H);
        P_dcur = xmalloc(sizeof(int32_t) * n); P_elim = xmalloc(sizeof(int32_t) * n);
        P_alive = xmalloc(n); P_n = n; P_h = H;
    }
    /* dynamic-input columns of W1, transposed for a tight inner loop */
    float Wd[NLIVE + NGLOB][64];
    for (int h = 0; h < H; h++)
        for (int k = 0; k < NLIVE + NGLOB; k++) Wd[k][h] = W1[(size_t)h * NF + NS + k];
    for (int v = 0; v < n; v++) {
        const float *sv = stat + (size_t)v * NS;
        float *z = P_zs + (size_t)v * H;
        for (int h = 0; h < H; h++) {
            const float *row = W1 + (size_t)h * NF;
            float acc = b1[h];
            for (int k = 0; k < NS; k++) acc += row[k] * sv[k];
            z[h] = acc;
        }
        P_dcur[v] = g->deg0[v]; P_elim[v] = 0; P_alive[v] = 1;
    }
    S *s = scratch_state(g);
    te_state_reset(g, s);
    const float tg = target / 500.0f, inv_n = 1.0f / n;
    float prog = 0.0f;
#define SCORE(v) do { \
        int d0_ = g->deg0[v], dc_ = P_dcur[v], el_ = P_elim[v]; \
        float x0 = dc_ * 0.01f, x1 = (dc_ - target) * 0.01f; \
        float x2 = d0_ ? (float)el_ / d0_ : 0.0f, x3 = (dc_ - (d0_ - el_)) * 0.01f; \
        const float *z_ = P_zs + (size_t)(v) * H; \
        float o_ = b2; \
        for (int h = 0; h < H; h++) { \
            float a_ = z_[h] + Wd[0][h] * x0 + Wd[1][h] * x1 + Wd[2][h] * x2 + Wd[3][h] * x3 \
                       + Wd[4][h] * prog + Wd[5][h] * tg; \
            o_ += w2[h] * (a_ > 0.0f ? a_ : 0.0f); \
        } \
        P_score[v] = o_; \
    } while (0)
    for (int v = 0; v < n; v++) SCORE(v);
    int first = -1;
    uint64_t *nb = g->nb;
    for (int i = 0; i < n; i++) {
        if (rescore_every > 0 && i > 0 && i % rescore_every == 0) {
            prog = i * inv_n;
            for (int v = 0; v < n; v++) if (P_alive[v]) SCORE(v);
        }
        int best = -1, low = -1;
        float bs = -INFINITY;
        for (int v = 0; v < n; v++) {
            if (!P_alive[v]) continue;
            if (low < 0 || P_dcur[v] < P_dcur[low]) low = v;
            if (P_dcur[v] <= MAX_TW && P_score[v] > bs) { bs = P_score[v]; best = v; }
        }
        if (best < 0) best = low;
        int u = best, lo, hi;
        P_alive[u] = 0;
        int d = eliminate(s, u, nb, &lo, &hi);
        perm[i] = u; deg[i] = d;
        if (d > MAX_TW && first < 0) first = i;
        const int32_t *nu = g->nbr + g->off[u];
        for (int k = 0; k < g->deg0[u]; k++) P_elim[nu[k]]++;
        for (int w = lo; w <= hi; w++) {
            uint64_t xx = nb[w];
            while (xx) {
                int bb = __builtin_ctzll(xx); xx &= xx - 1;
                int v = w * 64 + bb;
                P_dcur[v] = cur_degree(s, v);
                SCORE(v);
            }
        }
    }
#undef SCORE
    return first;
}

int te_policy_nweights(int NS, int H) { int NF = NS + NLIVE + NGLOB; return H * NF + H + H + 1; }

/* ---------------------------------------------------------- completion */

/* Guided greedy completion (EA-1, EA-2): eliminate prefix[0..plen-1] in the
 * given order, then complete the ordering greedily: at each step take the
 * alive vertex maximising
 *     -current degree + alpha * (share of its original neighbours already
 *      eliminated) + lam * prio[v] + tiny noise
 * among those whose current degree is <= MAX_TW (else the lowest degree).
 * alpha > 0 prefers the elimination frontier (cf. maximum cardinality
 * search); prio carries the parent's order (e.g. 1 - rank/n), so lam moves
 * the completion from pure greedy (lam 0) towards the parent (lam large).
 * Writes perm and exact degrees; returns the first cap breach or -1. */
int te_greedy_complete(const G *g, const int32_t *prefix, int plen, const float *prio,
                       float lam, float alpha, uint64_t *rng, int32_t *perm, int32_t *deg) {
    int n = g->n;
    if (P_n < n) {
        free(P_score); free(P_zs); free(P_dcur); free(P_elim); free(P_alive);
        P_score = xmalloc(sizeof(float) * n); P_zs = xmalloc(sizeof(float) * (size_t)n * (P_h ? P_h : 1));
        P_dcur = xmalloc(sizeof(int32_t) * n); P_elim = xmalloc(sizeof(int32_t) * n);
        P_alive = xmalloc(n); P_n = n;
    }
    S *s = scratch_state(g);
    te_state_reset(g, s);
    for (int v = 0; v < n; v++) { P_dcur[v] = g->deg0[v]; P_elim[v] = 0; P_alive[v] = 1; }
    int first = -1, lo, hi;
    uint64_t *nb = g->nb;
    float *noise = P_score;
    for (int v = 0; v < n; v++) noise[v] = (float)(rng_unit(rng) * 1e-3);
    for (int i = 0; i < n; i++) {
        int u;
        if (i < plen) {
            u = prefix[i];
        } else {
            int best = -1, low = -1;
            float bs = -INFINITY;
            for (int v = 0; v < n; v++) {
                if (!P_alive[v]) continue;
                if (low < 0 || P_dcur[v] < P_dcur[low]) low = v;
                if (P_dcur[v] > MAX_TW) continue;
                int d0 = g->deg0[v];
                float sc = -(float)P_dcur[v] + alpha * (d0 ? (float)P_elim[v] / d0 : 1.0f)
                           + (prio ? lam * prio[v] : 0.0f) + noise[v];
                if (sc > bs) { bs = sc; best = v; }
            }
            u = best >= 0 ? best : low;
        }
        P_alive[u] = 0;
        int d = eliminate(s, u, nb, &lo, &hi);
        perm[i] = u; deg[i] = d;
        if (d > MAX_TW && first < 0) first = i;
        const int32_t *nu = g->nbr + g->off[u];
        for (int k = 0; k < g->deg0[u]; k++) P_elim[nu[k]]++;
        for (int w = lo; w <= hi; w++) {
            uint64_t xx = nb[w];
            while (xx) {
                int bb = __builtin_ctzll(xx); xx &= xx - 1;
                int v = w * 64 + bb;
                P_dcur[v] = cur_degree(s, v);
            }
        }
    }
    return first;
}

/* Endpoint local search (EA-1's E island, EA-2's endpoint cell): minimise
 * the threshold-0 width W lexicographically as (W, number of steps at W,
 * sum over steps of (d - (W - 3))^2 for d > W - 3). Moves concentrate on the
 * steps at or near W: a position p with deg >= W - 1 is drawn uniformly, and
 * perm[p] is moved to, swapped with, or reversed with a position within
 * `window`. Each candidate is evaluated incrementally and banked. Keeps a move
 * when the cost does not get worse (ties with probability p_side). Returns
 * the number of strict improvements; *w_io gets the final W. */
static inline void endpoint_cost(const int32_t *deg, int n, int64_t *c) {
    int W = 0;
    for (int i = 0; i < n; i++) if (deg[i] > W) W = deg[i];
    int64_t cnt = 0, sq = 0;
    for (int i = 0; i < n; i++) {
        if (deg[i] == W) cnt++;
        int e = deg[i] - (W - 3);
        if (e > 0) sq += (int64_t)e * e;
    }
    c[0] = W; c[1] = cnt; c[2] = sq;
}
static inline int cost_cmp(const int64_t *a, const int64_t *b) {
    for (int k = 0; k < 3; k++) { if (a[k] < b[k]) return -1; if (a[k] > b[k]) return 1; }
    return 0;
}

int te_ls_endpoint(const G *g, INC *c, A *a, int iters, int window, double p_side,
                   uint64_t *rng, int32_t *w_io, int64_t stamp0) {
    int n = g->n;
    loop_scratch(n);
    int32_t *hot = g->tmp_c;
    int64_t cur[3], cand[3];
    endpoint_cost(c->deg, n, cur);
    int better = 0;
    if (window < 2) window = 2;
    for (int it = 0; it < iters; it++) {
        int W = (int)cur[0], nh = 0;
        for (int i = 0; i < n; i++) if (c->deg[i] >= W - 1 && c->deg[i] > 0) hot[nh++] = i;
        if (nh == 0) break;
        int p = hot[rng_below(rng, (uint32_t)nh)];
        int q = p + (int)rng_below(rng, (uint32_t)(2 * window + 1)) - window;
        if (q < 0) q = 0;
        if (q >= n) q = n - 1;
        if (q == p) q = (p + 1 < n) ? p + 1 : p - 1;
        memcpy(L_cand, c->perm, sizeof(int32_t) * n);
        int op = (int)rng_below(rng, 3), lo = p < q ? p : q, hi = p < q ? q : p;
        if (op == 0) move_elem(L_cand, n, p, q);
        else if (op == 1) { int32_t x = L_cand[p]; L_cand[p] = L_cand[q]; L_cand[q] = x; }
        else reverse_seg(L_cand, lo, hi);
        te_inc_eval(g, c, L_cand, lo, hi, L_deg);
        int m = te_staircase(L_deg, n, L_sw, L_st);
        if (m) te_arch_add(a, L_cand, L_sw, L_st, m, stamp0 + it);
        endpoint_cost(L_deg, n, cand);
        int cmp = cost_cmp(cand, cur);
        if (cmp < 0 || (cmp == 0 && rng_unit(rng) < p_side)) {
            te_inc_accept(g, c, L_cand, lo, hi);
            if (cmp < 0) better++;
            cur[0] = cand[0]; cur[1] = cand[1]; cur[2] = cand[2];
        }
    }
    *w_io = (int32_t)cur[0];
    return better;
}
