# Torso Decompositions — the problem, explained from scratch

This branch contains the **simplest possible** working solution to the ESA
SpOC-3 "Torso Decompositions" problem, written so that someone who has never
seen the problem can read it in one sitting and understand all of it.

Five files, **pure Python standard library** — no numpy, no packages, no
build step. If you have `python3`, everything here runs.

```
torso.py             the problem: graph, solution, evaluation, scoring
hill_climbing.py     the search: four operators, one accept rule
generate.py          make instances: toy, random, planted
validate.py          check an answer is legal and re-score it
test_correctness.py  22 checks proving the above is right
data/                the three competition graphs + a 12-vertex toy
```

Read this file top to bottom and you will know: what the problem is, how an
answer is written down, how to solve a small instance **by hand**, how the
score is computed, how the algorithm works, and how to run it.

---

# Part 1 — The problem

## 1.1 What you are given

An **undirected graph**: a set of vertices (numbered `0, 1, 2, …`) and edges
joining pairs of them. Nothing else. No weights, no directions.

Here is the toy instance in `data/toy.gr`, 12 vertices and 16 edges:

```
vertex : neighbours
   0   : 1, 2, 11
   1   : 0, 2, 3
   2   : 0, 1, 3
   3   : 1, 2, 4
   4   : 3, 5
   5   : 4, 6
   6   : 5, 7, 8
   7   : 6, 8, 9
   8   : 6, 7, 9
   9   : 7, 8, 10
  10   : 9, 11
  11   : 0, 10
```

Its shape is a ring made of four parts:

```
   ┌─────────────┐        ┌─────────────┐
   │  cluster A  │──4──5──│  cluster B  │──10──11──┐
   │  {0,1,2,3}  │ bridge │  {6,7,8,9}  │   tail   │
   └─────────────┘        └─────────────┘          │
          ▲                                        │
          └────────────────────────────────────────┘
                     (edge 11–0 closes the ring)
```

Cluster A is a triangle `{0,1,2}` with vertex `3` hanging off `1` and `2`.
Cluster B is the same shape: triangle `{6,7,8}` with `9` hanging off `7`
and `8`. The clusters are the dense parts; everything else is a thin path.

The `.gr` file format is one edge per line, two numbers:

```
0 1
0 2
0 11
1 2
...
```

## 1.2 Eliminating a vertex, and fill-in

The whole problem revolves around one operation: **eliminating** a vertex.

> To eliminate vertex `u`: delete `u` from the graph, then add edges so that
> **all of `u`'s remaining neighbours become joined to each other** (they
> form a clique).

Those newly added edges are called **fill-in**. This is the crucial part:
eliminating vertices does not simply shrink the graph, it also makes what
remains **denser**.

A three-line example. Suppose vertex `5` has neighbours `4` and `6`, and
`4–6` is not an edge:

```
   before                  eliminate 5             after
   4 ── 5 ── 6      →    remove 5, join      →    4 ──── 6
                          its neighbours           (new edge: fill-in)
```

We started with 2 edges and ended with 1, but that edge `4–6` did not exist
before. Do this a few hundred times and the surviving graph can become far
denser than the one you started with.

## 1.3 What you choose

Your answer to the problem is two things:

| you choose | what it is |
|---|---|
| **`perm`** | an **order** in which to eliminate the vertices — a permutation of `0 … n-1` |
| **`t`** | a **threshold**: how many vertices at the front of `perm` are eliminated *before we start measuring* |

The vertices from position `t` onwards are called the **torso**. Its size is
`n − t`.

```
[5, 4, 11, 10, 0, 1, 2, 3, 9, 6, 7, 8]   with t = 9
 └──────────────────┬──────────┘  └──┬──┘
   eliminated first            the "torso"
   (not measured for width)     (measured)
```

## 1.4 How an answer is written down — the decision vector

That has to be stored in a form a program can read. The competition's
format is deliberately plain: **one flat list of integers**, called a
**decision vector**.

```
decision vector = [ perm[0], perm[1], ..., perm[n-1],  t ]
                   └──────────── n numbers ─────────┘  └┬┘
                     the elimination order            threshold
```

Its length is always `n + 1`: the permutation, then one extra number on the
end. For the toy instance (`n = 12`) a decision vector is 13 numbers:

```
[5, 4, 11, 10, 0, 1, 2, 3, 9, 6, 7, 8, 9]
 └───────────── perm (12 numbers) ─────┘ └┬┘
                                       t = 9
```

Read it as: *eliminate 5 first, then 4, then 11, … and start measuring from
position 9.*

A decision vector is **legal** only if all of these hold — `validate.py`
checks every one:

| rule | why |
|---|---|
| length is exactly `n + 1` | one slot per vertex, plus the threshold |
| the first `n` entries are a permutation of `0 … n-1` | every vertex eliminated exactly once, none twice, none missing |
| `0 ≤ t < n` | the threshold must point at a real position |
| no elimination step exceeds width 500 | the organisers' hard cap (see 1.6) |

**One decision vector gives exactly one point** `(width, t)` on the
trade-off curve. To describe a whole curve you need several, so a
**submission** is a *list* of decision vectors — at most **20**:

```json
{
  "instance": "toy.gr",
  "n": 12,
  "score": -121,
  "decisionVector": [
    [7, 1, 8, 2, 9, 5, 0, 3, 10, 6, 11, 4, 11],
    [7, 1, 8, 2, 9, 5, 0, 3, 10, 6, 11, 4, 10],
    [7, 1, 8, 2, 9, 5, 0, 3, 10, 6, 11, 4,  2],
    [7, 1, 8, 2, 9, 5, 0, 3, 10, 6, 11, 4,  0]
  ]
}
```

Notice that all four vectors here share the **same permutation** and differ
only in their last number. That is the staircase of Part 2.1 written out:
one good ordering, read at four different thresholds, giving four different
trade-off points. Different vectors *may* use different permutations — on
the real instances they usually do — but they do not have to.

`hill_climbing.py` writes exactly this file; `validate.py` reads it.

## 1.5 The two objectives

Eliminate the vertices in order. At each step `i`, write down:

> `deg[i]` = how many of `perm[i]`'s neighbours were **still present** when
> it was eliminated (counting fill-in edges added by earlier steps)

Then the two numbers being judged are:

| objective | formula | meaning |
|---|---|---|
| **width** | `max(deg[i] for i ≥ t)` | the worst step in the torso |
| **t** | `t` | how many vertices you had to remove first |

**Both are minimised.** Lower width is better, and lower `t` is better.

Why lower `t` is better is worth pausing on: `t` counts the vertices you
*threw away*, so a smaller `t` means the surviving torso (`n − t` vertices)
is **larger**. You are trying to keep as much of the graph as possible while
keeping the width down.

These two goals **fight each other**:

- Want a tiny width? Eliminate almost everything first — but then `t` is
  huge and the torso is nearly empty.
- Want `t = 0` (keep the whole graph)? Then the width is whatever the best
  possible ordering of the entire graph gives — which may be large.

So there is no single best answer. There is a **trade-off curve**, and your
job is to map it out.

## 1.6 The one hard rule

If `deg[i] > 500` at **any** step — including the eliminated head, before
`t` — the whole answer is **void**. Not penalised: void. The code marks this
by returning a width of `501`.

This catches people out: it is tempting to think the head does not matter
because it is not measured. It is not measured for *width*, but it still has
to obey the cap.

---

# Part 2 — Solving the toy instance by hand

Let us do a complete example with no computer. Take this elimination order:

```
perm = [5, 4, 11, 10, 0, 1, 2, 3, 9, 6, 7, 8]
```

Eliminate them one at a time. At each step, look at which neighbours are
**still present** (i.e. appear *later* in the order), count them, and add
the fill-in edges.

| step | vertex | neighbours still present | `deg` | fill-in added |
|---:|---:|---|---:|---|
| 0 | 5 | 4, 6 | **2** | 4–6 |
| 1 | 4 | 3, 6 | **2** | 3–6 |
| 2 | 11 | 0, 10 | **2** | 0–10 |
| 3 | 10 | 0, 9 | **2** | 0–9 |
| 4 | 0 | 1, 2, 9 | **3** | 1–9, 2–9 |
| 5 | 1 | 2, 3, 9 | **3** | 3–9 |
| 6 | 2 | 3, 9 | **2** | — |
| 7 | 3 | 6, 9 | **2** | 6–9 |
| 8 | 9 | 6, 7, 8 | **3** | — |
| 9 | 6 | 7, 8 | **2** | — |
| 10 | 7 | 8 | **1** | — |
| 11 | 8 | — | **0** | — |

Follow step 1 to see fill-in working: vertex `4`'s original neighbours were
`3` and `5`. But `5` was already eliminated at step 0, and that step added
the edge `4–6`. So when we eliminate `4`, its surviving neighbours are `3`
and **`6`** — a vertex it was never originally connected to.

So:

```
deg = [2, 2, 2, 2, 3, 3, 2, 2, 3, 2, 1, 0]
```

## 2.1 Reading every answer off one table

Here is the fact that makes this problem tractable. The width is
`max(deg[i] for i ≥ t)`. So to get the width for **every** `t`, sweep the
`deg` array from the right, keeping a running maximum:

| `t` | 11 | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deg[t]` | 0 | 1 | 2 | 3 | 2 | 2 | 3 | 3 | 2 | 2 | 2 | 2 |
| **width** | **0** | **1** | **2** | **3** | 3 | 3 | 3 | 3 | 3 | 3 | 3 | **3** |

Each time the running maximum steps up, we have found the **smallest `t`**
that achieves that width — and smaller `t` is better, so that is exactly the
point we want. This one permutation therefore gives us four trade-off
points, for free:

| width | smallest `t` | torso size (`n − t`) |
|---:|---:|---:|
| 0 | 11 | 1 |
| 1 | 10 | 2 |
| 2 | 9 | 3 |
| 3 | 0 | 12 |

**One evaluation of one permutation yields a whole staircase of answers.**
This is `Solution.staircase()` in `torso.py`, and it is why the search code
is so short. It is also why the four decision vectors in Part 1.4 could
share a permutation.

---

# Part 3 — How the score is computed

We now have a set of `(width, t)` points. How good are they?

The competition uses **hypervolume**: the area those points dominate,
measured against the corner `(n, n)`.

Think of it as a graph with width across and `t` up. Each point `(w, t)`
"covers" the whole rectangle from itself out to the corner `(n, n)`. The
score is the **area of the union** of all those rectangles. More area is
better.

```
   t
   12 ┤ ← corner (n,n) = (12,12)
      │
   11 ┤██ ● (0,11)
      │████
   10 ┤█████ ● (1,10)
      │███████
    9 ┤████████ ● (2,9)
      │
      │        (everything below and right of a point is covered)
    0 ┤████████████████████ ● (3,0)
      └┬───┬───┬───┬────────┬
       0   1   2   3   ...  12   width
```

To compute it, sort the points by width and add up vertical strips. Each
point owns the strip from its own width to the **next** point's width, and
that strip is `n − t` tall:

```
point (0, 11):  strip width 1−0 = 1,   height 12−11 = 1    →   1 × 1  =   1
point (1, 10):  strip width 2−1 = 1,   height 12−10 = 2    →   1 × 2  =   2
point (2,  9):  strip width 3−2 = 1,   height 12− 9 = 3    →   1 × 3  =   3
point (3,  0):  strip width 12−3 = 9,  height 12− 0 = 12   →   9 × 12 = 108
                                                            ─────────────
                                                     total area = 114
```

**The official score is the negative of the area:**

```
score = −114
```

So **more negative is better**. A score of `−121` beats `−114`.

Two more rules:

- **At most 20 decision vectors** may be submitted. On the big graphs you
  will find hundreds of trade-off points and must choose the best 20.
- **Dominated points are wasted.** If point A has both a width no larger and
  a `t` no larger than point B, then B contributes nothing. `validate.py`
  warns about these.

---

# Part 4 — Why this is hard

The number of orderings is `n!`. For the toy that is 479 million; for
`small-graph` (n = 1357) it is a number with over 3,600 digits. You cannot
try them all, and there is no known formula for the best one.

Worse, the objective is **not smooth**: swapping two vertices can change the
fill-in cascade for every later step, so a tiny change to the input can
cause a large, unpredictable change to the output.

This is why we use a **heuristic** — a method that finds good answers
without proving they are the best.

---

# Part 5 — The algorithm: hill climbing

Hill climbing is the simplest local search that exists:

```
1.  start from some solution S
2.  make a small random change to a COPY of it   →  R
3.  if R is better than S, keep R; otherwise throw R away
4.  repeat until out of time
```

That is the entire algorithm. No population, no temperature, no memory, no
machine learning. In `HillClimber.climb` it is six lines of code.

The name comes from the picture: you are standing on a hillside in fog,
taking small steps, and only ever stepping *upward*. You will certainly
reach a hilltop. It might not be the highest hill — that is the known
weakness of the method, and it is honest to say so.

## 5.1 Handling two objectives with a single-objective method

Hill climbing compares two things and keeps the better one. But our problem
has *two* numbers, and we want a whole curve of answers. We resolve this in
the simplest way that still works:

> Pick a **target width `W`**. Hill-climb to **minimise `t`** at that width.

Now each climb is an ordinary single-objective climb — one number going
down — which is exactly what makes it easy to explain. Run one climb per
target width, collect the results, and you have your trade-off curve.

And remember Part 2.1: every permutation the climber looks at already
contains a point at *every* width. So each climb donates its whole staircase
to the shared front, even for the candidates it rejects. Choosing a target
width only decides which number is being pushed down.

## 5.2 The four moves

From `Operators` in `hill_climbing.py`. Each takes a permutation and returns
a new one — never modifying the original, because a rejected change must be
discarded cleanly.

| operator | what it does |
|---|---|
| `swap_neighbours` | swap two adjacent positions |
| `swap_any` | swap two positions anywhere |
| `move_vertex` | remove one vertex and reinsert it elsewhere |
| `reverse_segment` | reverse a short run of positions |

## 5.3 Where it starts

A random order is a terrible start: on the bigger graphs it blows past the
500 cap immediately, so every candidate is void and the climber has nothing
to compare. The default is therefore **min-degree**: repeatedly eliminate
whichever vertex currently has the fewest surviving neighbours. It is the
classic textbook heuristic for orderings like this and gives the climber a
legal, decent starting point.

## 5.4 What the climber actually finds

Here is the payoff, and the clearest single illustration of what the search
is *for*. Our hand-made answer from Part 2 scored `−114`. Run the climber
for three seconds and it finds `−121`. The difference is one point:

| | width | `t` | torso size |
|---|---:|---:|---:|
| our hand answer | 2 | 9 | 3 |
| climber's answer | 2 | **2** | **10** |

Same width, but the torso holds 10 vertices instead of 3. The order it
found:

```
perm = [7, 1, 8, 2, 9, 5, 0, 3, 10, 6, 11, 4]
deg  = [3, 3, 2, 2, 2, 2, 2, 2,  2, 2,  1, 0]
        └──┬──┘
       the only two expensive steps
```

Look at where the two `deg = 3` steps are: **positions 0 and 1**. With
`t = 2` they sit in the eliminated head, so they are **not counted** in the
width — while still obeying the 500 cap. Everything from position 2 onward
costs at most 2.

That is the whole game in one line: *arrange the ordering so the expensive
steps happen early, then set the threshold just past them.*

---

# Part 6 — The files

### `torso.py` — the problem

- **`Graph`** — loads a `.gr` file. Stores the graph twice: `adj` (a list of
  sets, the readable form) and `bits` (Python integers used as bitsets, the
  fast form). Both describe the same graph.
- **`Solution`** — one permutation, and everything derived from it:
  `degrees()`, `staircase()`, `best_t_for_width()`.
- **`Front`** — the collection of points we will submit, plus `best_k(20)`
  to choose which 20 and `score()` to grade them.
- **`hypervolume()`** — the area calculation from Part 3.

### `hill_climbing.py` — the search

`Operators` (the four moves), `Starts` (random or min-degree), and
`HillClimber` (steps 1–4).

### `generate.py` — making instances

- `toy` — the 12-vertex graph used throughout this document.
- `random` — Erdős–Rényi: every possible edge present with probability `p`.
  No structure; a neutral baseline.
- `planted` — mirrors the *shape* of the real competition instances: several
  low-width components glued by one dense core. The glue is what forces the
  width up, so you know in advance where the difficulty lives. Writes a
  `.meta.json` recording the structure it planted. This is our own
  generator, not the organisers' — it reproduces the published description
  of how the instances are built, not their exact code.

### `validate.py` — the referee

Re-implements the evaluation **from scratch**, deliberately sharing no code
with the search. See Part 8 for exactly what it is and is not.

### `test_correctness.py` — the proof

22 checks that pin the evaluator and the scoring against independent ground
truth. Run it first; if it passes, the numbers everything else prints can be
trusted.

---

# Part 7 — Running it

```bash
# prove the code is correct before trusting any of it
python3 test_correctness.py

# make the toy instance
python3 generate.py --kind toy --out data/toy.gr

# solve it, and verify the fast evaluator matches the obvious one
python3 hill_climbing.py --instance data/toy.gr --seconds 5 \
        --self-check --out out/toy.json

# have the independent referee check the answer
python3 validate.py --instance data/toy.gr --submission out/toy.json
```

The three real competition graphs:

```bash
python3 hill_climbing.py --instance data/small-graph.gr  --seconds 60  --out out/small.json
python3 hill_climbing.py --instance data/medium-graph.gr --seconds 300 --out out/medium.json
python3 hill_climbing.py --instance data/large-graph.gr  --seconds 600 --out out/large.json
```

Make your own:

```bash
python3 generate.py --kind random  --n 200 --p 0.05 --out data/rand200.gr
python3 generate.py --kind planted --n 400 --components 5 --glue-size 40 --out data/planted400.gr
```

Useful flags: `--seconds` (total budget), `--widths` (how many target widths
to climb at), `--seed` (reproducibility), `--start random|min_degree`,
`--self-check`.

---

# Part 8 — Why you can trust the numbers

## 8.1 What the validator is, and what it is not

**It is not ESA's code.** The organisers' official evaluator
(`graph_torso_udp`) is not included in this repository. `validate.py` is an
independent implementation of the **published rules** — the four legality
conditions in Part 1.4, the 500 cap of Part 1.6, and the hypervolume of
Part 3 — written from the specification rather than copied from anywhere.

What that buys you is real but worth stating precisely:

- it shares **no code with the search**, so a bug in `hill_climbing.py`
  cannot make the validator agree with it;
- it was cross-checked against the evaluator used throughout the main
  research project — **28 of 28** random `(perm, t)` pairs and **120 of 120**
  random hypervolume fronts agreed exactly;
- `test_correctness.py` pins it further against a hand-worked example and a
  brute-force area count.

What it does **not** do is guarantee ESA would return the same number. For
that you would run the official evaluator. Treat this as a strong
self-consistency check, not as the competition's own verdict.

## 8.2 The fast evaluator is checked against the obvious one

`torso.py` contains the evaluation twice: `degrees_slow()` with plain sets —
the version you should read — and `degrees()` with bitsets.
`Solution.check()` asserts they produce identical output, and `--self-check`
runs it before searching.

The bitsets are not decoration. Measured, one evaluation costs:

| instance | with sets | with bitsets |
|---|---:|---:|
| small-graph (n = 1357) | 0.04 s | 0.01 s |
| medium-graph (n = 1399) | 3.9 s | 0.15 s |
| large-graph (n = 2426) | 35.4 s | 0.63 s |

Hill climbing needs thousands of evaluations, so the readable version simply
cannot run the big instances — but it can prove the fast one honest on the
small ones.

## 8.3 What `test_correctness.py` checks

| # | check | ground truth used |
|---|---|---|
| 1 | the hand-worked toy example | the Part 2 table, typed in by hand |
| 2 | fast evaluator vs slow evaluator | two independent implementations |
| 3 | hypervolume formula | brute-force count of covered grid squares |
| 4 | every staircase point | re-evaluated directly at that `t` |
| 5 | validator rejects bad input | five kinds of malformed decision vector |

All 22 pass.

---

# Part 9 — What to expect

Indicative runs on a laptop, seconds to a minute each:

| instance | n | budget | score |
|---|---:|---:|---:|
| toy | 12 | 3 s | −121 |
| small-graph | 1357 | 25 s | −1,814,541 |
| medium-graph | 1399 | 22 s | −1,601,788 |
| large-graph | 2426 | 25 s | −4,794,745 |

These are honest plain-hill-climbing numbers with tiny budgets. The point of
this branch is that the method is simple enough to read in one sitting — not
that it is competitive. Longer budgets and more target widths improve all of
them.

One detail worth pointing at on `large-graph`: the climber reliably finds its
widest point at **width 499 with `t = 0`**, keeping the entire graph. That is
not luck, and it cannot be improved on. The graph contains a **500-vertex
clique** — 500 vertices all joined to each other. When you eliminate the
first vertex of a clique of size `k`, its other `k − 1` members are all still
present, so `deg ≥ k − 1` no matter what order you choose. With `k = 500`
that forces a width of at least 499. The simple hill climber finds the
provably optimal answer there.

---

# Glossary

| term | meaning |
|---|---|
| **vertex / edge** | a node of the graph / a connection between two nodes |
| **eliminate** | remove a vertex and join all its surviving neighbours |
| **fill-in** | the new edges created by an elimination |
| **elimination order** (`perm`) | the order in which vertices are eliminated |
| **threshold** (`t`) | how many vertices at the front are removed before measuring |
| **decision vector** | one answer, written as `perm + [t]` — a list of `n + 1` integers |
| **submission** | a list of at most 20 decision vectors |
| **torso** | the vertices from position `t` onwards; size `n − t` |
| **width** | the largest `deg` among the torso steps — minimise |
| **clique** | a set of vertices all joined to each other |
| **dominated** | point A dominates B if A is no worse on both objectives and better on at least one |
| **Pareto front** | the set of points not dominated by any other — the trade-off curve |
| **hypervolume** | the area the points dominate, measured to the corner `(n, n)` |
| **score** | negative hypervolume — more negative is better |
| **heuristic** | a method that finds good answers without proving they are best |
