# Torso Decompositions — the problem, explained from scratch

This branch contains the **simplest possible** working solution to the ESA
SpOC-3 "Torso Decompositions" problem, written so that someone who has never
seen the problem can read it in one sitting and understand all of it.

**Pure Python standard library** — no numpy, no packages, no build step. If
you have `python3`, everything here runs.

```
esa_eval.py            the official scoring, ported from the main project
torso.py               graphs, solutions, the front we submit
hill_climbing.py       the search: four operators, one accept rule
generate.py            make instances
validate.py            check an answer is legal and re-score it
test_correctness.py    22 checks proving the above is right
data/                  3 competition graphs, 7 synthetic, a toy
leaderboard-references/ the winning entry, and a CPU port of it
```

Read this file top to bottom and you will know: what the problem is, how an
answer is written down, how to solve a small instance **by hand**, how the
score is computed, how everything is stored in the code, why the problem is
hard, how the algorithm works, and how to run it.

A word on ambition: this branch is deliberately **primitive**. Plain hill
climbing, plain data structures, no cleverness. It is meant to be read and
understood, and to serve as the honest baseline — not to be competitive.

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

# Part 4 — How things are stored in the code

Part 1 described the problem on paper. This part is the bridge to what you
see on screen: how each idea is actually held in memory.

## 4.1 The map from idea to code

| idea from Part 1 | lives in | as |
|---|---|---|
| the graph | `Graph.adj` | list of sets — `adj[v]` is `v`'s neighbours |
| the graph, again | `Graph.bits` | list of ints used as bitsets (see 4.3) |
| elimination order `perm` | `Solution.perm` | a plain Python list |
| the `deg[]` table of Part 2 | `Solution.degrees()` | list of ints, computed once and cached |
| the staircase of Part 2.1 | `Solution.staircase()` | list of `(width, t)` pairs |
| threshold `t` | *not stored* | see 4.4 |
| the trade-off curve | `Front.points` | dict `width -> (t, perm)` |
| the score of Part 3 | `Front.score()` | one negative integer |
| a decision vector | `Front.decision_vectors()` | list `perm + [t]` |

## 4.2 The graph: a list of sets

The obvious way, and the one to read:

```python
adj = [set() for _ in range(n)]
for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)
```

For the toy instance `adj[0]` is `{1, 2, 11}`. Undirected means every edge is
stored twice — once at each end. Sets make the two questions we ask
constantly both fast and readable: *is `u` a neighbour of `v`?* is
`u in adj[v]`, and *join these vertices together* is `adj[u] |= others`.

## 4.3 The graph again: integers as bitsets

The same graph is also stored as a list of **integers**. A Python integer has
unlimited precision, so it can act as a set of bits of any size: bit number
`u` is `1` exactly when `u` is a neighbour.

```
vertex 0's neighbours are {1, 2, 11}

bit position :  11 10  9  8  7  6  5  4  3  2  1  0
value        :   1  0  0  0  0  0  0  0  0  1  1  0   =  2054
```

That single integer `2054` *is* the set `{1, 2, 11}`. And now set operations
become arithmetic that Python runs in C rather than in a loop:

| set operation | bitset version |
|---|---|
| intersection `A & B` | `a & b` |
| union `A ∪ B` | `a \| b` |
| size `len(A)` | `a.bit_count()` |
| membership `u in A` | `a >> u & 1` |

This is the *only* performance trick in the project, and it buys a lot —
one evaluation of `large-graph` drops from 35 seconds to 0.6. Both versions
of the evaluation are kept side by side in `torso.py` (`degrees_slow` with
sets, `degrees` with bitsets) and `test_correctness.py` proves they always
agree, so the trick never has to be taken on faith.

## 4.4 Why `Solution` does not store `t`

This surprises people reading the code. A `Solution` holds only a
permutation — no threshold. That is deliberate, and it follows directly from
Part 2.1: one permutation already answers the question for *every* `t` at
once. Storing a particular `t` inside the solution would throw away all the
other answers it contains for free.

So `t` appears only at the moment we *report* a result:
`staircase()` returns every `(width, t)` pair, and `decision_vectors()`
writes the chosen `t` onto the end of the permutation.

## 4.5 What `Front` accumulates

`Front` is a dict keyed by width, holding the best `t` seen at that width and
the permutation that achieved it:

```
points = {
     0: (11, [7, 1, 8, ...]),
     1: (10, [7, 1, 8, ...]),
     2: ( 2, [7, 1, 8, ...]),
     3: ( 0, [7, 1, 8, ...]),
}
```

Keying by width makes the update rule trivial — a new point at width `w`
replaces the old one only if its `t` is smaller. Every permutation the
climber evaluates is offered to the front, so good points are kept even when
the move that produced them was rejected.

`pareto()` then drops dominated entries, `best_k(20)` chooses which 20 to
submit, and `score()` grades them.

---

# Part 5 — Why this is hard

## 5.1 There are too many orderings to try

| instance | n | number of orderings (`n!`) |
|---|---:|---|
| toy | 12 | 479,001,600 |
| small-graph | 1357 | a number with 3,600+ digits |
| large-graph | 2426 | a number with 7,000+ digits |

For the toy you could brute-force it in a few minutes. For anything real,
the number of orderings exceeds the number of atoms in the observable
universe by thousands of orders of magnitude. There is no formula for the
best one either — the underlying problem (finding an elimination order of
minimum width) is **NP-hard**, so nobody has an efficient exact method and
almost certainly nobody will.

## 5.2 One small change can cascade

The objective is not smooth. Swapping two vertices does not nudge the answer
slightly; it can change the fill-in produced at that step, which changes
which vertices are adjacent later, which changes the fill-in *there*, and so
on to the end of the ordering.

We saw this in Part 2 at step 1: eliminating `5` created the edge `4–6`, so
by the time `4` was eliminated it had a neighbour it never originally had.
Move `5` elsewhere in the order and that edge is created at a different
moment — or not at all — and every later step can differ.

This is what "rugged landscape" means, and it is why you cannot reason your
way to a good ordering. You have to search.

## 5.3 The 500 cap makes cliffs

Most optimisation problems degrade gracefully: a slightly worse answer
scores slightly worse. Not here. One elimination step at width 501 makes the
**entire** answer void, no matter how good the other 2,425 steps were.

So the search space has cliffs in it. A single swap can take a perfectly
good ordering and make it worth nothing. This is also why a random starting
order is useless on the big instances — it lands off the cliff immediately,
every neighbouring order is also off the cliff, and hill climbing has no
signal to follow.

## 5.4 Two objectives, not one

Hill climbing compares two things and keeps the better. That only works when
"better" is a single number. Here there are two, and they conflict — so
there is no single best answer to climb towards, only a curve of compromises
(Part 1.5).

Part 6.1 explains the trick we use to get around this without adding any
machinery.

## 5.5 And it plateaus

Even with all that, the most common thing the climber runs into is simply
**nothing happening**. The width is an integer, and usually a `max` over
thousands of steps. Most single swaps do not change that maximum at all, so
most candidate moves score exactly the same as the current one, are not
strictly better, and get thrown away.

You can watch this in the output: the climber does hundreds of thousands of
evaluations and accepts perhaps a dozen. That is not a bug — it is what
plain hill climbing on a rugged plateau looks like, and it is the honest
baseline that more sophisticated methods have to beat.

---

# Part 6 — The algorithm: hill climbing

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

## 6.1 Handling two objectives with a single-objective method

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

## 6.2 The four moves

From `Operators` in `hill_climbing.py`. Each takes a permutation and returns
a new one — never modifying the original, because a rejected change must be
discarded cleanly.

| operator | what it does |
|---|---|
| `swap_neighbours` | swap two adjacent positions |
| `swap_any` | swap two positions anywhere |
| `move_vertex` | remove one vertex and reinsert it elsewhere |
| `reverse_segment` | reverse a short run of positions |

## 6.3 Where it starts

A random order is a terrible start: on the bigger graphs it blows past the
500 cap immediately (Part 5.3), so every candidate is void and the climber
has nothing to compare. The default is therefore **min-degree**: repeatedly
eliminate whichever vertex currently has the fewest surviving neighbours. It
is the classic textbook heuristic for orderings like this and gives the
climber a legal, decent starting point.

It is written with plain sets, exactly as you would say it out loud. The one
concession to speed is keeping a `degree` table and updating only the
vertices that changed, rather than recounting every vertex from scratch each
step — the difference between O(n²) and O(n³). Measured: toy 0.00 s,
small 0.09 s, medium 0.72 s, large **5.25 s**. Slow, but it runs once per
climb and it stays readable.

## 6.4 What the climber actually finds

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

# Part 7 — The files

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
with the search. See Part 9 for exactly what it is and is not.

### `test_correctness.py` — the proof

22 checks that pin the evaluator and the scoring against independent ground
truth. Run it first; if it passes, the numbers everything else prints can be
trusted.

---

# Part 8 — Running it

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

# Part 9 — Why you can trust the numbers

## 9.1 Where the scoring comes from

All scoring goes through `esa_eval.py`, which is a verbatim port of `core.py`
from the main research project (`test/gbdt-novelty`). That file is documented
as matching ESA's `graph_torso_udp._perm2fitness`, and it is the code every
leaderboard-verified score in that project was computed with. Ported across:
`MAX_TW = 500`, the bitset evaluator, `hypervolume_2d`, and
`top_k_by_hv_contribution` — the exact dynamic program that picks the best 20
points.

The port was checked against the original, not assumed: identical `MAX_TW`,
identical bitsets, and agreement on 34/34 random `(perm, t)` pairs, 160/160
random hypervolumes, and 50/50 HSSP subsets.

One caveat, stated plainly so it never becomes an overclaim: this is not
literally ESA's own source file, which was never published with the challenge
materials. It is the evaluator the main project validated against the live
leaderboard over several hundred submissions. If the official file turns up,
dropping it into `esa_eval.py` is the only change needed.

`validate.py` scores with `esa_eval.py` and *also* re-walks every vector with
a plain set-based implementation written separately. If those two ever
disagree it says so, rather than quietly trusting one.

## 9.2 The fast evaluator is checked against the obvious one

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

## 9.3 What `test_correctness.py` checks

| # | check | ground truth used |
|---|---|---|
| 1 | the hand-worked toy example | the Part 2 table, typed in by hand |
| 2 | fast evaluator vs slow evaluator | two independent implementations |
| 3 | hypervolume formula | brute-force count of covered grid squares |
| 4 | every staircase point | re-evaluated directly at that `t` |
| 5 | validator rejects bad input | five kinds of malformed decision vector |

All 22 pass.

---

# Part 10 — The instances, and comparing against the winners

## 10.1 The seven extra instances

`data/` holds the three competition graphs plus seven synthetic ones,
`synth-1` to `synth-7`: three in the small family, two medium, two large.
They exist so a solver can be tested on something it was not tuned on.

The obvious way to make them would be to keep the degree sequence and rewire
the edges — a double edge swap preserves every vertex's degree exactly. That
was the first attempt, and it does not work:

| template | its width | after degree-preserving rewiring |
|---|---:|---:|
| small-graph | **20** | 274 |
| medium-graph | **276** | 621 |
| large-graph | 499 | 499 |

small-graph and medium-graph are *structured* graphs that happen to have a
particular degree sequence. Rewiring keeps the degrees and throws the
structure away, leaving something an order of magnitude harder — a different
problem wearing the same costume. Only large-graph survived, because its
difficulty lives in three big cliques and those were frozen.

So each family is generated the way its template is built, and checked
against the template's measured width rather than its degree list:

| family | construction | count |
|---|---|---|
| small | grid + pendant vertices — sparse, triangle-free, small separators | 3 |
| medium | dense blocks joined by a controlled number of cross edges | 2 |
| large | the template's cliques kept, vertices relabelled, periphery rewired | 2 |

Large keeps its cliques deliberately. A K500 forces width ≥ 499 whatever you
do, so an instance in that family without it would be a different problem.
Relabelling gives the cliques fresh membership, so a twin is not the original
with a few edges moved.

## 10.2 Do they behave like the originals?

| instance | family | n | edges | min-degree width | hill climbing, 12s |
|---|---|---:|---:|---:|---:|
| small-graph | — *template* | 1357 | 2280 | 20 | −1,814,527 |
| **synth-1** | small | 1357 | 2282 | 18 | −1,818,560 |
| **synth-2** | small | 1357 | 2282 | 18 | −1,817,317 |
| **synth-3** | small | 1357 | 2282 | 16 | −1,819,889 |
| medium-graph | — *template* | 1399 | 13799 | 276 | −1,607,564 |
| **synth-4** | medium | 1399 | 14056 | 282 | −1,600,724 |
| **synth-5** | medium | 1399 | 14055 | 328 | −1,608,658 |
| large-graph | — *template* | 2426 | 253895 | 499 | −4,794,427 |
| **synth-6** | large | 2426 | 253895 | 499 | −4,795,089 |
| **synth-7** | large | 2426 | 253895 | 499 | −4,795,246 |

Each `synth-N.gr` has a `synth-N.gr.meta.json` beside it recording which
family it belongs to, the seed, and its measured width.

Scores land within 0.4% of their templates on small and medium, and within
0.02% on large. That is the property that matters: the extras are a fair
test, not an easier or harder one.

Make more with:

```bash
python3 generate.py --family small  --seed 11 --out data/synth-8.gr
python3 generate.py --family medium --seed 12 --out data/synth-9.gr
python3 generate.py --family large  --seed 13 --out data/synth-10.gr
```

## 10.3 Comparing against the leaderboard entries

`leaderboard-references/` holds the winning entry and a CPU reimplementation
of it. The original needs an NVIDIA GPU, a compiled CUDA kernel and PyTorch,
and has the three official graph sizes hardcoded — so it cannot run here, and
it is vendored for reference and attribution only. **It ships with no licence
file**; see that folder's README before publishing this repository anywhere.

Its method is worth understanding because it is so different from hill
climbing. A candidate is not a permutation but a *weight vector over
per-vertex features* — degree profile, Laplacian eigenvector coordinates,
and polynomial combinations of those — and the ordering is read off by
sorting `w · features`. Evolution searches the space of scoring rules rather
than the space of orderings.

`neuroevo_cpu.py` is that algorithm without the GPU, accepting any instance
and scoring through the same `esa_eval.py`, so the numbers are comparable:

```bash
cd leaderboard-references
python3 neuroevo_cpu.py --instance ../data/small-graph.gr --seconds 60
```

small-graph, ~30 s each on one laptop core:

| solver | score |
|---|---:|
| `hill_climbing.py` | −1,814,521 |
| `neuroevo_cpu.py` | −1,798,068 |

Plain hill climbing wins at this budget. That is the expected result and not
a claim about the original entry: the learned-scoring-rule approach needs a
large population to pay off, and a large population is exactly what the GPU
was for. Any comparison here is a statement about **the methods at equal CPU
budget**.

## 10.4 Starting points, and why they decide the table

Running four solvers for the same number of seconds is not yet a fair test.
They also have to *start* in comparable places, and on this problem the
starting point turns out to matter more than the search.

The four split into two kinds:

- **permutation-space** -- `hill_climbing.py` and `hri_lns.py` search
  orderings directly, so they need an ordering to begin from
- **weight-space** -- `neuroevo_cpu.py` and `cmaes.py` search weight vectors
  and read the ordering off `argsort(w . features)`

The obvious "fair" choice is to start everything from a random ordering. On
these instances that does not work, and the reason is the `MAX_TW = 500`
cap. Take one random permutation of each instance and count how many legal
points its staircase yields:

| instance | n | points from a random order | points from min-degree |
|---|---:|---:|---:|
| small-graph | 1357 | 117 | 22 |
| medium-graph | 1399 | **0** | 285 |
| large-graph | 2426 | **0** | 500 |
| synth-1 | 1357 | 250 | 18 |
| synth-2 | 1357 | 297 | 19 |
| synth-3 | 1357 | 295 | 19 |
| synth-4 | 1399 | **0** | 312 |
| synth-5 | 1399 | **0** | 349 |
| synth-6 | 2426 | **0** | 500 |
| synth-7 | 2426 | **0** | 500 |

On six of the ten, a random ordering scores **nothing at all** -- every
prefix busts the 500 cap, so there is not one legal point to stand on. And
a solver with no legal point has nothing to climb: every candidate costs
`n`, no candidate is ever an improvement, and the search degenerates into a
random walk. Measured on medium-graph, 25 s from a random start: Team HRI's
LNS ran 193 iterations and accepted **zero** of them, final score **0**.

The weight-space solvers never face this, because they never see a random
ordering. Their *first* candidate is already sorted by a degree-and-spectral
score, which is a constructive heuristic hiding inside the representation.
So "everyone starts random" would not equalise anything -- it would zero out
the two permutation solvers on six instances while the other two carried on
unaffected.

What the benchmark does instead: both permutation-space solvers start from
**the same min-degree construction**, computed fresh from the graph inside
each run. Nothing is carried over from a previous run, from another solver,
or from a saved solution -- each run begins from the graph and nothing else.

```bash
python3 benchmark.py --start min_degree      # the default: same start for both
python3 benchmark.py --start random          # literal random; void on 6 of 10
```

This matters enough to state plainly, because it is easy to fool yourself
here. Earlier versions of `benchmark.py` had hill climbing warm-starting
from min-degree while HRI started random, as HRI's paper specifies. On
synth-1 at 8 s that read HRI -1,543,392 against hill climbing -1,818,617,
which looks like a rout. Give HRI the same start and it reads -1,818,667
against -1,819,901 -- the same result to within a rounding error. **Almost
the whole apparent gap was the starting heuristic, not the search.**

There was a subtler version of the same trap inside `hill_climbing.py`.
`target_widths()` probes the graph to decide which widths to aim at, and
that probe used min-degree *regardless of the requested start* -- and added
it to the front. So `--start random` was quietly handing hill climbing a
free min-degree solution and reporting -1,607,522 on medium-graph for a run
that had accepted zero moves. The probe now follows whatever start it was
asked for, and an honest from-scratch run on medium-graph reports what it
actually earned, which is 0.

The general lesson is worth more than the table: **when a comparison shows a
large gap, check the starting conditions before believing the algorithm
caused it.**

## 10.5 One more confound: CPU cores

Equal wall-clock is not equal computation. Watching the benchmark run, the
process sat at **929% CPU** -- about nine cores busy. Hill climbing is pure
Python and uses exactly one. So the competitors are not merely matched on
time, they are being handed roughly nine times the processor.

It is worth knowing which way that cuts, so both solvers were measured on
small-graph at 30 s, once unrestricted and once pinned with
`OMP_NUM_THREADS=1`:

| solver | threads free | pinned to 1 core |
|---|---:|---:|
| Spacekangaroos | 1510 evals, -1,796,719 | **1997 evals, -1,800,321** |
| fast-cma-es | **1687 evals, -1,801,407** | 1345 evals, -1,794,332 |
| hill climbing | 9590 evals, -1,814,541 | 9778 evals, -1,814,542 |

Three separate things show up here.

**Hill climbing is unaffected**, as it must be -- pure Python, one thread
either way. The 2% difference is seed noise.

**fast-cma-es genuinely uses the cores.** Pinning it costs 20% of its
evaluations. It is a package built for parallel optimisation and throttling
it would misrepresent the method.

**Spacekangaroos is *hurt* by them.** Pinning it *gains* 32% more
evaluations and a better score. Its inner loop is a 104x1357 matrix-vector
product -- far too small to parallelise -- so the BLAS threads spend their
time spin-waiting instead of working. That is a configuration pathology, not
a flaw in the method, and it means this column is a slight underestimate.

The benchmark therefore lets every library thread as its authors intended
and holds wall-clock equal. That is the standard protocol, and the important
point for reading the table is the direction of the bias: **the handicap
runs against hill climbing, not for it.** Where hill climbing wins a row, it
wins it on one core against opponents using nine.

---

# Part 11 — What to expect

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
