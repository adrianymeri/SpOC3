# Torso Decompositions — plain hill climbing

A deliberately simple, self-contained implementation of the SpOC-3 Torso
Decompositions problem and a plain hill climber for it.

Four files, about 1,100 lines including comments, **pure Python standard
library** — no numpy, no external packages, no build step. If you have
`python3` you can run everything here.

```
torso.py           the problem: graph, solution, evaluation, scoring
hill_climbing.py   the search: four operators, one accept rule
generate.py        make instances: toy, random, planted
validate.py        check a submission is legal and rescore it
data/              the three competition graphs, plus generated ones
```

---

## The problem in one page

You are given an undirected graph on `n` vertices. You choose:

- **`perm`** — an elimination *order*: a permutation of `0 .. n-1`
- **`t`** — a *threshold*: how many vertices at the front of `perm` are
  eliminated before we start measuring

**Eliminating** a vertex means removing it and joining all of its
not-yet-eliminated neighbours into a clique. That added edge set is called
*fill-in*, and it is why eliminating vertices keeps making the surviving
graph denser.

At each step `i` record `deg[i]` — how many not-yet-eliminated neighbours
`perm[i]` had when it was eliminated. Then:

| objective | meaning | direction |
|---|---|---|
| `width = max(deg[i] for i >= t)` | how wide the torso got | minimise |
| `t` | how many vertices we had to remove | minimise |

Smaller `t` means we removed **fewer** vertices, so the surviving torso
(`n - t` vertices) is **larger**. That is the trade-off: allow a larger
width and you can keep more of the graph.

One hard rule: if `deg[i] > 500` at **any** step — including the eliminated
head, before `t` — the solution is void.

### The trick that keeps the code short

For one fixed `perm` you do **not** need a separate evaluation per `t`.
Compute `deg[]` once, then for every `t`

```
width(t) = max(deg[t], deg[t+1], ..., deg[n-1])
```

which is a running maximum from the back. So **one pass over one
permutation yields a whole staircase of (width, t) points.** That is
`Solution.staircase()`, and it is why the search can be as simple as it is.

### Scoring

The competition keeps at most **20** of your points and measures the area
they dominate relative to the corner `(n, n)`. The official score is the
**negative** of that area, so **more negative is better**.

---

## The algorithm

Hill climbing, in full:

1. start from some solution `S`
2. make a small random change to a **copy** of it → `R`
3. if `R` is better than `S`, keep `R`; otherwise throw it away
4. repeat until out of time

No population, no temperature, no memory, no learning. In
`HillClimber.climb` this is six lines.

**Handling two objectives.** We do the simplest thing that produces a real
front: pick a target width `W` and hill-climb to minimise `t` at that
width. Each climb is then an ordinary single-objective climb — one number
going down. Run one climb per target width and collect the results.

Because of the staircase trick, every permutation the climber looks at
donates points at *every* width to the shared front, even the ones it
rejects. Choosing a target width just decides which number is being pushed
down.

**Operators** (`Operators.ALL`): `swap_neighbours`, `swap_any`,
`move_vertex`, `reverse_segment`.

**Starting point** (`Starts`): `min_degree` (default) or `random`. On the
larger graphs a random order breaks the 500 cap immediately and the climber
has nothing to work with, so min-degree is the sensible default.

---

## Running it

```bash
# 1. make a tiny instance you can check by hand
python3 generate.py --kind toy --out data/toy.gr

# 2. climb, and prove the fast evaluator matches the obvious one
python3 hill_climbing.py --instance data/toy.gr --seconds 5 \
        --self-check --out out/toy.json

# 3. check the answer with code that shares nothing with the search
python3 validate.py --instance data/toy.gr --submission out/toy.json
```

The three competition graphs are in `data/`:

```bash
python3 hill_climbing.py --instance data/small-graph.gr  --seconds 60  --out out/small.json
python3 hill_climbing.py --instance data/medium-graph.gr --seconds 300 --out out/medium.json
python3 hill_climbing.py --instance data/large-graph.gr  --seconds 600 --out out/large.json
```

Generate your own:

```bash
python3 generate.py --kind random  --n 200 --p 0.05 --out data/rand200.gr
python3 generate.py --kind planted --n 400 --components 5 --glue-size 40 --out data/planted400.gr
```

`planted` mirrors the structure of the real competition instances — several
low-width components glued by one dense core. The glue is what forces the
width up, so you know in advance where the difficulty lives; the generator
writes a `.meta.json` recording the structure it planted.

---

## Two things worth demonstrating

**The fast evaluator is honest.** `torso.py` contains the evaluation twice:
`degrees_slow()` with plain sets (the obvious version — read this one) and
`degrees()` with Python-int bitsets. `Solution.check()` asserts they agree,
and `--self-check` runs it before the search starts.

Bitsets are not an optimisation for its own sake. Measured per evaluation:

| instance | sets | bitsets |
|---|---:|---:|
| small-graph (n=1357) | 0.04 s | 0.01 s |
| medium-graph (n=1399) | 3.9 s | 0.15 s |
| large-graph (n=2426) | 35.4 s | 0.63 s |

Hill climbing needs thousands of evaluations, so the set version simply
cannot run the big instances.

**The validator is independent.** `validate.py` deliberately re-implements
the evaluation from scratch with sets rather than importing anything from
the search. A validator that shared the search's code would believe the
search's bugs. It checks each vector is a genuine permutation with an
in-range threshold and no step over the cap, checks the submission has at
most 20 points with no duplicates or dominated entries, and recomputes the
score from the graph. Exit code 0 means everything passed.

---

## What to expect

Indicative runs on a laptop, a few seconds to a minute each:

| instance | n | budget | score |
|---|---:|---:|---:|
| toy | 12 | 3 s | −121 |
| small-graph | 1357 | 25 s | −1,814,541 |
| medium-graph | 1399 | 22 s | −1,601,788 |
| large-graph | 2426 | 25 s | −4,794,745 |

These are honest hill-climbing numbers with tiny budgets, and they are
meant as a **baseline** — the point of this branch is that the method is
simple enough to read in one sitting, not that it is competitive. Longer
budgets and more target widths improve all of them.

One detail worth pointing at on `large-graph`: the climber reliably finds
its widest point at **width 499 with `t = 0`** (the whole graph as torso).
That is not luck. The graph contains a 500-vertex clique, and a clique of
size `k` forces elimination width at least `k − 1`, so 499 is provably the
best any method can do there.
