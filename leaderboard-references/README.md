# Leaderboard references

Solutions from the ESA SpOC-3 Torso Decompositions leaderboard, kept here so
hill climbing can be compared against them on the same instances with the
same scoring.

```
cuda-torso/        Spacekangaroos' winning entry, vendored as published
neuroevo_cpu.py    CPU reimplementation of it  -> column "Spacekangaroos"
hri_lns.py         Team HRI's LNS, from their paper  -> column "Team HRI"
cmaes.py           the fast-cma-es approach  -> column "fast-cma-es"
```

All three run on any instance, score through `../esa_eval.py`, and are
driven together by `../benchmark.py`, which produces the comparison table
one column per team.

---

## cuda-torso — the winning entry

GPU neuro-evolution, by the team that won the challenge.
Upstream: <https://github.com/…/cuda-torso> (see `cuda-torso/README.md`).

**You cannot run this as-is on a laptop.** It needs an NVIDIA GPU, a CUDA
kernel compiled from `libeval.cu`, PyTorch, SciPy, and it has the three
official graph sizes hardcoded in `GRAPH_SIZES`. It is included for
reference and attribution, not because it will run here.

### How the method works

A candidate is *not* a permutation. It is a weight vector over per-vertex
features, and the ordering is read off by sorting:

```
score = w · features[v]        for every vertex v
perm  = argsort(score)
```

So the search is over *scoring rules*, not orderings — a much smaller and
smoother space. The features describe each vertex's position in the graph:

- **local degree profile** (5): its degree, and the min / max / mean / stdev
  of its neighbours' degrees
- **Laplacian positional encoding**: the first *k* eigenvectors of the
  normalised Laplacian, giving each vertex a spectral coordinate
- **polynomial expansion**: every feature squared, plus every pairwise
  product
- everything standardised to mean 0, stdev 1

Evolution is an elite scheme with two mutation operators: Gaussian noise on
a random subset of weights, and a Cosyne-style permutation that copies
individual weights between population members.

### Licensing

`cuda-torso/` **ships with no licence file.** Its copyright therefore rests
with its authors and no permission to redistribute has been granted in
writing. It is included here for academic comparison with attribution. If
this repository is ever published or redistributed, get the authors'
permission first or replace this directory with a link.

---

## neuroevo_cpu.py — the CPU port

The same algorithm, without the GPU, accepting any `.gr` instance.

```bash
python3 neuroevo_cpu.py --instance ../data/small-graph.gr --seconds 60
python3 neuroevo_cpu.py --instance ../data/synth-1.gr --seconds 60 --out out/ne1.json
```

Scoring goes through `../esa_eval.py` — the same evaluator hill climbing
uses — so the two are directly comparable, and `../validate.py` will check
its output exactly as it checks hill climbing's.

**What is faithful:** the feature construction, the `argsort(w · features)`
representation, the elite scheme, Gaussian mutation, and Cosyne permutation.

**What differs, and it matters:**

- CPU and numpy instead of CUDA and PyTorch. The original evolves thousands
  of candidates per generation on a GPU; this manages tens. **It is not
  competitive with the original and is not meant to be** — it is the same
  algorithm at a scale a laptop can run.
- Thresholds are chosen the same way `hill_climbing.py` does (every
  candidate donates its whole staircase to a shared front, cut to 20 points
  by the exact HSSP) rather than by the original's own routine.
- Without numpy the spectral features are skipped and only the degree
  profile is used. The program says so in its output when that happens.

Any comparison you draw from it is a statement about **the method at equal
CPU budget**, not about the original entry's leaderboard performance.

### Indicative numbers

small-graph, ~30 s each on one laptop core:

| solver | score |
|---|---:|
| `hill_climbing.py` | −1,814,521 |
| `neuroevo_cpu.py` | −1,798,068 |

Plain hill climbing wins at this budget, which is the expected result: the
learned-scoring-rule approach needs a large population to pay off, and a
large population is exactly what the GPU was for.

---

## hri_lns.py — Team HRI, second place

Multi-objective Large Neighbourhood Search, written from their paper
(`spoc_2023_team_hri.pdf`). Destroy-and-repair rather than mutate-and-test:

- **neighbour destroy** — pull a seed vertex and its neighbours out of the
  ordering, leaving a large structured hole
- **balanced repair** — put each one back at the **median position of its
  already-placed neighbours**, the balanced-insertion rule of Biedl et al.,
  *Discrete Applied Mathematics* 148 (2005), §5. Median placement leaves
  roughly as many neighbours before a vertex as after, which is what holds
  the width down.
- when that stalls, it swaps to a small random destroy-and-repair and back

**No source was published**, so this column is a reimplementation from the
paper, not their code.

The paper starts it from a random ordering. `../benchmark.py` does not: it
gives HRI the same min-degree start hill climbing gets, because a random
start is not a handicap here so much as a disqualification — on six of the
ten instances a random permutation busts the 500 cap at every prefix, so the
LNS has no legal point to work from and scores 0. Equalising the start is
what makes the column a comparison of *searches*. See `../README.md` §10.4
for the measurements.

---

## cmaes.py — fast-cma-es

Treats the problem as continuous optimisation. Same decode as
Spacekangaroos — `perm = argsort(features @ x)` — but the optimiser is a
separable CMA-ES, which adapts a diagonal covariance model of where good
weight vectors live instead of mutating elites. Ported from the main
project's implementation. Needs numpy.
