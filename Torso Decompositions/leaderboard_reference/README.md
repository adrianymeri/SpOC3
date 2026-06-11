# Leaderboard reference solutions (read-only study copies)

Two top ESA SpOC-3 leaderboard solutions, kept for reference until their ideas
are incorporated. See `docs/FUTURE.md` § 1g for the full analysis.

## cuda-torso-main/  (extracted)
GPU neuro-evolution over a **spectral node-scoring policy**. The decision is a
weight vector; `argsort(features @ weights)` is the elimination order. Features
= degree profile + Laplacian eigenvectors + polynomial expansion. Evolved by
CoSyNE with a custom CUDA fill-in kernel (`libeval.cu`), 1024 candidates/gen.
Key files: `run.py` (the whole method), `libeval.cu` (GPU evaluator).

## fast-cma-es-master.zip  (archive)
Dietmar Wolz's gradient-free toolkit: CMA-ES, BiteOpt, CR-FM-NES, MO-DE, with
massively parallel retry. The engine a top competitor used; the recipe is the
same continuous-vector -> argsort -> black-box-optimiser decode.

## Our CPU reproduction
`algorithms/continuous/cmaes_torso.py` replicates the paradigm on CPU
(sep-CMA-ES over the spectral policy; one fill-in pass -> whole front via
suffix-max). First runs already beat the entire permutation-space project.
