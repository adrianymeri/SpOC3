# OLA 2027 campaign

Evolutionary and neuroevolutionary algorithms for SpOC-3 torso decompositions,
run against the ESA leaderboard methods. Everything runs on the server and
nothing on the Mac; the Mac is only for reading results.

## What runs

Nine methods are compared:

| column | method | kind |
|---|---|---|
| EA-1 MEMETIC-PERM | memetic island GA on elimination orderings (`methods/memetic.py`) | ours |
| EA-2 TORSO-QD | width-indexed quality-diversity EA over head sets (`methods/torso_qd.py`) | ours |
| EA-3 NSGA-II | textbook NSGA-II on (ordering, threshold) (`methods/nsga2.py`) | ours |
| NE-1 ANE | adaptive neural decoder, one generalist, sep-CMA-ES (`methods/neuro.py`) | ours |
| NE-2 AQD-NE | adaptive neural decoders, one specialist per width, QD + emitters (`methods/neuro.py`) | ours |
| Team HRI | multi-objective LNS (`methods/baselines.py`) | baseline |
| Hill Climbing | this group's AlbShkenca method (`methods/baselines.py`) | baseline |
| fcmaes | CMA-ES stand-in on static features (`methods/baselines.py`) | baseline |
| Spacekangaroos | cuda-torso on the V100, as published (`gpu/cuda_torso.py`) | baseline |

How each one works, and every change made to a baseline, is written at the
top of its file.

### Instances

There are 33: the 3 official graphs and 30 synthetic ones.

| family | instances |
|---|---|
| small | small-graph, synth-1..3, synth-8..14 |
| medium | medium-graph, synth-4, 5, synth-15..22 |
| large | large-graph, synth-6, 7, synth-23..30 |

### Fairness

- All CPU methods evaluate through one C engine, `engine/torso_engine.c`.
  `tests/test_engine.py` checks it bit for bit against `esa_eval.py`.
- Every evaluated ordering's whole staircase is banked in the same archive.
- Every submission is the archive's exact best 20 points.
- Budgets are CPU time, with one physical core per run.

## Where and how long (server: 2× Xeon Gold 6130, V100)

| resource | runs | budget |
|---|---|---|
| core 0 | supervisor, aggregator, cuda-torso's host process | – |
| 24 cores | 3 official instances × 8 CPU methods, each pinned for the whole campaign | until the end |
| 7 cores | 30 synthetic instances × 8 CPU methods = 240 runs, in rounds | 1, 2, 4, 8, 12, 16, 20 ... h each |
| V100 | cuda-torso on all 33 instances, in rounds; official instances get 3× the time | 0.5, 1, 2, 4, 8 ... h each |

**Rounds.** No synthetic run gets its next level until every synthetic run
has finished the current one. So a fair, equal-budget table over all 30
synthetic instances exists at every level, whenever the campaign is stopped.

**End.** The campaign ends on 15 Nov 2026 at 23:59 server time (`config.END`),
or earlier with `stop.sh`.

## One-time setup on the server

Use a clone of its own, so no other checkout of the repository is touched:

```
git clone <repo> ~/SpOC3-ola2027 && cd ~/SpOC3-ola2027 && git checkout ola2027-evolutionary
campaign/install.sh          # checks, builds, tests, sets up the GPU, installs the cron entries
campaign/start.sh            # starts the supervisor
```

Everything the campaign writes stays inside that clone. The only things
outside it are the two cron lines and the usual pip/CuPy download caches
under `~/.cache` and `~/.cupy`.

`install.sh` does the following:

- builds the engine for the server's CPU (AVX-512);
- runs the engine test;
- caches each instance's structure;
- prepares cuda-torso with `gpu/setup.sh` (below);
- adds two cron lines: start at boot, and a check every 5 minutes.

`campaign/install.sh --no-gpu` skips cuda-torso. `campaign/gpu/setup.sh`
adds it later, also while the campaign is running:

- **Python:** the campaign's own Python if its torch runs on the GPU;
  otherwise `campaign/venv-gpu`, inheriting that torch and adding scipy, or
  with PyTorch's CUDA 12.6 build (the last one with code for Volta GPUs such
  as the V100).
- **Kernel:** built for the GPU's own architecture (`nvcc -arch=sm_XY`, from
  the PATH, then `/usr/local/cuda`), with CuPy/NVRTC as the fallback. CUDA 13
  compilers cannot build for Volta.
- **Self-check:** the build is kept only if its results equal an exact CPU
  emulation of the kernel (including its +1 self-loop) on small-graph and
  large-graph. Every cuda-torso session repeats this check before it
  searches. The published code never checks for CUDA errors, so a kernel
  that does not run would otherwise go unnoticed.
- **Smoke test:** about 4 minutes, including a resume.
- **Enable:** on success the supervisor is restarted with the GPU enabled.
  Its runs checkpoint and continue. On failure `state/NO_GPU` holds the
  reason; details are in `state/gpu_setup.log` and `state/gpu_smoke.log`.

## Daily

```
campaign/status.sh             # terminal summary (refreshed every 10 min)
campaign/status.sh --refresh   # re-aggregate now
campaign/snapshot.sh           # -> campaign/snapshots/ola2027-YYYYmmdd-HHMM.tgz
```

Copy the snapshot to the Mac into `SpOC3/campaign-results/` and unpack it.
Open `results/dashboard.html` in a browser. Tell Claude a new snapshot is
there, and it can read it.

`results/` contains:

| file | contents |
|---|---|
| `STATUS.txt` | the summary; alerts at the bottom |
| `dashboard.html` | official table with virtual ESA rank and 24 h change, run curves, fair synthetic table |
| `summary.csv` | one row per run: score, budget, 24 h change, last improvement, endpoint, checks |
| `ola2027_table.csv` | the OLA2027 tab's layout |
| `fair_synthetic.csv` | every synthetic run at the last level they all completed |
| `anytime_official.csv` | score over budget, official instances |

**Checks.**

- Every submission is re-scored with the engine on every pass.
- `esa_eval.py` confirms a few per pass, oldest first. `stop.sh` confirms all of them.

## Stop, resume, reboot

- `campaign/stop.sh` stops cleanly. Every run checkpoints, a final aggregation runs with `esa_eval` checks, and `state/STOP` is created. Cron then leaves the campaign alone.
- `campaign/start.sh --resume` continues every run from its checkpoint.
- **After a reboot or a crash**, cron starts the supervisor again, and every run resumes from its last checkpoint. Checkpoints are written every 10 minutes and on SIGTERM, atomically, with the previous one kept. At most about 10 minutes of a run's work is lost.
- **A run that dies** is restarted within seconds.
- **A run whose heartbeat is older than 15 minutes** is killed and restarted.
- **A run that crashes more than 6 times in an hour** is parked for an hour and listed in STATUS.txt.

### Without cron

Use a systemd user unit, `~/.config/systemd/user/ola2027.service`:

```
[Unit]
Description=OLA 2027 campaign supervisor
[Service]
WorkingDirectory=%h/SpOC3-ola2027
ExecStart=/usr/bin/python3 %h/SpOC3-ola2027/campaign/supervisor.py
Restart=always
[Install]
WantedBy=default.target
```

Then run:

```
systemctl --user enable --now ola2027
loginctl enable-linger $USER
```

## Files

```
campaign/
  config.py            instances, methods, rounds, end date
  supervisor.py        scheduling, pinning, restarts, rounds, stop
  run_method.py        one CPU run (resumes automatically)
  common.py            budget clock, checkpoints, logs, heartbeat
  torso_engine.py      bindings for engine/libtorso.so
  engine/              the C engine
  methods/             the eight CPU methods
  gpu/                 cuda-torso (kernel with the int32 fix only), setup.sh, smoke test
  structure.py         twins, cliques, blocks, cores, spectral coordinates (cached)
  seeding.py           structure-aware starting orderings (designed methods)
  aggregate.py         results, checks, dashboard
  tests/               engine test, end-to-end campaign test, all-method smoke test
  runs/  results/  state/  snapshots/  cache/   (created at run time, not in git)
```

## Tests

```
python3 campaign/tests/test_engine.py --full   # engine vs esa_eval, all 33 instances
campaign/tests/test_campaign.sh                # miniature campaign: crashes, orphans, rounds, stop
campaign/tests/test_power_loss.sh              # power cut: everything killed at once, a torn checkpoint, restart
campaign/tests/smoke_all.sh 8                  # every CPU method on every instance, with a resume
SMOKE_METHODS="ea2 hri" campaign/tests/smoke_all.sh 8   # the same for some methods only
campaign/gpu/smoke_test.sh                     # cuda-torso on the GPU
python3 campaign/gpu/cuda_torso.py --instance data/small-graph.gr --run-dir /tmp/chk --self-check-only
                                               # the kernel against the CPU emulation
```
