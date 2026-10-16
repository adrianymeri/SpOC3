"""
common.py -- the run framework every campaign method plugs into.

A run is one (method, instance) pair living in its own directory:

    runs/<instance>/<method>/
        ckpt.pkl        latest checkpoint (atomic: temp file + fsync + rename)
        ckpt.prev.pkl   the one before it, used if the latest is unreadable
        best.json       the current best 20-point submission (ESA format)
        anytime.csv     budget_s, wall time, score, evaluations, ... per improvement
        status.json     what the supervisor and the dashboard read
        heartbeat       touched every 30 s while the run is alive
        log.txt         human-readable progress

The budget is the process CPU time accumulated over every session of the run
(persisted in the checkpoint), so a run that is stopped and resumed -- by the
round scheduler, a reboot or a crash -- continues its clock where its last
checkpoint left it. Work done after the last checkpoint is lost on a crash
(at most ~10 minutes), and anytime.csv is rolled back to the checkpoint so the
anytime curve always matches a state the run actually had.

A method implements:

    class M(Method):
        name = "..."
        def init(self):            build the starting state
        def step(self):            one bounded chunk of work (< ~5 s)
        def get_state(self):       picklable dict
        def set_state(self, d):    inverse of get_state

and banks every permutation it evaluates through self.bank(perm) /
self.bank_deg(perm, deg), which offers its whole staircase to the shared
archive. The framework handles budgets, checkpoints, logs and signals.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import signal
import socket
import sys
import time
import traceback

# one thread per run: numpy must not spread over the other runs' cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import torso_engine as te  # noqa: E402

CKPT_EVERY_S = int(os.environ.get("CAMPAIGN_CKPT_EVERY", "600"))   # wall seconds between checkpoints
HEARTBEAT_EVERY_S = 30
FORMAT_VERSION = 1


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_bytes(path, data):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
    except OSError:
        pass


def atomic_write_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=1).encode())


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Method:
    """Base class. Subclasses set name and implement init/step/get_state/set_state."""

    name = "method"
    defaults: dict = {}

    def __init__(self, ctx, params):
        self.ctx = ctx
        self.eng = ctx.eng
        self.n = ctx.eng.n
        self.p = dict(self.defaults)
        self.p.update(params or {})

    # every evaluated permutation goes through one of these two
    def bank(self, perm):
        imp, deg, w, t = self.ctx.arch.eval_add(self.eng, perm)
        self.ctx.evals += 1
        return imp, deg, w, t

    def bank_deg(self, perm, deg):
        self.ctx.evals += 1
        return self.ctx.arch.add_deg(perm, deg)

    def note(self, msg):
        self.ctx.log(msg)

    def init(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def get_state(self):
        return {}

    def set_state(self, d):
        pass

    def extra_status(self):
        return {}


class Context:
    """What a method sees: engine, archive, counters, logging."""

    def __init__(self, run_dir, instance_path, method_name, seed):
        self.run_dir = run_dir
        self.instance_path = instance_path
        self.instance = os.path.splitext(os.path.basename(instance_path))[0]
        self.method_name = method_name
        self.seed = seed
        self.eng = te.Engine(instance_path)
        self.arch = te.Archive(self.eng.n)
        self.evals = 0
        self._logf = open(os.path.join(run_dir, "log.txt"), "a")

    def log(self, msg):
        line = f"{now_utc()} {msg}"
        self._logf.write(line + "\n")
        self._logf.flush()


class Runner:
    def __init__(self, method_cls, run_dir, instance_path, seed=1, params=None,
                 until_budget=None, deadline_epoch=None, quiet=False):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.method_cls = method_cls
        self.until_budget = until_budget
        self.deadline_epoch = deadline_epoch
        self.quiet = quiet
        self.ctx = Context(run_dir, instance_path, method_cls.name, seed)
        self.method = method_cls(self.ctx, params)
        self.budget_base = 0.0          # CPU seconds of earlier sessions
        self.cpu0 = time.process_time()
        self.sessions = 0
        self.best_score = 0
        self.best_budget = 0.0
        self.stop_flag = False
        self._pending_row = False
        self._last_row_budget = -1e9
        self._best_written = None
        self.ckpt_path = os.path.join(run_dir, "ckpt.pkl")
        self.prev_path = os.path.join(run_dir, "ckpt.prev.pkl")
        self.anytime_path = os.path.join(run_dir, "anytime.csv")
        self.started_wall = time.time()

    # ---------------------------------------------------------------- clock
    def budget(self):
        return self.budget_base + (time.process_time() - self.cpu0)

    # --------------------------------------------------------------- checkpoints
    def _load_ckpt(self):
        for path in (self.ckpt_path, self.prev_path):
            if not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    d = pickle.load(f)
                if d.get("version") != FORMAT_VERSION or d.get("method") != self.method_cls.name:
                    raise ValueError("checkpoint of another method or format")
                return d, path
            except Exception as e:                      # corrupt or partial
                self.ctx.log(f"checkpoint {os.path.basename(path)} unreadable: {e!r}")
        return None, None

    def save(self, reason):
        d = {
            "version": FORMAT_VERSION,
            "method": self.method_cls.name,
            "instance": self.ctx.instance,
            "seed": self.ctx.seed,
            "params": self.method.p,
            "budget": self.budget(),
            "evals": self.ctx.evals,
            "sessions": self.sessions,
            "best_score": self.best_score,
            "best_budget": self.best_budget,
            "archive": self.ctx.arch.export(),
            "method_state": self.method.get_state(),
            "saved_utc": now_utc(),
            "reason": reason,
        }
        data = pickle.dumps(d, protocol=4)
        if os.path.exists(self.ckpt_path):
            os.replace(self.ckpt_path, self.prev_path)
        atomic_write_bytes(self.ckpt_path, data)
        self._write_best()
        self._write_status("running" if reason == "periodic" else reason)
        self._anytime_row("ckpt")

    def _restore(self, d):
        self.budget_base = float(d["budget"])
        self.cpu0 = time.process_time()
        self.ctx.evals = int(d["evals"])
        self.sessions = int(d["sessions"])
        self.best_score = int(d["best_score"])
        self.best_budget = float(d["best_budget"])
        self.ctx.arch = te.Archive.restore(d["archive"])
        self.method.p.update(d.get("params", {}))
        self.method.set_state(d["method_state"])
        # anytime.csv: keep only what the checkpointed state had reached
        if os.path.exists(self.anytime_path):
            keep = []
            with open(self.anytime_path) as f:
                for i, line in enumerate(f):
                    if i == 0 or not line.strip():
                        keep.append(line)
                        continue
                    try:
                        if float(line.split(",")[0]) <= self.budget_base + 1e-6:
                            keep.append(line)
                    except ValueError:
                        pass
            atomic_write_bytes(self.anytime_path, "".join(keep).encode())

    # ------------------------------------------------------------------ outputs
    def _anytime_row(self, kind):
        self._last_row_budget = self.budget()
        new = not os.path.exists(self.anytime_path)
        with open(self.anytime_path, "a") as f:
            if new:
                f.write("budget_s,wall_utc,score,evaluations,points,kind\n")
            w, _ = self.ctx.arch.pareto()
            f.write(f"{self.budget():.1f},{now_utc()},{self.ctx.arch.score()},"
                    f"{self.ctx.evals},{len(w)},{kind}\n")

    def _write_best(self):
        arch = self.ctx.arch
        if self._best_written == arch.score() and os.path.exists(os.path.join(self.run_dir, "best.json")):
            return
        self._best_written = arch.score()
        best = {
            "challenge": "spoc-3-torso-decompositions",
            "problem": self.ctx.instance,
            "method": self.method_cls.name,
            "score": arch.score(),
            "budget_s": round(self.budget(), 1),
            "evaluations": self.ctx.evals,
            "saved_utc": now_utc(),
            "decisionVector": arch.decision_vectors(),
        }
        atomic_write_json(os.path.join(self.run_dir, "best.json"), best)

    def _write_status(self, state):
        ws, ts = self.ctx.arch.selected()
        st = {
            "method": self.method_cls.name,
            "instance": self.ctx.instance,
            "n": self.ctx.eng.n,
            "state": state,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "budget_s": round(self.budget(), 1),
            "until_budget_s": self.until_budget,
            "score": self.ctx.arch.score(),
            "best_budget_s": round(self.best_budget, 1),
            "evaluations": self.ctx.evals,
            "sessions": self.sessions,
            "front": [[int(w), int(t)] for w, t in zip(ws, ts)],
            "updated_utc": now_utc(),
            "extra": self.method.extra_status(),
        }
        atomic_write_json(os.path.join(self.run_dir, "status.json"), st)

    def _heartbeat(self):
        hb = os.path.join(self.run_dir, "heartbeat")
        with open(hb, "w") as f:
            f.write(f"{time.time():.0f} {os.getpid()} {self.budget():.0f} {self.ctx.arch.score()}\n")

    # --------------------------------------------------------------------- loop
    def _on_signal(self, signum, frame):
        self.stop_flag = True

    def _lock(self):
        """One process per run directory (an orphan of a crashed supervisor
        must never run beside its replacement)."""
        import fcntl
        self._lockf = open(os.path.join(self.run_dir, "lock"), "a+")
        try:
            fcntl.flock(self._lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"{self.run_dir} is already running", file=sys.stderr)
            sys.exit(3)
        self._lockf.seek(0)
        self._lockf.truncate()
        self._lockf.write(str(os.getpid()))
        self._lockf.flush()

    def run(self):
        self._lock()
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        d, path = self._load_ckpt()
        if d is not None:
            self._restore(d)
            self.ctx.log(f"resumed from {os.path.basename(path)} at budget {self.budget_base:.0f}s, "
                         f"score {self.ctx.arch.score():,}")
        else:
            self.ctx.log(f"start {self.method_cls.name} on {self.ctx.instance} seed {self.ctx.seed} "
                         f"params {self.method.p}")
            self.method.init()
            self.best_score = self.ctx.arch.score()
            self.best_budget = self.budget()
            self._anytime_row("init")
            self.save("initial")
        self.sessions += 1
        last_ckpt = time.time()
        last_hb = 0.0
        exit_reason = "stopped"
        try:
            while True:
                if self.until_budget is not None and self.budget() >= self.until_budget:
                    exit_reason = "slice-done"
                    break
                if self.deadline_epoch is not None and time.time() >= self.deadline_epoch:
                    exit_reason = "deadline"
                    break
                if self.stop_flag:
                    exit_reason = "stopped"
                    break
                self.method.step()
                sc = self.ctx.arch.score()
                if sc < self.best_score:
                    self.best_score = sc
                    self.best_budget = self.budget()
                    self._pending_row = True
                b = self.budget()
                if self._pending_row and b - self._last_row_budget >= max(5.0, 0.01 * b):
                    self._anytime_row("improve")
                    self._pending_row = False
                    if not self.quiet:
                        print(f"[{self.ctx.instance}/{self.method_cls.name}] {self.budget():9.1f}s  {sc:,}", flush=True)
                t = time.time()
                if t - last_hb >= HEARTBEAT_EVERY_S:
                    self._heartbeat()
                    last_hb = t
                if t - last_ckpt >= CKPT_EVERY_S:
                    self.save("periodic")
                    last_ckpt = t
        except Exception:
            self.ctx.log("CRASH\n" + traceback.format_exc())
            raise
        self.save(exit_reason)
        self._heartbeat()
        self.ctx.log(f"session end ({exit_reason}) at budget {self.budget():.0f}s, score {self.ctx.arch.score():,}")
        return exit_reason


# --------------------------------------------------------------- helpers used by methods

def py_rng_state(r: random.Random):
    return r.getstate()


def np_rng_state(g: np.random.Generator):
    return g.bit_generator.state


def np_rng_from_state(st):
    g = np.random.default_rng()
    g.bit_generator.state = st
    return g


def staircase_t_at(w_arr, t_arr, target, n):
    """Smallest t of a staircase (w asc, t desc) whose width <= target; n if none."""
    if len(w_arr) == 0:
        return n
    idx = np.searchsorted(w_arr, target, side="right") - 1
    if idx < 0:
        return n
    return int(t_arr[idx])
