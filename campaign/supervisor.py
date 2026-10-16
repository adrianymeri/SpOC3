#!/usr/bin/env python3
"""
supervisor.py -- keeps every campaign run going until the end date or STOP.

Started at boot by cron (campaign/start.sh, installed by campaign/install.sh)
and re-started by a 5-minute keepalive entry if it ever dies. Only one copy
runs (state/supervisor.lock).

Layout (one run per physical core, the hyper-thread sibling left idle):
    core 0          the supervisor, the aggregator; cuda-torso's host process
                    on the core's second thread
    next 24 cores   the official instances x 8 CPU methods, each pinned to its
                    own core for the whole campaign
    the rest        the synthetic runs, in rounds (config.SYNTH_LEVELS_H): a
                    run is started with --until-budget = the current level and
                    nobody gets the next level until every synthetic run has
                    reached this one, so a fair table exists at every level
    the GPU         cuda-torso, one instance at a time, in rounds
                    (config.GPU_LEVELS_H, official instances x3)

Every 10 s it restarts dead runs from their checkpoints, kills runs whose
heartbeat is older than 15 minutes, fills free synthetic slots and the GPU,
and every 10 minutes runs the aggregator (results/). A run that crashes more
than 6 times in an hour is parked for an hour and flagged in STATUS.txt; the
rounds do not wait for parked runs.

Stop: create campaign/state/STOP (campaign/stop.sh does), or reach config.END.
All runs get SIGTERM, write a final checkpoint, and the aggregator runs once more.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as C  # noqa: E402

EPS = 5.0   # seconds of budget counted as "level reached"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def physical_cores():
    """[(package, core) -> sorted logical CPUs] in a stable order."""
    fake = int(os.environ.get("CAMPAIGN_FAKE_CORES", "0"))
    if fake:                                   # tests: pretend, map onto the real CPUs
        real = sorted(os.sched_getaffinity(0))
        return [[real[i % len(real)]] for i in range(fake)]
    cores = {}
    base = "/sys/devices/system/cpu"
    allowed = os.sched_getaffinity(0)
    for name in os.listdir(base):
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        cpu = int(name[3:])
        if cpu not in allowed:
            continue
        try:
            pkg = int(open(f"{base}/{name}/topology/physical_package_id").read())
            core = int(open(f"{base}/{name}/topology/core_id").read())
        except OSError:
            pkg, core = 0, cpu
        cores.setdefault((pkg, core), []).append(cpu)
    return [sorted(v) for k, v in sorted(cores.items())]


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Child:
    def __init__(self, key, proc, cpu, until):
        self.key = key
        self.proc = proc
        self.cpu = cpu
        self.until = until
        self.started = time.time()


class Supervisor:
    def __init__(self, dry=False):
        os.makedirs(C.STATE, exist_ok=True)
        os.makedirs(C.RUNS, exist_ok=True)
        self.dry = dry
        self.lockf = open(os.path.join(C.STATE, "supervisor.lock"), "a+")
        try:
            fcntl.flock(self.lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("supervisor already running")
            sys.exit(0)
        self.lockf.seek(0)
        self.lockf.truncate()
        self.lockf.write(str(os.getpid()))
        self.lockf.flush()
        self.logf = open(os.path.join(C.STATE, "supervisor.log"), "a")
        cores = physical_cores()
        if len(cores) < C.RESERVED_CORES + 2:
            raise SystemExit(f"only {len(cores)} physical cores visible")
        sys_core = cores[0]
        self.sys_cpu = sys_core[0]
        self.gpu_cpu = sys_core[1] if len(sys_core) > 1 else sys_core[0]
        run_cpus = [c[0] for c in cores[C.RESERVED_CORES:]]
        self.official_keys = [(i, m) for i in C.OFFICIAL for m in C.CPU_METHODS]
        n_off = min(len(self.official_keys), len(run_cpus) - 1)
        if n_off < len(self.official_keys):
            self.log(f"WARNING: only {len(run_cpus)} run cores; {len(self.official_keys) - n_off} official runs will share the synthetic pool")
        self.official_cpu = {k: run_cpus[j] for j, k in enumerate(self.official_keys[:n_off])}
        self.pool_cpus = run_cpus[n_off:]
        self.synth_keys = [(i, m) for i in C.SYNTHETIC for m in C.CPU_METHODS] + \
                          [k for k in self.official_keys[n_off:]]
        self.gpu_keys = [(i, m) for m in C.GPU_METHODS for i in C.INSTANCES]
        self.children = {}            # key -> Child
        self.crashes = {}             # key -> [timestamps]
        self.parked = {}              # key -> until epoch
        self.gpu_child = None
        self.agg = None
        self.last_agg = 0.0
        self.stop_requested = False
        self.end_epoch = time.mktime(time.strptime(C.END, "%Y-%m-%dT%H:%M:%S"))
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)
        try:
            os.sched_setaffinity(0, {self.sys_cpu})
        except OSError:
            pass
        self.log(f"supervisor up, pid {os.getpid()}, {len(cores)} physical cores, system cpu {self.sys_cpu}, "
                 f"gpu host cpu {self.gpu_cpu}, {len(self.official_cpu)} official runs pinned, "
                 f"{len(self.pool_cpus)} pool cores, end {C.END}")

    def _sig(self, *a):
        self.stop_requested = True

    def log(self, msg):
        self.logf.write(f"{now()} {msg}\n")
        self.logf.flush()

    # ----------------------------------------------------------- helpers
    def budget(self, key):
        st = read_json(os.path.join(C.run_dir(*key), "status.json"), {})
        return float(st.get("budget_s", 0.0))

    def orphans(self):
        """Stop runs left behind by a previous supervisor (they hold run locks)."""
        pids = []
        for inst in C.INSTANCES:
            for m in C.METHODS:
                lf = os.path.join(C.run_dir(inst, m), "lock")
                if not os.path.exists(lf):
                    continue
                with open(lf, "a+") as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        try:
                            f.seek(0)
                            pid = int(f.read().strip() or 0)
                            if pid:
                                os.kill(pid, signal.SIGTERM)
                                pids.append(pid)
                        except (ValueError, ProcessLookupError, PermissionError):
                            pass
        found = len(pids)
        deadline = time.time() + 180
        while pids and time.time() < deadline:
            pids = [p for p in pids if self._alive(p)]
            time.sleep(1)
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if found:
            self.log(f"stopped {found} orphan runs (they checkpointed and will be restarted)")

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def start(self, key, cpu, until=None, gpu=False):
        inst, method = key
        rd = C.run_dir(inst, method)
        os.makedirs(rd, exist_ok=True)
        if gpu:
            cmd = [C.GPU_PYTHON, os.path.join(HERE, "gpu", "cuda_torso.py"), "--instance",
                   C.instance_path(inst), "--run-dir", rd]
            backend = C.CUDA_TORSO_BACKEND or (open(os.path.join(C.STATE, "cuda_backend")).read().strip()
                                              if os.path.exists(os.path.join(C.STATE, "cuda_backend")) else "nvcc")
            cmd += ["--backend", backend]
        else:
            cmd = [C.PYTHON, os.path.join(HERE, "run_method.py"), "--method", method, "--instance",
                   C.instance_path(inst), "--run-dir", rd, "--seed", str(C.SEED), "--quiet"]
        if until is not None:
            cmd += ["--until-budget", f"{until:.0f}"]
        if self.dry:
            cmd = [C.PYTHON, "-c", "import time; time.sleep(1)"]
        out = open(os.path.join(rd, "stdout.txt"), "a")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

        def pin():
            try:
                os.sched_setaffinity(0, {cpu})
            except OSError:
                pass
            os.setsid()
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, preexec_fn=pin, env=env, cwd=C.REPO)
        out.close()
        ch = Child(key, proc, cpu, until)
        self.log(f"start {inst}/{method} on cpu {cpu}" + (f" until {until / 3600:.2f} h" if until else ""))
        return ch

    def crashed(self, key, code):
        t = time.time()
        lst = [x for x in self.crashes.get(key, []) if t - x < 3600] + [t]
        self.crashes[key] = lst
        self.log(f"run {key[0]}/{key[1]} exited with code {code} ({len(lst)} in the last hour)")
        if len(lst) > C.MAX_CRASHES_PER_HOUR:
            self.parked[key] = t + 3600
            self.log(f"PARKED {key[0]}/{key[1]} for an hour after repeated crashes")

    def is_parked(self, key):
        u = self.parked.get(key)
        if u is None:
            return False
        if time.time() >= u:
            del self.parked[key]
            return False
        return True

    def reap(self, ch):
        """Returns True if the child has finished (and handles its exit)."""
        code = ch.proc.poll()
        if code is None:
            hb = os.path.join(C.run_dir(*ch.key), "heartbeat")
            age = time.time() - (os.path.getmtime(hb) if os.path.exists(hb) else ch.started)
            if time.time() - ch.started > C.HEARTBEAT_STALE_S and age > C.HEARTBEAT_STALE_S:
                self.log(f"run {ch.key[0]}/{ch.key[1]} silent for {age / 60:.0f} min: restarting")
                self.terminate([ch])
                self.crashed(ch.key, "stale")
                return True
            return False
        if code not in (0, 3):
            self.crashed(ch.key, code)
        return True

    def terminate(self, chs, timeout=180):
        for ch in chs:
            try:
                os.killpg(ch.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + timeout
        for ch in chs:
            while ch.proc.poll() is None and time.time() < deadline:
                time.sleep(0.5)
            if ch.proc.poll() is None:
                try:
                    os.killpg(ch.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                ch.proc.wait()

    # ------------------------------------------------------------- rounds
    @staticmethod
    def next_level(b, levels):
        for i, L in enumerate(levels):
            if b < L - EPS:
                return i, L
        return len(levels), None

    def synth_round(self):
        levels = [h * 3600 for h in C.SYNTH_LEVELS_H]
        info = []
        for k in self.synth_keys:
            b = self.budget(k)
            i, L = self.next_level(b, levels)
            info.append((k, b, i, L))
        active = [x for x in info if not self.is_parked(x[0]) and x[3] is not None]
        if not active:
            return None, []
        r = min(x[2] for x in active)
        eligible = [x for x in active if x[2] == r and x[0] not in self.children]
        return r, eligible

    def gpu_round(self):
        info = []
        for k in self.gpu_keys:
            w = C.OFFICIAL_GPU_WEIGHT if k[0] in C.OFFICIAL else 1
            levels = [h * 3600 * w for h in C.GPU_LEVELS_H]
            b = self.budget(k)
            i, L = self.next_level(b, levels)
            info.append((k, b, i, L))
        active = [x for x in info if not self.is_parked(x[0]) and x[3] is not None]
        if not active:
            return None, []
        r = min(x[2] for x in active)
        return r, [x for x in active if x[2] == r]

    # --------------------------------------------------------------- loop
    def aggregate(self, final=False):
        if self.agg is not None and self.agg.poll() is None:
            if not final:
                return
            self.agg.wait()
        cmd = [C.PYTHON, os.path.join(HERE, "aggregate.py")]
        if final:
            subprocess.run(cmd + ["--final"], cwd=C.REPO)
            return

        def pin():
            try:
                os.sched_setaffinity(0, {self.sys_cpu})
                os.nice(10)
            except OSError:
                pass
        self.agg = subprocess.Popen(cmd, cwd=C.REPO, preexec_fn=pin,
                                    stdout=open(os.path.join(C.STATE, "aggregate.log"), "a"),
                                    stderr=subprocess.STDOUT)
        self.last_agg = time.time()

    def write_status(self, sr, gr):
        st = {
            "updated": now(), "pid": os.getpid(), "end": C.END,
            "running": {f"{k[0]}/{k[1]}": {"cpu": ch.cpu, "until_h": (ch.until / 3600 if ch.until else None),
                                            "since": time.strftime("%Y-%m-%d %H:%M", time.localtime(ch.started))}
                        for k, ch in self.children.items()},
            "gpu": (f"{self.gpu_child.key[0]}" if self.gpu_child else None),
            "synthetic_round": sr, "synthetic_level_h": (C.SYNTH_LEVELS_H[sr] if sr is not None and sr < len(C.SYNTH_LEVELS_H) else None),
            "gpu_round": gr, "gpu_level_h": (C.GPU_LEVELS_H[gr] if gr is not None and gr < len(C.GPU_LEVELS_H) else None),
            "parked": [f"{k[0]}/{k[1]}" for k in self.parked],
            "crashes_last_hour": {f"{k[0]}/{k[1]}": len(v) for k, v in self.crashes.items() if v},
        }
        tmp = os.path.join(C.STATE, "supervisor_status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, os.path.join(C.STATE, "supervisor_status.json"))

    def stop_all(self, why):
        self.log(f"stopping everything: {why}")
        chs = list(self.children.values()) + ([self.gpu_child] if self.gpu_child else [])
        self.terminate(chs)
        self.children.clear()
        self.gpu_child = None
        self.aggregate(final=True)
        self.log("all runs stopped; final aggregation done")

    def loop(self, max_iterations=None):
        self.orphans()
        it = 0
        sr = gr = None
        while True:
            it += 1
            if max_iterations and it > max_iterations:
                break
            if self.stop_requested or os.path.exists(os.path.join(C.STATE, "STOP")) or time.time() >= self.end_epoch:
                why = "signal" if self.stop_requested else ("STOP file" if os.path.exists(os.path.join(C.STATE, "STOP")) else "end date")
                self.stop_all(why)
                if why == "end date":
                    open(os.path.join(C.STATE, "STOP"), "w").write(f"end date reached {now()}\n")
                return
            # reap
            for k, ch in list(self.children.items()):
                if self.reap(ch):
                    del self.children[k]
            if self.gpu_child is not None and self.reap(self.gpu_child):
                self.gpu_child = None
            # official runs: always running
            for k, cpu in self.official_cpu.items():
                if k not in self.children and not self.is_parked(k):
                    self.children[k] = self.start(k, cpu)
            # synthetic pool: rounds
            busy = {ch.cpu for k, ch in self.children.items() if k not in self.official_cpu}
            free = [c for c in self.pool_cpus if c not in busy]
            sr, eligible = self.synth_round()
            for cpu, (k, b, i, L) in zip(free, eligible):
                self.children[k] = self.start(k, cpu, until=L)
            # GPU
            if self.gpu_child is None:
                gr, ge = self.gpu_round()
                ge = [x for x in ge if not os.path.exists(os.path.join(C.STATE, "NO_GPU"))]
                if ge:
                    k, b, i, L = ge[0]
                    self.gpu_child = self.start(k, self.gpu_cpu, until=L, gpu=True)
            if time.time() - self.last_agg >= C.AGGREGATE_EVERY_S:
                self.aggregate()
            self.write_status(sr, gr)
            time.sleep(10 if not self.dry else 0.2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="start dummy processes (tests)")
    ap.add_argument("--iterations", type=int, default=None)
    a = ap.parse_args()
    Supervisor(dry=a.dry).loop(a.iterations)


if __name__ == "__main__":
    main()
