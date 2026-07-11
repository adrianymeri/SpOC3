# SpOC3 Ops Runbook — restart / repool / check

Two machines:
- **MAC**  : `cd ~/Desktop/SpOC3/"Torso Decompositions"`
- **SERVER**: `cd ~/SpOC3/"Torso Decompositions"`

The one number that matters per instance is `cap_submit … best-20`.
More negative = better. Targets: small −1,829,919 · medium −1,745,122 · large −5,493,062.

--------------------------------------------------------------------
## 1. CHECK STATUS

### MAC
```
cd ~/Desktop/SpOC3/"Torso Decompositions"
echo "=== MAC $(date '+%d.%m %H:%M') ==="
ps aux | grep -E "archive_evolve|gbfcpp|hri_lns" | grep -v grep | grep -v caffeinate | awk '{print "  "$2, $3"%cpu", $12, $13, $14}'
echo "  load:$(uptime | sed 's/.*load average://')"
for p in small-graph medium-graph large-graph; do echo "  $p:"; python3 tools/cap_submit.py --problem $p 2>/dev/null | grep -iE "best-20|residual"; done
```

### SERVER
```
cd ~/SpOC3/"Torso Decompositions"
echo "=== SERVER $(date '+%d.%m %H:%M') ==="
ps aux | grep -E "run_gbdt|run_capfocus|run\.py|gbfcpp|hri_lns|archive_evolve" | grep -v grep | awk '{print "  "$2, $3"%cpu", $12, $13, $14}'
nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader
echo "  load:$(uptime | sed 's/.*load average://')"
for p in small large medium; do cp ~/cuda-torso/submissions/$p-graph/*.json submissions/$p-graph/ 2>/dev/null; done
for p in small-graph medium-graph large-graph; do echo "  $p:"; python3 tools/cap_submit.py --problem $p 2>/dev/null | grep -iE "best-20|residual"; done
```

--------------------------------------------------------------------
## 2. RESTART anything that retired (either machine)

Arms that self-retire: **gbfcpp** (~33h, 1000 rounds) and **archive_evolve** (~days, 2M iters).
Everything else runs until killed or a reboot.

The resume script is idempotent — it starts ONLY what is missing, safe to run anytime:
```
cd ~/Desktop/SpOC3/"Torso Decompositions"     # MAC
bash resume_all.sh
```
```
cd ~/SpOC3/"Torso Decompositions"             # SERVER
bash resume_all.sh
```

After it runs, confirm no duplicates:
```
ps aux | grep -E "gbfcpp|hri_lns|archive_evolve" | grep -v grep | grep -v caffeinate | awk '{print $2,$12,$13,$14}'
```
If you ever see two arms with the SAME script+problem, kill the newer PID:  `kill <PID>`

--------------------------------------------------------------------
## 3. REPOOL  (share the best fronts so both machines build on them)

Do this ~once a day, **both directions** (2026-07-11: cap20-only sync let the pools
drift ~180 HV apart; `tools/sync_pool.sh` now ships the ENTIRE pool — ~6 MB gzipped,
all three problems in one URL — and re-scores automatically on pull).

### Step A — on MAC, publish the whole pool:
```
cd ~/Desktop/SpOC3/"Torso Decompositions"
bash tools/sync_pool.sh push
```
Prints one litterbox URL (24 h expiry; falls back to permanent catbox on failure).

### Step B — on SERVER, pull it in (peer files land as mac_*.json, replaced each sync):
```
cd ~/SpOC3/"Torso Decompositions"
bash tools/sync_pool.sh pull "PASTE_URL_HERE" mac_
```

### Step C — reverse direction, so the Mac also gets the GPU arms' points:
```
# on SERVER:
bash tools/sync_pool.sh push
# on MAC:
bash tools/sync_pool.sh pull "PASTE_URL_HERE" srv_
```
After a two-way sync both machines print identical best-20 lines. Pulls are
idempotent (unchanged files are skipped) and never clobber local arm files.

(The old single-file recipe still works for a quick one-off:
`curl -F "reqtype=fileupload" -F "fileToUpload=@submissions/large-graph/cap20.json" https://catbox.moe/user/api.php`)

--------------------------------------------------------------------
## 4. RESTART a single arm (e.g. one degraded / one you killed)

Find its PID with the check in §1, `kill <PID>`, then relaunch the matching line:

MAC (wrap in `caffeinate -i` so the laptop doesn't sleep it):
```
cd ~/Desktop/SpOC3/"Torso Decompositions"
caffeinate -i nohup python3 tools/gbfcpp.py --problem medium-graph --algo gbfcpp_capm --cap20 --rounds 1000 --round-budget 120 > medium_gbfcpp.log 2>&1 &
```

SERVER small GPU lottery (if `rate` collapses far below ~1200/s):
```
cd ~/cuda-torso
kill <old_pid>
nohup python3 run_gbdt.py --graph small-graph --batch_size 1024 --max_generations 100000000 > ~/SpOC3/"Torso Decompositions"/small_gpu_lottery.log 2>&1 &
```

--------------------------------------------------------------------
## 5. AFTER A REBOOT (power cut, etc.)

1. On the SERVER, resume the GPU + CPU arms:  `bash resume_all.sh`  (from the project dir)
2. On the MAC, resume its arms:               `bash resume_all.sh`
3. Repool (§3).
4. Check (§1).
The engines warm-start from their saved checkpoints, so a reboot loses only minutes, not the run.

--------------------------------------------------------------------
## NOTES / EXPECTATIONS
- Small is frozen at gap +5 (near-optimal wall). The GPU lottery can only *tie*, not beat.
- Large is near its floor (residual small); medium has the most room left.
- Movement is now tens–hundreds of HV/day — diminishing returns, this is normal.
- The two `0.0%cpu  -lc python3 …` lines in `ps` are harmless launcher wrappers, ignore them.
- Trust ONLY `cap_submit … best-20`. Ignore the engines' own internal "score/gap" log lines.
