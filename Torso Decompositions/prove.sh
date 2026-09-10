#!/usr/bin/env bash
# ============================================================
# prove.sh -- ONE command. Everything that matters, GBDT FIRST.
#
# Answers the three questions the thesis has to answer, in order:
#   1. Are we beating the leaderboard?            (scores, BEAT flags)
#   2. Is the result GBDT-driven?                 (attribution: which arm
#                                                  produced each submitted
#                                                  point, and who owns the
#                                                  decisive endpoint)
#   3. Does GBDT CAUSE better results?            (paired ablation: equal
#                                                  evaluation budget, GBDT
#                                                  ranker vs random control,
#                                                  with a sign-test p-value
#                                                  and the ranker correlation)
# Then: what is alive, and machine load.
#
#   bash prove.sh          full run (~2-3 min: recomputes attribution)
#   bash prove.sh quick    skip attribution; arms + ablation + load only
# Read-only. Starts and stops nothing.
# ============================================================
cd "$(dirname "$0")" || exit 1
MODE="${1:-full}"

hr(){ printf '%s\n' "----------------------------------------------------------------"; }
# count REAL arm processes: exclude the caffeinate / bash -lc wrappers that
# made the old roster triple-count every arm
narm(){ ps -eo args= 2>/dev/null | grep -F "$1" | grep -v grep \
        | grep -v caffeinate | grep -v -- "-lc" | wc -l | tr -d ' '; }

echo "================================================================"
echo " SpOC-3  |  $(hostname -s)  |  $(date '+%d.%m.%Y %H:%M')"
echo " targets: small -1,829,919 | medium -1,745,122 | large -5,493,062"
echo " (score MORE NEGATIVE than target = BEATING it)"
echo "================================================================"

if [ "$MODE" != "quick" ]; then
  echo
  echo "### 1+2.  SCORE  and  GBDT ATTRIBUTION"
  hr
  for p in small-graph medium-graph large-graph; do
    out=$(python3 tools/attribute.py --problem "$p" 2>/dev/null)
    hdr=$(echo "$out" | grep -E "^=== attribution")
    score=$(echo "$hdr" | sed 's/.*best-20 = //; s/ *===//')
    gb=$(echo "$out" | grep -E "^  GBDT " | sed 's/^  GBDT *: *//')
    cl=$(echo "$out" | grep -E "^  classical " | sed 's/^  classical *: *//')
    ep=$(echo "$out" | grep -E "^  ENDPOINT \(t=0\)" | sed 's/.*owner: //')
    if echo "$score" | grep -q "gap -"; then flag="*** BEAT ***"; else flag="still short"; fi
    printf "  %-13s %s   %s\n" "$p" "$score" "$flag"
    printf "  %-13s   GBDT      %s\n" "" "${gb:-n/a}"
    printf "  %-13s   classical %s\n" "" "${cl:-n/a}"
    printf "  %-13s   ENDPOINT owned by: %s\n" "" "${ep:-n/a}"
    echo
  done
  echo "  Quote POINT COUNTS and ENDPOINT OWNER. Do NOT quote the HV %% --"
  echo "  the endpoint strip runs to the reference point and carries ~99%% of"
  echo "  the hypervolume by construction, whoever owns it."
fi

echo
echo "### 3.  GBDT CAUSAL EVIDENCE  (paired ablations)"
hr
echo "  --- (A) ENDPOINT MOVE RANKER: gbdt vs random control ---"
running=$(narm "ablation_endpoint.sh")
found=0
for f in ablation_endpoint_*.txt; do
  [ -e "$f" ] || continue
  found=1
  n=$(grep -c "^RESULT" "$f" 2>/dev/null)
  echo "  $f : $n arm results so far"
  if [ "$n" -ge 2 ]; then
    python3 - "$f" << 'PYEOF'
import sys
rows={}
for line in open(sys.argv[1]):
    if not line.startswith("RESULT"): continue
    d=dict(kv.split("=",1) for kv in line.split()[1:])
    rows.setdefault(int(d["seed"]),{})[d["mode"]]=d
w=l=t=0; rr=[]
for s in sorted(rows):
    p=rows[s]
    if "gbdt" not in p or "random" not in p: continue
    g=(int(p['gbdt']['best_width']), int(p['gbdt']['best_bottleneck']))
    r=(int(p['random']['best_width']), int(p['random']['best_bottleneck']))
    if g<r: w+=1
    elif g>r: l+=1
    else: t+=1
    v=p['gbdt'].get('ranker_r','na')
    if v not in ("na",""):
        try: rr.append(float(v))
        except ValueError: pass
print(f"    GBDT wins {w} | losses {l} | ties {t}")
n=w+l
if n:
    from math import comb
    k=min(w,l); pv=min(sum(comb(n,i) for i in range(k+1))*2/(2**n),1.0)
    verdict = "SIGNIFICANT" if pv<0.05 else "not yet significant"
    print(f"    exact two-sided sign test: p = {pv:.4f}  ({verdict})")
else:
    print("    all pairs tied so far -- no signal yet")
if rr:
    m=sum(rr)/len(rr)
    q=("ranker learned a real signal" if m>0.10 else
       "WEAK/NO signal -- report honestly" if m<0.05 else "marginal")
    print(f"    mean ranker_r = {m:+.3f} over {len(rr)} arms  -> {q}")
else:
    print("    ranker_r not reported yet (arms still in warmup)")
PYEOF
  fi
done
[ "$found" -eq 1 ] || echo "  (A) no endpoint-ranker results file yet"
echo "  (A) endpoint-ranker driver running: $running"

echo
echo "  --- (B) SET-SPACE CONSTRUCTION: gbdt_grow vs --no-gbdt control ---"
echo "      (isolated pools, equal pass budget; metric = envelope HV gained)"
gfound=0
for f in ablation_grow_*.txt; do
  [ -e "$f" ] || continue
  gfound=1
  n=$(grep -c "^RESULT" "$f" 2>/dev/null)
  echo "  $f : $n arm results so far"
  if [ "$n" -ge 2 ]; then
    python3 - "$f" << 'PYEOF'
import sys
rows={}
for line in open(sys.argv[1]):
    if not line.startswith("RESULT"): continue
    d=dict(kv.split("=",1) for kv in line.split()[1:])
    rows.setdefault(int(d["seed"]),{})[d["mode"]]=d
w=l=t=0; zero=0
for s in sorted(rows):
    p=rows[s]
    if "gbdt" not in p or "nogbdt" not in p: continue
    g=float(p['gbdt']['delta_hv']); r=float(p['nogbdt']['delta_hv'])
    if g==0 and r==0: zero+=1
    if   g>r: w+=1
    elif g<r: l+=1
    else:     t+=1
print(f"    GBDT wins {w} | losses {l} | ties {t}")
n=w+l
if n:
    from math import comb
    k=min(w,l); pv=min(sum(comb(n,i) for i in range(k+1))*2/(2**n),1.0)
    verdict="SIGNIFICANT" if pv<0.05 else "not yet significant"
    print(f"    exact two-sided sign test: p = {pv:.4f}  ({verdict})")
    if pv<0.05:
        print(f"    -> {'LEARNED ranking causally better' if w>l else 'CLASSICAL control better -- report honestly'}")
else:
    print("    all pairs tied -- no signal yet")
if zero and zero==t and t:
    print(f"    NOTE: {zero} pair(s) BOTH gained 0 HV -- the starting front is")
    print("    too mature to discriminate. That is INCONCLUSIVE, not negative;")
    print("    rerun from a weakened snapshot so both arms have room to climb.")
PYEOF
  fi
done
[ "$gfound" -eq 1 ] || echo "  (B) no set-space results file yet"
echo "  (B) set-space driver running: $(narm ablation_grow.sh)"

echo
echo "### 4.  ARMS ALIVE  (wrappers excluded)"
hr
printf "  %-26s %-10s %s\n" "ARM" "KIND" "N"
for row in "tools/gbdt_grow.py:GBDT:gbdt_grow" \
           "tools/endpoint_gbdt.py:GBDT:endpoint_gbdt" \
           "tools/gbfcpp.py:GBDT:gbfcpp" \
           "run_gbdt.py:GBDT:gpu_lottery" \
           "tools/endpoint_ils.py:classical:endpoint_ils" \
           "tools/hri_lns.py:classical:hri_lns" \
           "tools/archive_evolve.py:classical:archive_evolve" \
           "tools/boundary_lns.py:classical:boundary_lns"; do
  pat="${row%%:*}"; rest="${row#*:}"; kind="${rest%%:*}"; label="${rest##*:}"
  printf "  %-26s %-10s %s\n" "$label" "$kind" "$(narm "$pat")"
done
g=$(( $(narm tools/gbdt_grow.py) + $(narm tools/endpoint_gbdt.py) + $(narm tools/gbfcpp.py) + $(narm run_gbdt.py) ))
c=$(( $(narm tools/endpoint_ils.py) + $(narm tools/hri_lns.py) + $(narm tools/archive_evolve.py) + $(narm tools/boundary_lns.py) ))
echo "  ------------------------------------------"
echo "  GBDT arms: $g    classical arms: $c"
[ "$g" -eq 0 ] && echo "  !! NO GBDT ARM RUNNING -- the thesis claim needs these alive !!"
echo "  load:$(uptime | sed 's/.*load average[s]*://')"

echo
echo "next: if a score flips to BEAT ->"
echo "  python3 tools/verify_submission.py submissions/<problem>/cap20.json"
