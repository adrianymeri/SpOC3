#!/usr/bin/env bash
# ============================================================
# stop_mac.sh -- FULL shutdown of the Mac fleet.
# Kills the respawn loops FIRST (otherwise they restart the arms),
# then every arm, then the caffeinate sleep-blockers.
# Nothing is lost: every arm banks its accepts to submissions/ on
# each improvement, so restarting later resumes from the pool.
#   usage:  cd ~/Desktop/SpOC3/"Torso Decompositions" && bash stop_mac.sh
# ============================================================
cd "$(dirname "$0")" || exit 1
echo "=== STOPPING MAC FLEET  $(date '+%d.%m %H:%M') ==="

# ---- 1. respawn / self-heal loops (MUST die first) ----
for pat in grow_forever.sh keepalive_mac.sh resume_all.sh ablation_watch.sh \
           log_scores.sh ablation_grow.sh ablation_endpoint.sh beat_mac.sh; do
  pkill -f "$pat" 2>/dev/null && echo "  killed loop : $pat"
done
sleep 2

# ---- 2. the arms ----
for pat in archive_evolve.py gbfcpp.py gbfcpp_swarm.py hri_lns.py hri_lns_gbdt.py \
           hri_plain.py endpoint_ils.py gbdt_grow.py gbdt_mapelites.py gbdt_moves.py \
           gbdt_sweep.py bandit_widths.py cqs.py quotient_lns.py boundary_lns.py \
           torso_deletion.py gaps_search.py band_climb.py grow_highband.py \
           clique_prefix.py breakpoint_nudge.py crossover_relinking.py diversify.py \
           ace_search.py; do
  pkill -f "tools/$pat" 2>/dev/null && echo "  killed arm  : $pat"
done
sleep 2

# ---- 3. sleep blockers (so the laptop can actually idle/sleep) ----
pkill caffeinate 2>/dev/null && echo "  killed caffeinate wrappers"

# ---- 4. verify ----
echo "--- STILL RUNNING (this list should be EMPTY) ---"
ps aux | grep -E "tools/[a-z_]*\.py|grow_forever|keepalive_mac|caffeinate" \
       | grep -v grep | awk '{print "  PID "$2"  "$11" "$12" "$13" "$14}'
echo "--- load average (should fall toward 0 within a minute) ---"
uptime | sed 's/.*load average://'
echo
echo "If anything is still listed above:  kill -9 <PID>"
echo "Do NOT run resume_all.sh / tools/keepalive_mac.sh on this Mac until you"
echo "want the fleet back -- both of them relaunch everything."
