#!/bin/bash
# sync_pool.sh -- full-pool Mac<->server sync via catbox/litterbox (no direct SSH needed).
#
# Ships the ENTIRE submissions pool (all top-level *.json per problem), not just
# cap20.json: the arms reseed from the pool, so cap20-only sync lets the machines
# drift (observed 2026-07-11: large -5,476,521 Mac vs -5,476,340 server).
# Peer files land with a fixed prefix (default srv_/mac_) and are REPLACED on each
# sync, so the pool doesn't grow unboundedly and no HV can be lost (cap20.json
# carries the pooled best-20 both ways).
#
# Usage:
#   bash tools/sync_pool.sh push                       # prints ONE url to paste
#   bash tools/sync_pool.sh pull <url> <prefix>        # e.g. prefix srv_ on Mac, mac_ on server
#
# push uploads to litterbox (24 h expiry, plenty for a daily sync); if that fails
# it falls back to permanent catbox (same API as the old runbook recipe).
set -euo pipefail
cd "$(dirname "$0")/.."
CMD="${1:-}"
PROBLEMS="small-graph medium-graph large-graph"

case "$CMD" in
  push)
    STAMP="$(hostname -s)_$(date +%Y%m%d_%H%M)"
    TGZ="/tmp/pool_${STAMP}.tgz"
    # top-level *.json only; skip peer copies (they came from the other machine),
    # platform wrappers, and hidden state files
    FILES=$(for p in $PROBLEMS; do
      for f in submissions/$p/*.json; do
        b=$(basename "$f")
        case "$b" in srv_*|mac_*|peer_*|*_platform.json) continue;; esac
        echo "$f"
      done
    done)
    # shellcheck disable=SC2086
    tar czf "$TGZ" $FILES
    echo "packed $(echo "$FILES" | wc -l | tr -d ' ') files -> $TGZ ($(du -h "$TGZ" | cut -f1))"
    URL=$(curl -sf -F "reqtype=fileupload" -F "time=24h" -F "fileToUpload=@$TGZ" \
          https://litterbox.catbox.moe/resources/internals/api.php) || \
    URL=$(curl -sf -F "reqtype=fileupload" -F "fileToUpload=@$TGZ" \
          https://catbox.moe/user/api.php)
    echo ""
    echo "PASTE ON THE OTHER MACHINE:"
    echo "  bash tools/sync_pool.sh pull $URL <srv_|mac_>"
    ;;

  pull)
    URL="${2:?usage: sync_pool.sh pull <url> <prefix>}"
    PREFIX="${3:?prefix required: srv_ (on Mac) or mac_ (on server)}"
    TMP=$(mktemp -d)
    if [ -f "$URL" ]; then cp "$URL" "$TMP/pool.tgz"; else curl -sfL -o "$TMP/pool.tgz" "$URL"; fi
    tar xzf "$TMP/pool.tgz" -C "$TMP"
    ADDED=0; SAME=0
    for p in $PROBLEMS; do
      [ -d "$TMP/submissions/$p" ] || continue
      mkdir -p "submissions/$p"
      for f in "$TMP/submissions/$p"/*.json; do
        [ -e "$f" ] || continue
        dst="submissions/$p/${PREFIX}$(basename "$f")"
        if [ -f "$dst" ] && cmp -s "$f" "$dst"; then SAME=$((SAME+1)); else
          cp "$f" "$dst"; ADDED=$((ADDED+1)); fi
      done
    done
    rm -rf "$TMP"
    echo "pulled: $ADDED new/updated, $SAME unchanged. Re-pooling..."
    for p in $PROBLEMS; do
      python3 tools/cap_submit.py --problem "$p" | grep -iE "best-20|residual" | sed "s/^/  [$p] /"
    done
    ;;

  *)
    echo "usage: sync_pool.sh push | pull <url> <srv_|mac_>"; exit 1;;
esac
