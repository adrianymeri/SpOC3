#!/bin/bash
cd "$(dirname "$0")/.."
for p in small-graph medium-graph large-graph; do echo "$p: $(python3 tools/cap_submit.py --problem $p 2>/dev/null | grep -i best-20)"; done
