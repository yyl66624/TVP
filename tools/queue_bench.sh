#!/bin/bash
# Run several benchmarks sequentially: queue_bench.sh "name:radius:log" ...
set -u
cd "${WORKSHOP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
for spec in "$@"; do
  IFS=: read -r NAME RADIUS LOG <<< "$spec"
  echo "=== $(date +%H:%M) $NAME radius $RADIUS"
  ./tools/run_benchmark.sh "$NAME" "" "$RADIUS" > "$LOG" 2>&1
  cat "$LOG"
done
echo "=== queue done $(date +%H:%M)"
