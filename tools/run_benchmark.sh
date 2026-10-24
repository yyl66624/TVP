#!/bin/bash
# Feature extraction + calibration for one benchmark, same protocol everywhere.
#   run_benchmark.sh <dataset_name> [dims] [positive_dist]
# dims defaults to round(N/15) clipped to [64, 4096] and rounded down to 32,
# the dimension rule in EXPERIMENTS.md.  positive_dist defaults to 25 m; pass
# 0.5 (CoVPR), 5 (NYC, ITLP-Indoor) or 2.5 explicitly for the indoor sets.
# This path uses equal weights and the dimension rule, so it reproduces the
# 15-benchmark suite, not the per-dataset tuned headline numbers (sqrt weights,
# e.g. MSLS 6 144/3 072/1 536) -- use run_eval.sh with explicit dims for those.
set -eu
NAME="$1"
DIMS="${2:-}"
THRESH="${3:-25}"
ROOT="${WORKSHOP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT/TF-VPR-main"

DATASET="$NAME" ./run_eval.sh "${NAME}_mvc" --tf_fusion_gem --tf_mean_view --save_features \
  --val_positive_dist_threshold "$THRESH" > "$ROOT/${NAME}_mvc.out" 2>&1
FEAT=$(ls -t test/"${NAME}"_mvc/*/features_hard_resize.npy | head -1)

if [ -z "$DIMS" ]; then
  N=$(.venv/bin/python -I -c "import numpy as np,sys; print(np.load(sys.argv[1], mmap_mode='r').shape[0])" "$FEAT")
  K=$(( N / 15 ))
  [ "$K" -lt 64 ] && K=64
  [ "$K" -gt 4096 ] && K=4096
  DIMS=$(( (K / 32) * 32 ))
  [ "$DIMS" -lt 64 ] && DIMS=64
fi

echo "$NAME dims ${DIMS} (positive radius ${THRESH} m):"
for fit in db all; do
  echo -n "  fit=$fit "
  .venv/bin/python ../tools/calib_postproc.py --features "$FEAT" \
      --eval_dataset_name "$NAME" --splits 17,13,1 --positive_dist "$THRESH" \
      --dims "$DIMS,$DIMS,$DIMS" --fit "$fit" --weights 1,1,1 2>/dev/null | tail -1
done
