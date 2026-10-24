#!/bin/bash
# Usage: ./run_eval.sh <run_name> [extra eval.py args...]
set -e
cd "$(dirname "$0")"
NAME=${1:?run name required}; shift || true
DATASET=${DATASET:-msls}
ROOT=${WORKSHOP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python eval.py \
  --eval_dataset_name "$DATASET" \
  --eval_datasets_folder "$ROOT/datasets" \
  --backbone dinov2 \
  --mode TF_VPR \
  --num_clusters 17 \
  --foundation_model_path "$ROOT/weights/dinov2_vitb14_pretrain.pth" \
  --infer_batch_size 32 \
  --num_workers 8 \
  --save_dir "$NAME" \
  "$@"
