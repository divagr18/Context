#!/usr/bin/env bash
# Track A RunPod entrypoint (W9, PLAN 10): regenerate data -> train -> eval.
# One run per pod invocation; artifacts land in runs/<run_tag>-<seed>/.
#
# Usage:
#   scripts/runpod_entry.sh configs/train_tiny.yaml
#   scripts/runpod_entry.sh configs/train_tiny.yaml --seed 22
set -euo pipefail

CONFIG="${1:-configs/train_tiny.yaml}"
shift || true
PYTHON="${PYTHON:-python}"
# Linux pods: expandable_segments defragments. (Windows dev boxes: use
# backend:cudaMallocAsync instead -- see docs/RUNPOD.md section 6.)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[runpod] environment"
"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
assert torch.__version__.startswith("2.8.0"), "torch pin drift (PLAN Q9: 2.8.x cu128)"
print("cuda:", torch.version.cuda, "| device:", torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo "[runpod] regenerating shards from committed split configs (PLAN 5.2)"
for split in train val test-id test-ood; do
  "$PYTHON" -m track_a.needle_gen.corpus_writer \
    --config "track_a/needle_gen/splits/${split}.yaml" \
    --out "data/shards/${split}.jsonl"
done

echo "[runpod] training: $CONFIG"
"$PYTHON" -m track_a.train --config "$CONFIG" "$@"

RUN_DIR=$(ls -dt runs/*/ | head -n 1)
RUN_DIR="${RUN_DIR%/}"
CKPT="$RUN_DIR/final.pt"
if [ -f "$RUN_DIR/best.pt" ]; then CKPT="$RUN_DIR/best.pt"; fi
echo "[runpod] run dir: $RUN_DIR | checkpoint: $CKPT"

echo "[runpod] eval battery (test-id + test-ood)"
"$PYTHON" -m track_a.eval --config "$CONFIG" --checkpoint "$CKPT" \
  --shards data/shards/test-id.jsonl data/shards/test-ood.jsonl \
  --out "$RUN_DIR/eval_report.json"
echo "[runpod] done -> $RUN_DIR/eval_report.json"
