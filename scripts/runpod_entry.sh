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

# REGEN=1 forces shard regeneration; otherwise regenerate only when shards
# are missing. Shards are deterministic per committed split config (PLAN
# 5.2). NOTE: after a split-config change (e.g. n_docs bump) force REGEN=1
# for the first run -- stale shards silently train the wrong corpus.
if [ "${REGEN:-0}" = "1" ] || [ ! -f data/shards/train.jsonl ]; then
  for split in train val test-id test-ood; do
    "$PYTHON" -m track_a.needle_gen.corpus_writer \
      --config "track_a/needle_gen/splits/${split}.yaml" \
      --out "data/shards/${split}.jsonl"
  done
else
  echo "[runpod] shards exist; skipping regeneration (REGEN=0)"
fi

echo "[runpod] training: $CONFIG"
"$PYTHON" -m track_a.train --config "$CONFIG" "$@"

RUN_DIR=$(ls -dt runs/*/ | head -n 1)
RUN_DIR="${RUN_DIR%/}"
# Track A checkpoint selection: final.pt, not best.pt. Val CE rises exactly
# when the model starts memorizing the facts -- the capability under test --
# so early-stopping on val CE picks a grammar-only checkpoint (G1 diagnosis).
CKPT="$RUN_DIR/final.pt"
echo "[runpod] run dir: $RUN_DIR | checkpoint: $CKPT"

# Scoped eval: primary needle metrics over 30 docs, 2 QA probes/doc.
# The QA probe is diagnostic-only (PLAN Q5+); the full battery's 8 Qs x 5
# ratios per doc is the slow part. Scope matches the G1-diagnosis runs so
# results are comparable; override with EVAL_ARGS for a full battery.
EVAL_ARGS="${EVAL_ARGS:---max-docs 30 --qa-per-doc 2}"
echo "[runpod] eval battery (test-id + test-ood) [${EVAL_ARGS}]"
"$PYTHON" -m track_a.eval --config "$CONFIG" --checkpoint "$CKPT" \
  --shards data/shards/test-id.jsonl data/shards/test-ood.jsonl \
  $EVAL_ARGS \
  --out "$RUN_DIR/eval_report.json"
echo "[runpod] done -> $RUN_DIR/eval_report.json"
