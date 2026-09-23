#!/usr/bin/env bash
# Optional single-node 16-GPU profile for full zai-org/GLM-5.3.
# The primary supported target of this branch is 2 nodes x 8 GPUs (TP8 x PP2).
set -euo pipefail

MODEL_HOST_PATH="${MODEL_HOST_PATH:-}"
[[ -n "${MODEL_HOST_PATH}" ]] || { echo "ERROR: set MODEL_HOST_PATH" >&2; exit 1; }
[[ -d "${MODEL_HOST_PATH}" ]] || { echo "ERROR: model path not found: ${MODEL_HOST_PATH}" >&2; exit 1; }

SIF_PATH="${SIF_PATH:-$(pwd)/glm53-full-sm80-vllm030-cu130.sif}"
MODEL_CONTAINER_PATH="/models/GLM-5.3"
PORT="${PORT:-8200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

[[ -f "${SIF_PATH}" ]] || { echo "ERROR: SIF not found: ${SIF_PATH}" >&2; exit 1; }
if command -v apptainer >/dev/null 2>&1; then
  R=apptainer
elif command -v singularity >/dev/null 2>&1; then
  R=singularity
else
  echo "ERROR: apptainer/singularity not found" >&2
  exit 1
fi

exec "${R}" exec --nv \
  --bind "${MODEL_HOST_PATH}:${MODEL_CONTAINER_PATH}:ro" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  "${SIF_PATH}" \
  vllm serve "${MODEL_CONTAINER_PATH}" \
    --served-model-name glm-5.3 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 16 \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --attention-config '{"backend":"TRITON_MLA_SPARSE"}' \
    --linear-backend marlin \
    --moe-backend marlin \
    --block-size "${BLOCK_SIZE}" \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tool-call-parser glm47 \
    --reasoning-parser glm47 \
    --enable-auto-tool-choice \
    "$@"
