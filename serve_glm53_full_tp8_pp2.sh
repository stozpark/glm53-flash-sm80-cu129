#!/usr/bin/env bash
# Two-node / 16-GPU baseline for full zai-org/GLM-5.3 on A100 80GB.
# Run on both nodes with NODE_RANK=0 and NODE_RANK=1.
set -euo pipefail

MODEL_HOST_PATH="${MODEL_HOST_PATH:-}"
MASTER_ADDR="${MASTER_ADDR:-}"
NODE_RANK="${NODE_RANK:-}"
if [[ -z "${MODEL_HOST_PATH}" || -z "${MASTER_ADDR}" || -z "${NODE_RANK}" ]]; then
  echo "ERROR: set MODEL_HOST_PATH, MASTER_ADDR, NODE_RANK={0|1}" >&2
  exit 1
fi

SIF_PATH="${SIF_PATH:-$(pwd)/glm53-full-sm80-vllm029-cu129.sif}"
MODEL_CONTAINER_PATH="/models/GLM-5.3"
CACHE_DIR="${CACHE_DIR:-/tmp/vllm_glm53_full_sm80}"
PORT="${PORT:-8200}"
MASTER_PORT="${MASTER_PORT:-29501}"
GMU="${GMU:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-2}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm5.3}"

mkdir -p "${CACHE_DIR}/vllm" "${CACHE_DIR}/triton" "${CACHE_DIR}/hf"
if command -v apptainer >/dev/null 2>&1; then R=apptainer; else R=singularity; fi

ARGS=(
  vllm serve "${MODEL_CONTAINER_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --tensor-parallel-size 8
  --pipeline-parallel-size 2
  --distributed-executor-backend mp
  --nnodes 2
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
  --distributed-timeout-seconds 3600
  --dtype bfloat16
  --kv-cache-dtype bfloat16
  --attention-config '{"backend":"TRITON_MLA_SPARSE"}'
  --linear-backend marlin
  --moe-backend marlin
  --enable-expert-parallel
  --gpu-memory-utilization "${GMU}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_SEQS}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
  --tool-call-parser glm47
  --reasoning-parser glm45
  --enable-auto-tool-choice
)
if [[ "${NODE_RANK}" == "0" ]]; then
  ARGS+=(--host 0.0.0.0 --port "${PORT}")
else
  ARGS+=(--headless)
fi

exec "${R}" exec --nv   --bind "${MODEL_HOST_PATH}:${MODEL_CONTAINER_PATH}:ro"   --bind "${CACHE_DIR}:/glm53_cache"   --env CUDA_VISIBLE_DEVICES="${GPUS}"   --env HF_HUB_OFFLINE=1   --env TRANSFORMERS_OFFLINE=1   --env HF_HOME=/glm53_cache/hf   --env XDG_CACHE_HOME=/glm53_cache   --env TRITON_CACHE_DIR=/glm53_cache/triton   --env VLLM_CACHE_ROOT=/glm53_cache/vllm   --env VLLM_ENABLE_CUDA_COMPATIBILITY=1   --env VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat   --env LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/cuda/lib64   --env NCCL_NVLS_ENABLE=0   --env PYTHONUNBUFFERED=1   "${SIF_PATH}" "${ARGS[@]}" "$@"
