#!/usr/bin/env bash
set -euo pipefail

MODEL_HOST_PATH="${MODEL_HOST_PATH:-}"
if [[ -z "${MODEL_HOST_PATH}" ]]; then
  echo "ERROR: set MODEL_HOST_PATH=/path/to/GLM-5.3-Flash" >&2
  exit 1
fi
SIF_PATH="${SIF_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/glm53-flash-sm80-cu129.sif}"
MODEL_CONTAINER_PATH="/models/GLM-5.3-Flash"
CACHE_DIR="${CACHE_DIR:-/tmp/vllm_glm53_sm80}"
PORT="${PORT:-8200}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
GMU="${GMU:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-8}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm5.3-flash}"
LOG_FILE="${LOG_FILE:-glm53_sm80_initial_${PORT}.log}"

mkdir -p "${CACHE_DIR}/vllm" "${CACHE_DIR}/triton" "${CACHE_DIR}/hf"

if command -v apptainer >/dev/null 2>&1; then R=apptainer; else R=singularity; fi

# Correctness-first launch: MTP and prefix caching intentionally OFF.
# sparse_mla_force_mqa=true intentionally exercises the new sparse path even for short prompts.
nohup "${R}" exec --nv \
  --bind "${MODEL_HOST_PATH}:${MODEL_CONTAINER_PATH}:ro" \
  --bind "${CACHE_DIR}:/glm53_cache" \
  --env CUDA_VISIBLE_DEVICES="${GPUS}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HOME=/glm53_cache/hf \
  --env XDG_CACHE_HOME=/glm53_cache \
  --env TRITON_CACHE_DIR=/glm53_cache/triton \
  --env VLLM_CACHE_ROOT=/glm53_cache/vllm \
  --env VLLM_TEST_FORCE_FP8_MARLIN=1 \
  --env VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE \
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --env PYTHONUNBUFFERED=1 \
  --env VLLM_LOGGING_LEVEL=INFO \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  --env VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat \
  --env LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/cuda/lib64 \
  --env NCCL_NVLS_ENABLE=0 \
  "${SIF_PATH}" \
  vllm serve "${MODEL_CONTAINER_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --enable-expert-parallel \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --gpu-memory-utilization "${GMU}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --kv-cache-dtype bfloat16 \
    --attention-config '{"sparse_mla_force_mqa": true}' \
    --moe-backend marlin \
    --no-enable-flashinfer-autotune \
    "$@" \
  >"${LOG_FILE}" 2>&1 &

PID=$!
echo "PID=${PID}"
echo "LOG=${LOG_FILE}"
echo "tail -f ${LOG_FILE}"
