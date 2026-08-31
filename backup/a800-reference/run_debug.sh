#!/usr/bin/env bash
# Foreground debug launcher (same config as deploy_glm53_flash_a800.sh, --rm + foreground).
# Use this to watch ALL startup logs directly; Ctrl+C to stop.
# Extra args are appended, e.g.: ./run_debug.sh --enforce-eager --max-model-len 32768

set -euo pipefail

[ -f "$(dirname "$0")/env.local" ] && source "$(dirname "$0")/env.local"
MODEL_HOST_PATH="${MODEL_HOST_PATH:?set MODEL_HOST_PATH in env.local (see env.local.example)}"
MODEL_CONTAINER_PATH="/models/GLM-5.3-Flash"
PORT="${PORT:-8007}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"
GMU="${GMU:-0.95}"
MAX_SEQS="${MAX_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
CACHE_DIR="$(pwd)/vllm_cache"

mkdir -p "${CACHE_DIR}"

docker run --rm \
  --name glm53-flash-debug \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=${GPUS} \
  --privileged --ipc=host \
  -p ${PORT}:8000 \
  -v "${MODEL_HOST_PATH}:${MODEL_CONTAINER_PATH}:ro" \
  -v "${CACHE_DIR}:/root/.cache/vllm" \
  -v "${CACHE_DIR}/triton:/root/.cache/triton" \
  -v "${CACHE_DIR}/tilelang:/root/.cache/tilelang" \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTHONUNBUFFERED=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  vllm/vllm-openai:glm53-flash-sm80 \
  "${MODEL_CONTAINER_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:-glm5.3-flash}" \
  --tensor-parallel-size ${TP} \
  --enable-expert-parallel \
  --moe-backend marlin \
  --gpu-memory-utilization ${GMU} \
  --max-num-seqs ${MAX_SEQS} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --attention-config '{"sparse_mla_force_mqa": true}' \
  "$@"
