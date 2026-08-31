#!/usr/bin/env bash
# GLM-5.3-Flash (zai-org/GLM-5.3-Flash, native FP8) on ALL 8x A800 80GB PCIe (sm_80), port 8008.
#
# Why these settings (A800 sm_80 has NO native FP8 tensor core):
#   - Weights are ~306 GiB -> need >=7 cards even before KV cache. TP=8 uses all GPUs;
#     per official MoE guidance TP must equal GPU count to avoid OOM from replicated dense layers.
#   - TEP8 = tensor-parallel-size 8 + --enable-expert-parallel (TP splits dense, EP splits experts).
#   - FP8 emulation on Ampere: VLLM_TEST_FORCE_FP8_MARLIN=1 + --moe-backend marlin
#     (weight-only FP8 Marlin kernels).
#   - --no-enable-flashinfer-autotune per official recipe for this image (autotune assumes Hopper+
#     FlashInfer attention paths; unsafe assumption on sm_80 for first boot).
#   - KV cache stays BF16 (default): FP8 KV cache needs SM89+/SM90+ for this model.
#   - MTP speculative decoding (model ships 1 draft layer), official value = 5 tokens.
#   - VLLM_ENGINE_READY_TIMEOUT_S=3600: NFS weight load of 306 GiB takes 15~30+ min.
#
# Overridable via env:
#   PORT=8008 GPUS="0,1,2,3,4,5,6,7" TP=8 NUM_SPEC_TOKENS=5 GMU=0.95 MAX_SEQS=32 MAX_MODEL_LEN=49152 EXTRA_ARGS="--enforce-eager"

set -euo pipefail

# Local-only overrides (git-ignored). Copy env.local.example -> env.local.
[ -f "$(dirname "$0")/env.local" ] && source "$(dirname "$0")/env.local"
MODEL_HOST_PATH="${MODEL_HOST_PATH:?set MODEL_HOST_PATH in env.local (see env.local.example)}"
MODEL_CONTAINER_PATH="/models/GLM-5.3-Flash"
PORT="${PORT:-8008}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"
GMU="${GMU:-0.95}"
MAX_SEQS="${MAX_SEQS:-32}"
# Keep the full 1M context: the GMU-0.95 pool (~1.87M tokens) serves one
# real 1M request with room to spare. Concurrency is not a priority for this
# deployment (MAX_SEQS=32); lower this for guaranteed many-way full-length.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm5.3-flash}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-${PORT}}"
CACHE_DIR="$(pwd)/vllm_cache"
# Patched image: adds TRITON_MLA_SPARSE backend for sm80 sparse MLA
# (see Dockerfile.glm53-sm80 / _port/patches_glm53_sm80).
IMAGE="${IMAGE:-vllm/vllm-openai:glm53-flash-sm80}"

mkdir -p "${CACHE_DIR}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
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
  "${IMAGE}" \
  "${MODEL_CONTAINER_PATH}" \
  --served-model-name ${SERVED_MODEL_NAME} \
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

echo ""
echo "Container '${CONTAINER_NAME}' started on port ${PORT} (TP${TP}, GPU ${GPUS})."
echo "Logs:   docker logs -f ${CONTAINER_NAME}"
echo "Probe:  curl -s http://localhost:${PORT}/v1/models    # ready after weight load (15-30 min)"
echo "Test:   ./test_api.sh ${PORT}"
