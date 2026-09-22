#!/usr/bin/env bash
# Single-node 16-GPU profile. Use only on a host that exposes >=16 A100s.
set -euo pipefail
MODEL_HOST_PATH="${MODEL_HOST_PATH:-}"
[[ -n "${MODEL_HOST_PATH}" ]] || { echo "ERROR: set MODEL_HOST_PATH" >&2; exit 1; }
SIF_PATH="${SIF_PATH:-$(pwd)/glm53-full-sm80-vllm029-cu129.sif}"
MODEL_CONTAINER_PATH="/models/GLM-5.3"
CACHE_DIR="${CACHE_DIR:-/tmp/vllm_glm53_full_sm80}"
PORT="${PORT:-8200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-2}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
mkdir -p "${CACHE_DIR}"
if command -v apptainer >/dev/null 2>&1; then R=apptainer; else R=singularity; fi

exec "${R}" exec --nv   --bind "${MODEL_HOST_PATH}:${MODEL_CONTAINER_PATH}:ro"   --bind "${CACHE_DIR}:/glm53_cache"   --env HF_HUB_OFFLINE=1   --env TRANSFORMERS_OFFLINE=1   --env VLLM_ENABLE_CUDA_COMPATIBILITY=1   --env VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat   --env LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/cuda/lib64   "${SIF_PATH}" vllm serve "${MODEL_CONTAINER_PATH}"     --served-model-name glm5.3     --host 0.0.0.0 --port "${PORT}"     --trust-remote-code     --tensor-parallel-size 16     --dtype bfloat16     --kv-cache-dtype bfloat16     --attention-config '{"backend":"TRITON_MLA_SPARSE"}'     --linear-backend marlin     --moe-backend marlin     --enable-expert-parallel     --gpu-memory-utilization 0.90     --max-model-len "${MAX_MODEL_LEN}"     --max-num-seqs "${MAX_SEQS}"     --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"     --tool-call-parser glm47     --reasoning-parser glm45     --enable-auto-tool-choice     "$@"
