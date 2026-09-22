#!/usr/bin/env bash
# Production-oriented 2-node launcher for full zai-org/GLM-5.3 on
# 2 x (8 x A100/A800 SM80).  vLLM topology: TP=8 inside each node, PP=2
# across nodes.
#
# Usage on BOTH nodes:
#   export MODEL_HOST_PATH=/models/GLM-5.3
#   export SIF_PATH=/path/glm53-full-sm80-vllm029-cu129.sif
#   export MASTER_ADDR=10.0.0.10       # routable IP of node 0
#   export NODE_RANK=0                 # node 0; use 1 on node 1
#   bash ./serve_glm53_full_tp8_pp2.sh start
#
# Actions: start | stop | restart | status | logs | health | run
set -euo pipefail

ACTION="${1:-start}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ----------------------------- required ---------------------------------
MODEL_HOST_PATH="${MODEL_HOST_PATH:-}"
SIF_PATH="${SIF_PATH:-$(pwd)/glm53-full-sm80-vllm029-cu129.sif}"
MASTER_ADDR="${MASTER_ADDR:-}"
NODE_RANK="${NODE_RANK:-}"

# ------------------------------ serving ---------------------------------
PORT="${PORT:-8200}"
MASTER_PORT="${MASTER_PORT:-29501}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.3}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Conservative first production profile. Raise only after the A100 runtime
# baseline is clean. With BF16 MLA KV, long-context concurrency is expensive.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"

# Current SM80 port supports BF16 main MLA KV. The DSA indexer side-cache
# remains FP8. Do not change this to fp8/fp8_e4m3 until an SM80 reader is added.
KV_CACHE_DTYPE="bfloat16"

# Correctness-safe defaults for the first real A100 deployment.
PREFIX_CACHING="${PREFIX_CACHING:-0}"   # 1 after baseline validation
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"     # 1 only for debugging
NUMA_BIND="${NUMA_BIND:-0}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"

# MTP under PP=2 is intentionally blocked in this branch. vLLM 0.29.0's
# NVIDIA DeepseekV32MTP does not declare SupportsPP; port/validate MTP+PP
# separately before enabling it.
SPEC_TOKENS="${SPEC_TOKENS:-0}"

# Optional API authentication.
API_KEY="${API_KEY:-}"

# Network interface. Set NET_IFACE explicitly on multi-NIC/IB hosts.
NET_IFACE="${NET_IFACE:-}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-}"

CACHE_DIR="${CACHE_DIR:-/tmp/vllm_glm53_full_sm80}"
RUN_DIR="${RUN_DIR:-/tmp/glm53-full-sm80}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
PID_FILE="${PID_FILE:-${RUN_DIR}/glm53.rank${NODE_RANK:-x}.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/glm53.rank${NODE_RANK:-x}.log}"

die() { echo "ERROR: $*" >&2; exit 1; }

detect_iface() {
  if [[ -n "${NET_IFACE}" ]]; then
    echo "${NET_IFACE}"
    return
  fi
  local iface
  iface="$(ip -o route get "${MASTER_ADDR}" 2>/dev/null | awk '{
    for (i=1; i<=NF; i++) if ($i=="dev") {print $(i+1); exit}
  }')"
  [[ -n "${iface}" ]] || die "cannot auto-detect network interface; set NET_IFACE"
  echo "${iface}"
}

iface_ip() {
  local iface="$1"
  ip -o -4 addr show dev "${iface}" scope global 2>/dev/null |
    awk '{split($4,a,"/"); print a[1]; exit}'
}

preflight() {
  [[ -n "${MODEL_HOST_PATH}" ]] || die "set MODEL_HOST_PATH"
  [[ -n "${MASTER_ADDR}" ]] || die "set MASTER_ADDR to node0 routable IP"
  [[ "${NODE_RANK}" == "0" || "${NODE_RANK}" == "1" ]] ||
    die "NODE_RANK must be 0 or 1"
  [[ -d "${MODEL_HOST_PATH}" ]] || die "model path not found: ${MODEL_HOST_PATH}"
  [[ -f "${MODEL_HOST_PATH}/config.json" ]] ||
    die "config.json not found under MODEL_HOST_PATH"
  [[ -f "${SIF_PATH}" ]] || die "SIF not found: ${SIF_PATH}"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  command -v ip >/dev/null || die "ip command not found"

  local ngpu
  ngpu="$(nvidia-smi -L | wc -l)"
  (( ngpu >= 8 )) || die "need at least 8 visible GPUs on this node; found ${ngpu}"

  if (( SPEC_TOKENS > 0 )); then
    die "SPEC_TOKENS>0 is not enabled for TP8xPP2 yet; use SPEC_TOKENS=0"
  fi

  local iface ipaddr
  iface="$(detect_iface)"
  ipaddr="$(iface_ip "${iface}")"
  [[ -n "${ipaddr}" ]] || die "no IPv4 address found on interface ${iface}"

  echo "node_rank=${NODE_RANK}"
  echo "master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "net_iface=${iface}"
  echo "local_ip=${ipaddr}"
  echo "gpus=${GPUS}"
  echo "model=${MODEL_HOST_PATH}"
  echo "sif=${SIF_PATH}"
}

runtime_bin() {
  if command -v apptainer >/dev/null 2>&1; then
    echo apptainer
  elif command -v singularity >/dev/null 2>&1; then
    echo singularity
  else
    die "apptainer/singularity not found"
  fi
}

build_args() {
  VLLM_ARGS=(
    vllm serve /models/GLM-5.3
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
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --block-size "${BLOCK_SIZE}"
    --attention-config '{"backend":"TRITON_MLA_SPARSE","indexer_kv_dtype":"fp8"}'

    --linear-backend marlin
    --moe-backend marlin
    --enable-expert-parallel

    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"

    --tool-call-parser glm47
    --reasoning-parser glm47
    --enable-auto-tool-choice
  )

  if [[ "${PREFIX_CACHING}" == "1" ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
  else
    VLLM_ARGS+=(--no-enable-prefix-caching)
  fi
  [[ "${ENFORCE_EAGER}" == "1" ]] && VLLM_ARGS+=(--enforce-eager)
  [[ "${NUMA_BIND}" == "1" ]] && VLLM_ARGS+=(--numa-bind)
  [[ "${DISABLE_LOG_REQUESTS}" == "1" ]] && VLLM_ARGS+=(--disable-log-requests)
  [[ -n "${API_KEY}" ]] && VLLM_ARGS+=(--api-key "${API_KEY}")

  if [[ "${NODE_RANK}" == "0" ]]; then
    VLLM_ARGS+=(--host 0.0.0.0 --port "${PORT}")
  else
    VLLM_ARGS+=(--headless)
  fi
}

run_server() {
  preflight
  mkdir -p "${CACHE_DIR}/hf" "${CACHE_DIR}/triton" "${CACHE_DIR}/vllm"            "${RUN_DIR}" "${LOG_DIR}"

  local iface local_ip rt
  iface="$(detect_iface)"
  local_ip="$(iface_ip "${iface}")"
  rt="$(runtime_bin)"
  build_args

  ENV_ARGS=(
    --env CUDA_VISIBLE_DEVICES="${GPUS}"
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env HF_HOME=/glm53_cache/hf
    --env XDG_CACHE_HOME=/glm53_cache
    --env TRITON_CACHE_DIR=/glm53_cache/triton
    --env VLLM_CACHE_ROOT=/glm53_cache/vllm
    --env VLLM_HOST_IP="${local_ip}"
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn
    --env VLLM_ENGINE_READY_TIMEOUT_S=3600
    --env PYTHONUNBUFFERED=1
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    --env NCCL_SOCKET_IFNAME="${iface}"
    --env GLOO_SOCKET_IFNAME="${iface}"
    --env NCCL_DEBUG="${NCCL_DEBUG}"
    --env NCCL_NVLS_ENABLE=0
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    --env VLLM_ENABLE_CUDA_COMPATIBILITY=1
    --env VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat
    --env LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/cuda/lib64
  )
  [[ -n "${NCCL_IB_HCA}" ]] && ENV_ARGS+=(--env NCCL_IB_HCA="${NCCL_IB_HCA}")
  [[ -n "${NCCL_IB_DISABLE}" ]] && ENV_ARGS+=(--env NCCL_IB_DISABLE="${NCCL_IB_DISABLE}")

  exec "${rt}" exec --nv     --bind "${MODEL_HOST_PATH}:/models/GLM-5.3:ro"     --bind "${CACHE_DIR}:/glm53_cache"     "${ENV_ARGS[@]}"     "${SIF_PATH}"     "${VLLM_ARGS[@]}"
}

is_running() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

start_server() {
  preflight
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"
  if is_running; then
    echo "already running: pid=$(cat "${PID_FILE}")"
    exit 0
  fi
  rm -f "${PID_FILE}"

  nohup setsid bash "${SCRIPT_PATH}" run >"${LOG_FILE}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"

  echo "started rank ${NODE_RANK}: pid=${pid}"
  echo "log: ${LOG_FILE}"
  if [[ "${NODE_RANK}" == "0" ]]; then
    echo "health: http://${MASTER_ADDR}:${PORT}/health"
  else
    echo "node1 is headless; check status/logs locally"
  fi
}

stop_server() {
  if ! [[ -f "${PID_FILE}" ]]; then
    echo "not running (no pid file)"
    exit 0
  fi
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${PID_FILE}"
    exit 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    # The detached launcher is a session/process-group leader. Kill the whole
    # group so vLLM local workers and the container runtime exit together.
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${PID_FILE}"
  echo "stopped rank ${NODE_RANK}"
}

status_server() {
  if is_running; then
    echo "RUNNING rank=${NODE_RANK} pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
  else
    echo "STOPPED rank=${NODE_RANK}"
    return 1
  fi
}

health_server() {
  if [[ "${NODE_RANK}" != "0" ]]; then
    status_server
    return
  fi
  curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null
  echo "HEALTHY http://127.0.0.1:${PORT}"
}

case "${ACTION}" in
  start)   start_server ;;
  stop)    stop_server ;;
  restart) stop_server || true; start_server ;;
  status)  status_server ;;
  logs)    touch "${LOG_FILE}"; tail -n 200 -f "${LOG_FILE}" ;;
  health)  health_server ;;
  run)     run_server ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs|health|run}" >&2
    exit 2
    ;;
esac
