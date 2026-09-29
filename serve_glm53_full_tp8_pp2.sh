#!/usr/bin/env bash
# Full zai-org/GLM-5.3 on 2 x (8 x A100/A800 SM80), vLLM 0.30.0.
#
# Minimal production launcher: official GLM-5.3 serving features plus only
# the options required by 2-node TP8xPP2 placement and the SM80 port.
#
# Required on both nodes:
#   MODEL_HOST_PATH=/models/GLM-5.3
#   SIF_PATH=/path/glm53-full-sm80-vllm030-cu130.sif
#   MASTER_ADDR=<routable node0 IPv4 or hostname>
#   NODE_RANK=0    # 1 on node1
#
# Optional on multi-NIC hosts:
#   NET_IFACE=eno1
#
# Actions: start | stop | restart | status | logs | health | run
set -euo pipefail

ACTION="${1:-start}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

MODEL_HOST_PATH="${MODEL_HOST_PATH:-}"
SIF_PATH="${SIF_PATH:-$(pwd)/glm53-full-sm80-vllm030-cu130.sif}"
MASTER_ADDR="${MASTER_ADDR:-}"
NODE_RANK="${NODE_RANK:-}"

PORT="${PORT:-8200}"
MASTER_PORT="${MASTER_PORT:-29501}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.3}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# A100 memory/correctness overrides. Everything else stays at the vLLM 0.30
# / official GLM-5.3 defaults.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
PP_LAYER_PARTITION="${PP_LAYER_PARTITION:-42,36}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"
EXPECTED_PORT_REVISION="glm53-full-sm80-cu130-v030-r20260929-10"

API_KEY="${API_KEY:-}"
NET_IFACE="${NET_IFACE:-}"

RUN_DIR="${RUN_DIR:-/tmp/glm53-full-sm80}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
PID_FILE="${PID_FILE:-${RUN_DIR}/glm53.rank${NODE_RANK:-x}.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/glm53.rank${NODE_RANK:-x}.log}"

die() { echo "ERROR: $*" >&2; exit 1; }

resolve_master_ipv4() {
  if [[ "${MASTER_ADDR}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "${MASTER_ADDR}"
    return
  fi
  local ipaddr
  ipaddr="$(getent ahostsv4 "${MASTER_ADDR}" 2>/dev/null |
    awk '$2 == "STREAM" {print $1; exit}')"
  [[ -n "${ipaddr}" ]] || die "cannot resolve MASTER_ADDR=${MASTER_ADDR} to IPv4"
  echo "${ipaddr}"
}

local_host_ip() {
  local master_ip ipaddr
  master_ip="$(resolve_master_ipv4)"

  # Rank 0 normally advertises MASTER_ADDR itself.
  if [[ "${NODE_RANK}" == "0" ]] &&
     ip -o -4 addr show scope global 2>/dev/null |
       awk -v target="${master_ip}" '{split($4,a,"/"); if (a[1]==target) found=1} END {exit !found}'; then
    echo "${master_ip}"
    return
  fi

  # Other ranks use the source address selected by the route to node0.
  ipaddr="$(ip -o route get "${master_ip}" 2>/dev/null |
    awk '{for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}')"
  [[ -n "${ipaddr}" && "${ipaddr}" != "127.0.0.1" ]] ||
    die "cannot determine this node's routable IP for MASTER_ADDR=${MASTER_ADDR}"
  echo "${ipaddr}"
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

preflight() {
  # Keep the launch contract deterministic. These host variables can override
  # the audited SM80 path or disable Marlin auto-selection.
  local var
  for var in \
    VLLM_ATTENTION_BACKEND \
    VLLM_KV_CACHE_LAYOUT \
    VLLM_USE_BREAKABLE_CUDAGRAPH \
    VLLM_USE_DEEP_GEMM \
    VLLM_MOE_USE_DEEP_GEMM \
    VLLM_BATCH_INVARIANT; do
    [[ -z "${!var:-}" ]] || die "unset conflicting host variable: ${var}"
  done

  [[ -n "${MODEL_HOST_PATH}" ]] || die "set MODEL_HOST_PATH"
  [[ -n "${MASTER_ADDR}" ]] || die "set MASTER_ADDR to node0 routable address"
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

  if [[ -n "${NET_IFACE}" ]]; then
    ip link show dev "${NET_IFACE}" >/dev/null 2>&1 ||
      die "NET_IFACE does not exist: ${NET_IFACE}"
  fi

  echo "node_rank=${NODE_RANK}"
  echo "master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "local_ip=$(local_host_ip)"
  echo "net_iface=${NET_IFACE:-auto}"
  echo "gpus=${GPUS}"
  echo "model=${MODEL_HOST_PATH}"
  echo "sif=${SIF_PATH}"
  echo "pp_layer_partition=${PP_LAYER_PARTITION}"
  echo "block_size=${BLOCK_SIZE}"
  echo "max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  echo "cudagraph_mode=${CUDAGRAPH_MODE}"
  echo "run_gpu_smoke=${RUN_GPU_SMOKE}"
}

validate_sif_runtime() {
  local rt
  rt="$(runtime_bin)"
  "${rt}" exec "${SIF_PATH}" python3 - <<'PY'
from importlib.metadata import version
from pathlib import Path

assert version("vllm") == "0.30.0", version("vllm")
revision = Path("/opt/glm53-full-sm80/PORT_REVISION").read_text().strip()
assert revision == "glm53-full-sm80-cu130-v030-r20260929-10", revision
root = Path("/usr/local/lib/python3.12/dist-packages/vllm")
if not root.exists():
    import importlib.util
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.origin is not None
    root = Path(spec.origin).resolve().parent

checks = {
    root / "v1/attention/backends/mla/triton_mla_sparse.py": [
        "def record_logical_topk_ready",
        "supports_dense_mha_prefill = False",
        "DeviceCapability(8, 0)",
    ],
    root / "models/deepseek_v32/common/kernels.py": [
        "SM80_SOFTWARE_E4M3FN",
        "index_q_fp8_storage",
    ],
    root / "models/deepseek_v32/attention.py": [
        "SM80_PIECEWISE_KV_BINDING_FIX",
        "topk_backend=self.indexer.indexer_op.topk_backend",
    ],
    root / "model_executor/layers/sparse_attn_indexer.py": [
        "_sm80_fp8_fp4_mqa_logits",
        "SM80_RAGGED_INDEXER_DECODE_FIX",
    ],
}
for path, markers in checks.items():
    text = path.read_text()
    for marker in markers:
        assert marker in text, f"{path}: missing {marker}"
assert Path("/opt/glm53-full-sm80/sm80_glm53_full_kernel_smoke.py").exists()
print("GLM53_FULL_SM80_SIF_RUNTIME_STATIC=PASS")
PY
}

run_gpu_smoke() {
  [[ "${RUN_GPU_SMOKE}" == "1" ]] || return 0
  local rt first_gpu
  rt="$(runtime_bin)"
  first_gpu="${GPUS%%,*}"
  echo "Running SM80 numerical/JIT smoke on physical GPU ${first_gpu}..."
  "${rt}" exec --nv \
    --env CUDA_VISIBLE_DEVICES="${first_gpu}" \
    --env PYTHONUNBUFFERED=1 \
    "${SIF_PATH}" \
    python3 /opt/glm53-full-sm80/sm80_glm53_full_kernel_smoke.py
}

build_args() {
  VLLM_ARGS=(
    vllm serve /models/GLM-5.3

    # Official GLM-5.3 recipe.
    --served-model-name "${SERVED_MODEL_NAME}"
    --tool-call-parser glm47
    --reasoning-parser glm47
    --enable-auto-tool-choice

    # 2 nodes x 8 A100 using vLLM's native multi-node multiprocessing.
    --distributed-executor-backend mp
    --tensor-parallel-size 8
    --pipeline-parallel-size 2
    --nnodes 2
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"

    # Only backend override required by the SM80 port. KV dtype is left at
    # vLLM's default "auto" -> model BF16; FP8 KV is not supported by this backend.
    --attention-config '{"backend":"TRITON_MLA_SPARSE"}'

    # A100-specific limits. v0.30 predates the all-backend block-size resolver
    # fix (#49845), so 64 is explicit. PIECEWISE graph profiling OOMs on the
    # full FP8 checkpoint; keep full graphs only for uniform decode.
    --max-model-len "${MAX_MODEL_LEN}"
    --block-size "${BLOCK_SIZE}"
    --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
    --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\"}"
  )

  [[ -n "${API_KEY}" ]] && VLLM_ARGS+=(--api-key "${API_KEY}")

  if [[ "${NODE_RANK}" == "0" ]]; then
    VLLM_ARGS+=(--port "${PORT}")
  else
    VLLM_ARGS+=(--headless)
  fi
}

run_server() {
  preflight
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"

  local rt host_ip
  rt="$(runtime_bin)"
  host_ip="$(local_host_ip)"
  validate_sif_runtime
  run_gpu_smoke
  build_args

  ENV_ARGS=(
    --env CUDA_VISIBLE_DEVICES="${GPUS}"
    # vLLM explicitly recommends a routable per-node address for multi-node.
    --env VLLM_HOST_IP="${host_ip}"
    # GLM-5.3 shares one DSA Top-K across four layers.  The default 39/39
    # split starts PP1 on a shared-index layer (39); 42/36 starts it on the
    # next full-indexer layer and removes cross-stage Top-K state.
    --env VLLM_PP_LAYER_PARTITION="${PP_LAYER_PARTITION}"
  )

  # Only pin NCCL/Gloo to an interface when the user asks for it. vLLM's
  # troubleshooting docs treat these as multi-NIC overrides, not defaults.
  if [[ -n "${NET_IFACE}" ]]; then
    ENV_ARGS+=(
      --env NCCL_SOCKET_IFNAME="${NET_IFACE}"
      --env GLOO_SOCKET_IFNAME="${NET_IFACE}"
    )
  fi

  # Optional NCCL tuning is inherited only when explicitly supplied.
  [[ -n "${NCCL_IB_HCA:-}" ]] &&
    ENV_ARGS+=(--env NCCL_IB_HCA="${NCCL_IB_HCA}")
  [[ -n "${NCCL_IB_DISABLE:-}" ]] &&
    ENV_ARGS+=(--env NCCL_IB_DISABLE="${NCCL_IB_DISABLE}")
  [[ -n "${NCCL_DEBUG:-}" ]] &&
    ENV_ARGS+=(--env NCCL_DEBUG="${NCCL_DEBUG}")

  exec "${rt}" exec --nv \
    --bind "${MODEL_HOST_PATH}:/models/GLM-5.3:ro" \
    "${ENV_ARGS[@]}" \
    "${SIF_PATH}" \
    "${VLLM_ARGS[@]}"
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
}

stop_server() {
  if ! [[ -f "${PID_FILE}" ]]; then
    echo "not running (no pid file)"
    return
  fi

  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    kill -0 "${pid}" 2>/dev/null &&
      { kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true; }
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
  *) echo "usage: $0 {start|stop|restart|status|logs|health|run}" >&2; exit 2 ;;
esac
