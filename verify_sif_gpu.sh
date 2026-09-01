#!/usr/bin/env bash
set -euo pipefail
SIF="${1:?usage: verify_sif_gpu.sh /path/to/image.sif}"
if command -v apptainer >/dev/null 2>&1; then R=apptainer; else R=singularity; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"

EXEC_ENV=(
  --env "CUDA_VISIBLE_DEVICES=${GPU}"
  --env VLLM_ENABLE_CUDA_COMPATIBILITY=1
  --env VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat
  --env LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/cuda/lib64
)

PYTHON_BIN="$("${R}" exec --nv "${EXEC_ENV[@]}" "${SIF}" sh -lc 'command -v python3 || command -v python')"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: neither python3 nor python exists inside ${SIF}" >&2
  exit 1
fi
echo "PYTHON_BIN=${PYTHON_BIN}"

"${R}" exec --nv "${EXEC_ENV[@]}" "${SIF}" "${PYTHON_BIN}" - <<'PY'
import pathlib, torch, vllm
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
root = pathlib.Path(vllm.__file__).resolve().parent
print("vllm_root", root)
assert torch.cuda.is_available(), "CUDA is not available inside the SIF"
assert torch.cuda.get_device_capability(0) == (8, 0), "expected SM80/A100"
PY

VLLM_ROOT="$("${R}" exec --nv "${EXEC_ENV[@]}" "${SIF}" "${PYTHON_BIN}" -c 'import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')"
"${R}" exec --nv "${EXEC_ENV[@]}" "${SIF}" "${PYTHON_BIN}" /opt/glm53-sm80/verify_static.py --vllm-root "${VLLM_ROOT}"

# Exercise the exact selector path needed by GLM-5.3 on A100 without loading
# model weights. This catches stale/unpatched SIFs before a multi-minute TP8
# model load.
"${R}" exec --nv "${EXEC_ENV[@]}" "${SIF}" "${PYTHON_BIN}" - <<'PY'
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

cfg = AttentionSelectorConfig(
    head_size=512,
    dtype=torch.bfloat16,
    kv_cache_dtype="bfloat16",
    block_size=None,
    use_mla=True,
    use_sparse=True,
)
backend = AttentionBackendEnum.TRITON_MLA_SPARSE
path = current_platform.get_attn_backend_cls(
    backend,
    attn_selector_config=cfg,
    num_heads=None,
)
print("EXPLICIT_SELECTOR_BACKEND=", backend.name)
print("EXPLICIT_SELECTOR_CLASS=", path)
assert path and "triton_mla_sparse" in path.lower(), path
print("EXPLICIT_TRITON_MLA_SPARSE_SELECTOR=PASS")
PY

"${R}" exec --nv "${EXEC_ENV[@]}" --bind "${ROOT}/tests:/glm53-tests:ro" "${SIF}" "${PYTHON_BIN}" /glm53-tests/test_nope512_kernel.py
"${R}" exec --nv "${EXEC_ENV[@]}" --bind "${ROOT}/tests:/glm53-tests:ro" "${SIF}" "${PYTHON_BIN}" /glm53-tests/test_mqa_sm80.py

echo "SIF_GPU_VERIFY=PASS"
