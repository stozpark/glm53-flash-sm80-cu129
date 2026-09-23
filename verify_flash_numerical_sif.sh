#!/usr/bin/env bash
set -euo pipefail

SIF="${1:?usage: verify_flash_numerical_sif.sh /path/to/glm53-sm80-4fa465c.sif}"
GPU="${GPU:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v apptainer >/dev/null 2>&1; then
  R=apptainer
elif command -v singularity >/dev/null 2>&1; then
  R=singularity
else
  echo "ERROR: apptainer/singularity not found" >&2
  exit 1
fi

EXEC_ENV=(
  --env "CUDA_VISIBLE_DEVICES=${GPU}"
  --env VLLM_ENABLE_CUDA_COMPATIBILITY=1
  --env VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat
  --env LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/usr/local/cuda/lib64
)

PYTHON_BIN="$("${R}" exec --nv "${EXEC_ENV[@]}" "${SIF}" sh -lc 'command -v python3 || command -v python')"
[[ -n "${PYTHON_BIN}" ]] || {
  echo "ERROR: python not found inside SIF" >&2
  exit 1
}

exec "${R}" exec --nv \
  "${EXEC_ENV[@]}" \
  --bind "${ROOT}/tests/test_flash_numerical_sm80.py:/tmp/test_flash_numerical_sm80.py:ro" \
  "${SIF}" \
  "${PYTHON_BIN}" /tmp/test_flash_numerical_sm80.py
