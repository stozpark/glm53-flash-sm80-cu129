#!/usr/bin/env bash
set -euo pipefail

SIF="${1:-${SIF_PATH:-$(pwd)/glm53-full-sm80-vllm030-cu130.sif}"
GPU="${GPU:-0}"

if [[ ! -f "${SIF}" ]]; then
  echo "ERROR: SIF not found: ${SIF}" >&2
  exit 1
fi

if command -v apptainer >/dev/null 2>&1; then
  R=apptainer
elif command -v singularity >/dev/null 2>&1; then
  R=singularity
else
  echo "ERROR: apptainer/singularity not found" >&2
  exit 1
fi

exec "${R}" exec --nv \
  --env CUDA_VISIBLE_DEVICES="${GPU}" \
  --env PYTHONUNBUFFERED=1 \
  "${SIF}" \
  python3 /opt/glm53-full-sm80/sm80_glm53_full_kernel_smoke.py
