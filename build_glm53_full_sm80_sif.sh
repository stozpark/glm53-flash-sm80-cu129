#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-glm53-full-sm80-vllm029-cu129.sif}"
DEF="${DEF:-Singularity.glm53-full-sm80.def}"
if command -v apptainer >/dev/null 2>&1; then R=apptainer; elif command -v singularity >/dev/null 2>&1; then R=singularity; else echo "ERROR: apptainer/singularity not found" >&2; exit 1; fi
# shellcheck disable=SC2086
"${R}" build ${BUILD_FLAGS:-} "${OUT}" "${DEF}"
echo "SIF=${OUT}"
