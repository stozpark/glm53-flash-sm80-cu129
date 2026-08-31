#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${ROOT}/glm53-flash-sm80-cu129.sif}"
DEF="${ROOT}/Singularity.def"

if [[ ! -s "${ROOT}/vendor/MANIFEST.txt" ]]; then
  echo "ERROR: vendor bundle is missing." >&2
  echo "Run ./prepare_vendor.sh on an internet-connected machine first." >&2
  exit 1
fi

if command -v apptainer >/dev/null 2>&1; then
  BUILDER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  BUILDER=singularity
else
  echo "ERROR: neither apptainer nor singularity is installed." >&2
  exit 1
fi

BUILD_FLAGS_STR="${BUILD_FLAGS:-}"
read -r -a BUILD_FLAGS_ARR <<< "${BUILD_FLAGS_STR}"

echo "Builder: ${BUILDER}"
echo "Output : ${OUT}"
"${BUILDER}" build "${BUILD_FLAGS_ARR[@]}" "${OUT}" "${DEF}"

echo
echo "Built: ${OUT}"
"${BUILDER}" inspect "${OUT}" || true
