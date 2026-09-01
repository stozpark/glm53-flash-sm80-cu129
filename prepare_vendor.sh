#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${ROOT}/vendor"
BACKUP_VENDOR="${BACKUP_VENDOR:-}"
mkdir -p "${VENDOR}/pr47629" "${VENDOR}/pr54031" "${VENDOR}/backport" "${VENDOR}/mrzhiyao" "${VENDOR}/a800"

PR47629_SHA="064801dd2bc6ac2e265dc3fa1f5d803d71bde25d"
PR54031_SHA="b325d908656d05e2a650ec60666ccec6f4f3eb0c"
BACKPORT_SHA="0ef4bff219c098d48cf16d3d63ebef329e9b74b0"
MRZHIYAO_SHA="daeccb983ec84756cde7408b0e29161d492ea2c5"

copy_backup() {
  local rel="$1"
  local out="$2"
  [[ -n "${BACKUP_VENDOR}" && -s "${BACKUP_VENDOR}/${rel}" ]] || return 1
  echo "[backup] ${BACKUP_VENDOR}/${rel}"
  cp "${BACKUP_VENDOR}/${rel}" "${out}"
}

fetch() {
  local rel="$1"
  local url="$2"
  local out="$3"
  if copy_backup "${rel}" "${out}"; then
    return 0
  fi
  echo "[fetch] ${url}"
  curl -fL --retry 3 --retry-delay 2 "${url}" -o "${out}"
}

fetch "pr47629/mqa_logits_triton.py" \
  "https://raw.githubusercontent.com/thomaslwang/vllm/${PR47629_SHA}/vllm/v1/attention/ops/mqa_logits_triton.py" \
  "${VENDOR}/pr47629/mqa_logits_triton.py"

# Keep #54031's FlashMLA-based plumbing because it matches the glm53-flash-cu129
# API and already provides MQA/paged-MQA warmup. The final sparse kernel itself
# is replaced below by Mrzhiyao's A800-validated derivative.
fetch "pr54031/triton_mla_sparse.py" \
  "https://raw.githubusercontent.com/ima-helikoptaaa/vllm/${PR54031_SHA}/vllm/v1/attention/backends/mla/triton_mla_sparse.py" \
  "${VENDOR}/pr54031/triton_mla_sparse.py"
fetch "pr54031/triton_mla_sparse_kernel.py" \
  "https://raw.githubusercontent.com/ima-helikoptaaa/vllm/${PR54031_SHA}/vllm/v1/attention/ops/triton_mla_sparse_kernel.py" \
  "${VENDOR}/pr54031/triton_mla_sparse_kernel.py"

# A800 production-validated sparse MLA implementation. We intentionally import
# only the kernel and use the backend file as a reference: its XPU plumbing is
# not copied over the cu129 FlashMLA plumbing.
fetch "mrzhiyao/triton_mla_sparse_kernel.py" \
  "https://raw.githubusercontent.com/Mrzhiyao/glm53-a800-vllm/${MRZHIYAO_SHA}/overrides/vllm/v1/attention/ops/triton_mla_sparse_kernel.py" \
  "${VENDOR}/mrzhiyao/triton_mla_sparse_kernel.py"
fetch "mrzhiyao/triton_mla_sparse.py" \
  "https://raw.githubusercontent.com/Mrzhiyao/glm53-a800-vllm/${MRZHIYAO_SHA}/overrides/vllm/v1/attention/backends/mla/triton_mla_sparse.py" \
  "${VENDOR}/mrzhiyao/triton_mla_sparse.py"

# Ampere-tested GLM-5.3 KPool write path. This replaces the Gitee/A800
# dependency in the actual build path. A800 is retained only as historical
# comparison material in the source-backup branch.
fetch "backport/fp8_sm80.py" \
  "https://raw.githubusercontent.com/wtdcode/vllm-backport/${BACKPORT_SHA}/vllm/v1/attention/ops/fp8_sm80.py" \
  "${VENDOR}/backport/fp8_sm80.py"
fetch "backport/kpool_compress.py" \
  "https://raw.githubusercontent.com/wtdcode/vllm-backport/${BACKPORT_SHA}/vllm/models/glm5next/nvidia/ops/kpool_compress.py" \
  "${VENDOR}/backport/kpool_compress.py"

# Compatibility shim for the first-stage patch_runtime.py. It expects the old
# A800 path, but the content is now our pinned Ampere backport copy. No Gitee
# network dependency remains in the build path.
cp "${VENDOR}/backport/kpool_compress.py" "${VENDOR}/a800/kpool_compress.py"

{
  echo "PR47629_SHA=${PR47629_SHA}"
  echo "PR54031_SHA=${PR54031_SHA}"
  echo "BACKPORT_SHA=${BACKPORT_SHA}"
  echo "MRZHIYAO_SHA=${MRZHIYAO_SHA}"
  echo
  sha256sum \
    "${VENDOR}/pr47629/mqa_logits_triton.py" \
    "${VENDOR}/pr54031/triton_mla_sparse.py" \
    "${VENDOR}/pr54031/triton_mla_sparse_kernel.py" \
    "${VENDOR}/mrzhiyao/triton_mla_sparse.py" \
    "${VENDOR}/mrzhiyao/triton_mla_sparse_kernel.py" \
    "${VENDOR}/backport/fp8_sm80.py" \
    "${VENDOR}/backport/kpool_compress.py"
} > "${VENDOR}/MANIFEST.txt"

cat "${VENDOR}/MANIFEST.txt"
echo "Vendor bundle prepared at ${VENDOR}"
