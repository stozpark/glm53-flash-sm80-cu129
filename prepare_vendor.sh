#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${ROOT}/vendor"
mkdir -p "${VENDOR}/pr47629" "${VENDOR}/pr54031" "${VENDOR}/a800"

PR47629_SHA="064801dd2bc6ac2e265dc3fa1f5d803d71bde25d"
PR54031_SHA="b325d908656d05e2a650ec60666ccec6f4f3eb0c"

fetch() {
  local url="$1"
  local out="$2"
  echo "[fetch] ${url}"
  curl -fL --retry 3 --retry-delay 2 "${url}" -o "${out}"
}

# Latest reviewed SM80 MQA-logits fallback from PR #47629.
fetch \
  "https://raw.githubusercontent.com/thomaslwang/vllm/${PR47629_SHA}/vllm/v1/attention/ops/mqa_logits_triton.py" \
  "${VENDOR}/pr47629/mqa_logits_triton.py"

# NoPE-512 sparse MLA backend/kernel from PR #54031.
fetch \
  "https://raw.githubusercontent.com/ima-helikoptaaa/vllm/${PR54031_SHA}/vllm/v1/attention/backends/mla/triton_mla_sparse.py" \
  "${VENDOR}/pr54031/triton_mla_sparse.py"
fetch \
  "https://raw.githubusercontent.com/ima-helikoptaaa/vllm/${PR54031_SHA}/vllm/v1/attention/ops/triton_mla_sparse_kernel.py" \
  "${VENDOR}/pr54031/triton_mla_sparse_kernel.py"

# A800 project is used ONLY for the SM80-specific GLM-5.3 KPool compressor.
# Pin the current master commit when possible, then record it in MANIFEST.txt.
A800_GIT="https://gitee.com/kill-life/glm5.3-flash-deployment-a800.git"
A800_SHA="${A800_SHA:-$(git ls-remote "${A800_GIT}" refs/heads/master | awk '{print $1}' | head -1 || true)}"
if [[ -z "${A800_SHA}" ]]; then
  echo "ERROR: could not resolve A800 project master SHA" >&2
  exit 1
fi
A800_RAW="https://gitee.com/kill-life/glm5.3-flash-deployment-a800/raw/${A800_SHA}/_port/patches_glm53_sm80/vllm"
fetch \
  "${A800_RAW}/models/glm5next/nvidia/ops/kpool_compress.py" \
  "${VENDOR}/a800/kpool_compress.py"
# Keep the A800 KPool indexer only as a reference; patch_runtime.py does not copy it.
fetch \
  "${A800_RAW}/model_executor/layers/sparse_attn_indexer_kpool.py" \
  "${VENDOR}/a800/sparse_attn_indexer_kpool.reference.py"

{
  echo "PR47629_SHA=${PR47629_SHA}"
  echo "PR54031_SHA=${PR54031_SHA}"
  echo "A800_SHA=${A800_SHA}"
  echo
  sha256sum \
    "${VENDOR}/pr47629/mqa_logits_triton.py" \
    "${VENDOR}/pr54031/triton_mla_sparse.py" \
    "${VENDOR}/pr54031/triton_mla_sparse_kernel.py" \
    "${VENDOR}/a800/kpool_compress.py" \
    "${VENDOR}/a800/sparse_attn_indexer_kpool.reference.py"
} > "${VENDOR}/MANIFEST.txt"

echo
cat "${VENDOR}/MANIFEST.txt"
echo
echo "Vendor bundle prepared at ${VENDOR}"
