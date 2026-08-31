#!/usr/bin/env bash
set -euo pipefail
SIF="${1:?usage: verify_sif_gpu.sh /path/to/image.sif}"
if command -v apptainer >/dev/null 2>&1; then R=apptainer; else R=singularity; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${R}" exec --nv "${SIF}" python - <<'PY'
import pathlib, torch, vllm
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
root = pathlib.Path(vllm.__file__).resolve().parent
print("vllm_root", root)
PY

VLLM_ROOT="$(${R} exec --nv "${SIF}" python -c 'import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')"
"${R}" exec --nv "${SIF}" python /opt/glm53-sm80/verify_static.py --vllm-root "${VLLM_ROOT}"
"${R}" exec --nv --bind "${ROOT}/tests:/glm53-tests:ro" "${SIF}" python /glm53-tests/test_nope512_kernel.py
"${R}" exec --nv --bind "${ROOT}/tests:/glm53-tests:ro" "${SIF}" python /glm53-tests/test_mqa_sm80.py
