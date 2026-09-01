#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def die(msg: str) -> None:
    raise RuntimeError(msg)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        die(f"{label}: expected exactly one match, found {n}")
    return text.replace(old, new, 1)


def patch_backend(text: str) -> str:
    """Keep #54031 FlashMLA plumbing, import only Mrzhiyao's SM80 tuning.

    Mrzhiyao's final backend inherits XPUMLASparseImpl. We intentionally do
    not replace the cu129-adapted FlashMLASparseImpl plumbing because it owns
    chunked-prefill routing, index globalization and the NoPE-aware MQA path
    already validated by the pinned base/source audit.

    The useful backend-level change is allowing 64-multiple KV block sizes,
    rather than forcing exactly 64. This keeps GLM's indexer constraint while
    permitting 128 when explicitly selected/profiled.
    """
    if "MultipleOf" not in text:
        anchor = "from vllm.utils.platform_utils import num_compute_units\n"
        text = replace_once(
            text,
            anchor,
            anchor + "from vllm.v1.attention.backend import MultipleOf\n",
            "MultipleOf import",
        )

    if "return [MultipleOf(64)]" not in text:
        anchor = (
            "    @staticmethod\n"
            "    def get_name() -> str:\n"
            "        return \"TRITON_MLA_SPARSE\"\n"
        )
        addition = (
            anchor
            + "\n"
            + "    @staticmethod\n"
            + "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
            + "        # GLM sparse indexer requires a 64-aligned cache group.\n"
            + "        # Mrzhiyao's A800 deployment validated allowing larger\n"
            + "        # multiples (e.g. 128) instead of forcing exactly 64.\n"
            + "        return [MultipleOf(64)]\n"
        )
        text = replace_once(text, anchor, addition, "MultipleOf(64) policy")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--vendor", required=True)
    args = ap.parse_args()

    root = Path(args.vllm_root).resolve()
    vendor = Path(args.vendor).resolve()
    backend = root / "v1/attention/backends/mla/triton_mla_sparse.py"
    kernel = root / "v1/attention/ops/triton_mla_sparse_kernel.py"
    mrz_kernel = vendor / "mrzhiyao/triton_mla_sparse_kernel.py"

    if not backend.exists():
        die(f"missing backend: {backend}")
    if not mrz_kernel.exists():
        die(f"missing pinned Mrzhiyao kernel: {mrz_kernel}")

    # Exact A800-validated kernel snapshot. It preserves the same public
    # triton_mla_sparse_attention() interface used by #54031 while carrying
    # the final 512/576 geometry cleanup and production-tested split-KV path.
    shutil.copy2(mrz_kernel, kernel)
    print(f"[mrzhiyao] kernel -> {kernel}")

    text = backend.read_text(encoding="utf-8")
    backend.write_text(patch_backend(text), encoding="utf-8")
    print(f"[mrzhiyao] backend tuning -> {backend}")


if __name__ == "__main__":
    main()
