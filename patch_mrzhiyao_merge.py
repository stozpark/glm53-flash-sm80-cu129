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
    """Keep #54031 FlashMLA plumbing, import only Mrzhiyao's SM80 tuning."""
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
            + "        return [MultipleOf(64)]\n"
        )
        text = replace_once(text, anchor, addition, "MultipleOf(64) policy")
    return text


def patch_kernel_long_context(text: str) -> str:
    """Promote KV element offsets to int64.

    At ~1M context the flat KV element offset can exceed int32 even though the
    row index itself still fits. This is the exact correctness hardening from
    wtdcode/vllm-backport commit 36c83fdc.
    """
    replacements = (
        (
            "indices[None, :] * stride_kv_token",
            "indices[None, :].to(tl.int64) * stride_kv_token",
        ),
        (
            "indices[:, None] * stride_kv_token",
            "indices[:, None].to(tl.int64) * stride_kv_token",
        ),
    )
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
    expected = text.count(".to(tl.int64) * stride_kv_token")
    if expected != 3:
        die(f"sparse MLA int64 row-offset hardening: expected 3 sites, found {expected}")
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

    # Start from the A800-validated Mrzhiyao kernel, then apply the later
    # long-context int64 row-offset correctness fix from wtdcode.
    shutil.copy2(mrz_kernel, kernel)
    ktext = patch_kernel_long_context(kernel.read_text(encoding="utf-8"))
    kernel.write_text(ktext, encoding="utf-8")
    print(f"[mrzhiyao] kernel + int64 long-context hardening -> {kernel}")

    text = backend.read_text(encoding="utf-8")
    backend.write_text(patch_backend(text), encoding="utf-8")
    print(f"[mrzhiyao] backend tuning -> {backend}")


if __name__ == "__main__":
    main()
