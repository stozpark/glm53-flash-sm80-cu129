#!/usr/bin/env python3
"""One-time A100/A800 sparse-indexer Top-K qualification and backend selector.

No model weights are needed. The probe runs the exact vLLM 0.30 prefill path,
the A100 persistent decode path, and FlashInfer against torch.topk on inputs
covering the oversized radix-threshold-bin regression tracked by upstream
vLLM #55314. It prints shell-friendly backend recommendations so the same SIF
can fall back without a rebuild.
"""
from __future__ import annotations

import os
from collections.abc import Callable

import torch

TOPK = 2048
WORKSPACE_BYTES = 1024 * 1024


def _cluster_row(length: int, shift: int = 0) -> torch.Tensor:
    """Strictly ordered positive fp32 values packed into a narrow radix range."""
    device = torch.device("cuda:0")
    # 0x3f000000 is 0.5f. Consecutive FP32 bit patterns are all distinct and
    # deliberately share a coarse radix prefix for the regression class.
    bits = torch.arange(
        0x3F000000,
        0x3F000000 + length,
        dtype=torch.int32,
        device=device,
    )
    values = bits.view(torch.float32)
    if length:
        values = torch.roll(values, shifts=shift % length)
    return values


def _make_case(lengths: list[int], mixed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda:0")
    width = max(lengths)
    rows = torch.empty((len(lengths), width), dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(20260930 + width + len(lengths))
    for r, length in enumerate(lengths):
        if mixed and r > 0:
            rows[r, :length] = torch.randn(length, generator=gen, device=device)
        else:
            rows[r, :length] = _cluster_row(length, shift=37 * r + 11)
        if length < width:
            # Dirty tail: a correct ragged selector must never pick it.
            rows[r, length:] = 1.0e30
    return rows.contiguous(), torch.tensor(lengths, dtype=torch.int32, device=device)


def _reference(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = torch.full(
        (logits.shape[0], TOPK), -1, dtype=torch.int32, device=logits.device
    )
    for r in range(logits.shape[0]):
        n = int(lengths[r].item())
        k = min(TOPK, n)
        if k:
            out[r, :k] = logits[r, :n].topk(
                k, dim=-1, largest=True, sorted=False
            ).indices.to(torch.int32)
    return out


def _same_selected_set(got: torch.Tensor, ref: torch.Tensor, lengths: torch.Tensor) -> bool:
    for r in range(got.shape[0]):
        k = min(TOPK, int(lengths[r].item()))
        if k:
            if not torch.equal(
                got[r, :k].sort().values,
                ref[r, :k].sort().values,
            ):
                return False
        if not torch.all(got[r, k:] == -1):
            return False
    return True


def _persistent(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = torch.full(
        (logits.shape[0], TOPK), -1, dtype=torch.int32, device=logits.device
    )
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=logits.device)
    torch.ops._C.persistent_topk(
        logits,
        lengths,
        out,
        workspace,
        TOPK,
        logits.shape[1],
    )
    return out


def _flashinfer(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    from flashinfer.topk import top_k_ragged_transform

    offsets = torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)
    return top_k_ragged_transform(logits, offsets, lengths, TOPK)


def _prefill(logits: torch.Tensor, lengths: torch.Tensor, backend: str) -> torch.Tensor:
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _sm80_top_k_per_row_prefill,
    )

    out = torch.full(
        (logits.shape[0], TOPK), -1, dtype=torch.int32, device=logits.device
    )
    starts = torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)
    old = os.environ.get("VLLM_SM80_PREFILL_TOPK_BACKEND")
    os.environ["VLLM_SM80_PREFILL_TOPK_BACKEND"] = backend
    try:
        _sm80_top_k_per_row_prefill(logits, starts, lengths, out, TOPK)
    finally:
        if old is None:
            os.environ.pop("VLLM_SM80_PREFILL_TOPK_BACKEND", None)
        else:
            os.environ["VLLM_SM80_PREFILL_TOPK_BACKEND"] = old
    return out


# Mirrors the failure families from upstream vLLM #55314 and adds the deployed
# 131072-token ceiling. The 40-row cases exercise the large-batch selector.
CASES: list[tuple[str, list[int], bool]] = [
    ("tight_5000", [5000], False),
    ("tight_9407", [9407], False),
    ("tight_17802", [17802], False),
    ("state_reuse", [17802, 3000], True),
    ("large_batch_9000", [9000] * 40, False),
    ("large_batch_40000", [40000] * 40, False),
    ("max_context_131072", [131072] * 4, False),
]


def _qualify(name: str, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> bool:
    try:
        for case_name, lens, mixed in CASES:
            logits, lengths = _make_case(lens, mixed)
            ref = _reference(logits, lengths)
            got = fn(logits, lengths)
            torch.cuda.synchronize()
            if not _same_selected_set(got, ref, lengths):
                print(f"{name}:{case_name}=FAIL")
                return False
            print(f"{name}:{case_name}=PASS")
        return True
    except Exception as exc:
        print(f"{name}=ERROR:{type(exc).__name__}:{exc}")
        return False


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    cap = torch.cuda.get_device_capability(0)
    print("device:", torch.cuda.get_device_name(0), "capability:", cap)
    assert cap == (8, 0), f"expected A100/A800 SM80, got {cap}"

    persistent_ok = _qualify("DECODE_PERSISTENT", _persistent)
    flashinfer_ok = _qualify("DECODE_FLASHINFER", _flashinfer)

    # Prefill has different row-start semantics, so validate the exact patched
    # helper independently. Prefer stock vLLM when correct, then FlashInfer.
    prefill_vllm_ok = _qualify(
        "PREFILL_VLLM", lambda x, n: _prefill(x, n, "vllm")
    )
    prefill_flashinfer_ok = _qualify(
        "PREFILL_FLASHINFER", lambda x, n: _prefill(x, n, "flashinfer")
    )

    decode_backend = (
        "persistent" if persistent_ok else "flashinfer" if flashinfer_ok else "torch"
    )
    prefill_backend = (
        "vllm" if prefill_vllm_ok else "flashinfer" if prefill_flashinfer_ok else "torch"
    )

    print(f"SM80_DECODE_TOPK_BACKEND={decode_backend}")
    print(f"SM80_PREFILL_TOPK_BACKEND={prefill_backend}")
    print("SM80_SPARSE_INDEXER_TOPK_PROBE=PASS")


if __name__ == "__main__":
    main()
