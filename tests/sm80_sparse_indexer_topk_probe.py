#!/usr/bin/env python3
"""A100 no-weight correctness probe for sparse-indexer Top-K backends.

Run inside an already-built vLLM 0.30 SM80 SIF.  No model weights are needed.
It stresses the threshold-bin overflow class from upstream vLLM #55314 using
strictly ordered, tightly clustered positive fp32 scores (DSA logits are
ReLU-weighted and can be highly clustered).

The probe compares:
  * vLLM top_k_per_row_prefill (actual prefill path)
  * vLLM persistent_topk (A100 auto decode path for topk=2048)
  * FlashInfer top_k_ragged_transform (candidate safe decode/prefill fallback)
against exact torch.topk selected index sets.
"""
from __future__ import annotations

import torch

import vllm._custom_ops as ops

TOPK = 2048
WORKSPACE_BYTES = 1024 * 1024


def make_clustered(rows: int, width: int, device: torch.device) -> torch.Tensor:
    # Consecutive positive float32 bit patterns around 1.0: all values are
    # distinct but share a very coarse radix prefix, stressing threshold-bin
    # refinement without tie ambiguity.
    base = torch.tensor([1.0], dtype=torch.float32, device=device).view(torch.int32)[0]
    bits = base + torch.arange(width, dtype=torch.int32, device=device)
    values = bits.view(torch.float32)
    g = torch.Generator(device=device).manual_seed(20260930 + width)
    rows_out = []
    for r in range(rows):
        perm = torch.randperm(width, generator=g, device=device)
        # Tiny row-dependent offset keeps rows independent while preserving
        # strict ordering inside each row.
        rows_out.append(values[perm] + r * 2.0 ** -18)
    return torch.stack(rows_out).contiguous()


def exact_indices(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    rows = []
    for r in range(logits.shape[0]):
        n = int(lengths[r].item())
        rows.append(
            logits[r, :n].topk(TOPK, largest=True, sorted=False).indices.to(torch.int32)
        )
    return torch.stack(rows)


def assert_same_set(
    name: str,
    logits: torch.Tensor,
    got: torch.Tensor,
    ref: torch.Tensor,
) -> None:
    got_s = got.sort(dim=1).values
    ref_s = ref.sort(dim=1).values
    if not torch.equal(got_s, ref_s):
        # Report value quality too; clustered values are strictly ordered, so
        # any set mismatch is a real ranking error rather than a tie choice.
        got_vals = torch.gather(logits, 1, got.clamp_min(0).to(torch.int64))
        ref_vals = torch.gather(logits, 1, ref.to(torch.int64))
        worst_gap = float(
            (ref_vals.sort(dim=1).values - got_vals.sort(dim=1).values)
            .abs()
            .max()
            .item()
        )
        mismatch = int((got_s != ref_s).sum().item())
        raise AssertionError(
            f"{name}: selected set differs from torch.topk "
            f"(index mismatches={mismatch}, max value gap={worst_gap:g})"
        )
    print(f"{name}=PASS")


def run_case(width: int, rows: int = 4) -> None:
    device = torch.device("cuda:0")
    logits = make_clustered(rows, width, device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    ref = exact_indices(logits, lengths)

    # 1) Exact path used by sparse-indexer PREFILL in vLLM 0.30.
    prefill_out = torch.full(
        (rows, TOPK), -1, dtype=torch.int32, device=device
    )
    starts = torch.zeros(rows, dtype=torch.int32, device=device)
    ends = lengths.clone()
    ops.top_k_per_row_prefill(
        logits,
        starts,
        ends,
        prefill_out,
        rows,
        logits.stride(0),
        logits.stride(1),
        TOPK,
    )
    torch.cuda.synchronize()
    assert_same_set(f"VLLM_PREFILL_TOPK_W{width}", logits, prefill_out, ref)

    # 2) Exact path selected by A100 'auto' decode for topk=2048.
    persistent_out = torch.full(
        (rows, TOPK), -1, dtype=torch.int32, device=device
    )
    workspace = torch.empty(
        WORKSPACE_BYTES, dtype=torch.uint8, device=device
    )
    torch.ops._C.persistent_topk(
        logits,
        lengths.reshape(rows, 1),
        persistent_out,
        workspace,
        TOPK,
        width,
    )
    torch.cuda.synchronize()
    assert_same_set(
        f"VLLM_PERSISTENT_TOPK_W{width}", logits, persistent_out, ref
    )

    # 3) Candidate Python-level escape hatch that avoids vLLM's C++ top-k ops.
    from flashinfer.topk import top_k_ragged_transform

    offsets = torch.zeros(rows, dtype=torch.int32, device=device)
    fi_out = top_k_ragged_transform(
        logits,
        offsets,
        lengths,
        TOPK,
    )
    torch.cuda.synchronize()
    assert_same_set(f"FLASHINFER_TOPK_W{width}", logits, fi_out, ref)


def main() -> None:
    assert torch.cuda.is_available()
    cap = torch.cuda.get_device_capability(0)
    print("device:", torch.cuda.get_device_name(0), "capability:", cap)
    assert cap == (8, 0), f"expected A100/A800 SM80, got {cap}"

    # 8192 exercises the short persistent path; 20000 the medium path; 40000
    # the long path. Four rows keep memory modest while forcing >stash clustered
    # candidates at the top-k boundary.
    for width in (8192, 20000, 40000):
        run_case(width)

    print("SM80_SPARSE_INDEXER_TOPK_PROBE=PASS")


if __name__ == "__main__":
    main()
