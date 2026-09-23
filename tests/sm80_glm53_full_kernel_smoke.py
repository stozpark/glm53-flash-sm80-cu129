#!/usr/bin/env python3
"""GPU smoke test for the full GLM-5.3 SM80 port.

Run this INSIDE the built SIF before loading the 753B checkpoint.  It forces
Triton JIT compilation of the exact A100-sensitive building blocks:
  * deepseek_v32 fused_norm_rope index-K FP8 cache writer
  * deepseek_v32 fused_q index-Q FP8 writer
  * Triton DSA prefill MQA logits
  * Triton DSA paged-decode MQA logits
  * Triton sparse MLA (576 -> 512)

No model weights are required.
"""
from __future__ import annotations

import math

import torch

from vllm.models.deepseek_v32.common import kernels as K
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    triton_mla_sparse_attention,
)

FP8 = torch.float8_e4m3fn
Q_LORA = 2048
KV_LORA = 512
ROPE_DIM = 64
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
EPS = 1e-6


def make_cos_sin(max_pos: int, rot_dim: int, device: torch.device) -> torch.Tensor:
    half = rot_dim // 2
    inv = 1.0 / (
        10000.0
        ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    f = torch.einsum("i,j->ij", t, inv)
    return torch.cat([f.cos(), f.sin()], dim=-1)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    dev = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(dev)
    print("device:", torch.cuda.get_device_name(dev))
    print("compute capability:", (major, minor))
    print("torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
    assert major == 8 and minor == 0, (
        "this smoke test is intended for A100/A800 SM80; "
        f"got sm_{major}{minor}"
    )

    torch.manual_seed(123)
    n = 2
    block_size = 64
    pos = torch.arange(n, device=dev, dtype=torch.int64)
    cos_sin = make_cos_sin(128, ROPE_DIM, dev)

    # ------------------------------------------------------------------
    # 1) Full index-K cache write through deepseek_v32 fused_norm_rope.
    # This is one of the two places that previously emitted fp8e4nv on SM80.
    # ------------------------------------------------------------------
    q_c = torch.randn(n, Q_LORA, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(n, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(n, ROPE_DIM, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(Q_LORA, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(KV_LORA, device=dev, dtype=torch.bfloat16)
    index_k = torch.randn(n, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
    index_kw = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    index_kb = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)

    idx_row_bytes = INDEX_HEAD_DIM + 4
    idx_cache = torch.zeros(
        1, block_size, idx_row_bytes, device=dev, dtype=torch.uint8
    )
    mla_cache = torch.zeros(
        1, block_size, KV_LORA + ROPE_DIM, device=dev, dtype=torch.bfloat16
    )
    slots = torch.arange(n, device=dev, dtype=torch.int64)
    topk = torch.full((n, 2048), 7, device=dev, dtype=torch.int32)

    q_norm = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        cos_sin,
        index_k,
        index_kw,
        index_kb,
        EPS,
        cos_sin,
        topk,
        slot_mapping=slots,
        indexer_slot_mapping=slots,
        indexer_k_cache=idx_cache,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype="bfloat16",
        mla_k_scale=None,
        has_indexer=True,
        index_rope_interleave=False,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(q_norm).all()
    assert (topk == -1).all(), "fused_norm_rope did not clear top-k buffer"

    # The cache is block-packed: all FP8 values first, then all fp32 scales.
    flat = idx_cache[0].reshape(-1)
    k_values = (
        flat[: block_size * INDEX_HEAD_DIM]
        .view(FP8)
        .reshape(block_size, INDEX_HEAD_DIM)
    )
    k_scales = flat[block_size * INDEX_HEAD_DIM :].view(torch.float32)
    assert torch.isfinite(k_values[:n].to(torch.float32)).all()
    assert torch.isfinite(k_scales[:n]).all() and torch.all(k_scales[:n] > 0)
    print("SM80_FUSED_NORM_ROPE_INDEX_K=PASS")

    # ------------------------------------------------------------------
    # 2) Full index-Q path through fused_q. Baseline BF16 MLA means
    # quantize_mqa=False, but index-Q itself is always FP8.
    # ------------------------------------------------------------------
    q_pe = torch.randn(n, 8, ROPE_DIM, device=dev, dtype=torch.bfloat16)
    ql_nope = torch.randn(n, 8, KV_LORA, device=dev, dtype=torch.bfloat16)
    index_q = torch.randn(
        n, INDEX_HEADS, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16
    )
    index_w = torch.randn(n, INDEX_HEADS, device=dev, dtype=torch.float32)
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32)

    iq_fp8, iw_out, q_pe_roped = K.fused_q(
        pos,
        q_pe,
        cos_sin,
        index_q,
        cos_sin,
        ql_nope,
        q_scale,
        index_w,
        INDEX_HEAD_DIM**-0.5,
        INDEX_HEADS**-0.5,
        has_indexer=True,
        index_rope_interleave=False,
        quantize_mqa=False,
    )
    torch.cuda.synchronize()
    assert iq_fp8.dtype == FP8
    assert q_pe_roped.dtype == torch.bfloat16
    assert torch.isfinite(iq_fp8.to(torch.float32)).all()
    assert torch.isfinite(iw_out).all()
    print("SM80_FUSED_Q_INDEX_Q=PASS")

    # ------------------------------------------------------------------
    # 3) DSA prefill MQA fallback.
    # ------------------------------------------------------------------
    ks = torch.zeros(n, device=dev, dtype=torch.int32)
    ke = torch.arange(1, n + 1, device=dev, dtype=torch.int32)
    logits = fp8_mqa_logits_triton(
        iq_fp8,
        (k_values, k_scales),
        iw_out,
        ks,
        ke,
        clean_logits=True,
    )
    torch.cuda.synchronize()
    assert logits.shape == (n, block_size)
    for row in range(n):
        assert torch.isfinite(logits[row, : row + 1]).all()
        assert torch.isneginf(logits[row, row + 1 :]).all()
    print("SM80_TRITON_MQA_PREFILL=PASS")

    # ------------------------------------------------------------------
    # 4) DSA paged decode MQA fallback.
    # ------------------------------------------------------------------
    q_decode = iq_fp8[:1].reshape(1, 1, INDEX_HEADS, INDEX_HEAD_DIM)
    context_lens = torch.tensor([n], dtype=torch.int32, device=dev)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    paged = fp8_paged_mqa_logits_triton(
        q_decode,
        idx_cache.unsqueeze(2),
        iw_out[:1],
        context_lens,
        block_table,
        max_model_len=block_size,
        clean_logits=True,
    )
    torch.cuda.synchronize()
    assert paged.shape == (1, block_size)
    assert torch.isfinite(paged[0, :n]).all()
    assert torch.isneginf(paged[0, n:]).all()
    print("SM80_TRITON_MQA_DECODE=PASS")

    # ------------------------------------------------------------------
    # 5) Sparse MLA kernel used by full GLM-5.3. 576-D Q/K, 512-D V output.
    # ------------------------------------------------------------------
    seq = 64
    q_sparse = torch.randn(1, 8, 576, device=dev, dtype=torch.bfloat16)
    kv_sparse = torch.randn(seq, 1, 576, device=dev, dtype=torch.bfloat16)
    indices = torch.arange(seq, device=dev, dtype=torch.int32).reshape(1, 1, seq)
    out = triton_mla_sparse_attention(
        q_sparse,
        kv_sparse,
        indices,
        sm_scale=1.0 / math.sqrt(576),
        num_kv_splits=1,
    )
    torch.cuda.synchronize()
    assert out.shape == (1, 8, 512)
    assert torch.isfinite(out).all()
    print("SM80_TRITON_MLA_SPARSE=PASS")

    print("GLM53_FULL_SM80_GPU_SMOKE=PASS")


if __name__ == "__main__":
    main()
