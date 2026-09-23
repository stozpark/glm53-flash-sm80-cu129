#!/usr/bin/env python3
"""A100/A800 (SM80) port for full zai-org/GLM-5.3 on vLLM 0.30.0 / CUDA 13.

Target architecture:
  GlmMoeDsaForCausalLM / glm_moe_dsa (DeepSeek-V3.2 DSA family)

This is intentionally separate from GLM-5.3-Flash.  Full GLM-5.3 does not use
KPool/KPoolTail; the Ampere blockers are the DSA FP8 indexer, sparse MLA and
DeepSeek-V3.2 fused FP8 quantization kernels.

The 2-node baseline deliberately uses VLLM_PP_LAYER_PARTITION=42,36 so each
pipeline stage starts its sparse-index sharing group on a full-indexer layer.
This avoids carrying DSA Top-K state through IntermediateTensors.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected one match, found {n}")
    return text.replace(old, new, 1)


def patch_registry(text: str) -> str:
    if "TRITON_MLA_SPARSE =" in text:
        return text
    anchor = '    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"\n'
    return replace_once(
        text,
        anchor,
        anchor
        + '    TRITON_MLA_SPARSE = (\n'
        + '        "vllm.v1.attention.backends.mla.triton_mla_sparse."\n'
        + '        "TritonMLASparseBackend"\n'
        + '    )\n',
        "registry: TRITON_MLA_SPARSE",
    )


def patch_cuda(text: str) -> str:
    marker = "AttentionBackendEnum.TRITON_MLA_SPARSE"
    if marker in text:
        return text
    old = """            sparse_tail = [
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
"""
    new = """            sparse_tail = [
                # SM80/Ampere fallback.  Hopper/Blackwell native sparse
                # backends remain higher-performance where supported, while
                # their supports_combination checks reject A100.
                AttentionBackendEnum.TRITON_MLA_SPARSE,
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
"""
    return replace_once(text, old, new, "cuda: sparse MLA priority")


def patch_indexer_metadata(text: str) -> str:
    # v0.30 bundles DeepGEMM, so importability alone is not enough on A100.
    text = text.replace(
        """from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
    native_next_n_supported,
)
""",
        """from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    is_deep_gemm_supported,
    native_next_n_supported,
)
""",
    )
    text = text.replace("has_deep_gemm()", "is_deep_gemm_supported()")
    if "has_deep_gemm" in text:
        raise RuntimeError("indexer metadata: stale has_deep_gemm reference remains")

    # split_decodes_and_prefills explicitly requires decode-first ordering.
    old = """        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        self.reorder_batch_threshold = None
"""
    new = """        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        # split_decodes_and_prefills assumes decode -> short-extend -> prefill.
        # Without a vote here, a decode behind a prefill can be scored by the
        # prefill indexer path and produce a different sparse Top-K.
        self.reorder_batch_threshold = self.decode_threshold
"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "self.reorder_batch_threshold = self.decode_threshold" not in text:
        raise RuntimeError("indexer metadata: decode-first reorder anchor not found")
    return text


def patch_deepseek_kernels(text: str) -> str:
    marker = "SM80_SOFTWARE_E4M3FN"
    if marker in text:
        return text

    import_anchor = "from vllm.utils.torch_utils import is_quantized_kv_cache\n"
    text = replace_once(
        text,
        import_anchor,
        import_anchor
        + "from vllm.v1.attention.ops.fp8_sm80 import _encode_e4m3fn_u8\n",
        "deepseek kernels: fp8_sm80 import",
    )

    old = """@triton.jit
def _fp8_ue8m0_quantize(vals):
    \"\"\"Quantize float32 values to FP8 E4M3 with a ue8m0 (power-of-2) scale.

    Returns (fp8_vals, scale) so the caller can store them or reuse the scale.
    \"\"\"
    vals = vals.to(tl.float32)
    amax = tl.max(tl.abs(vals))
    scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))
    fp8_vals = tl.div_rn(vals, scale).to(tl.float8e4nv)
    return fp8_vals, scale
"""
    new = """# SM80_SOFTWARE_E4M3FN: Triton cannot emit fp8e4nv conversions on
# Ampere.  Keep the exact E4M3FN bit pattern in uint8 storage and only view it
# as torch.float8_e4m3fn at Python boundaries.
@triton.jit
def _fp8_ue8m0_quantize(vals):
    \"\"\"Quantize to E4M3FN bytes with a ue8m0 (power-of-2) scale.\"\"\"
    vals = vals.to(tl.float32)
    amax = tl.max(tl.abs(vals))
    scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))
    fp8_bytes = _encode_e4m3fn_u8(tl.div_rn(vals, scale))
    return fp8_bytes, scale
"""
    text = replace_once(text, old, new, "deepseek kernels: ue8m0 software encode")

    # Keep the indexer cache pointer byte-addressed; storing uint8 E4M3 bytes
    # into an fp8 pointer would itself request an unsupported SM80 conversion.
    old = """        if indexer_k_cache.dtype == torch.uint8:
            indexer_k_cache = indexer_k_cache.view(torch.float8_e4m3fn)
"""
    new = """        # Keep raw uint8 storage on SM80.  _fp8_quant_and_cache_write writes
        # the E4M3FN bit pattern directly; the reader decodes the same bytes.
        if indexer_k_cache.dtype != torch.uint8:
            indexer_k_cache = indexer_k_cache.view(torch.uint8)
"""
    text = replace_once(text, old, new, "deepseek kernels: byte indexer cache")

    # The packed MQA-query branches are constexpr-dead with BF16 MLA KV, but
    # make them SM80-safe as well so enabling an fp8-query backend later does
    # not require another image.
    text = text.replace(
        "ql_nope_fp8 = (ql_nope / scale).to(tl.float8e4nv)",
        "ql_nope_fp8 = _encode_e4m3fn_u8(ql_nope / scale)",
    )
    text = text.replace(
        "(r1 / scale).to(tl.float8e4nv)",
        "_encode_e4m3fn_u8(r1 / scale)",
    )
    text = text.replace(
        "(r2 / scale).to(tl.float8e4nv)",
        "_encode_e4m3fn_u8(r2 / scale)",
    )

    old = """    if quantize_mqa:
        # fp8 path: pack [ql_nope; q_pe] into a single fp8 tensor.
        mqa_q_fp8 = torch.empty(
            q_pe.shape[0],
            q_pe.shape[1],
            ql_nope.shape[2] + q_pe.shape[2],
            dtype=torch.float8_e4m3fn,
            device=q_pe.device,
        )
        # Placeholder; pid 0 packs q_pe into mqa_q_fp8 instead.
        q_pe_out = mqa_q_fp8
        mqa_q = mqa_q_fp8
    else:
        # bf16 path: only the RoPE'd q_pe is produced; ql_nope used directly.
        q_pe_out = torch.empty_like(q_pe)
        mqa_q_fp8 = q_pe_out  # unused placeholder for the fp8 pack pointer
        mqa_q = q_pe_out

    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
"""
    new = """    if quantize_mqa:
        # Triton output is byte-addressed on SM80.  The consumer still sees
        # the canonical torch.float8_e4m3fn dtype through a zero-copy view.
        mqa_q_fp8 = torch.empty(
            q_pe.shape[0],
            q_pe.shape[1],
            ql_nope.shape[2] + q_pe.shape[2],
            dtype=torch.uint8,
            device=q_pe.device,
        )
        q_pe_out = mqa_q_fp8
        mqa_q = mqa_q_fp8.view(torch.float8_e4m3fn)
    else:
        q_pe_out = torch.empty_like(q_pe)
        mqa_q_fp8 = q_pe_out  # unused when QUANTIZE_MQA=False
        mqa_q = q_pe_out

    index_q_fp8_storage = torch.empty_like(index_q, dtype=torch.uint8)
    index_q_fp8 = index_q_fp8_storage.view(torch.float8_e4m3fn)
"""
    text = replace_once(text, old, new, "deepseek kernels: uint8 fused-q outputs")

    # CuTeDSL is a native-FP8 path.  Never select it on SM80.
    text = text.replace(
        "    if current_platform.is_cuda():\n"
        "        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (",
        "    if current_platform.is_cuda() and current_platform.supports_fp8():\n"
        "        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (",
        1,
    )

    # Triton receives byte storage, not an fp8 pointer.
    old = """        index_q_fp8,
        index_q_fp8.stride(0),
        index_q_fp8.stride(1),
        index_q_head_dim,
"""
    new = """        index_q_fp8_storage,
        index_q_fp8_storage.stride(0),
        index_q_fp8_storage.stride(1),
        index_q_head_dim,
"""
    text = replace_once(text, old, new, "deepseek kernels: fused-q byte pointer")

    # Baseline BF16 MLA KV keeps every remaining fp8 MLA-cache branch
    # constexpr-dead.  Fail loudly if the unconditional/indexer conversions
    # that caused the A100 crash survived.
    fused_q_start = text.index("def _fused_q_kernel(")
    fused_q_end = text.index("def fused_q(", fused_q_start)
    fused_q_body = text[fused_q_start:fused_q_end]
    if ".to(tl.float8e4nv)" in fused_q_body:
        raise RuntimeError("deepseek kernels: native fp8 cast remains in fused_q")
    helper = text[text.index("def _fp8_ue8m0_quantize"):text.index(
        "def _fp8_quant_and_cache_write"
    )]
    if "tl.float8e4nv" in helper:
        raise RuntimeError("deepseek kernels: native fp8 cast remains in index quantizer")
    return text


def patch_sparse_indexer(text: str) -> str:
    marker = "_sm80_fp8_fp4_mqa_logits"
    if marker in text:
        return text

    old_import = """from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
"""
    new_import = """from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits as _deepgemm_fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits as _deepgemm_fp8_fp4_paged_mqa_logits,
    is_deep_gemm_supported,
)
"""
    text = replace_once(text, old_import, new_import, "sparse indexer: DeepGEMM imports")

    common_import = (
        "from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton\n"
    )
    text = replace_once(
        text,
        common_import,
        common_import
        + "from vllm.v1.attention.ops.mqa_logits_triton import (\n"
        + "    fp8_mqa_logits_triton,\n"
        + "    fp8_paged_mqa_logits_triton,\n"
        + ")\n",
        "sparse indexer: Triton MQA import",
    )

    anchor = "MXFP4_BLOCK_SIZE = 32\n\n"
    helpers = r'''def _sm80_fp8_fp4_mqa_logits(
    q_pair,
    kv_pair,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    clean_logits=False,
):
    if is_deep_gemm_supported():
        return _deepgemm_fp8_fp4_mqa_logits(
            q_pair,
            kv_pair,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
        )
    q, q_scale = q_pair
    k, k_scale = kv_pair
    if q_scale is not None:
        raise RuntimeError("SM80 Triton DSA fallback does not support MXFP4")
    return fp8_mqa_logits_triton(
        q,
        (k, k_scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def _sm80_fp8_fp4_paged_mqa_logits(
    q_pair,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
    indices=None,
):
    if is_deep_gemm_supported():
        return _deepgemm_fp8_fp4_paged_mqa_logits(
            q_pair,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=clean_logits,
            indices=indices,
        )
    q, q_scale = q_pair
    if q_scale is not None:
        raise RuntimeError("SM80 Triton DSA fallback does not support MXFP4")
    # Baseline and flattened-spec decode use one effective context length per
    # request row.  The Triton fallback consumes that final effective length.
    if context_lens.ndim == 2:
        context_lens = context_lens[:, -1].contiguous()
    return fp8_paged_mqa_logits_triton(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len=max_model_len,
        clean_logits=clean_logits,
    )


'''
    text = replace_once(text, anchor, anchor + helpers, "sparse indexer: fallback helpers")

    text = text.replace(
        "logits = fp8_fp4_mqa_logits(",
        "logits = _sm80_fp8_fp4_mqa_logits(",
    )
    text = text.replace(
        "logits = fp8_fp4_paged_mqa_logits(",
        "logits = _sm80_fp8_fp4_paged_mqa_logits(",
    )

    old_gate = """        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )
"""
    new_gate = """        if current_platform.is_cuda() and not is_deep_gemm_supported():
            if self.use_fp4_cache:
                raise RuntimeError(
                    "SM80 Triton DSA fallback supports FP8 indexer cache only"
                )
            logger.warning_once(
                "DeepGEMM is unavailable on this architecture; "
                "using SM80 Triton DSA indexer fallback."
            )
"""
    text = replace_once(text, old_gate, new_gate, "sparse indexer: DeepGEMM gate")

    # JIT warmup must not instantiate a float8 output store on SM80.
    old = """            pack_dtype = torch.uint8 if use_fp4_cache else current_platform.fp8_dtype()
            _PACK_SEQ_TRITON_KERNEL.register_warmup(
                dtype=pack_dtype,
                pad_value=0 if use_fp4_cache else -float("inf"),
            )
"""
    new = """            pack_as_bytes = use_fp4_cache or not is_deep_gemm_supported()
            pack_dtype = torch.uint8 if pack_as_bytes else current_platform.fp8_dtype()
            _PACK_SEQ_TRITON_KERNEL.register_warmup(
                dtype=pack_dtype,
                pad_value=0 if pack_as_bytes else -float("inf"),
            )
"""
    text = replace_once(text, old, new, "sparse indexer: SM80 pack warmup")

    old = """            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
"""
    new = """            else:
                if is_deep_gemm_supported():
                    padded_q_quant_decode_tokens = pack_seq_triton(
                        q_quant[:num_decode_tokens], decode_lens
                    )
                else:
                    # Avoid fp32 -> fp8 Triton stores on SM80: copy exact E4M3
                    # bytes and restore the dtype with a zero-copy view.
                    packed_u8 = pack_seq_triton(
                        q_quant[:num_decode_tokens].view(torch.uint8),
                        decode_lens,
                        pad_value=0,
                    )
                    padded_q_quant_decode_tokens = packed_u8.view(q_quant.dtype)
                padded_q_scale = None
"""
    text = replace_once(text, old, new, "sparse indexer: byte decode padding")

    if "has_deep_gemm()" in text:
        raise RuntimeError("sparse indexer: stale has_deep_gemm() remains")
    return text


def patch_piecewise_kv_binding(text: str) -> str:
    """Backport the DeepSeek-V3.2 PIECEWISE CUDA-graph KV binding fix.

    During PIECEWISE capture, attention metadata is absent but the persistent
    slot-mapping buffers and bound KV caches must still be passed to
    fused_norm_rope.  Passing None bakes "never write KV" into the captured
    graph and can silently corrupt decode after the first token.
    """
    marker = "SM80_PIECEWISE_KV_BINDING_FIX"
    if marker in text:
        return text

    old = """        if forward_context.attn_metadata is None or self.use_pcp:
            mla_kv_cache = None
            mla_k_scale = None
            indexer_k_cache = None
            mla_slot = None
            indexer_slot = None
        else:
            mla_kv_cache = None if hisparse_cache is not None else self.kv_cache
            mla_k_scale = self._k_scale
"""

    new = """        # SM80_PIECEWISE_KV_BINDING_FIX: mirror the current upstream DSA
        # graph-capture rule.  Capture has no attention metadata, but the real
        # cache views and persistent slot buffers must be baked into the graph.
        if self.use_pcp:
            mla_kv_cache = None
            mla_k_scale = None
            indexer_k_cache = None
            mla_slot = None
            indexer_slot = None
        elif forward_context.attn_metadata is None:
            if (
                mla_slot is not None
                and hisparse_cache is None
                and self.kv_cache.numel() > 0
            ):
                mla_kv_cache = self.kv_cache
                mla_k_scale = self._k_scale
            else:
                mla_kv_cache = None
                mla_k_scale = None
                mla_slot = None

            if (
                indexer_slot is not None
                and self.indexer is not None
                and self.indexer.k_cache.kv_cache.numel() > 0
            ):
                indexer_k_cache = self.indexer.k_cache.kv_cache
            else:
                indexer_k_cache = None
                indexer_slot = None
        else:
            mla_kv_cache = None if hisparse_cache is not None else self.kv_cache
            mla_k_scale = self._k_scale
"""

    return replace_once(
        text, old, new, "deepseek attention: PIECEWISE KV binding"
    )


def patch_block_table(text: str) -> str:
    marker = "SM80_DSA_SELF_PAD"
    if marker in text:
        return text

    old = """        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
"""
    new = """        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
        # SM80_DSA_SELF_PAD: the indexer expands full-width rows.  Never leave
        # another request's stale block ids in the unused tail.
        end = start + num_blocks
        if num_blocks > 0 and end < self.block_table.np.shape[1]:
            self.block_table.np[row_idx, end:] = self.block_table.np[row_idx, end - 1]
"""
    text = replace_once(text, old, new, "block table: self-pad tail")

    old = """        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
"""
    new = """        # Move the padded tail too; a prefix-only copy would retain the
        # previous target row's stale tail.
        block_table_np[tgt] = block_table_np[src]
        self.num_blocks_per_row[tgt] = num_blocks
"""
    text = replace_once(text, old, new, "block table: full-row move")
    return text


def patch_file(path: Path, fn) -> None:
    old = path.read_text(encoding="utf-8")
    new = fn(old)
    path.write_text(new, encoding="utf-8")
    print(f"[glm53-full-sm80-v030] patched {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument(
        "--vendor",
        default=str(Path(__file__).resolve().parent / "vendor/glm53-full-sm80"),
    )
    args = ap.parse_args()
    root = Path(args.vllm_root).resolve()
    vendor = Path(args.vendor).resolve()
    if not (root / "__init__.py").exists():
        raise RuntimeError(f"not a vllm package root: {root}")

    for rel in (
        "v1/attention/ops/fp8_sm80.py",
        "v1/attention/ops/triton_mla_sparse_kernel.py",
        "v1/attention/ops/mqa_logits_triton.py",
        "v1/attention/backends/mla/triton_mla_sparse.py",
    ):
        src = vendor / rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[glm53-full-sm80-v030] installed {dst}")

    patch_file(root / "v1/attention/backends/registry.py", patch_registry)
    patch_file(root / "platforms/cuda.py", patch_cuda)
    patch_file(root / "v1/attention/backends/mla/indexer.py", patch_indexer_metadata)
    patch_file(root / "model_executor/layers/sparse_attn_indexer.py", patch_sparse_indexer)
    patch_file(root / "models/deepseek_v32/common/kernels.py", patch_deepseek_kernels)
    patch_file(root / "models/deepseek_v32/attention.py", patch_piecewise_kv_binding)
    patch_file(root / "v1/worker/block_table.py", patch_block_table)

    print("GLM53_FULL_SM80_V030_PATCH=PASS")


if __name__ == "__main__":
    main()
