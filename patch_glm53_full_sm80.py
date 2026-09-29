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

    return text


def patch_deepseek_kernels(text: str) -> str:
    marker = "SM80_SOFTWARE_E4M3FN_MQA"
    if marker in text:
        return text

    legacy_sm80 = "SM80_SOFTWARE_E4M3FN" in text
    import_anchor = "from vllm.utils.torch_utils import is_quantized_kv_cache\n"

    if not legacy_sm80:
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
# Ampere. Keep the exact E4M3FN bit pattern in uint8 storage and only view it
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

        old = """        if indexer_k_cache.dtype == torch.uint8:
            indexer_k_cache = indexer_k_cache.view(torch.float8_e4m3fn)
"""
        new = """        # Keep raw uint8 storage on SM80. _fp8_quant_and_cache_write writes
        # the E4M3FN bit pattern directly; the reader decodes the same bytes.
        if indexer_k_cache.dtype != torch.uint8:
            indexer_k_cache = indexer_k_cache.view(torch.uint8)
"""
        text = replace_once(text, old, new, "deepseek kernels: byte indexer cache")

        old = "    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)\n"
        new = """    index_q_fp8_storage = torch.empty_like(index_q, dtype=torch.uint8)
    index_q_fp8 = index_q_fp8_storage.view(torch.float8_e4m3fn)
"""
        text = replace_once(text, old, new, "deepseek kernels: uint8 index-Q output")

        text = text.replace(
            "    if current_platform.is_cuda():\n"
            "        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (",
            "    if current_platform.is_cuda() and current_platform.supports_fp8():\n"
            "        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (",
            1,
        )

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

    # SM80_SOFTWARE_E4M3FN_MQA: the fused MQA-query pack also contains native
    # fp8e4nv casts in v0.30. Keep those bytes architecture-neutral as well.
    text = replace_once(
        text,
        "                ql_nope_fp8 = (ql_nope / scale).to(tl.float8e4nv)\n",
        "                ql_nope_fp8 = _encode_e4m3fn_u8(ql_nope / scale)\n",
        "deepseek kernels: MQA NoPE software encode",
    )
    text = replace_once(
        text,
        "                        (r1 / scale).to(tl.float8e4nv),\n",
        "                        _encode_e4m3fn_u8(r1 / scale),\n",
        "deepseek kernels: MQA RoPE r1 software encode",
    )
    text = replace_once(
        text,
        "                        (r2 / scale).to(tl.float8e4nv),\n",
        "                        _encode_e4m3fn_u8(r2 / scale),\n",
        "deepseek kernels: MQA RoPE r2 software encode",
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
"""
    new = """    if quantize_mqa:
        # Byte-addressed storage avoids an unsupported Triton fp8e4nv store on SM80.
        mqa_q_fp8_storage = torch.empty(
            q_pe.shape[0],
            q_pe.shape[1],
            ql_nope.shape[2] + q_pe.shape[2],
            dtype=torch.uint8,
            device=q_pe.device,
        )
        mqa_q_fp8 = mqa_q_fp8_storage.view(torch.float8_e4m3fn)
        q_pe_out = mqa_q_fp8
        mqa_q = mqa_q_fp8
    else:
        # bf16 path: only the RoPE'd q_pe is produced; ql_nope used directly.
        q_pe_out = torch.empty_like(q_pe)
        mqa_q_fp8 = q_pe_out
        mqa_q_fp8_storage = q_pe_out  # unused placeholder for the byte pointer
        mqa_q = q_pe_out
"""
    text = replace_once(text, old, new, "deepseek kernels: uint8 MQA output")

    old = """        ql_nope,
        ql_nope.stride(0),
        ql_nope.stride(1),
        mqa_q_fp8,
        mqa_q_fp8.stride(0),
        mqa_q_fp8.stride(1),
        q_scale,
"""
    new = """        ql_nope,
        ql_nope.stride(0),
        ql_nope.stride(1),
        mqa_q_fp8_storage,
        mqa_q_fp8_storage.stride(0),
        mqa_q_fp8_storage.stride(1),
        q_scale,
"""
    text = replace_once(text, old, new, "deepseek kernels: fused-q MQA byte pointer")

    helper = text[text.index("def _fp8_ue8m0_quantize"):text.index(
        "def _fp8_quant_and_cache_write"
    )]
    if "tl.float8e4nv" in helper:
        raise RuntimeError("deepseek kernels: native fp8 cast remains in index quantizer")

    fq0 = text.index("def _fused_q_kernel")
    fq1 = text.index("def fused_q(", fq0)
    if "tl.float8e4nv" in text[fq0:fq1]:
        raise RuntimeError("deepseek kernels: native fp8 cast remains in fused_q kernel")
    return text

def patch_sparse_indexer(text: str) -> str:
    def patch_ragged_decode(text: str) -> str:
        marker = "SM80_RAGGED_INDEXER_DECODE_FIX"
        if marker in text:
            return text

        old = """        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
"""
        new = """        decode_lens = decode_metadata.decode_lens
        # SM80_RAGGED_INDEXER_DECODE_FIX: backport upstream #52500.  A
        # warmup/mixed decode batch can be ragged even when metadata does not
        # request padding.  Never reshape such a token stream as uniform.
        needs_padded_path = decode_metadata.requires_padding or (
            num_decode_tokens % decode_lens.shape[0] != 0
        )
        if num_decode_tokens == 0:
"""
        text = replace_once(
            text, old, new, "sparse indexer: ragged decode path detection"
        )
        text = replace_once(
            text,
            "        elif decode_metadata.requires_padding:\n",
            "        elif needs_padded_path:\n",
            "sparse indexer: ragged decode pack",
        )
        text = replace_once(
            text,
            "        if decode_metadata.requires_padding:\n"
            "            # if padded, we need to unpack\n",
            "        if needs_padded_path:\n"
            "            # if padded, we need to unpack\n",
            "sparse indexer: ragged decode unpack",
        )
        return text

    marker = "_sm80_fp8_fp4_mqa_logits"
    if marker in text:
        return patch_ragged_decode(text)

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
    return patch_ragged_decode(text)


def patch_topk_backend(text: str) -> str:
    """Backport upstream #58594 sparse-indexer Top-K backend propagation."""
    marker = "topk_backend=self.indexer.indexer_op.topk_backend"
    if marker in text:
        return text

    old = """                skip_topk_buffer_clear=True,
            )
"""
    new = """                skip_topk_buffer_clear=True,
                topk_backend=self.indexer.indexer_op.topk_backend,
            )
"""
    return replace_once(
        text, old, new, "deepseek attention: sparse top-k backend"
    )


def patch_mla_stride_alignment(text: str) -> str:
    """Publish the physical-row alignment required by TRITON_MLA_SPARSE.

    The sparse backend flattens a paged BF16 MLA cache into token rows via
    flat_kv_row_view().  In a packed block layout the inter-block stride may
    include other layers' pages, so the allocator must round that physical
    stride to a whole MLA row.
    """
    marker = "SM80_TRITON_MLA_BLOCK_STRIDE_ALIGNMENT"
    if marker in text:
        return text

    old = """        return MLAAttentionSpec(
            **common_kwargs,
            is_index_group_leader=self.indexer is not None,
            non_causal_multi_token_decode=self.non_causal_multi_token_decode,
        )
"""
    new = """        spec = MLAAttentionSpec(
            **common_kwargs,
            is_index_group_leader=self.indexer is not None,
            non_causal_multi_token_decode=self.non_causal_multi_token_decode,
        )
        # SM80_TRITON_MLA_BLOCK_STRIDE_ALIGNMENT: flat_kv_row_view addresses
        # physical blocks in whole latent+RoPE rows.  Packed cache blocks must
        # therefore start on an exact row boundary.
        if self.attn_backend.get_name() == "TRITON_MLA_SPARSE":
            spec = replace(
                spec, block_stride_alignment=spec.state_content_size_bytes
            )
        return spec
"""
    return replace_once(text, old, new, "MLA: Triton physical-row alignment")


def patch_kv_cache_interface_alignment(text: str) -> str:
    """Backport the v0.30-compatible part of upstream #56254/#55528."""
    marker = "block_stride_alignment: int | None = None"
    if marker in text:
        return text

    old = """    storage_block_size: int | None = None
    \"\"\"Token width used to view storage when it differs from the kernel block.\"\"\"
"""
    new = """    storage_block_size: int | None = None
    \"\"\"Token width used to view storage when it differs from the kernel block.\"\"\"
    block_stride_alignment: int | None = None
    \"\"\"Required byte alignment between consecutive physical cache blocks.\"\"\"
"""
    text = replace_once(text, old, new, "KV spec: block stride field")

    old = """        storage_block_size_set = set(spec.storage_block_size for spec in specs)
        assert (
"""
    new = """        storage_block_size_set = set(spec.storage_block_size for spec in specs)
        block_stride_alignment_set = {spec.block_stride_alignment for spec in specs}
        assert (
"""
    text = replace_once(text, old, new, "KV spec: merge alignment set")

    old = """            and len(index_group_leader_set) == 1
            and len(storage_block_size_set) == 1
        ), (
"""
    new = """            and len(index_group_leader_set) == 1
            and len(storage_block_size_set) == 1
            and len(block_stride_alignment_set) == 1
        ), (
"""
    text = replace_once(text, old, new, "KV spec: merge alignment invariant")

    old = """            cache_role=cache_role_set.pop(),
            is_index_group_leader=index_group_leader_set.pop(),
            storage_block_size=storage_block_size_set.pop(),
            non_causal_multi_token_decode=any(
"""
    new = """            cache_role=cache_role_set.pop(),
            is_index_group_leader=index_group_leader_set.pop(),
            storage_block_size=storage_block_size_set.pop(),
            block_stride_alignment=block_stride_alignment_set.pop(),
            non_causal_multi_token_decode=any(
"""
    text = replace_once(text, old, new, "KV spec: preserve block alignment")
    return text


def patch_kv_cache_allocator_alignment(text: str) -> str:
    """Round packed physical blocks to all MLA row-stride requirements."""
    marker = "SM80_MLA_STRIDE_ALIGNMENTS"
    if marker in text:
        return text

    old = """    if hot_page_sizes:
        bytes_per_block = round_up(bytes_per_block, math.lcm(*hot_page_sizes))
    return bytes_per_block
"""
    new = """    if hot_page_sizes:
        bytes_per_block = round_up(bytes_per_block, math.lcm(*hot_page_sizes))

    # SM80_MLA_STRIDE_ALIGNMENTS: upstream #56254/#55528.  A flat sparse-MLA
    # reader addresses rows across physical blocks, so packed blocks must honor
    # every layer's row-stride alignment requirement.
    stride_alignments = [
        spec.block_stride_alignment
        for group in kv_cache_groups
        for layer_name in group.layer_names
        if isinstance(spec := _get_per_layer_spec(group, layer_name), MLAAttentionSpec)
        and spec.block_stride_alignment
    ]
    if stride_alignments:
        bytes_per_block = round_up(bytes_per_block, math.lcm(*stride_alignments))
    return bytes_per_block
"""
    return replace_once(text, old, new, "KV allocator: MLA stride alignment")


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
    patch_file(root / "v1/attention/backends/mla/indexer.py", patch_indexer_metadata)
    patch_file(root / "model_executor/layers/sparse_attn_indexer.py", patch_sparse_indexer)
    patch_file(root / "models/deepseek_v32/common/kernels.py", patch_deepseek_kernels)
    patch_file(root / "models/deepseek_v32/attention.py", patch_topk_backend)
    patch_file(
        root / "model_executor/layers/attention/mla_attention.py",
        patch_mla_stride_alignment,
    )
    patch_file(root / "v1/kv_cache_interface.py", patch_kv_cache_interface_alignment)
    patch_file(root / "v1/core/kv_cache_utils.py", patch_kv_cache_allocator_alignment)

    print("GLM53_FULL_SM80_V030_PATCH=PASS")


if __name__ == "__main__":
    main()
