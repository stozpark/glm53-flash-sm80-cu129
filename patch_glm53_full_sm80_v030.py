#!/usr/bin/env python3
"""Port full zai-org/GLM-5.3 (glm_moe_dsa) to NVIDIA SM80 on vLLM 0.30.0.

Target:
  - vllm/vllm-openai:v0.30.0 (CUDA 13.0)
  - NVIDIA A100/A800 (SM80)
  - official zai-org/GLM-5.3 FP8 checkpoint
  - BF16 main MLA KV cache
  - Triton DSA indexer + Triton sparse MLA
  - Marlin W8A16 execution for checkpoint FP8 weights

This is intentionally a full-GLM-5.3 port, not a GLM-5.3-Flash/KPool patch.
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
        "registry TRITON_MLA_SPARSE",
    )


def patch_cuda_priority(text: str) -> str:
    if "AttentionBackendEnum.TRITON_MLA_SPARSE" in text:
        return text
    old = """            sparse_tail = [
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
"""
    new = """            sparse_tail = [
                # Pure-Triton DSA path used by this SM80 port. Explicit
                # --attention-backend selects it directly; keeping it in the
                # priority list also makes auto-selection aware of it.
                AttentionBackendEnum.TRITON_MLA_SPARSE,
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
"""
    return replace_once(text, old, new, "CUDA sparse MLA priority")


def patch_indexer_metadata(text: str) -> str:
    # v0.30 has both has_deep_gemm() (package availability) and
    # is_deep_gemm_supported() (package + runtime architecture). A100 can have
    # the package present but cannot execute DeepGEMM. All DSA metadata
    # decisions must therefore use the architecture-aware predicate.
    old = """from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
    native_next_n_supported,
)
"""
    new = """from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    is_deep_gemm_supported,
    native_next_n_supported,
)
"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "is_deep_gemm_supported" not in text:
        raise RuntimeError("indexer.py DeepGEMM import anchor not found")
    text = text.replace("has_deep_gemm()", "is_deep_gemm_supported()")

    # Our sparse-attention builder (Triton/XPU-derived) does not cast a reorder
    # vote. The indexer split itself assumes decode-first order, so make the
    # indexer vote its own threshold. This is especially important once mixed
    # prefill/decode batches appear.
    old_thr = """        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        self.reorder_batch_threshold = None
"""
    new_thr = """        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        # split_decodes_and_prefills assumes decode-first batch order. The
        # Triton sparse MLA builder used on SM80 does not provide a competing
        # reorder threshold, so the indexer must request its own threshold.
        self.reorder_batch_threshold = self.decode_threshold
"""
    if old_thr in text:
        text = text.replace(old_thr, new_thr, 1)
    elif "self.reorder_batch_threshold = self.decode_threshold" not in text:
        raise RuntimeError("indexer reorder-threshold anchor not found")
    return text


def patch_sparse_indexer(text: str) -> str:
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
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
"""
    if old_import in text:
        text = text.replace(old_import, new_import, 1)
    elif "_deepgemm_fp8_fp4_mqa_logits" not in text:
        raise RuntimeError("sparse indexer DeepGEMM import anchor not found")

    marker = "def _sm80_fp8_fp4_mqa_logits("
    if marker not in text:
        anchor = "MXFP4_BLOCK_SIZE = 32\n\n"
        helpers = r'''def _sm80_fp8_fp4_mqa_logits(
    q_pair,
    kv_pair,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    clean_logits=False,
):
    """Architecture-aware DSA prefill logits dispatch."""
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
        raise RuntimeError("SM80 Triton DSA indexer supports FP8, not MXFP4")
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
    """Architecture-aware DSA decode logits dispatch."""
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
        raise RuntimeError("SM80 Triton DSA indexer supports FP8, not MXFP4")
    # The SM80 kernel consumes one context length per flattened query row.
    # MTP is intentionally disabled in the baseline serve profile; if a 2-D
    # shape is present, the final column is the current visible length.
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
        text = replace_once(
            text, anchor, anchor + helpers, "SM80 DSA dispatch helpers"
        )

    text = text.replace(
        "logits = fp8_fp4_mqa_logits(",
        "logits = _sm80_fp8_fp4_mqa_logits(",
    )
    text = text.replace(
        "logits = fp8_fp4_paged_mqa_logits(",
        "logits = _sm80_fp8_fp4_paged_mqa_logits(",
    )

    # Avoid allocating a logits matrix at the configured maximum context on
    # the Triton fallback. In eager baseline mode max_seq_len is the active
    # batch width and therefore exact.
    old = """                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
"""
    new = """                max_model_len=(
                    max_model_len
                    if is_deep_gemm_supported()
                    else attn_metadata_narrowed.max_seq_len
                ),
                clean_logits=False,
                indices=decode_metadata.indices,
"""
    if old in text:
        text = text.replace(old, new, 1)

    old_gate = """        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )
"""
    new_gate = """        if current_platform.is_cuda() and not is_deep_gemm_supported():
            if self.use_fp4_cache:
                raise RuntimeError(
                    "SM80 Triton DSA fallback supports FP8 indexer KV, not MXFP4."
                )
            logger.warning_once(
                "DeepGEMM is unsupported on this CUDA architecture; "
                "using the SM80 Triton DSA indexer fallback."
            )
"""
    if old_gate in text:
        text = text.replace(old_gate, new_gate, 1)
    elif "using the SM80 Triton DSA indexer fallback" not in text:
        raise RuntimeError("SparseAttnIndexer DeepGEMM hard-gate anchor not found")

    # Any remaining has_deep_gemm() check can incorrectly select DeepGEMM on
    # an A100 simply because the Python package exists.
    text = text.replace("has_deep_gemm()", "is_deep_gemm_supported()")
    return text


def patch_deepseek_v32_kernels(text: str) -> str:
    """Remove active fp8e4nv conversions from DSA Q/K quantization on SM80.

    Triton cannot lower float8e4nv conversion instructions for SM80. Keep FP8
    as a storage *format*: software-encode E4M3FN bytes, store via uint8
    pointers, and expose float8 views only at the Python tensor boundary.
    """
    import_anchor = "from vllm.utils.torch_utils import is_quantized_kv_cache\n"
    import_line = (
        "from vllm.v1.attention.ops.fp8_sm80 import _encode_e4m3fn_u8\n"
    )
    if import_line not in text:
        text = replace_once(
            text,
            import_anchor,
            import_anchor + import_line,
            "deepseek_v32 software FP8 import",
        )

    old = "    fp8_vals = tl.div_rn(vals, scale).to(tl.float8e4nv)\n"
    new = "    fp8_vals = _encode_e4m3fn_u8(tl.div_rn(vals, scale))\n"
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError("_fp8_ue8m0_quantize conversion anchor not found")

    old_cache = """        if indexer_k_cache.dtype == torch.uint8:
            indexer_k_cache = indexer_k_cache.view(torch.float8_e4m3fn)
"""
    new_cache = """        # SM80 Triton cannot materialize fp8e4nv values. Keep the cache
        # pointer byte-addressed; _fp8_ue8m0_quantize emits exact E4M3FN bytes.
        indexer_k_cache = indexer_k_cache.view(torch.uint8)
"""
    if old_cache in text:
        text = text.replace(old_cache, new_cache, 1)
    elif "indexer_k_cache = indexer_k_cache.view(torch.uint8)" not in text:
        raise RuntimeError("indexer K-cache byte-view anchor not found")

    # MQA query FP8 pack. This path is not used with the baseline BF16 MLA KV,
    # but making it byte-addressed avoids another SM80 compile trap if the
    # query quantization flag is reached.
    old_mqa_alloc = """        mqa_q_fp8 = torch.empty(
            q_pe.shape[0],
            q_pe.shape[1],
            ql_nope.shape[2] + q_pe.shape[2],
            dtype=torch.float8_e4m3fn,
            device=q_pe.device,
        )
        # Placeholder; pid 0 packs q_pe into mqa_q_fp8 instead.
        q_pe_out = mqa_q_fp8
        mqa_q = mqa_q_fp8
"""
    new_mqa_alloc = """        mqa_q_fp8 = torch.empty(
            q_pe.shape[0],
            q_pe.shape[1],
            ql_nope.shape[2] + q_pe.shape[2],
            dtype=torch.uint8,
            device=q_pe.device,
        )
        # Byte storage inside Triton; consumers receive an E4M3FN tensor view.
        q_pe_out = mqa_q_fp8
        mqa_q = mqa_q_fp8.view(torch.float8_e4m3fn)
"""
    if old_mqa_alloc in text:
        text = text.replace(old_mqa_alloc, new_mqa_alloc, 1)

    text = text.replace(
        "                ql_nope_fp8 = (ql_nope / scale).to(tl.float8e4nv)\n",
        "                ql_nope_fp8 = _encode_e4m3fn_u8(ql_nope / scale)\n",
    )
    text = text.replace(
        "                        (r1 / scale).to(tl.float8e4nv),\n",
        "                        _encode_e4m3fn_u8(r1 / scale),\n",
    )
    text = text.replace(
        "                    (r2 / scale).to(tl.float8e4nv),\n",
        "                    _encode_e4m3fn_u8(r2 / scale),\n",
    )

    old_iq = (
        "    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)\n"
    )
    new_iq = (
        "    index_q_fp8 = torch.empty_like(index_q, dtype=torch.uint8)\n"
    )
    if old_iq in text:
        text = text.replace(old_iq, new_iq, 1)
    elif new_iq not in text:
        raise RuntimeError("index Q FP8 allocation anchor not found")

    # The CuTeDSL fused-Q path is SM89+ and expects a true FP8 output pointer.
    # This A100-specific port must never dispatch there.
    text = text.replace(
        "    if current_platform.is_cuda():\n"
        "        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (\n",
        "    if current_platform.is_cuda() and current_platform.has_device_capability(89):\n"
        "        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (\n",
        1,
    )

    # Both fused_q return sites expose the encoded byte tensor as E4M3FN.
    text = text.replace(
        "        return index_q_fp8, index_weights_out, mqa_q\n",
        "        return index_q_fp8.view(torch.float8_e4m3fn), index_weights_out, mqa_q\n",
        1,
    )
    # Final Triton fallback return.
    marker = "    return index_q_fp8, index_weights_out, mqa_q\n"
    if marker in text:
        text = text.replace(
            marker,
            "    return index_q_fp8.view(torch.float8_e4m3fn), index_weights_out, mqa_q\n",
            1,
        )

    if "index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)" in text:
        raise RuntimeError("native FP8 index-Q allocation remains")
    return text


def patch_pp_topk_relay(text: str) -> str:
    """Carry DSA shared top-k selections across PP stage boundaries."""
    if "_sm80_pp_topk_relay_buf" in text:
        return text

    alloc = """        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
"""
    alloc_new = alloc + """        # A PP stage may begin on a shared/skip-topk layer. Keep a private
        # relay reference so the current batch's selections can cross the
        # stage boundary. The public topk_indices_buffer remains untouched for
        # upstream MTP integration.
        self._sm80_pp_topk_relay_buf = self.topk_indices_buffer
"""
    text = replace_once(text, alloc, alloc_new, "PP top-k relay buffer")

    old_factory = """        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
"""
    new_factory = """        if parallel_config.pipeline_parallel_size > 1:
            _hidden_size = config.hidden_size
            _index_topk = config.index_topk

            def _make_empty_intermediate_tensors(batch_size, dtype, device):
                return IntermediateTensors(
                    {
                        "hidden_states": torch.zeros(
                            (batch_size, _hidden_size), dtype=dtype, device=device
                        ),
                        "residual": torch.zeros(
                            (batch_size, _hidden_size), dtype=dtype, device=device
                        ),
                        "topk_indices": torch.full(
                            (batch_size, _index_topk),
                            -1,
                            dtype=torch.int32,
                            device=device,
                        ),
                    }
                )

            self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors
        else:
            self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
"""
    text = replace_once(text, old_factory, new_factory, "PP top-k factory")

    old_recv = """        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
"""
    new_recv = """        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            incoming_topk = intermediate_tensors.tensors.get("topk_indices")
            if incoming_topk is not None:
                n = min(
                    incoming_topk.shape[0],
                    self._sm80_pp_topk_relay_buf.shape[0],
                )
                self._sm80_pp_topk_relay_buf[:n].copy_(incoming_topk[:n])
"""
    text = replace_once(text, old_recv, new_recv, "PP top-k receive")

    old_send = """            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
"""
    new_send = """            outgoing = {
                "hidden_states": hidden_states,
                "residual": residual,
                # Clone, never send a live view of the mutable rank-local
                # top-k buffer: PP transport may outlive this forward call.
                "topk_indices": self._sm80_pp_topk_relay_buf[
                    : positions.shape[0]
                ].clone(),
            }
            return IntermediateTensors(outgoing)
"""
    text = replace_once(text, old_send, new_send, "PP top-k send")
    return text


def patch_block_table(text: str) -> str:
    """Self-pad block-table tails so an over-wide DSA copy stays request-local."""
    if "SM80/DSA self-pad" not in text:
        old = """        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
"""
        new = old + """        # SM80/DSA self-pad: the indexer expands full block-table rows.
        # Do not leave stale block ids from a previous request in the tail.
        end = start + num_blocks
        if end < self.block_table.np.shape[1]:
            self.block_table.np[row_idx, end:] = self.block_table.np[row_idx, end - 1]
"""
        text = replace_once(text, old, new, "block-table self-pad")

        old_move = """        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
        # Clear the vacated source row: dummy-run batches dereference stale
        # rows as mamba state slots and write state in place there, possibly
        # after the blocks have been freed and reallocated.
        block_table_np[src, :num_blocks] = 0
        self.num_blocks_per_row[src] = 0
"""
        new_move = """        # Copy the full row so the self-padded tail moves with the request.
        block_table_np[tgt] = block_table_np[src]
        self.num_blocks_per_row[tgt] = num_blocks
        # Clear the full vacated row, including its padded tail.
        block_table_np[src] = 0
        self.num_blocks_per_row[src] = 0
"""
        text = replace_once(text, old_move, new_move, "block-table move full row")
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

    # Files derived from the SM80 sparse-MLA work plus our software FP8 codec.
    for rel in (
        "v1/attention/ops/fp8_sm80.py",
        "v1/attention/ops/triton_mla_sparse_kernel.py",
        "v1/attention/ops/mqa_logits_triton.py",
        "v1/attention/backends/mla/triton_mla_sparse.py",
    ):
        src = vendor / rel
        dst = root / rel
        if not src.exists():
            raise RuntimeError(f"missing vendored source: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[glm53-full-sm80-v030] installed {dst}")

    patch_file(root / "v1/attention/backends/registry.py", patch_registry)
    patch_file(root / "platforms/cuda.py", patch_cuda_priority)
    patch_file(root / "v1/attention/backends/mla/indexer.py", patch_indexer_metadata)
    patch_file(root / "model_executor/layers/sparse_attn_indexer.py", patch_sparse_indexer)
    patch_file(root / "models/deepseek_v32/common/kernels.py", patch_deepseek_v32_kernels)
    patch_file(root / "models/deepseek_v32/nvidia/model.py", patch_pp_topk_relay)
    patch_file(root / "v1/worker/block_table.py", patch_block_table)

    # Build-time invariants. These intentionally target the exact failure modes
    # seen on A100 rather than merely checking that strings were inserted.
    kernels = (root / "models/deepseek_v32/common/kernels.py").read_text()
    sparse = (root / "model_executor/layers/sparse_attn_indexer.py").read_text()
    idx = (root / "v1/attention/backends/mla/indexer.py").read_text()
    model = (root / "models/deepseek_v32/nvidia/model.py").read_text()

    assert "_encode_e4m3fn_u8" in kernels
    assert "index_q_fp8 = torch.empty_like(index_q, dtype=torch.uint8)" in kernels
    assert "indexer_k_cache = indexer_k_cache.view(torch.uint8)" in kernels
    assert "using the SM80 Triton DSA indexer fallback" in sparse
    assert "_sm80_fp8_fp4_paged_mqa_logits" in sparse
    assert "has_deep_gemm()" not in idx
    assert "self.reorder_batch_threshold = self.decode_threshold" in idx
    assert "_sm80_pp_topk_relay_buf" in model

    print("GLM53_FULL_SM80_V030_PATCH=PASS")


if __name__ == "__main__":
    main()
