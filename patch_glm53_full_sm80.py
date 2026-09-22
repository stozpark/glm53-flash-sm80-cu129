#!/usr/bin/env python3
"""Backport the SM80 DSA path required by full zai-org/GLM-5.3 to vLLM 0.29.0.

Target:
  vllm/vllm-openai:v0.29.0-cu129
  GlmMoeDsaForCausalLM / glm_moe_dsa
  NVIDIA A100/A800 (SM80)

The patch deliberately does NOT reuse GLM-5.3-Flash KPool/KPoolTail code.
Full GLM-5.3 follows the DeepSeek-V3.2 DSA path.
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
        text, anchor,
        anchor + '    TRITON_MLA_SPARSE = (\n'
        '        "vllm.v1.attention.backends.mla.triton_mla_sparse."\n'
        '        "TritonMLASparseBackend"\n'
        '    )\n',
        "registry TRITON_MLA_SPARSE",
    )


def patch_cuda(text: str) -> str:
    if "AttentionBackendEnum.TRITON_MLA_SPARSE" in text:
        return text
    old = """                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
"""
    new = """                AttentionBackendEnum.TRITON_MLA,
                # Pure Triton DSA path for SM80/other architectures where
                # DeepGEMM + FlashMLA-Sparse are unavailable.
                AttentionBackendEnum.TRITON_MLA_SPARSE,
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
"""
    return replace_once(text, old, new, "CUDA sparse MLA priority")


def patch_indexer_metadata(text: str) -> str:
    text = text.replace(
        "    has_deep_gemm,\n",
        "    is_deep_gemm_supported,\n",
    )
    text = text.replace(
        "and has_deep_gemm()",
        "and is_deep_gemm_supported()",
    )
    text = text.replace(
        "and has_deep_gemm()\n",
        "and is_deep_gemm_supported()\n",
    )
    if "has_deep_gemm" in text:
        raise RuntimeError("indexer.py: unconverted has_deep_gemm reference remains")
    return text


def patch_sparse_indexer(text: str) -> str:
    # Rename Hopper kernels, then install signature-compatible dispatch wrappers.
    text = text.replace(
        """from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
""",
        """from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits as _deepgemm_fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits as _deepgemm_fp8_fp4_paged_mqa_logits,
    is_deep_gemm_supported,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
""",
    )

    marker = "def _sm80_fp8_fp4_mqa_logits("
    if marker not in text:
        anchor = "MXFP4_BLOCK_SIZE = 32\n\n"
        helpers = r'''
def _sm80_fp8_fp4_mqa_logits(
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
        raise RuntimeError("SM80 Triton DSA indexer supports FP8, not MXFP4")
    return fp8_mqa_logits_triton(
        q, (k, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke,
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
        raise RuntimeError("SM80 Triton DSA indexer supports FP8, not MXFP4")
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
            text, anchor, anchor + helpers, "SM80 MQA dispatch helpers"
        )

    text = text.replace("logits = fp8_fp4_mqa_logits(", "logits = _sm80_fp8_fp4_mqa_logits(")
    text = text.replace(
        "logits = fp8_fp4_paged_mqa_logits(",
        "logits = _sm80_fp8_fp4_paged_mqa_logits(",
    )

    # On the fallback, allocate logits only to the active batch width instead of
    # configured 1M context; downstream top-k is bounded by seq_lens.
    old = "                max_model_len=max_model_len,\n                clean_logits=False,\n                indices=decode_metadata.indices,\n"
    new = "                max_model_len=(\n                    max_model_len\n                    if is_deep_gemm_supported()\n                    else attn_metadata_narrowed.max_seq_len\n                ),\n                clean_logits=False,\n                indices=decode_metadata.indices,\n"
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
                    "SM80 Triton DSA fallback supports the FP8 indexer cache, "
                    "not MXFP4."
                )
            logger.warning_once(
                "DeepGEMM is unavailable on this CUDA architecture; "
                "using the Triton DSA indexer fallback."
            )
"""
    if old_gate in text:
        text = text.replace(old_gate, new_gate, 1)
    elif "using the Triton DSA indexer fallback" not in text:
        raise RuntimeError("DeepGEMM hard-gate anchor not found")

    if "has_deep_gemm" in text:
        raise RuntimeError("sparse_attn_indexer.py: has_deep_gemm reference remains")
    return text


def patch_pp_topk_relay(text: str) -> str:
    """Carry DSA shared top-k selections across pipeline stage boundaries."""
    if "_sm80_pp_topk_relay_buf" in text:
        return text

    text = replace_once(
        text,
        """        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
""",
        """        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
        # PRIVATE name is intentional: speculative proposers probe public
        # topk_indices_buffer attributes and may alias their own draft buffer.
        self._sm80_pp_topk_relay_buf = topk_indices_buffer
""",
        "PP top-k private buffer",
    )

    text = replace_once(
        text,
        """        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
""",
        """        if parallel_config.pipeline_parallel_size > 1:
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
""",
        "PP top-k intermediate factory",
    )

    text = replace_once(
        text,
        """        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
""",
        """        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # A PP stage may start on a skip_topk/shared-indexer layer. Seed
            # this rank with the current batch's selections from the previous
            # stage instead of consuming stale selections from an older batch.
            incoming_topk = intermediate_tensors.tensors.get("topk_indices")
            if incoming_topk is not None:
                n = min(
                    incoming_topk.shape[0],
                    self._sm80_pp_topk_relay_buf.shape[0],
                )
                self._sm80_pp_topk_relay_buf[:n].copy_(incoming_topk[:n])
""",
        "PP top-k receive",
    )

    text = replace_once(
        text,
        """            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
""",
        """            outgoing = {
                "hidden_states": hidden_states,
                "residual": residual,
            }
            # Clone: PP send may be asynchronous while the rank-local sparse
            # index buffer is mutable and reused by the next microbatch.
            outgoing["topk_indices"] = self._sm80_pp_topk_relay_buf[
                : positions.shape[0]
            ].clone()
            return IntermediateTensors(outgoing)
""",
        "PP top-k send",
    )
    return text


def patch_file(path: Path, fn) -> None:
    old = path.read_text(encoding="utf-8")
    new = fn(old)
    path.write_text(new, encoding="utf-8")
    print(f"[glm53-full-sm80] patched {path}")


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

    # New pure-Triton SM80 files.
    for rel in (
        "v1/attention/ops/triton_mla_sparse_kernel.py",
        "v1/attention/ops/mqa_logits_triton.py",
        "v1/attention/backends/mla/triton_mla_sparse.py",
    ):
        src = vendor / rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[glm53-full-sm80] installed {dst}")

    patch_file(root / "v1/attention/backends/registry.py", patch_registry)
    patch_file(root / "platforms/cuda.py", patch_cuda)
    patch_file(root / "v1/attention/backends/mla/indexer.py", patch_indexer_metadata)
    patch_file(root / "model_executor/layers/sparse_attn_indexer.py", patch_sparse_indexer)
    patch_file(root / "models/deepseek_v32/nvidia/model.py", patch_pp_topk_relay)

    print("GLM53_FULL_SM80_PATCH=PASS")


if __name__ == "__main__":
    main()
