from pathlib import Path
import importlib.util

P = Path(__file__).resolve().parents[1] / 'patch_runtime.py'
spec = importlib.util.spec_from_file_location('patch_runtime', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

cuda = '''            sparse_tail = [\n                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,\n                AttentionBackendEnum.FLASHMLA_SPARSE,\n                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,\n            ]\n'''
out = m.patch_cuda_priorities(cuda)
assert 'AttentionBackendEnum.TRITON_MLA_SPARSE' in out

backend = '''class X:\n    def _bf16_flash_mla_kernel(\n        self,\n        q,\n    ) -> tuple[torch.Tensor, torch.Tensor | None]:\n        output = q\n        return output[:, : self.num_heads, :], None\n'''
out = m.patch_54031_backend(backend)
assert ') -> torch.Tensor:' in out
assert 'return output[:, : self.num_heads, :], None' not in out

kpool = '''from vllm.utils.deep_gemm import (\n    fp8_fp4_mqa_logits,\n    fp8_fp4_paged_mqa_logits,\n    has_deep_gemm,\n)\nfrom vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton\n\ndef f():\n    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens\n    logits = fp8_fp4_mqa_logits(\n        (q_slice_cast, q_scale_slice),\n        (k_quant_cast, k_scale_cast),\n        weights[chunk.token_start : chunk.token_end],\n        chunk.cu_seqlen_ks,\n        chunk.cu_seqlen_ke,\n        clean_logits=False,\n    )\n    logits = fp8_fp4_paged_mqa_logits(\n        (padded_q_quant_cast, padded_q_scale),\n        kv_cache,\n        padded_weights[:num_padded_tokens],\n        seq_lens,\n        decode_metadata.block_table,\n        decode_metadata.schedule_metadata,\n        max_model_len=max_model_len,\n        clean_logits=False,\n    )\n'''
out = m.patch_kpool_indexer(kpool)
assert 'fp8_mqa_logits_triton' in out
assert 'fp8_paged_mqa_logits_triton' in out
assert 'seq_lens[:, -1].contiguous()' not in out
assert 'Preserve exact per-token KPool lengths for native MTP1.' in out
assert 'use_deep_gemm = is_deep_gemm_supported()' in out

mqa = '''@triton.jit
def _fp8_paged_mqa_logits_kernel(
    context_lens_ptr,
    stride_l_t,
    stride_l_n,
    next_n: tl.constexpr,
):
    batch_id = 0
    next_n_id = 0
    block_rk = 0
    block_size = 64
    context_len = tl.load(context_lens_ptr + batch_id)
    if block_rk * block_size >= context_len:
        return

    q_offset = context_len - next_n + next_n_id

def fp8_paged_mqa_logits_triton(q, kv_cache, weights, context_lens, block_tables):
    \"\"\"
        context_lens:  [B] int32
    \"\"\"
    B, next_n, num_heads, head_dim = q.shape
    _, block_size, one, d_plus_4 = kv_cache.shape
    assert one == 1
    assert d_plus_4 == head_dim + 4
    _fp8_paged_mqa_logits_kernel[(1,)](
        q,
        kv_cache,
        weights,
        fp8_lut,
        context_lens,
        block_tables,
        logits,
        logits.stride(0),
        logits.stride(1),
        next_n=next_n,
    )
'''
out = m.patch_mqa_paged_context_lens(mqa)
assert 'CONTEXT_LENS_2D: tl.constexpr' in out
assert 'q_offset = context_len - 1' in out
assert 'context_lens_c = context_lens.contiguous()' in out
assert 'CONTEXT_LENS_2D=context_lens_c.ndim == 2' in out

print('PATCH_RUNTIME_UNIT=PASS')
