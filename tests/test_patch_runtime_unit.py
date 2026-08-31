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
assert 'seq_lens[:, -1].contiguous()' in out
assert 'use_deep_gemm = is_deep_gemm_supported()' in out
print('PATCH_RUNTIME_UNIT=PASS')
