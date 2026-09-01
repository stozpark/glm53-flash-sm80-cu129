#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


def must(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise RuntimeError(f"verification failed: {label}: missing {needle!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    args = ap.parse_args()
    root = Path(args.vllm_root).resolve()

    files = {
        "registry": root / "v1/attention/backends/registry.py",
        "cuda": root / "platforms/cuda.py",
        "backend": root / "v1/attention/backends/mla/triton_mla_sparse.py",
        "kernel": root / "v1/attention/ops/triton_mla_sparse_kernel.py",
        "mqa": root / "v1/attention/ops/mqa_logits_triton.py",
        "indexer_meta": root / "v1/attention/backends/mla/indexer.py",
        "kpool": root / "model_executor/layers/sparse_attn_indexer_kpool.py",
        "kpool_compress": root / "models/glm5next/nvidia/ops/kpool_compress.py",
        "runner": root / "v1/worker/gpu_model_runner.py",
        "model_runner": root / "v1/worker/gpu/model_runner.py",
        "block_table": root / "v1/worker/gpu/block_table.py",
        "mamba_hybrid": root / "v1/worker/gpu/model_states/mamba_hybrid.py",
        "speculator": root / "v1/worker/gpu/spec_decode/speculator.py",
        "fp8_sm80": root / "v1/attention/ops/fp8_sm80.py",
    }
    for name, p in files.items():
        if not p.exists():
            raise RuntimeError(f"verification failed: missing {name}: {p}")
        py_compile.compile(str(p), doraise=True)

    t = files["registry"].read_text()
    must(t, "TRITON_MLA_SPARSE", "backend registry")

    t = files["cuda"].read_text()
    must(t, "AttentionBackendEnum.TRITON_MLA_SPARSE", "SM80 backend priority")

    t = files["backend"].read_text()
    must(t, "class TritonMLASparseImpl(FlashMLASparseImpl)", "#54031 FlashMLA plumbing retained")
    must(t, "return [512, 576]", "NoPE/RoPE head sizes")
    must(t, 'return "TRITON_MLA_SPARSE"', "backend name")
    must(t, "return [MultipleOf(64)]", "Mrzhiyao 64-multiple block-size policy")

    t = files["kernel"].read_text()
    must(t, "_SUPPORTED_DIM_QK = (_BLOCK_DMODEL, _DIM_QK)", "Mrzhiyao 512/576 geometry")
    must(t, "KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)", "Mrzhiyao split-KV candidates")
    must(t, "block_dpe = dim_qk - _BLOCK_DMODEL", "NoPE geometry dispatch")
    must(t, "if BLOCK_DPE > 0:", "compile-time RoPE pruning")
    must(t, "e_sum_safe = tl.where", "empty-row NaN guard")
    must(t, "_SPLIT_MAX_OCCUPANCY = 4", "occupancy-aware split heuristic")

    t = files["mqa"].read_text()
    must(t, ".to(tl.int64)", "large-stride block index")
    must(t, "k_offset < context_len", "paged tail bound")
    must(t, "write -inf here for the early-exit tile", "chunked-prefill dirty buffer fix")

    t = files["indexer_meta"].read_text()
    must(t, "is_deep_gemm_supported", "architecture-aware metadata gate")

    t = files["kpool"].read_text()
    must(t, "fp8_mqa_logits_triton", "KPool prefill SM80 fallback")
    must(t, "fp8_paged_mqa_logits_triton", "KPool decode SM80 fallback")
    must(t, "seq_lens[:, -1].contiguous()", "MTP 2D seq_lens fix")
    must(t, "is_deep_gemm_supported", "KPool architecture gate")

    t = files["runner"].read_text()
    must(t, "if self.vllm_config.max_concurrent_batches > 1:", "#47644 PP race fix")

    t = files["indexer_meta"].read_text()
    must(t, "out_full=self.tail_slot_mapping_buffer", "KPoolTail persistent mapping")
    must(t, "out_full.fill_(-1)", "KPoolTail padded sentinel")
    must(t, "seq_lens[:num_decodes] // self.compress_ratio", "MTP padded seq_lens fix")
    must(t, "KpoolTailMetadataBuilder requires CommonAttentionMetadata.positions", "KPoolTail positions contract")

    t = files["model_runner"].read_text()
    must(t, "slot_mapping_enabled=slot_mapping_enabled", "KPoolTail generic slot-map exclusion wiring")

    t = files["block_table"].read_text()
    must(t, "enabled = tl.load(slot_mapping_enabled + group_id) != 0", "KPoolTail generic slot-map kernel skip")

    t = files["mamba_hybrid"].read_text()
    must(t, "positions=input_batch.positions", "hybrid positions propagation")

    t = files["speculator"].read_text()
    must(t, "positions=self.input_buffers.positions[:num_tokens_padded]", "MTP draft positions")

    t = files["fp8_sm80"].read_text()
    must(t, "def _encode_e4m3fn_u8", "SM80 software FP8 encoder")

    t = files["kpool_compress"].read_text()
    must(t, "native_fp8_cast_supported", "KPool SM80 FP8 write fallback")

    print("STATIC_VERIFY=PASS")
    print(f"VLLM_ROOT={root}")
    for name, p in files.items():
        print(f"OK {name}: {p.relative_to(root)}")


if __name__ == "__main__":
    main()
