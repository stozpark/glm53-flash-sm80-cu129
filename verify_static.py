#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
import re
from pathlib import Path


def must(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise RuntimeError(f"verification failed: {label}: missing {needle!r}")


def must_not(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise RuntimeError(f"verification failed: {label}: forbidden {needle!r}")


def verify_glm_sparse_selector(text: str) -> None:
    pat = re.compile(
        r"^[ \t]*sparse_tail = \[\n"
        r"(?P<body>(?:^[ \t]+AttentionBackendEnum\.[A-Z0-9_]+,\n)+)"
        r"^[ \t]*\]\n",
        re.MULTILINE,
    )
    matches = list(pat.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"verification failed: expected exactly one sparse_tail, found {len(matches)}"
        )
    names = [
        line.strip().rstrip(",")
        for line in matches[0].group("body").splitlines()
    ]
    triton = "AttentionBackendEnum.TRITON_MLA_SPARSE"
    fi_sm90 = "AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90"
    if triton not in names or fi_sm90 not in names:
        raise RuntimeError("verification failed: sparse selector backend missing")
    simulated = names.copy()
    popped = simulated.pop()
    if popped != fi_sm90:
        raise RuntimeError(
            "verification failed: GLM prefer_fi_sm90 sparse_tail.pop() would "
            f"remove {popped}, not {fi_sm90}"
        )
    if triton not in simulated:
        raise RuntimeError(
            "verification failed: TRITON_MLA_SPARSE disappears after GLM sparse_tail.pop()"
        )


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
        "glm_attention": root / "models/glm5next/nvidia/attention.py",
        "kv_cache_interface": root / "v1/kv_cache_interface.py",
        "runner": root / "v1/worker/gpu_model_runner.py",
        "model_runner": root / "v1/worker/gpu/model_runner.py",
        "block_table": root / "v1/worker/gpu/block_table.py",
        "mamba_hybrid": root / "v1/worker/gpu/model_states/mamba_hybrid.py",
        "speculator": root / "v1/worker/gpu/spec_decode/speculator.py",
        "autoregressive_speculator": root
        / "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
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
    verify_glm_sparse_selector(t)

    t = files["backend"].read_text()
    must(t, "class TritonMLASparseImpl(FlashMLASparseImpl)", "#54031 FlashMLA plumbing retained")
    must(t, "return [512, 576]", "NoPE/RoPE head sizes")
    must(t, 'return "TRITON_MLA_SPARSE"', "backend name")
    must(t, "return [MultipleOf(64)]", "64-multiple block-size policy")

    t = files["kernel"].read_text()
    must(t, "_SUPPORTED_DIM_QK = (_BLOCK_DMODEL, _DIM_QK)", "512/576 geometry")
    must(t, "KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)", "split-KV candidates")
    must(t, "block_dpe = dim_qk - _BLOCK_DMODEL", "NoPE geometry dispatch")
    must(t, "if BLOCK_DPE > 0:", "compile-time RoPE pruning")
    must(t, "e_sum_safe = tl.where", "empty-row NaN guard")
    must(t, "_SPLIT_MAX_OCCUPANCY = 4", "occupancy-aware split heuristic")
    if t.count(".to(tl.int64) * stride_kv_token") != 3:
        raise RuntimeError(
            "verification failed: sparse MLA KV row offsets are not int64 at all 3 load sites"
        )

    t = files["mqa"].read_text()
    must(t, ".to(tl.int64)", "large-stride block index")
    must(t, "k_offset < context_len", "paged tail bound")
    must(t, "write -inf here for the early-exit tile", "chunked-prefill dirty buffer fix")

    t = files["indexer_meta"].read_text()
    must(t, "is_deep_gemm_supported", "architecture-aware metadata gate")
    must(t, "out_full=self.tail_slot_mapping_buffer", "KPoolTail persistent mapping")
    must(t, "out_full.fill_(-1)", "KPoolTail padded sentinel")
    must(t, "seq_lens[:num_decodes] // self.compress_ratio", "MTP padded seq_lens fix")
    must(t, "KpoolTailMetadataBuilder requires CommonAttentionMetadata.positions", "KPoolTail positions contract")

    t = files["kpool"].read_text()
    must(t, "fp8_mqa_logits_triton", "KPool prefill SM80 fallback")
    must(t, "fp8_paged_mqa_logits_triton", "KPool decode SM80 fallback")
    must(t, "seq_lens[:, -1].contiguous()", "MTP 2D seq_lens fix")
    must(t, "is_deep_gemm_supported", "KPool architecture gate")
    must(t, "DeepGEMM unavailable; using SM80 Triton sparse-indexer fallback.", "DeepGEMM fallback marker")
    must(t, "persistent_topk operates on pool-granular logits", "KPool persistent_topk pool-width fix")
    must_not(t, "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.", "stale DeepGEMM hard gate")
    must_not(t, "select_k,\n                attn_metadata_narrowed.max_seq_len", "stale token-granular persistent_topk width")

    t = files["glm_attention"].read_text()
    must(t, "index_kpool * 32", "KPool storage-block alignment guard")
    must(t, "cache_config.block_size % index_kpool == 0", "chunked-prefill pool alignment guard")

    t = files["kv_cache_interface"].read_text()
    kpool_tail_start = t.find("class KpoolTailSpec")
    if kpool_tail_start < 0:
        raise RuntimeError("verification failed: KpoolTailSpec missing")
    kpool_tail_region = t[kpool_tail_start : kpool_tail_start + 5000]
    must(
        kpool_tail_region,
        "def participates_in_prefix_caching(self) -> bool:",
        "KPoolTail native prefix-cache exclusion",
    )
    must(kpool_tail_region, "return False", "KPoolTail non-shareable prefix state")

    t = files["runner"].read_text()
    must(t, "if self.vllm_config.max_concurrent_batches > 1:", "#47644 PP race fix")

    t = files["model_runner"].read_text()
    must(t, "slot_mapping_enabled=slot_mapping_enabled", "KPoolTail generic slot-map exclusion wiring")

    t = files["block_table"].read_text()
    must(t, "enabled = tl.load(slot_mapping_enabled + group_id) != 0", "KPoolTail generic slot-map kernel skip")

    t = files["mamba_hybrid"].read_text()
    must(t, "positions=input_batch.positions", "hybrid positions propagation")

    t = files["speculator"].read_text()
    must(t, "positions=self.input_buffers.positions[:num_tokens_padded]", "MTP draft positions")

    t = files["autoregressive_speculator"].read_text()
    if t.count("torch.accelerator.current_stream().synchronize()") != 2:
        raise RuntimeError("verification failed: expected two MTP draft stream fences")
    if t.count("if not torch.cuda.is_current_stream_capturing():") < 2:
        raise RuntimeError("verification failed: MTP stream fences are not graph-capture safe")

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
