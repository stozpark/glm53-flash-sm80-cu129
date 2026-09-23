#!/usr/bin/env python3
"""CPU/static validation for the full GLM-5.3 SM80 port.

This intentionally tests only properties that do not require an NVIDIA GPU.
GPU JIT/numerical parity is covered by sm80_glm53_full_kernel_smoke.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def must(text: str, needle: str, label: str) -> None:
    assert needle in text, f"{label}: missing {needle!r}"


def must_not(text: str, needle: str, label: str) -> None:
    assert needle not in text, f"{label}: unexpected {needle!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--port-root", required=True)
    args = ap.parse_args()

    vllm = Path(args.vllm_root)
    port = Path(args.port_root)

    launcher = (port / "serve_glm53_full_tp8_pp2.sh").read_text()
    patcher = (port / "patch_glm53_full_sm80.py").read_text()
    backend = (
        port
        / "vendor/glm53-full-sm80/v1/attention/backends/mla/triton_mla_sparse.py"
    ).read_text()
    smoke = (port / "tests/sm80_glm53_full_kernel_smoke.py").read_text()

    # ------------------------------------------------------------------
    # Published full GLM-5.3 DSA topology.
    # config: 78 layers, first 3 full-index layers, then full every 4
    # starting at layer 6.  The default 39/39 split starts PP1 on a shared
    # layer, while 42/36 starts PP1 on a full-index layer.
    # ------------------------------------------------------------------
    num_layers = 78
    full_index_layers = {0, 1, 2, *range(6, num_layers, 4)}
    assert 38 in full_index_layers
    assert 39 not in full_index_layers
    assert 42 in full_index_layers
    partitions = [42, 36]
    starts = [0, partitions[0]]
    assert sum(partitions) == num_layers
    assert all(x in full_index_layers for x in starts)
    # First three layers are dense; every later layer is MoE.
    moe_per_stage = [
        sum(i >= 3 for i in range(0, 42)),
        sum(i >= 3 for i in range(42, 78)),
    ]
    assert moe_per_stage == [39, 36]

    # Production launcher invariants.
    for needle in (
        "--tensor-parallel-size 8",
        "--pipeline-parallel-size 2",
        'PP_LAYER_PARTITION="${PP_LAYER_PARTITION:-42,36}"',
        'BLOCK_SIZE="${BLOCK_SIZE:-64}"',
        'ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"',
        'ENFORCE_EAGER="${ENFORCE_EAGER:-0}"',
        "--kv-cache-dtype bfloat16",
        '"backend":"TRITON_MLA_SPARSE"',
        "--linear-backend marlin",
        "--moe-backend marlin",
        "--tool-call-parser glm47",
        "--reasoning-parser glm47",
    ):
        must(launcher, needle, "launcher")
    must_not(launcher, "--speculative-config", "MTP is intentionally not enabled")
    must_not(launcher, "_sm80_pp_topk_relay_buf", "launcher")

    # Every critical launcher option must exist in the exact v0.30 parser.
    parser_text = (
        (vllm / "engine/arg_utils.py").read_text()
        + (vllm / "entrypoints/launchers/cli_args.py").read_text()
    )
    for flag in (
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--nnodes",
        "--node-rank",
        "--master-addr",
        "--master-port",
        "--headless",
        "--kv-cache-dtype",
        "--attention-config",
        "--linear-backend",
        "--moe-backend",
        "--block-size",
        "--enable-prefix-caching",
    ):
        must(parser_text, flag, "vLLM v0.30 CLI")

    # The active patch must stay narrow: no global BlockTable mutation, no old
    # PP relay, no legacy V1 pinned-buffer backport.
    for stale in (
        "patch_block_table",
        "SM80_DSA_SELF_PAD",
        "_sm80_pp_topk_relay_buf",
        "gpu_model_runner.py",
        "patch_47644",
        "reorder_batch_threshold = self.decode_threshold",
    ):
        must_not(patcher, stale, "patcher")

    # SM80 backend is deliberately limited to A100/A800 and exact 64-token
    # pages, matching DeepseekV32IndexerBackend in v0.30.
    must(backend, "return capability == DeviceCapability(8, 0)", "backend SM80 gate")
    must(backend, "return [64]", "backend block size")
    must(backend, "def record_logical_topk_ready", "v0.30 DSA API")
    must(backend, "_INDEXER_NUM_HEADS = 32", "GLM-5.3 indexer heads")
    must(backend, "_INDEXER_HEAD_DIM = 128", "GLM-5.3 indexer dim")
    must(smoke, "INDEX_HEADS = 32", "smoke config")
    assert smoke.count("index_rope_interleave=True") >= 2
    assert smoke.count("interleave=True") >= 2

    # Exact v0.30 indexer page-size contract.
    indexer = (vllm / "v1/attention/backends/mla/indexer.py").read_text()
    must(
        indexer,
        'return [1, MultipleOf(16)] if current_platform.is_rocm() else [64]',
        "DeepseekV32IndexerBackend page size",
    )

    # Marlin must support A100 and GLM-5.3's 128x128 block-FP8 weights.
    marlin_linear = (
        vllm / "model_executor/kernels/linear/scaled_mm/marlin.py"
    ).read_text()
    marlin_moe = (
        vllm / "model_executor/layers/fused_moe/experts/marlin_moe.py"
    ).read_text()
    must(marlin_linear, "FP8 Marlin requires compute capability 7.5 or higher", "linear")
    must(marlin_linear, "kFp8Static128BlockSym", "linear block-FP8")
    must(marlin_moe, "p.has_device_capability((7, 5))", "MoE A100 capability")
    must(marlin_moe, "kFp8Static128BlockSym", "MoE block-FP8")

    # Patched source invariants.
    patched_attention = (vllm / "models/deepseek_v32/attention.py").read_text()
    patched_kernels = (vllm / "models/deepseek_v32/common/kernels.py").read_text()
    patched_sparse = (
        vllm / "model_executor/layers/sparse_attn_indexer.py"
    ).read_text()
    must(
        patched_attention,
        "SM80_PIECEWISE_KV_BINDING_FIX",
        "PIECEWISE graph KV binding",
    )
    must(patched_kernels, "SM80_SOFTWARE_E4M3FN", "software E4M3")
    must(patched_kernels, "index_q_fp8_storage", "byte-addressed index Q")
    must(patched_sparse, "_sm80_fp8_fp4_mqa_logits", "Triton indexer fallback")
    must_not(
        patched_sparse,
        "Sparse Attention Indexer CUDA op requires DeepGEMM",
        "SM80 DeepGEMM hard gate",
    )

    q0 = patched_kernels.index("def _fp8_ue8m0_quantize")
    q1 = patched_kernels.index("def _fp8_quant_and_cache_write", q0)
    must_not(patched_kernels[q0:q1], "tl.float8e4nv", "index-K active quantizer")
    q0 = patched_kernels.index("def _fused_q_kernel")
    q1 = patched_kernels.index("def fused_q(", q0)
    must_not(patched_kernels[q0:q1], ".to(tl.float8e4nv)", "index-Q active kernel")

    print("GLM53_FULL_SM80_STATIC_SEMANTICS=PASS")


if __name__ == "__main__":
    main()
