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
    topk_probe = (port / "tests/sm80_sparse_indexer_topk_probe.py").read_text()

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

    # Production launcher invariants: official GLM-5.3 serving features plus
    # only SM80 / 2-node requirements.
    for needle in (
        "--distributed-executor-backend mp",
        "--tensor-parallel-size 8",
        "--pipeline-parallel-size 2",
        'PP_LAYER_PARTITION="${PP_LAYER_PARTITION:-42,36}"',
        'MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"',
        'RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-auto}"',
        "--compilation-config '{\"cudagraph_mode\":\"NONE\"}'",
        '"backend":"TRITON_MLA_SPARSE"',
        "--kv-cache-dtype bfloat16",
        "--sparse-indexer-topk-backend",
        "select_sparse_topk_backends",
        "VLLM_SM80_PREFILL_TOPK_BACKEND",
        "--tool-call-parser glm47",
        "--reasoning-parser glm47",
        "--enable-auto-tool-choice",
    ):
        must(launcher, needle, "launcher")
    for forbidden in (
        "--max-cudagraph-capture-size",
        "CUDAGRAPH_MODE=",
        "--linear-backend",
        "--moe-backend",
        "--enable-prefix-caching",
        "--env VLLM_USE_BREAKABLE_CUDAGRAPH",
        "--env VLLM_KV_CACHE_LAYOUT",
        "--speculative-config",
        "_sm80_pp_topk_relay_buf",
    ):
        must_not(launcher, forbidden, "minimal launcher")

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
        "--attention-config",
        "--max-model-len",
        "--kv-cache-dtype",
        "--compilation-config",
        "--sparse-indexer-topk-backend",
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
        "def patch_cuda(",
        "SM80_PIECEWISE_KV_BINDING_FIX",
    ):
        must_not(patcher, stale, "patcher")

    # SM80 backend is deliberately limited to A100/A800 and exact 64-token
    # pages, matching DeepseekV32IndexerBackend in v0.30.
    must(backend, "return capability == DeviceCapability(8, 0)", "backend SM80 gate")
    must(backend, "return [64]", "backend block size")
    must(
        backend,
        "return (KVCacheLayout.LBHNC,)",
        "safe V3.2 layer-compact KV layout",
    )
    must(backend, "def record_logical_topk_ready", "v0.30 DSA API")
    must(
        backend,
        "supports_quant_query_input = False",
        "SM80 BF16-query contract",
    )
    must(
        backend,
        '"bfloat16",',
        "SM80 BF16 KV contract",
    )
    must(
        backend,
        "supports_dense_mha_prefill = False",
        "sparse-only prefill routing",
    )
    must(
        patcher,
        "SM80_SPARSE_ONLY_PREFILL_TOPK_FIX",
        "short-prefill Top-K capability fix",
    )
    must(
        patcher,
        "SM80_PREFILL_TOPK_RUNTIME_BACKEND",
        "runtime-selectable prefill Top-K",
    )
    must(backend, "_INDEXER_NUM_HEADS = 32", "GLM-5.3 indexer heads")
    must(backend, "_INDEXER_HEAD_DIM = 128", "GLM-5.3 indexer dim")
    must(smoke, "INDEX_HEADS = 32", "smoke config")
    assert smoke.count("index_rope_interleave=True") >= 3
    assert smoke.count("interleave=True") >= 3
    must(smoke, "quantize_mqa=True", "SM80 fused-Q MQA smoke")
    must(smoke, "SM80_FUSED_Q_MQA_FP8_PACK=PASS", "SM80 MQA pack smoke")
    must(
        smoke,
        "SM80_SPARSE_MLA_METADATA_GUARD=PASS",
        "sparse-only metadata smoke",
    )
    must(
        topk_probe,
        "SM80_DECODE_TOPK_BACKEND=",
        "runtime decode Top-K selector output",
    )
    must(
        topk_probe,
        "SM80_PREFILL_TOPK_BACKEND=",
        "runtime prefill Top-K selector output",
    )
    must(
        topk_probe,
        "large_batch_40000",
        "oversized-bin large-batch regression",
    )
    must(
        topk_probe,
        "max_context_131072",
        "deployed max-context Top-K regression",
    )

    # Exact v0.30 indexer page-size contract.
    indexer = (vllm / "v1/attention/backends/mla/indexer.py").read_text()
    must(
        indexer,
        'return [1, MultipleOf(16)] if current_platform.is_rocm() else [64]',
        "DeepseekV32IndexerBackend page size",
    )

    # Exact full-GLM model routing and parser availability in v0.30.
    registry = (vllm / "model_executor/models/registry.py").read_text()
    deepseek_model = (vllm / "model_executor/models/deepseek_v2.py").read_text()
    tool_parsers = (vllm / "tool_parsers/__init__.py").read_text()
    reasoning_parsers = (vllm / "reasoning/__init__.py").read_text()
    must(
        registry,
        '"GlmMoeDsaForCausalLM": ("vllm.models.deepseek_v32", "GlmMoeDsaForCausalLM")',
        "full GLM registry",
    )
    must(deepseek_model, 'model_type", None) == "glm_moe_dsa"', "GLM model type")
    must(deepseek_model, "return torch.float32", "GLM FP32 MoE router")
    must(tool_parsers, '"glm47"', "GLM tool parser")
    must(reasoning_parsers, '"glm47"', "GLM reasoning parser")

    # Prefix caching stays on the vLLM default, but the official GLM-5.3
    # FP8-KV recipe is intentionally overridden: TRITON_MLA_SPARSE on SM80
    # consumes BF16 MLA KV and must never request FP8 query packing.
    cache_cfg = (vllm / "config/cache.py").read_text()
    must(cache_cfg, "enable_prefix_caching: bool = True", "prefix caching default")
    must(launcher, "--kv-cache-dtype bfloat16", "SM80 BF16 MLA KV override")
    must(
        launcher,
        'smoke.${EXPECTED_PORT_REVISION}.gpu${first_gpu}.ok',
        "revision-scoped one-time GPU smoke stamp",
    )

    vllm_cfg = (vllm / "config/vllm.py").read_text()
    b0 = vllm_cfg.index("DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES")
    b1 = vllm_cfg.index("@lru_cache", b0)
    must(
        vllm_cfg[b0:b1],
        '"GlmMoeDsaForCausalLM",',
        "GLM DSA breakable CUDA-graph default",
    )
    m0 = vllm_cfg.index("def _maybe_enable_breakable_cudagraph")
    m1 = vllm_cfg.index("@property", m0)
    must(
        vllm_cfg[m0:m1],
        "self.compilation_config.mode = CompilationMode.NONE",
        "breakable CUDA graphs disable Inductor",
    )
    deepseek_nvidia = (
        vllm / "models/deepseek_v32/nvidia/model.py"
    ).read_text()
    must_not(
        deepseek_nvidia,
        "@support_torch_compile",
        "v0.30 GlmMoeDsa must not enter Inductor compile path",
    )

    # Marlin must support A100 and GLM-5.3's 128x128 block-FP8 weights.
    fp8_quant = (
        vllm / "model_executor/layers/quantization/fp8.py"
    ).read_text()
    quant_utils = (
        vllm / "model_executor/layers/quantization/utils/quant_utils.py"
    ).read_text()
    marlin_linear = (
        vllm / "model_executor/kernels/linear/scaled_mm/marlin.py"
    ).read_text()
    marlin_moe = (
        vllm / "model_executor/layers/fused_moe/experts/marlin_moe.py"
    ).read_text()
    must(fp8_quant, "GroupShape(*self.weight_block_size)", "FP8 block quant mapping")
    must(quant_utils, "GroupShape(128, 128)", "128x128 FP8 quant key")
    must(quant_utils, "kFp8Static128BlockSym", "128x128 FP8 quant key")
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
        "topk_backend=self.indexer.indexer_op.topk_backend",
        "upstream #58594 GLM-5.3 sparse top-k backend selection",
    )
    must(
        patched_attention,
        "SM80_SPARSE_ONLY_PREFILL_TOPK_FIX",
        "sparse-only short-prefill Top-K guard",
    )
    short_skip = patched_attention[
        patched_attention.index("enable_short_prefill_scoring_skip"):
        patched_attention.index("self._dense_mha_metadata_layer_name"),
    ]
    must(
        short_skip,
        "self.impl.supports_dense_mha_prefill",
        "short-prefill skip requires backend dense-MHA capability",
    )
    must(patched_kernels, "SM80_SOFTWARE_E4M3FN_MQA", "software E4M3 MQA")
    must(patched_kernels, "index_q_fp8_storage", "byte-addressed index Q")
    must(patched_kernels, "mqa_q_fp8_storage", "byte-addressed MQA query")
    must(
        patched_kernels,
        "q_pe_out = q_pe",
        "SM80-safe BF16 dummy pointer for fused-Q signature",
    )
    must_not(
        patched_kernels,
        "q_pe_out = mqa_q_fp8",
        "FP8 pointer leaked into Triton fused-Q signature",
    )
    must(patched_sparse, "_sm80_fp8_fp4_mqa_logits", "Triton indexer fallback")
    must(
        patched_sparse,
        "SM80_RAGGED_INDEXER_DECODE_FIX",
        "upstream #52500 ragged decode path",
    )
    must(
        patched_sparse,
        "elif needs_padded_path:",
        "upstream #52500 ragged decode pack",
    )
    must(
        patched_sparse,
        "SM80_PREFILL_TOPK_RUNTIME_BACKEND",
        "runtime-selectable prefill Top-K helper",
    )
    must(
        patched_sparse,
        'os.getenv("VLLM_SM80_PREFILL_TOPK_BACKEND", "vllm")',
        "prefill Top-K runtime backend env",
    )
    must_not(
        patched_sparse,
        "Sparse Attention Indexer CUDA op requires DeepGEMM",
        "SM80 DeepGEMM hard gate",
    )

    # #48285: SM80 paged-MQA graph capture requires a 1-D effective
    # context length and the configured (constant) max_model_len.
    must(
        patched_sparse,
        "context_lens = context_lens[:, -1].contiguous()",
        "#48285 2-D decode context lengths",
    )
    must(
        patched_sparse,
        "max_model_len=max_model_len",
        "#48285 fixed decode logits width",
    )

    # #47522: Marlin repacks FP8 weights to int32.  Chunked/prefix prefill
    # must cast activations to params_dtype, never to the packed int32 weight.
    mla_common = (
        vllm / "model_executor/layers/attention/mla_attention.py"
    ).read_text()
    must(
        mla_common,
        "def _get_kv_b_proj_input_dtype",
        "#47522 Marlin prefill dtype helper",
    )
    must(
        mla_common,
        "if weight_dtype == torch.int32:",
        "#47522 packed Marlin weight handling",
    )
    must(
        mla_common,
        "return kv_b_proj.params_dtype",
        "#47522 activation dtype recovery",
    )

    mla_attention = (
        vllm / "model_executor/layers/attention/mla_attention.py"
    ).read_text()
    kv_interface = (vllm / "v1/kv_cache_interface.py").read_text()
    kv_utils = (vllm / "v1/core/kv_cache_utils.py").read_text()
    must(
        mla_attention,
        "SM80_SPARSE_MLA_NO_DENSE_PREFILL_GUARD",
        "sparse-only MLA metadata guard",
    )
    use_sparse_mha = mla_attention[
        mla_attention.index("def _use_sparse_mha"):
        mla_attention.index("def process_weights_after_loading")
    ]
    assert use_sparse_mha.index("supports_dense_mha_prefill") < use_sparse_mha.index(
        "prefill = attn_metadata.prefill"
    ), "dense-prefill capability guard must precede metadata.prefill access"
    must(
        mla_attention,
        "SM80_TRITON_MLA_BLOCK_STRIDE_ALIGNMENT",
        "Triton MLA packed-block row alignment",
    )
    must(
        mla_attention,
        'self.attn_backend.get_name() == "TRITON_MLA_SPARSE"',
        "alignment limited to Triton MLA",
    )
    must(
        kv_interface,
        "block_stride_alignment: int | None = None",
        "MLA cache stride contract",
    )
    must(
        kv_utils,
        "SM80_MLA_STRIDE_ALIGNMENTS",
        "allocator stride alignment",
    )
    must(
        kv_utils,
        "math.lcm(*stride_alignments)",
        "allocator LCM alignment",
    )

    # #58068 / #55569 risk audit: the stock sparse-indexer allocator uses
    # variable-width FP32 prefill logits.  On A100 this is a fragmentation/
    # headroom risk rather than the unified-memory failure reported on GB10.
    # Lock the production arithmetic so a future scheduler/default change does
    # not silently increase a single logits allocation beyond 512 MiB.
    envs_text = (vllm / "envs.py").read_text()
    must(
        envs_text,
        'os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")',
        "sparse indexer logits budget",
    )
    arg_utils = (vllm / "engine/arg_utils.py").read_text()
    must(
        arg_utils,
        "UsageContext.OPENAI_API_SERVER: 2048",
        "A100 server batched-token default",
    )
    max_model_len = 131072
    max_batched_tokens = 2048
    logits_budget_bytes = 512 * 1024 * 1024
    # At 128K, each launch is capped at 1024 query rows x 131072 keys x fp32
    # = 512 MiB, so one 2048-token scheduler chunk is split into two launches.
    rows_per_launch = logits_budget_bytes // (max_model_len * 4)
    assert rows_per_launch == 1024
    assert (max_batched_tokens + rows_per_launch - 1) // rows_per_launch == 2
    assert rows_per_launch * max_model_len * 4 == logits_budget_bytes

    q0 = patched_kernels.index("def _fp8_ue8m0_quantize")
    q1 = patched_kernels.index("def _fp8_quant_and_cache_write", q0)
    must_not(patched_kernels[q0:q1], "tl.float8e4nv", "index-K active quantizer")
    fq0 = patched_kernels.index("def _fused_q_kernel")
    fq1 = patched_kernels.index("def fused_q(", fq0)
    must_not(
        patched_kernels[fq0:fq1],
        "tl.float8e4nv",
        "fused-Q SM80 quantization",
    )
    must(patched_kernels, "index_q_fp8_storage", "byte-addressed index-Q output")
    must(patched_kernels, "mqa_q_fp8_storage", "byte-addressed MQA output")

    print("GLM53_FULL_SM80_STATIC_SEMANTICS=PASS")


if __name__ == "__main__":
    main()
