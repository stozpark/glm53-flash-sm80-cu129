# Full GLM-5.3 / SM80 correctness audit

Audit target: **full `zai-org/GLM-5.3`**, not GLM-5.3-Flash.

- GPU target: A100/A800 80GB (SM80)
- topology: 2 nodes x 8 GPUs, TP8 x PP2
- base image: vLLM 0.30.0 / CUDA 13.0
- architecture: `GlmMoeDsaForCausalLM / glm_moe_dsa`
- attention: DeepSeek-V3.2 DSA sparse MLA
- port revision: `glm53-full-sm80-cu130-v030-r20260930-14`

The old GLM-5.3-Flash/KPool audit is intentionally not used for this branch.

## Current model contract

CI downloads the live `zai-org/GLM-5.3` config and safetensors index and
verifies the assumptions used by this port.

Verified:

```text
num_hidden_layers        = 78
hidden_size              = 6144
first_k_dense_replace    = 3
n_routed_experts         = 256
num_experts_per_tok      = 8
q_lora_rank              = 2048
kv_lora_rank             = 512
qk_nope_head_dim         = 192
qk_rope_head_dim         = 64
index_n_heads            = 32
index_head_dim           = 128
index_topk               = 2048
index_topk_freq          = 4
indexer_rope_interleave  = true
weight FP8 block         = 128 x 128
```

The current checkpoint index reports:

```text
checkpoint size          = 703.723 GiB
average weight / 16 GPU  = 43.983 GiB
BF16 MLA KV @128K PP0    = 5.906 GiB/rank
BF16 MLA KV @128K PP1    = 5.062 GiB/rank
```

The actual stage weight distribution is not exactly the 16-GPU average; PP0
contains 39 MoE layers plus the first three dense layers and is expected to be
the heavier stage.

## Applied correctness fixes

### SM80 sparse-MLA / indexer

- PR #47629 lineage: Triton sparse MLA and FP8 paged-MQA fallback for SM80.
- Native DeepGEMM use is gated by actual architecture support.
- Main MLA KV is explicitly BF16 on SM80; this intentionally overrides the
  official Hopper/Blackwell GLM-5.3 recipe's FP8 KV setting.
- DSA index Q/K and the generic fused MQA-query pack use software E4M3FN byte
  encoding on SM80.
- The complete `_fused_q_kernel` contains no native `tl.float8e4nv`
  conversion after the r14 patch.
- sparse MLA and indexer pages use an exact 64-token block size.
- physical sparse-MLA indices use int64 address arithmetic for long context.

### Post-v0.30 fixes backported

- **#58594**: propagate the layer-selected sparse Top-K backend into the direct
  `sparse_attn_indexer()` call.
- **#55528/#56254 minimal backport**: publish and honor physical block-stride
  alignment for row-addressed BF16 sparse MLA.  For this backend the alignment
  is one 576-element BF16 row = **1152 bytes**.
- **#51395 capability pattern**: `TRITON_MLA_SPARSE` declares
  `supports_dense_mha_prefill=False`; all prefills remain on the implemented
  sparse-MQA path.
- **#48285 fixes**: normalize decode context lengths to the effective 1-D form
  and keep decode logits width tied to configured `max_model_len`.
- **#47522 protection**: chunked/prefix prefill recovers the activation dtype
  from Marlin params rather than casting activations to packed int32 weights.
- **#49844 avoidance**: vLLM 0.30 already classifies `GlmMoeDsaForCausalLM`
  as a breakable-CUDA-graph architecture and disables Inductor compilation for
  that path. Target A100 profiling exhausted the remaining HBM even before
  serving, so the startup-safe r14 launcher explicitly selects
  `cudagraph_mode=NONE`.
- **#52500**: use the padded pack/unpack path for ragged warmup/mixed decode
  batches even when metadata incorrectly reports `requires_padding=False`.

## PP correctness

The default 39/39 split is not used.

```text
layer 38 = full indexer producer
----- default PP boundary -----
layer 39 = shared indexer consumer
```

Current production partition:

```text
VLLM_PP_LAYER_PARTITION=42,36
PP0 = layers 0..41
PP1 = layers 42..77
```

Layer 42 is a full-indexer producer, so PP1 generates its own logical Top-K and
no cross-stage Top-K relay is required.

The old custom Top-K relay is absent.

## Explicitly reviewed but not backported

### #47644

Old V1 `gpu_model_runner.py` pinned-input-buffer race.

The production path uses the v0.30 V2 runner
`vllm/v1/worker/gpu/model_runner.py`, whose UVA buffer pool is sized from
`max_concurrent_batches`.  #47644 is therefore not applied.

### #51915

The significant correctness issue in this PR is the AMD/AITER shuffled indexer
cache layout.  CUDA's base indexer layout reports no shuffle; ROCm FP8/FNUZ and
AITER BMM changes are not applicable to A100.

### #55431

An explicit block-outermost `BLHNC` layout can make V3.2 sparse MLA inherit
a mixed MLA+indexer physical block stride that is not an integer number of
576-element MLA rows.  The SM80 backend declares only the known-good layer-compact `LBHNC`
(legacy HND) layout. The launcher leaves layout selection to vLLM's backend
resolver and rejects a conflicting inherited `VLLM_KV_CACHE_LAYOUT` instead
of pinning the same value twice.

### #54296

Generic slot-mapping OOB guard for enabled cache groups whose block table can be
narrower than raw token positions.

Full GLM-5.3 in this port uses `tokens_per_state=1` and block size 64 for the
main MLA/indexer caches.  Its block tables span the raw token sequence, so the
reported narrow/compressed-group condition is not present.  The global
BlockTable implementation is intentionally left untouched.

### #58215

DeepSelect sentinel remap fix.  DeepSelect requires SM100a/SM103a; A100
`auto` Top-K resolves through the CUDA persistent/per-row path, so this fix is
outside the active SM80 execution path.

### #58450

GLM metadata preparation performance optimization.  It is not a correctness
requirement for this port.

### #49845

Upstream auto block-size selection fix. The r14 launcher does not pin
`--block-size`; the selected DSA/indexer backend advertises its exact
64-token contract and vLLM resolves it automatically. Target-hardware logs
confirm `DEEPSEEK_V32_INDEXER` selected block size 64.

## Observed A100 fused-Q SM80 compile failure

A later target-hardware run progressed through backend resolution
(`DEEPSEEK_V32_INDEXER` block size 64 and `LBHNC` layout) and failed during
the ordinary vLLM profile forward, before graph capture, at:

```text
deepseek_v32/attention.py -> fused_q()
_fused_q_kernel:
    ql_nope_fp8 = (ql_nope / scale).to(tl.float8e4nv)

ValueError: type fp8e4nv not supported in this architecture
```

Root cause: r10 made the index-Q/index-K quantizers SM80-safe but the
`QUANTIZE_MQA=True` branch in the same fused-Q Triton kernel still used native
`tl.float8e4nv` casts. The old GPU smoke only exercised
`quantize_mqa=False`, so the gap escaped the preflight.

r14 closes both sides of the issue:

- `--kv-cache-dtype bfloat16` is explicit in the A100 launcher, matching the
  `TRITON_MLA_SPARSE` BF16 main-KV contract and preventing FP8 query
  quantization in the supported production configuration;
- `TritonMLASparseImpl.supports_quant_query_input=False` is explicit and the
  backend rejects unsupported main-KV dtypes;
- all FP8 stores in `_fused_q_kernel` use software E4M3FN encoding into
  byte-addressed storage, so an accidental FP8-query route no longer fails at
  Triton compile time;
- the GPU smoke now explicitly invokes `fused_q(..., quantize_mqa=True)` and
  checks the packed NoPE+RoPE query against an FP8 reference;
- the launcher automatically runs that hardware smoke once per port revision
  before loading the 703.7 GiB checkpoint.

## Observed A100 CUDA-graph profiling OOM

On the target 2-node A100 deployment, PP0 loaded successfully at approximately
47.05 GiB per rank, then failed before KV-cache allocation inside
`profile_cudagraph_memory() -> capture_model()`.  The first PIECEWISE capture
descriptor exhausted the remaining device memory (20 MiB allocation attempted
with only ~16.8 MiB free).

This is not a model-weight load failure and not evidence of a sparse-MLA kernel
numerical failure.  vLLM sorts CUDA-graph capture descriptors largest-first; with
`max_cudagraph_capture_size=32`, the automatically generated PIECEWISE set is
`[32, 24, 16, 8, 4, 2, 1]`, so failure at 0/7 means the 32-token PIECEWISE
warmup itself is too expensive for the full FP8 checkpoint on PP0.

The intermediate `FULL_DECODE_ONLY / max_capture=32` mitigation was still
not a sufficiently conservative bring-up contract: vLLM profiles CUDA-graph
memory for every mode except `NONE`.  The r14 startup baseline is therefore:

```text
cudagraph_mode = NONE
```

This deliberately gives up CUDA-graph speed until full TP8×PP2 serving and
generation correctness are proven on A100. It removes graph profiling/capture
from startup instead of iteratively trying smaller capture envelopes.

## Open memory-risk item: #58068

#58068 is **not a demonstrated SM80 correctness failure** and is not
backported.  It addresses caching-allocator growth caused by changing prefill
logits widths.  Its published E2E evidence is GLM-5.3-Flash on GB10 unified
memory with the DeepGEMM path.

The SM80 Triton prefill fallback also allocates an `[M, N]` fp32 logits tensor,
so the allocation-shape pattern is relevant even though the kernel is
different.

For the production defaults:

```text
max_model_len            = 131072
max_num_batched_tokens   = 2048
max logits budget        = 512 MiB
compress_ratio           = 1
```

a single long prefill reaches the 512 MiB per-launch cap at ~64K context and is
sub-chunked beyond that.  Before the cap, progressively larger allocations can
create caching-allocator fragmentation.  vLLM's profile run accounts for the
**512 MiB peak allocation** when sizing KV memory, but the profiling context
calls `empty_cache()` afterward and therefore cannot prove that runtime
fragmentation from a sequence of differently-sized allocations is harmless.

This is therefore a **runtime memory telemetry item**, not a source blocker.
Do not claim 128K production memory stability until it has been measured on the
actual A100 deployment.

## Minimal SIF patch scope

The SIF is based directly on `vllm/vllm-openai:v0.30.0`; it installs no
additional runtime packages. The patcher copies four SM80-only modules and
edits only the upstream files needed to connect them or backport post-v0.30
correctness fixes.

| File / change | Why it is required |
|---|---|
| `v1/attention/backends/mla/triton_mla_sparse.py` | SM80 sparse-MLA backend; upstream v0.30 has no CUDA sparse MLA backend usable on A100 |
| `v1/attention/ops/mqa_logits_triton.py` | DSA indexer MQA/paged-MQA fallback because DeepGEMM is unsupported on SM80 |
| `v1/attention/ops/triton_mla_sparse_kernel.py` | sparse MLA attention kernel for SM80 |
| `v1/attention/ops/fp8_sm80.py` | portable E4M3FN encode for fused index-Q/index-K and fused MQA-query packing; native Triton FP8 conversion does not compile on SM80 |
| `v1/attention/backends/registry.py` | register `TRITON_MLA_SPARSE` |
| `v1/attention/backends/mla/indexer.py` | gate DeepGEMM by actual hardware support instead of importability |
| `model_executor/layers/sparse_attn_indexer.py` | route prefill/decode logits to Triton on SM80; includes #48285 and #52500 correctness fixes |
| `models/deepseek_v32/common/kernels.py` | make every active fused-Q/indexer FP8 store legal on SM80 while preserving FP8 byte semantics |
| `models/deepseek_v32/attention.py` | backport #58594 Top-K backend propagation |
| `model_executor/layers/attention/mla_attention.py` | publish sparse-MLA packed-block row alignment required by the Triton reader |
| `v1/kv_cache_interface.py`, `v1/core/kv_cache_utils.py` | carry and honor that physical block-stride alignment (#55528 lineage) |

Explicitly **not** patched: PP pinned-buffer V1 fix #47644, global BlockTable,
PIECEWISE KV-binding workarounds, GLM-5.3-Flash/KPool code, DeepSelect,
speculative decoding, generic scheduler code, NCCL code, or model weight
loading. #47522's Marlin packed-int32 prefill fix is already present in exact
vLLM 0.30.0 and is validated rather than patched.

The SIF also contains the patch script, revision marker, and GPU smoke script
for verification. Those files do not alter vLLM runtime behavior.

## Source/CI validation

Latest audited workflow is tracked at the branch HEAD; CI must pass exact
vLLM 0.30 patch application, idempotence, semantic checks, and launcher/SIF
contracts before the revision is considered buildable.

The workflow performs:

1. live GLM-5.3 config/index validation;
2. checkout of exact vLLM `v0.30.0`;
3. complete patch application;
4. touched-file scope check;
5. second patch application and byte-identical idempotence check;
6. `git diff --check`;
7. Python compile of every modified/installed module;
8. static semantic validation;
9. E4M3FN numerical reference validation:
   - 100,514 random/edge float32 values,
   - all 65,536 FP16 bit patterns,
   - all 65,536 BF16 bit patterns;
10. active-path native-FP8 exclusion checks;
11. long-context int64 address checks;
12. production launcher/SIF revision checks.

Observed markers from the successful run:

```text
GLM53_CURRENT_MODEL_CONTRACT=PASS
GLM53_FULL_SM80_V030_PATCH=PASS
SM80_PATCH_SCOPE=PASS
SM80_PATCH_IDEMPOTENT=PASS
GLM53_FULL_SM80_STATIC_SEMANTICS=PASS
SM80_E4M3FN_REFERENCE=PASS
SM80_FP8_ACTIVE_PATHS=PASS
GLM53_FULL_SM80_V030_STATIC=PASS
CUDA13_PROFILE=PASS
```

## What cannot be proven without A100 hardware

The source audit cannot validate:

- CUDA/Triton JIT on SM80;
- numerical parity of the five GPU smoke kernels on actual A100;
- Marlin checkpoint load/repack peak memory;
- NCCL/Gloo initialization across the two physical nodes;
- TP8 x PP2 full checkpoint initialization;
- CUDA-graph performance on A100 (r14 intentionally runs graph-free after
  target-hardware graph profiling exhausted HBM);
- prefix-cache behavior with real requests;
- runtime allocator fragmentation during 64K/128K prefills;
- end-to-end generated-token correctness.

The production launcher runs the A100 GPU smoke once per `PORT_REVISION` on
each node before the expensive full-model load, caches a success stamp, and
rejects a stale SIF by `PORT_REVISION`.

## Release gate

```text
SOURCE / CONFIG / CPU REFERENCE AUDIT = PASS
SIF STATIC CONTRACT                 = PASS
A100 GPU KERNEL AUDIT               = PENDING HARDWARE
16-GPU FULL SERVING AUDIT           = PENDING HARDWARE
128K MEMORY-STABILITY AUDIT         = PENDING HARDWARE
```

The next meaningful evidence must therefore come from the target A100 nodes,
not from additional source-only assertions.
