# GLM-5.3-Flash on SM80 / CUDA 12.9

[Korean README](README.md)

Build and patch bundle for running `zai-org/GLM-5.3-Flash` on NVIDIA A100/A800 (SM80) using the official `vllm/vllm-openai:glm53-flash-cu129` image as the base.

> This is a selected backport for SM80 rather than official upstream GLM-5.3 support.

## Validation status

### ✅ Validated on A100 80GB ×8

As of **2026-09-01**, the `serve_tp8_ideal.sh` profile has been confirmed to serve GLM-5.3-Flash correctly on a single node with **8× NVIDIA A100-SXM4-80GB**.

Validated operating profile:

```text
GPU=A100-SXM4-80GB x8
TP=8
Expert Parallel=ON
gpu_memory_utilization=0.90
max_model_len=524288 (512K configured)
max_num_seqs=8
max_num_batched_tokens=8192
KV cache=BF16
MTP=3
prefix caching=ON
attention backend=TRITON_MLA_SPARSE
sparse_mla_force_mqa=true
MoE backend=marlin
NCCL_ALGO=Ring
NCCL_PROTO=Simple
custom all-reduce=OFF
```

Recommended validated single-node topology:

```text
A100 80GB x8
TP=8
EP=ON
PP=1
serve_tp8_ideal.sh
```

> Successful serving with `max_model_len=524288` does not by itself mean that a full 512K-token request has been stress-tested end-to-end. Very long 256K/512K/1M requests should still be validated separately if required.

> TP8×PP2 multi-node serving is not considered validated by this repository yet.

## Selected patch strategy

- **vLLM PR #54031** for native NoPE `dim_qk=512` `TRITON_MLA_SPARSE`.
- **PR #47629** for SM80 FP8 MQA/paged-MQA Triton fallback, large-stride int64 addressing and chunked-prefill correctness fixes.
- **Pinned Ampere backport** (`wtdcode/vllm-backport@0ef4bff...`) for SM80 software E4M3FN KPool writes and GLM-5.3 KPoolTail/MTP fixes.
- **Mrzhiyao A800 reference kernel** for the A800-tested sparse MLA kernel while retaining the #54031-compatible CUDA sparse-attention plumbing.
- **Sparse backend selector repair** so `TRITON_MLA_SPARSE` survives the GLM sparse backend priority `pop()` logic on Ampere.
- **DeepGEMM hard-gate removal** so A100 correctly falls through to the SM80 Triton sparse-indexer path.
- **MTP stream fences** to avoid draft-buffer reuse races.
- **PR #47644 compatibility patch** for the PP pinned-input-buffer race.

Notable correctness fixes include:

- unified-KV row/stride address promotion to int64
- paged-MQA final-block bounds
- chunked-prefill dirty-logit protection
- KPool compressed-width argument for persistent top-k
- persistent KPoolTail slot mappings for CUDA graph replay
- padded KPoolTail slots initialized to `-1`
- KPoolTail exclusion from generic slot mapping
- MTP seq-len and position propagation fixes

## Build

Prepare pinned vendor sources:

```bash
bash ./prepare_vendor.sh
```

Build the SIF:

```bash
bash ./build_sif.sh /path/to/glm53-flash-sm80-cu129.sif
```

The build should finish with:

```text
STATIC_VERIFY=PASS
```

Validate kernels on one A100 before loading the full model:

```bash
GPU=0 bash ./verify_sif_gpu.sh /path/to/glm53-flash-sm80-cu129.sif
```

Expected result:

```text
SIF_GPU_VERIFY=PASS
```

## Offline source backup

The `source-backup` branch stores pinned vendor sources, exact-base reconstruction files, reference implementations and checksums so the build can be recovered even if upstream repositories or PRs disappear.

## vLLM profiles

### Initial diagnostic profile

```text
TP=8
Expert Parallel=ON
gpu_memory_utilization=0.85
max_model_len=131072
max_num_seqs=8
max_num_batched_tokens=8192
KV cache=BF16
MTP=OFF
prefix caching=OFF
TRITON_MLA_SPARSE
sparse_mla_force_mqa=true
```

Run:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=initial bash ./serve_tp8.sh
```

### Recommended operating profile — validated on A100 ×8

Run:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=ideal bash ./serve_tp8.sh
```

or directly:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
./serve_tp8_ideal.sh
```

Default settings:

```text
TP=8
Expert Parallel=ON
gpu_memory_utilization=0.90
max_model_len=524288
max_num_seqs=8
max_num_batched_tokens=8192
KV cache=BF16
MTP=3
prefix caching=ON
TRITON_MLA_SPARSE
sparse_mla_force_mqa=true
MoE backend=marlin
NCCL_ALGO=Ring
NCCL_PROTO=Simple
custom all-reduce=OFF
```

Test MTP5 separately before adopting it:

```bash
NUM_SPEC_TOKENS=5 PROFILE=ideal bash ./serve_tp8.sh
```

## Still requiring separate stress validation

- actual 256K / 512K full-context requests
- 1M context
- MTP5+
- TP8×PP2 multi-node
- long-duration/high-concurrency soak tests

## Upstream references

- vLLM: https://github.com/vllm-project/vllm
- SM80 Triton MQA fallback: PR #47629
- GLM-5.3 NoPE 512 sparse MLA: PR #54031
- PP pinned-buffer race: PR #47644
- Ampere GLM-5.3 backport: https://github.com/wtdcode/vllm-backport
- A800 native-FP8 reference: https://github.com/Mrzhiyao/glm53-a800-vllm
- A800 historical comparison: https://gitee.com/kill-life/glm5.3-flash-deployment-a800
