# GLM-5.3-Flash on SM80 / CUDA 12.9

[Korean README](README.md)

Build and patch bundle for running `zai-org/GLM-5.3-Flash` on NVIDIA A100/A800 (SM80) using the official `vllm/vllm-openai:glm53-flash-cu129` image as the base.

> This is an experimental selected backport, not official vLLM support for GLM-5.3 on SM80.

## Selected patch strategy

- **vLLM PR #54031** for native NoPE `dim_qk=512` `TRITON_MLA_SPARSE` based on `FlashMLASparseImpl`.
- **Latest pinned PR #47629** for the SM80 FP8 MQA/paged-MQA Triton fallback and long-context correctness fixes.
- **GLM-5.3 KPool routing patch** so `sparse_attn_indexer_kpool.py` also uses the SM80 Triton fallback, including the MTP `(B,next_n)` seq-lens collapse required by the paged MQA kernel.
- **A800 reference project** only for the SM80-specific `kpool_compress.py` workaround.
- **PR #47644** for the PP pinned-input-buffer race.

`xpu_mla_sparse.py` is intentionally left stock; GLM-5.3 NoPE sparse attention is routed through the #54031-style Triton backend instead.

## Build

Prepare pinned vendor sources on a machine with Internet access:

```bash
bash ./prepare_vendor.sh
```

Build the SIF:

```bash
bash ./build_sif.sh /path/to/glm53-flash-sm80-cu129.sif
```

Validate kernels on one A100 before loading the full model:

```bash
bash ./verify_sif_gpu.sh /path/to/glm53-flash-sm80-cu129.sif
```

## vLLM profiles

Two profiles are provided.

### Initial correctness profile

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
sparse_mla_force_mqa=true
```

Run:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=initial bash ./serve_tp8.sh
```

### Target operating profile

Use only after the initial profile passes long-context, tool-call and WebFetch correctness checks.

```text
TP=8
Expert Parallel=ON
gpu_memory_utilization=0.90
max_model_len=524288
max_num_seqs=8
max_num_batched_tokens=8192
KV cache=BF16
MTP=5
prefix caching=ON
```

Run:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=ideal bash ./serve_tp8.sh
```

The 512K profile is a practical target for A100 80GB x8. Validate 1M context separately after 512K is stable.

For two nodes, prefer TP8 x PP2 over TP16 across nodes. PR #47644 is included for the PP batch-queue input-buffer race. Validate PP with MTP disabled first.

## Suggested validation order

1. short deterministic prompt
2. 32K / 64K / 128K needle retrieval
3. prompt longer than 8192 tokens to exercise chunked prefill
4. Claude Code tool calling
5. Claude Code WebFetch
6. MTP5
7. prefix caching
8. 256K / 512K / optional 1M
9. TP8 x PP2

## Upstream references

- vLLM: https://github.com/vllm-project/vllm
- SM80 Triton sparse MLA: PR #47629
- GLM-5.3 NoPE 512: PR #54031
- PP pinned-buffer race: PR #47644
- A800 reference project: https://gitee.com/kill-life/glm5.3-flash-deployment-a800
