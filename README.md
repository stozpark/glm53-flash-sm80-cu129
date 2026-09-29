# Full GLM-5.3 on 16x A100/A800 (SM80)

이 브랜치는 **full `zai-org/GLM-5.3`** 를 A100/A800 80GB 16장
(**2 nodes x 8 GPUs**)에서 vLLM으로 서비스하기 위한 SM80 포트입니다.

- 대상 architecture: `GlmMoeDsaForCausalLM / glm_moe_dsa`
- 대상 attention: DeepSeek-V3.2 계열 DSA sparse MLA
- 대상 topology: **TP8 x PP2**
- container base: **vLLM 0.30.0 / CUDA 13.0**
- model weights: native block-FP8
- Linear/MoE: vLLM **auto** (SM80에서는 지원 가능한 fallback 선택)
- main MLA KV: **BF16** (`--kv-cache-dtype bfloat16`; official FP8-KV recipe를 SM80에서 override)
- DSA indexer cache: FP8 E4M3FN
- sparse MLA backend: **TRITON_MLA_SPARSE**

> 이 브랜치는 `GLM-5.3-Flash` 포트가 아닙니다.
> `glm5_next`, KPool/KPoolTail, Flash 모델 전용 NoPE 패치는 이 경로에 사용하지 않습니다.

## Upstream 기준

SM80 sparse-MLA 구현은 vLLM PR **#47629**의 실제 A800 E2E 경로를
기준으로 삼고, 현재 DeepSeek-V3.2/GLM DSA API와 SM80 correctness 이슈를
반영했습니다.

추가로 확인한 upstream 변경:

- **#55173**: pre-SM89 CUDA용 portable E4M3 conversion
- **#58594**: full GLM-5.3 sparse-indexer Top-K backend propagation (v0.30 이후 merge)
- **#55528/#56254**: V3.2 sparse MLA packed physical-block stride alignment
- **#51395**: sparse-only MLA backend가 dense-MHA prefill을 광고하면 안 된다는 capability fix
- **#47644**: 구형 V1 model runner용 PP pinned-buffer fix이며 현재 V2 runner에는 사용하지 않음

최신 vLLM main도 검토했지만, NVIDIA SM80용 `TRITON_MLA_SPARSE`가
upstream main에 정식 포함된 상태는 아니므로 이 브랜치에서 필요한 SM80
부분만 유지합니다.

## SM80에서 필요한 변경

### 1. DSA indexer: DeepGEMM -> Triton fallback

A100에서는 DeepGEMM FP8 MQA 경로를 사용할 수 없으므로 다음 fallback을
사용합니다.

```text
FP8 index-Q x FP8 index-K
        |
        +-- SM90+ : DeepGEMM
        |
        +-- SM80  : Triton FP8 MQA / paged-MQA
```

파일:

```text
vendor/glm53-full-sm80/v1/attention/ops/mqa_logits_triton.py
```

long-context correctness를 위해 physical block address 계산은 int64로
승격하고, 마지막 paged block은 `k_offset < context_len`으로 mask합니다.

### 2. Sparse MLA: TRITON_MLA_SPARSE

파일:

```text
vendor/glm53-full-sm80/v1/attention/backends/mla/triton_mla_sparse.py
vendor/glm53-full-sm80/v1/attention/ops/triton_mla_sparse_kernel.py
```

full GLM-5.x DSA의 compressed latent geometry:

```text
DQK = 576
DV  = 512
```

A100에서는 split-KV decode를 사용합니다.

### 3. SM80 software E4M3FN

현재 DeepSeek-V3.2 fused index-Q/index-K kernel은 native
`tl.float8e4nv` conversion을 사용하며 SM80에서 실패할 수 있습니다.

이 브랜치는 E4M3FN bit pattern을 `uint8`에 직접 기록하고 Python 경계에서
`torch.float8_e4m3fn` zero-copy view를 사용합니다.

```text
vendor/glm53-full-sm80/v1/attention/ops/fp8_sm80.py
```

실제 GLM-5.3 DSA config에 맞춰:

```text
index_n_heads          = 32
index_head_dim         = 128
index_topk             = 2048
indexer_rope_interleave = true
```

를 사용합니다.

### 4. 최신 GLM/DSA correctness backport

v0.30 릴리스 이후 merge된 full GLM-5.3/DSA 수정도 필요한 부분만 backport합니다.

- **#58594**: `sparse_attn_indexer()`에 실제 선택된 `topk_backend` 전달
- **#55528/#56254 최소 backport**: `TRITON_MLA_SPARSE`의 packed BF16 MLA
  physical block stride를 576-element row(1152 bytes) 경계에 정렬
- **#51395 패턴**: Triton sparse backend는 dense `forward_mha()`를 구현하지
  않으므로 `supports_dense_mha_prefill=False`를 명시하고 short/chunked
  prefill도 지원되는 sparse-MQA 경로로 보냄

## TP8 x PP2 partition

GLM-5.3은 indexer Top-K 하나를 4개 layer 묶음에서 공유합니다.

기본 39/39 split은:

```text
layer 38 = full indexer
--------- PP boundary ---------
layer 39 = shared indexer
```

가 되어 PP1이 자기 rank에서 생성되지 않은 Top-K를 소비할 수 있습니다.

따라서 production launcher는 기본적으로:

```bash
VLLM_PP_LAYER_PARTITION=42,36
```

을 사용합니다.

```text
PP0: layers 0..41
PP1: layers 42..77
```

layer 42가 full-indexer layer이므로 stage 간 Top-K relay가 필요 없습니다.

## Production defaults

별도의 bring-up profile을 두지 않습니다. 기본 launcher 자체가 운영 설정입니다.

```text
TP=8
PP=2
PP partition=42,36
CUDA graph=NONE (A100 80GB startup-safe baseline)
prefix caching=vLLM default (ON)
block size=vLLM/backend auto-resolution (TRITON_MLA_SPARSE -> 64)
KV cache layout=vLLM/backend auto-resolution (TRITON_MLA_SPARSE -> LBHNC)
KV dtype=BF16 (SM80 backend contract)
Linear/MoE=vLLM auto
MTP=OFF
max model len=131072
```

scheduler concurrency는 vLLM 0.30의 A100 OpenAI-server 기본값을 그대로
사용합니다.

```text
max_num_batched_tokens = 2048
max_num_seqs           = 128
```

환경변수로 명시했을 때만 override합니다.

## Build

```bash
git checkout glm53-full-sm80-cu130-v030

bash ./build_glm53_full_sm80_sif.sh \
  /path/to/glm53-full-sm80-vllm030-cu130.sif
```

빌드 시 `patch_glm53_full_sm80.py`가 vLLM 0.30.0 source에 모든 SM80
변경을 적용합니다.

## GPU smoke

full checkpoint를 읽기 전에 SIF 안의 실제 Triton kernels를 A100에서
검증합니다. Production launcher는 새 `PORT_REVISION`마다 각 노드의 첫 GPU에서
smoke를 **한 번만 자동 실행**하고 성공 stamp를 남깁니다
(`RUN_GPU_SMOKE=auto`). 같은 revision 재시작에서는 다시 실행하지 않습니다.

```bash
GPU=0 bash ./verify_glm53_full_sm80_sif.sh \
  /path/to/glm53-full-sm80-vllm030-cu130.sif
```

PASS 항목:

```text
SM80_FUSED_NORM_ROPE_INDEX_K=PASS
SM80_FUSED_Q_INDEX_Q=PASS
SM80_FUSED_Q_MQA_FP8_PACK=PASS
SM80_TRITON_MQA_PREFILL=PASS
SM80_TRITON_MQA_DECODE=PASS
SM80_TRITON_MLA_SPARSE=PASS
GLM53_FULL_SM80_GPU_SMOKE=PASS
```

smoke test도 실제 GLM-5.3처럼 `indexer_rope_interleave=true`를 사용합니다.
또한 launcher는 SIF 내부의 `PORT_REVISION`과 vLLM 0.30.0, 필수 patch marker를
확인하므로 오래된 SIF를 실수로 실행하면 full model load 전에 실패합니다.

## Run: 2 nodes x 8 A100

두 노드 모두 같은 model path와 SIF를 사용합니다.

node 0:

```bash
MODEL_HOST_PATH=/models/GLM-5.3 \
SIF_PATH=/path/to/glm53-full-sm80-vllm030-cu130.sif \
MASTER_ADDR=<NODE0_IP> \
NODE_RANK=0 \
bash ./serve_glm53_full_tp8_pp2.sh
```

node 1:

```bash
MODEL_HOST_PATH=/models/GLM-5.3 \
SIF_PATH=/path/to/glm53-full-sm80-vllm030-cu130.sif \
MASTER_ADDR=<NODE0_IP> \
NODE_RANK=1 \
bash ./serve_glm53_full_tp8_pp2.sh
```

multi-NIC이면 필요할 때만:

```bash
NET_IFACE=ens11np0
```

등을 지정합니다.

## 현재 검증 상태

GitHub CI에서 현재 다음을 모두 검증합니다.

- live `zai-org/GLM-5.3/config.json` contract
  - `GlmMoeDsaForCausalLM / glm_moe_dsa`
  - 78 layers / 6144 hidden / 256 routed experts
  - indexer 32x128 / top-k 2048 / interleaved RoPE
  - layer 38 full, 39 shared, 42 full
  - dynamic E4M3 / weight block `[128,128]`
- exact vLLM 0.30.0 source에 patch 적용
- patch가 의도한 source set에만 한정되는지 scope 검사
- patch를 두 번 적용해도 byte-identical인지 idempotence 검사
- `git diff --check` 및 모든 patched Python module compile
- v0.30 model registry가 full GLM을 DeepSeek-V3.2 path로 라우팅하는지 검사
- GLM MoE router FP32 처리와 `glm47` tool/reasoning parser 존재 확인
- v0.30 CLI에 production launcher의 모든 option 존재 확인
- FP8 128x128 block quantization의 SM80 fallback 지원 여부 확인
- SM80 software E4M3FN reference 검사
  - random/edge float32 100k+
  - FP16 전체 65,536 bit patterns
  - BF16 전체 65,536 bit patterns
  - signed NaN/Inf/overflow saturating semantics
- active index-Q/index-K와 fused MQA-query pack에 native `tl.float8e4nv` cast가 남지 않았는지 검사
- `quantize_mqa=True` 경로를 실제 SM80 JIT + <=1 ULP reference로 검사
- `record_logical_topk_ready()` compatibility hook
- #58594 Top-K backend propagation
- #55528/#56254 packed MLA block-stride alignment
- sparse-only backend의 dense-MHA prefill 비활성화 (#51395 패턴)
- #48285의 2-D decode seq_lens / fixed logits width 회귀 방지
- #47522의 Marlin packed-int32 chunked-prefill dtype 회귀 방지
- A100 startup baseline에서 CUDA graph를 완전히 비활성화해 observed graph-profile OOM 회피
- exact 64-token DSA page contract
- 42/36 PP partition이 두 stage 모두 full-indexer layer에서 시작하는지 검사
- obsolete V1 #47644 / custom PP Top-K relay / global BlockTable mutation 부재 확인
- production launcher: native MP / BF16 MLA KV / graph NONE / prefix cache default / SIF revision preflight
- shell syntax

- 128K sparse-indexer prefill logits budget
  - v0.30 A100 server default: 2048 batched tokens
  - logits allocation cap: 512 MiB
  - 128K context: at most 1024 rows/launch, therefore two <=512 MiB launches per 2048-token chunk

이 CI가 확인할 수 없는 것은 **실제 SM80 GPU 실행 자체**입니다. 따라서 남은
하드웨어 의존 검증은 A100에서의 전체 Triton JIT/numerical smoke, 16-GPU NCCL/PP
initialization, weight load/repack peak memory 및 실제 generation E2E입니다. CUDA graph는
현재 startup-safe baseline에서 의도적으로 비활성화되어 있습니다.

## Main files

```text
patch_glm53_full_sm80.py
vendor/glm53-full-sm80/
Singularity.glm53-full-sm80.def
build_glm53_full_sm80_sif.sh
verify_glm53_full_sm80_sif.sh
serve_glm53_full_tp8_pp2.sh
tests/sm80_glm53_full_kernel_smoke.py
tests/validate_glm53_full_static.py
tests/test_fp8e4m3fn_reference.py
PORT_REVISION
```
