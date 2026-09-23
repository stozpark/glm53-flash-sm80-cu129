# Full GLM-5.3 on 16x A100/A800 (SM80)

이 브랜치는 **full `zai-org/GLM-5.3`** 를 A100/A800 80GB 16장
(**2 nodes x 8 GPUs**)에서 vLLM으로 서비스하기 위한 SM80 포트입니다.

- 대상 architecture: `GlmMoeDsaForCausalLM / glm_moe_dsa`
- 대상 attention: DeepSeek-V3.2 계열 DSA sparse MLA
- 대상 topology: **TP8 x PP2**
- container base: **vLLM 0.30.0 / CUDA 13.0**
- model weights: native block-FP8
- Linear/MoE: **Marlin**
- main MLA KV: **BF16**
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
- **#54851**: DeepSeek-V3.2/GLM-5.x PIECEWISE CUDA-graph KV binding fix
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

### 4. PIECEWISE CUDA-graph KV binding

vLLM #54851에서 보고된 DSA graph-capture 문제를 v0.30 코드에 맞춰
backport했습니다.

PIECEWISE capture 시 attention metadata가 없어도 실제 KV cache view와
persistent slot-mapping buffer를 graph에 연결해 두어야 합니다. 그렇지 않으면
decode graph가 KV write를 영구히 생략할 수 있습니다.

marker:

```text
SM80_PIECEWISE_KV_BINDING_FIX
```

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
CUDA graph=ON
prefix caching=ON
block size=64
main MLA KV=BF16
Linear=Marlin
MoE=Marlin
MTP=OFF (별도 enable 전까지)
max model len=131072
```

scheduler concurrency는 vLLM 0.30의 A100 OpenAI-server 기본값을 그대로
사용합니다.

```text
max_num_batched_tokens = 2048
max_num_seqs           = 256
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
검증할 수 있습니다. Production launcher도 기본적으로 각 노드에서 이 smoke를
한 번 실행한 뒤 서버를 시작합니다 (`RUN_GPU_SMOKE=1`).

```bash
GPU=0 bash ./verify_glm53_full_sm80_sif.sh \
  /path/to/glm53-full-sm80-vllm030-cu130.sif
```

PASS 항목:

```text
SM80_FUSED_NORM_ROPE_INDEX_K=PASS
SM80_FUSED_Q_INDEX_Q=PASS
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
- patch가 의도한 10개 source에만 한정되는지 scope 검사
- patch를 두 번 적용해도 byte-identical인지 idempotence 검사
- `git diff --check` 및 모든 patched Python module compile
- v0.30 model registry가 full GLM을 DeepSeek-V3.2 path로 라우팅하는지 검사
- GLM MoE router FP32 처리와 `glm47` tool/reasoning parser 존재 확인
- v0.30 CLI에 production launcher의 모든 option 존재 확인
- FP8 128x128 block quantization -> Marlin linear/MoE support 확인
- SM80 software E4M3FN reference 검사
  - random/edge float32 100k+
  - FP16 전체 65,536 bit patterns
  - BF16 전체 65,536 bit patterns
  - signed NaN/Inf/overflow saturating semantics
- active index-Q/index-K path에 native `tl.float8e4nv` cast가 남지 않았는지 검사
- `record_logical_topk_ready()` compatibility hook
- #54851-equivalent PIECEWISE KV-binding fix
- exact 64-token DSA page contract
- 42/36 PP partition이 두 stage 모두 full-indexer layer에서 시작하는지 검사
- obsolete V1 #47644 / custom PP Top-K relay / global BlockTable mutation 부재 확인
- production launcher: native MP / graph ON / prefix cache ON / SIF revision preflight
- shell syntax

이 CI가 확인할 수 없는 것은 **실제 SM80 GPU 실행 자체**입니다. 따라서 남은
하드웨어 의존 검증은 A100에서의 Triton JIT/numerical smoke, 16-GPU NCCL/PP
initialization, Marlin weight load/repack peak memory, CUDA-graph replay 및 실제
generation E2E입니다.

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
