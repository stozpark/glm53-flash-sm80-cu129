# GLM-5.3 on A100/A800 (SM80) / CUDA 13.0

이 브랜치는 **full `zai-org/GLM-5.3`**를 NVIDIA A100/A800(SM80)에서 실행하기 위한 포팅 브랜치입니다.

> Flash 모델(`GLM-5.3-Flash`)용 브랜치가 아닙니다. full GLM-5.3은 `GlmMoeDsaForCausalLM / glm_moe_dsa`이고 DeepSeek-V3.2 계열 DSA sparse MLA 경로를 사용합니다.

## 현재 상태

- 기준 vLLM: **v0.30.0**
- 기준 CUDA image: **`vllm/vllm-openai:v0.30.0-cu129`**
- 대상 GPU: **A100/A800 (SM80)**
- 권장 16-GPU 토폴로지: **2 nodes × 8 A100, TP8 × PP2**
- main KV cache: **BF16**
- DSA indexer cache: **FP8**
- sparse MLA: **`TRITON_MLA_SPARSE`**
- FP8 linear/MoE: **Marlin**
- MTP: **초기 검증에서는 OFF**
- prefix caching: 초기 correctness 확인 뒤 단계적으로 켜는 것을 권장

### 확인한 것

2026-09-22 기준으로 이 브랜치의 patcher를 **vLLM v0.30.0 원본 checkout에 실제 적용**했고 다음을 확인했습니다.

```text
GLM53_FULL_SM80_PATCH=PASS
Python compile: PASS
GLM53_FULL_SM80_STATIC=PASS
```

GitHub Actions에서도 동일한 검사가 통과했습니다.

### 아직 확인하지 않은 것

**A100 16장 실제 하드웨어에서 full GLM-5.3을 아직 끝까지 기동한 상태는 아닙니다.**

따라서 현재 상태는:

```text
코드 포팅 + vLLM 0.29.0 적용/컴파일 확인 완료
                         ↓
A100 ×16 실제 모델 로딩/serving 검증 필요
```

입니다.

---

# 왜 GLM-5.3-Flash 포팅과 다른가

full GLM-5.3과 Flash는 attention 구조가 다릅니다.

| | GLM-5.3 | GLM-5.3-Flash |
|---|---|---|
| vLLM architecture | `GlmMoeDsaForCausalLM` | `Glm5NextForCausalLM` |
| model type | `glm_moe_dsa` | `glm5_next` |
| 계열 | DeepSeek-V3.2 DSA | GLM5Next/KDA |
| KPool/KPoolTail | 없음 | 있음 |
| sparse MLA | DSA | KPool 기반 GLM5Next sparse MLA |
| 이 브랜치의 대상 | **O** | X |

따라서 Flash용으로 만들었던 KPool/KPoolTail/NoPE-512 패치는 이 브랜치에 재사용하지 않습니다.

full GLM-5.3은 기존 DeepSeek-V3.2 sparse-attention stack을 SM80에서 실행 가능하게 만드는 게 핵심입니다.

---

# A100에서 원래 막히는 부분

Hopper에서 GLM-5.3 DSA는 크게 다음 accelerated path에 의존합니다.

```text
DSA indexer
  └─ DeepGEMM FP8 MQA / paged-MQA

Sparse MLA
  └─ FlashMLA-Sparse / FlashInfer sparse

FP8 Linear / MoE
  └─ Hopper-optimized FP8 kernels
```

A100은 SM80이라 이 경로를 그대로 사용할 수 없습니다.

이 브랜치는 이를 다음으로 바꿉니다.

```text
DSA indexer
  DeepGEMM
      ↓
  Triton FP8 MQA / paged-MQA

Sparse MLA
  FlashMLA-Sparse
      ↓
  TRITON_MLA_SPARSE

FP8 Linear / MoE
      ↓
  Marlin
```

---

# 기반 구현

핵심은 vLLM PR **#38476**입니다.

```text
[Feature] TRITON_MLA_SPARSE backend for SM8x/11x/12x DSA Sparse MLA Support
head: 3740c02bb1223d37823593664ae3eafa397b9937
```

이 PR은 명시적으로 **GLM-5 / DeepSeek-V3.2를 A100/A800(SM80)에서 실행**하기 위한 구현이며, 8×A100에서 GLM-5.1 계열 모델을 실제 벤치마크했습니다.

다만 이 브랜치는 PR을 그대로 복사하지 않고 vLLM 0.29.0에 맞춰 다음을 추가했습니다.

1. 최신 vLLM 0.29.0 indexer/API에 맞춘 dispatch
2. 장문 context용 int64 address 계산
3. paged-MQA 마지막 block OOB store 방지
4. PP stage 간 DSA top-k relay
5. SM80에서는 BF16 main KV cache 강제
6. native FP8 weight는 Marlin으로 실행

---

# 1. Triton DSA indexer

DSA indexer는 실제 sparse attention 전에 어느 KV token을 볼지 선택합니다.

원래 CUDA 경로는:

```text
FP8 Q × FP8 index K
        ↓
DeepGEMM fp8_mqa_logits
        ↓
Top-K
```

입니다.

A100에서는 DeepGEMM을 쓸 수 없으므로:

```text
is_deep_gemm_supported()
  ├─ true  → 기존 DeepGEMM
  └─ false → Triton fallback
```

으로 dispatch합니다.

추가되는 kernel:

```text
vllm/v1/attention/ops/mqa_logits_triton.py
```

Prefill:
- FP8 Q/K를 BF16으로 decode
- Triton `tl.dot`
- ReLU + head weight
- chunked-prefill 범위 밖에는 반드시 `-inf`

Decode:
- paged indexer K cache를 직접 읽음
- E4M3FN byte → BF16 LUT decode
- causal mask 적용
- Top-K용 FP32 logits 생성

MXFP4 indexer cache는 이 SM80 fallback에서 지원하지 않습니다.

---

# 2. TRITON_MLA_SPARSE

추가 파일:

```text
vllm/v1/attention/backends/mla/triton_mla_sparse.py
vllm/v1/attention/ops/triton_mla_sparse_kernel.py
```

full GLM-5.3의 sparse MLA는 compressed latent KV 기준으로:

```text
DMODEL = 512
DPE    = 64
DQK    = 576
DV     = 512
```

를 사용합니다.

이는 config의 `qk_nope_head_dim=192`, `qk_rope_head_dim=64`와 다른 레벨의 차원입니다. sparse MLA kernel은 projection 전 원래 attention-head QK 256차원이 아니라 **MLA compressed latent + RoPE representation**을 처리합니다.

## split-KV

decode batch가 작을 때 A100 SM을 충분히 채우기 위해 Top-K 축을 나눕니다.

```text
top-k
 ├─ split 0 ─┐
 ├─ split 1 ─┤
 ├─ split 2 ─┼─ partial output/LSE → online-softmax merge
 └─ split N ─┘
```

후보:

```text
1, 2, 4, 8, 16
```

입니다.

---

# 3. 장문 context correctness 보강

PR #38476 원본에 추가한 수정입니다.

## Sparse MLA

주소 계산:

```text
token_index × stride_kv_token
```

은 긴 context에서 int32 범위를 넘을 수 있습니다.

K / KPE / V 세 load 모두 index를 `tl.int64`로 올린 뒤 offset을 계산합니다.

## paged MQA

```text
physical_block_index × block_stride
```

도 같은 문제가 있으므로 block-table에서 읽은 physical block id를 즉시 `tl.int64`로 승격합니다.

마지막 page는 항상 완전히 차 있지 않으므로 store mask도:

```text
k_offset < context_len
```

을 포함합니다.

이 종류의 버그는 crash보다 **잘못된 KV를 조용히 읽는 형태**가 더 위험하기 때문에 초기부터 포함했습니다.

---

# 4. TP8 × PP2에서 DSA top-k를 stage 사이에 전달

16×A100을 2노드로 쓸 경우 권장 구성은:

```text
node 0: A100 ×8 ─ TP8 ┐
                      ├─ PP2
node 1: A100 ×8 ─ TP8 ┘
```

입니다.

TP16을 노드 사이에 걸면 매 layer TP collective가 네트워크를 통과합니다. TP8×PP2는 TP traffic을 각 노드 NVLink 안에 두고, PP activation만 노드 사이로 넘길 수 있습니다.

하지만 DSA의 `index_topk_freq > 1` 때문에 별도 correctness 문제가 있습니다.

일부 layer는 indexer를 새로 계산하지 않고 이전 full-indexer layer의 Top-K를 공유합니다.

PP stage가 그 shared layer 중간에서 시작하면 rank-local `topk_indices_buffer`에는 **현재 batch의 Top-K가 없을 수 있습니다.**

그래서 PP boundary에서:

```text
previous PP stage
  current topk_indices.clone()
             ↓
IntermediateTensors
             ↓
next PP stage
  local topk buffer에 copy
             ↓
leading shared layer 실행
```

하도록 수정했습니다.

`clone()`을 쓰는 이유는 PP send가 비동기인데 rank-local top-k buffer는 다음 microbatch에서 재사용되기 때문입니다.

buffer attribute도 private name인:

```text
_sm80_pp_topk_relay_buf
```

를 사용합니다. speculative proposer가 public `topk_indices_buffer`를 찾아 draft buffer와 alias하는 것을 피하기 위해서입니다.

---

# 5. FP8 weight: Marlin

GLM-5.3 native checkpoint는 FP8 weight를 사용합니다.

A100에서는 Hopper FP8 GEMM 대신:

```bash
--linear-backend marlin
--moe-backend marlin
```

을 사용합니다.

vLLM의 FP8 Marlin 경로는 Ampere를 지원합니다.

GLM DSA router는 FP32 routing이 필요합니다. vLLM 0.29.0의 `deepseek_v2.py`에는 이미:

```text
model_type == glm_moe_dsa
→ router dtype = torch.float32
```

처리가 들어 있으므로 별도 패치를 하지 않습니다.

---

# 6. 왜 main KV는 BF16인가

초기 SM80 포팅에서는:

```bash
--kv-cache-dtype bfloat16
```

을 사용합니다.

PR #38476의 SM80 Triton sparse MLA는 BF16 main KV를 대상으로 검증되어 있습니다.

여기서 FP8인 것은 **DSA indexer용 별도 K cache**입니다.

```text
main MLA KV     = BF16
indexer K cache = FP8
model weights   = FP8
```

세 가지를 구분해야 합니다.

---

# 빌드

```bash
git checkout glm53-full-sm80

bash ./build_glm53_full_sm80_sif.sh \
  /path/to/glm53-full-sm80-vllm030-cu130.sif
```

fakeroot가 필요하면:

```bash
BUILD_FLAGS=--fakeroot \
  bash ./build_glm53_full_sm80_sif.sh \
  /path/to/glm53-full-sm80-vllm030-cu130.sif
```

빌드 중 다음이 나와야 합니다.

```text
GLM53_FULL_SM80_PATCH=PASS
GLM53_FULL_SM80_STATIC=PASS
```

---

# 권장 초기 실행: 2 nodes × A100 8장

두 노드에:
- 같은 SIF
- 같은 model path
- 같은 CUDA/driver 환경

이 있어야 합니다.

## node 0

```bash
MODEL_HOST_PATH=/models/GLM-5.3 \
SIF_PATH=/path/to/glm53-full-sm80-vllm030-cu130.sif \
MASTER_ADDR=<NODE0_IP> \
NODE_RANK=0 \
bash ./serve_glm53_full_tp8_pp2.sh
```

## node 1

```bash
MODEL_HOST_PATH=/models/GLM-5.3 \
SIF_PATH=/path/to/glm53-full-sm80-vllm030-cu130.sif \
MASTER_ADDR=<NODE0_IP> \
NODE_RANK=1 \
bash ./serve_glm53_full_tp8_pp2.sh
```

기본값:

```text
TP=8
PP=2
EP=ON
KV=BF16
Sparse MLA=TRITON_MLA_SPARSE
Linear=Marlin
MoE=Marlin
max_model_len=128K
max_num_seqs=2
max_num_batched_tokens=4096
MTP=OFF
```

처음부터 1M context나 MTP를 켜지 마십시오. baseline correctness가 먼저입니다.

---

# 단일 노드 A100 ×16

한 서버에 물리적으로 16장이 달려 있다면:

```bash
MODEL_HOST_PATH=/models/GLM-5.3 \
SIF_PATH=/path/to/glm53-full-sm80-vllm030-cu130.sif \
bash ./serve_glm53_full_tp16.sh
```

를 사용할 수 있습니다.

일반적인 8-GPU ×2 node 구성에서는 TP8×PP2를 우선 권장합니다.

---

# 검증 순서

실제 A100에서는 아래 순서로 올리는 것을 권장합니다.

1. **128K / MTP OFF / prefix cache OFF**
2. 짧은 deterministic prompt의 H100/reference output 비교
3. chunked prefill 32K/64K/128K
4. 동시 request 2개
5. PP boundary가 포함된 반복/교차 request correctness
6. prefix caching ON
7. 256K
8. 512K
9. MTP1
10. MTP3 이상
11. 1M context

특히 2-node PP2에서는 **서로 다른 두 request를 번갈아 넣는 테스트**가 중요합니다. stale top-k relay 문제가 이런 workload에서 가장 잘 드러납니다.

---

# 주요 파일

```text
patch_glm53_full_sm80.py
  vLLM 0.29.0에 SM80 DSA 경로와 PP top-k relay 적용

vendor/glm53-full-sm80/
  외부 PR을 빌드 시 다시 다운로드하지 않도록 고정한 Triton source

Singularity.glm53-full-sm80.def
  vllm-openai:v0.30.0-cu129 기반 SIF

build_glm53_full_sm80_sif.sh
  SIF 빌드

serve_glm53_full_tp8_pp2.sh
  2-node × 8-GPU 권장 baseline

serve_glm53_full_tp16.sh
  single-node 16-GPU profile
```

---

# 출처

- vLLM v0.30.0
- vLLM PR #38476 — SM80 DSA Triton sparse MLA/indexer
- `zai-org/GLM-5.3`
- vLLM GLM-5.3 recipe
- `bayley/vllm-170hx-glm5` — SM80 DSA/PP correctness 사례

PR #38476 코드는 Apache-2.0 vLLM 코드에 기반합니다.

---

# 현재 결론

이 브랜치에서 구현한 A100 경로는:

```text
GLM-5.3 FP8 checkpoint
        │
        ├─ FP8 Linear/MoE → Marlin
        │
        ├─ DSA indexer
        │      DeepGEMM → Triton FP8 MQA
        │
        ├─ Sparse MLA
        │      Hopper sparse backend → TRITON_MLA_SPARSE
        │
        └─ TP8 × PP2
               └─ top-k selections relay across PP stages
```

입니다.

**vLLM 0.29.0 source-level 적용/컴파일은 통과했습니다. 다음 단계는 A100 ×16에서 실제 GLM-5.3 checkpoint를 로드해 runtime 문제를 잡는 것입니다.**
