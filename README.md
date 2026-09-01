# GLM-5.3-Flash on A100/A800 (SM80) / CUDA 12.9

[English README](README.en.md)

`zai-org/GLM-5.3-Flash`를 NVIDIA A100/A800(SM80)에서 돌리기 위한 vLLM + Singularity/Apptainer 패치입니다.

기준 이미지는 공식 GLM-5.3용 CUDA 12.9 이미지입니다.

```text
vllm/vllm-openai:glm53-flash-cu129
```

이 저장소의 방향은 **최신 vLLM을 새로 빌드하는 것**이 아닙니다. GLM-5.3 모델 구현이 이미 들어 있는 공식 이미지를 그대로 두고, **Hopper 전용인 실행 경로만 Ampere에서 동작하는 Triton/Marlin 경로로 교체**합니다.

2026-09-01 기준으로 **A100-SXM4-80GB ×8 단일 노드에서 `serve_tp8_ideal.sh`로 실제 serving이 정상 동작하는 것까지 확인했습니다.**

---

# 한눈에 보기

A100에서 막히던 핵심은 모델 자체가 아니라 아래 세 군데였습니다.

| 구간 | 원래 GLM-5.3 CUDA 경로 | A100에서 문제 | 이 저장소에서 쓰는 경로 |
|---|---|---|---|
| Sparse MLA | FlashMLA / SM90+ sparse kernel | Hopper/Blackwell 전용 | `TRITON_MLA_SPARSE` |
| Sparse indexer score | DeepGEMM FP8 MQA | SM80에서 사용 불가 | Triton FP8 MQA / paged-MQA |
| KPool FP8 cache write/read | native `fp8e4nv` convert | Triton이 pre-SM89 convert를 허용하지 않음 | software E4M3FN encode + BF16 LUT decode |
| MoE/FP8 weight 실행 | Hopper 쪽 고성능 backend 우선 | A100에서 그대로 못 씀 | vLLM Marlin backend |

즉 A100 지원을 위해 모델 구조를 바꾼 게 아닙니다.

```text
GLM-5.3 model graph / checkpoint
        그대로 유지
             │
             ├─ sparse indexer: DeepGEMM → Triton
             ├─ sparse MLA:     FlashMLA sparse → Triton sparse MLA
             ├─ KPool FP8 I/O:  native FP8 convert → software byte encode/decode
             └─ MoE:            Marlin backend 사용
```

중요한 점 하나:

> **메인 MLA KV cache는 BF16입니다.**
>
> 여기서 별도로 FP8 처리가 필요한 부분은 sparse token을 고르는 **indexer의 KPool cache**입니다. `--kv-cache-dtype bfloat16`을 쓰는 이유도 이 둘을 구분하기 위해서입니다.

---

# 실제 확인한 환경

현재 확인된 단일 노드 설정입니다.

| 항목 | 값 |
|---|---:|
| GPU | **NVIDIA A100-SXM4-80GB ×8** |
| Compute Capability | **SM80** |
| TP | **8** |
| Expert Parallel | **ON** |
| PP | **1** |
| GPU memory utilization | **0.90** |
| max model len | **524288 (512K configured)** |
| max num seqs | **8** |
| max num batched tokens | **8192** |
| Main KV cache | **BF16** |
| MTP | **3 draft tokens** |
| Prefix caching | **ON** |
| Sparse MLA | **TRITON_MLA_SPARSE** |
| `sparse_mla_force_mqa` | **true** |
| MoE backend | **Marlin** |
| NCCL | **Ring / Simple** |
| custom all-reduce | **OFF** |

권장 실행은 `serve_tp8_ideal.sh`입니다.

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
./serve_tp8_ideal.sh
```

`max_model_len=524288`로 서버가 올라오고 일반 요청이 정상 처리되는 것까지 확인했습니다. **실제 512K 토큰 요청을 끝까지 채운 stress test를 했다는 뜻은 아닙니다.** 256K/512K/1M full-context는 별도로 확인하는 게 좋습니다.

---

# GLM-5.3 sparse attention이 어떻게 동작하는가

패치 이유를 이해하려면 GLM-5.3의 sparse attention을 먼저 보는 게 빠릅니다.

대략적인 흐름은 다음과 같습니다.

```text
hidden states
    │
    ├─ Indexer Q
    │
    ├─ Indexer K
    │      │
    │      └─ KPool compression
    │             여러 token → 하나의 pooled K
    │
    ├─ query × pooled-K score
    │
    ├─ pool-level Top-K
    │
    ├─ 선택된 pool을 token index로 다시 확장
    │      + 아직 pool이 완성되지 않은 최신 tail token 추가
    │
    └─ Sparse MLA
           선택된 token만 실제 attention
```

## KPool을 쓰는 이유

GLM-5.3 indexer는 과거 모든 token을 매번 token 단위로 scoring하지 않습니다. 연속된 token 여러 개를 하나의 pool로 압축해서 indexer cache에 저장합니다.

`compress_ratio == index_kpool`이므로 vLLM의 indexer metadata도 자동으로 pool 단위가 됩니다.

```text
token sequence
0 1 2 3 | 4 5 6 7 | 8 9 10 11 | ...
   pool 0     pool 1       pool 2
```

실제 KPool 압축은 단순 average가 아닙니다.

```text
score_i = gate_i + APE_i
weight_i = softmax(score_i)
pooled_K = Σ weight_i * K_i
```

그 뒤:

1. Hadamard-128 rotation
2. per-vector absmax scale 계산
3. E4M3FN FP8 quantization
4. indexer K cache에 pooled entry 저장

을 한 Triton kernel에서 처리합니다.

Indexer가 Top-K를 구할 때도 token이 아니라 **pool을 먼저 선택**합니다.

```text
pool Top-K
   ↓
선택된 각 pool을 원래 token index들로 확장
   ↓
아직 압축되지 않은 마지막 incomplete pool의 token도 뒤에 추가
```

따라서 indexer logits의 가로 길이는 **token sequence length가 아니라 pool length**입니다. 뒤에서 설명하는 `persistent_topk()` 길이 버그가 여기서 생깁니다.

---

# H100에서는 되는데 A100에서 바로 안 되는 이유

## 1. Sparse MLA kernel이 SM90+ 기준

기본 GLM-5.3 CUDA 경로는 sparse MLA에서 FlashMLA 계열 backend를 우선합니다.

A100은 SM80이므로 native sparse MLA 후보들이 compute capability 검사에서 빠집니다.

또 일반 `TRITON_MLA`는 dense MLA backend라 `use_sparse=True` 요청을 받을 수 없습니다.

따라서 A100에서는 별도의 sparse backend가 필요합니다.

```text
TRITON_MLA              → dense MLA
TRITON_MLA_SPARSE       → sparse MLA  ← 추가
```

## 2. GLM-5.3은 NoPE 512 layout

GLM-5.3의 sparse MLA는 다음 형태입니다.

```text
latent / NoPE dim = 512
RoPE suffix       = 0
qk dim            = 512
```

기존 DeepSeek/GLM 계열 sparse MLA 구현에는 다음 576 layout도 있습니다.

```text
latent dim = 512
RoPE dim   = 64
qk dim     = 576
```

그래서 새 Triton kernel은 512와 576을 둘 다 처리하도록 되어 있습니다.

```python
_SUPPORTED_DIM_QK = (512, 576)
block_dpe = dim_qk - 512
```

GLM-5.3에서는 `block_dpe == 0`이 되고:

```python
if BLOCK_DPE > 0:
    ... RoPE load / dot ...
```

분기가 compile-time에 제거됩니다. 즉 576 kernel을 억지로 512에 맞추는 게 아니라 **NoPE 512가 실제 kernel geometry의 한 경우**가 됩니다.

## 3. Sparse indexer가 DeepGEMM에 묶여 있었음

원래 KPool indexer는 score 계산에서 다음 DeepGEMM op를 사용합니다.

```text
fp8_fp4_mqa_logits
fp8_fp4_paged_mqa_logits
```

Hopper에서는 좋은 경로지만 A100에서는 이걸 전제로 하면 모델 로딩 단계에서 바로 막힙니다.

그래서 `has_deep_gemm()` 같은 단순 설치 여부가 아니라 **현재 GPU에서 실제 DeepGEMM을 쓸 수 있는지**를 보는 `is_deep_gemm_supported()`로 바꾸고, 지원되지 않으면 Triton fallback으로 보냅니다.

```text
DeepGEMM supported
    ├─ yes → 기존 DeepGEMM path
    └─ no  → fp8_mqa_logits_triton / fp8_paged_mqa_logits_triton
```

## 4. A100에서는 Triton native FP8 convert도 안 됨

A100은 SM80입니다. Triton의 `fp8e4nv` 변환은 pre-SM89 CUDA에서 사용할 수 없습니다.

즉 아래 같은 연산을 kernel에서 그대로 쓸 수 없습니다.

```python
x.to(tl.float8e4nv)
```

하지만 **FP8 데이터를 저장하는 것 자체가 불가능한 건 아닙니다.** E4M3FN bit pattern을 직접 만들고 `uint8`로 쓰면 됩니다.

이 저장소에서는:

- write: software E4M3FN encoder
- read: 256-entry BF16 lookup table

을 사용합니다.

이게 A100에서 GLM-5.3 KPool FP8 cache를 유지할 수 있게 만든 핵심입니다.

---

# 패치를 이런 식으로 섞은 이유

한 저장소나 PR을 통째로 가져오지 않았습니다. 기준 API가 서로 다르고, 각 구현에서 필요한 부분도 다르기 때문입니다.

| 출처 | 가져온 것 | 그대로 가져오지 않은 것 | 이유 |
|---|---|---|---|
| 공식 `glm53-flash-cu129` | GLM-5.3 model/runtime 전체 | - | 모델 integration과 CUDA 12.9 환경을 기준점으로 유지 |
| vLLM PR #54031 | `TRITON_MLA_SPARSE` backend/plumbing | 최종 sparse kernel | CUDA FlashMLA metadata 흐름을 그대로 재사용하기 좋음 |
| vLLM PR #47629 | SM80 Triton MQA/paged-MQA | PR 전체 | indexer fallback과 long-context fix만 필요 |
| Mrzhiyao A800 | A800에서 실제 사용한 sparse MLA kernel | XPU 기반 backend plumbing | kernel은 검증 가치가 높지만 backend 구조는 현재 cu129 CUDA 경로와 다름 |
| wtdcode backport | SM80 FP8 encoder, KPool writer 및 Ampere correctness fix | fork 전체 | 현재 공식 image를 최대한 유지하면서 Ampere 관련 부분만 적용 |
| vLLM PR #47644 | overlapping batch event fix | PR 전체 | base에 이미 blocking event가 있어 필요한 조건만 backport |
| Gitee A800 repo | 비교 자료 | runtime dependency | 빌드 재현성을 위해 실제 build path에서는 제거 |

## 왜 #54031 backend + Mrzhiyao kernel 조합인가

Mrzhiyao 구현은 A800에서 실제로 돌린 코드라는 장점이 있습니다. 다만 backend가:

```python
TritonMLASparseImpl(XPUMLASparseImpl)
```

형태입니다.

현재 공식 GLM CUDA image에서는 이미 `FlashMLASparseImpl` 쪽에 다음 로직이 들어 있습니다.

- CUDA sparse metadata
- per-request index → global slot 변환
- prefill/decode routing
- NoPE MQA path
- MQA kernel warmup

이걸 버리고 XPU plumbing 전체로 갈 이유가 없습니다.

그래서 backend는 #54031의:

```python
TritonMLASparseImpl(FlashMLASparseImpl)
```

을 유지하고, **실제 attention kernel만 Mrzhiyao의 A800-tested 버전으로 교체**합니다.

이 조합이 현재 코드의 핵심 설계입니다.

## 왜 wtdcode fork를 통째로 쓰지 않았나

wtdcode 쪽에는 Ampere에서 필요한 수정이 빠르게 들어가 있지만, fork 전체를 기준으로 갈 경우 공식 `glm53-flash-cu129`와 다른 변경까지 같이 들어옵니다.

이 저장소는 다음 원칙을 잡았습니다.

```text
공식 GLM image를 기준으로 유지
+ SM80에서 필요한 kernel/correctness fix만 backport
```

그래서 wtdcode에서는 특히 의미가 명확한:

- FP8 software encode/decode
- KPool compress/write
- KPoolTail/MTP 관련 correctness fix

만 가져옵니다.

---

# 패치별로 정확히 무엇을 바꾸는가

빌드 때 적용되는 순서입니다.

```text
1. patch_47644_compat.py
2. patch_runtime.py
3. patch_cuda_sparse_priority.py
4. patch_deepgemm_gate.py
5. patch_kpool_topk_len.py
6. patch_mrzhiyao_merge.py
7. patch_glm53_tail.py
8. patch_mtp_stream_fence.py
9. verify_static.py
```

순서에도 의미가 있습니다. 먼저 cu129 코드에 backend와 fallback을 연결한 다음, A800 kernel과 Ampere fix를 덮고, 마지막에 전체 invariant를 검사합니다.

---

## `patch_runtime.py`

가장 큰 1차 패치입니다.

### `TRITON_MLA_SPARSE` 등록

vLLM backend registry에:

```text
TRITON_MLA_SPARSE
→ vllm.v1.attention.backends.mla.triton_mla_sparse.TritonMLASparseBackend
```

를 추가합니다.

등록만 하면 끝이 아니라 CUDA backend 후보에도 들어가야 합니다. 이 부분은 뒤의 `patch_cuda_sparse_priority.py`에서 최종 순서를 한 번 더 보정합니다.

### #54031 API를 cu129에 맞춤

#54031은 더 최신 vLLM API를 기준으로 작성됐습니다.

새 API의 sparse kernel hook은:

```python
(output, lse)
```

를 반환하지만 `glm53-flash-cu129@487ecf187`은 output tensor 하나를 기대합니다.

그래서 backend logic은 유지하면서 return contract만 현재 base에 맞춥니다.

### DeepGEMM 여부를 GPU 기준으로 판단

```text
has_deep_gemm()
→ is_deep_gemm_supported()
```

로 바꿉니다.

Python package가 설치돼 있는 것과 현재 GPU architecture에서 kernel을 실행할 수 있는 것은 다른 문제입니다. A100에서는 이 구분이 필요합니다.

### KPool prefill indexer fallback

기존:

```text
fp8_fp4_mqa_logits(...)
```

A100 fallback:

```text
fp8_mqa_logits_triton(...)
```

으로 바꿉니다.

### KPool decode indexer fallback

기존:

```text
fp8_fp4_paged_mqa_logits(...)
```

A100 fallback:

```text
fp8_paged_mqa_logits_triton(...)
```

으로 바꿉니다.

MTP에서는 `seq_lens`가 `[B, next_n]` 형태가 될 수 있지만 Triton paged-MQA kernel은 request당 현재 길이 하나인 `[B]`가 필요합니다.

그래서:

```python
seq_lens[:, -1].contiguous()
```

로 마지막 길이만 전달합니다.

---

## `patch_cuda_sparse_priority.py`

이 패치는 단순 backend 등록보다 중요합니다.

GLM CUDA selector에는 대략 다음 로직이 있습니다.

```python
sparse_tail = [
    ...,
    FLASHINFER_MLA_SPARSE_SM90,
]

if prefer_fi_sm90:
    sparse_tail.pop()
```

마지막 원소가 SM90 FlashInfer backend라는 가정으로 `pop()`을 합니다.

`TRITON_MLA_SPARSE`를 단순히 맨 뒤에 붙이면:

```text
[..., FLASHINFER_MLA_SPARSE_SM90, TRITON_MLA_SPARSE]
```

가 되고 `pop()`이 **A100용 Triton backend를 지워버립니다.**

그래서 최종 순서를 반드시:

```text
...
TRITON_MLA_SPARSE
FLASHINFER_MLA_SPARSE_SM90   ← pop 대상
```

으로 만듭니다.

패치 스크립트 안에서 실제로 `pop()`을 simulation해서:

- 빠지는 값이 `FLASHINFER_MLA_SPARSE_SM90`인지
- `TRITON_MLA_SPARSE`가 여전히 남아 있는지

둘 다 확인합니다.

---

## `patch_deepgemm_gate.py`

기존 KPool constructor에는:

```text
Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.
```

라는 hard error가 있었습니다.

이미 Triton fallback을 넣어도 constructor가 먼저 죽으면 아무 의미가 없습니다.

그래서 이 조건을:

```text
DeepGEMM unavailable
→ warning
→ Triton fallback 계속 진행
```

으로 바꿉니다.

FP4 index cache까지 무조건 지원하는 건 아닙니다. 현재 SM80 fallback은 **GLM-5.3에서 사용하는 FP8 K cache 경로**를 대상으로 합니다.

---

# Sparse MLA kernel

## #54031에서 가져온 핵심

GLM-5.3 NoPE 512 layout을 Triton에서 직접 처리합니다.

```python
_BLOCK_DMODEL = 512
_BLOCK_DPE = 64
_SUPPORTED_DIM_QK = (512, 576)
```

런타임 `dim_qk`로:

```python
block_dpe = dim_qk - 512
```

를 계산합니다.

GLM-5.3이면 0이므로 RoPE load/dot가 compile-time에 사라집니다.

## Mrzhiyao kernel을 최종 kernel로 쓰는 이유

`patch_mrzhiyao_merge.py`는 #54031 kernel을 최종적으로 Mrzhiyao A800 kernel로 바꿉니다.

이 kernel에는 A100/A800에서 중요한 low-batch decode 최적화가 들어 있습니다.

### Split-KV

decode에서 query가 몇 개 없으면 일반 `(token, head)` grid만으로 A100의 SM을 충분히 채우지 못합니다.

그래서 Top-K 축을 여러 조각으로 나눕니다.

```text
Top-K indices
│
├─ split 0 ─┐
├─ split 1 ─┤
├─ split 2 ─┼─ partial output + LSE
└─ split N ─┘
              ↓
            merge
```

후보 split 수:

```python
KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)
```

선택 기준은:

- 현재 `(num_tokens × head_groups)` grid가 이미 SM을 충분히 채우면 split하지 않음
- split당 Top-K가 너무 작아지지 않게 최소 128개 정도의 work를 유지
- A100의 실제 SM count를 기준으로 power-of-two split 선택

입니다.

즉 split-KV는 계산량을 줄이기 위한 게 아니라 **작은 decode batch에서 GPU occupancy를 높이기 위한 방법**입니다.

### online softmax

각 Top-K block을 순회하면서 전체 score를 한 번에 materialize하지 않고:

```text
e_max
e_sum
acc
```

을 갱신합니다.

split-KV에서는 각 split의 output과 LSE를 FP32 mid buffer에 저장한 다음 LSE 기준으로 다시 합칩니다.

### empty row NaN 방지

모든 index가 `-1`인 tile에서 `-inf - -inf`를 계산하면 NaN이 생길 수 있습니다.

그래서 kernel은 `-inf` 대신 큰 음수 finite sentinel을 쓰고:

```python
e_sum_safe = where(e_sum > 0, e_sum, 1.0)
```

로 zero-valid-KV row도 안전하게 처리합니다.

---

# `patch_mrzhiyao_merge.py`

Mrzhiyao에서 가져오는 것은 **kernel**입니다. backend 전체가 아닙니다.

그리고 두 가지를 추가로 적용합니다.

## KV cache block size는 64의 배수

backend에:

```python
return [MultipleOf(64)]
```

를 추가합니다.

고정 `[64]`가 아니라 `MultipleOf(64)`인 이유는 128, 256 같은 더 큰 block size도 허용하기 위해서입니다.

vLLM의 여러 attention/indexer backend가 하나의 KV cache group을 공유할 때 공통 block size를 찾아야 하는데, 아무 제한이 없으면 16 같은 값이 선택돼 sparse indexer 조건과 충돌할 수 있습니다.

GLM KPool 쪽에는 이보다 더 강한 조건도 있습니다.

```text
cache block_size % index_kpool == 0
storage block_size % 32 == 0
```

즉 `MultipleOf(64)`는 planner가 처음부터 말이 안 되는 작은 block을 고르지 않게 하고, 최종 KPool 조건은 GLM cache spec이 다시 확인합니다.

## 장문 context에서 KV offset을 int64로

Sparse MLA kernel의 실제 주소는:

```text
indices × stride_kv_token
```

입니다.

sequence index 자체는 int32에 들어가더라도, **flat element offset**은 긴 context에서 int32 범위를 넘을 수 있습니다.

그래서 K / KPE / V 세 load 모두:

```python
indices.to(tl.int64) * stride_kv_token
```

으로 계산합니다.

이건 1M급 context에서 조용히 잘못된 KV를 읽는 문제를 막기 위한 correctness fix입니다.

---

# SM80용 MQA indexer

`#47629`의 `mqa_logits_triton.py`를 사용합니다.

Indexer score는 대략 다음 구조입니다.

```text
FP8 Q
  ×
FP8 pooled K
  ↓
head별 dot-product
  ↓
ReLU
  ↓
head weight 적용
  ↓
head reduction
  ↓
각 pool의 index score
```

## Prefill

Prefill은 `M` query와 긴 `N` prefix를 처리합니다.

A100에서 FP8 LUT lookup을 inner loop마다 하는 것보다, Q/K를 BF16으로 먼저 decode한 뒤 `tl.dot` 하는 쪽이 더 효율적인 구조라 prefill kernel은 이 경로를 씁니다.

chunked prefill에서는 각 query row가 실제로 봐야 하는 `[ks, ke)` 범위가 다릅니다.

`clean_logits=False`일 때 output buffer를 `torch.empty()`로 잡기 때문에, range 밖 tile을 그냥 early return하면 이전 메모리 값이 남습니다.

그 값이 Top-K로 들어가면 crash가 아니라 **조용히 잘못된 token을 선택**할 수 있습니다.

그래서 early-exit tile에도 반드시:

```text
-inf
```

를 씁니다.

이 fix가 긴 tool result / WebFetch / chunked prefill에서 correctness에 특히 중요합니다.

## Paged decode

Decode는 block table을 따라 indexer K cache를 읽습니다.

통합 KV pool에서는 block 하나의 stride가 매우 크기 때문에:

```text
block_idx * stride
```

를 int32로 계산하면 긴 context나 여러 sequence에서 wraparound가 날 수 있습니다.

그래서:

```python
block_idx = ... .to(tl.int64)
```

로 승격합니다.

마지막 physical block도 항상 꽉 차 있는 게 아니므로 store에는:

```python
k_offset < context_len
```

mask가 들어갑니다.

causal 조건도:

```python
k_offset <= q_offset
```

으로 유지합니다.

---

# A100에서 FP8 KPool cache를 어떻게 유지하는가

## 문제

Triton은 SM80에서 다음 변환을 compile하지 못합니다.

```text
fp32/bf16 ↔ fp8e4nv
```

GLM-5.3 indexer K cache는 FP8 E4M3FN을 사용하므로 KPool write 단계에서 막힙니다.

## 해결: FP8을 숫자 type이 아니라 byte format으로 다룸

`wtdcode/vllm-backport`의 `fp8_sm80.py`를 사용합니다.

### encode

FP32 bit pattern에서 직접:

- sign
- exponent
- mantissa
- round-to-nearest-even
- finite saturation

을 계산해 E4M3FN byte를 만듭니다.

최대 finite magnitude는 `±448`입니다.

결과는 `uint8` view로 같은 FP8 tensor storage에 씁니다.

즉 tensor의 논리 dtype은 `float8_e4m3fn`으로 유지하면서 **kernel write만 byte 단위로 우회**합니다.

### decode

256개 가능한 FP8 byte를 미리 BF16 값으로 바꿔 둔 lookup table을 사용합니다.

```text
uint8 FP8 bit pattern
        ↓
256-entry LUT
        ↓
BF16
```

A100에서는 이 방식이 software ALU unpack보다 register pressure가 낮고, indexer decode에서 쓰기 좋은 구조입니다.

SM89+에서는 같은 코드가 native FP8 path를 그대로 사용합니다.

---

# KPool writer

`patch_glm53_tail.py`에서 최종 KPool writer를 wtdcode Ampere 버전으로 교체합니다.

Pool 하나에 대해 한 Triton program이:

```text
gate + APE
   ↓
softmax
   ↓
weighted K sum
   ↓
Hadamard-128
   ↓
absmax FP8 quantization
   ↓
KPool cache write
```

를 한 번에 처리합니다.

A100에서는 마지막 FP8 write만 software byte encoder로 바뀌고, 수학적 KPool 정의는 그대로 유지합니다.

---

# `patch_kpool_topk_len.py`

이건 작은 수정이지만 의미가 큽니다.

KPool이 켜지면 logits는 token 길이가 아니라 **pool 길이**입니다.

그런데 기존 `persistent_topk()` 호출은 마지막 인자로:

```python
attn_metadata_narrowed.max_seq_len
```

즉 token 단위 길이를 넘기고 있었습니다.

host 쪽에서 알고 있는 row width와 실제 device logits width가 달라질 수 있습니다.

그래서:

```python
logits.shape[1]
```

을 넘기도록 바꿉니다.

원칙은 단순합니다.

> Top-K가 읽는 tensor의 row length는 그 tensor의 실제 width를 써야 한다.

---

# KPoolTail이 필요한 이유

완성된 pool은 FP8 KPool cache에 들어가지만, 최신 token들은 아직 pool을 다 채우지 못했을 수 있습니다.

예를 들어 pool size가 4라면:

```text
... | 100 101 102 103 | 104 105
        complete pool    incomplete tail
```

`104, 105`를 버리면 다음 decode에서 `106, 107`이 들어왔을 때 올바른 pool을 만들 수 없습니다.

그래서 request마다 최근 incomplete pool의:

```text
raw BF16 K
raw BF16 gate score
```

를 별도 circular tail cache에 보관합니다.

```text
position % kpool
```

로 ring slot을 정합니다.

pool 마지막 token이 들어오면 tail에 저장해 둔 이전 token들과 현재 token을 모아 정상적인 KPool compression을 수행합니다.

---

# `patch_glm53_tail.py`

KPoolTail과 MTP가 같이 동작할 때 필요한 여러 correctness fix를 묶어 둔 패치입니다.

## `positions`를 반드시 전달

KPoolTail의 물리 slot은 단순 block table만으로 계산할 수 없습니다.

```text
request tail block + position % kpool
```

이 필요합니다.

그래서 hybrid model metadata와 MTP draft metadata에 token-level `positions`를 전달합니다.

## generic slot mapping에서 KPoolTail 제외

일반 KV cache slot mapping은 position이 계속 증가한다고 가정합니다.

KPoolTail은 circular buffer라:

```text
position → position % kpool
```

이어야 합니다.

generic mapping을 그대로 적용하면 장문 context에서 tail block 밖 주소를 만들 수 있습니다.

그래서 KV cache group별로 `slot_mapping_enabled` flag를 두고 `KpoolTailSpec` group은 generic Triton slot-mapping kernel을 건너뜁니다.

## padded slot은 항상 `-1`

MTP나 padded batch에서는 실제 token이 아닌 row가 생깁니다.

이 row에 이전 값이 남아 있으면 tail writer가 stale physical slot에 데이터를 쓸 수 있습니다.

따라서 persistent buffer를 매번:

```python
fill_(-1)
```

하고 실제 token만 새 slot을 씁니다.

## CUDA graph 때문에 mapping buffer를 persistent하게 유지

CUDA graph는 capture 시 사용한 tensor pointer를 replay에서도 그대로 사용합니다.

매 step마다:

```python
slot_mapping.clone()
```

같이 temporary tensor를 만들면 capture 뒤 tensor가 해제되고 같은 주소가 다른 allocation에 재사용될 수 있습니다.

그래서 runner lifetime 동안 유지되는:

```text
tail_slot_mapping_buffer
```

를 미리 만들고 계속 재사용합니다.

이건 CUDA graph replay correctness를 위한 수정입니다.

## MTP padded seq_lens

MTP warmup에서는 `seq_lens` tensor가 실제 decode request보다 크게 padding될 수 있습니다.

따라서:

```python
seq_lens[:num_decodes] // compress_ratio
```

만 사용합니다.

전체 tensor를 그대로 넣으면 destination slice 크기와 맞지 않거나 padded row가 실제 request처럼 처리될 수 있습니다.

## Prefix caching에서 KPoolTail은 공유하지 않음

KPoolTail은 request의 **진행 중인 pool 상태**입니다. 같은 prefix를 가진 request라도 현재 tail의 write phase까지 일반 KV block처럼 공유하면 안 됩니다.

그래서 `KpoolTailSpec.participates_in_prefix_caching()`은 `False`입니다.

메인 KV/prefix cache는 그대로 사용하면서 mutable tail만 공유 대상에서 제외됩니다.

---

# MTP와 KPool

MTP에서는 한 request가 한 step에 여러 draft token을 verify할 수 있습니다.

```text
[B, next_n]
```

형태가 되므로 plain decode보다 신경 쓸 부분이 많습니다.

## request별 token 순서를 유지해야 함

한 request 안에서:

```text
token t   → tail에 stash
token t+1 → 앞 token을 포함해서 pool completion 검사
```

순서가 중요합니다.

그래서 KPool decode kernel은 request 하나를 한 Triton program이 맡고 `next_n` token을 position 순서대로 처리합니다.

request끼리는 서로 다른 tail block을 쓰므로 병렬 처리할 수 있습니다.

## Triton paged-MQA에는 1D seq_lens 전달

vLLM MTP metadata는 `[B, next_n]` seq_lens를 가질 수 있지만 paged-MQA kernel은 request당 현재 context length 하나가 필요합니다.

```python
seq_lens[:, -1]
```

을 넘깁니다.

반면 downstream Top-K/tail expansion에서는 token별 position 정보가 필요하므로 원래 2D metadata를 무조건 없애면 안 됩니다. 두 의미를 분리해서 사용합니다.

---

# `patch_mtp_stream_fence.py`

MTP draft input buffer는 다음 step에서도 재사용됩니다.

비동기 CUDA stream에서 write가 끝나기 전에 다음 step이 같은 buffer를 읽으면 아주 가끔 illegal memory access나 잘못된 입력이 발생할 수 있습니다.

그래서 non-captured path에서는 draft buffer update 뒤 현재 stream을 synchronize합니다.

```python
if not torch.cuda.is_current_stream_capturing():
    torch.accelerator.current_stream().synchronize()
```

CUDA graph capture 중에는 synchronization을 넣으면 안 되고, graph 안에서는 replay ordering 자체가 고정되므로 skip합니다.

Prefill 이후와 multi-step decode 이후 두 군데에 fence가 들어갑니다.

---

# chunked prefill에서 왜 틀린 답이 나올 수 있었나

이 포크에서 가장 위험한 종류의 버그는 crash보다 **silent wrong output**입니다.

`max_num_batched_tokens=8192`보다 긴 prompt는 chunked prefill로 들어갑니다.

Indexer logits tensor를 `clean_logits=False`로 `torch.empty()`에서 만들면 모든 위치가 초기화돼 있다는 보장이 없습니다.

각 query가 보는 `[ks, ke)`와 겹치지 않는 Triton tile이 그냥 return하면:

```text
이전 GPU memory 값
→ logits에 남음
→ Top-K가 높은 값으로 오인
→ 엉뚱한 pool/token 선택
→ sparse MLA가 잘못된 context를 읽음
→ 모델은 crash 없이 이상한 답을 생성
```

이 때문에 `mqa_logits_triton.py`는 early-exit tile에도 `-inf`를 명시적으로 씁니다.

A100에서 짧은 prompt는 정상인데 긴 tool result/WebFetch에서만 이상해지는 경우 특히 먼저 의심해야 하는 지점입니다.

---

# 장문 context에서 int64 주소 계산이 필요한 이유

두 군데가 서로 다른 문제입니다.

## Paged MQA

```text
physical block index × unified-KV block stride
```

가 큽니다.

block index는 작아 보여도 stride가 매우 크므로 결과가 int32를 넘을 수 있습니다.

그래서 block table에서 읽자마자 `tl.int64`로 바꿉니다.

## Sparse MLA

```text
token index × stride_kv_token
```

의 **element offset**이 긴 context에서 int32를 넘을 수 있습니다.

K / KPE / V 세 주소 계산을 모두 int64로 바꿉니다.

둘 다 Python tensor shape은 정상이고 kernel도 실행되기 때문에, overflow가 나면 crash보다 잘못된 memory 위치를 읽는 형태가 될 수 있습니다.

---

# `patch_47644_compat.py`

PP나 concurrent engine batch에서는 pinned CPU input tensor를 다음 batch가 재사용하기 전에 이전 GPU 소비가 끝났는지 동기화해야 합니다.

upstream #47644의 취지는 맞지만 현재 base는 이미:

```python
torch.cuda.Event(blocking=True)
```

를 사용하고 있습니다.

그래서 upstream patch를 그대로 덮지 않고 blocking semantics는 유지한 채 event 생성 조건만:

```python
max_concurrent_batches > 1
```

로 넓힙니다.

TP8 단일 노드에서는 거의 영향이 없고 PP 확장을 위한 방어 코드에 가깝습니다.

현재 **TP8×PP2는 아직 이 저장소에서 실제 운영 확인한 설정으로 보지 않습니다.**

---

# Marlin을 쓰는 이유

이 저장소에서 sparse attention만 따로 고쳤다고 모델 전체가 A100에서 끝나는 건 아닙니다. GLM-5.3 checkpoint의 FP8 weight/MoE 실행 경로도 A100에서 가능한 backend를 써야 합니다.

`serve_tp8_ideal.sh`는:

```text
VLLM_TEST_FORCE_FP8_MARLIN=1
--moe-backend marlin
```

을 사용합니다.

즉 weight/MoE 쪽은 새 kernel을 이 저장소에서 만드는 대신 **vLLM에 이미 있는 Ampere-compatible Marlin 경로를 사용**합니다.

이 저장소가 직접 수정하는 핵심은 sparse MLA/indexer/KPool입니다.

---

# 실행 옵션을 이렇게 잡은 이유

`serve_tp8_ideal.sh`의 기본값입니다.

## `--kv-cache-dtype bfloat16`

메인 sparse MLA의 KV를 BF16으로 둡니다.

A100에서 native FP8 sparse MLA까지 억지로 열지 않고, indexer KPool만 software FP8 cache를 사용합니다.

## `--attention-config '{"backend":"TRITON_MLA_SPARSE","sparse_mla_force_mqa":true}'`

A100에서 반드시 우리가 추가한 sparse backend로 가게 합니다.

`force_mqa`는 sparse MLA의 MQA-compatible 경로를 명시적으로 사용합니다.

## `--enable-prefix-caching`

KPoolTail만 prefix-sharing 대상에서 제외하도록 고쳤기 때문에 메인 prefix cache는 켤 수 있습니다.

## MTP3

```text
num_speculative_tokens=3
```

으로 실제 동작을 확인했습니다.

MTP5는 별도 A/B test 후 쓰는 것을 권장합니다.

## `NCCL_NVLS_ENABLE=0`

이 설정은 sparse attention 알고리즘 자체와는 관계없습니다. A100 단일 노드에서 불필요한 NVLS 경로를 피하기 위한 운영 설정입니다.

## `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `--disable-custom-all-reduce`

최대 throughput보다는 A100 8장 환경에서 안정적인 통신 경로를 우선한 설정입니다. 모델 correctness를 만드는 핵심 patch는 아닙니다.

---

# 빌드 재현성

`prepare_vendor.sh`는 가져오는 파일과 commit을 모두 고정합니다.

```text
base      vLLM 487ecf187
#47629    064801dd2bc6ac2e265dc3fa1f5d803d71bde25d
#54031    b325d908656d05e2a650ec60666ccec6f4f3eb0c
wtdcode   0ef4bff219c098d48cf16d3d63ebef329e9b74b0
Mrzhiyao  daeccb983ec84756cde7408b0e29161d492ea2c5
```

각 파일 SHA256도 `vendor/MANIFEST.txt`에 기록합니다.

`source-backup` 브랜치에는 실제 사용하는 vendor 파일과 기준 vLLM 파일을 같이 보관합니다.

```text
backup/vendor/
backup/base487/
backup/backport-reference/
backup/mrzhiyao-reference/
backup/a800-reference/
backup/SHA256SUMS.txt
```

외부 PR이나 저장소가 없어져도 해당 snapshot으로 다시 만들 수 있습니다.

---

# CI에서 확인하는 것

`.github/workflows/audit-patch-chain.yml`은 `source-backup`에 저장된 정확한 base 파일을 복원하고 patch를 처음부터 다시 적용합니다.

확인 항목은 다음과 같습니다.

- 모든 patch가 지정한 위치에 정확히 한 번 적용되는지
- 수정된 Python 파일이 compile되는지
- `TRITON_MLA_SPARSE`가 registry에 있는지
- GLM selector의 `pop()` 뒤에도 Triton sparse backend가 남는지
- sparse kernel이 512/576을 지원하는지
- split-KV 후보와 empty-row guard가 있는지
- K/KPE/V 세 offset이 모두 int64인지
- paged-MQA block index가 int64인지
- 마지막 page OOB mask가 있는지
- chunked-prefill early-exit가 `-inf`를 쓰는지
- DeepGEMM hard gate가 남아 있지 않은지
- KPool `persistent_topk`가 실제 logits width를 쓰는지
- KPoolTail buffer가 persistent + `-1` 초기화인지
- generic slot mapping에서 KPoolTail이 빠지는지
- MTP positions / padded seq_lens fix가 있는지
- MTP stream fence 두 군데가 있고 graph-capture safe한지
- SM80 software FP8 encoder를 실제 KPool writer가 사용하는지
- pinned source checksum이 맞는지

정상 빌드에서는 마지막에:

```text
STATIC_VERIFY=PASS
```

가 나옵니다.

이건 GPU runtime test를 대신하는 건 아닙니다. 그래서 SIF를 만든 다음 `verify_sif_gpu.sh`도 따로 실행합니다.

---

# 빌드

## 1. vendor 준비

온라인:

```bash
./prepare_vendor.sh
```

오프라인:

```bash
git fetch origin source-backup
git worktree add ../glm53-source-backup origin/source-backup

BACKUP_VENDOR=../glm53-source-backup/backup/vendor \
  ./prepare_vendor.sh
```

## 2. SIF 생성

```bash
./build_sif.sh /path/to/glm53-flash-sm80-cu129.sif
```

fakeroot가 필요하면:

```bash
BUILD_FLAGS='--fakeroot' \
  ./build_sif.sh /path/to/glm53-flash-sm80-cu129.sif
```

## 3. GPU kernel 확인

```bash
GPU=0 ./verify_sif_gpu.sh /path/to/glm53-flash-sm80-cu129.sif
```

정상이면:

```text
SIF_GPU_VERIFY=PASS
```

가 나옵니다.

---

# 실행

## 권장 설정

A100 80GB ×8에서 실제로 정상 동작을 확인한 설정입니다.

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
./serve_tp8_ideal.sh
```

`serve_tp8.sh` wrapper를 쓰면:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=ideal ./serve_tp8.sh
```

## 문제를 분리해서 보고 싶을 때

MTP와 prefix caching을 끈 보수적인 설정도 남겨 두었습니다.

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=initial ./serve_tp8.sh
```

`initial`은:

```text
128K max model len
GMU 0.85
MTP OFF
Prefix caching OFF
```

입니다.

---

# 아직 따로 확인할 것

현재 A100 ×8 / TP8 / EP / MTP3 / prefix caching ON 조합은 정상 동작을 확인했습니다.

아래는 별도 stress test 대상으로 남겨 둡니다.

- 실제 256K / 512K full-context 요청
- 1M context
- MTP5 이상
- TP8×PP2 멀티노드
- 장시간 high-concurrency soak test

---

# 파일별 역할

```text
Singularity.def
  공식 glm53-flash-cu129 이미지에 patch를 순서대로 적용

prepare_vendor.sh
  upstream 파일을 정확한 commit으로 고정해서 준비

patch_runtime.py
  Triton sparse backend 등록, #54031 API 적응,
  DeepGEMM→Triton MQA fallback wiring

patch_cuda_sparse_priority.py
  GLM selector의 pop() 이후에도 TRITON_MLA_SPARSE가 남도록 순서 수정

patch_deepgemm_gate.py
  A100에서 DeepGEMM hard error 대신 Triton fallback 허용

patch_kpool_topk_len.py
  pool-granular logits에 token-granular max_seq_len을 넘기던 문제 수정

patch_mrzhiyao_merge.py
  A800-tested sparse MLA kernel 적용,
  MultipleOf(64), long-context int64 offset 추가

patch_glm53_tail.py
  SM80 FP8 writer + KPoolTail + positions + CUDA graph + MTP 관련 수정

patch_mtp_stream_fence.py
  MTP draft input buffer stream race 방지

patch_47644_compat.py
  overlapping engine batch / PP pinned-input reuse 동기화

verify_static.py
  위 조건이 실제 최종 vLLM source에 모두 남아 있는지 검사
```

---

# 참고한 upstream

- vLLM: https://github.com/vllm-project/vllm
- SM80 Triton MQA fallback: vLLM PR #47629
- GLM-5.3 NoPE 512 sparse MLA: vLLM PR #54031
- PP pinned-buffer race: vLLM PR #47644
- Ampere GLM-5.3 backport: https://github.com/wtdcode/vllm-backport
- A800 sparse MLA reference: https://github.com/Mrzhiyao/glm53-a800-vllm
- 초기 A800 비교 자료: https://gitee.com/kill-life/glm5.3-flash-deployment-a800

---

# 라이선스

vendor/upstream 파일에는 각 원본 프로젝트의 라이선스가 적용됩니다. vLLM 코드는 Apache-2.0입니다.
