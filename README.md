# GLM-5.3-Flash on SM80 / CUDA 12.9

[English README](README.en.md)

`zai-org/GLM-5.3-Flash`를 NVIDIA A100/A800(SM80)에서 CUDA 12.9 기반으로 실행하기 위한 vLLM + Singularity/Apptainer 패치 번들입니다.

기준 이미지는 공식 GLM-5.3 전용 이미지인 다음 태그입니다.

```text
vllm/vllm-openai:glm53-flash-cu129
```

> 이 저장소는 공식 vLLM의 GLM-5.3 SM80 지원이 아닙니다. 아직 merge되지 않은 upstream PR과 실제 Ampere backport에서 확인된 GLM-5.3 correctness 수정 중 필요한 부분을 선별하여 `glm53-flash-cu129`에 backport하는 실험적 빌드입니다.

## 목표

- GPU: NVIDIA A100/A800 80GB, SM80
- 모델: `zai-org/GLM-5.3-Flash` 원본 FP8 checkpoint
- CUDA runtime: 12.9
- 단일 노드: TP=8 + Expert Parallel
- 멀티 노드 확장: TP=8 × PP=2 등을 고려할 수 있도록 #47644 포함
- Sparse MLA: `TRITON_MLA_SPARSE`
- Attention KV cache: BF16
- A100 FP8 weight 실행: Marlin fallback
- Claude Code tool calling / WebFetch 같은 긴 tool-result workload의 correctness 검증을 주요 목표로 함

## 패치 구성

### 1. PR #54031: GLM-5.3 NoPE 512 Sparse MLA

GLM-5.3의 attention geometry는 다음과 같습니다.

```text
dim_qk = 512
qk_rope_head_dim = 0
```

A800 프로젝트의 generic XPU sparse path 대신 #54031의 구조를 채택합니다.

- `TritonMLASparseImpl(FlashMLASparseImpl)` 기반
- `dim_qk=512` NoPE 직접 지원
- `BLOCK_DPE=0`이면 RoPE 계산을 compile-time 제거
- 512/576 geometry를 동일 Triton kernel에서 처리
- split-KV decode
- empty-row NaN guard

단, #54031은 최신 main API 기준이므로 파일을 그대로 덮지 않고 `glm53-flash-cu129` API에 맞도록 `patch_runtime.py`에서 return interface를 조정합니다.

### 2. PR #47629: SM80 MQA indexer fallback

최신 고정 commit:

```text
064801dd2bc6ac2e265dc3fa1f5d803d71bde25d
```

주요 correctness 수정:

- SM80용 FP8 MQA / paged-MQA Triton fallback
- large unified-KV stride에서 `block_idx`를 int64로 승격하여 address overflow 방지
- paged decode 마지막 block OOB store 방지
- chunked-prefill `clean_logits=False` 경로에서 stale logits 방지

#47629은 A800에서 long-context/tool-call/needle retrieval까지 시험된 SM80 fallback입니다.

### 3. GLM-5.3 KPool SM80 routing

GLM-5.3은 generic `sparse_attn_indexer.py`만 쓰지 않고 실제로 `sparse_attn_indexer_kpool.py`를 사용합니다.

따라서 KPool indexer에도 동일한 SM80 Triton MQA fallback을 이식합니다.

MTP에서 `seq_lens`가 `(B, next_n)`일 때 paged Triton kernel에는 1-D context length가 필요하므로 다음 형태로 변환합니다.

```python
triton_seq_lens = (
    seq_lens[:, -1].contiguous()
    if seq_lens.ndim == 2
    else seq_lens
)
```

원본 2-D `seq_lens`는 이후 KPool top-k 처리용으로 그대로 유지합니다.

### 4. Ampere KPool FP8 writer 및 KPoolTail correctness 보강

실제 빌드 경로에서는 더 이상 Gitee/A800 프로젝트의 `kpool_compress.py`에 의존하지 않습니다. A800 코드는 비교 자료로만 유지합니다.

대신 `wtdcode/vllm-backport`의 Ampere 검증 구현에서 다음을 고정 SHA로 가져옵니다.

```text
BACKPORT_SHA = 0ef4bff219c098d48cf16d3d63ebef329e9b74b0
```

주요 내용:

- SM80에서 Triton `fp8e4nv` native cast를 쓰지 않고 E4M3FN byte를 software encode
- KPool prefill/decode FP8 cache write를 SM80에서 안전하게 처리
- hybrid model runner에서 KPoolTail metadata에 token positions 전달
- padded KPoolTail slot은 `-1`로 초기화
- CUDA graph replay를 위해 KPoolTail slot mapping을 persistent buffer로 유지
- MTP warmup의 padded `seq_lens`를 active request 수에 맞춰 slice
- KPoolTail group을 generic position-based slot mapping에서 제외
- MTP draft attention metadata에도 positions 전달

이 수정들은 GLM-5.3을 Ampere에서 장문/MTP/CUDA graph로 구동하면서 실제로 발견된 OOB·stale pointer·shape mismatch 문제를 대상으로 합니다.

### 5. PR #47644: PP pinned-buffer race fix

PP batch queue에서 CPU pinned input buffer가 다음 step에 의해 너무 일찍 재사용되는 race를 방지합니다.

- TP-only에서는 사실상 영향 없음
- 향후 TP8 × PP2 같은 2노드 구성을 위해 포함

## 빌드

### 1. 인터넷 가능한 머신에서 vendor source 준비

```bash
bash ./prepare_vendor.sh
```

고정되는 upstream:

```text
#47629  064801dd2bc6ac2e265dc3fa1f5d803d71bde25d
#54031   b325d908656d05e2a650ec60666ccec6f4f3eb0c
backport 0ef4bff219c098d48cf16d3d63ebef329e9b74b0
```

이후 디렉터리 전체를 폐쇄망으로 복사할 수 있습니다.

### 2. SIF 생성

```bash
bash ./build_sif.sh /path/to/glm53-flash-sm80-cu129.sif
```

fakeroot가 필요하면:

```bash
BUILD_FLAGS='--fakeroot' bash ./build_sif.sh /path/to/glm53-flash-sm80-cu129.sif
```

### 3. A100에서 kernel 검증

모델을 로딩하기 전에 먼저 실행하십시오.

```bash
bash ./verify_sif_gpu.sh /path/to/glm53-flash-sm80-cu129.sif
```

검증 항목:

1. 패치된 Python module 전체 compile
2. #54031 NoPE 512 sparse MLA vs PyTorch reference
3. #47629 SM80 FP8 MQA vs PyTorch reference
4. int64 block addressing / MTP context-lens / KPoolTail persistent mapping / padded-slot sentinel 검증
5. hybrid/MTP positions propagation 및 KPoolTail generic slot-map exclusion
6. #47644 존재 여부

GPU kernel test가 실패하면 full model serving으로 넘어가지 않는 것을 권장합니다.

## 오프라인/소스 백업

참조 PR이나 외부 저장소가 나중에 사라져도 재구성할 수 있도록 `source-backup` 브랜치에 pinned source snapshot과 patch 자료를 별도로 보관합니다.

```bash
git fetch origin source-backup
git worktree add ../glm53-source-backup origin/source-backup

# source-backup에서 vendor를 재조립한 후
bash ../glm53-source-backup/backup/reassemble_vendor.sh
BACKUP_VENDOR=../glm53-source-backup/backup/vendor bash ./prepare_vendor.sh
```

`source-backup`은 실행 브랜치가 아니라 재현용 보관 브랜치입니다. 실제 빌드 스크립트의 기준은 `main`입니다.

# vLLM 실행 설정

두 가지 프로파일을 제공합니다.

```text
serve_tp8.sh
  ├─ PROFILE=initial → serve_tp8_initial.sh
  └─ PROFILE=ideal   → serve_tp8_ideal.sh
```

## A. 초기 검증 설정 — 권장 시작점

정확성 문제를 찾기 위한 설정입니다.

| 항목 | 값 |
|---|---:|
| GPU | A100/A800 80GB × 8 |
| TP | 8 |
| Expert Parallel | ON |
| GPU memory utilization | **0.85** |
| max model len | **131072 (128K)** |
| max num seqs | **8** |
| max num batched tokens | **8192** |
| KV cache | **BF16** |
| MTP | **OFF** |
| Prefix caching | **OFF** |
| Sparse MLA | `TRITON_MLA_SPARSE` |
| `sparse_mla_force_mqa` | **true** |
| CUDA Graph | 허용 |

실행:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=initial bash ./serve_tp8.sh
```

또는 직접:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
bash ./serve_tp8_initial.sh
```

이 프로파일의 목적은 성능이 아니라 다음 문제를 분리하는 것입니다.

- SM80 Sparse MLA kernel 자체 correctness
- `max_num_batched_tokens=8192`를 넘는 chunked prefill
- 긴 tool result
- Claude Code tool calling / WebFetch
- CUDA graph와 무관한 silent wrong output 여부

## B. 이상적 운영 설정 — 초기 검증 통과 후

단일 노드 A100 80GB ×8에서 현재 목표로 잡는 운영 프로파일입니다.

| 항목 | 값 |
|---|---:|
| GPU | A100/A800 80GB × 8 |
| TP | 8 |
| Expert Parallel | ON |
| GPU memory utilization | **0.90** |
| max model len | **524288 (512K)** |
| max num seqs | **8** |
| max num batched tokens | **8192** |
| KV cache | **BF16** |
| MTP | **3 draft tokens** (기본) |
| Prefix caching | **ON** |
| Sparse MLA | `TRITON_MLA_SPARSE` |
| `sparse_mla_force_mqa` | **true** |
| NCCL | `Ring` / `Simple` |
| custom all-reduce | **OFF** |
| CUDA Graph | 허용 |

실행:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=ideal bash ./serve_tp8.sh
```

이 설정은 **목표 운영 설정**이며 새 패치 SIF에서 아직 검증됐다고 가정하지 않습니다. 반드시 initial 프로파일에서 WebFetch/tool-call까지 통과한 뒤 사용하십시오.

### 왜 8192를 유지하는가

`max_num_batched_tokens=8192`는 4096보다 긴 prompt의 chunk 수를 줄여 SM80 MLA/indexer의 chunked-prefill correctness 문제를 검증하고 완화하기 위한 값입니다.

다만 8192도 prompt가 8192 tokens를 넘으면 chunked prefill 자체를 없애지는 않습니다. 16K 이상 prompt를 단일 prefill로 대조하고 싶다면 진단용으로 16384 이상을 별도 시험해야 합니다.

### 왜 512K인가

GLM-5.3은 1M context를 지원하지만 A100에서 Claude Code 운영과 correctness를 우선하면 512K가 메모리 headroom과 실제 활용 범위 사이의 현실적인 기본값입니다.

1M 검증은 512K가 안정화된 후 별도로 올리는 것을 권장합니다.

## 2노드 확장

권장 topology는 일반적으로 노드 간 TP16보다 다음 구조입니다.

```text
Node 0: TP8
Node 1: TP8
전체: TP=8, PP=2
```

#47644는 이 PP 경로를 위해 포함되어 있습니다.

다만 MTP + PP에는 vLLM core scheduler의 추가 race가 보고된 적이 있으므로 최초 2노드 검증은 다음 순서를 권장합니다.

```text
TP8×PP2, MTP OFF
→ long prefill / tool-call 검증
→ 안정화 후 MTP ON
```

이 저장소는 아직 특정 Ray/Slurm launcher에 종속된 2노드 실행 스크립트는 제공하지 않습니다.

## Correctness 검증 순서

1. 짧은 deterministic prompt
2. 32K needle retrieval
3. 64K needle retrieval
4. 128K needle retrieval
5. 8192보다 긴 prompt로 chunked prefill
6. Claude Code tool call
7. Claude Code WebFetch
8. single-node MTP3 (필요 시 `NUM_SPEC_TOKENS=5`를 별도 A/B)
9. prefix caching
10. 256K → 512K → 필요 시 1M
11. TP8 × PP2

간단한 API 확인:

```bash
bash ./serve_test.sh
```

## 주의사항

- `--enforce-eager`에서도 동일한 wrong-output이 발생한다면 CUDA Graph를 주원인으로 보기 어렵습니다.
- `gpu-memory-utilization`, `max-model-len`, `max-num-seqs`는 주로 메모리/동시성 관련 옵션이며 tool-result 의미가 깨지는 현상을 직접 설명하지는 못합니다.
- MTP와 prefix caching은 초기 correctness 검증에서는 끄는 것이 좋습니다.
- 이 patch set은 open/unmerged upstream work를 포함합니다. upstream vLLM이 GLM-5.3 + SM80을 공식 지원하게 되면 이 저장소보다 공식 구현을 우선하십시오.

## 출처

- vLLM: https://github.com/vllm-project/vllm
- SM80 Triton sparse MLA: vLLM PR #47629
- GLM-5.3 NoPE 512: vLLM PR #54031
- PP pinned-buffer race: vLLM PR #47644
- Ampere GLM-5.3 backport: https://github.com/wtdcode/vllm-backport
- A800 historical comparison: https://gitee.com/kill-life/glm5.3-flash-deployment-a800

## 라이선스

이 저장소의 스크립트는 실험/배포 자동화를 위한 것이며, vendor 단계에서 가져오는 vLLM 코드에는 upstream의 Apache-2.0 라이선스가 적용됩니다. 각 upstream 프로젝트의 라이선스를 확인하십시오.
