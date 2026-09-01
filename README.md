# GLM-5.3-Flash on SM80 / CUDA 12.9

[English README](README.en.md)

`zai-org/GLM-5.3-Flash`를 NVIDIA A100/A800(SM80)에서 CUDA 12.9 기반으로 실행하기 위한 vLLM + Singularity/Apptainer 패치 번들입니다.

기준 이미지는 다음입니다.

```text
vllm/vllm-openai:glm53-flash-cu129
```

> 공식 vLLM의 GLM-5.3 SM80 지원이 아니라, 아직 merge되지 않은 upstream PR과 Ampere에서 발견된 correctness fix를 `glm53-flash-cu129`에 선별 backport한 빌드입니다.

---

# 현재 검증 상태

## ✅ A100 80GB ×8 실기기 검증 완료

**2026-09-01 기준**, NVIDIA **A100-SXM4-80GB ×8** 단일 노드에서 `serve_tp8_ideal.sh` 프로파일로 GLM-5.3-Flash serving이 정상 동작하는 것을 확인했습니다.

검증에 사용한 운영 프로파일:

| 항목 | 값 |
|---|---:|
| GPU | **A100-SXM4-80GB ×8** |
| TP | **8** |
| Expert Parallel | **ON** |
| GPU memory utilization | **0.90** |
| max model len | **524288 (512K configured)** |
| max num seqs | **8** |
| max num batched tokens | **8192** |
| KV cache | **BF16** |
| MTP | **3 draft tokens** |
| Prefix caching | **ON** |
| Sparse MLA | **TRITON_MLA_SPARSE** |
| `sparse_mla_force_mqa` | **true** |
| MoE backend | **Marlin** |
| NCCL | **Ring / Simple** |
| custom all-reduce | **OFF** |

즉, 현재 권장 단일 노드 실행 경로는 다음입니다.

```text
A100 80GB ×8
TP=8
EP=ON
PP=1
serve_tp8_ideal.sh
```

> `max_model_len=524288`로 서버가 정상 기동하고 일반 serving이 동작하는 것은 확인했습니다. 다만 이것이 512K 길이의 실제 full-context 요청을 끝까지 스트레스 테스트했다는 의미는 아닙니다. 256K/512K/1M 장문 입력은 필요 시 별도 검증하십시오.

> TP8×PP2 멀티노드는 아직 실기기 검증 완료 상태로 보지 않습니다.

---

# 현재 패치 구성

## 1. PR #54031 — GLM-5.3 NoPE 512 Sparse MLA

GLM-5.3의 `dim_qk=512`, `qk_rope_head_dim=0`을 native `TRITON_MLA_SPARSE` kernel에서 직접 처리합니다.

- `TritonMLASparseImpl(FlashMLASparseImpl)` 기반
- 512/576 geometry 통합
- `BLOCK_DPE=0`에서 RoPE 계산 compile-time 제거
- split-KV decode
- empty-row NaN guard
- long-context KV row-offset int64 처리

#54031은 newer-main API이므로 `patch_runtime.py`에서 `glm53-flash-cu129` API에 맞게 적응합니다.

## 2. PR #47629 — SM80 MQA indexer fallback

고정 commit:

```text
064801dd2bc6ac2e265dc3fa1f5d803d71bde25d
```

포함하는 주요 fix:

- SM80 FP8 MQA / paged-MQA Triton fallback
- unified-KV large-stride `block_idx` int64 처리
- paged decode 마지막 block OOB store 방지
- chunked-prefill dirty logits 방지
- GLM-5.3 KPool indexer에도 동일 fallback 적용
- MTP `(B,next_n)` `seq_lens`를 Triton paged MQA용 `[B]`로 변환
- KPool `persistent_topk()`에 실제 compressed-logit width 전달

## 3. GLM-5.3 KPool / KPoolTail Ampere correctness fix

다음 Ampere backport를 고정하여 사용합니다.

```text
wtdcode/vllm-backport
0ef4bff219c098d48cf16d3d63ebef329e9b74b0
```

포함한 보강:

- SM80 software E4M3FN byte encoder
- KPool prefill/decode FP8 cache write fallback
- Hybrid runner의 KPoolTail positions 전달
- padded KPoolTail slot을 `-1` sentinel로 초기화
- CUDA graph replay용 persistent KPoolTail slot-mapping buffer
- MTP padded `seq_lens[:num_decodes]` 처리
- KPoolTail group을 generic slot-mapping kernel에서 제외
- MTP draft metadata에도 positions 전달
- MTP draft-buffer stream synchronization fence

## 4. Sparse MLA backend selector fix

Ampere에서 `TRITON_MLA_SPARSE`가 sparse backend 후보에 등록되어도 기존 GLM 우선순위 로직의 `pop()`에 의해 제거될 수 있는 문제를 수정합니다.

현재 patch는 `TRITON_MLA_SPARSE`를 SM90 전용 후보 앞에 배치하고, CI에서 실제 `pop()` 이후에도 후보가 남는 것을 control-flow 수준으로 검증합니다.

## 5. DeepGEMM fallback gate fix

A100에서는 DeepGEMM이 없는 것이 정상입니다.

GLM-5.3 KPool constructor의 DeepGEMM hard gate를 제거하고 다음 경로로 진행하도록 합니다.

```text
DeepGEMM unavailable
→ SM80 Triton sparse-indexer fallback
```

## 6. PR #47644 — PP pinned-buffer race

`glm53-flash-cu129@487ecf187`의 기존 `blocking=True` semantics를 유지하면서 concurrent batch에서 필요한 event 생성 조건을 보강합니다.

이 patch는 포함되어 있지만 **TP8×PP2 자체는 아직 이 저장소에서 실기기 검증 완료 상태가 아닙니다.**

---

# 소스 감사 상태

`main`에는 GitHub Actions source audit가 포함되어 있습니다.

```text
.github/workflows/audit-patch-chain.yml
```

이 workflow는 `source-backup`의 pinned snapshot으로 exact base를 재구성한 뒤 전체 patch chain을 다시 적용합니다.

검증 대상에는 다음이 포함됩니다.

1. patch unit tests
2. `vLLM@487ecf187` source subset 재구성
3. PP compatibility patch
4. DeepGEMM gate 제거
5. runtime / Sparse MLA patch
6. Sparse backend selector control-flow 검증
7. Mrzhiyao A800 sparse kernel merge
8. GLM KPool/KPoolTail correctness fix
9. long-context int64 offset fix
10. MTP stream fence
11. patched Python compile/static verification
12. backup SHA256 verification

현재 전체 source audit는 **PASS**입니다.

상세 기록은 [`AUDIT.md`](AUDIT.md)를 참고하십시오.

---

# 오프라인 소스 백업

`source-backup` 브랜치에 참조 소스를 저장해 두었습니다.

```text
backup/vendor/             실제 빌드에 필요한 pinned vendor source
backup/base487/            glm53-flash-cu129 기준 vLLM source
backup/backport-reference/ Ampere GLM-5.3 reference source
backup/mrzhiyao-reference/ A800 native-FP8 reference source
backup/a800-reference/     Gitee A800 프로젝트 snapshot (비교용)
backup/SHA256SUMS.txt      snapshot checksum
```

오프라인 vendor 준비:

```bash
git fetch origin source-backup
git worktree add ../glm53-source-backup origin/source-backup

BACKUP_VENDOR=../glm53-source-backup/backup/vendor \
  bash ./prepare_vendor.sh
```

`source-backup`은 보관/재현용이며 실제 개발·빌드 기준은 `main`입니다.

---

# 빌드

## 1. vendor 준비

온라인:

```bash
bash ./prepare_vendor.sh
```

오프라인:

```bash
BACKUP_VENDOR=/path/to/source-backup/backup/vendor \
  bash ./prepare_vendor.sh
```

고정 revision:

```text
base      487ecf187
#47629    064801dd2bc6ac2e265dc3fa1f5d803d71bde25d
#54031    b325d908656d05e2a650ec60666ccec6f4f3eb0c
backport  0ef4bff219c098d48cf16d3d63ebef329e9b74b0
```

## 2. SIF 생성

```bash
bash ./build_sif.sh /path/to/glm53-flash-sm80-cu129.sif
```

fakeroot가 필요하면:

```bash
BUILD_FLAGS='--fakeroot' \
  bash ./build_sif.sh /path/to/glm53-flash-sm80-cu129.sif
```

빌드 중 다음이 통과해야 합니다.

```text
STATIC_VERIFY=PASS
```

## 3. A100 GPU kernel 검증

```bash
GPU=0 bash ./verify_sif_gpu.sh /path/to/glm53-flash-sm80-cu129.sif
```

정상 완료:

```text
SIF_GPU_VERIFY=PASS
```

---

# vLLM 실행

두 프로파일을 제공합니다.

```text
PROFILE=initial → serve_tp8_initial.sh
PROFILE=ideal   → serve_tp8_ideal.sh
```

## A. 초기 진단 프로파일

문제 분리를 위한 보수적 설정입니다.

| 항목 | 값 |
|---|---:|
| GPU | A100/A800 80GB ×8 |
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

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=initial bash ./serve_tp8.sh
```

## B. 권장 운영 프로파일 — A100 ×8 검증 완료

현재 A100 80GB ×8 단일 노드에서 정상 serving을 확인한 프로파일입니다.

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=ideal bash ./serve_tp8.sh
```

또는 직접:

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
./serve_tp8_ideal.sh
```

기본값:

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

MTP5는 별도 A/B 검증 후 사용하십시오.

```bash
NUM_SPEC_TOKENS=5 PROFILE=ideal bash ./serve_tp8.sh
```

---

# 추가 검증이 필요한 범위

현재 A100×8 ideal serving은 확인되었지만 다음은 별도 stress test 대상으로 남겨 둡니다.

- 실제 256K / 512K full-context 요청
- 1M context
- MTP5 이상
- TP8×PP2 멀티노드
- 장시간/고동시성 soak test

---

## 출처

- vLLM: https://github.com/vllm-project/vllm
- SM80 Triton MQA fallback: vLLM PR #47629
- GLM-5.3 NoPE 512 Sparse MLA: vLLM PR #54031
- PP pinned-buffer race: vLLM PR #47644
- Ampere GLM-5.3 backport: https://github.com/wtdcode/vllm-backport
- A800 native-FP8 reference: https://github.com/Mrzhiyao/glm53-a800-vllm
- A800 historical comparison: https://gitee.com/kill-life/glm5.3-flash-deployment-a800

## 라이선스

vendor/upstream source에는 각 upstream 프로젝트의 라이선스가 적용됩니다. vLLM 코드는 Apache-2.0입니다.
