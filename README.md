# GLM-5.3-Flash on SM80 / CUDA 12.9

[English README](README.en.md)

`zai-org/GLM-5.3-Flash`를 NVIDIA A100/A800(SM80)에서 CUDA 12.9 기반으로 실행하기 위한 vLLM + Singularity/Apptainer 패치 번들입니다.

기준 이미지는 다음입니다.

```text
vllm/vllm-openai:glm53-flash-cu129
```

> 공식 vLLM의 GLM-5.3 SM80 지원이 아니라, 아직 merge되지 않은 upstream PR과 Ampere에서 발견된 correctness fix를 `glm53-flash-cu129`에 선별 backport하는 실험적 빌드입니다.

## 현재 패치 구성

### 1. PR #54031 — GLM-5.3 NoPE 512 Sparse MLA

GLM-5.3의 `dim_qk=512`, `qk_rope_head_dim=0`을 native `TRITON_MLA_SPARSE` kernel에서 직접 처리합니다.

- `TritonMLASparseImpl(FlashMLASparseImpl)` 기반
- 512/576 geometry 통합
- `BLOCK_DPE=0`에서 RoPE 계산 compile-time 제거
- split-KV decode
- empty-row NaN guard

#54031은 newer-main API이므로 `patch_runtime.py`에서 `glm53-flash-cu129` API에 맞게 적응합니다.

### 2. PR #47629 — SM80 MQA indexer fallback

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

### 3. GLM-5.3 KPool / KPoolTail Ampere correctness fix

실제 빌드 경로는 더 이상 Gitee A800 프로젝트의 KPool writer에 의존하지 않습니다. 다음 Ampere backport를 고정하여 사용합니다.

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

### 4. PR #47644 — PP pinned-buffer race

`glm53-flash-cu129@487ecf187`은 upstream PR 당시 코드와 달리 이미 `torch.cuda.Event(blocking=True)`를 사용합니다.

따라서 `patch_47644_compat.py`에서 **blocking semantics는 유지하고** event 생성 조건만 다음으로 확대합니다.

```python
if self.vllm_config.max_concurrent_batches > 1:
    self.prepare_inputs_event = torch.cuda.Event(blocking=True)
```

TP-only에는 사실상 영향이 없고, TP8×PP2 같은 PP 구성을 위한 fix입니다.

---

# 소스 감사 상태

`main`에는 GitHub Actions source audit가 포함되어 있습니다.

```text
.github/workflows/audit-patch-chain.yml
```

이 workflow는 외부 upstream을 다시 받지 않고 `source-backup`의 고정 snapshot만 사용해:

1. patch unit tests 실행
2. `vLLM@487ecf187` source subset 재구성
3. `patch_47644_compat.py` 적용
4. `patch_runtime.py` 전체 적용
5. `patch_glm53_tail.py` 전체 적용
6. patched Python 전체 compile/static verification
7. backup SHA256 검증

을 수행합니다.

현재 전체 source audit는 **PASS**입니다.

상세 기록은 [`AUDIT.md`](AUDIT.md)를 참고하십시오.

> 단, 이 PASS는 source/patch compatibility 검증입니다. 실제 A100 GPU에서 Triton kernel과 full GLM-5.3 serving correctness를 검증했다는 뜻은 아닙니다. 최종 확인은 아래 `verify_sif_gpu.sh`와 실제 tool/WebFetch workload로 해야 합니다.

---

# 오프라인 소스 백업

`source-backup` 브랜치에 실제 참조 소스를 저장해 두었습니다.

```text
backup/vendor/             실제 빌드에 필요한 pinned vendor source
backup/base487/            glm53-flash-cu129 기준 vLLM source
backup/backport-reference/ Ampere GLM-5.3 reference source
backup/a800-reference/     Gitee A800 프로젝트 전체 snapshot (비교용)
backup/SHA256SUMS.txt      snapshot checksum
```

참조 저장소가 사라져도 다음처럼 완전 오프라인으로 vendor를 준비할 수 있습니다.

```bash
git fetch origin source-backup
git worktree add ../glm53-source-backup origin/source-backup

BACKUP_VENDOR=../glm53-source-backup/backup/vendor \
  bash ./prepare_vendor.sh
```

`prepare_vendor.sh`는 `BACKUP_VENDOR`에 파일이 있으면 네트워크를 사용하지 않습니다.

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

## 3. A100 GPU kernel 검증

```bash
bash ./verify_sif_gpu.sh /path/to/glm53-flash-sm80-cu129.sif
```

이 단계가 실패하면 full serving으로 넘어가지 않는 것을 권장합니다.

---

# vLLM 실행 설정

두 프로파일을 제공합니다.

```text
PROFILE=initial → serve_tp8_initial.sh
PROFILE=ideal   → serve_tp8_ideal.sh
```

## A. 초기 검증 설정

정확성 우선입니다.

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

검증 목표:

- 짧은 deterministic output
- 32K / 64K / 128K needle retrieval
- 8192 tokens 초과 chunked prefill
- 긴 tool result
- Claude Code tool calling
- Claude Code WebFetch

## B. 이상적 운영 설정

초기 설정이 모두 통과한 뒤 사용하는 목표 프로파일입니다.

| 항목 | 값 |
|---|---:|
| GPU | A100/A800 80GB ×8 |
| TP | 8 |
| Expert Parallel | ON |
| GPU memory utilization | **0.90** |
| max model len | **524288 (512K)** |
| max num seqs | **8** |
| max num batched tokens | **8192** |
| KV cache | **BF16** |
| MTP | **3 draft tokens** 기본 |
| Prefix caching | **ON** |
| Sparse MLA | `TRITON_MLA_SPARSE` |
| `sparse_mla_force_mqa` | **true** |
| NCCL | Ring / Simple |
| custom all-reduce | OFF |

```bash
MODEL_HOST_PATH=/path/to/GLM-5.3-Flash \
SIF_PATH=/path/to/glm53-flash-sm80-cu129.sif \
PROFILE=ideal bash ./serve_tp8.sh
```

MTP5를 시험하려면 초기 안정화 후:

```bash
NUM_SPEC_TOKENS=5 PROFILE=ideal bash ./serve_tp8.sh
```

처럼 별도 A/B 검증하십시오.

512K가 안정된 뒤 1M으로 올리는 것을 권장합니다.

---

# 2노드 확장

일반적으로 다음 topology를 권장합니다.

```text
Node 0: TP8
Node 1: TP8
전체: TP=8, PP=2
```

노드 간 TP16보다 inter-node tensor-parallel collective 부담이 작습니다.

PP 최초 검증은:

```text
TP8×PP2 + MTP OFF
→ long prefill / tool call
→ 안정화 후 MTP ON
```

순서를 권장합니다.

---

# 최종 correctness 검증 순서

1. `verify_sif_gpu.sh`
2. 짧은 deterministic prompt
3. 32K / 64K / 128K needle
4. chunked prefill
5. Claude Code tool call
6. Claude Code WebFetch
7. MTP3
8. prefix caching
9. 256K → 512K → 필요 시 1M
10. TP8×PP2

---

## 출처

- vLLM: https://github.com/vllm-project/vllm
- SM80 Triton Sparse MLA: vLLM PR #47629
- GLM-5.3 NoPE 512: vLLM PR #54031
- PP pinned-buffer race: vLLM PR #47644
- Ampere GLM-5.3 backport: https://github.com/wtdcode/vllm-backport
- A800 historical comparison: https://gitee.com/kill-life/glm5.3-flash-deployment-a800

## 라이선스

vendor/upstream source에는 각 upstream 프로젝트의 라이선스가 적용됩니다. vLLM 코드는 Apache-2.0입니다.
