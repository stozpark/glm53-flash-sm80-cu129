# GLM-5.3 SM80 patch audit

최종 source-level audit 기준: 2026-09-01

## 감사 기준

```text
Base vLLM:       487ecf187
PR #47629:       064801dd2bc6ac2e265dc3fa1f5d803d71bde25d
PR #54031:       b325d908656d05e2a650ec60666ccec6f4f3eb0c
Ampere backport: 0ef4bff219c098d48cf16d3d63ebef329e9b74b0
```

## 발견하여 수정한 누락

초기 저장소 작성 후 전체 재감사 과정에서 다음 누락을 발견했고 `main`에 반영했습니다.

1. GLM-5.3 KPoolTail hybrid metadata에 token positions가 전달되지 않는 문제
2. CUDA graph padded KPoolTail slot이 stale main-cache slot을 유지할 수 있는 문제
3. KPoolTail mapping tensor가 persistent buffer가 아니어서 graph replay pointer 안정성이 깨질 수 있는 문제
4. MTP warmup에서 padded `seq_lens` source를 active decode request 수만큼 slice하지 않는 문제
5. KPoolTail cache group에 generic position-based slot mapping이 적용되는 문제
6. MTP draft metadata에 positions가 전달되지 않는 문제
7. Gitee/A800 KPool writer를 actual build dependency로 두고 있던 문제
8. #47644 patch가 `glm53-flash-cu129@487ecf187`의 blocking-event 코드 형태와 맞지 않는 문제

현재 실제 build path는 Gitee/A800 구현을 사용하지 않습니다. A800 snapshot은 비교용으로만 `source-backup`에 보관합니다.

## source audit 결과

GitHub Actions:

```text
Audit full patch chain against pinned base
run: 33421750542
result: SUCCESS
```

검증한 항목:

- `test_patch_runtime_unit.py`: PASS
- `test_patch_glm53_tail_unit.py`: PASS
- self-contained `source-backup` checkout: PASS
- exact `487ecf187` touched-source subset reconstruction: PASS
- `patch_47644_compat.py`: PASS
- `patch_runtime.py`: PASS
- `patch_glm53_tail.py`: PASS
- patched Python `py_compile`: PASS
- `verify_static.py`: PASS
- backup SHA256 verification: PASS

즉 **현재 patch chain은 고정한 base source에 실제로 적용 가능하고 syntax/static invariant가 모두 통과함을 확인했습니다.**

## source backup 결과

`source-backup` branch snapshot workflow:

```text
Snapshot pinned upstream sources
run: 33421255994
result: SUCCESS
```

보관:

- pinned build vendor source
- base `487ecf187` source files
- Ampere backport reference files
- Gitee A800 repository snapshot
- SHA256 checksums

## 아직 검증되지 않은 것

이 환경에는 A100과 실제 `glm53-flash-cu129` SIF runtime이 없으므로 아래는 아직 PASS라고 주장하지 않습니다.

- 실제 Singularity/Apptainer SIF build
- A100 SM80에서 Triton compilation
- #54031 NoPE-512 GPU numerical reference test
- SM80 FP8 MQA GPU numerical reference test
- 원본 GLM-5.3 FP8 checkpoint TP8 loading
- 128K/512K long-context correctness
- Claude Code tool-call / WebFetch correctness
- MTP3 acceptance/correctness
- prefix-cache correctness
- TP8×PP2 runtime correctness

따라서 release gate는 다음입니다.

```text
SOURCE_AUDIT = PASS
GPU/SIF_AUDIT = PENDING
FULL_SERVING_AUDIT = PENDING
```

A100 머신에서는 먼저:

```bash
bash ./verify_sif_gpu.sh /path/to/glm53-flash-sm80-cu129.sif
```

을 통과한 뒤 `PROFILE=initial`로 full serving 검증을 진행해야 합니다.
