# Source backup

이 디렉터리는 `main`의 GLM-5.3 SM80 빌드를 외부 참조 저장소 없이 재구성하기 위한 소스 스냅샷입니다.

## 고정 revision

```text
vLLM glm53-flash-cu129 source base: 487ecf187
PR #47629: 064801dd2bc6ac2e265dc3fa1f5d803d71bde25d
PR #54031: b325d908656d05e2a650ec60666ccec6f4f3eb0c
Ampere backport: 0ef4bff219c098d48cf16d3d63ebef329e9b74b0
```

## 디렉터리

- `vendor/`: 실제 `main/prepare_vendor.sh`가 필요로 하는 pinned vendor source
- `base487/`: `glm53-flash-cu129`의 기준 vLLM source 중 이 프로젝트가 수정하는 파일
- `backport-reference/`: Ampere GLM-5.3 reference implementation
- `a800-reference/`: Gitee A800 프로젝트 전체 snapshot; 실제 빌드에는 사용하지 않고 비교용으로 보존
- `SHA256SUMS.txt`: 자동 snapshot 시점의 checksum

## 완전 오프라인 vendor 준비

`source-backup`을 별도 worktree 또는 directory에 checkout한 뒤 `main`에서 다음처럼 사용합니다.

```bash
BACKUP_VENDOR=/path/to/source-backup/backup/vendor bash ./prepare_vendor.sh
```

`prepare_vendor.sh`는 `BACKUP_VENDOR`에 필요한 파일이 있으면 네트워크에 접근하지 않고 그것을 복사합니다.

## 주의

`source-backup`은 실행/개발 기준 브랜치가 아닙니다. 실제 빌드 스크립트와 패치 로직의 기준은 `main`입니다. 이 브랜치는 외부 원본 삭제, force-push, 서비스 장애 등에 대비한 재현용 보관본입니다.
