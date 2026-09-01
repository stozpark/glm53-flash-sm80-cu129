#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


TRITON = "AttentionBackendEnum.TRITON_MLA_SPARSE"
FI_SM90 = "AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    args = ap.parse_args()

    path = Path(args.vllm_root).resolve() / "platforms/cuda.py"
    text = path.read_text(encoding="utf-8")

    # glm53-flash-cu129 uses sparse_tail.pop() in the GLM NoPE branch to
    # remove FLASHINFER_MLA_SPARSE_SM90 before re-inserting it at the head.
    # Therefore TRITON_MLA_SPARSE must appear BEFORE FI_SM90 in sparse_tail.
    pat = re.compile(
        r"(?P<head>^[ \t]*sparse_tail = \[\n)"
        r"(?P<body>(?:^[ \t]+AttentionBackendEnum\.[A-Z0-9_]+,\n)+)"
        r"(?P<tail>^[ \t]*\]\n)",
        re.MULTILINE,
    )
    matches = list(pat.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one sparse_tail block, found {len(matches)}")

    m = matches[0]
    lines = m.group("body").splitlines(keepends=True)
    names = [line.strip().rstrip(",") for line in lines]
    if FI_SM90 not in names:
        raise RuntimeError("FLASHINFER_MLA_SPARSE_SM90 missing from sparse_tail")

    # Remove any existing Triton entry, then insert it immediately before the
    # SM90 entry. This fixes both the original base and the previously patched
    # buggy order [.., FI_SM90, TRITON].
    lines = [line for line in lines if TRITON not in line]
    fi_idx = next(i for i, line in enumerate(lines) if FI_SM90 in line)
    indent = re.match(r"^[ \t]*", lines[fi_idx]).group(0)
    lines.insert(fi_idx, f"{indent}{TRITON},\n")

    new_block = m.group("head") + "".join(lines) + m.group("tail")
    text = text[: m.start()] + new_block + text[m.end() :]

    # Control-flow verification: GLM-5.3's prefer_fi_sm90 branch executes
    # sparse_tail.pop(). After that operation Triton must still be present and
    # FI_SM90 must be the removed element.
    names = [line.strip().rstrip(",") for line in lines]
    simulated = names.copy()
    popped = simulated.pop()
    if popped != FI_SM90:
        raise RuntimeError(f"GLM sparse_tail.pop() would remove {popped}, expected {FI_SM90}")
    if TRITON not in simulated:
        raise RuntimeError("TRITON_MLA_SPARSE disappears from GLM selector after pop()")

    path.write_text(text, encoding="utf-8")
    print(f"[cuda-sparse-priority] patched and control-flow verified: {path}")


if __name__ == "__main__":
    main()
