#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


ERROR_TEXT = "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
WARNING_TEXT = "DeepGEMM unavailable; using SM80 Triton sparse-indexer fallback."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    args = ap.parse_args()

    root = Path(args.vllm_root).resolve()
    path = root / "model_executor/layers/sparse_attn_indexer_kpool.py"
    text = path.read_text(encoding="utf-8")

    if ERROR_TEXT not in text:
        if WARNING_TEXT in text:
            print(f"[deepgemm-gate] already patched: {path}")
            return
        raise RuntimeError(
            "KPool DeepGEMM hard gate not found and fallback marker is absent"
        )

    pattern = re.compile(
        r'(?P<i>^[ \t]*)if current_platform\.is_cuda\(\) and not '
        r'(?:has_deep_gemm|is_deep_gemm_supported)\(\):\n'
        r'(?P=i)    raise RuntimeError\(\n'
        r'(?P=i)        "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed\."\n'
        r'(?P=i)    \)',
        re.MULTILINE,
    )

    def repl(m: re.Match[str]) -> str:
        i = m.group("i")
        return (
            f"{i}if current_platform.is_cuda() and not is_deep_gemm_supported():\n"
            f"{i}    logger.warning_once(\n"
            f'{i}        "{WARNING_TEXT}"\n'
            f"{i}    )"
        )

    patched, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"expected exactly one KPool DeepGEMM hard gate, found {count}")
    if ERROR_TEXT in patched:
        raise RuntimeError("DeepGEMM hard gate message still present after patch")

    path.write_text(patched, encoding="utf-8")
    print(f"[deepgemm-gate] patched: {path}")


if __name__ == "__main__":
    main()
