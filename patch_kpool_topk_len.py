#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    args = ap.parse_args()

    path = Path(args.vllm_root).resolve() / "model_executor/layers/sparse_attn_indexer_kpool.py"
    text = path.read_text(encoding="utf-8")

    old = (
        "                select_k,\n"
        "                attn_metadata_narrowed.max_seq_len,\n"
        "            )"
    )
    new = (
        "                select_k,\n"
        "                # persistent_topk operates on pool-granular logits;\n"
        "                # keep its host-side width consistent with device lengths.\n"
        "                logits.shape[1],\n"
        "            )"
    )

    count = text.count(old)
    if count == 0:
        if "persistent_topk operates on pool-granular logits" in text:
            print(f"[kpool-topk-len] already patched: {path}")
            return
        raise RuntimeError("KPool persistent_topk max_seq_len anchor not found")

    # There are prefill/decode persistent_topk call sites in the pinned base.
    text = text.replace(old, new)
    if "select_k,\n                attn_metadata_narrowed.max_seq_len" in text:
        raise RuntimeError("stale token-granular persistent_topk length remains")
    path.write_text(text, encoding="utf-8")
    print(f"[kpool-topk-len] patched {count} call site(s): {path}")


if __name__ == "__main__":
    main()
