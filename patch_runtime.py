#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def die(msg: str) -> None:
    raise RuntimeError(msg)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        die(f"{label}: expected exactly one match, found {n}")
    return text.replace(old, new, 1)


def regex_replace_once(text: str, pattern: str, repl, label: str, flags=0) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        die(f"{label}: expected exactly one regex match, found {n}")
    return out


def backup(path: Path, backup_root: Path, vllm_root: Path) -> None:
    rel = path.relative_to(vllm_root)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not dst.exists():
        shutil.copy2(path, dst)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[patched] {path}")


def patch_54031_backend(src: str) -> str:
    # PR #54031 targets a newer FlashMLASparseImpl whose hook returns
    # (output, lse). glm53-flash-cu129 @ 487ecf187 expects output only.
    src = regex_replace_once(
        src,
        r"(def _bf16_flash_mla_kernel\([\s\S]*?\n\s*\)) -> tuple\[torch\.Tensor, torch\.Tensor \| None\]:",
        r"\1 -> torch.Tensor:",
        "#54031 backend return annotation",
    )
    src = replace_once(
        src,
        "return output[:, : self.num_heads, :], None",
        "return output[:, : self.num_heads, :]",
        "#54031 backend return value",
    )
    return src


def patch_registry(text: str) -> str:
    if "TRITON_MLA_SPARSE" in text:
        return text
    anchor = '    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"\n'
    insert = (
        anchor
        + '    TRITON_MLA_SPARSE = (\n'
        + '        "vllm.v1.attention.backends.mla.triton_mla_sparse.TritonMLASparseBackend"\n'
        + '    )\n'
    )
    return replace_once(text, anchor, insert, "registry TRITON_MLA_SPARSE")


def patch_cuda_priorities(text: str) -> str:
    if "AttentionBackendEnum.TRITON_MLA_SPARSE" in text:
        return text

    # glm53-flash-cu129 @ 487ecf187 keeps sparse backends in sparse_tail.
    base_anchor = (
        "            sparse_tail = [\n"
        "                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,\n"
        "                AttentionBackendEnum.FLASHMLA_SPARSE,\n"
        "                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,\n"
        "            ]\n"
    )
    if base_anchor in text:
        repl = (
            "            sparse_tail = [\n"
            "                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,\n"
            "                AttentionBackendEnum.FLASHMLA_SPARSE,\n"
            "                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,\n"
            "                AttentionBackendEnum.TRITON_MLA_SPARSE,\n"
            "            ]\n"
        )
        return text.replace(base_anchor, repl, 1)

    # Newer-main layout used by PR #47629.
    newer_anchor = (
        "                AttentionBackendEnum.TRITON_MLA,\n"
        "                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,\n"
        "                AttentionBackendEnum.FLASHMLA_SPARSE,\n"
    )
    if newer_anchor in text:
        return text.replace(
            newer_anchor,
            newer_anchor + "                AttentionBackendEnum.TRITON_MLA_SPARSE,\n",
            1,
        )
    die("cuda SM80 sparse backend priority anchor not found")


def patch_mla_indexer_metadata(text: str) -> str:
    if "is_deep_gemm_supported" not in text:
        anchor = "    has_deep_gemm,\n"
        text = replace_once(
            text,
            anchor,
            anchor + "    is_deep_gemm_supported,\n",
            "mla/indexer import is_deep_gemm_supported",
        )
    old = "if current_platform.is_cuda() and has_deep_gemm():"
    if old in text:
        text = text.replace(
            old, "if current_platform.is_cuda() and is_deep_gemm_supported():", 1
        )
    elif "if current_platform.is_cuda() and is_deep_gemm_supported():" not in text:
        die("mla/indexer DeepGEMM metadata gate not found")
    return text


def _indent_block(block: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in block.splitlines())


def patch_kpool_indexer(text: str) -> str:
    # Architecture-aware DeepGEMM gate.
    text = text.replace("    has_deep_gemm,\n", "    is_deep_gemm_supported,\n")
    text = text.replace("has_deep_gemm()", "is_deep_gemm_supported()")
    if "fp8_mqa_logits_triton" not in text:
        anchor = "from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton\n"
        add = (
            anchor
            + "from vllm.v1.attention.ops.mqa_logits_triton import (\n"
            + "    fp8_mqa_logits_triton,\n"
            + "    fp8_paged_mqa_logits_triton,\n"
            + ")\n"
        )
        text = replace_once(text, anchor, add, "kpool Triton MQA imports")

    if "use_deep_gemm = is_deep_gemm_supported()" not in text:
        anchor = "    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens\n"
        add = (
            anchor
            + "    use_deep_gemm = is_deep_gemm_supported()\n"
            + "    if not use_deep_gemm and use_fp4_cache:\n"
            + "        raise RuntimeError(\n"
            + '            "SM80 Triton sparse-indexer fallback supports FP8 K cache only"\n'
            + "        )\n"
        )
        text = replace_once(text, anchor, add, "kpool use_deep_gemm gate")

    # Prefill: replace the one direct DeepGEMM call with an SM80 Triton fallback.
    prefill_pat = re.compile(
        r"^(?P<indent>[ \t]*)logits = fp8_fp4_mqa_logits\(\n"
        r"(?P<body>[\s\S]*?)\n(?P=indent)\)",
        re.MULTILINE,
    )
    m = prefill_pat.search(text)
    if m:
        indent = m.group("indent")
        original = m.group(0).lstrip("\n")
        original_indented = _indent_block(original, 4)
        fallback = (
            f"{indent}if use_deep_gemm:\n"
            f"{original_indented}\n"
            f"{indent}else:\n"
            f"{indent}    logits = fp8_mqa_logits_triton(\n"
            f"{indent}        q_slice_cast,\n"
            f"{indent}        (k_quant_cast, k_scale_cast),\n"
            f"{indent}        weights[chunk.token_start : chunk.token_end],\n"
            f"{indent}        chunk.cu_seqlen_ks,\n"
            f"{indent}        chunk.cu_seqlen_ke,\n"
            f"{indent}        clean_logits=False,\n"
            f"{indent}    )"
        )
        text = text[: m.start()] + fallback + text[m.end() :]
    elif "logits = fp8_mqa_logits_triton(" not in text:
        die("kpool prefill DeepGEMM call not found")

    # Decode: replace the one direct DeepGEMM paged call. Keep seq_lens 2-D for
    # downstream top-k, but collapse to [B] for the Triton kernel (MTP fix).
    decode_pat = re.compile(
        r"^(?P<indent>[ \t]*)logits = fp8_fp4_paged_mqa_logits\(\n"
        r"(?P<body>[\s\S]*?)\n(?P=indent)\)",
        re.MULTILINE,
    )
    m = decode_pat.search(text)
    if m:
        indent = m.group("indent")
        original = m.group(0).lstrip("\n")
        original_indented = _indent_block(original, 4)
        fallback = (
            f"{indent}if use_deep_gemm:\n"
            f"{original_indented}\n"
            f"{indent}else:\n"
            f"{indent}    triton_seq_lens = (\n"
            f"{indent}        seq_lens[:, -1].contiguous()\n"
            f"{indent}        if seq_lens.ndim == 2\n"
            f"{indent}        else seq_lens\n"
            f"{indent}    )\n"
            f"{indent}    active_max_model_len = int(attn_metadata_narrowed.max_seq_len)\n"
            f"{indent}    logits = fp8_paged_mqa_logits_triton(\n"
            f"{indent}        padded_q_quant_cast,\n"
            f"{indent}        kv_cache,\n"
            f"{indent}        padded_weights[:num_padded_tokens],\n"
            f"{indent}        triton_seq_lens,\n"
            f"{indent}        decode_metadata.block_table,\n"
            f"{indent}        max_model_len=active_max_model_len,\n"
            f"{indent}        clean_logits=False,\n"
            f"{indent}    )"
        )
        text = text[: m.start()] + fallback + text[m.end() :]
    elif "triton_seq_lens" not in text:
        die("kpool decode DeepGEMM call not found")

    # If this day-0 file carries the generic hard error, downgrade it.
    hard_error = re.compile(
        r"(?P<i>\s*)if current_platform\.is_cuda\(\) and not is_deep_gemm_supported\(\):\n"
        r"(?P=i)    raise RuntimeError\(\n"
        r"(?P=i)        \"Sparse Attention Indexer CUDA op requires DeepGEMM support in \"\n"
        r"(?P=i)        \"the current vLLM environment\.\"\n"
        r"(?P=i)    \)",
    )
    text, n = hard_error.subn(
        lambda mm: (
            f"{mm.group('i')}if current_platform.is_cuda() and not is_deep_gemm_supported():\n"
            f"{mm.group('i')}    logger.warning_once(\n"
            f'{mm.group("i")}        "DeepGEMM unsupported; using SM80 Triton sparse-indexer fallback."\n'
            f"{mm.group('i')}    )"
        ),
        text,
        count=1,
    )
    return text


def patch_gpu_model_runner(text: str) -> str:
    if "if self.vllm_config.max_concurrent_batches > 1:" in text:
        return text
    old = (
        "        if self.use_async_scheduling:\n"
        "            self.async_output_copy_stream = torch.cuda.Stream()\n"
        "            self.prepare_inputs_event = torch.Event()\n"
    )
    new = (
        "        if self.use_async_scheduling:\n"
        "            self.async_output_copy_stream = torch.cuda.Stream()\n"
        "\n"
        "        self.prepare_inputs_event = None\n"
        "        if self.vllm_config.max_concurrent_batches > 1:\n"
        "            self.prepare_inputs_event = torch.Event()\n"
    )
    return replace_once(text, old, new, "#47644 prepare_inputs_event")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--backup-root", default="/opt/glm53-sm80-stock")
    args = ap.parse_args()

    root = Path(args.vllm_root).resolve()
    vendor = Path(args.vendor).resolve()
    backup_root = Path(args.backup_root).resolve()
    if not (root / "__init__.py").exists():
        die(f"not a vllm package root: {root}")

    # Paths we modify.
    p_registry = root / "v1/attention/backends/registry.py"
    p_cuda = root / "platforms/cuda.py"
    p_mla_indexer = root / "v1/attention/backends/mla/indexer.py"
    p_kpool = root / "model_executor/layers/sparse_attn_indexer_kpool.py"
    p_kpool_compress = root / "models/glm5next/nvidia/ops/kpool_compress.py"
    p_runner = root / "v1/worker/gpu_model_runner.py"
    p_backend = root / "v1/attention/backends/mla/triton_mla_sparse.py"
    p_kernel = root / "v1/attention/ops/triton_mla_sparse_kernel.py"
    p_mqa = root / "v1/attention/ops/mqa_logits_triton.py"

    for p in [
        p_registry,
        p_cuda,
        p_mla_indexer,
        p_kpool,
        p_kpool_compress,
        p_runner,
        p_backend,
        p_kernel,
        p_mqa,
    ]:
        if p.exists():
            backup(p, backup_root, root)

    # New/replacement kernel files from pinned upstream PRs.
    backend_src = (vendor / "pr54031/triton_mla_sparse.py").read_text()
    backend_src = patch_54031_backend(backend_src)
    write(p_backend, backend_src)
    shutil.copy2(vendor / "pr54031/triton_mla_sparse_kernel.py", p_kernel)
    print(f"[patched] {p_kernel}")
    shutil.copy2(vendor / "pr47629/mqa_logits_triton.py", p_mqa)
    print(f"[patched] {p_mqa}")

    # GLM-5.3 KPool compressor: only component retained from the A800 project.
    shutil.copy2(vendor / "a800/kpool_compress.py", p_kpool_compress)
    print(f"[patched] {p_kpool_compress}")

    write(p_registry, patch_registry(p_registry.read_text()))
    write(p_cuda, patch_cuda_priorities(p_cuda.read_text()))
    write(p_mla_indexer, patch_mla_indexer_metadata(p_mla_indexer.read_text()))
    write(p_kpool, patch_kpool_indexer(p_kpool.read_text()))
    write(p_runner, patch_gpu_model_runner(p_runner.read_text()))

    print("Patch application complete.")


if __name__ == "__main__":
    main()
