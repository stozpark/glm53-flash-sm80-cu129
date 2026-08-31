#!/usr/bin/env python3
"""GLM-5.3 SM80 KPoolTail/MTP correctness hardening for glm53-flash-cu129.

Applied after patch_runtime.py. These fixes are intentionally kept separate from
#47629/#54031 plumbing because they are GLM-5.3 hybrid/KPoolTail correctness
fixes discovered later on Ampere.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        fail(f"{label}: expected one match, found {n}")
    return text.replace(old, new, 1)


def patch_indexer(text: str) -> str:
    # Persistent, -1-filled KPoolTail mapping: required by FULL CUDA graph replay.
    if "out_full: torch.Tensor" not in text:
        sig = '''def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
) -> torch.Tensor:
'''
        sig2 = sig.replace(
            "    kpool: int,\n) -> torch.Tensor:\n",
            "    kpool: int,\n    out_full: torch.Tensor,\n) -> torch.Tensor:\n",
        )
        text = replace_once(text, sig, sig2, "KPoolTail mapping signature")
        text = replace_once(
            text,
            "    out = slot_mapping.clone()\n",
            "    n = slot_mapping.shape[0]\n"
            "    out_full.fill_(-1)\n"
            "    out = out_full[:n]\n",
            "KPoolTail padded sentinel",
        )

    if "self.tail_slot_mapping_buffer = torch.full(" not in text:
        anchor = (
            "        super().__init__(kv_cache_spec, layer_names, vllm_config, device)\n"
            "\n"
            "    def build(\n"
        )
        repl = (
            "        super().__init__(kv_cache_spec, layer_names, vllm_config, device)\n"
            "        self.tail_slot_mapping_buffer = torch.full(\n"
            "            (vllm_config.scheduler_config.max_num_batched_tokens,),\n"
            "            -1,\n"
            "            dtype=torch.int64,\n"
            "            device=device,\n"
            "        )\n"
            "\n"
            "    def build(\n"
        )
        text = replace_once(text, anchor, repl, "KPoolTail persistent buffer")

    old = '''        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            # Circular per-request layout; the generic kernel output collapses
            # onto tail block 0 for pos >= kpool (see compute_... docstring).
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
'''
    if old in text:
        new = '''        positions = common_attn_metadata.positions
        if positions is None:
            raise ValueError(
                "KpoolTailMetadataBuilder requires CommonAttentionMetadata.positions"
            )
        slot_mapping = compute_kpool_tail_slot_mapping(
            common_attn_metadata.slot_mapping,
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.query_start_loc,
            positions,
            common_attn_metadata.num_actual_tokens,
            common_attn_metadata.num_reqs,
            self.kv_cache_spec.block_size,
            out_full=self.tail_slot_mapping_buffer,
        )
'''
        text = text.replace(old, new, 1)
    elif "out_full=self.tail_slot_mapping_buffer" not in text:
        fail("KPoolTail builder mapping anchor not found")

    # MTP warmup can have padded seq_lens rows beyond num_decodes.
    old = '''                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
'''
    new = '''                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens[:num_decodes] // self.compress_ratio
                    )
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "seq_lens[:num_decodes] // self.compress_ratio" not in text:
        fail("MTP padded seq_lens fix anchor not found")
    return text


def patch_mamba_hybrid(text: str) -> str:
    if "positions=input_batch.positions" in text:
        return text
    anchor = "            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,\n"
    return replace_once(
        text,
        anchor,
        anchor + "            positions=input_batch.positions,\n",
        "hybrid positions propagation",
    )


def patch_speculator(text: str) -> str:
    if "positions=self.input_buffers.positions[:num_tokens_padded]" in text:
        return text
    anchor = "            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,\n"
    return replace_once(
        text,
        anchor,
        anchor + "            positions=self.input_buffers.positions[:num_tokens_padded],\n",
        "MTP draft positions",
    )


def patch_model_runner(text: str) -> str:
    if "KpoolTailSpec" not in text[:6000]:
        old = "from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec\n"
        new = (
            "from vllm.v1.kv_cache_interface import (\n"
            "    KVCacheConfig,\n"
            "    KpoolTailSpec,\n"
            "    MambaSpec,\n"
            ")\n"
        )
        text = replace_once(text, old, new, "KpoolTailSpec import")

    if "slot_mapping_enabled=slot_mapping_enabled" not in text:
        anchor = "        self.block_tables = BlockTables(\n"
        flags = (
            "        slot_mapping_enabled = [\n"
            "            not isinstance(group.kv_cache_spec, KpoolTailSpec)\n"
            "            for group in kv_cache_config.kv_cache_groups\n"
            "        ]\n"
        )
        text = replace_once(text, anchor, flags + anchor, "tail group slot-map flags")
        anchor = "            cp_interleave=self.cp_interleave,\n        )\n"
        text = replace_once(
            text,
            anchor,
            "            cp_interleave=self.cp_interleave,\n"
            "            slot_mapping_enabled=slot_mapping_enabled,\n"
            "        )\n",
            "BlockTables tail flag wiring",
        )
    return text


def patch_block_table(text: str) -> str:
    if "slot_mapping_enabled: list[bool] | None = None" not in text:
        text = replace_once(
            text,
            "        cp_interleave: int = 1,\n    ):\n",
            "        cp_interleave: int = 1,\n"
            "        slot_mapping_enabled: list[bool] | None = None,\n"
            "    ):\n",
            "BlockTables slot-map flag arg",
        )
        anchor = "        self.num_kv_cache_groups = len(self.block_sizes)\n"
        text = replace_once(
            text,
            anchor,
            anchor
            + "        if slot_mapping_enabled is None:\n"
            + "            slot_mapping_enabled = [True] * self.num_kv_cache_groups\n"
            + "        assert len(slot_mapping_enabled) == self.num_kv_cache_groups\n"
            + "        self.slot_mapping_enabled = torch.tensor(\n"
            + "            slot_mapping_enabled, dtype=torch.int32, device=self.device\n"
            + "        )\n",
            "BlockTables slot-map flag tensor",
        )

    if "self.slot_mapping_enabled," not in text:
        text = replace_once(
            text,
            "            self.block_sizes_tensor,\n            slot_mappings,\n",
            "            self.block_sizes_tensor,\n"
            "            self.slot_mapping_enabled,\n"
            "            slot_mappings,\n",
            "slot mapping kernel flag arg",
        )

    if "slot_mapping_enabled,  # [num_kv_cache_groups]" not in text:
        text = replace_once(
            text,
            "    block_sizes,  # [num_kv_cache_groups]\n"
            "    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]\n",
            "    block_sizes,  # [num_kv_cache_groups]\n"
            "    slot_mapping_enabled,  # [num_kv_cache_groups]\n"
            "    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]\n",
            "slot mapping kernel signature",
        )
        old = (
            "    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride\n"
            "\n"
            "    if batch_idx == tl.num_programs(1) - 1:\n"
        )
        new = (
            "    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride\n"
            "\n"
            "    enabled = tl.load(slot_mapping_enabled + group_id) != 0\n"
            "    if not enabled:\n"
            "        if batch_idx == tl.num_programs(1) - 1:\n"
            "            for i in range(0, max_num_tokens, TRITON_BLOCK_SIZE):\n"
            "                offset = i + tl.arange(0, TRITON_BLOCK_SIZE)\n"
            "                tl.store(\n"
            "                    slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens\n"
            "                )\n"
            "        return\n"
            "\n"
            "    if batch_idx == tl.num_programs(1) - 1:\n"
        )
        text = replace_once(text, old, new, "generic tail slot-map skip")
    return text


def write_patched(path: Path, fn) -> None:
    path.write_text(fn(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"[glm53-tail] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--vendor", required=True)
    args = ap.parse_args()
    root = Path(args.vllm_root).resolve()
    vendor = Path(args.vendor).resolve()

    # Replace the KPool writer with the pinned Ampere implementation and add
    # the shared software FP8 encoder used by its Triton kernels.
    fp8 = root / "v1/attention/ops/fp8_sm80.py"
    kpool = root / "models/glm5next/nvidia/ops/kpool_compress.py"
    shutil.copy2(vendor / "backport/fp8_sm80.py", fp8)
    shutil.copy2(vendor / "backport/kpool_compress.py", kpool)
    print(f"[glm53-tail] {fp8}")
    print(f"[glm53-tail] {kpool}")

    write_patched(root / "v1/attention/backends/mla/indexer.py", patch_indexer)
    write_patched(root / "v1/worker/gpu/model_states/mamba_hybrid.py", patch_mamba_hybrid)
    write_patched(root / "v1/worker/gpu/spec_decode/speculator.py", patch_speculator)
    write_patched(root / "v1/worker/gpu/model_runner.py", patch_model_runner)
    write_patched(root / "v1/worker/gpu/block_table.py", patch_block_table)


if __name__ == "__main__":
    main()
