from pathlib import Path
import importlib.util

P = Path(__file__).resolve().parents[1] / "patch_glm53_tail.py"
spec = importlib.util.spec_from_file_location("patch_glm53_tail", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

idx = '''def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
) -> torch.Tensor:
    out = slot_mapping.clone()

class KpoolTailMetadataBuilder:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        slot_mapping = common_attn_metadata.slot_mapping
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

                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
'''
out = m.patch_indexer(idx)
assert "out_full.fill_(-1)" in out
assert "self.tail_slot_mapping_buffer = torch.full(" in out
assert "out_full=self.tail_slot_mapping_buffer" in out
assert "seq_lens[:num_decodes] // self.compress_ratio" in out

hybrid = "            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,\n"
assert "positions=input_batch.positions" in m.patch_mamba_hybrid(hybrid)

speculator = "            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,\n"
assert "positions=self.input_buffers.positions[:num_tokens_padded]" in m.patch_speculator(speculator)

runner = '''from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
        self.block_tables = BlockTables(
            cp_interleave=self.cp_interleave,
        )
'''
out = m.patch_model_runner(runner)
assert "KpoolTailSpec" in out
assert "slot_mapping_enabled=slot_mapping_enabled" in out

bt = '''class BlockTables:
    def __init__(
        self,
        cp_interleave: int = 1,
    ):
        self.device = device
        self.num_kv_cache_groups = len(self.block_sizes)

    def f(self):
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            self.block_sizes_tensor,
            slot_mappings,
        )

@triton.jit
def _compute_slot_mappings_kernel(
    block_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
):
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        pass
'''
out = m.patch_block_table(bt)
assert "slot_mapping_enabled: list[bool] | None = None" in out
assert "enabled = tl.load(slot_mapping_enabled + group_id) != 0" in out
print("PATCH_GLM53_TAIL_UNIT=PASS")
