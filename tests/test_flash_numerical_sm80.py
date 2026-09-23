import math

import torch
import torch.nn.functional as F

from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    fwht128_quant_fp8,
    kpool_compress_and_write_cache,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_sm80 import _encode_e4m3fn_u8
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    triton_mla_sparse_attention,
)


FP8 = torch.float8_e4m3fn
D_INDEX = 128


def stats(name: str, got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    g = got.float().reshape(-1)
    r = ref.float().reshape(-1)
    finite = torch.isfinite(g) & torch.isfinite(r)
    g = g[finite]
    r = r[finite]
    if g.numel() == 0:
        raise AssertionError(f"{name}: no finite elements")
    d = g - r
    absd = d.abs()
    ref_rms = torch.sqrt(torch.mean(r * r)).clamp_min(1e-12)
    out = {
        "mae": float(absd.mean().item()),
        "max_abs": float(absd.max().item()),
        "rmse": float(torch.sqrt(torch.mean(d * d)).item()),
        "rel_l2": float((torch.linalg.vector_norm(d) / torch.linalg.vector_norm(r).clamp_min(1e-12)).item()),
        "nrmse": float((torch.sqrt(torch.mean(d * d)) / ref_rms).item()),
        "cos": float(F.cosine_similarity(g[None], r[None], dim=1).item()),
    }
    print(
        f"{name}: "
        f"mae={out['mae']:.6g} max_abs={out['max_abs']:.6g} "
        f"rmse={out['rmse']:.6g} rel_l2={out['rel_l2']:.6g} "
        f"nrmse={out['nrmse']:.6g} cos={out['cos']:.9f}"
    )
    return out


def byte_agreement(name: str, got: torch.Tensor, ref: torch.Tensor) -> float:
    gu = got.contiguous().view(torch.uint8)
    ru = ref.contiguous().view(torch.uint8)
    rate = float((gu == ru).float().mean().item())
    mismatch = int((gu != ru).sum().item())
    print(f"{name}: byte_agreement={rate:.9f} mismatches={mismatch}/{gu.numel()}")
    return rate


def topk_recall(name: str, got: torch.Tensor, ref: torch.Tensor, k: int) -> tuple[float, float]:
    gi = torch.topk(got.float(), k, dim=-1).indices
    ri = torch.topk(ref.float(), k, dim=-1).indices
    hit = (gi[..., :, None] == ri[..., None, :]).any(dim=-1).float().mean(dim=-1)
    mean = float(hit.mean().item())
    minimum = float(hit.min().item())
    print(f"{name}: top{k}_recall_mean={mean:.6f} min={minimum:.6f}")
    return mean, minimum


def fwht128_ref(x: torch.Tensor) -> torch.Tensor:
    y = x.float().contiguous()
    rows = y.shape[0]
    h = 1
    while h < 128:
        y = y.view(rows, -1, 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y = torch.stack((a + b, a - b), dim=2).reshape(rows, 128)
        h *= 2
    y = y * 0.08838834764831845
    # The production Triton path materializes BF16 before FP8 quantization.
    return y.to(torch.bfloat16).float()


def ue8m0_quant_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    absmax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    q = torch.clamp(x.float() / scale, -448.0, 448.0).to(FP8)
    return q, scale


@triton.jit
def _encode_probe_kernel(inp, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(inp + offs, mask=mask, other=0.0)
    u = _encode_e4m3fn_u8(x)
    tl.store(out + offs, u, mask=mask)


def test_encoder(device: torch.device) -> None:
    torch.manual_seed(11)
    x = torch.cat(
        [
            torch.linspace(-448.0, 448.0, 16384, device=device),
            torch.randn(49152, device=device) * 64.0,
            torch.tensor(
                [-448.0, -240.0, -1.0, -0.0, 0.0, 1.0, 240.0, 448.0],
                device=device,
            ),
        ]
    ).float()
    x = x.clamp(-448.0, 448.0).contiguous()
    out = torch.empty(x.numel(), dtype=torch.uint8, device=device)
    block = 256
    _encode_probe_kernel[(triton.cdiv(x.numel(), block),)](
        x, out, x.numel(), BLOCK=block
    )
    ref = x.to(FP8).view(torch.uint8)
    mismatches = int((out != ref).sum().item())
    print(f"FLASH_E4M3FN_ENCODER: mismatches={mismatches}/{x.numel()}")
    assert mismatches == 0, "software E4M3FN encoder is not bit-exact"
    print("FLASH_E4M3FN_ENCODER=PASS")


def make_q(
    m: int, h: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.randn(m * h, D_INDEX, device=device, dtype=torch.bfloat16)
    q_fp8, q_scale = fwht128_quant_fp8(raw.contiguous())
    q_rot_ref = fwht128_ref(raw)
    q_ref, q_scale_ref = ue8m0_quant_ref(q_rot_ref)

    byte_rate = byte_agreement("FLASH_Q_FWHT_FP8_BYTES", q_fp8, q_ref)
    scale_err = stats("FLASH_Q_FWHT_SCALE", q_scale, q_scale_ref)
    deq = q_fp8.float() * q_scale
    deq_ref = q_ref.float() * q_scale_ref
    dq = stats("FLASH_Q_FWHT_DEQUANT", deq, deq_ref)

    # This path should be essentially identical to the reference. Keep the
    # threshold tight enough to catch a broken SM80 byte encoder / FWHT order
    # without requiring every quantized byte to match at rounding boundaries.
    assert byte_rate > 0.999
    assert scale_err["max_abs"] == 0.0
    assert dq["rel_l2"] < 5e-3
    print("FLASH_Q_FWHT_QUANT=PASS")

    return (
        q_fp8.view(m, h, D_INDEX),
        q_scale.view(m, h),
        q_rot_ref.view(m, h, D_INDEX),
    )


def make_kpool(
    n_pool: int,
    pool_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    slot_k = torch.randn(
        n_pool, pool_size, D_INDEX, device=device, dtype=torch.bfloat16
    )
    slot_score = (
        torch.randn(
            n_pool, pool_size, D_INDEX, device=device, dtype=torch.float32
        )
        * 0.35
    ).to(torch.bfloat16)
    ape = torch.randn(pool_size, D_INDEX, device=device, dtype=torch.float32) * 0.1
    loc = torch.arange(n_pool, device=device, dtype=torch.int64)

    num_blocks = math.ceil(n_pool / block_size)
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        D_INDEX + 4,
        dtype=torch.uint8,
        device=device,
    )

    k_fp8, k_scale = kpool_compress_and_write_cache(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size=pool_size,
        return_compressed=True,
        write_cache=False,
    )

    logits = slot_score.float() + ape[None, :, :]
    prob = torch.softmax(logits, dim=1)
    pooled = (prob * slot_k.float()).sum(dim=1).to(torch.bfloat16).float()
    k_rot_ref = fwht128_ref(pooled)
    k_ref, k_scale_ref_2d = ue8m0_quant_ref(k_rot_ref)
    k_scale_ref = k_scale_ref_2d.squeeze(-1)

    byte_rate = byte_agreement("FLASH_KPOOL_FP8_BYTES", k_fp8, k_ref)
    scale_err = stats("FLASH_KPOOL_SCALE", k_scale, k_scale_ref)
    k_deq = k_fp8.float() * k_scale[:, None]
    k_deq_ref = k_ref.float() * k_scale_ref[:, None]
    dq = stats("FLASH_KPOOL_DEQUANT", k_deq, k_deq_ref)
    ideal = stats("FLASH_KPOOL_VS_UNQUANTIZED", k_deq, k_rot_ref)

    print(
        "FLASH_KPOOL_NOTE: KPOOL_VS_UNQUANTIZED includes the intended FP8 "
        "approximation; it is reported, not used as a kernel-correctness gate."
    )
    assert scale_err["rel_l2"] < 1e-6
    assert dq["rel_l2"] < 3e-2
    assert dq["cos"] > 0.999
    # Byte agreement can be slightly lower at FP8 rounding boundaries because
    # Triton's exp implementation and torch.softmax are not bit-identical.
    assert byte_rate > 0.98

    # Exercise the actual cache writer too, then verify its physical layout.
    kpool_compress_and_write_cache(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size=pool_size,
        return_compressed=False,
        write_cache=True,
    )
    flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * D_INDEX
    cached_k = (
        flat[:, :k_end]
        .reshape(num_blocks, block_size, D_INDEX)
        .reshape(-1, D_INDEX)[:n_pool]
        .view(FP8)
    )
    cached_scale = flat[:, k_end:].view(torch.float32).reshape(-1)[:n_pool]
    assert torch.equal(cached_k.view(torch.uint8), k_fp8.view(torch.uint8))
    torch.testing.assert_close(cached_scale, k_scale, rtol=0, atol=0)
    print("FLASH_KPOOL_CACHE_LAYOUT=PASS")

    return k_fp8, k_scale, k_rot_ref, kv_cache


def mqa_semantic_ref(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    dot = torch.einsum(
        "mhd,nd->mhn",
        q_fp8.float(),
        k_fp8.float(),
    )
    dot = dot * k_scale[None, None, :]
    return (torch.relu(dot) * weights[:, :, None]).sum(dim=1)


def mqa_ideal_ref(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    base_weights: torch.Tensor,
) -> torch.Tensor:
    dot = torch.einsum("mhd,nd->mhn", q_rot.float(), k_rot.float())
    return (torch.relu(dot) * base_weights[:, :, None]).sum(dim=1)


def test_mqa_prefill(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    q_rot: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    k_rot: torch.Tensor,
) -> None:
    m, h, _ = q_fp8.shape
    n = k_fp8.shape[0]
    base_weights = torch.randn(m, h, device=q_fp8.device, dtype=torch.float32) / math.sqrt(h)
    # Production FP8 indexer folds per-head Q scale into weights.
    weights = base_weights * q_scale

    ks = torch.zeros(m, device=q_fp8.device, dtype=torch.int32)
    ke = torch.full((m,), n, device=q_fp8.device, dtype=torch.int32)
    got = fp8_mqa_logits_triton(q_fp8, (k_fp8, k_scale), weights, ks, ke)

    quant_ref = mqa_semantic_ref(q_fp8, k_fp8, k_scale, weights)
    ideal_ref = mqa_ideal_ref(q_rot, k_rot, base_weights)

    kernel_err = stats("FLASH_MQA_PREFILL_KERNEL_VS_QUANT_REF", got, quant_ref)
    approx_err = stats("FLASH_MQA_PREFILL_END2END_VS_UNQUANT", got, ideal_ref)
    topk_recall("FLASH_MQA_PREFILL_END2END", got, ideal_ref, min(64, n))

    print(
        "FLASH_MQA_PREFILL_NOTE: END2END_VS_UNQUANT includes Q/K FP8 and "
        "KPool approximation; KERNEL_VS_QUANT_REF isolates Triton arithmetic."
    )
    assert kernel_err["rel_l2"] < 2e-2
    assert kernel_err["cos"] > 0.999
    print("FLASH_MQA_PREFILL_NUMERICAL=PASS")

    # Also verify masking/clean_logits=False, the production prefill mode.
    ks2 = torch.arange(m, device=q_fp8.device, dtype=torch.int32) * 3
    ke2 = torch.minimum(ks2 + n // 2, torch.full_like(ks2, n))
    masked = fp8_mqa_logits_triton(
        q_fp8, (k_fp8, k_scale), weights, ks2, ke2, clean_logits=False
    )
    ar = torch.arange(n, device=q_fp8.device)[None, :]
    valid = (ar >= ks2[:, None]) & (ar < ke2[:, None])
    assert torch.equal(torch.isneginf(masked), ~valid)
    masked_ref = quant_ref.masked_fill(~valid, float("-inf"))
    e = stats(
        "FLASH_MQA_PREFILL_MASKED_KERNEL_VS_QUANT_REF",
        masked[valid],
        masked_ref[valid],
    )
    assert e["rel_l2"] < 2e-2
    print("FLASH_MQA_PREFILL_MASKING=PASS")


def test_mqa_decode(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    kv_cache_3d: torch.Tensor,
) -> None:
    h = q_fp8.shape[1]
    block_size = kv_cache_3d.shape[1]
    n = k_fp8.shape[0]
    b = 2
    next_n = 3
    rows = b * next_n
    assert q_fp8.shape[0] >= rows

    q = q_fp8[:rows].view(b, next_n, h, D_INDEX).contiguous()
    base_weights = torch.randn(rows, h, device=q.device, dtype=torch.float32) / math.sqrt(h)
    weights = base_weights * q_scale[:rows]

    num_blocks = kv_cache_3d.shape[0]
    block_tables = torch.arange(
        num_blocks, device=q.device, dtype=torch.int32
    ).repeat(b, 1)
    if num_blocks >= 4:
        block_tables[1] = torch.tensor(
            [2, 0, 3, 1] + list(range(4, num_blocks)),
            dtype=torch.int32,
            device=q.device,
        )

    context_lens = torch.tensor(
        [n, max(next_n, n - 17)], dtype=torch.int32, device=q.device
    )
    got = fp8_paged_mqa_logits_triton(
        q,
        kv_cache_3d.unsqueeze(-2),
        weights,
        context_lens,
        block_tables,
        max_model_len=n,
        clean_logits=True,
    )

    refs = []
    valid_masks = []
    for bi in range(b):
        blocks = block_tables[bi].long()
        logical_k = torch.cat(
            [
                k_fp8[
                    int(block.item()) * block_size :
                    min((int(block.item()) + 1) * block_size, n)
                ]
                for block in blocks
            ],
            dim=0,
        )
        logical_s = torch.cat(
            [
                k_scale[
                    int(block.item()) * block_size :
                    min((int(block.item()) + 1) * block_size, n)
                ]
                for block in blocks
            ],
            dim=0,
        )
        clen = int(context_lens[bi].item())
        logical_k = logical_k[:clen]
        logical_s = logical_s[:clen]
        for ti in range(next_n):
            ridx = bi * next_n + ti
            qrow = q_fp8[ridx : ridx + 1]
            wrow = weights[ridx : ridx + 1]
            r = mqa_semantic_ref(qrow, logical_k, logical_s, wrow)[0]
            q_offset = clen - next_n + ti
            valid_len = q_offset + 1
            row = torch.full((n,), float("-inf"), device=q.device)
            row[:valid_len] = r[:valid_len]
            mask = torch.zeros(n, dtype=torch.bool, device=q.device)
            mask[:valid_len] = True
            refs.append(row)
            valid_masks.append(mask)

    ref = torch.stack(refs)
    valid = torch.stack(valid_masks)
    assert torch.equal(torch.isneginf(got), ~valid)
    e = stats(
        "FLASH_MQA_DECODE_KERNEL_VS_QUANT_REF",
        got[valid],
        ref[valid],
    )
    assert e["rel_l2"] < 2e-2
    assert e["cos"] > 0.999

    recalls = []
    for i in range(rows):
        k = min(32, int(valid[i].sum().item()))
        gi = torch.topk(got[i, valid[i]], k).indices
        ri = torch.topk(ref[i, valid[i]], k).indices
        recalls.append(
            float((gi[:, None] == ri[None, :]).any(dim=1).float().mean().item())
        )
    print(
        "FLASH_MQA_DECODE: "
        f"topk_recall_mean={sum(recalls)/len(recalls):.6f} "
        f"min={min(recalls):.6f}"
    )
    assert min(recalls) > 0.98
    print("FLASH_MQA_DECODE_NUMERICAL=PASS")


def test_sparse_mla(device: torch.device) -> None:
    torch.manual_seed(29)
    t, h, d, seq, topk = 2, 16, 512, 512, 256
    q = torch.randn(t, h, d, device=device, dtype=torch.bfloat16)
    kv = torch.randn(seq, 1, d, device=device, dtype=torch.bfloat16)
    indices = torch.empty(t, 1, topk, device=device, dtype=torch.int32)
    for ti in range(t):
        indices[ti, 0] = torch.randperm(seq, device=device)[:topk].to(torch.int32)

    scale = 1.0 / math.sqrt(d)
    refs = []
    for ti in range(t):
        idx = indices[ti, 0].long()
        k = kv[idx, 0].float()
        score = q[ti].float() @ k.T * scale
        prob = torch.softmax(score, dim=-1)
        refs.append(prob @ k[:, :512])
    ref = torch.stack(refs)

    for splits in (1, 2, 4, None):
        got = triton_mla_sparse_attention(
            q, kv, indices, sm_scale=scale, num_kv_splits=splits
        ).float()
        e = stats(f"FLASH_SPARSE_MLA_SPLIT_{splits}", got, ref)
        assert e["rel_l2"] < 3e-2
        assert e["cos"] > 0.999
    print("FLASH_SPARSE_MLA_NUMERICAL=PASS")


def main() -> None:
    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(device)
    print("device:", torch.cuda.get_device_name(device))
    print("compute capability:", cap)
    print("torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
    assert cap == (8, 0), f"expected SM80/A100, got {cap}"

    # Keep the FP32 reference path as a reference rather than silently using TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)

    test_encoder(device)

    m, h = 8, 32
    q_fp8, q_scale, q_rot = make_q(m, h, device)

    n_pool, pool_size, block_size = 256, 16, 64
    k_fp8, k_scale, k_rot, kv_cache = make_kpool(
        n_pool, pool_size, block_size, device
    )

    test_mqa_prefill(q_fp8, q_scale, q_rot, k_fp8, k_scale, k_rot)
    test_mqa_decode(q_fp8, q_scale, k_fp8, k_scale, kv_cache)
    test_sparse_mla(device)

    print("FLASH_SM80_NUMERICAL_VERIFY=PASS")


if __name__ == "__main__":
    main()
