"""SM80 probe: patched kpool_compress vs pure-torch oracle. Run on A800 box."""
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "kpc",
    "/w/_port/patches_glm53_sm80/vllm/models/glm5next/nvidia/ops/kpool_compress.py",
)
kpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kpc)

torch.manual_seed(0)
dev = "cuda"
fails = []


def act_quant_ref(x_f32):  # sglang chain: absmax>=1e-4, ue8m0 scale, +-448 clamp
    x = x_f32.to(torch.bfloat16).to(torch.float32)
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / 448.0))))
    y = (x / scale).clamp(-448.0, 448.0)
    return y.to(torch.float8_e4m3fn), scale


# ---------- Test A: fwht128_quant_fp8 ----------
rows = 257
q = torch.randn(rows, 128, device=dev, dtype=torch.bfloat16)
H = kpc._hadamard128_torch(
    torch.eye(128, device=dev, dtype=torch.float32)
)  # [128,128] exact-ish
rot_ref = (q.float() @ H).to(torch.bfloat16)  # unfused chain materializes bf16
qref, sref = act_quant_ref(rot_ref.float())
qnew, snew = kpc.fwht128_quant_fp8(q.clone())
same_bytes = (
    qref.view(torch.uint8) == qnew.view(torch.uint8)
).float().mean().item()
rel = ((sref - snew).abs() / sref).max().item()
print(f"[A] byte-match={same_bytes*100:.2f}% scale_rel_err_max={rel:.3e}")
if same_bytes < 0.999 or rel > 1e-6:
    fails.append("A")

# ---------- Test B: prefill compress-write ----------
POOL, HD = 16, 128
n_pages, page_size, n_pools = 4, 64, 37
kv = torch.zeros(n_pages * page_size * (HD + 4), dtype=torch.uint8, device=dev)
slot_k = torch.randn(n_pools, POOL, HD, device=dev, dtype=torch.bfloat16)
slot_score = torch.randn(n_pools, POOL, HD, device=dev, dtype=torch.bfloat16)
ape = torch.randn(POOL, HD, device=dev, dtype=torch.float32)
locs = torch.randperm(n_pools, device=dev).to(torch.int64)
mask_rows = torch.ones(n_pools, dtype=torch.bool, device=dev)
mask_rows[5::7] = False

buf_fp32 = kv.view(torch.float32)
kpc.kpool_compress_and_write_cache(
    kv.view(n_pages, page_size, HD + 4),
    slot_k,
    slot_score,
    ape,
    locs,
    POOL,
    write_mask=mask_rows,
)
# torch oracle for selected rows
sel = mask_rows.nonzero(as_tuple=True)[0]
sc = slot_score.float() + ape[None]
prob = torch.softmax(sc, dim=1)
x = (slot_k.float() * prob).sum(dim=1).to(torch.bfloat16).float()  # kernel rounds pre-H
x = (x @ H).to(torch.bfloat16).float()  # kernel rounds post-H too
qref_b, sref_b = act_quant_ref(x)

off_bytes = locs[sel] // page_size * kv.view(n_pages, -1).stride(0)[()] if False else None
page = locs[sel] // page_size
tok = locs[sel] % page_size
base = page * kv.view(n_pages, -1).stride(0) + tok * HD
ref_by = qref_b[sel].view(torch.uint8)
got_by = torch.stack(
    [kv.view(-1)[b : b + HD] for b in base]
)
mism = (ref_by != got_by).sum().item()
s_base = page * kv.view(n_pages, -1).stride(0) + page_size * HD + tok * 4
got_s = torch.stack([kv.view(torch.float32)[b // 4 : b // 4 + 1][0] for b in s_base])
serr = ((got_s - sref_b[sel, 0]).abs() / sref_b[sel, 0]).max().item()
tot = ref_by.numel()
print(f"[B] rows={len(sel)} byte-mismatch={mism}/{tot} scale_rel_err_max={serr:.3e}")
if mism or serr > 1e-6:
    fails.append("B")

# ---------- Test C: decode completion (single pool fill) ----------
kv2 = torch.zeros(n_pages * page_size * (HD + 4), dtype=torch.uint8, device=dev)
tail = torch.randn(2, 2, POOL, HD, device=dev, dtype=torch.bfloat16)  # [B,2,K,HD]
tail_slot = torch.tensor([[30], [20]], dtype=torch.int32, device=dev)  # blk 1 each
tail_slot = tail_slot.repeat(1, POOL).contiguous()  # [B, next_n]
key = torch.randn(2, POOL, HD, device=dev, dtype=torch.bfloat16)  # next_n=POOL
sscore = torch.randn(2, POOL, HD, device=dev, dtype=torch.bfloat16)
pos = torch.arange(POOL, device=dev, dtype=torch.int32)
pos_last = pos[-1].item()
# position requests so token POOL-1 completes pool 0 for both reqs
positions = pos.repeat(2, 1).contiguous()
slot_map = torch.full((2, POOL), 7, dtype=torch.int32, device=dev)
slot_map[:, -1] = torch.tensor([3, 130], dtype=torch.int32, device=dev)  # cache locs
ape2 = torch.randn(POOL, HD, device=dev, dtype=torch.float32)
kpc.kpool_decode_update_and_maybe_write_cache_batched(
    kv2.view(n_pages, page_size, HD + 4),
    tail,
    tail_slot,
    key,
    sscore,
    ape2,
    slot_map,
    positions,
    POOL,
)
for r, loc_expect in [(0, 3), (1, 130)]:
    lp, tk = loc_expect // page_size, loc_expect % page_size
    b = lp * kv2.view(n_pages, -1).stride(0) + tk * HD
    got = kv2.view(-1)[b : b + HD]
    nz = (got != 0).sum().item()
    sb = lp * kv2.view(n_pages, -1).stride(0) + page_size * HD + tk * 4
    gs = kv2.view(torch.float32)[sb // 4]
    print(f"[C] req{r} nonzero_k_bytes={nz} scale={gs.item():.4f} finite={torch.isfinite(gs).item()}")
    if nz < HD or not torch.isfinite(gs):
        fails.append(f"C{r}")

print("PROBE", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
