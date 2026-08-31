"""Probe donor triton MQA logits kernels vs bf16 torch oracle on A800."""
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "mlt", "/w/_port/donor/mqa_logits_triton.py"
)
mlt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlt)

torch.manual_seed(7)
dev = "cuda"
fails = []

# ---------- non-paged ----------
M, N, H, D = 33, 517, 16, 128
q = torch.randn(M, H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k = torch.randn(N, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
ks = torch.rand(N, device=dev, dtype=torch.float32) + 0.5
w = torch.randn(M, H, device=dev, dtype=torch.float32)
ks_i = torch.randint(0, 100, (M,), device=dev, dtype=torch.int32)
ke = (ks_i + 200).clamp(max=N).to(torch.int32)

got = mlt.fp8_mqa_logits_triton(q, (k, ks), w, ks_i, ke, clean_logits=False)
qb, kb = q.to(torch.bfloat16), k.to(torch.bfloat16)  # kernel runs bf16 tl.dot
s = torch.einsum("mhd,nd->mhn", qb, kb).float() * ks[None, None, :]
s = torch.relu(s)  # kernel line 339: positive-gating semantics
ref = (s * w[:, :, None]).sum(dim=1)  # k_scale applied once (kernel line 332)
mask = (torch.arange(N, device=dev)[None, :] >= ks_i[:, None]) & (
    torch.arange(N, device=dev)[None, :] < ke[:, None]
)
ref = ref * mask
valid = mask
d = (got - ref).abs()
d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
err = d.max().item() / ref.abs().max().item()
print(f"[paged=off] rel_err_max={err:.4e}")
if err > 0.02:
    fails.append("nonpaged")

# ---------- paged (cache layout: [nb, bs, 1, D+4] uint8) ----------
B, NN, nb, bs = 2, 3, 64, 64
q2 = torch.randn(B, NN, H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
cache = torch.zeros(nb, bs, 1, D + 4, dtype=torch.uint8, device=dev)
# fill pages 0..1 with random k + scales (per page: bs*D bytes then bs*4B scales)
kflat = torch.randn(2 * bs, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
sflat = torch.rand(2 * bs, device=dev, dtype=torch.float32) + 0.5
for p in range(2):
    pg = cache[p].view(-1)
    pg[: bs * D] = kflat[p * bs : (p + 1) * bs].view(torch.uint8).reshape(-1)
    pg[bs * D :].view(torch.float32).copy_(sflat[p * bs : (p + 1) * bs])
ctx = torch.tensor([50, 120], device=dev, dtype=torch.int32)
bt = torch.zeros(B, 8, dtype=torch.int32, device=dev)
bt[0, 0] = 0
bt[1, 0] = 0
bt[1, 1] = 1
w2 = torch.randn(B * NN, H, device=dev, dtype=torch.float32)
got2 = mlt.fp8_paged_mqa_logits_triton(
    q2, cache, w2, ctx, bt, 128, clean_logits=False
)
ref2 = torch.zeros(B * NN, 128, device=dev, dtype=torch.float32)
for b in range(B):
    kk = kflat[: min(ctx[b].item(), bs)]
    if ctx[b] > bs:
        kk = torch.cat([kk, kflat[bs : ctx[b]]])
    qb2 = q2[b].to(torch.bfloat16)  # [NN,H,D] bf16 like kernel
    s2 = (
        torch.einsum("thd,nd->thn", qb2, kk.to(torch.bfloat16)).float()
        * sflat[: ctx[b]][None, None, :]
    )
    s2 = torch.relu(s2)
    lo = (s2 * w2[b * NN : (b + 1) * NN, :, None]).sum(dim=1)
    ref2[b * NN : (b + 1) * NN, : ctx[b]] = lo
v2 = torch.zeros_like(ref2, dtype=torch.bool)
for b in range(B):
    for t in range(NN):
        # kernel causal contract: row t attends cols <= ctx-NN+t
        v2[b * NN + t, : ctx[b] - NN + t + 1] = True
v2f = v2
n_inf_valid = int((torch.isinf(got2) & v2f).sum().item())
print(f"[paged] infs inside valid region: {n_inf_valid} (total inf={torch.isinf(got2).sum().item()})")
if n_inf_valid:
    iv = (torch.isinf(got2) & v2f).nonzero()[:6]
    for r, c in iv.tolist():
        print(f"  INF at row={r} col={c}")
d2 = (got2 - ref2).abs()
d2 = torch.where(torch.isfinite(d2), d2, torch.zeros_like(d2))
err2 = d2.max().item() / ref2.abs().max().item()
print(f"[paged] rel_err_max={err2:.4e}")
if err2 > 0.02:
    fails.append("paged")


# ---- per-head decomposition at worst-diff cell (nonpaged) ----
dn = (got - ref).abs()
dn = torch.where(torch.isfinite(dn), dn, torch.zeros_like(dn))
m0, n0 = divmod(torch.argmax(dn).item(), got.shape[1])
print(f"DECOMP cell m={m0} n={n0}: got={got[m0,n0].item():.4f} ref={ref[m0,n0].item():.4f}")
s_cell = torch.relu(s[m0, :, n0]) * w[m0, :]  # per-head relu*scale*w
for hi in range(1, H + 1):
    part = s_cell[:hi].sum().item()
    if abs(part - got[m0, n0].item()) < 0.01 * max(1.0, abs(got[m0, n0].item())):
        print(f"  got == sum(heads[0:{hi}])  ({part:.4f})   <-- {H-hi} heads missing")
        break
else:
    print("  no contiguous prefix matches; per-head:", [round(x, 3) for x in s_cell.tolist()])
print("LOGITS PROBE", "PASS" if not fails else f"FAIL {fails}")

# ---- diagnostics ----
print("health: got nan/inf:", torch.isnan(got).sum().item(), torch.isinf(got).sum().item(),
      "| ref nan/inf:", torch.isnan(ref).sum().item(), torch.isinf(ref).sum().item(),
      "| got2 nan/inf:", torch.isnan(got2).sum().item(), torch.isinf(got2).sum().item(),
      "| ref2 nan/inf:", torch.isnan(ref2).sum().item(), torch.isinf(ref2).sum().item())
gd = (got2 - ref2).abs()
bidx = torch.argmax(torch.where(v2 & torch.isfinite(gd), gd, torch.zeros_like(gd)))
r, c = divmod(bidx.item(), got2.shape[1])
print(f"[diag paged] worst at row={r} n={c} got={got2[r,c].item():.4f} ref={ref2[r,c].item():.4f}")
d = (got - ref).abs()
dm = torch.where(valid & torch.isfinite(d), d, torch.zeros_like(d))
bidx = torch.argmax(dm)
m, n = divmod(bidx.item(), got.shape[1])
print(f"[diag nonpaged] worst at m={m} n={n} got={got[m,n].item():.4f} ref={ref[m,n].item():.4f} ks={ks_i[m].item()} ke={ke[m].item()}")

sys.exit(0 if not fails else 1)
