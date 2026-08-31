"""Boundary scan: does the nonpaged logits kernel drop heads at BLOCK_H boundaries?
Clean settings each time: w=+1, ks=0, ke=N (full range), clean=True."""
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("mlt", "/w/donor/mqa_logits_triton.py")
mlt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlt)

torch.manual_seed(3)
dev = "cuda"
M, N, D = 8, 256, 128
k = torch.randn(N, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
ks = torch.rand(N, device=dev, dtype=torch.float32) + 0.5
ks_i = torch.zeros(M, device=dev, dtype=torch.int32)
ke = torch.full((M,), N, device=dev, dtype=torch.int32)

fails = []
for H in (4, 8, 16, 32):
    q = torch.randn(M, H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.ones(M, H, device=dev, dtype=torch.float32)
    got = mlt.fp8_mqa_logits_triton(q, (k, ks), w, ks_i, ke, clean_logits=True)
    s = (
        torch.einsum("mhd,nd->mhn", q.to(torch.bfloat16), k.to(torch.bfloat16)).float()
        * ks
    )
    s = torch.relu(s)
    ref = (s * w[:, :, None]).sum(1)
    d = (got - ref).abs()
    rel = d.max().item() / ref.abs().max().item()
    m0, n0 = divmod(torch.argmax(d).item(), got.shape[1])
    # try to find which head-subset of the oracle matches got at the worst cell
    s_cell = s[m0, :, n0]  # [H] per-head contributions (relu, ks applied)
    got_cell = got[m0, n0].item()
    match = "n/a"
    for lo in range(0, H + 1):
        for hi in range(lo, H + 1):
            if abs(s_cell[lo:hi].sum().item() - got_cell) < 1e-2 * max(
                1.0, abs(got_cell)
            ):
                match = f"sum(h[{lo}:{hi}])"
                break
        if match != "n/a":
            break
    status = "OK " if rel < 0.01 else "BAD"
    if rel >= 0.01:
        fails.append(H)
    print(
        f"[{status}] H={H:2d} rel_err={rel:.3e} worst(m{n0 and ''}={m0},n={n0}) "
        f"got={got_cell:.3f} ref={ref[m0, n0].item():.3f} match={match}"
    )

print("SCAN", "PASS" if not fails else f"FAIL at H={fails}")
sys.exit(0 if not fails else 1)
