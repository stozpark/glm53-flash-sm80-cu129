import torch

from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton

assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability()
assert cap == (8, 0), f"expected SM80, got {cap}"
torch.manual_seed(1)

device = "cuda"
M, N, H, D = 8, 256, 32, 128
q_bf16 = torch.randn(M, H, D, dtype=torch.bfloat16, device=device)
k_bf16 = torch.randn(N, D, dtype=torch.bfloat16, device=device)
q = q_bf16.to(torch.float8_e4m3fn)
amax = k_bf16.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4)
kscale = (amax / 448.0).squeeze(-1)
k = (k_bf16.float() / kscale[:, None]).to(torch.float8_e4m3fn)
weights = torch.randn(M, H, dtype=torch.float32, device=device)
ks = torch.arange(M, dtype=torch.int32, device=device) * 3
ke = torch.minimum(ks + 97, torch.full_like(ks, N))

out = fp8_mqa_logits_triton(q, (k, kscale), weights, ks, ke, clean_logits=False)
q_ref = q.to(torch.bfloat16)
k_ref = k.to(torch.bfloat16)
score = torch.einsum("mhd,nd->hmn", q_ref, k_ref).float() * kscale
ref = (score.relu() * weights.T.unsqueeze(-1)).sum(dim=0)
ar = torch.arange(N, device=device)[None, :]
mask = (ar >= ks[:, None]) & (ar < ke[:, None])
ref = ref.masked_fill(~mask, float("-inf"))

# Critical correctness property for clean_logits=False: masked positions are -inf.
assert torch.equal(torch.isneginf(out), torch.isneginf(ref))
finite = torch.isfinite(ref)
torch.testing.assert_close(out[finite], ref[finite], atol=1.0, rtol=0.2)
print("MQA_PREFILL_SM80=PASS")
