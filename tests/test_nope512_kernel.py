import math
import torch

from vllm.v1.attention.ops.triton_mla_sparse_kernel import triton_mla_sparse_attention

assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability()
print("GPU", torch.cuda.get_device_name(), "CC", cap)
assert cap == (8, 0), f"expected SM80 for this validation, got {cap}"

torch.manual_seed(0)
device = "cuda"
T, H, D, S, TOPK = 1, 16, 512, 64, 64
q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
kv = torch.randn(S, 1, D, device=device, dtype=torch.bfloat16)
indices = torch.full((T, 1, TOPK), -1, device=device, dtype=torch.int32)
valid = torch.randperm(S, device=device)[:40]
indices[0, 0, :40] = valid.to(torch.int32)
scale = 1.0 / math.sqrt(D)

def ref():
    k = kv[valid, 0].float()
    scores = q[0].float() @ k.T * scale
    p = torch.softmax(scores, dim=-1)
    return (p @ k[:, :512]).unsqueeze(0)

expected = ref()
for splits in (1, 2, 4, None):
    out = triton_mla_sparse_attention(
        q, kv, indices, sm_scale=scale, num_kv_splits=splits
    ).float()
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)
    assert not torch.isnan(out).any()
    print("PASS split", splits)

# All-invalid row must be zero, never NaN.
indices.fill_(-1)
out = triton_mla_sparse_attention(q, kv, indices, sm_scale=scale).float()
assert not torch.isnan(out).any()
assert torch.count_nonzero(out) == 0
print("NOPE512_KERNEL=PASS")
