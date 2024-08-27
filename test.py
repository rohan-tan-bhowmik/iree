import torch
import numpy as np

# softmax(Q * K^t) * V
if __name__ == "__main__":
    m = 4096
    k2 = 64
    n = 64
    b0 = 1
    b1 = 1
    q = (torch.rand(b0,b1,m,n).to(torch.float32) - 0.5) * 2
    k = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 3
    v = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 4
    mask = (torch.rand(b0,b1,m,k2).to(torch.float32) - 0.5) * 0
    # mask = torch.triu(torch.ones(b0, b1, m, k2), diagonal=1)
    # mask[mask == 1] = True #float('-inf')
    # mask[mask == 0] = False #0.0
    zeros = (torch.rand(b0,b1,m,k2).to(torch.float32) - 0.5) * 0
    mask = zeros

    np.save("attn_q.npy", q.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_k.npy", k.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_v.npy", v.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_mask.npy", mask.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_zeros.npy", zeros.detach().to(dtype=torch.float16, device="cpu").numpy())

    # Post attention func
    out = torch._scaled_dot_product_flash_attention_for_cpu(q, k, v, attn_mask=mask).output
    # out = torch._scaled_dot_product_flash_attention_for_cpu(q, k, v, is_causal=True).output
    np.save("attn_ref.npy", out.detach().to(dtype=torch.float32, device="cpu").numpy())