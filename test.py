import torch
import numpy as np
import random

# softmax(Q * K^t) * V
if __name__ == "__main__":
    m = 4096
    k2 = 64
    n = 64
    b0 = 1
    b1 = 1
    q = (torch.rand(b0,b1,m,n).to(torch.float32) - 0.5) * 1
    k = (torch.rand(b0,b1,k2,n).to(torch.float32) - 0.5) * 1
    v = torch.rand(b0,b1,k2,n).to(torch.float32) * 1
    # v = torch.eye(64).reshape(b0,b1,k2,n)

    mask = (torch.rand(b0,b1,m,k2).to(torch.float32) - 0.5) * 1

    # rowdiv = 1
    # coldiv = 64
    # for i in range(m//rowdiv):
    #     for j in range(k2//coldiv):
    #         tile = (torch.ones(b0,b1,rowdiv,coldiv).to(torch.float32)) * random.random()
    #         mask[:,:,i*rowdiv:(i+1)*rowdiv,j*coldiv:(j+1)*coldiv] = tile

    # print(mask)

    # tile = torch.arange(0,32).reshape(2,16).repeat(8, 1)
    # print(tile)
    # for i in range(32):
    #     tile = np.where(tile == i, i, tile)
    # mask = torch.tensor(np.tile(tile, (b0,b1,m//16,k2//16))).to(torch.float32)
    # print(mask.shape)


    # mask = torch.triu(torch.ones(b0, b1, m, k2), diagonal=1)
    # mask[mask == 1] = True #float('-inf')
    # mask[mask == 0] = False #0.0
    zeros = (torch.rand(b0,b1,m,k2).to(torch.float32) - 0.5) * 0

    np.save("attn_q.npy", q.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_k.npy", k.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_v.npy", v.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_mask.npy", mask.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_zeros.npy", zeros.detach().to(dtype=torch.float16, device="cpu").numpy())


    # Post attention func
    out = torch._scaled_dot_product_flash_attention_for_cpu(q, k, v, attn_mask=mask).output
    expected = torch.softmax(mask, dim=-1)

    print("we here")
    print(out)
    print(torch.softmax(q @ k.transpose(-2, -1) + mask, dim=-1)@v)

    np.save("attn_ref.npy", out.detach().to(dtype=torch.float32, device="cpu").numpy())