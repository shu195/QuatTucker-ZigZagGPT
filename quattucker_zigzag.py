"""
Quaternion-Tucker ZigZag Networks (QuatTucker-ZigZagGPT)
Shubham Barmase (Independent Researcher), 2026
DOI: 10.5281/zenodo.24424502
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalFlashAttn(nn.Module):
    """Independent per-layer dense Causal FlashAttention."""
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads, self.hd = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class QuatLinear(nn.Module):
    """4D Quaternion Linear layer (4x fewer params, fused bf16 Tensor-Core Hamilton GEMM)."""
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        assert in_f % 4 == 0 and out_f % 4 == 0
        iq, oq = in_f // 4, out_f // 4
        s = in_f ** -0.5
        self.wr = nn.Parameter(torch.randn(oq, iq) * s)
        self.wi = nn.Parameter(torch.randn(oq, iq) * s)
        self.wj = nn.Parameter(torch.randn(oq, iq) * s)
        self.wk = nn.Parameter(torch.randn(oq, iq) * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r0 = torch.cat([self.wr, -self.wi, -self.wj, -self.wk], dim=1)
        r1 = torch.cat([self.wi,  self.wr, -self.wk,  self.wj], dim=1)
        r2 = torch.cat([self.wj,  self.wk,  self.wr, -self.wi], dim=1)
        r3 = torch.cat([self.wk, -self.wj,  self.wi,  self.wr], dim=1)
        return F.linear(x, torch.cat([r0, r1, r2, r3], dim=0))


class QuatTuckerZigZagDiamond(nn.Module):
    """4-layer Braided ZigZag Diamond (0 -> 2 -> 1 -> 3) with Shared 4D Quaternion SwiGLU-Tucker FFN."""
    def __init__(self, d: int = 384, n_heads: int = 6, rank: int = 4096, blk: int = 64):
        super().__init__()
        self.blk, self.n_blk = blk, rank // blk
        self.attns = nn.ModuleList([CausalFlashAttn(d, n_heads) for _ in range(4)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.U_gate = QuatLinear(d, rank)
        self.U_val  = QuatLinear(d, rank)
        self.V_out  = QuatLinear(rank, d)
        eye = torch.eye(blk).unsqueeze(0).repeat(4, 1, 1)
        self.G = nn.Parameter((eye + 0.05 * torch.randn(4, blk, blk)) * (rank ** -0.5))

    def _block(self, x: torch.Tensor, l: int) -> torch.Tensor:
        B, T, _ = x.shape
        x = x + self.attns[l](self.ln1[l](x))
        xn = self.ln2[l](x)
        z = F.silu(self.U_gate(xn)) * self.U_val(xn)
        z_mix = (z.view(B, T, self.n_blk, self.blk) @ self.G[l]).view(B, T, -1)
        return x + self.V_out(z_mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self._block(x, 0)
        h2 = self._block(h0, 2)
        h1 = self._block(0.5 * (h0 + h2), 1)
        h3 = self._block(0.5 * (h1 + h2), 3)
        return h3


class QuatTuckerZigZagGPT(nn.Module):
    """QuatTucker-ZigZagGPT Language Model (default: 8 layers = 2 stacked diamonds, ~26.5M total / 7.2M backbone)."""
    def __init__(self, vocab_size: int = 50257, max_seq_len: int = 256, d_model: int = 384,
                 n_heads: int = 6, n_diamonds: int = 2, rank: int = 4096, blk: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.diamonds = nn.ModuleList([
            QuatTuckerZigZagDiamond(d_model, n_heads, rank, blk) for _ in range(n_diamonds)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight  # Weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for d in self.diamonds:
            x = d(x)
        return self.head(self.ln_f(x))
