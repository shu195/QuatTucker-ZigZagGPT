"""
Quaternion-Tucker ZigZag Networks v2.0 (QuatTucker-ZigZagGPT)
RoPE + RMSNorm + FlashAttention + 4D Quaternion-Tucker SwiGLU + Exact Zero-Loss Depth Growth
Shubham Barmase (Independent Researcher), 2026 | DOI: 10.5281/zenodo.24424502
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def precompute_rope(hd: int, max_len: int = 512, theta: float = 10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    return torch.cos(freqs)[None, None, :, :], torch.sin(freqs)[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.size(2)
    c, s = cos[:, :, :T, :], sin[:, :, :T, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class RoPEFlashAttn(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads, self.hd = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = apply_rope(q.view(B, T, self.n_heads, self.hd).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, T, self.n_heads, self.hd).transpose(1, 2), cos, sin)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class QuatLinear(nn.Module):
    """4D Quaternion Hamilton Linear (4x fewer params, single fused bf16 GEMM)."""
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
    """4-layer Braided ZigZag Diamond (0 -> 2 -> 1 -> 3) with Shared 4D QuatTucker SwiGLU."""
    def __init__(self, d: int = 768, n_heads: int = 12, rank: int = 8192, blk: int = 64):
        super().__init__()
        assert rank % (4 * blk) == 0, "rank must be a multiple of 4*blk (256 for blk=64)"
        self.blk, self.n_blk = blk, rank // blk
        self.attns = nn.ModuleList([RoPEFlashAttn(d, n_heads) for _ in range(4)])
        self.ln1 = nn.ModuleList([RMSNorm(d) for _ in range(4)])
        self.ln2 = nn.ModuleList([RMSNorm(d) for _ in range(4)])
        self.U_gate = QuatLinear(d, rank)
        self.U_val  = QuatLinear(d, rank)
        self.V_out  = QuatLinear(rank, d)
        eye = torch.eye(blk).unsqueeze(0).repeat(4, 1, 1)
        self.G = nn.Parameter((eye + 0.05 * torch.randn(4, blk, blk)) * (rank ** -0.5))

    def zero_init_residual_outputs_(self):
        """Exact identity initialization for zero-loss depth growth (max_diff = 0.000000)."""
        with torch.no_grad():
            for l in range(4):
                self.attns[l].proj.weight.zero_()
            self.V_out.wr.zero_(); self.V_out.wi.zero_()
            self.V_out.wj.zero_(); self.V_out.wk.zero_()

    def _block(self, x: torch.Tensor, l: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = x + self.attns[l](self.ln1[l](x), cos, sin)
        xn = self.ln2[l](x)
        z = F.silu(self.U_gate(xn)) * self.U_val(xn)
        z_mix = (z.view(B, T, self.n_blk, self.blk) @ self.G[l]).view(B, T, -1)
        return x + self.V_out(z_mix)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h0 = self._block(x, 0, cos, sin)
        h2 = self._block(h0, 2, cos, sin)
        h1 = self._block(0.5 * (h0 + h2), 1, cos, sin)
        h3 = self._block(0.5 * (h1 + h2), 3, cos, sin)
        return h3


class QuatTuckerZigZagGPT(nn.Module):
    """QuatTucker-ZigZagGPT v2.0 (default: L=16 / 4 Diamonds, 95.3M total / 56.7M backbone)."""
    def __init__(self, vocab_size: int = 50257, d_model: int = 768, n_heads: int = 12,
                 n_diamonds: int = 4, rank: int = 8192, blk: int = 64, max_len: int = 512):
        super().__init__()
        self.d_model, self.n_heads, self.rank, self.blk = d_model, n_heads, rank, blk
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.diamonds = nn.ModuleList([
            QuatTuckerZigZagDiamond(d_model, n_heads, rank, blk) for _ in range(n_diamonds)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        cos, sin = precompute_rope(d_model // n_heads, max_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def grow_depth(self):
        """Append a new 4-layer Braided ZigZag Diamond with exact zero-loss preservation."""
        new_d = QuatTuckerZigZagDiamond(self.d_model, self.n_heads, self.rank, self.blk).to(self.emb.weight.device)
        new_d.zero_init_residual_outputs_()
        self.diamonds.append(new_d)
        return self

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        for d in self.diamonds:
            x = d(x, self.cos, self.sin)
        return self.head(self.ln_f(x))
