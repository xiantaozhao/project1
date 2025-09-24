# File: src/model/unet.py
"""
UNet backbone for diffusion models (predicts noise [epsilon]).
- Sinusoidal time embedding + MLP projection
- Residual blocks with time conditioning
- Optional self-attention at chosen resolutions
- Downsample via strided conv; upsample via transposed conv

Input:  x  -> [B, C, H, W]
Timestep: t -> [B] long/int in [0..T-1]
Output: same shape as x (noise prediction)
"""
from __future__ import annotations
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utils
# ------------------------------

def gn(ch: int, groups: int = 32) -> nn.GroupNorm:
    # clamp groups so that groups | ch
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)

# ------------------------------
# Sinusoidal timestep embedding
# ------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Classic transformer-style positional embedding adapted for timesteps.
    Returns a vector of size `dim`, then a small MLP to mix features.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B], has integer or float timesteps in [0, T-1]
        if t.dtype not in (torch.float32, torch.float64):
            t = t.float()
        half = self.dim // 2
        # log-spaced frequencies (safer range than strict 1..10000)
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device))
        angles = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)

# ------------------------------
# Building blocks
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # PreNorm -> Conv -> add time -> Norm -> Act -> Drop -> Conv
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)

class AttnBlock(nn.Module):
    """Channel-wise multi-head self-attention over spatial tokens.
    Keeps spatial dims intact and preserves channels.
    """
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # each [B, C, H, W]
        # reshape to heads
        head_dim = C // self.heads
        q = q.reshape(B, self.heads, head_dim, H * W)
        k = k.reshape(B, self.heads, head_dim, H * W)
        v = v.reshape(B, self.heads, head_dim, H * W)
        # attention: (B,heads,N,D) @ (B,heads,D,N) -> (B,heads,N,N)
        scale = head_dim ** -0.5
        attn = torch.softmax((q.transpose(-2, -1) @ k) * scale, dim=-1)
        # (B,heads,N,N) @ (B,heads,N,D) -> (B,heads,N,D)
        out = attn @ v.transpose(-2, -1)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(out)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)

# ------------------------------
# UNet
# ------------------------------
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 128,
        channel_mults: List[int] | tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        time_embed_dim: int = 512,
        use_attention: List[bool] | tuple[bool, ...] | None = None,
        attn_heads: int = 4,
        dropout: float = 0.0,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        if use_attention is None:
            use_attention = [False] * len(channel_mults)
        assert len(use_attention) == len(channel_mults), "use_attention must align with channel_mults"

        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # --- Down path ---
        self.down = nn.ModuleList()
        self.skip_channels: List[int] = []  # record channels after each ResBlock
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResidualBlock(ch, out_ch, time_embed_dim, dropout))
                ch = out_ch
                if use_attention[i]:
                    self.down.append(AttnBlock(ch, attn_heads))
                self.skip_channels.append(ch)
            if i != len(channel_mults) - 1:
                self.down.append(Downsample(ch))

        # --- Middle ---
        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttnBlock(ch, attn_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout),
        ])

        # --- Up path ---
        self.up = nn.ModuleList()
        for i, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up.append(ResidualBlock(ch + skip_ch, out_ch, time_embed_dim, dropout))
                ch = out_ch
                if use_attention[i]:
                    self.up.append(AttnBlock(ch, attn_heads))
            if i != 0:
                self.up.append(Upsample(ch))

        self.out_norm = gn(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels or in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_emb(t)  # [B, time_embed_dim]
        hs: List[torch.Tensor] = []

        h = self.in_conv(x)
        for mod in self.down:
            if isinstance(mod, ResidualBlock):
                h = mod(h, temb)
                hs.append(h)
            else:  # Attn or Downsample
                h = mod(h)

        for mod in self.mid:
            if isinstance(mod, ResidualBlock):
                h = mod(h, temb)
            else:
                h = mod(h)

        for mod in self.up:
            if isinstance(mod, ResidualBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = mod(h, temb)
            else:  # Attn or Upsample
                h = mod(h)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)

if __name__ == "__main__":
    # quick shape test
    net = UNet(in_channels=1, base_channels=64, channel_mults=(1,2,2), num_res_blocks=2, use_attention=(False, True, False))
    x = torch.randn(2, 1, 128, 128)
    t = torch.randint(0, 1000, (2,))
    with torch.no_grad():
        y = net(x, t)
    print("in:", x.shape, "out:", y.shape)
