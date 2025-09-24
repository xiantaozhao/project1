# File: src/model/unet_basic.py
"""
A minimal supervised UNet (no diffusion, no timestep) for image-to-image regression.
Use case: limited-angle FBP (input) -> full-angle target (output), trained with L1/L2.

- GroupNorm + SiLU (small-batch friendly)
- Downsample via stride-2 conv; Upsample via transposed conv
- Classic UNet skip connections
- Default: in_channels=1, out_channels=1

Input  : x in [B, C, H, W]  (e.g., C=1, H=W=256)
Output : y in [B, C, H, W]
Note    : Ensure H and W are divisible by 2**num_levels (default: 4 levels -> divisible by 16).
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# helpers
# ------------------------------

def gn(ch: int, groups: int = 32) -> nn.GroupNorm:
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            gn(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            gn(out_ch), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)

# ------------------------------
# UNet (supervised baseline)
# ------------------------------
class UNetBasic(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Encoder
        ch = base_channels
        self.in_conv = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        enc_channels: List[int] = []

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(DoubleConv(ch, out_ch, dropout))
            ch = out_ch
            enc_channels.append(ch)
            if i != len(channel_mults) - 1:
                self.downsamples.append(Down(ch))

        # Bottleneck
        self.mid = DoubleConv(ch, ch, dropout)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i, mult in list(enumerate(channel_mults))[::-1]:
            if i == 0:
                break
            self.upsamples.append(Up(ch))
            skip_ch = enc_channels[i - 1]
            self.up_blocks.append(DoubleConv(ch + skip_ch, skip_ch, dropout))
            ch = skip_ch

        self.out_head = nn.Sequential(
            gn(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        h = self.in_conv(x)
        for i, block in enumerate(self.down_blocks):
            h = block(h)
            feats.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid(h)

        # decode (reverse levels)
        for i in range(len(self.up_blocks)):
            h = self.upsamples[i](h)
            skip = feats[-(i + 2)]  # align with previous encoder level
            # handle any odd size due to rounding
            if h.shape[-2:] != skip.shape[-2:]:
                dh = skip.shape[-2] - h.shape[-2]
                dw = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, (0, max(dw, 0), 0, max(dh, 0)))
                skip = F.pad(skip, (0, max(-dw, 0), 0, max(-dh, 0)))
            h = torch.cat([h, skip], dim=1)
            h = self.up_blocks[i](h)

        return self.out_head(h)

if __name__ == "__main__":
    # quick shape test
    net = UNetBasic(in_channels=1, out_channels=1, base_channels=64, channel_mults=(1,2,4,8))
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = net(x)
    print("in:", x.shape, "out:", y.shape)
