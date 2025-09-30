import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.temb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.temb(temb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    """A tiny UNet for DDPM on small images (1 or 3 channels)."""

    def __init__(self, in_ch: int = 1, base_ch: int = 64, ch_mult=(1, 2, 2), temb_dim: int = 256):
        super().__init__()
        self.in_ch = in_ch
        self.temb_dim = temb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(64, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        chs = [base_ch * m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down
        self.down1 = ConvBlock(base_ch, chs[0], temb_dim)
        self.down2 = ConvBlock(chs[0], chs[1], temb_dim)
        self.down3 = ConvBlock(chs[1], chs[2], temb_dim)
        self.downsample1 = nn.Conv2d(chs[0], chs[0], 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(chs[1], chs[1], 3, stride=2, padding=1)

        # Middle
        self.mid1 = ConvBlock(chs[2], chs[2], temb_dim)
        self.mid2 = ConvBlock(chs[2], chs[2], temb_dim)

        # Up
        self.upsample1 = nn.ConvTranspose2d(chs[2], chs[1], 4, stride=2, padding=1)
        self.up1 = ConvBlock(chs[1] + chs[1], chs[1], temb_dim)
        self.upsample2 = nn.ConvTranspose2d(chs[1], chs[0], 4, stride=2, padding=1)
        self.up2 = ConvBlock(chs[0] + chs[0], chs[0], temb_dim)

        self.out_norm = nn.GroupNorm(32, chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(chs[0], in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # time embedding
        temb = timestep_embedding(t, 64)
        temb = self.time_mlp(temb)

        # in
        x0 = self.in_conv(x)

        # down
        d1 = self.down1(x0, temb)
        d1s = self.downsample1(d1)
        d2 = self.down2(d1s, temb)
        d2s = self.downsample2(d2)
        d3 = self.down3(d2s, temb)

        # mid
        m = self.mid1(d3, temb)
        m = self.mid2(m, temb)

        # up
        u1 = self.upsample1(m)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1, temb)
        u2 = self.upsample2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up2(u2, temb)

        out = self.out_conv(self.out_act(self.out_norm(u2)))
        return out
