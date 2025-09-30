# src/utils/layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_act(config):
    """获取激活函数"""
    act_name = getattr(config.model, 'activation', 'swish')
    if act_name.lower() in ['swish', 'silu']:
        return nn.SiLU()
    elif act_name.lower() == 'relu':
        return nn.ReLU()
    elif act_name.lower() == 'gelu':
        return nn.GELU()
    else:
        return nn.SiLU()  # 默认使用SiLU


def default_init():
    """默认权重初始化函数"""
    def _init(weight):
        return torch.nn.init.xavier_uniform_(weight)
    return _init


def ddpm_conv3x3(in_planes: int, out_planes: int, stride: int = 1, bias: bool = True, init_scale: float = 1.0) -> nn.Conv2d:
    """3x3卷积层"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
    if init_scale != 1.0:
        with torch.no_grad():
            conv.weight *= init_scale
    return conv


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    时间步嵌入，使用正弦位置编码
    
    Args:
        timesteps: shape [batch_size]
        embedding_dim: 嵌入维度
    
    Returns:
        embeddings: shape [batch_size, embedding_dim]
    """
    assert len(timesteps.shape) == 1  # [batch_size]
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # 奇数维度，零填充
        emb = F.pad(emb, (0, 1))
    
    return emb


class AttnBlock(nn.Module):
    """自注意力块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # 计算注意力
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # [b, hw, c]
        k = k.reshape(b, c, h * w)  # [b, c, hw]
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # [b, hw, c]
        
        attn = torch.bmm(q, k) * (int(c) ** (-0.5))  # [b, hw, hw]
        attn = F.softmax(attn, dim=2)
        
        h_ = torch.bmm(attn, v)  # [b, hw, c]
        h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)  # [b, c, h, w]
        
        h_ = self.proj_out(h_)
        return x + h_


class ResnetBlockDDPM(nn.Module):
    """DDPM风格的残差块"""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: Optional[int] = None,
        temb_dim: Optional[int] = None,
        dropout: float = 0.0,
        act: nn.Module = nn.SiLU()
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch or in_ch
        self.temb_dim = temb_dim
        
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, self.out_ch, kernel_size=3, padding=1)
        
        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, self.out_ch)
        
        self.norm2 = nn.GroupNorm(32, self.out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1)
        
        self.act = act
        
        if in_ch != self.out_ch:
            self.shortcut = nn.Conv2d(in_ch, self.out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        if temb is not None and self.temb_dim is not None:
            temb = self.act(temb)
            temb = self.temb_proj(temb)[:, :, None, None]
            h = h + temb
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class Upsample(nn.Module):
    """上采样层"""
    
    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """下采样层"""
    
    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            return self.conv(x)
        else:
            return F.avg_pool2d(x, kernel_size=2, stride=2)


class RefineBlock(nn.Module):
    """优化块（可选使用）"""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """基础残差块"""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.act = nn.SiLU()
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.shortcut(x)