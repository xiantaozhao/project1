"""
Conditional UNet for DOLCE model
Based on DOLCE architecture with FBP/RLS conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1D Tensor of N indices, one per batch element
        dim: Dimension of output
        max_period: Controls minimum frequency of embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    """Residual block with time embedding and optional conditioning."""
    
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb(F.silu(t))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum("bhqd,bhkd->bhqk", q, k) * scale, dim=-1)
        h = torch.einsum("bhqk,bhvd->bhqd", attn, v)
        
        # Reshape back
        h = h.transpose(-2, -1).reshape(B, C, H, W)
        h = self.proj_out(h)
        
        return x + h


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """
    Conditional UNet for DOLCE.
    Supports FBP and/or RLS reconstruction conditioning.
    
    Args:
        in_channels: Number of input channels (1 for CT)
        out_channels: Number of output channels (1 for CT)
        model_channels: Base channel count
        num_res_blocks: Number of residual blocks per resolution
        channel_mult: Channel multiplier for each resolution level
        attention_resolutions: Resolutions to apply attention at
        dropout: Dropout probability
        use_fbp_condition: Whether to use FBP conditioning
        use_rls_condition: Whether to use RLS conditioning
        condition_dropout: Probability of dropping condition during training
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 4),
        attention_resolutions=(8, 16),
        dropout=0.0,
        use_fbp_condition=True,
        use_rls_condition=False,
        condition_dropout=0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.use_fbp_condition = use_fbp_condition
        self.use_rls_condition = use_rls_condition
        self.condition_dropout = condition_dropout
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Calculate input channels (image + conditions)
        input_ch = in_channels
        if use_fbp_condition:
            input_ch += 1
        if use_rls_condition:
            input_ch += 1
            
        # Input convolution
        self.input_conv = nn.Conv2d(input_ch, model_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        input_block_channels = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_embed_dim, dropout)]
                ch = out_ch
                
                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=1))
                    
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(ch)
                
            if level != len(channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
                input_block_channels.append(ch)
                ds *= 2
            else:
                self.down_samples.append(None)
                
        # Middle blocks
        self.middle_block1 = ResBlock(ch, ch, time_embed_dim, dropout)
        self.middle_attention = AttentionBlock(ch, num_heads=1)
        self.middle_block2 = ResBlock(ch, ch, time_embed_dim, dropout)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                # Skip connection from downsampling
                skip_ch = input_block_channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_embed_dim, dropout)]
                ch = out_ch
                
                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=1))
                    
                self.up_blocks.append(nn.ModuleList(layers))
                
                if i == num_res_blocks and level != 0:
                    self.up_samples.append(Upsample(ch))
                    ds //= 2
                else:
                    self.up_samples.append(None)
                    
        # Output convolution
        self.output_norm = nn.GroupNorm(32, ch)
        self.output_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
    def forward(self, x, timesteps, condition_fbp=None, condition_rls=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            timesteps: Timestep tensor (B,)
            condition_fbp: FBP reconstruction condition (B, 1, H, W) or None
            condition_rls: RLS reconstruction condition (B, 1, H, W) or None
            
        Returns:
            Output tensor (B, C, H, W)
        """
        # Time embedding
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Concatenate conditions
        inputs = [x]
        
        # Apply condition dropout during training
        if self.training and self.condition_dropout > 0:
            if condition_fbp is not None and torch.rand(1) < self.condition_dropout:
                condition_fbp = torch.zeros_like(condition_fbp)
            if condition_rls is not None and torch.rand(1) < self.condition_dropout:
                condition_rls = torch.zeros_like(condition_rls)
        
        if self.use_fbp_condition and condition_fbp is not None:
            inputs.append(condition_fbp)
        if self.use_rls_condition and condition_rls is not None:
            inputs.append(condition_rls)
            
        h = torch.cat(inputs, dim=1)
        h = self.input_conv(h)
        
        # Downsampling
        hs = [h]
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for layer in blocks:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            hs.append(h)
            
            if downsample is not None:
                h = downsample(h)
                hs.append(h)
                
        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attention(h)
        h = self.middle_block2(h, t_emb)
        
        # Upsampling
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            # Skip connection
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            
            for layer in blocks:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
                    
            if upsample is not None:
                h = upsample(h)
                
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h


def create_conditional_unet(**kwargs):
    """Factory function to create ConditionalUNet with default parameters."""
    defaults = {
        'in_channels': 1,
        'out_channels': 1,
        'model_channels': 128,
        'num_res_blocks': 2,
        'channel_mult': (1, 2, 2, 4),
        'attention_resolutions': (8, 16),
        'dropout': 0.0,
        'use_fbp_condition': True,
        'use_rls_condition': False,
        'condition_dropout': 0.1,
    }
    defaults.update(kwargs)
    return ConditionalUNet(**defaults)
