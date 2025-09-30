# src/utils/normalization.py
import torch
import torch.nn as nn
from typing import Optional


def get_normalization(config, num_channels: int) -> nn.Module:
    """
    根据配置获取归一化层
    
    Args:
        config: 配置对象
        num_channels: 通道数
    
    Returns:
        归一化层
    """
    norm_type = getattr(config.model, 'normalization', 'group')
    
    if norm_type.lower() == 'group':
        # Group Normalization，默认32个组
        num_groups = min(32, num_channels)
        if num_channels % num_groups != 0:
            # 调整组数以确保能整除
            for g in [16, 8, 4, 2, 1]:
                if num_channels % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups, num_channels, eps=1e-6)
    
    elif norm_type.lower() == 'batch':
        return nn.BatchNorm2d(num_channels, eps=1e-6)
    
    elif norm_type.lower() == 'instance':
        return nn.InstanceNorm2d(num_channels, eps=1e-6)
    
    elif norm_type.lower() == 'layer':
        return nn.LayerNorm([num_channels], eps=1e-6)
    
    elif norm_type.lower() == 'none':
        return nn.Identity()
    
    else:
        # 默认使用Group Normalization
        num_groups = min(32, num_channels)
        if num_channels % num_groups != 0:
            for g in [16, 8, 4, 2, 1]:
                if num_channels % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups, num_channels, eps=1e-6)


class GroupNorm32(nn.GroupNorm):
    """32组的Group Normalization"""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        num_groups = min(32, num_channels)
        if num_channels % num_groups != 0:
            # 调整组数以确保能整除
            for g in [16, 8, 4, 2, 1]:
                if num_channels % g == 0:
                    num_groups = g
                    break
        super().__init__(num_groups, num_channels, eps=eps)


class ConditionalGroupNorm(nn.Module):
    """条件Group Normalization（支持时间步条件）"""
    
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.num_groups = min(num_groups, num_channels)
        if num_channels % self.num_groups != 0:
            for g in [16, 8, 4, 2, 1]:
                if num_channels % g == 0:
                    self.num_groups = g
                    break
        
        self.eps = eps
        self.num_channels = num_channels
        
        # 默认的gamma和beta参数
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
            condition: 条件信息 [B, condition_dim] （可选）
        """
        # 基础Group Normalization
        B, C, H, W = x.shape
        x = x.view(B, self.num_groups, C // self.num_groups, H, W)
        
        # 计算均值和方差
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        
        # 标准化
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(B, C, H, W)
        
        # 应用学习的缩放和偏移
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        
        return x * weight + bias


class AdaptiveGroupNorm(nn.Module):
    """自适应Group Normalization（根据通道数自动调整组数）"""
    
    def __init__(self, num_channels: int, target_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        
        # 自动选择最佳组数
        self.num_groups = min(target_groups, num_channels)
        for g in [target_groups, 16, 8, 4, 2, 1]:
            if num_channels % g == 0:
                self.num_groups = g
                break
        
        self.group_norm = nn.GroupNorm(self.num_groups, num_channels, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm(x)


# 便捷函数
def group_norm(num_channels: int, num_groups: int = 32, eps: float = 1e-6) -> nn.GroupNorm:
    """创建Group Normalization层"""
    actual_groups = min(num_groups, num_channels)
    if num_channels % actual_groups != 0:
        for g in [16, 8, 4, 2, 1]:
            if num_channels % g == 0:
                actual_groups = g
                break
    return nn.GroupNorm(actual_groups, num_channels, eps=eps)


def batch_norm(num_channels: int, eps: float = 1e-6) -> nn.BatchNorm2d:
    """创建Batch Normalization层"""
    return nn.BatchNorm2d(num_channels, eps=eps)


def instance_norm(num_channels: int, eps: float = 1e-6) -> nn.InstanceNorm2d:
    """创建Instance Normalization层"""
    return nn.InstanceNorm2d(num_channels, eps=eps)