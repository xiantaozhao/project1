# src/utils/model_utils.py
import torch
import numpy as np
from typing import Dict, Any, Union, Optional


# 全局模型注册表
_MODELS = {}


def register_model(name: str):
    """模型注册装饰器"""
    def _register(cls):
        if name in _MODELS:
            raise ValueError(f"Model {name} already registered")
        _MODELS[name] = cls
        return cls
    return _register


def get_model_by_name(name: str):
    """根据名称获取模型类"""
    if name not in _MODELS:
        raise ValueError(f"Model {name} not found. Available: {list(_MODELS.keys())}")
    return _MODELS[name]


def get_sigmas(config) -> np.ndarray:
    """
    根据配置生成sigma序列（用于噪声调度）
    """
    diffusion_config = config.diffusion
    
    # 获取扩散步数
    num_timesteps = getattr(diffusion_config, 'steps', 1000)
    
    # 获取beta调度类型
    beta_schedule = getattr(diffusion_config, 'beta_schedule', 'cosine')
    beta_start = getattr(diffusion_config, 'beta_start', 1e-4)
    beta_end = getattr(diffusion_config, 'beta_end', 0.02)
    
    # 生成beta序列
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps)
    elif beta_schedule == "cosine":
        betas = cosine_beta_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")
    
    # 计算alpha和sigma
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    # sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
    sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
    
    return sigmas


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """
    余弦beta调度
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    从序列a中提取索引t对应的值，并调整形状以便广播
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def make_ddpm_config(config_dict: Dict[str, Any]) -> Any:
    """
    将字典配置转换为属性访问格式
    """
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    
    return Config(config_dict)


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(config) -> torch.device:
    """获取设备"""
    if hasattr(config, 'project'):
        device_name = getattr(config.project, 'device', 'cuda')
    elif isinstance(config, dict):
        device_name = config.get('project', {}).get('device', 'cuda')
    else:
        device_name = 'cuda'
    
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """加载检查点"""
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint