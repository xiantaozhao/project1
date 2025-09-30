# src/utils/diffusion_utils.py
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    余弦beta调度，产生更平滑的噪声添加过程
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    线性beta调度
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionScheduler:
    """
    扩散过程调度器，管理前向扩散和反向去噪过程
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        
        # 生成beta序列
        if beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == "linear":
            self.betas = linear_beta_schedule(num_train_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 计算alpha和alpha_cumprod
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # 计算用于前向扩散的sqrt值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 计算用于反向去噪的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 计算后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def to(self, device: torch.device):
        """将调度器的所有张量移动到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向扩散：给原始图像添加噪声
        
        Args:
            original_samples: 原始图像 [B, C, H, W]
            noise: 随机噪声 [B, C, H, W]
            timesteps: 时间步 [B]
        
        Returns:
            noisy_samples: 添加噪声后的图像 [B, C, H, W]
        """
        # 确保调度器参数在正确的设备上
        device = original_samples.device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha_prod = sqrt_alphas_cumprod[timesteps].flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDPM单步去噪
        
        Args:
            model_output: 模型预测的噪声 [B, C, H, W]
            timestep: 当前时间步
            sample: 当前噪声图像 [B, C, H, W]
            generator: 随机数生成器
        
        Returns:
            prev_sample: 去噪后的图像 [B, C, H, W]
            pred_original_sample: 预测的原始图像 [B, C, H, W]
        """
        t = timestep
        
        # 1. 计算预测的原始样本 x0
        pred_original_sample = (
            sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
        ) / self.sqrt_alphas_cumprod[t]

        # 2. 使用 DDPM 的后验均值公式
        alpha_t = self.alphas[t]
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]

        # 等价形式：mu = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-\bar{alpha}_t) * eps)
        posterior_mean = (
            (sample - (beta_t / torch.sqrt(1 - alpha_prod_t)) * model_output)
            / torch.sqrt(alpha_t)
        )

        prev_sample = posterior_mean

        # 3. 添加噪声（除了最后一步）
        if t > 0:
            if generator is not None:
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            else:
                noise = torch.randn_like(sample)
            prev_sample = prev_sample + torch.sqrt(self.posterior_variance[t]) * noise
        
        return prev_sample, pred_original_sample


class DDIMScheduler:
    """
    DDIM调度器，支持确定性采样
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        eta: float = 0.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        # 生成beta序列
        if beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == "linear":
            self.betas = linear_beta_schedule(num_train_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 计算alpha和alpha_cumprod
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 设置推理时间步
        self.set_timesteps(num_inference_steps)
    
    def to(self, device: torch.device):
        """将调度器张量移动到目标设备（用于避免 CPU/CUDA 混用错误）。"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        # timesteps 本身是索引，保持在 CPU/GPU 都可，但为一致性也转到 device
        if hasattr(self, 'timesteps'):
            self.timesteps = self.timesteps.to(device)
        return self
    
    def set_timesteps(self, num_inference_steps: int):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps, dtype=torch.float) * step_ratio).round().long()
        self.timesteps = torch.flip(timesteps, [0])  # 从大到小
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM单步去噪
        
        Args:
            model_output: 模型预测的噪声 [B, C, H, W]
            timestep: 当前时间步
            sample: 当前噪声图像 [B, C, H, W]
            eta: 随机性系数，0表示完全确定性
            generator: 随机数生成器
        
        Returns:
            prev_sample: 去噪后的图像 [B, C, H, W]
            pred_original_sample: 预测的原始图像 [B, C, H, W]
        """
        if eta is None:
            eta = self.eta
        
        device = sample.device
        dtype = sample.dtype
        # 获取前一个时间步
        step_stride = self.num_train_timesteps // self.num_inference_steps
        prev_timestep = timestep - step_stride
        
        # 1. 计算alphas（保持与sample同设备/类型）
        alpha_prod_t = self.alphas_cumprod[timestep].to(device=device, dtype=dtype)
        if prev_timestep >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(device=device, dtype=dtype)
        else:
            alpha_prod_t_prev = torch.ones((), device=device, dtype=dtype)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # 2. 计算预测的原始样本
        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # 3. 计算方向方差
        variance = eta**2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = torch.clamp(variance, min=0.0)
        std_dev_t = torch.sqrt(variance)
        
        # 4. 计算方向指向xt的系数
        coeff = torch.clamp(1 - alpha_prod_t_prev - variance, min=0.0)
        pred_sample_direction = torch.sqrt(coeff) * model_output
        
        # 5. 计算前一时间步的样本
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        # 6. 添加噪声
        if eta > 0:
            if generator is not None:
                # 使用generator生成噪声
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            else:
                noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev_t * noise
        
        return prev_sample, pred_original_sample


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    从序列a中提取索引t对应的值，并调整形状以便广播
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)