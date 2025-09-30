# scripts/infer_ddpm.py
#!/usr/bin/env python3
from __future__ import annotations
import sys
import argparse
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目根目录到路径
def _add_repo_root_to_syspath():
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

_add_repo_root_to_syspath()

from src.configs.configloading import load_config
from src.model.ddpm import DDPM
from src.utils.model_utils import make_ddpm_config, get_device, load_checkpoint
from src.utils.diffusion_utils import DiffusionScheduler, DDIMScheduler

# 可选的图像保存
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def save_images(
    images: np.ndarray,
    save_dir: Path,
    prefix: str = "sample",
    save_individual: bool = True,
    save_grid: bool = True,
    nrow: int = 4
):
    """
    保存生成的图像
    
    Args:
        images: 图像数组 [N, H, W] 或 [N, C, H, W]
        save_dir: 保存目录
        prefix: 文件名前缀
        save_individual: 是否保存单独的图像
        save_grid: 是否保存网格图像
        nrow: 网格行数
    """
    ensure_dir(save_dir)
    
    # 确保图像格式正确
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]  # 去掉单通道维度 [N, H, W]
    
    # 归一化到[0, 255]
    images = np.clip(images * 255, 0, 255).astype(np.uint8)
    
    # 保存单独图像
    if save_individual:
        for i, img in enumerate(images):
            save_path = save_dir / f"{prefix}_{i:03d}.png"
            
            if HAS_IMAGEIO:
                imageio.imwrite(save_path, img, format='PNG')
            elif HAS_MATPLOTLIB:
                plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
            else:
                # 保存为numpy文件
                np.save(save_dir / f"{prefix}_{i:03d}.npy", img)
    
    # 保存网格图像
    if save_grid and HAS_MATPLOTLIB:
        n_images = len(images)
        ncol = math.ceil(n_images / nrow)
        
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
        if nrow == 1:
            axes = axes.reshape(1, -1)
        elif ncol == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(nrow):
            for j in range(ncol):
                idx = i * ncol + j
                if idx < n_images:
                    axes[i, j].imshow(images[idx], cmap='gray')
                    axes[i, j].set_title(f'Sample {idx+1}')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_grid.png", dpi=150, bbox_inches='tight')
        plt.close()


def ddpm_sample(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    shape: Tuple[int, ...],
    device: torch.device,
    num_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    show_progress: bool = True
) -> torch.Tensor:
    """
    DDPM采样生成图像
    
    Args:
        model: 训练好的DDPM模型
        scheduler: 扩散调度器
        shape: 生成图像的形状 (batch_size, channels, height, width)
        device: 设备
        num_steps: 采样步数，None表示使用完整步数
        generator: 随机数生成器
        show_progress: 是否显示进度条
    
    Returns:
        生成的图像 [B, C, H, W]，范围[0, 1]
    """
    model.eval()
    
    # 初始化为纯噪声
    x = torch.randn(shape, device=device, generator=generator)
    
    # 采样步数
    timesteps = list(range(scheduler.num_train_timesteps))[::-1]  # 从T-1到0
    if num_steps is not None:
        step_size = scheduler.num_train_timesteps // num_steps
        timesteps = timesteps[::step_size]
    
    # 反向扩散过程
    iterator = tqdm(timesteps, desc="DDPM Sampling") if show_progress else timesteps
    
    with torch.no_grad():
        for t in iterator:
            # 创建时间步张量
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 模型预测噪声
            pred_noise = model(x, t_tensor)
            
            # 去噪一步
            x, _ = scheduler.step(pred_noise, t, x, generator=generator)
    
    # 限制到[0, 1]范围
    x = torch.clamp(x, 0.0, 1.0)
    
    return x


def ddim_sample(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    shape: Tuple[int, ...],
    device: torch.device,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
    show_progress: bool = True
) -> torch.Tensor:
    """
    DDIM采样生成图像（更快，可确定性）
    
    Args:
        model: 训练好的DDPM模型
        scheduler: DDIM调度器
        shape: 生成图像的形状
        device: 设备
        num_inference_steps: 推理步数
        eta: 随机性参数，0表示确定性采样
        generator: 随机数生成器
        show_progress: 是否显示进度条
    
    Returns:
        生成的图像 [B, C, H, W]，范围[0, 1]
    """
    model.eval()
    
    # 设置推理步数
    scheduler.set_timesteps(num_inference_steps)
    
    # 初始化为纯噪声
    x = torch.randn(shape, device=device, generator=generator)
    
    # 反向扩散过程
    iterator = tqdm(scheduler.timesteps, desc="DDIM Sampling") if show_progress else scheduler.timesteps
    
    with torch.no_grad():
        for t in iterator:
            # 创建时间步张量
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 模型预测噪声
            pred_noise = model(x, t_tensor)
            
            # DDIM去噪一步
            x, _ = scheduler.step(pred_noise, t, x, eta=eta, generator=generator)
    
    # 限制到[0, 1]范围
    x = torch.clamp(x, 0.0, 1.0)
    
    return x


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Any,
    device: torch.device
) -> torch.nn.Module:
    """从检查点加载模型"""
    # 创建模型
    model = DDPM(config).to(device)
    
    # 加载权重
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="DDPM图像生成推理脚本")
    parser.add_argument("--config", type=str, default="configs/ddpm/chest.yaml",
                        help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="outputs/ddpm/chest/checkpoints/best.pth",
                        help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认使用配置中的sample_dir）")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="生成样本数量")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="推理批大小")
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim",
                        help="采样器类型")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="采样步数")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM随机性参数（0=确定性）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备：cuda/cpu/auto")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    
    # 加载配置
    print(f"Loading config from {args.config}")
    config_dict = load_config(args.config, default_path=None)
    config = make_ddpm_config(config_dict)
    
    # 设备
    if args.device == "auto":
        device = get_device(config)
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 输出目录
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        dataset_name = config.data.dataset_name
        sample_dir = config.training.sample_dir.replace("${training.output_dir}", 
                                                        config.training.output_dir)
        sample_dir = sample_dir.replace("${data.dataset_name}", dataset_name)
        output_dir = Path(sample_dir) / "inference"
    
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    model.eval()
    
    # 创建调度器
    if args.sampler == "ddpm":
        scheduler = DiffusionScheduler(
            num_train_timesteps=config.diffusion.steps,
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.get('beta_start', 1e-4),
            beta_end=config.diffusion.get('beta_end', 0.02)
        )
    else:  # ddim
        scheduler = DDIMScheduler(
            num_train_timesteps=config.diffusion.steps,
            num_inference_steps=args.num_steps,
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.get('beta_start', 1e-4),
            beta_end=config.diffusion.get('beta_end', 0.02),
            eta=args.eta
        )
    # 与模型设备一致
    if hasattr(scheduler, 'to'):
        scheduler = scheduler.to(device)
    
    # 生成图像
    print(f"Generating {args.num_samples} samples using {args.sampler.upper()}")
    
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        start_idx = i * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.num_samples)
        current_batch_size = end_idx - start_idx
        
        # 生成形状
        shape = (current_batch_size, config.data.num_channels, 
                config.data.image_size, config.data.image_size)
        
        print(f"Batch {i+1}/{num_batches}: generating {current_batch_size} samples")
        
        if args.sampler == "ddpm":
            samples = ddpm_sample(
                model=model,
                scheduler=scheduler,
                shape=shape,
                device=device,
                num_steps=args.num_steps,
                generator=generator,
                show_progress=True
            )
        else:  # ddim
            samples = ddim_sample(
                model=model,
                scheduler=scheduler,
                shape=shape,
                device=device,
                num_inference_steps=args.num_steps,
                eta=args.eta,
                generator=generator,
                show_progress=True
            )
        
        # 转换为numpy并收集
        samples_np = samples.cpu().numpy()
        all_samples.append(samples_np)
    
    # 合并所有样本
    all_samples = np.concatenate(all_samples, axis=0)
    print(f"Generated {len(all_samples)} samples with shape {all_samples.shape}")
    
    # 保存图像
    print("Saving images...")
    save_images(
        images=all_samples,
        save_dir=output_dir,
        prefix=f"{args.sampler}_samples",
        save_individual=True,
        save_grid=True,
        nrow=4
    )
    
    # 保存样本统计信息
    stats = {
        'num_samples': len(all_samples),
        'shape': all_samples.shape,
        'mean': float(all_samples.mean()),
        'std': float(all_samples.std()),
        'min': float(all_samples.min()),
        'max': float(all_samples.max()),
        'sampler': args.sampler,
        'num_steps': args.num_steps,
        'eta': args.eta,
        'seed': args.seed
    }
    
    # 保存统计信息
    import json
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Generation completed! Images saved to {output_dir}")
    print(f"Sample statistics: mean={stats['mean']:.3f}, std={stats['std']:.3f}")


if __name__ == "__main__":
    main()