# src/train/train_ddpm.py
from __future__ import annotations
import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到路径
def _add_repo_root_to_syspath():
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

_add_repo_root_to_syspath()

from src.configs.configloading import load_config
from src.data.dataset_ddpm import DDPMChestDataset, create_chest_ddpm_datasets
from src.model.ddpm import DDPM
from src.utils.model_utils import make_ddpm_config, get_device, save_checkpoint, count_parameters
from src.utils.diffusion_utils import DiffusionScheduler
from src.utils.logging_utils import CSVLogger


class EMAModel:
    """指数移动平均模型"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """更新EMA参数"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        """应用EMA参数到模型"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """恢复原始参数"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def expand_var(template: str, **kwargs) -> str:
    """展开配置中的变量"""
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"${{{key}}}", str(value))
    return result


def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def compute_ddpm_loss(
    model: nn.Module,
    x: torch.Tensor,
    scheduler: DiffusionScheduler,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算DDPM训练损失
    
    Args:
        model: DDPM模型
        x: 输入图像 [B, C, H, W]，范围[0,1]
        scheduler: 扩散调度器
        device: 设备
    
    Returns:
        loss: 损失值
        metrics: 指标字典
    """
    batch_size = x.shape[0]
    
    # 随机采样时间步
    t = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
    
    # 生成随机噪声
    noise = torch.randn_like(x)
    
    # 前向扩散：添加噪声
    noisy_x = scheduler.add_noise(x, noise, t)
    
    # 模型预测噪声
    pred_noise = model(noisy_x, t)
    
    # 计算MSE损失
    loss = F.mse_loss(pred_noise, noise)
    
    # 计算指标
    with torch.no_grad():
        mse = F.mse_loss(pred_noise, noise, reduction='mean')
        mae = F.l1_loss(pred_noise, noise, reduction='mean')
    
    metrics = {
        'mse': mse.item(),
        'mae': mae.item(),
    }
    
    return loss, metrics


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    scheduler: DiffusionScheduler,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """验证模型"""
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for i, batch in enumerate(pbar):
            if max_batches is not None and i >= max_batches:
                break
                
            x = batch['image'].to(device)
            
            loss, metrics = compute_ddpm_loss(model, x, scheduler, device)
            
            total_loss += loss.item()
            total_mse += metrics['mse']
            total_mae += metrics['mae']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{metrics['mse']:.4f}"
            })
    
    model.train()
    
    return {
        'val_loss': total_loss / num_batches,
        'val_mse': total_mse / num_batches,
        'val_mae': total_mae / num_batches,
    }


def save_sample_images(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    device: torch.device,
    save_path: Path,
    num_samples: int = 8,
    image_size: int = 256,
    num_inference_steps: int = 50,
    use_ddim: bool = True
):
    """生成并保存样本图像"""
    model.eval()
    
    with torch.no_grad():
        # 生成随机噪声
        shape = (num_samples, 1, image_size, image_size)
        x = torch.randn(shape, device=device)
        
        if use_ddim:
            # 使用DDIM采样（更快，质量更好）
            from src.utils.diffusion_utils import DDIMScheduler
            ddim_scheduler = DDIMScheduler(
                num_train_timesteps=scheduler.num_train_timesteps,
                num_inference_steps=num_inference_steps,
                beta_schedule="cosine",
                eta=0.1  # 添加少量随机性提升质量
            )
            ddim_scheduler = ddim_scheduler.to(device)
            ddim_scheduler.set_timesteps(num_inference_steps)
            
            timesteps = ddim_scheduler.timesteps
            for t in tqdm(timesteps, desc="DDIM Sampling", leave=False):
                t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                pred_noise = model(x, t_tensor)
                x, _ = ddim_scheduler.step(pred_noise, t, x)
        else:
            # 使用完整DDPM采样（1000步）
            timesteps = list(range(scheduler.num_train_timesteps))[::-1]
            for t in tqdm(timesteps, desc="DDPM Sampling", leave=False):
                t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                pred_noise = model(x, t_tensor)
                x, _ = scheduler.step(pred_noise, t, x)
        
        # 转换为图像格式并保存
        x = torch.clamp(x, 0, 1)
        x = x.cpu().numpy()
        
        # 保存为网格图像
        save_grid_images(x, save_path, nrow=4)
    
    model.train()


def save_grid_images(images: np.ndarray, save_path: Path, nrow: int = 4):
    """保存图像网格"""
    try:
        import matplotlib.pyplot as plt
        
        n_images = images.shape[0]
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
                    img = images[idx, 0]  # 去掉通道维度
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].set_title(f'Sample {idx+1}')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Warning: matplotlib not available, skipping sample image saving")


def main():
    # 加载配置
    config_path = "configs/ddpm/chest.yaml"
    config_dict = load_config(config_path, default_path=None)
    config = make_ddpm_config(config_dict)
    
    # 设备
    device = get_device(config)
    print(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建输出目录
    dataset_name = config.data.dataset_name
    output_dir = Path(expand_var(config.training.output_dir, **{"data.dataset_name": dataset_name}))
    checkpoint_dir = Path(expand_var(config.training.checkpoint_dir, **{"training.output_dir": str(output_dir)}))
    sample_dir = Path(expand_var(config.training.sample_dir, **{"training.output_dir": str(output_dir)}))
    log_dir = Path(expand_var(config.training.log_dir, **{"training.output_dir": str(output_dir)}))
    
    ensure_dir(checkpoint_dir)
    ensure_dir(sample_dir)
    ensure_dir(log_dir)
    
    print(f"Output directory: {output_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Samples: {sample_dir}")
    print(f"Logs: {log_dir}")
    
    # 创建数据集
    print("Creating datasets...")
    
    # 准备数据集参数
    dataset_params = {
        'image_size': config.data.image_size,
        'use_mu': config.data.use_mu,
        'mu_water': config.data.mu_water,
        'hu_clip_range': config.data.hu_clip_range,
        'data_augmentation': config.data.data_augmentation,
        'random_flip': config.data.random_flip,
        'random_rot90': config.data.random_rot90,
        'cache_volumes': config.data.cache_volumes
    }
    
    # 检查是否手动指定了患者分组
    if hasattr(config.data, 'train_patient_ids') and hasattr(config.data, 'val_patient_ids'):
        # 手动指定模式
        train_dataset, val_dataset, test_dataset = create_chest_ddpm_datasets(
            train_patient_ids=config.data.train_patient_ids,
            val_patient_ids=config.data.val_patient_ids,
            test_patient_ids=getattr(config.data, 'test_patient_ids', []),
            **dataset_params
        )
    else:
        # 自动划分模式
        train_dataset, val_dataset, test_dataset = create_chest_ddpm_datasets(
            patient_ids=config.data.patient_ids,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            **dataset_params
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=True if config.data.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=True if config.data.num_workers > 0 else False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # 创建模型
    print("Creating model...")
    model = DDPM(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 创建扩散调度器
    scheduler = DiffusionScheduler(
        num_train_timesteps=config.diffusion.steps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=getattr(config.diffusion, 'beta_start', 1e-4),
        beta_end=getattr(config.diffusion, 'beta_end', 0.02)
    )
    
    # 将调度器移动到正确的设备
    scheduler = scheduler.to(device)
    
    # 优化器
    if config.optimizer.name.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas
        )
    
    # 学习率调度器
    if config.scheduler.name.lower() == 'cosine':
        scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.max_epochs,
            eta_min=config.optimizer.lr * 0.01
        )
        
        # 预热
        if config.scheduler.warmup_steps > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=config.scheduler.warmup_steps
            )
    else:
        scheduler_lr = None
        warmup_scheduler = None
    
    # EMA模型
    ema_model = None
    if config.training.ema.use_ema:
        ema_model = EMAModel(model, decay=config.training.ema.decay)
    
    # 混合精度训练
    scaler = GradScaler(enabled=config.training.amp and device.type == 'cuda')
    
    # 日志记录
    writer = SummaryWriter(log_dir / "tensorboard")
    csv_logger = CSVLogger(log_dir / "train_log.csv")
    
    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(1, config.training.max_epochs + 1):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_mae = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.max_epochs}")
        for batch in pbar:
            x = batch['image'].to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=config.training.amp and device.type == 'cuda'):
                loss, metrics = compute_ddpm_loss(model, x, scheduler, device)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新EMA
            if ema_model is not None:
                ema_model.update(model)
            
            # 更新学习率（预热阶段）
            if warmup_scheduler is not None and global_step < config.scheduler.warmup_steps:
                warmup_scheduler.step()
            
            # 记录指标
            epoch_loss += loss.item()
            epoch_mse += metrics['mse']
            epoch_mae += metrics['mae']
            num_batches += 1
            global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # 记录到TensorBoard
            if global_step % config.training.log_interval == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/mse', metrics['mse'], global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
        
        # 更新学习率调度器（主调度器）
        if scheduler_lr is not None and (warmup_scheduler is None or global_step >= config.scheduler.warmup_steps):
            scheduler_lr.step()
        
        # 验证阶段
        if epoch % config.training.val_interval == 0:
            val_metrics = validate_model(model, val_loader, scheduler, device, max_batches=50)
            
            # 记录验证指标
            writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
            writer.add_scalar('val/mse', val_metrics['val_mse'], epoch)
            
            print(f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}")
            
            # 保存最佳模型
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                
                # 如果使用EMA，保存EMA模型
                if ema_model is not None:
                    ema_model.apply_shadow(model)
                
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=best_val_loss,
                    filepath=str(checkpoint_dir / "best.pth"),
                    config=config_dict,
                    global_step=global_step
                )
                
                if ema_model is not None:
                    ema_model.restore(model)
                
                print(f"Saved best model with val_loss = {best_val_loss:.4f}")
        
        # 保存检查点
        if epoch % config.training.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=epoch_loss / num_batches,
                filepath=str(checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"),
                config=config_dict,
                global_step=global_step
            )
        
        # 生成样本图像
        if epoch % (config.training.save_interval * 2) == 0:
            sample_path = sample_dir / f"samples_epoch_{epoch}.png"
            
            # 使用EMA模型生成样本
            if ema_model is not None:
                ema_model.apply_shadow(model)
            
            save_sample_images(
                model=model,
                scheduler=scheduler,
                device=device,
                save_path=sample_path,
                num_samples=8,
                image_size=config.data.image_size,
                num_inference_steps=50,  # 使用DDIM加速采样
                use_ddim=True
            )
            
            if ema_model is not None:
                ema_model.restore(model)
        
        # 记录到CSV
        csv_logger.log({
            'epoch': epoch,
            'train_loss': epoch_loss / num_batches,
            'train_mse': epoch_mse / num_batches,
            'train_mae': epoch_mae / num_batches,
            'lr': optimizer.param_groups[0]['lr'],
            'global_step': global_step
        })
        
        print(f"Epoch {epoch} completed: Loss = {epoch_loss / num_batches:.4f}")
    
    # 保存最终模型
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.training.max_epochs,
        loss=epoch_loss / num_batches,
        filepath=str(checkpoint_dir / "final.pth"),
        config=config_dict,
        global_step=global_step
    )
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()