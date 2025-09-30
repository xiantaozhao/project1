# src/data/dataset_ddpm.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from src.data.data_load import data_load_chest


def hu_to_mu(hu: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
    """
    HU转换为线衰减系数mu
    
    Args:
        hu: HU值数组
        mu_water: 水的线衰减系数 (mm^-1)
    
    Returns:
        mu: 线衰减系数数组
    """
    return mu_water * (1.0 + hu.astype(np.float32) / 1000.0)


def normalize_to_01(img: np.ndarray, clip_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    将图像归一化到 [0, 1] 范围
    
    Args:
        img: 输入图像
        clip_range: 可选的裁剪范围 (min, max)
    
    Returns:
        归一化后的图像
    """
    if clip_range is not None:
        img = np.clip(img, clip_range[0], clip_range[1])
    
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)


class DDPMChestDataset(Dataset):
    """
    DDPM训练数据集，从CT体数据中提取2D切片
    
    特点：
    1. 自动从多个patient加载CT体数据
    2. 将HU转换为mu值
    3. 逐切片归一化到[0,1]
    4. 支持数据增强（翻转、旋转）
    """
    
    def __init__(
        self,
        patient_ids: List[Union[str, int]],
        image_size: int = 256,
        use_mu: bool = True,
        mu_water: float = 0.02,
        hu_clip_range: Optional[Tuple[float, float]] = (-1000, 1000),
        data_augmentation: bool = True,
        random_flip: bool = True,
        random_rot90: bool = False,
        cache_volumes: bool = True,
    ):
        """
        Args:
            patient_ids: 患者ID列表，如 ["1", "2", "3"] 或 [1, 2, 3]
            image_size: 目标图像尺寸（正方形）
            use_mu: 是否转换HU到mu值
            mu_water: 水的线衰减系数
            hu_clip_range: HU值裁剪范围
            data_augmentation: 是否启用数据增强
            random_flip: 是否随机翻转
            random_rot90: 是否随机90度旋转
            cache_volumes: 是否缓存加载的体数据
        """
        self.patient_ids = [str(pid) for pid in patient_ids]
        self.image_size = image_size
        self.use_mu = use_mu
        self.mu_water = mu_water
        self.hu_clip_range = hu_clip_range
        self.data_augmentation = data_augmentation
        self.random_flip = random_flip
        self.random_rot90 = random_rot90
        self.cache_volumes = cache_volumes
        
        # 缓存
        self._volume_cache = {} if cache_volumes else None
        
        # 构建切片索引：(patient_id, slice_idx)
        self.slice_indices = []
        self._build_slice_indices()
        
        print(f"[DDPMChestDataset] Loaded {len(self.patient_ids)} patients, "
              f"total {len(self.slice_indices)} slices")
    
    def _build_slice_indices(self):
        """构建所有切片的索引"""
        for patient_id in self.patient_ids:
            try:
                vol_shape = self._get_volume_shape(patient_id)
                num_slices = vol_shape[0]  # Z维度
                
                # 添加该患者的所有切片索引
                for slice_idx in range(num_slices):
                    self.slice_indices.append((patient_id, slice_idx))
                    
                print(f"[DDPMChestDataset] Patient {patient_id}: {num_slices} slices")
                
            except Exception as e:
                print(f"[DDPMChestDataset] Warning: Skipping patient {patient_id}, error: {e}")
    
    def _get_volume_shape(self, patient_id: str) -> Tuple[int, int, int]:
        """获取体数据形状（不加载完整数据）"""
        if self.cache_volumes and patient_id in self._volume_cache:
            return self._volume_cache[patient_id].shape
        
        # 快速加载获取形状
        vol_HU_zyx, _, _ = data_load_chest.load_data_chest(patient_id, "CT")
        return vol_HU_zyx.shape
    
    def _load_volume(self, patient_id: str) -> np.ndarray:
        """加载并处理体数据"""
        if self.cache_volumes and patient_id in self._volume_cache:
            return self._volume_cache[patient_id]
        
        # 加载原始HU数据
        vol_HU_zyx, spacing_dzyx, meta = data_load_chest.load_data_chest(patient_id, "CT")
        
        # HU裁剪
        if self.hu_clip_range is not None:
            vol_HU_zyx = np.clip(vol_HU_zyx, self.hu_clip_range[0], self.hu_clip_range[1])
        
        # HU转mu
        if self.use_mu:
            vol_processed = hu_to_mu(vol_HU_zyx, self.mu_water)
        else:
            vol_processed = vol_HU_zyx.astype(np.float32)
        
        # 缓存
        if self.cache_volumes:
            self._volume_cache[patient_id] = vol_processed
        
        return vol_processed
    
    def _resize_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """调整切片尺寸到目标大小"""
        if slice_2d.shape[0] == self.image_size and slice_2d.shape[1] == self.image_size:
            return slice_2d
        
        # 使用最近邻插值调整尺寸（保持医学图像特性）
        try:
            import cv2
            resized = cv2.resize(slice_2d, (self.image_size, self.image_size), 
                               interpolation=cv2.INTER_LINEAR)
            return resized
        except ImportError:
            # 如果没有cv2，使用简单的双线性插值
            from scipy import ndimage
            zoom_factor = self.image_size / min(slice_2d.shape)
            resized = ndimage.zoom(slice_2d, (zoom_factor, zoom_factor), order=1)
            
            # 裁剪或填充到精确尺寸
            h, w = resized.shape
            if h > self.image_size or w > self.image_size:
                start_h = (h - self.image_size) // 2
                start_w = (w - self.image_size) // 2
                resized = resized[start_h:start_h+self.image_size, 
                                start_w:start_w+self.image_size]
            elif h < self.image_size or w < self.image_size:
                padded = np.zeros((self.image_size, self.image_size), dtype=resized.dtype)
                start_h = (self.image_size - h) // 2
                start_w = (self.image_size - w) // 2
                padded[start_h:start_h+h, start_w:start_w+w] = resized
                resized = padded
                
            return resized
    
    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        if not self.data_augmentation:
            return img
        
        # 随机水平翻转
        if self.random_flip and random.random() < 0.5:
            img = np.fliplr(img)
        
        # 随机垂直翻转
        if self.random_flip and random.random() < 0.5:
            img = np.flipud(img)
        
        # 随机90度旋转
        if self.random_rot90 and random.random() < 0.5:
            k = random.randint(1, 3)  # 旋转90, 180, 或270度
            img = np.rot90(img, k)
        
        return img
    
    def __len__(self) -> int:
        return len(self.slice_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个切片的数据
        
        Returns:
            dict: {
                'image': torch.Tensor shape [1, H, W], 归一化到[0,1]
                'patient_id': str
                'slice_idx': int
            }
        """
        patient_id, slice_idx = self.slice_indices[idx]
        
        # 加载体数据并提取切片
        volume = self._load_volume(patient_id)
        slice_2d = volume[slice_idx]  # shape: [H, W]
        
        # 调整尺寸
        slice_2d = self._resize_slice(slice_2d)
        
        # 归一化到[0,1]
        slice_2d = normalize_to_01(slice_2d)
        
        # 数据增强
        slice_2d = self._apply_augmentation(slice_2d)
        
        # 确保数组是连续的（修复负步长问题）
        if not slice_2d.flags['C_CONTIGUOUS']:
            slice_2d = slice_2d.copy()
        
        # 转换为PyTorch张量并添加通道维度
        image = torch.from_numpy(slice_2d).float().unsqueeze(0)  # [1, H, W]
        
        return {
            'image': image,
            'patient_id': patient_id,
            'slice_idx': slice_idx
        }
    
    def get_slice_info(self, idx: int) -> Dict[str, Any]:
        """获取切片信息（用于调试）"""
        patient_id, slice_idx = self.slice_indices[idx]
        volume_shape = self._get_volume_shape(patient_id)
        
        return {
            'idx': idx,
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'volume_shape': volume_shape,
            'total_slices_this_patient': volume_shape[0]
        }


def create_chest_ddpm_datasets(
    patient_ids: List[Union[str, int]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_patient_ids: List[Union[str, int]] = None,
    val_patient_ids: List[Union[str, int]] = None,
    test_patient_ids: List[Union[str, int]] = None,
    **dataset_kwargs
) -> Tuple[DDPMChestDataset, DDPMChestDataset, DDPMChestDataset]:
    """
    创建训练、验证、测试数据集
    
    Args:
        patient_ids: 患者ID列表 (自动划分模式)
        train_ratio: 训练集比例 (自动划分模式)
        val_ratio: 验证集比例 (自动划分模式)
        test_ratio: 测试集比例 (自动划分模式)
        train_patient_ids: 手动指定训练集患者ID
        val_patient_ids: 手动指定验证集患者ID  
        test_patient_ids: 手动指定测试集患者ID
        **dataset_kwargs: 传递给DDPMChestDataset的参数
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    
    # 优先使用手动指定的患者分组
    if train_patient_ids is not None and val_patient_ids is not None:
        train_ids = [str(pid) for pid in train_patient_ids]
        val_ids = [str(pid) for pid in val_patient_ids]
        test_ids = [str(pid) for pid in test_patient_ids] if test_patient_ids else []
        
        print(f"[create_chest_ddpm_datasets] Manual dataset split:")
        print(f"  Training set: {len(train_ids)} patients - {train_ids}")
        print(f"  Validation set: {len(val_ids)} patients - {val_ids}")
        print(f"  Test set: {len(test_ids)} patients - {test_ids}")
        
    else:
        # 使用自动划分模式
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, validation, and test ratios must sum to 1"
        
        # 打乱患者ID
        patient_ids = list(patient_ids)
        random.shuffle(patient_ids)
        
        n_total = len(patient_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_ids = patient_ids[:n_train]
        val_ids = patient_ids[n_train:n_train+n_val]
        test_ids = patient_ids[n_train+n_val:]
        
        print(f"[create_chest_ddpm_datasets] Automatic dataset split:")
        print(f"  Training set: {len(train_ids)} patients - {train_ids}")
        print(f"  Validation set: {len(val_ids)} patients - {val_ids}")
        print(f"  Test set: {len(test_ids)} patients - {test_ids}")
    
    # 创建数据集（验证和测试集不使用数据增强）
    train_kwargs = dataset_kwargs.copy()
    train_kwargs['data_augmentation'] = train_kwargs.get('data_augmentation', True)
    
    val_test_kwargs = dataset_kwargs.copy()
    val_test_kwargs['data_augmentation'] = False
    
    train_dataset = DDPMChestDataset(train_ids, **train_kwargs)
    val_dataset = DDPMChestDataset(val_ids, **val_test_kwargs)
    test_dataset = DDPMChestDataset(test_ids, **val_test_kwargs)
    
    return train_dataset, val_dataset, test_dataset


# 测试代码
if __name__ == "__main__":
    # 测试数据集
    print("Testing DDPMChestDataset...")
    
    # 使用少量患者进行测试
    test_patient_ids = ["1", "2"]
    
    dataset = DDPMChestDataset(
        patient_ids=test_patient_ids,
        image_size=256,
        use_mu=True,
        data_augmentation=True,
        cache_volumes=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试几个样本
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        info = dataset.get_slice_info(i)
        
        print(f"\nSample {i}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"  Patient ID: {sample['patient_id']}")
        print(f"  Slice index: {sample['slice_idx']}")
        print(f"  Details: {info}")