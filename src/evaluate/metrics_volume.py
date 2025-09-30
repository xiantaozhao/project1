# src/evaluate/metrics_volume.py
"""
体积数据质量评估模块

提供SSIM和PSNR指标的计算功能，用于评估3D医学影像重建质量。
支持逐切片计算和结果保存。
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import csv
from typing import Dict, Optional, Tuple, Union
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def _check_shapes(gt: np.ndarray, rec: np.ndarray) -> Tuple[int, int, int]:
    """
    检查输入数组的形状是否合法
    
    Args:
        gt: 真值数组，期望形状为[S,H,W]
        rec: 重建数组，期望形状为[S,H,W]
        
    Returns:
        数组形状元组(S, H, W)
        
    Raises:
        ValueError: 当数组不是3D或形状不匹配时
    """
    if gt.ndim != 3 or rec.ndim != 3:
        raise ValueError(f"Expected 3D arrays [S,H,W], got gt{gt.shape}, rec{rec.shape}")
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: gt{gt.shape} vs rec{rec.shape}")
    return gt.shape


def _stem_from_cfg(cfg: Optional[Dict], case_id: Union[str, int]) -> str:
    """
    根据配置生成文件名前缀
    
    Args:
        cfg: 配置字典，包含投影角度信息
        case_id: 案例标识符
        
    Returns:
        格式化的文件名前缀，如 "case1_180@1"
    """
    # 提取投影角度配置
    projection_cfg = (cfg or {}).get("projection", {})
    angles_cfg = projection_cfg.get("angles", {})
    stop_deg = angles_cfg.get("stop_deg", 180)
    step_deg = angles_cfg.get("step_deg", 1)
    
    def format_number(x):
        """格式化数字：整数显示为整数，浮点数保持原样"""
        try:
            x_float = float(x)
            x_int = int(round(x_float))
            return str(x_int) if abs(x_float - x_int) < 1e-6 else str(x)
        except (ValueError, TypeError):
            return str(x)
    
    # 处理案例ID
    case_str = str(case_id if case_id not in (None, "", "None") else "nocase")
    
    return f"{case_str}_{format_number(stop_deg)}@{format_number(step_deg)}"

def _normalize_arrays(gt: np.ndarray, rec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对数组进行全局min-max归一化到[0,1]范围
    
    Args:
        gt: 真值数组
        rec: 重建数组
        
    Returns:
        归一化后的(gt_normalized, rec_normalized)元组
    """
    # 转换为float32类型以节省内存
    gt = gt.astype(np.float32, copy=False)
    rec = rec.astype(np.float32, copy=False)
    
    # 计算全局最值
    gt_min, gt_max = float(gt.min()), float(gt.max())
    rec_min, rec_max = float(rec.min()), float(rec.max())
    
    # 归一化真值数组
    if gt_max > gt_min:
        gt_normalized = (gt - gt_min) / (gt_max - gt_min)
    else:
        # 如果真值数组为常数，设为0.5（中间值）
        gt_normalized = np.full_like(gt, 0.5)
    
    # 归一化重建数组
    if rec_max > rec_min:
        rec_normalized = (rec - rec_min) / (rec_max - rec_min)
    else:
        # 如果重建数组为常数，设为0.5（中间值）
        rec_normalized = np.full_like(rec, 0.5)
    
    return gt_normalized, rec_normalized


def _compute_slice_metrics(gt_slice: np.ndarray, rec_slice: np.ndarray) -> Tuple[float, float]:
    """
    计算单个切片的SSIM和PSNR
    
    Args:
        gt_slice: 真值切片，已归一化到[0,1]
        rec_slice: 重建切片，已归一化到[0,1]
        
    Returns:
        (ssim_value, psnr_value)元组
    """
    # 计算SSIM
    ssim_value = ssim(
        gt_slice, rec_slice,
        data_range=1.0,                    # 数据范围[0,1]
        gaussian_weights=False,            # 不使用高斯权重
        use_sample_covariance=True,        # 使用样本协方差
        win_size=None                      # 使用默认窗口大小
    )
    
    # 计算PSNR
    psnr_value = psnr(gt_slice, rec_slice, data_range=1.0)
    
    return float(ssim_value), float(psnr_value)


def _save_results(
    ssim_list: np.ndarray, 
    psnr_list: np.ndarray, 
    save_dir: Path, 
    stem: str, 
    case_id: Union[str, int], 
    shape: Tuple[int, int, int]
) -> None:
    """
    保存计算结果到文件
    
    Args:
        ssim_list: 每切片SSIM值数组
        psnr_list: 每切片PSNR值数组
        save_dir: 保存目录
        stem: 文件名前缀
        case_id: 案例标识符
        shape: 数据形状(S,H,W)
    """
    # 创建输出目录
    result_dir = save_dir / f"result_{case_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存SSIM结果
    with open(result_dir / f"ssim_{stem}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "ssim"])
        for i, value in enumerate(ssim_list):
            writer.writerow([i, float(value)])
    
    # 保存PSNR结果
    with open(result_dir / f"psnr_{stem}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "psnr"])
        for i, value in enumerate(psnr_list):
            writer.writerow([i, float(value)])
    
    # 保存汇总信息
    S, H, W = shape
    ssim_mean = float(np.mean(ssim_list))
    psnr_mean = float(np.mean(psnr_list))
    
    with open(result_dir / f"summary_{stem}.txt", "w") as f:
        f.write(f"Case: {case_id}\n")
        f.write(f"Shape: {S}x{H}x{W}\n")
        f.write(f"SSIM mean: {ssim_mean:.6f}\n")
        f.write(f"PSNR mean: {psnr_mean:.6f}\n")
        f.write("Normalization: global min-max to [0,1]\n")
        f.write("SSIM settings: data_range=1.0, gaussian_weights=False\n")
        f.write("PSNR settings: data_range=1.0\n")
    
    print(f"[Metrics] Results saved to {result_dir} (stem={stem})")


def evaluate_ssim_psnr(
    gt: np.ndarray,             
    rec: np.ndarray,            
    *,
    cfg: Optional[Dict] = None,
    case_id: Union[str, int] = "nocase",
    save_dir: Optional[Union[str, Path]] = "outputs/FBP",
    center_circle_ratio: Optional[float] = None  # 保留接口但不使用
) -> Dict:
    """
    计算3D体积数据的SSIM和PSNR指标
    
    对输入的真值和重建数据进行全局min-max归一化后，逐切片计算SSIM和PSNR指标。
    
    Args:
        gt: 真值数据，形状为[S,H,W]，其中S为切片数，H为高度，W为宽度
        rec: 重建数据，形状为[S,H,W]，必须与gt形状相同
        cfg: 配置字典，可包含投影角度等信息，用于生成文件名
        case_id: 案例标识符，用于文件命名和结果标识
        save_dir: 结果保存目录，None则不保存文件
        center_circle_ratio: 保留的接口参数，当前版本中不使用
        
    Returns:
        包含以下键的字典：
        - "shape": 数据形状(S,H,W)
        - "ssim": {"mean": 平均SSIM, "per_slice": 每切片SSIM数组}
        - "psnr": {"mean": 平均PSNR, "per_slice": 每切片PSNR数组}
        - "mask": {"type": "full", "center_circle_ratio": None}  # 兼容性保留
        
    Raises:
        ValueError: 当输入数组形状不符合要求时
        
    Example:
        >>> gt = np.random.rand(10, 256, 256)    # 10个切片的真值
        >>> rec = np.random.rand(10, 256, 256)   # 对应的重建结果
        >>> result = evaluate_ssim_psnr(gt, rec, case_id="test_case")
        >>> print(f"Average SSIM: {result['ssim']['mean']:.4f}")
        >>> print(f"Average PSNR: {result['psnr']['mean']:.4f}")
    """
    # 1. 检查输入数据形状
    shape = _check_shapes(gt, rec)
    S, H, W = shape
    
    # 2. 数据归一化
    gt_normalized, rec_normalized = _normalize_arrays(gt, rec)
    
    # 3. 初始化结果数组
    ssim_list = np.zeros(S, dtype=np.float32)
    psnr_list = np.zeros(S, dtype=np.float32)
    
    # 4. 逐切片计算指标
    for slice_idx in range(S):
        ssim_val, psnr_val = _compute_slice_metrics(
            gt_normalized[slice_idx], 
            rec_normalized[slice_idx]
        )
        ssim_list[slice_idx] = ssim_val
        psnr_list[slice_idx] = psnr_val
    
    # 5. 计算平均值
    ssim_mean = float(np.mean(ssim_list))
    psnr_mean = float(np.mean(psnr_list))
    
    # 6. 构建结果字典
    result = {
        "shape": shape,
        "ssim": {
            "mean": ssim_mean, 
            "per_slice": ssim_list
        },
        "psnr": {
            "mean": psnr_mean, 
            "per_slice": psnr_list
        },
        "mask": {
            "type": "full",
            "center_circle_ratio": None
        }  # 为了保持接口兼容性
    }
    
    # 7. 保存结果（如果指定了保存目录）
    if save_dir is not None:
        save_path = Path(save_dir)
        stem = _stem_from_cfg(cfg, case_id)
        _save_results(ssim_list, psnr_list, save_path, stem, case_id, shape)
    
    return result
