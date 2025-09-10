"""
数据处理函数：HU→衰减系数、衰减→灰度、体数据最小-最大归一化
"""

import numpy as np

def hu_to_attenuation(data: np.ndarray) -> np.ndarray:
    """
    将 HU 转换为衰减系数 (mu).

    公式：
        mu = mu_water + (mu_water - mu_air) / 1000 * HU
    其中：
        mu_water = 0.206
        mu_air   = 0.0004

    参数
    -----
    data : np.ndarray
        体数据 (X, Y, Z) 或 (Z, Y, X)，单位 HU。

    返回
    -----
    mu : np.ndarray
        衰减系数（与输入同形状）。
    """
    HU = data
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000.0 * HU
    return mu


def attenuation_to_gray(mu: np.ndarray, vmin: float = 0.0004, vmax: float = 0.5) -> np.ndarray:
    """
    将衰减系数映射为 0-255 的灰度图像。

    参数
    -----
    mu : np.ndarray
        衰减系数 (cm^-1)
    vmin, vmax : float
        归一化范围，默认覆盖空气到骨头。

    返回
    -----
    gray : np.ndarray
        8-bit 灰度图像（与输入同形状）。
    """
    mu_clip = np.clip(mu, vmin, vmax)
    norm = (mu_clip - vmin) / (vmax - vmin + 1e-12)
    gray = (norm * 255.0).astype(np.uint8)
    return gray


def normalize_volume_minmax(volume: np.ndarray) -> np.ndarray:
    """
    最小-最大归一化到 [0, 1] 范围。

    参数
    -----
    volume : np.ndarray
        任意维度体数据。

    返回
    -----
    volume_normalized : np.ndarray
        归一化到 [0,1] 的 float32 数组。
    """
    vol_min = float(np.min(volume))
    vol_max = float(np.max(volume))
    if vol_max - vol_min <= 1e-12:
        # 常量体：直接返回零数组
        return np.zeros_like(volume, dtype=np.float32)
    volume_normalized = (volume.astype(np.float32) - vol_min) / (vol_max - vol_min)
    return volume_normalized
