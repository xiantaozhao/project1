"""
数据处理函数：HU→衰减系数、衰减→灰度、体数据最小-最大归一化
"""

import numpy as np

def raw_to_attenuation(data: np.ndarray, rescale_slope: float, rescale_intercept: float) -> np.ndarray:
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
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    # mu = mu * 100
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


def reorient_for_axis(vol_xyz, geo, axis='z', voxel_spacing=None):
    """
    将当前体数据 vol_xyz（形状为 (x,y,z)）重排，使“想绕的轴 axis”映射为 z。
    同步更新 geo['nVoxel'], geo['dVoxel'], offOrigin，并返回新的 voxel_spacing。
    参数:
      - vol_xyz: ndarray, 形状 (x,y,z)
      - geo: dict, 至少包含键 'nVoxel', 'dVoxel', 'offOrigin'
      - axis: 'x' | 'y' | 'z'，表示你想“绕”的原始轴
      - voxel_spacing: 可选，(dx,dy,dz)。若为 None，则使用 geo['dVoxel'] 作为来源
    返回:
      - vol_perm: 重排后的体 (x',y',z')
      - geo2: 同步更新后的 geo
      - spacing_new: 重排后的 (dx',dy',dz')，np.float32
    说明:
      - 若 axis='y'，就把“旧的 y”映射为新 z'；若 axis='x'，把“旧的 x”映射为新 z'。
      - angles 仍按“绕 z”的约定传入投影函数即可（因为我们已把目标轴映射到 z）。
    """
    assert axis in ('x','y','z'), f"axis 必须是 'x'/'y'/'z'，收到 {axis}"

    # perm: 新 (x',y',z') 分别取自旧 (x,y,z) 的哪个下标
    if axis == 'z':
        perm = (0, 1, 2)     # 不变：绕 z（默认）
    elif axis == 'y':
        perm = (0, 2, 1)     # 让旧 y -> 新 z'
    else:  # axis == 'x'
        perm = (2, 1, 0)     # 让旧 x -> 新 z'

    # 1) 重排体数据
    vol_perm = np.transpose(vol_xyz, perm)

    # 2) 生成新的 spacing 来源
    if voxel_spacing is None:
        spacing_src = np.asarray(geo['dVoxel'], dtype=np.float32)  # [dx,dy,dz]
    else:
        spacing_src = np.asarray(voxel_spacing, dtype=np.float32)  # [dx,dy,dz]

    # 3) 同步更新 geo
    geo2 = geo.copy()

    # nVoxel/dVoxel 按 perm 重排
    nx, ny, nz = vol_xyz.shape
    nvoxel_new = np.array([ [nx,ny,nz][i] for i in perm ], dtype=np.int32)
    dvoxel_new = np.array([ spacing_src[i] for i in perm ], dtype=np.float32)
    geo2['nVoxel'] = nvoxel_new
    geo2['dVoxel'] = dvoxel_new

    # offOrigin 同步重排（若存在）
    if 'offOrigin' in geo2 and len(geo2['offOrigin']) == 3:
        ox, oy, oz = map(float, geo2['offOrigin'])
        geo2['offOrigin'] = np.array([ [ox,oy,oz][i] for i in perm ], dtype=np.float32)

    # 其余几何量（DSD/DSO/nDetector/dDetector/offDetector/COR/rotDetector）保持不变
    # 返回 triple：体、geo、spacing
    return vol_perm, geo2, dvoxel_new
