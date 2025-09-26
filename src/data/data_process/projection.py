# src/data/data_process/projection.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import os

try:
    import astra
except ImportError:
    astra = None

try:
    from PIL import Image
except ImportError:
    Image = None

def _ensure_astra_cuda():
    if astra is None:
        raise ImportError("ASTRA not installed. Please install astra-toolbox.")
    # 简单检查 CUDA 插件是否可用（失败也能回退到 line）
    return True

def _normalize_axis_order(vol: np.ndarray, axis_order: str, target_axis: str = "z") -> np.ndarray:
    """
    将体数据重排到 [Z, Y, X]，并保证我们按 Z 切片。
    axis_order: 声明 vol 当前的轴顺序（如 'zyx' 或 'xyz'）
    """
    order = axis_order.lower()
    assert set(order) == set("zyx")
    idx = {ch: i for i, ch in enumerate(order)}
    vol_zyx = np.moveaxis(vol, [idx['z'], idx['y'], idx['x']], [0, 1, 2])
    if target_axis != 'z':
        raise NotImplementedError("当前仅支持按 z 轴切片（slice_axis='z'）")
    return vol_zyx

def _hu_to_mu(vol_hu: np.ndarray, mu_water: float) -> np.ndarray:
    return mu_water * (1.0 + vol_hu.astype(np.float32) / 1000.0)

def _angles_from_step(start: float, stop: float, step: float, include_endpoint: bool) -> np.ndarray:
    # 与 np.arange 类似，但可选是否包含末端
    if include_endpoint:
        n = int(np.floor((stop - start) / step + 0.5)) + 1
        arr = start + np.arange(n) * step
        if arr[-1] > stop:  # 裁剪
            arr = arr[arr <= stop + 1e-6]
        return arr
    else:
        n = int(np.floor((stop - start) / step + 1e-9))
        return start + np.arange(n) * step

def _build_proj_geom_fanflat(cfg_geom: Dict, angles_deg: np.ndarray,
                             dx_mm: float, width_px: int):
    """
    构造 fanflat 投影几何（严格按官方签名）：
    create_proj_geom('fanflat', det_spacing, det_count, angles_rad, DSO, ODD)
    """
    det_cfg = cfg_geom['det']
    if bool(det_cfg.get('from_image', False)):
        det_count   = int(width_px)
        det_spacing = float(dx_mm)  # 像素间距(mm)，用 slice 的 dx
    else:
        det_count   = int(det_cfg['det_count'])
        det_spacing = float(det_cfg['det_pixel_mm'])

    if det_count <= 0:
        raise ValueError(f"det_count must be > 0, got {det_count}")
    if not np.isfinite(det_spacing) or det_spacing <= 0:
        raise ValueError(f"det_spacing must be > 0, got {det_spacing}")

    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    if angles_rad.ndim != 1 or angles_rad.size == 0:
        raise ValueError(f"angles must be 1-D & non-empty, got shape {angles_rad.shape}")

    DSO = float(cfg_geom['source_origin_mm'])
    ODD = float(cfg_geom['origin_det_mm'])
    if DSO <= 0 or ODD <= 0:
        raise ValueError(f"DSO/ODD must be > 0, got DSO={DSO}, ODD={ODD}")

    # 正确顺序：det_spacing -> det_count -> angles -> DSO -> ODD
    proj_geom = astra.create_proj_geom( # type: ignore # type: ignore
        'fanflat',
        det_spacing,
        det_count,
        angles_rad,
        DSO,
        ODD
    )
    return proj_geom, det_count



def _build_vol_geom_2d(h: int, w: int, dy_mm: float, dx_mm: float):
    """
    ASTRA 2D 体素几何（长形式）。
    注意顺序：先行(h, 对应Y)，后列(w, 对应X)。
    """
    return astra.create_vol_geom( # type: ignore
        h, w,
        -w * dx_mm / 2.0,  w * dx_mm / 2.0,   # x_min, x_max
        -h * dy_mm / 2.0,  h * dy_mm / 2.0    # y_min, y_max
    )

def _to_uint8(img: np.ndarray, method: str = "percentile",
              p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    """把浮点数组归一化到 uint8（0..255）。支持百分位裁剪或 min-max。"""
    arr = img.astype(np.float32, copy=False)
    if method == "percentile":
        lo, hi = np.percentile(arr, [p_lo, p_hi])
    else:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)

def project_volume_with_astra(
    vol_in: np.ndarray,
    spacing_dzyx: Tuple[float, float, float] | None,
    cfg: Dict,
    case_id: str = "case",
) -> np.ndarray:
    """
    前向投影：对 3D 体数据逐 z-slice 做 2D fan-beam 投影，返回形状 [S, A, D] 的 sinogram。
    - vol_in: 体数据；可能是 HU 或灰度；输入轴顺：
        * chest:  (Z, Y, X)
        * other:  (X, Y, Z)  —— 本函数会内部规范到 (Z, Y, X)
    - spacing_dzyx: 对 chest: (dz, dy, dx)；对 other: 通常是 (dx, dy, dz)，本函数会重排为 (dz, dy, dx)
    - cfg: YAML 字典
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    assert cfg['projection']['geom']['type'] == 'fanflat', "当前实现仅 fanflat 标量版"

    # ---- 1) 轴顺与单位归一化 ----
    data_name = cfg['data']['name'].lower()  # 'chest' or others

    # 输入轴顺声明：chest=zyx；other=xyz（若你在cfg里自定义，会以cfg为准）
    if cfg['projection']['volume']['axis_order'] == 'auto':
        axis_order = 'zyx' if data_name == 'chest' else cfg['data'].get('axis_order', 'xyz')
    else:
        axis_order = cfg['projection']['volume']['axis_order']

    # 统一到 (Z,Y,X)，确保“沿XY平面旋转，按Z切片”
    vol_zyx = _normalize_axis_order(vol_in, axis_order, target_axis="z")

    # ---- 1.1 spacing：一次性确定为 (dz,dy,dx)（不要在下文再从cfg读）----
    if spacing_dzyx is None or cfg['data']['spacing']['enabled']:
        # 从 cfg.data.spacing 读取，并按其声明的顺序重排到 (dz,dy,dx)
        order = cfg['data']['spacing']['order'].lower()  # 例如 'dzyx' 或 'xyz'
        vals = [float(v) for v in cfg['data']['spacing']['values']]
        # 把 'd' 当作 'z' 处理
        order = order.replace('d', 'z')
        # 构造映射
        assert set(order) >= set('zyx'), f"spacing order 需包含 z,y,x，当前为 {order}"
        mp = {ch: vals[i] for i, ch in enumerate(order)}
        dz, dy, dx = float(mp['z']), float(mp['y']), float(mp['x'])
    else:
        # 调用方已提供 spacing：
        if data_name == 'chest':
            # chest 约定传入就是 (dz,dy,dx)
            dz, dy, dx = (float(spacing_dzyx[0]), float(spacing_dzyx[1]), float(spacing_dzyx[2]))
        else:
            # other 输入体是 (X,Y,Z)，常见 loader 给 spacing 为 (dx,dy,dz) —— 这里重排为 (dz,dy,dx)
            dx_in, dy_in, dz_in = spacing_dzyx  # (dx,dy,dz)
            dz, dy, dx = float(dz_in), float(dy_in), float(dx_in)

    # ---- 2) 单位：HU -> mu 或直接使用 ----
    units = cfg['projection']['volume']['input_units'].lower()
    if units == 'auto':
        units = 'hu' if data_name == 'chest' else 'gray'

    if units == 'hu':
        if cfg['projection']['volume']['hu_to_mu']['enabled']:
            mu_water = float(cfg['projection']['volume']['hu_to_mu']['mu_water_mm_inv'])
            lo, hi = cfg['projection']['volume']['hu_to_mu']['hu_clip']
            vol_zyx = np.clip(vol_zyx, lo, hi)
            vol_mu = _hu_to_mu(vol_zyx, mu_water)
        else:
            raise ValueError("HU 输入但 hu_to_mu.disabled，ASTRA 期待线衰减系数 mu（建议开启转换）")
    elif units in ('mu', 'gray'):
        vol_mu = vol_zyx.astype(np.float32)
    else:
        raise ValueError(f"Unsupported input_units: {units}")

    # 统一形状
    S, H, W = vol_mu.shape  # S=z-slices, H=Y, W=X

    # ---- 3) 角度（step） ----
    ang_cfg = cfg['projection']['angles']
    assert ang_cfg['mode'] == 'range_step'
    angles = _angles_from_step(
        float(ang_cfg['start_deg']),
        float(ang_cfg['stop_deg']),
        float(ang_cfg['step_deg']),
        bool(ang_cfg['include_endpoint'])
    )
    if len(angles) == 0:
        raise ValueError("angles is empty: check start/stop/step and include_endpoint")
    A = len(angles)

    # ---- 4) ASTRA 初始化（沿 XY 平面旋转，按 Z 切片）----
    use_cuda = (cfg['projection']['astra']['projector_2d'] == 'cuda')
    if use_cuda:
        _ensure_astra_cuda()

    # 投影几何（fanflat：det_spacing, det_count, angles, DSO, ODD）
    proj_geom, det_count = _build_proj_geom_fanflat(
        cfg['projection']['geom'], angles, dx_mm=dx, width_px=W
    )
    # 体素几何（2D）：(H,W) + 物理范围
    vol_geom = _build_vol_geom_2d(H, W, dy_mm=dy, dx_mm=dx)

    # Projector（只建一次；签名：proj_geom 在前，vol_geom 在后）
    projector_id = astra.create_projector('cuda' if use_cuda else 'line', proj_geom, vol_geom) # type: ignore

    # 输出 [S, A, D]
    sino_out = np.zeros((S, A, det_count), dtype=np.float32)

    batch = int(cfg['projection']['astra']['batch_slices'])
    try:
        for s0 in range(0, S, batch):
            s1 = min(S, s0 + batch)
            for s in range(s0, s1):
                slice2d = np.ascontiguousarray(vol_mu[s], dtype=np.float32)  # (H, W)
                sid = astra.data2d.create('-vol', vol_geom, slice2d) # type: ignore

                sino_id, sino = astra.create_sino(sid, projector_id)   # type: ignore # [A, D]
                sino_out[s, :, :] = sino

                astra.data2d.delete(sid) # pyright: ignore[reportOptionalMemberAccess]
                astra.data2d.delete(sino_id) # type: ignore
    finally:
        astra.projector.delete(projector_id) # type: ignore

    # ---- 5) 保存 PNG（每个角度一张，把所有 slice 堆叠；按 spacing 校正纵横比）----
    png_cfg = cfg['projection']['save'].get('pngs', {})
    if png_cfg.get('enabled', False):
        if Image is None:
            raise ImportError("需要安装 pillow： pip install pillow")

        out_root = cfg['projection']['save']['dir'].format(dataset_name=cfg['data']['dataset_name'])
        out_dir = os.path.join(out_root, "projs")
        os.makedirs(out_dir, exist_ok=True)

        method = png_cfg.get('normalize', 'percentile')
        p_lo, p_hi = png_cfg.get('percentiles', [0.5, 99.5])

        aspect_cfg = png_cfg.get('aspect', {})
        use_spacing = bool(aspect_cfg.get('use_spacing', True))
        scale = float(aspect_cfg.get('scale', 1.0))
        max_h = int(aspect_cfg.get('max_height_px', 4096))
        interp_name = str(aspect_cfg.get('interpolation', 'nearest')).lower()
        resample = Image.NEAREST if interp_name == 'nearest' else Image.BILINEAR # type: ignore

        det_cfg = cfg['projection']['geom']['det']
        det_spacing_mm = float(dx) if bool(det_cfg.get('from_image', False)) \
                         else float(det_cfg['det_pixel_mm'])

        for ai, ang_deg in enumerate(angles):
            img_SD = sino_out[:, ai, :]  # [S, D]
            img_u8 = _to_uint8(img_SD, method=method, p_lo=p_lo, p_hi=p_hi)

            if use_spacing and det_spacing_mm > 0:
                target_h = int(round((img_u8.shape[0] * dz / det_spacing_mm) * scale))
            else:
                target_h = int(round(img_u8.shape[0] * max(1.0, scale)))
            target_h = max(1, min(target_h, max_h))

            H0, W0 = img_u8.shape
            if target_h != H0:
                img_resized = Image.fromarray(img_u8).resize((W0, target_h), resample=resample)
            else:
                img_resized = Image.fromarray(img_u8)

            pos = int(round(float(ang_deg))) % 360
            fpath = os.path.join(out_dir, f"{pos:03d}.png")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            img_resized.save(fpath)

        print(f"[Projection] Saved projection PNGs to: {out_dir}")

    # ---- 6) 保存 sinogram----
    save_cfg = cfg['projection']['save']
    if save_cfg['enabled']:
        rng_str = f"{ang_cfg['stop_deg']:.0f}@{ang_cfg['step_deg']}"
        out_dir = save_cfg['dir'].format(dataset_name=cfg['data']['dataset_name'])
        os.makedirs(out_dir, exist_ok=True)
        out_name = save_cfg['filename'].format(case_id=case_id, range=rng_str)
        np.save(os.path.join(out_dir, out_name), sino_out)

        print(f"[Projection] Saved sinogram to: {os.path.join(out_dir, out_name)}")

    return sino_out  # [S, A, D]
