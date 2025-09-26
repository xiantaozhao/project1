# -*- coding: utf-8 -*-
"""
File: src/model/FBP.py
Author: <Ming>
Purpose:
    使用 ASTRA Toolbox 完成 2D FBP 重建（逐 z-slice），并对 3D 体 [S,H,W] 进行
    可选的圆形 FOV 遮罩、归一化与落盘。仅修改“重建端 [reconstruction side]”参数；
    前向投影的几何/角度从已合并的配置里读取（继承自 configs/default/...）。

Input:
    sino_SAD : np.ndarray
        Shape [S, A, D]
        - S: slice 数 (沿 z 轴逐 slice 重建)
        - A: angle 数 (必须与 cfg.projection.angles 生成的一致)
        - D: 探测器通道数 (detector bins)

    cfg_merged : Dict
        已合并的配置字典 (default.yaml + FBP.yaml)
        - 提供角度范围、几何参数、滤波器、重建大小、后处理、IO 路径等

    case_id : str | int, default="case"
        文件名前缀里的 {case_id}

    ground_truth_zyx : np.ndarray | None, default=None
        可选的参考体数据，Shape [S, H, W]
        - 若提供，则保存为 PNG 到 groudtruth/ 目录
        - 不参与重建和数值计算

    spacing_dzyx : tuple(float,float,float) | None
        (dz, dy, dx) in mm
        - 若提供，用于构建 ASTRA vol_geom 的物理尺寸
        - 否则尝试从 cfg.data.spacing 中读取

Output:
    recon : np.ndarray
        Shape [S, H, W], dtype=float32
        - 每 slice 重建图像
        - 已根据 cfg.fbp.post 进行归一化/裁剪/dtype 转换
        - 默认范围 [0,1]


Key Dependencies:
    - astra (GPU: 'FBP_CUDA' + 'cuda' projector; CPU: 'FBP' + 'line' projector)
    - numpy
    - imageio (可选，仅用于保存 PNG)
    - 项目内的 load_config: 需返回已做 inherit 的 cfg（含 default+FBP）

Expected Config Schema (摘录):
    cfg["projection"]["angles"]:
        mode: "range_step"
        start_deg: float
        stop_deg : float
        step_deg : float
        include_endpoint: bool
    cfg["projection"]["geom"]:
        type: "fanflat" | "parallel"
        det:
            from_image: bool
            det_pixel_mm: float | "auto"
        source_origin_mm: float  # DSO
        origin_det_mm  : float  # ODD
    cfg["data"]["spacing"]:
        enabled: bool
        order  : "dzyx"  # (dz,dy,dx)
        values : [dz,dy,dx]  # mm
    cfg["fbp"]:
        algo      : "FBP_CUDA" | "FBP"
        projector : "cuda" | "line"
        recon:
            img_size: [H, W]
            crop_to_circle: bool (未在算法内裁切，仅 post 可遮罩)
        filter:
            type: "ram-lak" | "hann" | "hamming" | "shepp-logan" | ...
            d_param    : Optional[float]
            parameter  : Optional[float]
            cutoff     : Optional[float] (not always used by ASTRA)
        short_scan:
            enabled: bool       # Parker weighting
            pixel_supersampling: int  # 'PixelSuperSampling'
        post:
            # 归一化 [normalization]：三选一
            # - "minmax_per_slice": 每 slice min-max 到 [0,1]
            # - "percentile"     : 每 slice 用百分位 [p_lo,p_hi] 归一化到 [0,1]
            # - "none"/None      : 不做归一化
            normalize: "minmax_per_slice" | "percentile" | "none"
            percentiles: [p_lo, p_hi]  # 当 normalize="percentile" 时有效
            mask_circle: bool          # 是否应用圆形 FOV 遮罩（外圈置零）
            mask_radius_factor: float  # 遮罩半径 = 0.5*min(H,W)*factor
            scale: float               # 归一化后的整体缩放
            clip : [lo, hi] | null     # 归一化/缩放后的裁剪范围
            dtype: "float32" | ...
        io:
            out_root: "outputs/FBP"
            save_npz_all: bool
            save_png_each_slice: bool
            case_prefix: "recon_{case_id}_{stop}@{step}"

Shapes & Conventions:
    - Sinogram [sino_SAD]: [S, A, D]
        S: slice 数（沿 cfg["projection"]["slice_axis"]="z" 切）
        A: angle 数（必须与 cfg 中生成的 angles 长度一致）
        D: detector 通道数（本文件以输入 sino 的 D 为准）
        单 slice 期望为 [A, D]（若收到 [D, A] 会自动转置）
        数值应为线积分 [line integral], i.e., -log(I/I0)

    - Reconstruction [recon]: [S, H, W]
        H,W 由 cfg["fbp"]["recon"]["img_size"] 决定（默认 256×256 或 512×512）
        经 _apply_post 处理后，通常归一化到 [0,1] 并转换为 float32

    - Physical Spacing:
        若传入 spacing_dzyx=(dz,dy,dx) 或 cfg.data.spacing.enabled=True，
        则重建体素的物理范围会基于 (dy,dx) 构造 ASTRA 的长形式 vol_geom。

I/O Behavior:
    - 若 cfg.fbp.io.save_npz_all=True:
        存为: {out_root}/{prefix}.npz，键名 "recon" -> [S,H,W]
        其中 prefix 模板缺省为 "recon_{case_id}_{stop}@{step}"

    - 若 cfg.fbp.io.save_png_each_slice=True:
        存为: {out_root}/{prefix}/0000.png, 0001.png, ...
        PNG 默认写入经过 post 的结果（浮点会乘 255 转 uint8）

    - ground_truth_zyx (可选):
        若传入 [S,H,W] 的 GT，会额外保存到 {out_root}/groudtruth/*.png，
        仅用于参考可视化，不写入 npz（函数不参与评估）。

Errors & Validations:
    - angles 长度必须等于 sino.shape[1] (A)
    - det_pixel_mm 的解析顺序：
        1) 若是数值，直接用；
        2) 若为 "auto"：需要 dx；优先 spacing_dzyx，其次 cfg.data.spacing；否则报错；
        3) 若 det.from_image=True 且未给数值：需要 dx；否则报错。
    - 几何类型仅支持 "fanflat"/"parallel"

Public API:
    fbp_reconstruct_with_astra(
        sino_SAD: np.ndarray,            # [S,A,D]
        cfg_merged: Dict,                # 已合并的配置（含 default+FBP）
        case_id: str | int = "case",     # 文件命名前缀中的 {case_id}
        ground_truth_zyx: np.ndarray | None = None,  # 可选 GT，只保存 PNG
        spacing_dzyx: Tuple[float,float,float] | None = None  # (dz,dy,dx) mm
    ) -> np.ndarray                      # 返回 [S,H,W] float32

Typical Usage:
    from src.configs.configloading import load_config
    from src.data.data_load import data_load_chest
    from src.data.data_process.projection import project_volume_with_astra
    from src.model.FBP import fbp_reconstruct_with_astra

    # 1) 读取 default 并做前向投影（示例）
    cfg_default = load_config("configs/default/chest.yaml", default_path=None)
    vol_HU_zyx, spacing_dzyx, meta = data_load_chest.load_data_chest("1", "CT")
    sino = project_volume_with_astra(vol_HU_zyx, spacing_dzyx, cfg_default, case_id=meta.get("case_id","1"))

    # 2) 读取 FBP 配置（已继承 default）
    cfg_fbp = load_config("configs/FBP/chest.yaml", default_path=None)

    # 3) FBP 重建（仅动重建端参数）
    recon_zyx = fbp_reconstruct_with_astra(
        sino_SAD=sino,                  # [S,A,D]
        cfg_merged=cfg_fbp,             # merged (default + FBP)
        case_id=meta.get("case_id", "nocase"),
        ground_truth_zyx=None,          # 可传入 GT 以保存参考 PNG
        spacing_dzyx=spacing_dzyx       # 用于 vol_geom 物理范围/推断 dx
    )

Notes:
    - 想提升观感、削弱条纹：优先在 cfg.fbp.filter 里换窗（如 "hann"/"shepp-logan"），
      并适当提高 short_scan.pixel_supersampling（例如 2）。
    - 若需要严格数值评估（SSIM/PSNR），请使用 src/evaluate/metrics_volume.py，
      并注意与 GT 的归一化一致性（建议 normalize="per_slice"）。
"""



from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import astra

try:
    import imageio.v3 as iio  # 写 PNG
except Exception:
    iio = None


def _get_from(cfg: Dict, keys: Tuple[str, ...], default=None):
    """在多级字典里按一串候选键路径获取值，便于兼容不同 default 结构。"""
    for k in keys:
        node = cfg
        ok = True
        for part in k.split('.'):
            if not (isinstance(node, dict) and part in node):
                ok = False
                break
            node = node[part]
        if ok:
            return node
    if default is not None:
        return default
    raise KeyError(f"Missing config key in any of: {keys}")


def _angles_from_default(cfg: Dict) -> np.ndarray:
    """
    从 cfg['projection']['angles'] 读取角度（单位：弧度）。
    仅实现 mode='range_step' + include_endpoint 语义：
      start_deg, stop_deg, step_deg, include_endpoint
    """
    ang = cfg["projection"]["angles"]
    mode = str(ang.get("mode", "range_step")).lower()
    if mode != "range_step":
        raise ValueError(f"Only angles.mode='range_step' is supported, got {mode}")

    start = float(ang.get("start_deg", 0.0))
    stop  = float(ang.get("stop_deg", 180.0))
    step  = float(ang.get("step_deg", 1.0))
    include_endpoint = bool(ang.get("include_endpoint", False))

    # 用步长生成度数数组
    if include_endpoint:
        # 等价于 np.arange 并补齐终点（若恰好落在步长网格上避免重复）
        vals = list(np.arange(start, stop, step, dtype=np.float32))
        if len(vals) == 0 or abs(vals[-1] - stop) > 1e-6:
            vals.append(np.float32(stop))
        angles_deg = np.array(vals, dtype=np.float32)
    else:
        angles_deg = np.arange(start, stop, step, dtype=np.float32)

    return np.deg2rad(angles_deg.astype(np.float32))


def _proj_geom_from_default(cfg: Dict, angles: np.ndarray, sino_D: int,
                            spacing_dzyx: Tuple[float, float, float] | None = None):
    """
    从 cfg['projection']['geom'] 构建 ASTRA 的 proj_geom。
    - det_count：强制与 sino 的 D 对齐。
    - det_pixel_mm 解析优先级（高→低）：
        A) 若 det_pixel_mm 是数值，直接用；
        B) 若 det_pixel_mm == "auto"：
            1) 优先用 spacing_dzyx 提供的 dx；
            2) 否则用 cfg.data.spacing (order='dzyx') 的 dx；
            3) 用公式 det_pixel_mm = dx * (DSO+ODD) / DSO；
        C) 若 det.from_image = true 且没给数值：
            1) 优先用 spacing_dzyx 的 dx；
            2) 否则用 cfg.data.spacing 的 dx；
        D) 仍然缺失 => 报错。
    """
    g = cfg["projection"]["geom"]
    gtype = str(g.get("type", "fanflat")).lower()

    det_cfg = g.get("det", {})
    det_from_img = bool(det_cfg.get("from_image", False))

    # 通道数：强制与 sino 的 D 对齐（更安全）
    det_count = int(sino_D)

    # DSO/ODD
    DSO = float(g.get("source_origin_mm", 600.0))
    ODD = float(g.get("origin_det_mm",  400.0))

    # ---- 准备 dx 值（优先用函数入参 spacing_dzyx）----
    dx_val = None
    if spacing_dzyx is not None and len(spacing_dzyx) == 3:
        dx_val = float(spacing_dzyx[2])
    else:
        data_sp = cfg.get("data", {}).get("spacing", {})
        if bool(data_sp.get("enabled", False)) and str(data_sp.get("order", "")).lower() == "dzyx":
            vals = data_sp.get("values", None)
            if isinstance(vals, (list, tuple)) and len(vals) == 3:
                dx_val = float(vals[2])  # dx

    # ---- 解析 det_pixel_mm ----
    det_pixel_mm = det_cfg.get("det_pixel_mm", None)
    if isinstance(det_pixel_mm, str) and det_pixel_mm.lower() == "auto":
        if dx_val is None:
            raise ValueError("det_pixel_mm=auto 需要 dx；请传 spacing_dzyx 或在 cfg.data.spacing(dzyx) 里提供。")
        det_pixel_mm = dx_val * (DSO + ODD) / DSO
    elif det_pixel_mm is None and det_from_img:
        # from_image: 若未显式给数值，则用 dx 作为探测器像素宽度
        if dx_val is not None:
            det_pixel_mm = dx_val

    if det_pixel_mm is None:
        raise ValueError(
            "无法确定 det_pixel_mm。请在 YAML 设置 projection.geom.det.det_pixel_mm "
            "(数值或 'auto')，或开启 det.from_image 并提供 dx（通过 spacing_dzyx 或 cfg.data.spacing）。"
        )

    det_spacing = float(det_pixel_mm)

    # ---- 构建 ASTRA 几何 ----
    if gtype == "parallel":
        return astra.create_proj_geom('parallel', det_spacing, det_count, angles)
    elif gtype in ("fanflat", "fanbeam", "fan"):
        return astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)
    else:
        raise ValueError(f"Unsupported projection.geom.type: {gtype}")


def _vol_geom_from_fbp(cfg_fbp: Dict, dy_mm: float | None, dx_mm: float | None):
    """
    构建重建体素几何：
    - 若提供 dy_mm、dx_mm（来自 spacing_dzyx 或 cfg.data.spacing），用“长形式”带物理尺寸；
    - 否则退回 ASTRA 简写（无物理范围）。
    """
    rh, rw = cfg_fbp["fbp"]["recon"]["img_size"]
    rh, rw = int(rh), int(rw)
    if (dy_mm is not None) and (dx_mm is not None) and dy_mm > 0 and dx_mm > 0:
        return astra.create_vol_geom(
            rh, rw,
            -rw * dx_mm / 2.0,  rw * dx_mm / 2.0,
            -rh * dy_mm / 2.0,  rh * dy_mm / 2.0
        )
    else:
        return astra.create_vol_geom(rh, rw)


def _algo_and_projector(cfg_fbp: Dict) -> Tuple[str, str]:
    algo = str(cfg_fbp["fbp"]["algo"])
    projector = str(cfg_fbp["fbp"]["projector"])
    if algo not in ("FBP", "FBP_CUDA"):
        raise ValueError(f"fbp.algo must be 'FBP' or 'FBP_CUDA', got {algo}")
    return algo, projector


def _apply_fov_mask_circle(arr_3d: np.ndarray, radius_factor: float = 0.99) -> np.ndarray:
    """
    对 [S,H,W] 重建结果做圆形 FOV 遮罩（外圈置零）。
    radius_factor: 半径相对 min(H,W)/2 的比例，留一点边避免截断。
    """
    S, H, W = arr_3d.shape
    yy, xx = np.ogrid[-H//2:H//2, -W//2:W//2]
    r2 = (min(H, W) * 0.5 * float(radius_factor)) ** 2
    mask = ((xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2) <= r2).astype(np.float32)
    return arr_3d * mask[None, ...]


def _apply_post(recon: np.ndarray, cfg_fbp: Dict) -> np.ndarray:
    """按照配置做遮罩/归一化/裁剪/dtype。"""
    post = _get_from(cfg_fbp, ("fbp.post",))
    # ① 兼容两处开关：fbp.post.mask_circle 或 projection.astra.pad_to_fov
    mask_circle = bool(post.get("mask_circle", False)) \
                  or bool(_get_from(cfg_fbp, ("projection.astra",), {}).get("pad_to_fov", False))
    if mask_circle:
        rfac = float(post.get("mask_radius_factor", 1))
        recon = _apply_fov_mask_circle(recon, radius_factor=rfac)

    # ② 归一化
    mode = post.get("normalize", "minmax_per_slice")

    if mode == "minmax_per_slice":
        # 逐 slice min-max 到 [0,1]
        S = recon.shape[0]
        for s in range(S):
            v = recon[s]
            vmin, vmax = float(np.min(v)), float(np.max(v))
            if vmax > vmin:
                recon[s] = (v - vmin) / (vmax - vmin)
            else:
                recon[s] = 0.0

    elif mode == "percentile":
        # 逐 slice 百分位归一化到 [0,1]
        p_lo, p_hi = post.get("percentiles", [1.0, 99.0])
        S = recon.shape[0]
        for s in range(S):
            v = recon[s]
            lo = np.percentile(v, p_lo)
            hi = np.percentile(v, p_hi)
            if hi > lo:
                recon[s] = np.clip((v - lo) / (hi - lo), 0, 1)
            else:
                recon[s] = 0.0

    elif mode == "minmax":
        # 全局 min-max 到 [0,1]（跨 slice 一致）
        gmin = float(np.min(recon))
        gmax = float(np.max(recon))
        if gmax > gmin:
            recon = (recon - gmin) / (gmax - gmin)
        else:
            recon = np.zeros_like(recon, dtype=np.float32)

    elif mode == "fixed":
        # 固定范围到 [0,1]（适合 HU/μ 的物理口径）
        # cfg: fbp.post.fixed_range: [lo, hi]
        lo, hi = post.get("fixed_range", [-1000.0, 3000.0])
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            raise ValueError(f"[global_fixed] invalid fixed_range: {lo}, {hi}")
        recon = np.clip((recon - lo) / (hi - lo), 0.0, 1.0)

    elif mode in (None, "none"):
        pass

    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")

    # （可选）再次遮罩，确保外圈保持 0
    if mask_circle:
        rfac = float(post.get("mask_radius_factor", 1.0))
        recon = _apply_fov_mask_circle(recon, radius_factor=rfac)


    # ③ 缩放/裁剪
    scale = float(post.get("scale", 1.0))
    if scale != 1.0:
        recon = recon * scale

    clip = post.get("clip", None)
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        recon = np.clip(recon, lo, hi)

    # ④ dtype
    dtype = post.get("dtype", "float32")
    return recon.astype(dtype, copy=False)


def _to_uint8_slicewise(arr: np.ndarray, mode: str = "percentile",
                        p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    """
    将 [S,H,W] 浮点阵列逐 slice 归一化为 uint8（0..255），仅用于可视化输出。
    """
    S, H, W = arr.shape
    out = np.empty((S, H, W), dtype=np.uint8)
    for s in range(S):
        v = arr[s].astype(np.float32, copy=False)
        if mode == "percentile":
            lo, hi = np.percentile(v, [p_lo, p_hi])
        else:
            lo, hi = float(np.min(v)), float(np.max(v))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        u = np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        out[s] = (u * 255.0 + 0.5).astype(np.uint8)
    return out

def _fmt_num(x: float) -> str:
    """Format float like 3 -> '3', 3.5 -> '3.5', 3.250 -> '3.25' (max 6 decimals)."""
    s = f"{x:.6f}".rstrip('0').rstrip('.')
    return s if s else "0"

def _save_outputs(recon: np.ndarray, cfg_fbp: Dict, case_id: str | int | None,
                  ground_truth_zyx: np.ndarray | None = None):
    """
    保存重建结果；若提供 ground_truth_zyx，则仅保存 PNG 到 groudtruth 目录（不存 npz）。
    - 前缀支持小数角度
    - .npz 保存到 fbp.io.save_npz_dir（按 {dataset_name} 模板展开）
    - dataset_name 来自 data.name
    """
    io_cfg = cfg_fbp["fbp"]["io"]

    # ---- dataset_name from data.name ----
    dataset_name = cfg_fbp.get("data", {}).get("name", "unknown")

    # ---- output roots (expand {dataset_name}) ----
    out_root_tpl = io_cfg.get("out_root", "outputs/FBP/{dataset_name}")
    out_root = Path(out_root_tpl.format(dataset_name=dataset_name))
    out_root.mkdir(parents=True, exist_ok=True)

    # 角度信息（保留小数）
    ang_cfg = cfg_fbp["projection"]["angles"]
    stop_deg = float(ang_cfg.get("stop_deg", 180.0))
    step_deg = float(ang_cfg.get("step_deg", 1.0))

    # case_id 兜底
    case_id_safe = str(case_id) if case_id not in (None, "", "None") else "nocase"

    # 前缀（与 .npz 同名，也作为 PNG 文件夹名）；角度小数友好显示
    prefix_tpl = io_cfg.get("case_prefix", "recon_{case_id}_{stop}@{step}")
    prefix = prefix_tpl.format(
        case_id=case_id_safe,
        stop=_fmt_num(stop_deg),
        step=_fmt_num(step_deg),
    )

    # 1) 保存重建整体 npz（如开启） -> 到 save_npz_dir（按 {dataset_name} 展开）
    if bool(io_cfg.get("save_npz_all", True)):
        save_npz_dir_tpl = io_cfg.get("save_npz_dir", "data/interim/recon/{dataset_name}")
        save_npz_dir = Path(save_npz_dir_tpl.format(dataset_name=dataset_name))
        save_npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_npz_dir / f"{prefix}.npz", recon=recon)

        print(f"[FBP] Saved reconstruction npz to: {save_npz_dir / f'{prefix}.npz'}")

    # 2) 保存重建 PNG
    if bool(io_cfg.get("save_png_each_slice", True)):
        if iio is None:
            print("[FBP] imageio 未安装，跳过 PNG 保存")
        else:

            # 改成 out_root/recon_{caseid}/{prefix}
            png_dir = out_root / f"recon_{case_id_safe}" / prefix
            png_dir.mkdir(parents=True, exist_ok=True)

            arr = recon
            # 假定 recon 已归一化到 [0,1]；若不是浮点则直接写
            if arr.dtype.kind in "fc":
                arr_to_write = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
            else:
                arr_to_write = arr

            for s in range(arr.shape[0]):
                iio.imwrite(png_dir / f"{s:04d}.png", arr_to_write[s])

            print(f"[FBP] Saved PNG slices to: {png_dir}")


    # 3) Ground Truth：只保存 PNG 到 groudtruth 目录 (normalize)
    if ground_truth_zyx is not None and iio is not None:
        gt = np.asarray(ground_truth_zyx, dtype=np.float32)
        if gt.ndim != 3:
            raise ValueError(f"ground_truth_zyx must be [S,H,W], got shape {gt.shape}")
        
        # case_id 兜底
        case_id_safe = str(case_id) if case_id not in (None, "", "None") else "nocase"

        # groundtruth 目录改成带 case_id
        gt_dir = out_root / f"groundtruth_{case_id_safe}"
        gt_dir.mkdir(parents=True, exist_ok=True)

        S = gt.shape[0]
        for s in range(S):
            v = gt[s]
            vmin, vmax = float(np.min(v)), float(np.max(v))
            if vmax > vmin:
                v_norm = (v - vmin) / (vmax - vmin)
            else:
                v_norm = np.zeros_like(v, dtype=np.float32)
            v_u8 = (v_norm * 255.0 + 0.5).astype(np.uint8)
            iio.imwrite(gt_dir / f"{s:04d}.png", v_u8)

        print(f"[FBP] Saved ground truth PNG slices to: {gt_dir}")



def fbp_reconstruct_with_astra(
    sino_SAD: np.ndarray,
    cfg_merged: Dict,
    case_id: str | int = "case",
    ground_truth_zyx: np.ndarray | None = None,
    spacing_dzyx: Tuple[float, float, float] | None = None
) -> np.ndarray:
    """
    用 ASTRA 的 [FBP / FBP_CUDA] 对扇束/平行束正弦图逐 slice 重建。
    - sino_SAD: [S, A, D] 浮点；由 ASTRA 前向投影得到
    - cfg_merged: 经过 load_config 继承合并后的配置（包含 default+FBP）
    - ground_truth_zyx: 可选，形状 [S,H,W] 的 GT，只保存 PNG 到 groudtruth/
    - spacing_dzyx: 可选 (dz,dy,dx)。若提供，将用于构建 vol_geom 的物理范围；否则尝试从 cfg.data.spacing 读取；再否则用简写 vol_geom。
    - 返回: [S, H, W]
    """
    assert sino_SAD.ndim == 3, "sino must be [S, A, D]"
    S, A, D = sino_SAD.shape

    # 角度/几何
    angles = _angles_from_default(cfg_merged)
    if angles.shape[0] != A:
        raise ValueError(f"[angles] length {angles.shape[0]} != sino A {A}")

    proj_geom = _proj_geom_from_default(cfg_merged, angles, sino_D=D, spacing_dzyx=spacing_dzyx)

    # ---- 解析 dy, dx：优先用入参 spacing_dzyx；否则尝试 cfg.data.spacing（order='dzyx'） ----
    dy_mm = dx_mm = None
    if spacing_dzyx is not None:
        if len(spacing_dzyx) != 3:
            raise ValueError("spacing_dzyx must be (dz,dy,dx)")
        dy_mm = float(spacing_dzyx[1]); dx_mm = float(spacing_dzyx[2])
    else:
        data_sp = cfg_merged.get("data", {}).get("spacing", {})
        if bool(data_sp.get("enabled", False)) and str(data_sp.get("order", "")).lower() == "dzyx":
            vals = data_sp.get("values", None)
            if isinstance(vals, (list, tuple)) and len(vals) == 3:
                dy_mm = float(vals[1]); dx_mm = float(vals[2])

    vol_geom  = _vol_geom_from_fbp(cfg_merged, dy_mm=dy_mm, dx_mm=dx_mm)
    algo, projector = _algo_and_projector(cfg_merged)

    # 预创建投影器
    projector_id = astra.create_projector(projector, proj_geom, vol_geom)

    fil = _get_from(cfg_merged, ("fbp.filter",))
    ss  = _get_from(cfg_merged, ("fbp.short_scan",))

    # Parker / Cosine（向后兼容 projection.astra.*）
    proj_astra_opts = _get_from(cfg_merged, ("projection.astra",), default={})
    parker_from_proj  = bool(proj_astra_opts.get("parker_weighting", False))
    cosine_from_proj  = bool(proj_astra_opts.get("cosine_weight", False))  # 占位
    short_scan_enabled = bool(ss.get("enabled", False)) or parker_from_proj
    pixel_ss = int(ss.get("pixel_supersampling", 1))

    H = vol_geom["GridRowCount"]; W = vol_geom["GridColCount"]
    recon = np.zeros((S, H, W), dtype=np.float32)

    for s in range(S):
        sino_AD = np.asarray(sino_SAD[s], dtype=np.float32)  # [A, D]
        if sino_AD.shape != (A, D):
            # 保险：如果是 [D, A] 就转置
            if sino_AD.shape == (D, A):
                sino_AD = sino_AD.T
            else:
                raise ValueError(f"Unexpected slice sino shape: {sino_AD.shape}, expect {(A, D)}")

        sinogram_id = astra.data2d.create('-sino', proj_geom, sino_AD)
        recon_id    = astra.data2d.create('-vol',  vol_geom)

        cfg = astra.astra_dict(algo)
        cfg["ProjectionDataId"]     = sinogram_id
        cfg["ReconstructionDataId"] = recon_id
        cfg["ProjectorId"] = projector_id

        # 过滤器/短扫描选项（字段名按 ASTRA FBP/FBP_CUDA）
        if fil.get("type"):
            cfg.setdefault("option", {})
            cfg["option"]["FilterType"] = fil["type"]
            if fil.get("d_param") is not None:
                cfg["option"]["FilterD"] = float(fil["d_param"])
            if fil.get("parameter") is not None:
                cfg["option"]["FilterParameter"] = float(fil["parameter"])

        # Parker 短扫描：任一来源为真都启用（ASTRA 内部即应用 Parker weighting）
        if short_scan_enabled:
            cfg.setdefault("option", {})
            cfg["option"]["ShortScan"] = True

        # 像素超采样（抗走样）
        if pixel_ss > 1:
            cfg.setdefault("option", {})
            cfg["option"]["PixelSuperSampling"] = pixel_ss

        # 余弦加权：ASTRA 的 fan-beam FBP 内部会做，保留占位不额外手搓
        # if cosine_from_proj: pass

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        recon[s] = astra.data2d.get(recon_id).astype(np.float32, copy=False)

        # 清理 slice 级资源
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(recon_id)

    # 清理 projector
    astra.projector.delete(projector_id)

    # 后处理（遮罩/归一化/裁剪/dtype）
    recon = _apply_post(recon, cfg_merged)

    # 保存（按开关）+ Ground Truth（只 PNG）
    _save_outputs(recon, cfg_merged, case_id=case_id, ground_truth_zyx=ground_truth_zyx)

    return recon
