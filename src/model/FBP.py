# -*- coding: utf-8 -*-
"""使用 ASTRA Toolbox 对正弦图执行逐 slice 的 FBP 重建。

功能概览：
    * 读取 `cfg.projection.*` 构造角度与投影几何。
    * 复用 ASTRA 投影器逐 slice 调用 FBP / FBP_CUDA。
    * 根据 `cfg.fbp.post` 执行遮罩、归一化、裁剪与 dtype 转换。
    * 可选落盘重建体与参考真值 PNG。

参数:
    sino_SAD (np.ndarray): 形状 [S, A, D] 的正弦图数据。
    cfg_merged (Dict): `default` 与 `FBP` 配置合并后的字典。
    case_id (str | int): 输出文件名前缀，默认 "case"。
    ground_truth_zyx (np.ndarray | None): 形状 [S, H, W] 的参考体数据，仅用于保存 PNG。
    spacing_dzyx (Tuple[float, float, float] | None): (dz, dy, dx) 的体素间距，单位 mm。

返回:
    np.ndarray: 形状 [S, H, W]、dtype=float32 的重建体数据。

注意事项:
    * angles 数量必须与 `sino_SAD` 的第二维一致。
    * 仅支持 fanflat 或 parallel 投影几何。
    * 若 `det_pixel_mm` 为 "auto" 或依赖图像尺寸，需要能够解析体素 dx。
"""



from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast
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

    # 优先从入参或数据配置解析体素 dx
    dx_val = None
    if spacing_dzyx is not None and len(spacing_dzyx) == 3:
        dx_val = float(spacing_dzyx[2])
    else:
        data_sp = cfg.get("data", {}).get("spacing", {})
        if bool(data_sp.get("enabled", False)) and str(data_sp.get("order", "")).lower() == "dzyx":
            vals = data_sp.get("values", None)
            if isinstance(vals, (list, tuple)) and len(vals) == 3:
                dx_val = float(vals[2])  # dx

    # 解析探测器像素尺寸 det_pixel_mm
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

    # 构建 ASTRA 投影几何
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

class _FBPReconstructionPipeline:
    """封装 FBP 重建全流程，避免重复提取配置。"""

    def __init__(self, cfg: Dict, spacing_dzyx: Optional[Tuple[float, float, float]]):
        self.cfg = cfg
        if spacing_dzyx is not None:
            if len(spacing_dzyx) != 3:
                raise ValueError("spacing_dzyx must be (dz,dy,dx)")
            self.spacing_dzyx: Optional[Tuple[float, float, float]] = (
                float(spacing_dzyx[0]),
                float(spacing_dzyx[1]),
                float(spacing_dzyx[2]),
            )
        else:
            self.spacing_dzyx = None

        self.post_cfg = _get_from(cfg, ("fbp.post",))
        self.filter_cfg = _get_from(cfg, ("fbp.filter",))
        self.short_scan_cfg = _get_from(cfg, ("fbp.short_scan",))
        self.proj_astra_cfg = _get_from(cfg, ("projection.astra",), default={})
        self.proj_volume_cfg = _get_from(cfg, ("projection.volume",), default={})
        self.io_cfg = cfg["fbp"]["io"]
        self.dataset_name = cfg.get("data", {}).get("name", "unknown")

        self.algo: Optional[str] = None
        self.projector: Optional[str] = None
        self.angles: Optional[np.ndarray] = None
        self.proj_geom = None
        self.vol_geom = None
        self.S = self.A = self.D = 0

        self.mask_circle_enabled = bool(self.post_cfg.get("mask_circle", False)) or bool(
            self.proj_astra_cfg.get("pad_to_fov", False)
        )
        self.mask_radius_factor = float(self.post_cfg.get("mask_radius_factor", 1.0))
        self.normalization_mode = self.post_cfg.get("normalize", "minmax_per_slice")
        percentiles = self.post_cfg.get("percentiles", [1.0, 99.0])
        self.percentiles = (float(percentiles[0]), float(percentiles[1])) if len(percentiles) >= 2 else (1.0, 99.0)
        fixed_range = self.post_cfg.get("fixed_range", [-1000.0, 3000.0])
        self.fixed_range = (float(fixed_range[0]), float(fixed_range[1])) if len(fixed_range) >= 2 else (-1000.0, 3000.0)
        self.scale_factor = float(self.post_cfg.get("scale", 1.0))
        self.clip_range = self.post_cfg.get("clip", None)
        self.target_dtype = self.post_cfg.get("dtype", "float32")

        self.parker_from_proj = bool(self.proj_astra_cfg.get("parker_weighting", False))
        self.pixel_ss = int(self.short_scan_cfg.get("pixel_supersampling", 1))
        self.short_scan_enabled = bool(self.short_scan_cfg.get("enabled", False)) or self.parker_from_proj

        hu_to_mu_cfg = self.proj_volume_cfg.get("hu_to_mu", {}) if isinstance(self.proj_volume_cfg, dict) else {}
        self.gt_hu_to_mu_enabled = bool(hu_to_mu_cfg.get("enabled", False))
        clip_vals = hu_to_mu_cfg.get("hu_clip", [-1024.0, 3071.0])
        if isinstance(clip_vals, (list, tuple)) and len(clip_vals) >= 2:
            self.gt_hu_clip = (float(clip_vals[0]), float(clip_vals[1]))
        else:
            self.gt_hu_clip = (-1024.0, 3071.0)
        self.gt_mu_water = float(hu_to_mu_cfg.get("mu_water_mm_inv", 0.02))

    def run(
        self,
        sino_SAD: np.ndarray,
        *,
        case_id: str | int = "case",
        ground_truth: np.ndarray | None = None,
    ) -> np.ndarray:
        assert sino_SAD.ndim == 3, "sino must be [S, A, D]"
        self._prepare_geometry(sino_SAD)

        projector_id = astra.create_projector(self.projector, self.proj_geom, self.vol_geom)
        try:
            recon = self._reconstruct_volume(sino_SAD, projector_id)
        finally:
            astra.projector.delete(projector_id)

        recon = self._apply_post(recon)
        self._save_outputs(recon, case_id, ground_truth)
        return recon

    def _prepare_geometry(self, sino_SAD: np.ndarray) -> None:
        self.S, self.A, self.D = sino_SAD.shape
        self.angles = _angles_from_default(self.cfg)
        if self.angles.shape[0] != self.A:
            raise ValueError(f"[angles] length {self.angles.shape[0]} != sino A {self.A}")

        self.proj_geom = _proj_geom_from_default(
            self.cfg,
            self.angles,
            sino_D=self.D,
            spacing_dzyx=self.spacing_dzyx,
        )

        dy_mm, dx_mm = self._resolve_voxel_spacing()
        self.vol_geom = _vol_geom_from_fbp(self.cfg, dy_mm=dy_mm, dx_mm=dx_mm)
        self.algo, self.projector = _algo_and_projector(self.cfg)

    def _resolve_voxel_spacing(self) -> Tuple[Optional[float], Optional[float]]:
        if self.spacing_dzyx is not None:
            return float(self.spacing_dzyx[1]), float(self.spacing_dzyx[2])

        data_sp = self.cfg.get("data", {}).get("spacing", {})
        if bool(data_sp.get("enabled", False)) and str(data_sp.get("order", "")).lower() == "dzyx":
            vals = data_sp.get("values", None)
            if isinstance(vals, (list, tuple)) and len(vals) == 3:
                return float(vals[1]), float(vals[2])
        return None, None

    def _reconstruct_volume(self, sino_SAD: np.ndarray, projector_id: int) -> np.ndarray:
        vol_geom = self.vol_geom
        if vol_geom is None:
            raise RuntimeError("Volume geometry has not been prepared.")

        H = cast(int, vol_geom["GridRowCount"])
        W = cast(int, vol_geom["GridColCount"])
        recon = np.zeros((self.S, H, W), dtype=np.float32)

        for idx in range(self.S):
            recon[idx] = self._run_slice(sino_SAD[idx], projector_id)

        return recon

    def _run_slice(self, sino_slice: np.ndarray, projector_id: int) -> np.ndarray:
        sino_AD = np.asarray(sino_slice, dtype=np.float32)
        if sino_AD.shape != (self.A, self.D):
            if sino_AD.shape == (self.D, self.A):
                sino_AD = sino_AD.T
            else:
                raise ValueError(f"Unexpected slice sino shape: {sino_AD.shape}, expect {(self.A, self.D)}")

        proj_geom = self.proj_geom
        vol_geom = self.vol_geom
        algo = self.algo
        if proj_geom is None or vol_geom is None or algo is None:
            raise RuntimeError("Geometry or algorithm is not prepared before reconstruction.")

        sinogram_id = astra.data2d.create('-sino', proj_geom, sino_AD)
        recon_id = astra.data2d.create('-vol', vol_geom)

        cfg: Dict[str, Any] = astra.astra_dict(algo)
        cfg["ProjectionDataId"] = sinogram_id
        cfg["ReconstructionDataId"] = recon_id
        cfg["ProjectorId"] = projector_id

        options: Dict[str, Any] = {}
        self._configure_filter(options)
        self._configure_short_scan(options)
        if options:
            cfg["option"] = options

        alg_id = astra.algorithm.create(cfg)
        try:
            astra.algorithm.run(alg_id)
            result = astra.data2d.get(recon_id).astype(np.float32, copy=False)
        finally:
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(sinogram_id)
            astra.data2d.delete(recon_id)

        return result

    def _configure_filter(self, options: Dict[str, Any]) -> None:
        if self.filter_cfg.get("type"):
            options["FilterType"] = self.filter_cfg["type"]
            if self.filter_cfg.get("d_param") is not None:
                options["FilterD"] = float(self.filter_cfg["d_param"])
            if self.filter_cfg.get("parameter") is not None:
                options["FilterParameter"] = float(self.filter_cfg["parameter"])

    def _configure_short_scan(self, options: Dict[str, Any]) -> None:
        if self.short_scan_enabled:
            options["ShortScan"] = True
        if self.pixel_ss > 1:
            options["PixelSuperSampling"] = self.pixel_ss

    def _apply_post(self, recon: np.ndarray) -> np.ndarray:
        if self.mask_circle_enabled:
            recon = _apply_fov_mask_circle(recon, radius_factor=self.mask_radius_factor)

        mode = self.normalization_mode

        if mode == "minmax_per_slice":
            for s in range(recon.shape[0]):
                v = recon[s]
                vmin, vmax = float(np.min(v)), float(np.max(v))
                if vmax > vmin:
                    recon[s] = (v - vmin) / (vmax - vmin)
                else:
                    recon[s] = 0.0

        elif mode == "percentile":
            p_lo, p_hi = self.percentiles
            for s in range(recon.shape[0]):
                v = recon[s]
                lo = np.percentile(v, p_lo)
                hi = np.percentile(v, p_hi)
                if hi > lo:
                    recon[s] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
                else:
                    recon[s] = 0.0

        elif mode == "minmax":
            gmin = float(np.min(recon))
            gmax = float(np.max(recon))
            if gmax > gmin:
                recon = (recon - gmin) / (gmax - gmin)
            else:
                recon = np.zeros_like(recon, dtype=np.float32)

        elif mode == "fixed":
            lo, hi = self.fixed_range
            if hi <= lo:
                raise ValueError(f"[global_fixed] invalid fixed_range: {lo}, {hi}")
            recon = np.clip((recon - lo) / (hi - lo), 0.0, 1.0)

        elif mode in (None, "none"):
            pass

        else:
            raise ValueError(f"Unsupported normalize mode: {mode}")

        if self.mask_circle_enabled:
            recon = _apply_fov_mask_circle(recon, radius_factor=self.mask_radius_factor)

        if self.scale_factor != 1.0:
            recon = recon * self.scale_factor

        if self.clip_range is not None:
            lo, hi = float(self.clip_range[0]), float(self.clip_range[1])
            recon = np.clip(recon, lo, hi)

        return recon.astype(self.target_dtype, copy=False)

    def _convert_gt_hu_to_mu(self, gt: np.ndarray) -> np.ndarray:
        if not self.gt_hu_to_mu_enabled:
            return gt
        lo, hi = self.gt_hu_clip
        gt_clipped = np.clip(gt, lo, hi).astype(np.float32, copy=False)
        return self.gt_mu_water * (1.0 + gt_clipped / 1000.0)

    def _save_outputs(
        self,
        recon: np.ndarray,
        case_id: str | int | None,
        ground_truth: np.ndarray | None,
    ) -> None:
        io_cfg = self.io_cfg

        out_root_tpl = io_cfg.get("out_root", "outputs/FBP/{dataset_name}")
        out_root = Path(out_root_tpl.format(dataset_name=self.dataset_name))
        out_root.mkdir(parents=True, exist_ok=True)

        ang_cfg = self.cfg["projection"]["angles"]
        stop_deg = float(ang_cfg.get("stop_deg", 180.0))
        step_deg = float(ang_cfg.get("step_deg", 1.0))

        case_id_safe = str(case_id) if case_id not in (None, "", "None") else "nocase"

        prefix_tpl = io_cfg.get("case_prefix", "recon_{case_id}_{stop}@{step}")
        prefix = prefix_tpl.format(
            case_id=case_id_safe,
            stop=_fmt_num(stop_deg),
            step=_fmt_num(step_deg),
        )

        save_numpy_flag = io_cfg.get("save_numpy_all")
        if save_numpy_flag is None:
            save_numpy_flag = io_cfg.get("save_npz_all", True)
        if bool(save_numpy_flag):
            save_numpy_dir_tpl = io_cfg.get(
                "save_numpy_dir",
                io_cfg.get("save_npz_dir", "data/interim/recon/{dataset_name}")
            )
            save_numpy_dir = Path(save_numpy_dir_tpl.format(dataset_name=self.dataset_name))
            save_numpy_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_numpy_dir / f"{prefix}.npy"
            np.save(out_path, recon)
            print(f"[FBP] Saved reconstruction array to: {out_path}")

        if bool(io_cfg.get("save_png_each_slice", True)):
            if iio is None:
                print("[FBP] imageio 未安装，跳过 PNG 保存")
            else:
                png_dir = out_root / f"recon_{case_id_safe}" / prefix
                png_dir.mkdir(parents=True, exist_ok=True)

                arr = recon
                if arr.dtype.kind in "fc":
                    arr_to_write = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
                else:
                    arr_to_write = arr

                for s in range(arr.shape[0]):
                    iio.imwrite(png_dir / f"{s:04d}.png", arr_to_write[s])

                print(f"[FBP] Saved PNG slices to: {png_dir}")

        if ground_truth is not None and iio is not None:
            gt = np.asarray(ground_truth, dtype=np.float32)
            if gt.ndim != 3:
                raise ValueError(f"ground_truth_zyx must be [S,H,W], got shape {gt.shape}")

            gt_mu = self._convert_gt_hu_to_mu(gt)

            gt_dir = out_root / f"groundtruth_{case_id_safe}"
            gt_dir.mkdir(parents=True, exist_ok=True)

            for s in range(gt_mu.shape[0]):
                v = gt_mu[s]
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
    pipeline = _FBPReconstructionPipeline(cfg_merged, spacing_dzyx)
    return pipeline.run(sino_SAD, case_id=case_id, ground_truth=ground_truth_zyx)