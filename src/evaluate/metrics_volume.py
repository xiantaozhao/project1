# src/evaluate/metrics_volume.py
from __future__ import annotations
import numpy as np
from pathlib import Path
import csv
from typing import Dict, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr  # 保留导入以兼容原实现

def _check_shapes(gt: np.ndarray, rec: np.ndarray) -> Tuple[int,int,int]:
    if gt.ndim != 3 or rec.ndim != 3:
        raise ValueError(f"expect [S,H,W], got gt{gt.shape}, rec{rec.shape}")
    if gt.shape != rec.shape:
        raise ValueError(f"shape mismatch: gt{gt.shape} vs rec{rec.shape}")
    return gt.shape  # type: ignore # (S,H,W)

def _stem_from_cfg(cfg: Optional[Dict], case_id: str | int) -> str:
    ang = (cfg or {}).get("projection", {}).get("angles", {})
    stop = ang.get("stop_deg", 180)
    step = ang.get("step_deg", 1)
    def fmt(x):
        try:
            xi = int(round(float(x)))
            return str(xi) if abs(float(x)-xi) < 1e-6 else str(x)
        except Exception:
            return str(x)
    case_s = str(case_id if case_id not in (None, "", "None") else "nocase")
    return f"{case_s}_{fmt(stop)}@{fmt(step)}"

def _center_circle_mask(H: int, W: int, ratio: float) -> np.ndarray:
    """
    生成中心圆形掩膜（bool）。ratio∈(0,1]，半径 = ratio * min(H,W)/2。
    """
    if not (0 < ratio <= 1.0):
        raise ValueError(f"center_circle_ratio must be in (0,1], got {ratio}")
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    r = ratio * min(H, W) / 2.0
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy)**2 + (xx - cx)**2 <= r**2
    if not np.any(mask):
        raise ValueError("center circle mask is empty; increase center_circle_ratio")
    return mask

def evaluate_ssim_psnr(
    gt: np.ndarray,             # [S,H,W]
    rec: np.ndarray,            # [S,H,W]
    *,
    cfg: Optional[Dict] = None,
    case_id: str | int = "nocase",
    save_dir: Optional[str | Path] = "outputs/FBP",
    center_circle_ratio: Optional[float] = None
) -> Dict:
    """
    简洁版: min-max 到[0,1]，再算 SSIM + PSNR。
    与小脚本口径一致。
    新增:
      - center_circle_ratio: 若给定 (0,1]，则只在中心圆内计算指标。
    """
    S,H,W = _check_shapes(gt, rec)
    gt = gt.astype(np.float32, copy=False)
    rec = rec.astype(np.float32, copy=False)

    # 预生成掩膜（若需要）
    mask = None
    if center_circle_ratio is not None:
        mask = _center_circle_mask(H, W, center_circle_ratio)

    ssim_list = np.zeros((S,), dtype=np.float32)
    psnr_list = np.zeros((S,), dtype=np.float32)

    # ===== 全局归一化 =====
    gmin_all, gmax_all = float(gt.min()), float(gt.max())
    rmin_all, rmax_all = float(rec.min()), float(rec.max())

    if gmax_all > gmin_all:
        gt_n = (gt - gmin_all) / (gmax_all - gmin_all)
    else:
        gt_n = np.zeros_like(gt)

    if rmax_all > rmin_all:
        rec_n = (rec - rmin_all) / (rmax_all - rmin_all)
    else:
        rec_n = np.zeros_like(rec)

    for s in range(S):
        g = gt_n[s]; r = rec_n[s]

        if mask is None:
            # -------- 原口径 --------
            # SSIM（标量）
            ssim_val = ssim(
                g, r,
                data_range=1.0,
                gaussian_weights=False,
                use_sample_covariance=True,
                win_size=None
            )
            # PSNR（标量）
            psnr_val = psnr(g, r, data_range=1.0)
        else:
            # -------- 只在中心圆区域计算 --------
            # SSIM：拿到 SSIM map，再在掩膜内平均
            ssim_map = ssim(
                g, r,
                data_range=1.0,
                gaussian_weights=False,
                use_sample_covariance=True,
                win_size=None,
                full=True
            )[1]
            ssim_val = float(np.mean(ssim_map[mask]))

            # PSNR：按掩膜手动计算 MSE -> PSNR
            diff = (g - r)[mask]
            mse = float(np.mean(diff * diff)) if diff.size > 0 else 0.0
            if mse <= 0.0:
                psnr_val = float("inf")
            else:
                psnr_val = float(10.0 * np.log10(1.0 / mse))  # data_range=1.0

        ssim_list[s] = ssim_val
        psnr_list[s] = psnr_val

    ssim_mean = float(np.mean(ssim_list))
    psnr_mean = float(np.mean(psnr_list))
    result = {
        "shape": (S,H,W),
        "ssim": {"mean": ssim_mean, "per_slice": ssim_list},
        "psnr": {"mean": psnr_mean, "per_slice": psnr_list},
        "mask": {
            "type": "center_circle" if mask is not None else "full",
            "center_circle_ratio": float(center_circle_ratio) if center_circle_ratio is not None else None
        }
    }

    # 保存
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir = out_dir / f"result_{case_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _stem_from_cfg(cfg, case_id)
        suffix = "" if center_circle_ratio is None else f"_centerR{center_circle_ratio}"

        with open(out_dir / f"ssim_{stem}{suffix}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slice","ssim"])
            for i, v in enumerate(ssim_list): w.writerow([i, float(v)])

        with open(out_dir / f"psnr_{stem}{suffix}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slice","psnr"])
            for i, v in enumerate(psnr_list): w.writerow([i, float(v)])

        with open(out_dir / f"summary_{stem}{suffix}.txt", "w") as f:
            f.write(f"Case: {case_id}\n")
            f.write(f"Shape: {S}x{H}x{W}\n")
            f.write(f"SSIM mean: {ssim_mean:.6f}\n")
            f.write(f"PSNR mean: {psnr_mean:.6f}\n")
            f.write("Mode: per-slice independent min-max, data_range=1.0\n")
            if center_circle_ratio is not None:
                f.write(f"Mask: center circle, ratio={center_circle_ratio}\n")

        print(f"[Metrics] saved to {out_dir} (stem={stem}{suffix})")

    return result
