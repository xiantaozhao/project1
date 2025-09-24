# src/evaluate/metrics_volume.py
from __future__ import annotations
import numpy as np
from pathlib import Path
import csv
from typing import Dict, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def _check_shapes(gt: np.ndarray, rec: np.ndarray) -> Tuple[int,int,int]:
    if gt.ndim != 3 or rec.ndim != 3:
        raise ValueError(f"expect [S,H,W], got gt{gt.shape}, rec{rec.shape}")
    if gt.shape != rec.shape:
        raise ValueError(f"shape mismatch: gt{gt.shape} vs rec{rec.shape}")
    return gt.shape  # (S,H,W)

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

def evaluate_ssim_psnr(
    gt: np.ndarray,             # [S,H,W]
    rec: np.ndarray,            # [S,H,W]
    *,
    cfg: Optional[Dict] = None,
    case_id: str | int = "nocase",
    save_dir: Optional[str | Path] = "outputs/FBP"
) -> Dict:
    """
    简洁版：逐 slice 独立 min-max 到[0,1]，再算 SSIM + PSNR。
    与小脚本口径一致。
    """
    S,H,W = _check_shapes(gt, rec)
    gt = gt.astype(np.float32, copy=False)
    rec = rec.astype(np.float32, copy=False)

    ssim_list = np.zeros((S,), dtype=np.float32)
    psnr_list = np.zeros((S,), dtype=np.float32)

    for s in range(S):
        g = gt[s]; r = rec[s]
        gmin, gmax = float(g.min()), float(g.max())
        rmin, rmax = float(r.min()), float(r.max())
        g = (g - gmin) / (gmax - gmin + 1e-6) if gmax > gmin else np.zeros_like(g)
        r = (r - rmin) / (rmax - rmin + 1e-6) if rmax > rmin else np.zeros_like(r)

        # SSIM
        ssim_list[s] = ssim(
            g, r,
            data_range=1.0,
            gaussian_weights=False,
            use_sample_covariance=True,
            win_size=None
        )
        # PSNR
        psnr_list[s] = psnr(g, r, data_range=1.0)

    ssim_mean = float(np.mean(ssim_list))
    psnr_mean = float(np.mean(psnr_list))
    result = {
        "shape": (S,H,W),
        "ssim": {"mean": ssim_mean, "per_slice": ssim_list},
        "psnr": {"mean": psnr_mean, "per_slice": psnr_list},
    }

    # 保存
    if save_dir is not None:
        out_dir = Path(save_dir); out_dir.mkdir(parents=True, exist_ok=True)
        stem = _stem_from_cfg(cfg, case_id)

        with open(out_dir / f"ssim_{stem}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slice","ssim"])
            for i, v in enumerate(ssim_list): w.writerow([i, float(v)])

        with open(out_dir / f"psnr_{stem}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slice","psnr"])
            for i, v in enumerate(psnr_list): w.writerow([i, float(v)])

        with open(out_dir / f"summary_{stem}.txt", "w") as f:
            f.write(f"Case: {case_id}\n")
            f.write(f"Shape: {S}x{H}x{W}\n")
            f.write(f"SSIM mean: {ssim_mean:.6f}\n")
            f.write(f"PSNR mean: {psnr_mean:.6f}\n")
            f.write("Mode: per-slice independent min-max, data_range=1.0\n")

    return result
