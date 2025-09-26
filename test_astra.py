# -*- coding: utf-8 -*-
# Compare μ-domain vs gray-domain with ASTRA projection+FBP
# pip install astra-toolbox numpy
from __future__ import annotations
import numpy as np

try:
    import astra
except ImportError as e:
    raise SystemExit("Please install astra-toolbox: pip install astra-toolbox") from e


def stats(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    print(f"{name:14s} | shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def make_mu_phantom(h=256, w=256, mu_bg=0.0000, mu_obj=0.0200):
    """简单圆形物体 + 小圆洞，单位：mm^-1（μ）"""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h/2, w/2
    r_big = min(h, w) * 0.30
    r_hole = min(h, w) * 0.10

    img = np.full((h, w), mu_bg, dtype=np.float32)
    mask_big  = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_big ** 2
    mask_hole = (xx - (cx + 0.12*w)) ** 2 + (yy - (cy - 0.10*h)) ** 2 <= r_hole ** 2
    img[mask_big] = mu_obj
    img[mask_hole] = mu_bg
    return img


def do_astra_pipeline(image_2d: np.ndarray,
                      geom: str = 'parallel',
                      det_spacing: float = 1.0,
                      det_count: int = 384,
                      n_angles: int = 360,
                      filter_type: str = 'Ram-Lak'):
    """image_2d -> sino -> FBP recon，返回 (sino, recon)"""
    H, W = image_2d.shape
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)

    if geom == 'parallel':
        proj_geom = astra.create_proj_geom('parallel', det_spacing, det_count, angles)
        vol_geom  = astra.create_vol_geom(H, W)
    else:
        raise NotImplementedError("Only 'parallel' demo here; fan-beam can be added similarly.")

    # CPU projector，避免 CUDA 依赖；FBP(CPU) 需要 ProjectorId
    projector_id = astra.create_projector('line', proj_geom, vol_geom)

    vol_id = astra.data2d.create('-vol', vol_geom, image_2d.astype(np.float32, copy=False))
    sino_id, sino = astra.create_sino(vol_id, projector_id)  # [A, D]

    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId']     = sino_id
    cfg['ProjectorId']          = projector_id
    cfg['FilterType']           = filter_type

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon = astra.data2d.get(rec_id)

    # 清理
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(vol_id)
    astra.projector.delete(projector_id)

    return sino, recon


def main():
    # ---------- Case 1: μ-domain ----------
    mu = make_mu_phantom(h=256, w=256, mu_bg=0.0000, mu_obj=0.0200)  # mm^-1
    stats("mu input", mu)
    sino_mu, recon_mu = do_astra_pipeline(mu)
    stats("sino (μ)", sino_mu)
    stats("recon (μ)", recon_mu)
    print(f"MSE(μ vs recon): {mse(mu, recon_mu):.8f}  Corr: {corr(mu, recon_mu):.6f}")
    print("-" * 80)

    # ---------- Case 2: Gray-domain ----------
    # 把同一张图归一化到 [0,1] 再拉伸到 [0,255] 当作“灰度”
    mu_min, mu_max = float(mu.min()), float(mu.max())
    gray = (mu - mu_min) / (mu_max - mu_min + 1e-12) * 255.0
    gray = gray.astype(np.float32)
    stats("gray input", gray)
    sino_gray, recon_gray = do_astra_pipeline(gray)
    stats("sino (gray)", sino_gray)
    stats("recon(gray)", recon_gray)
    print(f"MSE(gray vs recon): {mse(gray, recon_gray):.8f}  Corr: {corr(gray, recon_gray):.6f}")
    print("-" * 80)

    # ---------- Cross checks（展示线性但“物理上不对应”） ----------
    # 线性算子下，若 gray 是 mu 的仿射变换，则投影/FBP也近似做相应仿射（边界与滤波使得不完全理想）
    # 这里做个最简单的“最佳线性拟合”来对齐 recon_gray 与 gray：
    A = np.vstack([recon_gray.ravel(), np.ones(recon_gray.size)]).T
    coef, _, _, _ = np.linalg.lstsq(A, gray.ravel(), rcond=None)  # gray ≈ a*recon_gray + b
    a, b = coef
    linfit = a * recon_gray + b
    print(f"Linear fit gray ≈ a*recon_gray + b: a={a:.6f}, b={b:.6f}")
    print(f"MSE(gray vs a*recon_gray+b): {mse(gray, linfit):.8f}  Corr: {corr(gray, linfit):.6f}")

    # 对比 μ 与灰度重建之间的相关性（仅数学相关性，无物理意义）
    print(f"Corr(recon μ, recon gray): {corr(recon_mu, recon_gray):.6f}")


if __name__ == "__main__":
    main()
