# vis/proj/proj_vis.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from matplotlib.image import imsave  # 直接写文件，不走 pyplot

def save_projs_2d(
    projs: np.ndarray,
    angles_deg: np.ndarray,
    category: str = "default",
    save_path: str | Path | None = None,
    window: tuple[float, float] = (1, 99),
    origin: str = "upper",  # "upper" 或 "lower"
) -> list[Path]:
    """
    将投影逐张保存为灰度图，不显示。

    参数
    ----
    projs: np.ndarray
        形状 (nAngles, H, W) 或 (H, W, nAngles)。
    angles_deg: np.ndarray
        对应角度（度），长度 = nAngles。
    category: str
        类别名，用于默认保存目录 figs/projections/(category)。
    save_path: str | Path | None
        保存目录；None 时用默认 figs/projections/(category)。
    window: (p_low, p_high)
        统一窗宽的百分位，默认 (1, 99)。
    origin: {"upper","lower"}
        图像原点；保存时仅影响是否翻转。

    返回
    ----
    保存的文件路径列表。
    """
    # —— 统一形状为 (nAngles, H, W) ——
    if projs.ndim != 3:
        raise ValueError(f"projs 需为3维，当前 {projs.shape}")
    if projs.shape[0] == angles_deg.size:
        stack = projs
    elif projs.shape[-1] == angles_deg.size:
        stack = np.moveaxis(projs, -1, 0)
    else:
        raise ValueError(f"角度维不匹配：projs={projs.shape}, angles_deg={angles_deg.shape}")

    # —— 目标目录 —— 
    save_dir = Path(save_path) if save_path is not None else Path("figs") / "projections" / category / "projections"
    save_dir.mkdir(parents=True, exist_ok=True)

    # —— 统一窗宽（基于全部投影） ——
    arr_all = np.nan_to_num(stack.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p_low, p_high = np.percentile(arr_all, window)

    saved: list[Path] = []
    for deg, img in zip(angles_deg, stack):
        arr = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # 翻转原点（upper=默认；lower=上下翻转）
        if origin == "lower":
            arr = np.flipud(arr)

        # 窗宽 + 归一化到 [0,1]
        arr = np.clip(arr, p_low, p_high)
        arr = (arr - p_low) / (p_high - p_low + 1e-8)

        # 保存：deg_XX度.png
        fname = f"deg_{int(round(float(deg))):02d}.png"
        fpath = save_dir / fname
        # 直接写，不创建 Figure
        imsave(fpath, arr, cmap="gray", origin="upper")
        saved.append(fpath)

    return saved
