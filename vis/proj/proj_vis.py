# vis/proj/proj_vis.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v2 as iio

def save_projs_png_uint8(
    projs: np.ndarray,
    angles_deg: np.ndarray,
    category: str = "default",
    save_path: str | Path | None = None,
    origin: str = "upper",
    per_frame: bool = False,  # True: 每帧单独缩放; False: 用全局 min/max 缩放
) -> list[Path]:
    """
    保存投影为 0–255 的 PNG (uint8 灰度)。

    参数
    ----
    projs: np.ndarray
        形状 (nAngles, H, W) 或 (H, W, nAngles)。
    angles_deg: np.ndarray
        投影角度 (度)。
    category: str
        默认保存目录 figs/projections/(category)/png_uint8。
    save_path: str | Path | None
        自定义保存目录。
    origin: {"upper","lower"}
        图像原点，"lower" 会上下翻转。
    per_frame: bool
        True = 每帧单独拉伸到 [0,255]。
        False = 全局用同一 min/max（跨角度可对比）。
    """
    if projs.ndim != 3:
        raise ValueError(f"projs 需为3维，当前 {projs.shape}")
    if projs.shape[0] == angles_deg.size:
        stack = projs
    elif projs.shape[-1] == angles_deg.size:
        stack = np.moveaxis(projs, -1, 0)
    else:
        raise ValueError(f"角度维不匹配：projs={projs.shape}, angles_deg={angles_deg.shape}")

    save_dir = Path(save_path) if save_path is not None else Path("figs") / "projections" / category / "png_uint8"
    save_dir.mkdir(parents=True, exist_ok=True)

    stack = stack.astype(np.float32, copy=False)
    saved: list[Path] = []

    # 全局 min/max
    if not per_frame:
        global_min, global_max = float(np.min(stack)), float(np.max(stack))

    for deg, img in zip(angles_deg, stack):
        arr = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if origin == "lower":
            arr = np.flipud(arr)

        if per_frame:
            lo, hi = float(np.min(arr)), float(np.max(arr))
        else:
            lo, hi = global_min, global_max

        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)  # 动态范围为0时全黑

        arr = arr.astype(np.uint8)

        fname = f"deg_{int(round(float(deg))):02d}.png"
        fpath = save_dir / fname
        iio.imwrite(fpath, arr)
        saved.append(fpath)

    return saved
