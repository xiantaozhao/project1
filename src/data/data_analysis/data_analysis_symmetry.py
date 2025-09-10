"""
Data Analysis Utilities for Projection Symmetry

This module provides reusable functions to:
- Load PNG projections named like `deg_00.png`, `deg_03.png`, ...
- Compute symmetry metrics between angle θ and (θ+180) with optional horizontal flip
- Export metrics to CSV
- Generate visualizations (histograms/distributions, angle-curves, scatter plots, and example image grids)

Usage (CLI):
    python -m src.data.data_analysis.data_analysis_symmetry \
        --input ./figs/projections/chest/projections \
        --output ./figs/projections/chest/analysis \
        --pattern "deg_*.png" \
        --dtype uint8 \
        --data-range auto \
        --flip180
        
    python -m src.data.data_analysis.data_analysis_symmetry \
        --input ./figs/projections/chest/parallel \
        --output ./figs/projections/chest/parallel_analysis \
        --pattern "deg_*.png" \
        --dtype uint8 \
        --data-range auto \
        --flip180

    python -m src.data.data_analysis.data_analysis_symmetry \
        --input ./figs/projections/chest/projections \
        --output ./figs/projections/chest/analysis_masked \
        --pattern "deg_*.png" \
        --dtype uint8 \
        --data-range auto \
        --flip180 \
        --mask-dir ./figs/projections/chest/mask \
        --mask-pattern "deg_*.csv"

    python -m src.data.data_analysis.data_analysis_symmetry \
        --input ./figs/projections/chest/projections \
        --output ./figs/projections/chest/analysis_masked_edge \
        --pattern "deg_*.png" \
        --dtype uint8 \
        --data-range auto \
        --flip180 \
        --mask-dir ./figs/projections/chest/mask_edge \
        --mask-pattern "deg_*.csv"

Dependencies:
    pip install numpy pandas pillow scikit-image matplotlib
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, cast

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# -----------------------------
# I/O & Parsing
# -----------------------------

ANGLE_REGEX = re.compile(r"deg[_-]?(\d{1,3})\.(?:png|PNG)$")

def read_mask_csv(path: Path) -> np.ndarray:
    """
    读取 mask CSV（0/1），返回二维 numpy 数组（0 或 1）。
    对于非0的数统一当作1，保证鲁棒。
    """
    m = pd.read_csv(path, header=None).to_numpy()
    # 任何非0视为1
    m = (m != 0).astype(np.uint8)
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got shape={m.shape} at {path}")
    return m

def load_masks(folder: Path, pattern: str = "deg_*.csv") -> Dict[int, np.ndarray]:
    """
    遍历 mask 目录，返回 angle->mask 的映射。
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    angle_to_mask: Dict[int, np.ndarray] = {}
    for f in files:
        ang = extract_angle_from_filename(f.name.replace(".csv", ".png"))
        if ang is None:
            continue
        angle_to_mask[ang] = read_mask_csv(f)
    return angle_to_mask

def apply_masks_to_projections(
    angle_to_img: Dict[int, np.ndarray],
    angle_to_mask: Dict[int, np.ndarray],
    strict_shape: bool = False,
    verbose: bool = True,
) -> Dict[int, np.ndarray]:
    """
    将同角度的 mask 应用到投影：img = img * mask
    - mask 为 0/1。若 img 为 float32，mask 会转为 float32；若 img 为 uint8，mask 为 uint8。
    - 形状不一致时：strict_shape=True 抛异常；否则跳过该角度并提示。
    返回新的 angle->img 字典（不在原地修改）。
    """
    out: Dict[int, np.ndarray] = {}
    for ang, img in angle_to_img.items():
        if ang not in angle_to_mask:
            out[ang] = img  # 没有对应 mask，保持原样
            continue

        m = angle_to_mask[ang]
        if m.shape != img.shape:
            msg = f"[mask] angle {ang}: shape mismatch mask={m.shape} vs img={img.shape}"
            if strict_shape:
                raise ValueError(msg)
            if verbose:
                print("WARN:", msg, "-> skip applying mask for this angle.")
            out[ang] = img
            continue

        if img.dtype == np.float32:
            mm = m.astype(np.float32)
        else:  # uint8
            mm = m.astype(np.uint8)

        out[ang] = (img * mm)
    return out


def extract_angle_from_filename(name: str) -> Optional[int]:
    """Extract integer angle from filename like 'deg_03.png' -> 3.
    Returns None if not found.
    """
    m = ANGLE_REGEX.search(name)
    if not m:
        return None
    return int(m.group(1)) % 360


def read_png_gray(path: Path, dtype: str = "uint8") -> np.ndarray:
    """Read a PNG as grayscale array of given dtype ("uint8" or "float32").
    - If dtype == "uint8": values in [0,255]
    - If dtype == "float32": values normalized to [0,1]
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    if dtype == "float32":
        return (arr.astype(np.float32) / 255.0)
    return arr.astype(np.uint8)


def load_projections(folder: Path, pattern: str = "deg_*.png", dtype: str = "uint8") -> Dict[int, np.ndarray]:
    """Load all PNGs in folder matching pattern, keyed by angle (0-359).
    Later files with same angle overwrite earlier ones.
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    angle_to_img: Dict[int, np.ndarray] = {}
    for f in files:
        ang = extract_angle_from_filename(f.name)
        if ang is None:
            continue
        angle_to_img[ang] = read_png_gray(f, dtype=dtype)
    return angle_to_img


# -----------------------------
# Metrics
# -----------------------------

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = (a.astype(np.float64) - b.astype(np.float64))
    return float(np.mean(diff * diff))


def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    ax = a.astype(np.float64).ravel()
    bx = b.astype(np.float64).ravel()
    ax -= ax.mean()
    bx -= bx.mean()
    denom = np.linalg.norm(ax) * np.linalg.norm(bx)
    return float(np.dot(ax, bx) / denom) if denom != 0 else 0.0


# -----------------------------
# Pairing & Evaluation
# -----------------------------

def compute_pair_metrics(
    angle_to_img: Dict[int, np.ndarray],
    flip180: bool = True,
    data_range: str | float = "auto",  # "auto", 1.0, or 255
) -> pd.DataFrame:
    """Compute metrics for all available (θ, θ+180) pairs.

    Args:
        angle_to_img: dict angle->image
        flip180: if True, horizontally flip the (θ+180) image for alignment
        data_range: 'auto' to infer from dtype; else numeric like 1.0 or 255
    Returns:
        DataFrame with metrics for each unique pair
    """
    rows: List[dict] = []
    present = sorted(angle_to_img.keys())

    # 显式为固定长度的 Tuple[int,int]，避免 Pylance 认为是可变长元组
    seen: Set[Tuple[int, int]] = set()

    for a in present:
        b = (a + 180) % 360
        if b not in angle_to_img:
            continue
        key: Tuple[int, int] = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)

        A = angle_to_img[a]
        B = angle_to_img[b]
        if A.shape != B.shape:
            # skip mismatched shapes
            continue

        B_aligned = np.fliplr(B) if flip180 else B
        flip_used = "flip_lr" if flip180 else "none"

        # Determine data range
        if data_range == "auto":
            dr: float = 1.0 if A.dtype == np.float32 else 255.0
        else:
            dr = float(data_range)

        # ssim 的类型提示常见歧义：若未显式 full=False，Pylance 可能推断成返回 (float, ndarray)
        ssim_val = cast(float, ssim(A, B_aligned, data_range=dr, channel_axis=None, full=False))
        # psnr 的类型提示也强制转成 float 更稳妥
        psnr_val = float(psnr(A, B_aligned, data_range=dr))

        rows.append({
            "deg_a": int(a),
            "deg_b": int(b),
            "h": int(A.shape[0]),
            "w": int(A.shape[1]),
            "flip_used": flip_used,
            "MAE": mae(A, B_aligned),
            "MSE": mse(A, B_aligned),
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "Pearson": pearsonr(A, B_aligned),
        })

    df = pd.DataFrame(rows).sort_values(["deg_a"]).reset_index(drop=True)
    return df


# -----------------------------
# Visualization Helpers
# -----------------------------

def ensure_outdir(outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _to_numpy_float(col: pd.Series) -> np.ndarray:
    """Robustly convert a pandas Series (possibly ExtensionArray) to np.ndarray[float]."""
    # to_numpy(copy=False) + dtype=float 避免 Pylance 对 pandas ArrayLike 的报错
    return cast(np.ndarray, col.to_numpy(dtype=float, copy=False))


def plot_metric_distributions(df: pd.DataFrame, outdir: Path, bins: int = 30) -> None:
    """Histogram distributions for all metrics."""
    outdir = ensure_outdir(outdir)
    metrics = ["MAE", "MSE", "PSNR", "SSIM", "Pearson"]
    for m in metrics:
        vals = _to_numpy_float(df[m])
        plt.figure()
        plt.hist(vals, bins=int(bins))
        plt.title(f"Distribution of {m}")
        plt.xlabel(m)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outdir / f"dist_{m}.png", dpi=150)
        plt.close()


def plot_pair_hist_cdf(
    angle_to_img: Dict[int, np.ndarray],
    df: pd.DataFrame,
    outdir: Path,
    bins: Optional[int] = None,
    flip180: bool = True,
    top_k: Optional[int] = None,
    sort_by: str = "MAE",
    largest: bool = True,
    verbose: bool = False,
) -> int:
    """
    为每个 (θ, θ+180) 对生成直方图+CDF 对比图（左θ，右θ+180对齐）。
    返回成功保存的图片数量。
    """
    outdir = ensure_outdir(outdir)

    # 1) 基本校验
    if df is None or df.empty:
        raise ValueError("[plot_pair_hist_cdf] df 为空：没有可画的配对。请检查是否生成了 metrics_180pairs。")

    # 2) 选择子集
    pairs_df = df.copy()
    if top_k is not None and sort_by in pairs_df.columns:
        pairs_df = pairs_df.sort_values(sort_by, ascending=not largest).head(top_k)

    if pairs_df.empty:
        raise ValueError(f"[plot_pair_hist_cdf] 经过筛选后为空（top_k={top_k}, sort_by={sort_by}）。")

    saved = 0

    for row in pairs_df.itertuples(index=False):
        a, b = int(row.deg_a), int(row.deg_b)
        if a not in angle_to_img or b not in angle_to_img:
            if verbose:
                print(f"[plot_pair_hist_cdf] 跳过 {a} vs {b}：角度数据缺失。")
            continue

        A = angle_to_img[a]
        B = angle_to_img[b]
        if A.shape != B.shape:
            if verbose:
                print(f"[plot_pair_hist_cdf] 跳过 {a} vs {b}：尺寸不一致 {A.shape} vs {B.shape}")
            continue

        B_aligned = np.fliplr(B) if flip180 else B

        # 3) 设定范围 & bins
        is_float = np.issubdtype(A.dtype, np.floating)
        vmin, vmax = (0.0, 1.0) if is_float else (0.0, 255.0)
        nbins = int(bins) if bins is not None else (100 if is_float else 256)

        def _hist_cdf(x: np.ndarray):
            vals = x.ravel().astype(np.float64)
            # 防止全常量导致边界奇异
            if np.all(vals == vals[0]):
                # 人工加一点点抖动，避免width为0
                vals = vals + 1e-12 * np.random.randn(*vals.shape)
            h, edges = np.histogram(vals, bins=nbins, range=(vmin, vmax), density=True)
            dx = np.diff(edges).astype(np.float64)
            # 防御：dx 有可能全0（极端情况），加一个极小值
            dx[dx == 0] = 1e-12
            c = np.cumsum(h * dx)
            c = np.clip(c, 0, 1)  # 数值稳定
            # 用区间中心画bar
            centers = (edges[:-1] + edges[1:]) / 2.0
            return centers, h, c

        xA, hA, cA = _hist_cdf(A)
        xB, hB, cB = _hist_cdf(B_aligned)

        # 4) 画图并保存（用 subplots 避免 plt.subplot 的类型歧义）
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

        def _panel(ax, x, h, c, title):
            # 直方图
            width = float((x[1] - x[0])) if x.size >= 2 else float((vmax - vmin) / max(nbins, 1))
            ax.bar(x, h, width=width, alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel("Pixel value")
            ax.set_ylabel("Density")
            # CDF（第二坐标轴）
            ax2 = ax.twinx()
            ax2.plot(x, c)
            ax2.set_ylabel("CDF")
            ax2.set_ylim(0, 1)

        _panel(axes[0], xA, hA, cA, f"Histogram + CDF — {a}°")
        _panel(axes[1], xB, hB, cB, f"Histogram + CDF — {b}° (aligned)")

        fig.suptitle(f"Pair {a}° vs {b}° — bins={nbins}")
        fig.tight_layout()

        out_path = outdir / f"histcdf_pair_{a:03d}_{b:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close()

        saved += 1
        if verbose:
            print(f"[plot_pair_hist_cdf] Saved -> {out_path.resolve()}")

    if verbose and saved == 0:
        print("[plot_pair_hist_cdf] 没有任何图片保存。请检查：角度是否成对、尺寸是否一致、筛选是否过严。")

    return saved


def plot_pair_distributions(
    angle_to_img: Dict[int, np.ndarray],
    df: pd.DataFrame,
    outdir: Path,
    bins: int = 50,
    flip180: bool = True,
) -> None:
    """
    对每个 (θ, θ+180) 对，画灰度值直方图对比：
    - 左：θ 图的灰度直方图
    - 右：θ+180 (可选 flip) 图的灰度直方图
    """
    outdir = ensure_outdir(outdir)
    for row in df.itertuples(index=False):
        a, b = int(row.deg_a), int(row.deg_b)
        A = angle_to_img[a]
        B = angle_to_img[b]
        B_aligned = np.fliplr(B) if flip180 else B

        # 改为 subplots，避免 plt.subplot 引起的 ConvertibleToInt 误报
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].hist(A.ravel().astype(np.float64), bins=int(bins), alpha=0.7)
        axs[0].set_title(f"Histogram of {a}°")
        axs[0].set_xlabel("Pixel value")
        axs[0].set_ylabel("Count")

        axs[1].hist(B_aligned.ravel().astype(np.float64), bins=int(bins), alpha=0.7)
        axs[1].set_title(f"Histogram of {b}° (aligned)")
        axs[1].set_xlabel("Pixel value")
        axs[1].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(outdir / f"hist_pair_{a}_{b}.png", dpi=150)
        plt.close(fig)


def plot_metric_vs_angle(df: pd.DataFrame, outdir: Path) -> None:
    """Line plots of metric vs deg_a."""
    outdir = ensure_outdir(outdir)
    metrics = ["MAE", "MSE", "PSNR", "SSIM", "Pearson"]
    x = _to_numpy_float(df["deg_a"])
    for m in metrics:
        y = _to_numpy_float(df[m])
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.title(f"{m} vs Angle")
        plt.xlabel("Angle (deg)")
        plt.ylabel(m)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"angle_{m}.png", dpi=150)
        plt.close()


def plot_scatter_relations(df: pd.DataFrame, outdir: Path) -> None:
    """Scatter plots for common metric relationships."""
    outdir = ensure_outdir(outdir)
    pairs = [("MAE", "PSNR"), ("MAE", "SSIM"), ("MSE", "PSNR"), ("SSIM", "Pearson")]
    for xk, yk in pairs:
        xv = _to_numpy_float(df[xk])
        yv = _to_numpy_float(df[yk])
        plt.figure()
        plt.scatter(xv, yv, s=18)
        plt.title(f"{yk} vs {xk}")
        plt.xlabel(xk)
        plt.ylabel(yk)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"scatter_{xk}_vs_{yk}.png", dpi=150)
        plt.close()


def save_example_pairs(angle_to_img: Dict[int, np.ndarray], df: pd.DataFrame, outdir: Path, k: int = 6, flip180: bool = True) -> None:
    """Save example image grids for some pairs (lowest MAE / highest SSIM)."""
    outdir = ensure_outdir(outdir)
    # pick top-k best (by SSIM) and worst (by MAE)
    best = df.sort_values("SSIM", ascending=False).head(k)
    worst = df.sort_values("SSIM", ascending=True).head(k)

    def _plot_rows(rows: pd.DataFrame, tag: str):
        if rows.empty:
            return
        n = len(rows)
        plt.figure(figsize=(6, 3 * n))
        for i, row in enumerate(rows.itertuples(index=False), start=1):
            a = int(row.deg_a); b = int(row.deg_b)
            A = angle_to_img[a]
            B = angle_to_img[b]
            B_aligned = np.fliplr(B) if flip180 else B

            # show A, B_aligned, and absolute diff
            diff = np.abs(A.astype(np.float32) - B_aligned.astype(np.float32))

            for j, (img, title) in enumerate(((A, f"{a}°"), (B_aligned, f"{b}° (aligned)"), (diff, "|A-B|")), start=1):
                ax = plt.subplot(n, 3, (i - 1) * 3 + j)
                ax.imshow(img, cmap="gray")
                ax.set_title(title)
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(outdir / f"examples_{tag}.png", dpi=150)
        plt.close()

    _plot_rows(best, "best")
    _plot_rows(worst, "worst")


# -----------------------------
# CLI Entrypoint
# -----------------------------

def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    dtype: str,
    data_range: str | float,
    flip180: bool,
    mask_dir: Optional[Path] = None,
    mask_pattern: str = "deg_*.csv",
) -> Path:
    output_dir = ensure_outdir(output_dir)
    angle_to_img = load_projections(input_dir, pattern=pattern, dtype=dtype)
    if not angle_to_img:
        raise FileNotFoundError(f"No images matching {pattern} under {input_dir}")

    # ====== 可选：读取并应用 mask ======
    if mask_dir is not None:
        masks = load_masks(mask_dir, pattern=mask_pattern)
        if masks:
            print(f"Applying masks from: {mask_dir} (found {len(masks)})")
            angle_to_img = apply_masks_to_projections(angle_to_img, masks, strict_shape=False, verbose=True)
        else:
            print(f"No masks matched {mask_pattern} under {mask_dir}; proceed without masking.")

    df_metrics = compute_pair_metrics(angle_to_img, flip180=flip180, data_range=data_range)
    metrics_csv = output_dir / "metrics_180pairs.csv"
    df_metrics.to_csv(metrics_csv, index=False)

    # Visualizations
    print("Generating plots...")
    plot_metric_distributions(df_metrics, output_dir / "plots")
    plot_metric_vs_angle(df_metrics, output_dir / "plots")
    plot_scatter_relations(df_metrics, output_dir / "plots")
    save_example_pairs(angle_to_img, df_metrics, output_dir / "plots", k=6, flip180=flip180)

    print("Generating pair histograms...")
    plot_pair_distributions(angle_to_img, df_metrics, output_dir / "pair_hists", bins=50, flip180=flip180)

    print("Generating pair hist+CDF plots...")
    plot_pair_hist_cdf(angle_to_img, df_metrics, output_dir / "pair_hist_cdf", bins=None, flip180=flip180, top_k=None, sort_by="MAE", largest=True)

    return metrics_csv



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Projection symmetry analysis (θ vs θ+180)")
    p.add_argument("--input", type=str, required=True, help="Folder containing deg_XX.png files")
    p.add_argument("--output", type=str, required=True, help="Output folder for CSV and plots")
    p.add_argument("--pattern", type=str, default="deg_*.png", help="Glob pattern for PNG files")
    p.add_argument("--dtype", type=str, choices=["uint8", "float32"], default="uint8", help="Array dtype when reading PNGs")
    p.add_argument("--data-range", type=str, default="auto", help="'auto', '1.0', or '255'")
    p.add_argument("--flip180", action="store_true", help="Flip (θ+180) image horizontally before comparison")

    # -------- 新增：mask 相关，可选 --------
    p.add_argument("--mask-dir", type=str, default=None,
                   help="Optional folder containing mask CSV files (deg_*.csv). If not provided, masks are not used.")
    p.add_argument("--mask-pattern", type=str, default="deg_*.csv",
                   help="Glob pattern for mask CSV files inside --mask-dir (default: deg_*.csv)")
    return p



def main():
    parser = build_argparser()
    args = parser.parse_args()

    data_range: str | float
    if args.data_range.lower() == "auto":
        data_range = "auto"
    else:
        try:
            data_range = float(args.data_range)
        except ValueError:
            raise ValueError("--data-range must be 'auto' or a number like 1.0 or 255")

    metrics_csv = run_pipeline(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        pattern=args.pattern,
        dtype=args.dtype,
        data_range=data_range,
        flip180=args.flip180,
        mask_dir=Path(args.mask_dir) if args.mask_dir is not None else None,
        mask_pattern=args.mask_pattern,
    )
    print(f"Saved metrics to: {metrics_csv}")


if __name__ == "__main__":
    main()
