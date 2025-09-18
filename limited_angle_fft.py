# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from typing import Tuple, List, Dict
from pathlib import Path
import os
import pydicom

# ---------------------------
# Config
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "output" / "limited_angle_fft"
os.makedirs(OUTDIR, exist_ok=True)

USE_DICOM = True  # whether to load a real DICOM slice from LIDC-IDRI for demo

IMG_SIZE = (256, 256)   # 如果 USE_DICOM=True，会用 DICOM 尺寸
PIXEL_SIZE = 1.0        # mm (未直接使用，仅保留)

# Angle ranges to test (inclusive endpoints), and a list of N views
RANGES = {
    "0_90": (0.0, 90.0),
    "90_180": (90.0, 180.0),
    "0_180": (0.0, 180.0),
    "0_360": (0.0, 360.0),
}
N_LIST = [50, 100, 150, 200, 250, 300]
range_names = list(RANGES.keys())

# Frequency sector settings
NUM_SECTORS = 12  # 180° 上 12 个桶（每个 15°）
LOW_FRAC  = (0.0,  0.33)  # 低频半径带（相对 Nyquist 半径）
MID_FRAC  = (0.33, 0.66)
HIGH_FRAC = (0.66, 1.00)

# 统一图像显示的灰度范围（跨图可比），用 GT 的 min/max
GLOBAL_VMIN = None
GLOBAL_VMAX = None

plt.rcParams["figure.dpi"] = 120

# ---------------------------
# Phantom (256x256) for fallback
# ---------------------------
import numpy as np
from typing import Tuple

def make_hu_phantom(size: Tuple[int, int]) -> np.ndarray:
    H, W = size
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0

    img = np.full((H, W), -1000.0, dtype=np.float32)  # background air

    # ---- Big soft-tissue ellipse (中心对称, 居中) ----
    a1, b1 = 0.38*W, 0.5*H
    mask1 = (((xx-cx)/a1)**2 + ((yy-cy)/b1)**2) <= 1.0
    img[mask1] = 40.0

    # ---- Bone-like ellipse (左右对称) ----
    a2, b2 = 0.12*W, 0.18*H
    dx2, dy2 = 0.12*W, 0.15*H
    # 左
    mask2L = (((xx-(cx-dx2))/a2)**2 + ((yy-(cy-dy2))/b2)**2) <= 1.0
    # 右
    mask2R = (((xx-(cx+dx2))/a2)**2 + ((yy-(cy-dy2))/b2)**2) <= 1.0
    img[mask2L | mask2R] = 800.0

    # ---- Rotated ellipse (左右对称) ----
    a3, b3 = 0.10*W, 0.08*H
    angle = np.deg2rad(25.0)
    dx3, dy3 = 0.15*W, 0.10*H
    # 左
    xrL = (xx-(cx-dx3))*np.cos(angle) + (yy-(cy+dy3))*np.sin(angle)
    yrL = -(xx-(cx-dx3))*np.sin(angle) + (yy-(cy+dy3))*np.cos(angle)
    mask3L = (xrL/a3)**2 + (yrL/b3)**2 <= 1.0
    # 右
    xrR = (xx-(cx+dx3))*np.cos(angle) + (yy-(cy+dy3))*np.sin(angle)
    yrR = -(xx-(cx+dx3))*np.sin(angle) + (yy-(cy+dy3))*np.cos(angle)
    mask3R = (xrR/a3)**2 + (yrR/b3)**2 <= 1.0
    img[mask3L | mask3R] = 450.0

    # ---- Low-density ellipse (上下对称, 居中) ----
    a4, b4 = 0.20*W, 0.12*H
    # 上
    mask4U = (((xx-cx)/a4)**2 + ((yy-(cy-0.12*H))/b4)**2) <= 1.0
    # 下
    mask4D = (((xx-cx)/a4)**2 + ((yy-(cy+0.12*H))/b4)**2) <= 1.0
    img[mask4U | mask4D] = -80.0

    return img

# ---------------------------
# Load GT
# ---------------------------
if USE_DICOM:
    dcm_path = BASE_DIR / "data/raw/chest/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-090.dcm"
    ds = pydicom.dcmread(dcm_path)
    img_raw = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    GT = img_raw * slope + intercept
else:
    GT = make_hu_phantom(IMG_SIZE)

GLOBAL_VMIN = float(GT.min())
GLOBAL_VMAX = float(GT.max())

# ---------------------------
# Utils: FFT coordinates / masks / recon
# ---------------------------
def build_frequency_coords(h: int, w: int):
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    KX, KY = np.meshgrid(kx, ky)
    R = np.sqrt(KX**2 + KY**2)
    THETA = np.arctan2(KY, KX)   # [-pi, pi]
    THETA_MOD = np.mod(THETA, np.pi)  # [0, pi)
    return KX, KY, R, THETA_MOD

def angular_distance(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, np.pi - d)

def angular_mask(h: int, w: int, angles_deg: np.ndarray) -> np.ndarray:
    """
    频域角扇形掩膜：每个角度对应一条扇形，半宽 ~ 角步长的一半。
    """
    _, _, R, TH = build_frequency_coords(h, w)
    if len(angles_deg) < 2:
        half_width = np.deg2rad(0.5)
    else:
        spacings = np.diff(np.sort(angles_deg))
        spacings = np.append(spacings, (180 - (angles_deg.max() - angles_deg.min())))
        half_width = np.deg2rad(np.median(spacings) / 2.0)
        half_width = max(half_width, np.deg2rad(0.5))

    mask = np.zeros((h, w), dtype=bool)
    for ang in angles_deg:
        a = np.deg2rad(ang) % np.pi
        mask |= (angular_distance(TH, a) <= half_width)
    # 始终保留 DC 附近一小圈，避免振铃
    r_keep = 2.0
    mask |= (R <= r_keep)
    return mask

def reconstruct_via_fft_mask(gt_img: np.ndarray, angles_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = gt_img.shape
    F = np.fft.fft2(gt_img)
    mask = angular_mask(H, W, angles_deg)
    F_masked = F * mask
    rec = np.fft.ifft2(F_masked).real
    return rec, mask

# ---------------------------
# Metrics: SSIM（全图）/ ROI-SSIM / Coverage
# ---------------------------
def global_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    简化版 SSIM（全图、不加窗口）。L 取 HU 动态范围的经验尺度。
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = x.mean(); mu_y = y.mean()
    vx = x.var();   vy = y.var()
    cxy = ((x - mu_x) * (y - mu_y)).mean()
    L = 2200.0   # ~[-1000, 1200] 的量级
    k1, k2 = 0.01, 0.03
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    num = (2*mu_x*mu_y + C1) * (2*cxy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (vx + vy + C2)
    return float(num / den)

def global_ssim_masked(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """ROI-SSIM：只在 mask=True 的区域上评估（不做窗口化）。"""
    xm = x[mask].astype(np.float64)
    ym = y[mask].astype(np.float64)
    if xm.size == 0:
        return np.nan
    mu_x = xm.mean(); mu_y = ym.mean()
    vx = xm.var();   vy = ym.var()
    cxy = ((xm - mu_x) * (ym - mu_y)).mean()
    L = 2200.0
    k1, k2 = 0.01, 0.03
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    num = (2*mu_x*mu_y + C1) * (2*cxy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (vx + vy + C2)
    return float(num / den)

def try_otsu(img: np.ndarray) -> float:
    """尽量用 Otsu 阈值，失败则回退到 25% 分位数。"""
    try:
        from skimage.filters import threshold_otsu
        return float(threshold_otsu(img.astype(np.float32)))
    except Exception:
        return float(np.quantile(img, 0.25))

def get_roi_mask_from_gt(gt_img: np.ndarray) -> np.ndarray:
    thr = try_otsu(gt_img)
    roi = gt_img > thr
    if not np.any(roi):
        roi = np.ones_like(gt_img, dtype=bool)
    return roi

def sector_retention(rec_img: np.ndarray, gt_img: np.ndarray, num_sectors: int = 12,
                     r_frac_band: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, float]:
    """
    频域每个扇区的能量保留：|F_rec|^2 / |F_gt|^2。
    返回 (每桶保留度, 覆盖打分=扇区平均)。
    """
    H, W = rec_img.shape
    _, _, R, TH = build_frequency_coords(H, W)
    r = R / (R.max() + 1e-12)

    F_rec = np.fft.fft2(rec_img)
    F_gt  = np.fft.fft2(gt_img)
    P_rec = np.abs(F_rec)**2
    P_gt  = np.abs(F_gt)**2 + 1e-12

    r0, r1 = r_frac_band
    band_mask = (r >= r0) & (r <= r1)

    edges = np.linspace(0.0, np.pi, num_sectors+1)
    ret = np.zeros(num_sectors, dtype=np.float64)
    for k in range(num_sectors):
        th0, th1 = edges[k], edges[k+1]
        sect_mask = (TH >= th0) & (TH < th1)
        m = band_mask & sect_mask
        nume = P_rec[m].sum()
        deno = P_gt[m].sum()
        ret[k] = float(nume / deno) if deno > 0 else 0.0

    coverage = float(ret.mean())
    return ret, coverage

# ---------------------------
# Plot Helpers
# ---------------------------
def save_image(img: np.ndarray, path: Path, title: str = ""):
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)
    if title:
        plt.title(title)
    plt.axis('off')
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_fft_image(img: np.ndarray, path: Path, title: str = ""):
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(F))
    plt.figure()
    plt.imshow(mag, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def radar_plot(values_dict: Dict[str, np.ndarray], path: Path, title: str):
    labels = [f"S{k}" for k in range(NUM_SECTORS)]
    angles = np.linspace(0, 2*np.pi, NUM_SECTORS, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure()
    ax = plt.subplot(111, polar=True)
    for name, vals in values_dict.items():
        data = vals.tolist(); data += data[:1]
        ax.plot(angles, data, linewidth=1.5, label=name)
    ax.set_xticks(np.linspace(0, 2*np.pi, NUM_SECTORS, endpoint=False))
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def bar_plot(values: np.ndarray,
             labels: List[str],
             path: Path,
             title: str,
             ylabel: str,
             rotation: int = 45,   # 旋转角度（度）
             show_every: int = 1    # 每隔多少个标签显示一个（1=全显示）
             ):
    n = len(values)
    # 根据标签数自适应图宽，避免拥挤
    fig_w = max(7, 0.55 * n)  # 你也可把 0.55 调大/调小
    fig, ax = plt.subplots(figsize=(fig_w, 3))

    x = np.arange(n)
    ax.bar(x, values, width=0.85)

    # 只显示部分标签（如 show_every=2 表示隔一个显示一个）
    ax.set_xticks(x[::show_every])
    ax.set_xticklabels([labels[i] for i in range(0, n, show_every)],
                       rotation=rotation, ha='right')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------
# Prepare GT products
# ---------------------------
save_image(GT, OUTDIR / "GT.png", "Ground Truth")
save_fft_image(GT, OUTDIR / "GT_fft.png", "GT FFT (log magnitude)")

# GT 方向能量分布（低/中/高频三个频段）
def orientation_energy(img: np.ndarray, num_sectors: int, band: Tuple[float,float]) -> np.ndarray:
    H, W = img.shape
    _, _, R, TH = build_frequency_coords(H, W)
    r = R / (R.max() + 1e-12)
    F = np.fft.fft2(img)
    P = np.abs(F)**2
    edges = np.linspace(0.0, np.pi, num_sectors+1)
    r0, r1 = band
    band_mask = (r >= r0) & (r <= r1)
    out = np.zeros(num_sectors, dtype=np.float64)
    for k in range(num_sectors):
        th0, th1 = edges[k], edges[k+1]
        sect_mask = (TH >= th0) & (TH < th1)
        m = band_mask & sect_mask
        out[k] = float(P[m].sum())
    # 归一到 [0,1] 便于比较
    s = out.sum()
    return out / (s + 1e-12)

gt_low  = orientation_energy(GT, NUM_SECTORS, LOW_FRAC)
gt_mid  = orientation_energy(GT, NUM_SECTORS, MID_FRAC)
gt_high = orientation_energy(GT, NUM_SECTORS, HIGH_FRAC)
sector_labels = [f"{int(i*180/NUM_SECTORS)}–{int((i+1)*180/NUM_SECTORS)}°" for i in range(NUM_SECTORS)]
bar_plot(gt_low,  sector_labels, OUTDIR/"GT_orientation_low.png",  "GT Orientation Energy (Low band)",  "Normalized energy")
bar_plot(gt_mid,  sector_labels, OUTDIR/"GT_orientation_mid.png",  "GT Orientation Energy (Mid band)",  "Normalized energy")
bar_plot(gt_high, sector_labels, OUTDIR/"GT_orientation_high.png", "GT Orientation Energy (High band)", "Normalized energy")

# ---------------------------
# Runner: iterate ranges and N
# ---------------------------
def linspace_inclusive(start: float, stop: float, n: int) -> np.ndarray:
    return np.linspace(start, stop, n)  # 保持你原来的“含端点”策略

def make_angles(range_name: str, N: int) -> np.ndarray:
    a, b = RANGES[range_name]
    return linspace_inclusive(a, b, N)

rows = []
REP_N = 100   # 代表性 N，用于保存面板图

# ROI（用 GT 取一次，后续复用）
ROI_MASK = get_roi_mask_from_gt(GT)

for N in N_LIST:
    for range_name in range_names:
        angles = make_angles(range_name, N)
        rec, mask = reconstruct_via_fft_mask(GT, angles)

        # Metrics
        ssim_full = global_ssim(rec, GT)
        ssim_roi  = global_ssim_masked(rec, GT, ROI_MASK)

        # Sector retention（ALL/LOW/MID/HIGH）
        sect_all,  cov_all  = sector_retention(rec, GT, NUM_SECTORS, (0.0, 1.0))
        sect_low,  cov_low  = sector_retention(rec, GT, NUM_SECTORS, LOW_FRAC)
        sect_mid,  cov_mid  = sector_retention(rec, GT, NUM_SECTORS, MID_FRAC)
        sect_high, cov_high = sector_retention(rec, GT, NUM_SECTORS, HIGH_FRAC)

        rows.append({
            "range": range_name,
            "N_views": N,
            "SSIM_full": ssim_full,
            "SSIM_roi":  ssim_roi,
            "Coverage_all":  cov_all,
            "Coverage_low":  cov_low,
            "Coverage_mid":  cov_mid,
            "Coverage_high": cov_high,
            **{f"Sector_all_{k}":  sect_all[k]  for k in range(NUM_SECTORS)},
            **{f"Sector_high_{k}": sect_high[k] for k in range(NUM_SECTORS)},
        })

        # 保存代表性 N 的对照图（GT / Recon / Diff / FFT / FreqMask）
        if N == REP_N:
            save_dir = OUTDIR / f"repN_{REP_N}" / range_name
            # 展示面板：GT / Recon / |Diff|
            plt.figure(figsize=(8.5, 3))
            plt.subplot(1,3,1); plt.imshow(GT,  cmap='gray', vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX); plt.title("GT");      plt.axis('off')
            plt.subplot(1,3,2); plt.imshow(rec, cmap='gray', vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX); plt.title(f"Recon {range_name}"); plt.axis('off')
            plt.subplot(1,3,3); plt.imshow(np.abs(rec - GT), cmap='inferno'); plt.title("|Recon−GT|"); plt.axis('off')
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "panel_GT_Recon_Diff.png", bbox_inches='tight'); plt.close()

            # Recon 的 FFT
            save_fft_image(rec, save_dir / "recon_fft.png", f"Recon FFT {range_name} N={N}")

            # 频域掩膜（shift 后显示）
            plt.figure()
            plt.imshow(np.fft.fftshift(mask).astype(float), cmap='gray')
            plt.title(f"Frequency Mask {range_name} N={N}")
            plt.axis('off')
            plt.savefig(save_dir / "freq_mask.png", bbox_inches='tight')
            plt.close()

# 保存指标 CSV
df = pd.DataFrame(rows).sort_values(["range", "N_views"])
csv_path = OUTDIR / "metrics.csv"
df.to_csv(csv_path, index=False)

# ---------------------------
# 你原有图 1：SSIM vs N（保留）
# ---------------------------
plt.figure()
for rn in range_names:
    vals = df[df["range"] == rn].sort_values("N_views")
    plt.plot(vals["N_views"], vals["SSIM_full"], marker='o', label=rn)
plt.xlabel("Number of views (N)")
plt.ylabel("SSIM vs GT")
plt.title("SSIM vs N for angle ranges")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig(OUTDIR / "SSIM_vs_N.png", bbox_inches='tight')
plt.close()

# ---------------------------
# 你原有图 2：Coverage vs N（保留）
# ---------------------------
plt.figure()
for rn in range_names:
    vals = df[df["range"] == rn].sort_values("N_views")
    plt.plot(vals["N_views"], vals["Coverage_all"], marker='o', label=rn)
plt.xlabel("Number of views (N)")
plt.ylabel("Coverage score (all frequencies)")
plt.title("Coverage vs N by angle range")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig(OUTDIR / "Coverage_vs_N.png", bbox_inches='tight')
plt.close()

# ---------------------------
# 你原有图 3：Radar（高频扇区保留，REP_N）（保留）
# ---------------------------
vals_dict = {}
for rn in range_names:
    row = df[(df["range"] == rn) & (df["N_views"] == REP_N)].iloc[0]
    vals = np.array([row[f"Sector_high_{k}"] for k in range(NUM_SECTORS)], dtype=float)
    vals_dict[rn] = vals
radar_plot(vals_dict, OUTDIR / f"radar_highfreq_N{REP_N}.png",
           f"Sector retention (high freq) at N={REP_N}")

# ---------------------------
# 新增图 1：ROI-SSIM vs N
# ---------------------------
plt.figure()
for rn in range_names:
    vals = df[df["range"] == rn].sort_values("N_views")
    plt.plot(vals["N_views"], vals["SSIM_roi"], marker='o', label=rn)
plt.xlabel("Number of views (N)")
plt.ylabel("ROI-SSIM vs GT")
plt.title("ROI-SSIM vs N for angle ranges")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig(OUTDIR / "ROI_SSIM_vs_N.png", bbox_inches='tight')
plt.close()

# ---------------------------
# 新增图 2：GT 方向能量雷达图（高频带）
# ---------------------------
radar_plot({"GT-high": gt_high}, OUTDIR / "GT_orientation_high_radar.png",
           "GT Orientation Energy (High band)")

# ---------------------------
# 新增图 3：对比 0_90 vs 90_180（全图 SSIM & ROI-SSIM）
# ---------------------------
plt.figure(figsize=(7,3))
for metric, fname, ylabel in [
    ("SSIM_full", "SSIM_vs_N_0_90_vs_90_180.png", "SSIM vs GT"),
    ("SSIM_roi",  "ROI_SSIM_vs_N_0_90_vs_90_180.png", "ROI-SSIM vs GT")
]:
    plt.clf()
    for rn in ["0_90", "90_180"]:
        vals = df[df["range"] == rn].sort_values("N_views")
        plt.plot(vals["N_views"], vals[metric], marker='o', label=rn)
    plt.xlabel("Number of views (N)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} (0_90 vs 90_180)")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(OUTDIR / fname, bbox_inches='tight')
plt.close()

print("[OK] All figures and metrics are saved to:", OUTDIR)
print("CSV:", csv_path)
# %%
