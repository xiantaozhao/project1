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


IMG_SIZE = (256, 256)  # 你的真实数据是 512x512，这里用 256x256 的模拟切片
PIXEL_SIZE = 1.0       # mm (not used directly but kept for completeness)

# Angle ranges to test (inclusive endpoints), and a list of N views
RANGES = {
    "0_90": (0.0, 90.0),
    "90_180": (90.0, 180.0),
    "0_180": (0.0, 180.0),
    "0_360": (0.0, 360.0),
}
N_LIST = [50, 100, 150, 200, 250, 300]

# Frequency sector settings
NUM_SECTORS = 12  # 15° each over 180° unique orientations
LOW_FRAC = (0.0, 0.33)  # low frequency band (as fraction of Nyquist radius)
MID_FRAC = (0.33, 0.66)
HIGH_FRAC = (0.66, 1.0)

# ---------------------------
# Utility: build HU phantom (256x256)
# A few ellipses with HU values to mimic air (-1000), soft tissue (~0-60), bone (~300-1200)
# ---------------------------
def make_hu_phantom(size: Tuple[int, int]) -> np.ndarray:
    H, W = size
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0

    img = np.full((H, W), -1000.0, dtype=np.float32)  # background air
    # Big soft-tissue ellipse
    a1, b1 = 0.38*W, 0.5*H
    mask1 = (((xx-cx)/a1)**2 + ((yy-cy)/b1)**2) <= 1.0
    img[mask1] = 40.0  # soft tissue ~ 40 HU

    # Bone-like ellipse
    a2, b2 = 0.12*W, 0.18*H
    mask2 = (((xx-(cx-0.12*W))/a2)**2 + ((yy-(cy-0.15*H))/b2)**2) <= 1.0
    img[mask2] = 800.0  # cortical bone-like

    # Another bone-ish ellipse
    a3, b3 = 0.1*W, 0.08*H
    angle = np.deg2rad(25.0)
    xr = (xx-(cx+0.15*W))*np.cos(angle) + (yy-(cy+0.1*H))*np.sin(angle)
    yr = -(xx-(cx+0.15*W))*np.sin(angle) + (yy-(cy+0.1*H))*np.cos(angle)
    mask3 = (xr/a3)**2 + (yr/b3)**2 <= 1.0
    img[mask3] = 450.0

    # Low-density ellipse (fat-like, negative but not air)
    a4, b4 = 0.2*W, 0.12*H
    mask4 = (((xx-(cx*0.9))/a4)**2 + ((yy-(cy*1.05))/b4)**2) <= 1.0
    img[mask4] = -80.0

    return img

phantom_hu = make_hu_phantom(IMG_SIZE)

dcm_path = BASE_DIR / "data/raw/chest/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-090.dcm"
ds = pydicom.dcmread(dcm_path)
img_raw = ds.pixel_array.astype(np.float32)
slope = float(getattr(ds, "RescaleSlope", 1))
intercept = float(getattr(ds, "RescaleIntercept", 0))
img_hu = img_raw * slope + intercept



# Save the phantom
plt.figure()
plt.imshow(phantom_hu, cmap='gray')
plt.title("HU Phantom (256x256)")
plt.axis('off')
plt.savefig(OUTDIR / "phantom_hu.png", bbox_inches='tight')
plt.close()


# Save the chest slice
plt.figure()
plt.imshow(img_hu, cmap='gray')
plt.title("HU Chest (512x512)")
plt.axis('off')
plt.savefig(OUTDIR / "img_hu.png", bbox_inches='tight')
plt.close()

# ---------------------------
# Frequency mask to emulate limited-angle coverage via Fourier Slice Theorem
# We create angular "wedges" for each projection angle; the wedge half-width equals half the angular step.
# This approximates the coverage provided by N projections spanning [alpha, beta].
# ---------------------------
def build_frequency_coords(h: int, w: int):
    ky = np.fft.fftfreq(h) * h  # cycles per image (integer grid)
    kx = np.fft.fftfreq(w) * w
    KX, KY = np.meshgrid(kx, ky)
    R = np.sqrt(KX**2 + KY**2)
    THETA = np.arctan2(KY, KX)  # [-pi, pi]
    # Map to [0, pi): orientation equivalence of theta and theta+pi
    THETA_MOD = np.mod(THETA, np.pi)
    return KX, KY, R, THETA_MOD

def angular_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest absolute distance between angles in [0, pi) on a circle of length pi."""
    d = np.abs(a - b)
    return np.minimum(d, np.pi - d)

def angular_mask(h: int, w: int, angles_deg: np.ndarray) -> np.ndarray:
    """
    Construct a binary mask in frequency domain: pass regions whose angle matches
    any projection angle within half the angular step. Always keep DC.
    """
    _, _, R, TH = build_frequency_coords(h, w)
    if len(angles_deg) < 2:
        half_width = np.deg2rad(0.5)  # minimal wedge
    else:
        # effective bin half-width equals half the median angular spacing (in radians)
        spacings = np.diff(np.sort(angles_deg))
        # handle wrap for ranges that include 0 and 180/360
        # but since we map to [0, 180), use modulo 180
        spacings = np.append(spacings, (180 - (angles_deg.max() - angles_deg.min())))
        half_width = np.deg2rad(np.median(spacings) / 2.0)
        half_width = max(half_width, np.deg2rad(0.5))

    mask = np.zeros((h, w), dtype=bool)
    for ang in angles_deg:
        a = np.deg2rad(ang) % np.pi  # map to [0, pi)
        mask |= (angular_distance(TH, a) <= half_width)
    # Always keep very low frequencies (a small disk) to avoid ringing in DC neighborhood
    r_keep = 2.0
    mask |= (R <= r_keep)
    return mask

# ---------------------------
# Reconstruction via masked inverse FFT (coverage proxy)
# ---------------------------
def reconstruct_via_fft_mask(gt_img: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    H, W = gt_img.shape
    F = np.fft.fft2(gt_img)
    mask = angular_mask(H, W, angles_deg)
    F_masked = F * mask
    rec = np.fft.ifft2(F_masked).real
    return rec, mask

# ---------------------------
# Global SSIM (no window) for simplicity and robustness without extra deps
# ---------------------------
def global_ssim(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = x.mean()
    mu_y = y.mean()
    vx = x.var()
    vy = y.var()
    cxy = ((x - mu_x) * (y - mu_y)).mean()
    # Constants (scaled for HU dynamic range ~ [-1000, 1200])
    L = 2200.0
    k1, k2 = 0.01, 0.03
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    num = (2*mu_x*mu_y + C1) * (2*cxy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (vx + vy + C2)
    return float(num / den)

# ---------------------------
# Sector energy retention
# ---------------------------
def sector_retention(rec_img: np.ndarray, gt_img: np.ndarray, num_sectors: int = 12,
                     r_frac_band: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, float]:
    """
    Compute energy retention per angular sector in frequency domain: |F_rec|^2 / |F_gt|^2.
    Returns (retention_per_sector, coverage_score = mean over sectors).
    """
    H, W = rec_img.shape
    _, _, R, TH = build_frequency_coords(H, W)
    r = R / R.max()  # normalize radius to [0,1]

    F_rec = np.fft.fft2(rec_img)
    F_gt = np.fft.fft2(gt_img)
    P_rec = np.abs(F_rec)**2
    P_gt  = np.abs(F_gt)**2 + 1e-12  # avoid zero

    # radial band mask
    r0, r1 = r_frac_band
    band_mask = (r >= r0) & (r <= r1)

    # sector bins in [0, pi)
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
# Runner: iterate ranges and N, produce recon, metrics, and plots
# ---------------------------
def linspace_inclusive(start: float, stop: float, n: int) -> np.ndarray:
    return np.linspace(start, stop, n)

def make_angles(range_name: str, N: int) -> np.ndarray:
    a, b = RANGES[range_name]
    return linspace_inclusive(a, b, N)

def save_image(img: np.ndarray, path: Path, title: str = ""):
    plt.figure()
    plt.imshow(img, cmap='gray')
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
    # Simple radar (polar) with num_sectors dimensions
    labels = [f"S{k}" for k in range(NUM_SECTORS)]
    angles = np.linspace(0, 2*np.pi, NUM_SECTORS, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure()
    ax = plt.subplot(111, polar=True)
    for name, vals in values_dict.items():
        data = vals.tolist()
        data += data[:1]
        ax.plot(angles, data, linewidth=1.5, label=name)
    ax.set_xticks(np.linspace(0, 2*np.pi, NUM_SECTORS, endpoint=False))
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# Prepare GT: we treat the phantom itself as GT here (you can switch to 0-360 dense recon later)
if USE_DICOM:
    GT = img_hu.copy()
else:
    GT = phantom_hu.copy()

# Save GT images
save_image(GT, OUTDIR / "GT.png", "Ground Truth (HU Phantom)")
save_fft_image(GT, OUTDIR / "GT_fft.png", "GT FFT (log magnitude)")

rows = []
# For one representative N to draw qualitative panels
REP_N = 100

for N in N_LIST:
    for range_name in RANGES.keys():
        angles = make_angles(range_name, N)
        rec, mask = reconstruct_via_fft_mask(GT, angles)
        ssim_val = global_ssim(rec, GT)

        # Sector retention (all / low / mid / high)
        sect_all, cov_all = sector_retention(rec, GT, NUM_SECTORS, (0.0, 1.0))
        sect_low, cov_low = sector_retention(rec, GT, NUM_SECTORS, LOW_FRAC)
        sect_mid, cov_mid = sector_retention(rec, GT, NUM_SECTORS, MID_FRAC)
        sect_high, cov_high = sector_retention(rec, GT, NUM_SECTORS, HIGH_FRAC)

        rows.append({
            "range": range_name,
            "N_views": N,
            "SSIM": ssim_val,
            "Coverage_all": cov_all,
            "Coverage_low": cov_low,
            "Coverage_mid": cov_mid,
            "Coverage_high": cov_high,
            **{f"Sector_all_{k}": sect_all[k] for k in range(NUM_SECTORS)},
            **{f"Sector_high_{k}": sect_high[k] for k in range(NUM_SECTORS)},
        })

        # Save representative images for REP_N
        if N == REP_N:
            save_dir = OUTDIR / f"repN_{REP_N}" / range_name
            save_image(rec, save_dir / "recon.png", f"Recon {range_name} N={N}")
            save_fft_image(rec, save_dir / "fft.png", f"FFT {range_name} N={N}")

            # Also save the binary mask used in frequency domain for visualization
            plt.figure()
            plt.imshow(np.fft.fftshift(mask).astype(float), cmap='gray')
            plt.title(f"Frequency Mask {range_name} N={N}")
            plt.axis('off')
            plt.savefig(save_dir / "freq_mask.png", bbox_inches='tight')
            plt.close()

# Metrics dataframe
df = pd.DataFrame(rows)
csv_path = OUTDIR / "metrics.csv"
df.to_csv(csv_path, index=False)



# ---------------------------
# Plot: SSIM degradation curves vs N
# ---------------------------
plt.figure()
for range_name in RANGES.keys():
    vals = df[df["range"] == range_name].sort_values("N_views")
    plt.plot(vals["N_views"], vals["SSIM"], marker='o', label=range_name)
plt.xlabel("Number of views (N)")
plt.ylabel("SSIM vs GT")
plt.title("SSIM vs N for angle ranges")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig(OUTDIR / "SSIM_vs_N.png", bbox_inches='tight')
plt.close()

# ---------------------------
# Plot: Coverage vs N (all-band)
# ---------------------------
plt.figure()
for range_name in RANGES.keys():
    vals = df[df["range"] == range_name].sort_values("N_views")
    plt.plot(vals["N_views"], vals["Coverage_all"], marker='o', label=range_name)
plt.xlabel("Number of views (N)")
plt.ylabel("Coverage score (all frequencies)")
plt.title("Coverage vs N by angle range")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.savefig(OUTDIR / "Coverage_vs_N.png", bbox_inches='tight')
plt.close()

# ---------------------------
# Radar plot for sector retention at REP_N (high-frequency band, most diagnostic)
# ---------------------------
vals_dict = {}
for range_name in RANGES.keys():
    row = df[(df["range"] == range_name) & (df["N_views"] == REP_N)].iloc[0]
    vals = np.array([row[f"Sector_high_{k}"] for k in range(NUM_SECTORS)], dtype=float)
    vals_dict[range_name] = vals
radar_plot(vals_dict, OUTDIR / f"radar_highfreq_N{REP_N}.png",
           f"Sector retention (high freq) at N={REP_N}")

# ---------------------------
# Save a small README for you
# ---------------------------
with open(OUTDIR / "README.txt", "w") as f:
    f.write(
        "This demo emulates limited-angle CT information loss using Fourier-domain coverage masks.\n"
        "GT: HU-like phantom (256x256). Four angle ranges: 0-90, 90-180, 0-180, 0-360.\n"
        "For each N in {16,32,64,100,180}, we build an angular frequency mask based on sampled views\n"
        "and reconstruct via masked inverse FFT (a proxy via the Fourier Slice Theorem). We compute global SSIM\n"
        "vs GT, frequency-sector energy retention (all/low/mid/high bands), and we plot:\n"
        "- SSIM_vs_N.png: degradation curves with N\n"
        "- Coverage_vs_N.png: coverage score curves with N\n"
        "- radar_highfreq_N100.png: sector retention radar at N=100 (high-frequency band)\n"
        "Representative images for N=100 are under repN_100/<range>/ (recon.png, fft.png, freq_mask.png)\n"
    )

OUTDIR, csv_path, OUTDIR / "SSIM_vs_N.png", OUTDIR / "Coverage_vs_N.png", OUTDIR / f"radar_highfreq_N{REP_N}.png"

# %%
