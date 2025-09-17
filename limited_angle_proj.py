# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict
import os
import pydicom
from skimage.metrics import structural_similarity as ssim 
from skimage.filters import threshold_otsu

# ===============================
# Config
# ===============================
BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "output" / "limited_angle_proj"
os.makedirs(OUTDIR, exist_ok=True)

IMG_SIZE = (256, 256)         # simulate first on 256x256
PIXEL_MM = 0.8                # pixel size (mm)
MU_WATER = 0.02               # mm^-1, for HU->mu conversion  mu = mu_water * (1 + HU/1000)

USE_DICOM = True  # whether to load a real DICOM slice from LIDC-IDRI for demo

# Geometry (defaults you approved)
DSO = 600.0   # mm, source to isocenter
DSD = 1000.0  # mm, source to detector
# derived detector-to-isocenter distance
DID = DSD - DSO

# detector
N_DET = 800             # number of detector channels (moderate for runtime)
DET_PITCH = 1.0         # mm
U0 = 0.0                # detector center offset (channels)

# angles
RANGES = {
    "0_90":  (0.0, 90.0),
    "90_180": (90.0, 180.0),
    "0_180": (0.0, 180.0),
    "0_360": (0.0, 360.0),
}
N_VIEWS_DEFAULT = 200
N_LIST = [50, 100, 150, 200, 250, 300]

# filters
def ram_lak_filter(n: int, du: float, hann_cutoff: float = None) -> np.ndarray:
    """
    Build 1D Ram-Lak ramp filter (frequency domain) with optional Hann window (cutoff in [0,1], 1.0 = Nyquist).
    du: detector spacing (parallel domain), here we'll pass effective t spacing.
    """
    # frequency samples
    freqs = np.fft.fftfreq(n, d=du)  # cycles/mm
    filt = np.abs(freqs) * 2.0  # ramp magnitude (scale factor not critical for qualitative)
    if hann_cutoff is not None:
        f_nyq = np.max(np.abs(freqs))
        cutoff = hann_cutoff * f_nyq
        w = 0.5 * (1 + np.cos(np.pi * np.clip(np.abs(freqs)/cutoff, 0, 1)))
        w[np.abs(freqs) > cutoff] = 0.0
        filt = filt * w
    return filt

# ===============================
# Phantom (HU) and HU->mu
# ===============================
def make_hu_phantom(size: Tuple[int, int]) -> np.ndarray:
    H, W = size
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0

    img = np.full((H, W), -1000.0, dtype=np.float32)  # air
    # body soft tissue ellipse
    a1, b1 = 0.40*W, 0.48*H
    mask1 = (((xx-cx)/a1)**2 + ((yy-cy)/b1)**2) <= 1.0
    img[mask1] = 40.0

    # bone
    a2, b2 = 0.13*W, 0.18*H
    mask2 = (((xx-(cx-0.12*W))/a2)**2 + ((yy-(cy-0.12*H))/b2)**2) <= 1.0
    img[mask2] = 900.0

    # second bone-ish
    a3, b3 = 0.1*W, 0.08*H
    angle = np.deg2rad(30.0)
    xr = (xx-(cx+0.15*W))*np.cos(angle) + (yy-(cy+0.10*H))*np.sin(angle)
    yr = -(xx-(cx+0.15*W))*np.sin(angle) + (yy-(cy+0.10*H))*np.cos(angle)
    mask3 = (xr/a3)**2 + (yr/b3)**2 <= 1.0
    img[mask3] = 500.0

    # fat-like low density
    a4, b4 = 0.22*W, 0.12*H
    mask4 = (((xx-(cx*0.9))/a4)**2 + ((yy-(cy*1.05))/b4)**2) <= 1.0
    img[mask4] = -80.0
    return img

def hu_to_mu(hu: np.ndarray, mu_water: float = MU_WATER) -> np.ndarray:
    return mu_water * (1.0 + hu / 1000.0)

HU = make_hu_phantom(IMG_SIZE)
MU = hu_to_mu(HU)

dcm_path = BASE_DIR / "data/raw/chest/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-090.dcm"
ds = pydicom.dcmread(dcm_path)
img_raw = ds.pixel_array.astype(np.float32)
slope = float(getattr(ds, "RescaleSlope", 1))
intercept = float(getattr(ds, "RescaleIntercept", 0))
img_hu = img_raw * slope + intercept
img_MU = hu_to_mu(img_hu)

# save phantom images
plt.figure(); plt.imshow(HU, cmap='gray'); plt.title("HU phantom (256x256)"); plt.axis('off')
plt.savefig(OUTDIR / "phantom_HU.png", bbox_inches='tight'); plt.close()
plt.figure(); plt.imshow(MU, cmap='gray'); plt.title("mu (mm^-1) from HU"); plt.axis('off')
plt.savefig(OUTDIR / "phantom_mu.png", bbox_inches='tight'); plt.close()

# save chest slice images
plt.figure(); plt.imshow(img_hu, cmap='gray'); plt.title("HU chest (512x512)"); plt.axis('off')
plt.savefig(OUTDIR / "chest_HU.png", bbox_inches='tight'); plt.close()
plt.figure(); plt.imshow(img_MU, cmap='gray'); plt.title("mu chest (mm^-1) from HU"); plt.axis('off')
plt.savefig(OUTDIR / "chest_mu.png", bbox_inches='tight'); plt.close()

# ===============================
# Bilinear sampler
# ===============================
def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sample img at continuous (x,y) in image index coordinates (x: col, y: row).
    """
    H, W = img.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W-1); x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1); y1 = np.clip(y1, 0, H-1)

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    return Ia*wa + Ib*wb + Ic*wc + Id*wd

# ===============================
# Fan-beam forward projector (ray-driven)
# ===============================
def fanbeam_forward(mu_img: np.ndarray,
                    beta_deg: np.ndarray,
                    n_det: int = N_DET,
                    det_pitch_mm: float = DET_PITCH,
                    dso_mm: float = DSO,
                    dsd_mm: float = DSD,
                    pixel_mm: float = PIXEL_MM,
                    u0_offset: float = U0,
                    n_samples: int = 400) -> Tuple[np.ndarray, Dict]:
    H, W = mu_img.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    u = ((np.arange(n_det) - (n_det-1)/2.0 + u0_offset) * det_pitch_mm).astype(np.float32)
    P = np.zeros((len(beta_deg), n_det), dtype=np.float32)

    DID = dsd_mm - dso_mm
    t_vals = np.linspace(0.0, 1.0, n_samples).astype(np.float32)

    for ib, beta in enumerate(np.deg2rad(beta_deg)):
        xs = -dso_mm * np.cos(beta)
        ys = -dso_mm * np.sin(beta)

        xc = DID * np.cos(beta)
        yc = DID * np.sin(beta)
        tx = -np.sin(beta)
        ty =  np.cos(beta)

        xd = xc + u * tx
        yd = yc + u * ty

        dx = (xd - xs)[None, :]
        dy = (yd - ys)[None, :]

        X = xs + t_vals[:, None] * dx
        Y = ys + t_vals[:, None] * dy

        x_img = X / pixel_mm + cx
        y_img = Y / pixel_mm + cy

        mu_samp = bilinear_sample(mu_img, x_img, y_img)

        L = np.sqrt(dx**2 + dy**2)  # (1, n_det)
        ds = (L / (n_samples-1)).astype(np.float32)

        line_int = (mu_samp.sum(axis=0) * ds).ravel()
        P[ib, :] = line_int.astype(np.float32)

    info = {"u_mm": u, "det_pitch_mm": det_pitch_mm, "beta_deg": beta_deg,
            "dso_mm": dso_mm, "dsd_mm": dsd_mm}
    return P, info

# ===============================
# Fan->Parallel rebin and Parallel FBP
# ===============================
def fan2par_rebin(P: np.ndarray, info: Dict, n_t: int = None, theta_deg_uniform: np.ndarray = None):
    beta_deg = info["beta_deg"]
    u = info["u_mm"]
    dso = info["dso_mm"]
    dsd = info["dsd_mm"]

    beta = np.deg2rad(beta_deg)[:, None]
    gamma = np.arctan2(u, dsd)[None, :]

    theta = (beta + gamma)
    t = dso * np.sin(gamma)

    if theta_deg_uniform is None:
        th_min = np.rad2deg(theta.min())
        th_max = np.rad2deg(theta.max())
        ntheta = P.shape[0]
        theta_deg_uniform = np.linspace(th_min, th_max, ntheta)
    else:
        ntheta = len(theta_deg_uniform)

    if n_t is None:
        n_t = P.shape[1]

    TH_target = np.deg2rad(theta_deg_uniform)[:, None]
    t_min, t_max = float(t.min()), float(t.max())
    t_target = np.linspace(t_min, t_max, n_t)[None, :]

    P_par = np.zeros((ntheta, n_t), dtype=np.float32)

    theta_flat = theta
    # interpolate along theta for each u
    for j in range(len(u)):
        th_src = theta_flat[:, j]
        vals = P[:, j]
        # ensure sorted (beta is sorted; gamma is constant per j; so th_src is sorted)
        idx = np.searchsorted(th_src, TH_target.ravel(), side='left')
        idx = np.clip(idx, 1, len(th_src)-1)
        th0 = th_src[idx-1]; th1 = th_src[idx]
        v0 = vals[idx-1];    v1 = vals[idx]
        w = (TH_target.ravel() - th0) / (th1 - th0 + 1e-12)
        vtheta = (1-w)*v0 + w*v1
        if j == 0:
            Vtheta_stack = np.zeros((len(u), len(vtheta)), dtype=np.float32)
        Vtheta_stack[j, :] = vtheta.astype(np.float32)

    t_src = t.ravel()
    order = np.argsort(t_src)
    t_src_sorted = t_src[order]
    Vtheta_sorted = Vtheta_stack[order, :]

    idx_t = np.searchsorted(t_src_sorted, t_target.ravel(), side='left')
    idx_t = np.clip(idx_t, 1, len(t_src_sorted)-1)
    t0 = t_src_sorted[idx_t-1]; t1 = t_src_sorted[idx_t]
    w_t = (t_target.ravel() - t0) / (t1 - t0 + 1e-12)
    for it in range(len(t_target.ravel())):
        v0 = Vtheta_sorted[idx_t[it]-1, :]
        v1 = Vtheta_sorted[idx_t[it], :]
        P_par[:, it] = ((1-w_t[it])*v0 + w_t[it]*v1)

    return P_par, {"theta_deg": theta_deg_uniform, "t_mm": t_target.ravel()}

def parallel_fbp(P_par: np.ndarray, theta_deg: np.ndarray, t_mm: np.ndarray,
                 out_size: Tuple[int, int], pixel_mm: float,
                 filter_name: str = "hann", hann_cutoff: float = 0.8) -> np.ndarray:
    ntheta, nt = P_par.shape
    du = float(t_mm[1] - t_mm[0])

    if filter_name.lower() == "ram-lak":
        filt = ram_lak_filter(nt, du, hann_cutoff=None)
    elif filter_name.lower() == "hann":
        filt = ram_lak_filter(nt, du, hann_cutoff=hann_cutoff)
    elif filter_name.lower() == "shepp-logan":
        freqs = np.fft.fftfreq(nt, d=du)
        ram = np.abs(freqs) * 2.0
        sinc = np.sinc(freqs / (np.max(np.abs(freqs))+1e-12))
        filt = ram * np.abs(sinc)
    else:
        filt = ram_lak_filter(nt, du, hann_cutoff=0.8)

    P_f = np.zeros_like(P_par, dtype=np.float32)
    for i in range(ntheta):
        F = np.fft.fft(P_par[i, :])
        P_f[i, :] = np.fft.ifft(F * filt).real.astype(np.float32)

    H, W = out_size
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    xs = (np.arange(W) - cx) * pixel_mm
    ys = (np.arange(H) - cy) * pixel_mm
    X, Y = np.meshgrid(xs, ys)

    recon = np.zeros((H, W), dtype=np.float32)
    for i, th in enumerate(np.deg2rad(theta_deg)):
        t_xy = X * np.cos(th) + Y * np.sin(th)
        idx = (t_xy - t_mm[0]) / (t_mm[1] - t_mm[0])
        idx0 = np.floor(idx).astype(int)
        idx1 = idx0 + 1
        w = (idx - idx0).astype(np.float32)

        idx0 = np.clip(idx0, 0, len(t_mm)-1)
        idx1 = np.clip(idx1, 0, len(t_mm)-1)

        v0 = P_f[i, idx0]
        v1 = P_f[i, idx1]
        val = (1.0 - w) * v0 + w * v1
        recon += val.astype(np.float32)

    if len(theta_deg) > 1:
        delta_theta = np.deg2rad(np.mean(np.diff(theta_deg)))
    else:
        delta_theta = 0.0
    recon *= delta_theta

    return recon

def resize_to(img: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = out_shape
    H, W = img.shape
    yy, xx = np.meshgrid(
        np.linspace(0, H - 1, Ht), 
        np.linspace(0, W - 1, Wt), 
        indexing="ij"
    )
    return bilinear_sample(img, xx, yy).astype(np.float32)


# ===============================
# Run one range
# ===============================

if USE_DICOM:
    # 用 chest 作为底图
    BASE_MU = img_MU.astype(np.float32)
    # 覆盖像素间距与图像尺寸（优先读 DICOM 的 PixelSpacing）
    dicom_spacing = getattr(ds, "PixelSpacing", [PIXEL_MM, PIXEL_MM])
    PIXEL_MM = float(dicom_spacing[0])  # 假设方形像素
    IMG_SIZE = BASE_MU.shape  # e.g., (512, 512)
else:
    # 用模拟幻影
    BASE_MU = MU  # 已经是 mu (mm^-1)


range_names = list(RANGES.keys())

# —— 固定每个角度范围的参考角集（长度 6000，保证被 N_LIST 整除）——
THETA_REF_LEN = 6000
THETA_REF = {rn: np.linspace(*RANGES[rn], THETA_REF_LEN, endpoint=False) for rn in range_names}

def theta_subset(rn: str, N: int) -> np.ndarray:
    """从参考角集等步长取 N 个角度，保证不同 N 是嵌套子集。"""
    step = THETA_REF_LEN // N
    idx = np.arange(0, step * N, step, dtype=int)
    return THETA_REF[rn][idx]

def rot90_cw(a: np.ndarray) -> np.ndarray:
    return np.rot90(a, k=-1)  # clockwise 90°


def run_one_range(range_name: str, N_views: int, mode: str = "rebin", filt: str = "hann",
                  beta_override_deg: np.ndarray | None = None):

    beta = theta_subset(range_name, N_views) if beta_override_deg is None else beta_override_deg

    # 用选择后的底图 + 当前像素间距
    sino_fan, info = fanbeam_forward(BASE_MU, beta, pixel_mm=PIXEL_MM, dso_mm=DSO, dsd_mm=DSD)
    # 下面保持不变
    if mode in ("rebin", "fanbeam_fbp"):
        P_par, par_info = fan2par_rebin(sino_fan, info, n_t=512)
        recon = parallel_fbp(P_par, par_info["theta_deg"], par_info["t_mm"],
                             out_size=IMG_SIZE, pixel_mm=PIXEL_MM, filter_name=filt)
    else:
        P_par, par_info = fan2par_rebin(sino_fan, info, n_t=512)
        recon = parallel_fbp(P_par, par_info["theta_deg"], par_info["t_mm"],
                             out_size=IMG_SIZE, pixel_mm=PIXEL_MM, filter_name=filt)
    return recon, sino_fan, {"beta_deg": beta, "par": par_info}


MODE = "rebin"
FILT = "ram-lak"

results = {}
for rn in range_names:
    rec, sino, info = run_one_range(rn, N_VIEWS_DEFAULT, mode=MODE, filt=FILT,
                                beta_override_deg=theta_subset(rn, N_VIEWS_DEFAULT))

    results[rn] = {"recon": rec, "sino": sino, "info": info}

    # --- Sinogram：带坐标轴（x=detector, y=angle）---
    beta_deg = results[rn]["info"]["beta_deg"]  # 角度数组
    plt.figure()
    plt.imshow(
        sino, cmap='gray', aspect='auto', origin='lower',
        extent=[0, sino.shape[1]-1, float(beta_deg.min()), float(beta_deg.max())]
    )
    plt.xlabel("detector")
    plt.ylabel("angle (deg)")
    plt.title(f"Fan-beam sinogram {rn} (N={N_VIEWS_DEFAULT})")
    plt.savefig(OUTDIR / f"sino_{rn}.png", bbox_inches='tight')
    plt.close()

    # 2) 重建图：同样旋转90°顺时针
    plt.figure()
    plt.imshow(rot90_cw(rec), cmap='gray')
    plt.title(f"Recon {rn} (N={N_VIEWS_DEFAULT}, {MODE}/{FILT})")
    plt.axis('off')
    plt.savefig(OUTDIR / f"recon_{rn}.png", bbox_inches='tight')
    plt.close()


fig, axes = plt.subplots(1, len(range_names), figsize=(3*len(range_names), 3), constrained_layout=True)
for i, rn in enumerate(range_names):
    axes[i].imshow(rot90_cw(results[rn]["recon"]), cmap='gray')
    axes[i].set_title(rn)
    axes[i].axis('off')
fig.suptitle(f"Fan-beam -> FBP (mode={MODE}, filter={FILT}), N={N_VIEWS_DEFAULT}")
fig.savefig(OUTDIR / f"recon_panel_N{N_VIEWS_DEFAULT}.png", bbox_inches='tight')
plt.close(fig)

# --- SSIM vs Ground Truth ---
def to_float01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    amin, amax = float(a.min()), float(a.max())
    if amax > amin:
        return (a - amin) / (amax - amin)
    return np.zeros_like(a, dtype=np.float32)

GT = BASE_MU  # 作为 ground truth
gt_n = to_float01(GT)

# 依据 GT 取前景 ROI（示例：Otsu）
thr = threshold_otsu(gt_n)
roi = gt_n > thr
ys, xs = np.where(roi)
y0, y1 = ys.min(), ys.max()+1
x0, x1 = xs.min(), xs.max()+1

# 计算 ROI 内的 SSIM（裁剪到 ROI 的包围盒）
def ssim_roi(gt_img, rec_img):
    return ssim(gt_img[y0:y1, x0:x1], rec_img[y0:y1, x0:x1], data_range=1.0)

ssim_scores = {}
for rn in range_names:
    rec = results[rn]["recon"]
    rec_aligned = rot90_cw(rec)        # 只把重建旋转到正确取向
    rec_n = to_float01(rec_aligned)

    # 如有尺寸不一致（例如你把 DICOM 重采样到 256x256），需要在这一步对齐尺寸
    assert gt_n.shape == rec_n.shape, f"SSIM shape mismatch: GT{gt_n.shape} vs REC{rec_n.shape}"

    ssim_scores[rn] = ssim_roi(gt_n, rec_n)

# 画柱状图
plt.figure(figsize=(6, 3))
plt.bar(range_names, [ssim_scores[rn] for rn in range_names])
plt.ylabel("SSIM vs Ground Truth")
plt.xlabel("Angle Range")
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.title(f"SSIM vs GT (GT={'DICOM' if USE_DICOM else 'Phantom'})")
plt.savefig(OUTDIR / "ssim_vs_gt.png", bbox_inches='tight')
plt.close()


rows = []
plt.figure(figsize=(6, 3.5))
for rn in range_names:
    ssim_vals = []
    for N in N_LIST:
        rec, _, _ = run_one_range(rn, N, mode=MODE, filt=FILT,
                          beta_override_deg=theta_subset(rn, N))

        rec_aligned = rot90_cw(rec)  # 只旋转重建，GT 不旋转

        # 尺寸对齐
        if rec_aligned.shape != gt_n.shape:
            rec_aligned = resize_to(rec_aligned, gt_n.shape)

        rec_n = to_float01(rec_aligned)
        score = ssim_roi(gt_n, rec_n)

        RECONS_ROOT = OUTDIR / "ssim_vs_N_recons"
        os.makedirs(RECONS_ROOT, exist_ok=True)

        subdir = RECONS_ROOT / rn 
        os.makedirs(subdir, exist_ok=True)
        save_path = subdir / f"recon_{rn}_N{N:03d}_{MODE}_{FILT}.png"

        plt.figure()
        plt.imshow(rec_aligned, cmap="gray")
        plt.axis("off")
        plt.title(f"{rn} | N={N} | {MODE}/{FILT}\nSSIM={score:.4f}")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()

        ssim_vals.append(score)
        rows.append({"range": rn, "N": N, "SSIM": float(score)})
        print(f"[Info] SSIM for range {rn}, N={N}: {score:.4f}")
    plt.plot(N_LIST, ssim_vals, marker="o", label=rn)

plt.xlabel("Number of views (N)")
plt.ylabel("SSIM vs GT")
plt.ylim(0.4, 0.75)  # 视数据可调
plt.grid(alpha=0.3)
plt.legend()
plt.title("SSIM vs N for angle ranges")
plt.tight_layout()
plt.savefig(OUTDIR / "ssim_vs_N.png", bbox_inches="tight")
plt.close()

# 保存 CSV
df_ssim_n = pd.DataFrame(rows).sort_values(["range", "N"])
df_ssim_n.to_csv(OUTDIR / "ssim_vs_N.csv", index=False)
print(f"[Info] Saved SSIM vs N figure and CSV to: {OUTDIR}")

OUTDIR, [str((OUTDIR / f"recon_{rn}.png")) for rn in ["0_90","90_180","0_180","0_360"]], str(OUTDIR / f"recon_panel_N{N_VIEWS_DEFAULT}.png")

