from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

# 读图（保持原始范围）
gt = imread("outputs/FBP/chest/groundtruth_1/0000.png").astype(np.float32)
gt = imread("outputs/FBP/chest/recon_1/recon_1_360@0.25/0000.png").astype(np.float32)
rec = imread("outputs/FBP/chest/recon_1/recon_1_180@0.25/0000.png").astype(np.float32)

print("Ground truth stats:")
print(f"  shape: {gt.shape}, dtype: {gt.dtype}")
print(f"  min={gt.min()}, max={gt.max()}, mean={gt.mean()}, std={gt.std()}")

print("Reconstruction stats:")
print(f"  shape: {rec.shape}, dtype: {rec.dtype}")
print(f"  min={rec.min()}, max={rec.max()}, mean={rec.mean()}, std={rec.std()}")

# 差异分析
diff = rec - gt
print("Difference stats (rec - gt):")
print(f"  min={diff.min()}, max={diff.max()}, mean={diff.mean()}, std={diff.std()}")
print(f"  MSE={np.mean(diff**2)}, RMSE={np.sqrt(np.mean(diff**2))}")

# ---- 不归一化 ----
ssim_raw = ssim(gt, rec, data_range=255)
psnr_raw = psnr(gt, rec, data_range=255)
print(f"No Normalize SSIM = {ssim_raw:.6f}")
print(f"No Normalize PSNR = {psnr_raw:.6f} dB")

# ---- 归一化到 [0,1] ----
gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-6)
rec_n = (rec - rec.min()) / (rec.max() - rec.min() + 1e-6)

ssim_norm = ssim(gt_n, rec_n, data_range=1.0)
psnr_norm = psnr(gt_n, rec_n, data_range=1.0)
print(f"Normalize SSIM = {ssim_norm:.6f}")
print(f"Normalize PSNR = {psnr_norm:.6f} dB")

# ---- 圆形 FOV 统计 ----
height, width = gt.shape
radius = min(height, width) / 2.0
yy, xx = np.ogrid[:height, :width]
cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

ssim_full = ssim(gt_n, rec_n, data_range=1.0, full=True)
ssim_map = ssim_full[1] if isinstance(ssim_full, tuple) else ssim_full
mask_sum = mask.sum()
ssim_circle = float((ssim_map * mask).sum() / mask_sum) if mask_sum > 0 else float("nan")
print(f"Circular SSIM (radius={radius:.1f}) = {ssim_circle:.6f}")
