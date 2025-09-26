from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

# 读图（保持原始范围）
gt = imread("outputs/FBP/chest/groundtruth_1/0000.png").astype(np.float32)
rec = imread("outputs/FBP/chest/recon_1/recon_1_60@3/0000.png").astype(np.float32)

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
ssim_raw = ssim(gt, rec, data_range=gt.max() - gt.min())
psnr_raw = psnr(gt, rec, data_range=gt.max() - gt.min())
print(f"No Normalize SSIM = {ssim_raw:.6f}")
print(f"No Normalize PSNR = {psnr_raw:.6f} dB")

# ---- 归一化到 [0,1] ----
gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-6)
rec_n = (rec - rec.min()) / (rec.max() - rec.min() + 1e-6)

ssim_norm = ssim(gt_n, rec_n, data_range=1.0)
psnr_norm = psnr(gt_n, rec_n, data_range=1.0)
print(f"Normalize SSIM = {ssim_norm:.6f}")
print(f"Normalize PSNR = {psnr_norm:.6f} dB")
