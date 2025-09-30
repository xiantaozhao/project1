#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# --- make repo importable ---
def _add_repo_root_to_syspath():
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

_add_repo_root_to_syspath()

from src.simple_ddpm.model import SimpleUNet
from src.simple_ddpm.diffusion import Diffusion, DDIM
from src.data.data_load import data_load_chest


# =====================
# Hard-coded parameters
# =====================
CASE_ID = 1                 # 注意：整数 ID
MODALITY = 'CT'             # 模态
IMAGE_SIZE = 512            # 训练图像尺寸（H=W=512）
CHANNELS = 1                # 单通道 CT slice
EPOCHS = 1000                 # 训练轮数
BATCH_SIZE = 8              # 批大小（512x512 建议 4~8，视显存调节）
LR = 2e-4                   # 学习率
STEPS = 1000                # 扩散步数 T（DDPM）
DDIM_step = 100              # DDIM 采样步数
DEVICE = 'cuda'             # 'cuda' 或 'cpu'
MU_WATER = 0.02            # 水的线衰减系数 (mm^-1)，用于 HU->mu 转换
OUT_DIR = 'outputs/simple_ddpm_chest'
SEED = 42
SAVE_IMAGE_EVERY = 10       # 可视化图片保存间隔（单位：epoch）


class SliceDataset(Dataset):
    """把 HU 体数据按 slice 切成单张图，归一化到 [0,1]，并调整到 IMAGE_SIZE。"""
    def __init__(self, vol_hu_zyx: np.ndarray, image_size: int = 512):
        assert vol_hu_zyx.ndim == 3, 'Expected volume shape [Z,H,W]'
        self.vol = vol_hu_zyx.astype(np.float32)
        self.Z, self.H, self.W = self.vol.shape
        self.image_size = image_size

    def __len__(self):
        return self.Z

    def __getitem__(self, idx: int):
        sl_hu = self.vol[idx]  # [H, W] in HU
        # 1) HU -> mu（不裁剪 HU）
        sl_mu = MU_WATER * (1.0 + sl_hu.astype(np.float32) / 1000.0)
        # 2) 每张切片独立归一化到 [0,1]
        mu_min = float(sl_mu.min())
        mu_max = float(sl_mu.max())
        if mu_max > mu_min:
            sl_norm = (sl_mu - mu_min) / (mu_max - mu_min)
        else:
            sl_norm = np.zeros_like(sl_mu, dtype=np.float32)
        # 3) 转为张量并 resize 到 [1, IMAGE_SIZE, IMAGE_SIZE]
        x = torch.from_numpy(sl_norm).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        if x.shape[-1] != IMAGE_SIZE or x.shape[-2] != IMAGE_SIZE:
            x = torch.nn.functional.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        x = x.squeeze(0)  # [1,H,W]
        return { 'image': x, 'slice_idx': idx }


def save_png_grid(x: torch.Tensor, path: Path, nrow: int = 4):
    """保存网格图，若缺少 matplotlib 则保存为 .npy。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        np.save(path.with_suffix('.npy'), x.cpu().numpy())
        return
    x = x.clamp(0, 1)
    x = x.cpu().numpy()
    if x.shape[1] == 1:
        x = x[:, 0]
    N = x.shape[0]
    ncol = (N + nrow - 1) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*3))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)
    for i in range(nrow):
        for j in range(ncol):
            idx = i*ncol + j
            axes[i, j].axis('off')
            if idx < N:
                img = x[idx]
                if img.ndim == 2:
                    axes[i, j].imshow(img, cmap='gray')
                else:
                    import numpy as _np
                    axes[i, j].imshow(_np.transpose(img, (1,2,0)))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_triplet_grid(gt: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor, path: Path,
                      col_titles: tuple[str, str, str] = ("GT", "Noisy", "Denoised")):
    """将 N 张样本的 GT/Noisy/Denoised 拼成一个 N×3 的网格，并在每列顶部加标题。"""
    try:
        import matplotlib.pyplot as plt
        import numpy as _np
    except Exception:
        # 回退为分别保存 npy
        np.save(path.with_name(path.stem + '_gt.npy'), gt.cpu().numpy())
        np.save(path.with_name(path.stem + '_noisy.npy'), noisy.cpu().numpy())
        np.save(path.with_name(path.stem + '_denoised.npy'), denoised.cpu().numpy())
        return

    def to_hw(x: torch.Tensor):
        x = x.clamp(0, 1).detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] == 1:
            return x[:, 0]  # [N,H,W]
        elif x.ndim == 4:
            # CHW -> HWC for display
            return _np.transpose(x, (0, 2, 3, 1))
        elif x.ndim == 3:
            return x  # [N,H,W]
        else:
            raise ValueError("Unexpected tensor shape for image grid")

    g = to_hw(gt)
    n = to_hw(noisy)
    d = to_hw(denoised)
    N = g.shape[0]

    fig, axes = plt.subplots(N, 3, figsize=(3*3, N*3))  # 每图 3x3 inch
    if N == 1:
        axes = _np.expand_dims(axes, axis=0)

    # 顶部列标题
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)

    for i in range(N):
        imgs = [g[i], n[i], d[i]]
        for j in range(3):
            ax = axes[i, j]
            ax.axis('off')
            img = imgs[j]
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE if (DEVICE == 'cuda' and torch.cuda.is_available()) else 'cpu')

    # 读取 HU 体数据 [Z,H,W]
    vol_HU_zyx, spacing_dzyx, meta = data_load_chest.load_data_chest(CASE_ID, MODALITY)
    print(f"Loaded volume: shape={vol_HU_zyx.shape}, spacing(dzyx)={spacing_dzyx}")

    # 数据集/加载器
    ds = SliceDataset(vol_HU_zyx, image_size=IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 模型与扩散
    model = SimpleUNet(in_ch=CHANNELS).to(device)
    diff = Diffusion(T=STEPS).to(device)
    ddim = DDIM(T=STEPS, eta=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl, desc=f'Epoch {epoch}/{EPOCHS}')
        for batch in pbar:
            x0 = batch['image'].to(device)  # [B,1,512,512]
            B = x0.size(0)
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(x0)
            xt = diff.q_sample(x0, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 保存检查点（每 SAVE_IMAGE_EVERY 个 epoch）
        ckpt = {
            'model': model.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'config': {
                'CASE_ID': CASE_ID,
                'MODALITY': MODALITY,
                'IMAGE_SIZE': IMAGE_SIZE,
                'CHANNELS': CHANNELS,
                'EPOCHS': EPOCHS,
                'BATCH_SIZE': BATCH_SIZE,
                'LR': LR,
                'STEPS': STEPS,
                'DEVICE': DEVICE,
                'MU_WATER': MU_WATER,
            }
        }
        if epoch % SAVE_IMAGE_EVERY == 0:
            torch.save(ckpt, out_dir/f'checkpoint_epoch_{epoch}.pth')
            torch.save(ckpt, out_dir/'last.pth')

        # 每隔 SAVE_IMAGE_EVERY 个 epoch 才保存可视化样本
        if epoch % SAVE_IMAGE_EVERY == 0:
            # 采样几个样本看效果（减少步数加速 & 显示进度）
            model.eval()
            with torch.no_grad():
                    samples = ddim.sample(
                        model,
                        (4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
                        device,
                        num_steps=DDIM_step,             # DDIM 50 步通常已有结构
                        show_progress=True
                    )
            save_png_grid(samples, out_dir/f'samples_epoch_{epoch}.png', nrow=2)

        # 每隔 SAVE_IMAGE_EVERY 个 epoch 才保存 debug 三联图
        if epoch % SAVE_IMAGE_EVERY == 0:
            # 额外：从真实切片出发的去噪可视化（帮助判断是否学到东西），三图拼接到一张
            try:
                model.eval()
                with torch.no_grad():
                    # 取两张真实切片
                    x0_list = []
                    for i in range(min(2, len(ds))):
                        x0_list.append(ds[i]['image'])
                    if x0_list:
                        x0 = torch.stack(x0_list, dim=0).to(device)  # [N,1,512,512]
                        # 选一个较大的起始时间步（噪声较重）
                        t0 = min(600, STEPS-1)
                        t = torch.full((x0.size(0),), t0, device=device, dtype=torch.long)
                        noise = torch.randn_like(x0)
                        xt = diff.q_sample(x0, t, noise)

                        # 使用 DDIM 从 t0 开始反推 ~50 步，速度快且更稳定
                        num_back_steps = 50
                        # 构建一个从 t0 逐步到 0 的子时间表
                        ts = ddim.set_timesteps(num_back_steps)
                        # 找到最接近 t0 的起点索引
                        start_idx = 0
                        for i, tval in enumerate(ts):
                            if tval <= t0:
                                start_idx = i
                                break
                        x_cur = xt
                        for i in range(start_idx, len(ts)):
                            t = ts[i]
                            t_prev = ts[i+1] if i+1 < len(ts) else -1
                            tvec = torch.full((x0.size(0),), t, device=device, dtype=torch.long)
                            eps = model(x_cur, tvec)
                            x_cur, _ = ddim.step_from_to(eps, t, t_prev, x_cur, eta=0.0)
                        x_rec = x_cur.clamp(0, 1)

                        # 合成一张：GT / Noisy(t=t0) / Denoised（在同一作用域，避免未绑定变量）
                        save_triplet_grid(
                            x0, xt, x_rec,
                            out_dir/f'debug_epoch_{epoch}_triplet.png',
                            col_titles=("GT", f"Noisy t={t0}", "Denoised")
                        )
            except Exception as e:
                print(f"[Warn] debug denoise preview failed: {e}")

    # 最终权重
    torch.save(model.state_dict(), out_dir/'final_weights.pth')
    print(f"Training done. Checkpoints and samples saved under: {out_dir}")


if __name__ == '__main__':
    main()
