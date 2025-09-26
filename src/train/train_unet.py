# src/train/train_unet.py
from __future__ import annotations
from pathlib import Path
import time
from typing import Iterable, List, Dict, Any
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.configs.configloading import load_config
from src.data.dataset_unet import UnetDataset, UnetSampler
from src.model.unet import UNet2D
from src.utils.logging_utils import CSVLogger, make_tb_writer, expand_var


def _extract_metas(metas: Any, batch_size: int) -> List[Dict[str, Any]]:
    """
    将各种可能的 meta 结构统一成长度为 batch_size 的 list[dict]。
    支持：
      - None -> [{}]*B
      - list[dict] （长度>=B）
      - dict[str -> (list/np.ndarray/torch.Tensor/标量)]
    """
    out: List[Dict[str, Any]] = [dict() for _ in range(batch_size)]
    if metas is None:
        return out

    # 已经是 list[dict]
    if isinstance(metas, list):
        for i in range(min(len(metas), batch_size)):
            m = metas[i]
            if isinstance(m, dict):
                out[i].update(m)
        return out

    # 是 dict[str -> values]
    if isinstance(metas, dict):
        for k, v in metas.items():
            # 统一转可索引序列
            if isinstance(v, (list, tuple)):
                seq = v
            elif hasattr(v, "detach"):  # torch.Tensor
                v = v.detach().cpu().numpy()
                seq = v
            elif isinstance(v, np.ndarray):
                seq = v
            else:
                # 标量，给整个 batch 都填同一个
                seq = [v] * batch_size

            # 填充每个样本
            for i in range(batch_size):
                try:
                    vi = seq[i]
                except Exception:
                    vi = seq[-1] if len(seq) else None
                # 转为 python 标量
                if isinstance(vi, np.ndarray):
                    vi = vi.item() if vi.shape == () else vi.tolist()
                out[i][k] = vi
        return out

    # 其他无法识别的类型：返回空 meta
    return out


def _to_u8(img: np.ndarray) -> np.ndarray:
    """逐张归一化到[0,1]再转uint8，避免 μ 与 HU 亮度不一致。"""
    img = img.astype(np.float32, copy=False)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return (img * 255.0).round().astype(np.uint8)


# --- 轻量指标（基于张量，避免额外依赖） ---
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

def ssim(pred: torch.Tensor, target: torch.Tensor,
         C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    # 简化版 SSIM（窗口=11，padding=5，stride=1）
    mu_x = torch.nn.functional.avg_pool2d(pred, 11, 1, 5)
    mu_y = torch.nn.functional.avg_pool2d(target, 11, 1, 5)
    sigma_x = torch.nn.functional.avg_pool2d(pred * pred, 11, 1, 5) - mu_x ** 2
    sigma_y = torch.nn.functional.avg_pool2d(target * target, 11, 1, 5) - mu_y ** 2
    sigma_xy = torch.nn.functional.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean()

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def make_loader(cfg: Dict[str, Any], split: str):
    ds = UnetDataset(cfg, split_role=split)
    bs = UnetSampler(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=(split == "train"),
        drop_last=False,
        group_by="patient",
    )
    num_workers = int(cfg["data"].get("num_workers", 8))
    pin_mem = bool(cfg["data"].get("pin_memory", True)) and torch.cuda.is_available()
    dl = DataLoader(
        ds,
        batch_sampler=bs,                # 注意：使用 batch_sampler，不要再传 batch_size/shuffle
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
    )
    return ds, dl

# 替换你的 save_grid（签名增加 metas 参数）
def save_grid(x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor,
              out_path: Path, metas: Optional[Any] = None, max_n: int = 6) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    # ---- detach & cpu -> numpy ----
    x_np = x.detach().float().cpu().numpy()   # [B,1,H,W]
    y_np = y.detach().float().cpu().numpy()
    p_np = pred.detach().float().cpu().numpy()

    B = x_np.shape[0]
    b = min(B, max_n)

    # 元信息统一为 list[dict]
    meta_list = _extract_metas(metas, B)

    # 构造三行
    tiles_x, tiles_p, tiles_y, labels = [], [], [], []
    for i in range(b):
        xi = _to_u8(x_np[i, 0])
        pi = _to_u8(p_np[i, 0])
        yi = _to_u8(y_np[i, 0])
        tiles_x.append(xi); tiles_p.append(pi); tiles_y.append(yi)

        m = meta_list[i] if i < len(meta_list) else {}
        pid = m.get("patient", m.get("patient_id", m.get("pid", "?")))
        sid = m.get("slice", m.get("slice_idx", m.get("index", "?")))
        labels.append(f"p{pid} s{sid}")

    row_x = np.concatenate(tiles_x, axis=1)
    row_p = np.concatenate(tiles_p, axis=1)
    row_y = np.concatenate(tiles_y, axis=1)
    grid  = np.concatenate([row_x, row_p, row_y], axis=0)     # H*3 x (W*b)

    # 画标签（每列左上角）+ 行标题
    img  = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)
    H, W = tiles_x[0].shape  # 单图高宽
    # 行标题
    draw.text((5, 5),        "Input (μ, norm)", fill=255)
    draw.text((5, H + 5),    "Pred  (μ, norm)", fill=255)
    draw.text((5, 2*H + 5),  "GT    (HU, norm)", fill=255)
    # 每列样本标签
    for i, txt in enumerate(labels):
        x0 = i * W + 5
        draw.text((x0, 5), txt, fill=255)              # 第一行角落
        draw.text((x0, H + 5), txt, fill=255)          # 第二行
        draw.text((x0, 2*H + 5), txt, fill=255)        # 第三行

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    # ====== 配置与设备 ======
    cfg = load_config("configs/unet/chest.yaml", default_path=None)
    device = torch.device("cuda" if torch.cuda.is_available()
                          and cfg["project"]["device"] == "cuda" else "cpu")
    dname = cfg["data"]["dataset_name"]

    # 允许更快的 matmul（Ampere+）
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ====== 路径展开 ======
    ckpt_dir   = Path(expand_var(cfg["train"]["ckpt_dir"],   dname))
    sample_dir = Path(expand_var(cfg["train"]["sample_dir"], dname))
    tb_dir     = Path(expand_var(cfg["train"]["logging"]["tensorboard_dir"], dname))
    csv_path   = Path(expand_var(cfg["train"]["logging"]["csv_path"], dname))
    ensure_dir(ckpt_dir); ensure_dir(sample_dir); ensure_dir(tb_dir); ensure_dir(csv_path.parent)

    # ====== 日志器 ======
    lg_cfg = cfg["train"]["logging"]
    tb = make_tb_writer(bool(lg_cfg.get("use_tensorboard", True)), tb_dir)
    csv_logger = CSVLogger(csv_path) if bool(lg_cfg.get("save_csv", True)) else None

    # ====== 数据 ======
    train_ds, train_dl = make_loader(cfg, "train")
    val_ds,   val_dl   = make_loader(cfg, "val")

    # ====== 模型 ======
    mcfg = cfg["model"]
    net = UNet2D(
        in_ch=mcfg["in_ch"],
        out_ch=mcfg["out_ch"],
        base_ch=mcfg["base_ch"],
        depth=mcfg["depth"],
        bilinear=mcfg["bilinear_upsample"],
        norm=mcfg["norm"],
    ).to(device)

    # ====== 优化器 & 调度器 ======
    ocfg = cfg["optim"]
    if ocfg["name"].lower() == "adamw":
        opt = optim.AdamW(net.parameters(), lr=ocfg["lr"], weight_decay=ocfg["weight_decay"])
    else:
        opt = optim.Adam(net.parameters(), lr=ocfg["lr"])
    scfg = ocfg.get("scheduler", {})
    if scfg.get("name", "").lower() == "cosineannealinglr":
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(scfg.get("T_max", cfg["train"]["max_epochs"])))
    else:
        sch = None

    # ====== AMP / 其他 ======
    criterion = nn.L1Loss()
    best_score = -1.0
    gstep = 0
    amp_enabled = bool(cfg["train"].get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    log_interval = int(cfg["train"].get("log_interval", 50))
    save_best_by = str(cfg["train"].get("save_best_by", "SSIM")).upper()

    # ====== 训练循环 ======
    for epoch in range(1, int(cfg["train"]["max_epochs"]) + 1):
        # ---- Train ----
        net.train()
        t0 = time.time()
        running = 0.0

        train_iter = tqdm(train_dl, desc=f"Epoch {epoch:03d} [train]", ncols=100)
        for it, batch in enumerate(train_iter, 1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                pred = net(x)
                loss = criterion(pred, y)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            running += loss.item()
            gstep += 1

            # 进度条上显示即时指标
            train_iter.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.3e}")

            # TensorBoard（按 iter）
            if tb:
                tb.add_scalar("train/loss_iter", loss.item(), gstep)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], gstep)

            # 也可保留原有的定期 print（可选）
            if (it % log_interval) == 0:
                print(f"[E{epoch:03d} I{it:04d}] loss={loss.item():.4f}")

        if sch:
            sch.step()
        t1 = time.time()
        epoch_train_loss = running / max(1, it)

        # ---- Val ----
        net.eval()
        loss_v, ssim_v, psnr_v, n = 0.0, 0.0, 0.0, 0
        val_iter = tqdm(val_dl, desc=f"Epoch {epoch:03d} [val]  ", ncols=100, leave=False)
        with torch.no_grad():
            for batch in val_iter:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                with autocast(enabled=amp_enabled):
                    pred = net(x)
                    loss_step = criterion(pred, y)
                pred = pred.clamp(0, 1)

                loss_v += loss_step.item()
                ssim_v += ssim(pred, y).item()
                psnr_v += psnr(pred, y).item()
                n += 1

                # 进度条尾部显示当前均值
                val_iter.set_postfix(
                    loss=f"{(loss_v / n):.4f}",
                    ssim=f"{(ssim_v / n):.4f}",
                    psnr=f"{(psnr_v / n):.2f}dB",
                )

                # 只在第一个 batch 保存一张拼图
                if n == 1:
                    save_grid(x, y, pred, sample_dir / f"epoch_{epoch:03d}.png", metas=batch)


        loss_v /= max(n, 1); ssim_v /= max(n, 1); psnr_v /= max(n, 1)

        # ---- Logging (epoch-level) ----
        print(f"[VAL E{epoch:03d}] loss={loss_v:.4f} SSIM={ssim_v:.4f} PSNR={psnr_v:.2f}dB ({t1 - t0:.1f}s)")
        if tb:
            tb.add_scalar("train/loss_epoch", epoch_train_loss, epoch)
            tb.add_scalar("val/loss",  loss_v, epoch)
            tb.add_scalar("val/ssim",  ssim_v, epoch)
            tb.add_scalar("val/psnr",  psnr_v, epoch)

        if csv_logger:
            csv_logger.log({
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "val_loss": loss_v,
                "val_ssim": ssim_v,
                "val_psnr": psnr_v,
                "lr": opt.param_groups[0]["lr"],
                "time_sec": round(t1 - t0, 3),
            })

        # ---- Save best / last ----
        score = {"SSIM": ssim_v, "PSNR": psnr_v, "LOSS": -loss_v}.get(save_best_by, ssim_v)
        if score > best_score:
            best_score = score
            best_path = ckpt_dir / "best.pth"
            ckpt = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "best_score": best_score,
                "cfg": cfg,
            }
            ensure_dir(best_path.parent)
            torch.save(ckpt, best_path)
            print(f"[SAVE] best({save_best_by}={score:.4f}) -> {best_path}")

        last_path = ckpt_dir / "last.pth"
        torch.save({
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "best_score": best_score,
            "cfg": cfg,
        }, last_path)

    if tb: tb.close()
    if csv_logger: csv_logger.close()
    print("Done.")


if __name__ == "__main__":
    main()
