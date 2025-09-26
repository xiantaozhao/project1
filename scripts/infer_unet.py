#!/usr/bin/env python3
# scripts/infer_unet.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys, time, inspect, csv, argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader

# --------- 将仓库根目录加入 sys.path ----------
def _add_repo_root_to_syspath():
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src").is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return
_add_repo_root_to_syspath()

# --------- 项目内模块 ----------
from src.configs.configloading import load_config
from src.model.unet import UNet2D
from src.data.dataset_unet import UnetDataset  # 我们会做自适应构造

# 写 PNG（可选）
try:
    import imageio.v3 as iio
except Exception:
    iio = None

# 进度条（可选）
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

# ----------------- 工具 -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_image_png(path: Path, arr01: np.ndarray):
    if iio is None:
        return
    arr = np.clip(arr01, 0.0, 1.0)
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    iio.imwrite(str(path), arr8)

def to_device(x, device):
    return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

# ----------------- 轻量指标（与训练一致） -----------------
def torch_psnr(pred: torch.Tensor, target: torch.Tensor, eps=1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

def torch_ssim(pred: torch.Tensor, target: torch.Tensor,
               C1=0.01**2, C2=0.03**2) -> torch.Tensor:
    import torch.nn.functional as F
    k = 11
    mu_x = F.avg_pool2d(pred, k, 1, k//2)
    mu_y = F.avg_pool2d(target, k, 1, k//2)
    sigma_x = F.avg_pool2d(pred*pred, k, 1, k//2) - mu_x*mu_x
    sigma_y = F.avg_pool2d(target*target, k, 1, k//2) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(pred*target, k, 1, k//2) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / (
        (mu_x*mu_x + mu_y*mu_y + C1)*(sigma_x + sigma_y + C2) + 1e-12
    )
    return ssim_map.mean()

# ----------------- 模型构建 -----------------
def build_model_from_cfg(cfg: Dict[str, Any]) -> torch.nn.Module:
    mcfg = (cfg.get("model") or {})
    in_ch    = int(mcfg.get("in_ch",     mcfg.get("in_channels", 1)))
    out_ch   = int(mcfg.get("out_ch",    mcfg.get("out_channels", 1)))
    base_ch  = int(mcfg.get("base_ch",   mcfg.get("base_channels", 48)))  # 你的默认是 48
    depth    = int(mcfg.get("depth",     mcfg.get("num_down", 4)))
    bilinear = bool(mcfg.get("bilinear", True))
    norm     = str(mcfg.get("norm", "none"))

    model = UNet2D(
        in_ch=in_ch,
        out_ch=out_ch,
        base_ch=base_ch,
        depth=depth,
        bilinear=bilinear,
        norm=norm,
    )
    print(f"[INFO] Build UNet2D(in_ch={in_ch}, out_ch={out_ch}, base_ch={base_ch}, "
          f"depth={depth}, bilinear={bilinear}, norm='{norm}')")
    return model

def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: Path, strict: bool = True):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict):
        state = None
        for k in ["model", "state_dict", "model_state", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]; break
        if state is None:
            maybe = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
            state = maybe if maybe else ckpt
    else:
        state = ckpt
    new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing:    print(f"[load_checkpoint] missing keys: {missing}")
    if unexpected: print(f"[load_checkpoint] unexpected keys: {unexpected}")

# ----------------- batch 解包 & 命名 -----------------
def unpack_batch(batch):
    """
    安全解包，避免对 Tensor 做布尔判断。
    支持：
      - (x, y, meta)
      - (x, meta)
      - (x,)
      - dict 各种常见键名
    """
    x = y = meta = None

    # 情况 1：tuple/list
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            x, y, meta = batch
        elif len(batch) == 2:
            x, meta = batch
        elif len(batch) == 1:
            x = batch[0]
        else:
            x = batch
        return x, y, meta

    # 情况 2：dict
    if isinstance(batch, dict):
        # 明确按键名顺序查找，避免使用 `or`
        for k in ("x", "input", "image"):
            if k in batch:
                x = batch[k]; break

        for k in ("y", "target", "label"):
            if k in batch:
                y = batch[k]; break

        for k in ("meta", "info", "meta_info"):
            if k in batch:
                meta = batch[k]; break

        # 兜底：如果还没找到 x，则取第一个 tensor-like
        if x is None:
            for v in batch.values():
                if torch.is_tensor(v):
                    x = v; break

        # 防御：如果 meta 被写成 tensor/ndarray，就当作没有 meta
        if torch.is_tensor(meta):
            meta = None

        return x, y, meta

    # 情况 3：其它，直接当作 x
    return batch, None, None


def meta_to_name(meta: Optional[dict], fallback_idx: int) -> Tuple[str, int]:
    patient = "case"; sl = fallback_idx
    if isinstance(meta, dict):
        if "patient" in meta: patient = str(meta["patient"])
        for alt in ["pid", "case_id", "case", "patient_id"]:
            if alt in meta: patient = str(meta[alt])
        if "slice" in meta: sl = int(meta["slice"])
    return patient, sl

# ----------------- 数据集自适应构造 -----------------
def build_dataset_from_cfg(cfg: Dict[str, Any], split: str):
    """
    尝试多种常见 __init__ 方式：
      UnetDataset(cfg)
      UnetDataset(cfg, split)
      UnetDataset(cfg, mode=split)
      UnetDataset(cfg, phase=split)
      UnetDataset(split=split)
      UnetDataset(mode=split)
      UnetDataset(phase=split)
      UnetDataset(config=cfg, split=split)
    如果仍失败，则抛出更清晰的错误并提示 __init__ 签名。
    """
    tried: List[str] = []
    def _try(msg, fn):
        tried.append(msg)
        try:
            return fn()
        except TypeError:
            return None

    # 1) (cfg)
    ds = _try("UnetDataset(cfg)", lambda: UnetDataset(cfg))
    if ds is not None: return ds

    # 2) (cfg, split)
    ds = _try("UnetDataset(cfg, split)", lambda: UnetDataset(cfg, split))
    if ds is not None: return ds

    # 3) (cfg, mode=split)
    ds = _try("UnetDataset(cfg, mode=split)", lambda: UnetDataset(cfg, mode=split))
    if ds is not None: return ds

    # 4) (cfg, phase=split)
    ds = _try("UnetDataset(cfg, phase=split)", lambda: UnetDataset(cfg, phase=split))
    if ds is not None: return ds

    # 5) (split=split)
    ds = _try("UnetDataset(split=split)", lambda: UnetDataset(split=split))
    if ds is not None: return ds

    # 6) (mode=split)
    ds = _try("UnetDataset(mode=split)", lambda: UnetDataset(mode=split))
    if ds is not None: return ds

    # 7) (phase=split)
    ds = _try("UnetDataset(phase=split)", lambda: UnetDataset(phase=split))
    if ds is not None: return ds

    # 8) (config=cfg, split=split)
    ds = _try("UnetDataset(config=cfg, split=split)", lambda: UnetDataset(config=cfg, split=split))
    if ds is not None: return ds

    # 如果都不行，打印签名帮助
    sig = None
    try:
        sig = str(inspect.signature(UnetDataset.__init__))
    except Exception:
        sig = "<unknown>"
    tried_msg = "; ".join(tried)
    raise TypeError(
        f"无法自适应构造 UnetDataset。已尝试：{tried_msg}\n"
        f"请告知 UnetDataset.__init__ 签名（当前检测到：{sig}，"
        f"或直接贴出 src/data/dataset_unet.py 中该类的 __init__ 头部。"
    )

# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser(description="UNet 推理脚本")
    parser.add_argument("--config", type=str, default="configs/unet/chest.yaml",
                        help="配置 [YAML] 路径")
    parser.add_argument("--ckpt", type=str, default="outputs/unet/chest/checkpoints/best.pth",
                        help="权重 [.pth] 路径")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--split", type=str, default="test", help="数据划分：train/val/test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_png", action="store_true", help="不保存 [PNG]")
    parser.add_argument("--half", action="store_true", help="启用 [AMP] 半精度推理")
    parser.add_argument("--device", type=str, default=None, help="覆盖 cfg.project.device")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"配置不存在: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"权重不存在: {ckpt_path}")

    # 读取配置（按你要求）
    cfg: Dict[str, Any] = load_config(str(cfg_path))

    device_str = args.device or cfg.get("project", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model = build_model_from_cfg(cfg)
    load_checkpoint_weights(model, ckpt_path, strict=False)
    model.to(device).eval()

    # 输出目录
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = ckpt_path.parent.parent / "infer" / args.split / ts
    ensure_dir(out_dir)
    print(f"[INFO] 输出目录: {out_dir}")

    # —— 自适应构造数据集 ——
    dataset = build_dataset_from_cfg(cfg, split=args.split)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=False,
                        pin_memory=True)

    # 推理
    save_png = (not args.no_png) and (iio is not None)
    have_target = False
    psnr_list: List[float] = []
    ssim_list: List[float] = []
    name_list: List[str] = []

    use_amp = args.half and (device.type == "cuda")
    with torch.no_grad():
        it = tqdm(loader, desc=f"[infer] split={args.split}", ncols=100)
        global_idx = 0
        for batch in it:
            x, y, meta = unpack_batch(batch)
            if x is None:
                raise RuntimeError("DataLoader 返回的 batch 无法解包到输入张量 x。请检查 UnetDataset 的 __getitem__。")

            x = to_device(x, device).float()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)  # 期望 [B,1,H,W]

            pred_np = pred.detach().float().cpu().numpy()
            if pred_np.ndim == 4 and pred_np.shape[1] == 1:
                pred_np = pred_np[:, 0]  # [B,H,W]

            # 指标（若有目标）
            if y is not None:
                have_target = True
                y = to_device(y, device).float()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    p = torch.clamp(pred, 0.0, 1.0)
                    t = torch.clamp(y,    0.0, 1.0)
                    psnr_list.append(float(torch_psnr(p, t).detach().cpu().item()))
                    ssim_list.append(float(torch_ssim(p, t).detach().cpu().item()))

            # 保存
            B = pred_np.shape[0]
            metas = None
            if isinstance(meta, (list, tuple)) and len(meta) == B:
                metas = meta
            elif isinstance(meta, dict):
                keys = list(meta.keys())
                if keys and isinstance(meta[keys[0]], (list, tuple)):
                    metas = [{k: meta[k][i] for k in keys} for i in range(B)]
                else:
                    metas = [meta] * B

            for b in range(B):
                this_meta = metas[b] if metas is not None else None
                patient, sl = meta_to_name(this_meta, global_idx)
                name = f"{patient}_{sl:04d}"
                name_list.append(name)

                np.save(str(out_dir / f"{name}.npy"), pred_np[b].astype(np.float32))
                if save_png:
                    save_image_png(out_dir / f"{name}.png", pred_np[b])
                global_idx += 1

    if have_target:
        with open(out_dir / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["name", "psnr", "ssim"])
            for n, p, s in zip(name_list, psnr_list, ssim_list):
                w.writerow([n, f"{p:.6f}", f"{s:.6f}"])
        print(f"[INFO] 平均 PSNR={np.mean(psnr_list):.4f}, 平均 SSIM={np.mean(ssim_list):.4f}")
        print(f"[INFO] 指标写入: {out_dir/'metrics.csv'}")

    print("[DONE] 推理完成.")

if __name__ == "__main__":
    main()
