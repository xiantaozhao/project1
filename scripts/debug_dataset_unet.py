# scripts/debug_dataset_unet.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any, Iterable
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 项目内导入
from src.configs.configloading import load_config
from src.data.dataset_unet import UnetDataset, UnetSampler

# 尝试优先用 imageio 保存 PNG；不可用则回退到 PIL
try:
    import imageio.v3 as iio
except Exception:
    iio = None
from PIL import Image


# -------------------- 工具函数 --------------------
def to_uint8(img: np.ndarray) -> np.ndarray:
    """将 [H,W] float 转为 uint8，可容错范围外数据。"""
    arr = np.asarray(img, dtype=np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 1e-6:
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        # 若未在[0,1]，做一次 min-max 拉伸后再 clip
        if not (vmin >= 0.0 and vmax <= 1.0):
            arr = (arr - vmin) / (vmax - vmin + 1e-6)
        arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = to_uint8(img) if img.dtype != np.uint8 else img
    if iio is not None:
        iio.imwrite(path, out)
    else:
        Image.fromarray(out).save(path)


def make_side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 4) -> np.ndarray:
    """左右拼接 [H,W] + [H,W] -> [H, 2W+gap]"""
    L = to_uint8(left); R = to_uint8(right)
    h, w = L.shape
    canvas = np.zeros((h, w * 2 + max(gap, 0)), dtype=np.uint8)
    canvas[:, :w] = L
    if gap > 0:
        canvas[:, w:w+gap] = 200
    canvas[:, w + max(gap, 0):] = R
    return canvas


def tensor_chw_to_hw(x: torch.Tensor) -> np.ndarray:
    """将 [1,H,W] 或 [H,W] Tensor 转 [H,W] np.float32"""
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return x.detach().cpu().numpy().astype(np.float32)


def _viz_scale(img: np.ndarray, mode: str) -> np.ndarray:
    """
    仅用于可视化的拉伸方式：
      - volume: 不额外缩放（忠实展示 dataset 输出）
      - slice: 逐图 min-max 到 [0,1]，便于观察结构
    """
    arr = np.asarray(img, dtype=np.float32)
    if mode == "slice":
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax - vmin < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - vmin) / (vmax - vmin + 1e-6)
    return arr


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将样本列表堆叠成一个 batch；meta 保留为列表。"""
    xs = torch.stack([s["x"] for s in samples], dim=0)  # [B,1,H,W]
    ys = torch.stack([s["y"] for s in samples], dim=0)
    metas = [s["meta"] for s in samples]
    return {"x": xs, "y": ys, "meta": metas}


def save_batch_grid(xs: torch.Tensor, ys: torch.Tensor, out_path: Path,
                    nrow: int | None = None, gap: int = 2, viz_mode: str = "volume"):
    """
    保存一个 batch 的二维网格图：
      - 第一块（上）：所有 x
      - 第二块（下）：所有 y
    修复了行间分隔宽度不一致导致的拼接报错。
    """
    B, _, H, W = xs.shape
    if nrow is None:
        nrow = min(B, 8)
    ncol = int(np.ceil(B / nrow))

    def grid_from_stack(stack: torch.Tensor) -> np.ndarray:
        imgs: List[np.ndarray] = []
        for i in range(stack.shape[0]):
            hw = tensor_chw_to_hw(stack[i])
            hw = _viz_scale(hw, viz_mode)  # 仅可视化缩放
            imgs.append(to_uint8(hw))
        total = nrow * ncol
        if len(imgs) < total:
            imgs.extend([np.zeros((H, W), dtype=np.uint8) for _ in range(total - len(imgs))])

        rows = []
        grid_width: int | None = None
        for r in range(ncol):
            row_imgs = imgs[r * nrow:(r + 1) * nrow]
            row_concat = np.concatenate(row_imgs + ([255 * np.ones((H, gap), np.uint8)] if gap > 0 else []), axis=1)
            row_trim = row_concat[:, :-gap] if gap > 0 else row_concat
            if grid_width is None:
                grid_width = row_trim.shape[1]
            else:
                if row_trim.shape[1] != grid_width:
                    row_trim = row_trim[:, :grid_width]
            rows.append(row_trim)

        assert grid_width is not None
        if gap > 0:
            sep_row = 255 * np.ones((gap, grid_width), dtype=np.uint8)
            # rows 与分隔行交错拼接
            interleaved: List[np.ndarray] = []
            for rr in rows:
                interleaved.append(rr)
                interleaved.append(sep_row)
            grid = np.concatenate(interleaved[:-1], axis=0)  # 去掉最后一个分隔
        else:
            grid = np.concatenate(rows, axis=0)
        return grid

    gx = grid_from_stack(xs)
    gy = grid_from_stack(ys)

    if gap > 0:
        mid = 255 * np.ones((gap, gx.shape[1]), dtype=np.uint8)
        big = np.concatenate([gx, mid, gy], axis=0)
    else:
        big = np.concatenate([gx, gy], axis=0)

    save_png(out_path, big)


# -------------------- 主程序 --------------------
def main():
    parser = argparse.ArgumentParser(description="Debug UnetDataset reading & visualization")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置路径，如 configs/unet/chest.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="使用哪个 split")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batches", type=int, default=3, help="最多保存多少个 batch 的可视化")
    parser.add_argument("--out_dir", type=str, default="outputs/unet/chest/debug")
    parser.add_argument("--group_by", type=str, default="patient", choices=["patient", "patient_angle"],
                        help="UnetSampler 的分组方式")
    parser.add_argument("--shuffle", action="store_true", help="是否打乱分组与组内顺序")
    parser.add_argument("--viz_mode", type=str, default="volume", choices=["volume", "slice"],
                        help="仅影响保存图片时的拉伸方式；不影响 dataset 输出。")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据集 & 采样器
    dataset = UnetDataset(cfg, split_role=args.split)
    print(f"[INFO] Dataset size = {len(dataset)}  files={len(dataset.files)}  norm_mode={dataset.norm_mode}")
    if len(dataset) == 0:
        print("[WARN] 数据集为空，请检查 recon_root / file_glob / filters / split 配置")
        return

    sampler = UnetSampler(dataset, batch_size=args.batch_size,
                          shuffle=args.shuffle, drop_last=False,
                          group_by=args.group_by)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_batch,
    )

    saved_batches = 0
    for bidx, batch in enumerate(tqdm(loader, desc="Iter")):
        xs: torch.Tensor = batch["x"]  # [B,1,H,W]
        ys: torch.Tensor = batch["y"]
        metas: List[Dict[str, Any]] = batch["meta"]

        # 打印若干 meta
        print("---- Batch", bidx, "----")
        for j, m in enumerate(metas[:min(len(metas), 3)]):
            print(f"  [{j}] patient={m['patient']} slice={m['slice_id']} stop={m['stop_deg']} step={m['step_deg']}")
            print(f"      recon_path={m['recon_path']}")

        # 保存单张 & 对照
        for j in range(xs.shape[0]):
            m = metas[j]
            pid = m["patient"]; sid = m["slice_id"]
            base = out_dir / f"b{bidx:03d}_i{j:02d}_p{pid}_z{sid:04d}"

            x_hw = tensor_chw_to_hw(xs[j])
            y_hw = tensor_chw_to_hw(ys[j])

            # 仅可视化拉伸
            x_v = _viz_scale(x_hw, args.viz_mode)
            y_v = _viz_scale(y_hw, args.viz_mode)

            save_png(base.with_name(base.name + "_x.png"), x_v)
            save_png(base.with_name(base.name + "_y.png"), y_v)
            save_png(base.with_name(base.name + "_xy.png"), make_side_by_side(x_v, y_v, gap=4))

        # 保存 batch 网格（上=x，下=y）
        grid_path = out_dir / f"batch_{bidx:03d}_grid.png"
        save_batch_grid(xs, ys, grid_path, nrow=min(args.batch_size, 8), viz_mode=args.viz_mode)

        saved_batches += 1
        if saved_batches >= args.batches:
            break

    print(f"[DONE] Saved {saved_batches} batch(es) to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
