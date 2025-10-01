"""tools/run_ddpm_evaluate.py
对 `outputs/ddpm/chest/restore/<recon_folder>` 中按切片顺序保存的 PNG 序列进行质量评估。

行为与 `run_FBP_evaluate.py` 类似：对每个 case_id 下的重建文件夹逐个读取按序 PNG 切片，
并与指定的 ground-truth 体进行 SSIM/PSNR 计算。结果保存到
`outputs/ddpm/chest/restore/result_<case_id>/...`（evaluate 模块生成的文件名）。

使用示例：
    python run_ddpm_evaluate.py --case_ids 1 --use_recon_gt
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np

import imageio.v2 as imageio

from src.configs.configloading import load_config
from src.data.data_load import data_load_chest
from src.evaluate.metrics_volume import evaluate_ssim_psnr


# ===== User-editable =====
CFG_FBP_PATH = "configs/FBP/chest.yaml"
CASE_IDS = ["1"]
MODALITY = "CT"
RESTORE_ROOT = Path("outputs/ddpm/chest/restore")
RECON_NPY_ROOT = Path("data/interim/recon/chest")  # used when --use_recon_gt
GT_STOP_DEG = 360.0    # 参考体（ground truth）对应的 stop_deg
GT_STEP_DEG = 0.25     # 参考体（ground truth）对应的 step_deg

# =========================
USE_RECON_GT = True
# Evaluation sweep parameters (kept consistent with run_FBP_evaluate.py)
STOP_LIST: Iterable[float] = [60, 90, 120, 180.0]
STEP_LIST: Iterable[float] = [0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
CENTER_CIRCLE_RATIO = 1.0
# =========================


def _load_png_sequence(folder: Path) -> np.ndarray:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files in folder: {folder}")

    # keep only png/jpg etc and sort by numeric stem if possible
    img_files = [p for p in files if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
    if not img_files:
        raise FileNotFoundError(f"No image files found in: {folder}")

    def key_fn(p: Path):
        try:
            return int(p.stem)
        except Exception:
            return p.name

    img_files = sorted(img_files, key=key_fn)

    imgs = []
    for p in img_files:
        arr = imageio.imread(p)
        # If RGB, convert to gray by taking first channel
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.asarray(arr, dtype=np.float32)
        # if uint8-like, scale to [0,1]
        if arr.max() > 1.0:
            arr = arr / 255.0
        imgs.append(arr)

    vol = np.stack(imgs, axis=0)  # [S,H,W]
    return vol.astype(np.float32, copy=False)


def _parse_stop_step_from_name(name: str) -> Tuple[float, float]:
    # expected like: recon_{case}_{stop}@{step}
    # we simply split on '_' and '@'
    try:
        parts = name.split("_")
        last = parts[-1]
        stop_str, step_str = last.split("@")
        return float(stop_str), float(step_str)
    except Exception:
        return 180.0, 1.0


def _format_number(value: float) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    text = f"{val}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def main() -> None:
    cfg_base = load_config(CFG_FBP_PATH, default_path=None)

    for case_id in CASE_IDS:
        # find all restored folders for this case
        pattern = f"recon_{case_id}_*"
        folders = sorted([p for p in RESTORE_ROOT.glob(pattern) if p.is_dir()])
        if not folders:
            print(f"[Skip] No restored folders for case {case_id} under {RESTORE_ROOT}")
            continue

        # prepare GT per user's choice: use explicit GT_STOP_DEG/GT_STEP_DEG if USE_RECON_GT
        use_recon_gt = USE_RECON_GT
        if use_recon_gt:
            gt_npy_name = f"recon_{case_id}_{_format_number(GT_STOP_DEG)}@{_format_number(GT_STEP_DEG)}.npy"
            gt_path = RECON_NPY_ROOT / gt_npy_name
            if not gt_path.exists():
                print(f"[Skip] GT recon .npy not found for case {case_id}: {gt_path}")
                continue
            gt_volume = np.load(gt_path)
            gt_volume = gt_volume.astype(np.float32, copy=False)
        else:
            gt_volume, _, _ = data_load_chest.load_data_chest(case_id, MODALITY)
            gt_volume = gt_volume.astype(np.float32, copy=False)

        # iterate over requested angle sweep and evaluate matching folders
        for stop_deg, step_deg in itertools.product(STOP_LIST, STEP_LIST):
            stem_name = f"recon_{case_id}_{_format_number(stop_deg)}@{_format_number(step_deg)}"
            folder = RESTORE_ROOT / stem_name
            if not folder.exists() or not folder.is_dir():
                # no folder for this combination -> skip
                continue

            # load restored png sequence
            try:
                rec_volume = _load_png_sequence(folder)
            except FileNotFoundError:
                print(f"[Skip] Missing images in folder: {folder}")
                continue

            # ground truth already prepared above (explicit GT_STOP_DEG/GT_STEP_DEG when use_recon_gt)

            if gt_volume.shape != rec_volume.shape:
                print(f"[Error] Shape mismatch for {folder}: gt{gt_volume.shape} vs rec{rec_volume.shape}")
                continue

            # build a minimal cfg to allow evaluate_ssim_psnr to name outputs
            cfg_eval = copy.deepcopy(cfg_base)
            proj = cfg_eval.get("projection", {})
            angs = proj.get("angles", {})
            angs["stop_deg"] = float(stop_deg)
            angs["step_deg"] = float(step_deg)
            proj["angles"] = angs
            cfg_eval["projection"] = proj

            print("=" * 80)
            print(f"[EVAL] case_id={case_id} folder={folder.name} -> slices={rec_volume.shape[0]} stop={stop_deg} step={step_deg}")

            res = evaluate_ssim_psnr(
                gt=gt_volume,
                rec=rec_volume,
                cfg=cfg_eval,
                case_id=case_id,
                save_dir=str(RESTORE_ROOT),  # will create RESTORE_ROOT/result_{case_id}
                center_circle_ratio=CENTER_CIRCLE_RATIO,
            )

            print(f"SSIM(mean): {res['ssim']['mean']:.6f}")
            print(f"PSNR(mean): {res['psnr']['mean']:.6f}")

    print("=" * 80)
    print("All DDPM evaluations completed.")


if __name__ == "__main__":
    main()
