"""tools/run_FBP_evaluate.py
仅执行重建结果的指标评估，不再进行前向投影与FBP重建。

读取 `data/interim/recon/chest/` 中的 `.npy` 体数据，并与参考体数据对比，
计算 SSIM/PSNR 指标并按照原有逻辑保存结果。
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from src.configs.configloading import load_config
from src.data.data_load import data_load_chest
from src.evaluate.metrics_volume import evaluate_ssim_psnr

# ===== User-editable =====
CFG_FBP_PATH = "configs/FBP/chest.yaml"
CASE_IDS = ["1"]
MODALITY = "CT"
RECON_ROOT = Path("data/interim/recon/chest")
GT_STOP_DEG = 360.0    # 参考体（ground truth）对应的 stop_deg
GT_STEP_DEG = 0.25     # 参考体（ground truth）对应的 step_deg
# =========================

STOP_LIST: Iterable[float] = [60, 90, 120, 180.0]
# STOP_LIST: Iterable[float] = [60, 180.0]
STEP_LIST: Iterable[float] = [0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
# STEP_LIST: Iterable[float] = [0.25, 3.0]
CENTER_CIRCLE_RATIO = 1.0
# CENTER_CIRCLE_RATIO = None
USE_RECON_GT = True    # False 时使用原始 HU 体（通过 data_load_chest 加载）
# USE_RECON_GT = False    # False 时使用原始 HU 体（通过 data_load_chest 加载）

# =========================

def _format_number(value: float) -> str:
    """格式化角度参数，保证文件名与配置一致。"""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    text = f"{val}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _make_recon_path(case_id: str, stop_deg: float, step_deg: float) -> Path:
    stop_str = _format_number(stop_deg)
    step_str = _format_number(step_deg)
    return RECON_ROOT / f"recon_{case_id}_{stop_str}@{step_str}.npy"


def _set_angles(cfg: dict, *, stop_deg: float, step_deg: float) -> None:
    proj = cfg.get("projection", {})
    angs = proj.get("angles", {})
    angs["stop_deg"] = float(stop_deg)
    angs["step_deg"] = float(step_deg)
    proj["angles"] = angs
    cfg["projection"] = proj


def _get_current_angles(cfg: dict) -> Tuple[float, float]:
    proj = cfg.get("projection", {})
    angs = proj.get("angles", {})
    return angs.get("stop_deg"), angs.get("step_deg")


def _load_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Volume file not found: {path}")
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def main() -> None:
    cfg_base_fbp = load_config(CFG_FBP_PATH, default_path=None)

    for case_id in CASE_IDS:
        if USE_RECON_GT:
            gt_path = _make_recon_path(case_id, GT_STOP_DEG, GT_STEP_DEG)
            gt_volume = _load_volume(gt_path)
        else:
            gt_volume, _, _ = data_load_chest.load_data_chest(case_id, MODALITY)
            gt_volume = gt_volume.astype(np.float32, copy=False)

        for stop_deg, step_deg in itertools.product(STOP_LIST, STEP_LIST):
            recon_path = _make_recon_path(case_id, stop_deg, step_deg)
            try:
                recon_volume = _load_volume(recon_path)
            except FileNotFoundError:
                print(f"[Skip] Missing recon file: {recon_path}")
                continue

            if gt_volume.shape != recon_volume.shape:
                raise ValueError(
                    f"Shape mismatch for case {case_id}, stop={stop_deg}, step={step_deg}: "
                    f"gt{gt_volume.shape} vs rec{recon_volume.shape}"
                )

            cfg_eval = copy.deepcopy(cfg_base_fbp)
            _set_angles(cfg_eval, stop_deg=stop_deg, step_deg=step_deg)

            dataset_name = cfg_eval.get("data", {}).get("name", "unknown")
            out_root_tpl = cfg_eval["fbp"]["io"].get("out_root", "outputs/FBP/{dataset_name}")
            out_dir = Path(out_root_tpl.format(dataset_name=dataset_name))
            out_dir.mkdir(parents=True, exist_ok=True)

            stop_cfg, step_cfg = _get_current_angles(cfg_eval)
            print("=" * 80)
            print(
                f"[EVAL] case_id={case_id} stop_deg={stop_cfg} step_deg={step_cfg} "
                f"-> {recon_path.name}"
            )

            res = evaluate_ssim_psnr(
                gt=gt_volume,
                rec=recon_volume,
                cfg=cfg_eval,
                case_id=case_id,
                save_dir=str(out_dir),
                center_circle_ratio=CENTER_CIRCLE_RATIO,
            )

            print(f"SSIM(mean): {res['ssim']['mean']:.6f}")
            print(f"PSNR(mean): {res['psnr']['mean']:.6f}")
            print(f"[Saved to] {out_dir}")

    print("=" * 80)
    print("All evaluations completed.")


if __name__ == "__main__":
    main()
