# tools/run_angle_grid.py
from __future__ import annotations
import itertools
import copy
from pathlib import Path

import numpy as np

from src.configs.configloading import load_config
from src.data.data_load import data_load_chest
from src.data.data_process.projection import project_volume_with_astra
from src.model.FBP import fbp_reconstruct_with_astra
from src.evaluate.metrics_volume import evaluate_ssim_psnr

# ===== User-editable =====
CFG_FORWARD_PATH = "configs/default/chest.yaml"
CFG_FBP_PATH     = "configs/FBP/chest.yaml"
CASE_ID          = "1"
MODALITY         = "CT"

STOP_LIST = [60, 90, 120, 180.0]
# STOP_LIST = [360.0]
STOP_LIST = [1, 5, 10, 30]
STEP_LIST = [0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
STEP_LIST = [1]
# STEP_LIST = [0.25]
USE_RECON_GT = False  # False 时直接使用 loader 输出的原始 HU 体作为评估基准
GT_RECON_ROOT = Path("data/interim/recon/chest")
GT_RECON_STOP_DEG = 360.0
GT_RECON_STEP_DEG = 0.25
# =========================


def _format_number(value: float) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    rounded = round(val)
    if abs(val - rounded) < 1e-6:
        return str(int(rounded))
    text = f"{val}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _make_recon_path(case_id: str, stop_deg: float, step_deg: float) -> Path:
    stop_str = _format_number(stop_deg)
    step_str = _format_number(step_deg)
    return GT_RECON_ROOT / f"recon_{case_id}_{stop_str}@{step_str}.npy"

def _set_angles(cfg: dict, *, stop_deg: float, step_deg: float):
    """
    Safely set angle fields inside cfg['projection']['angles'] if present.
    Keeps start_deg/include_endpoint/mode as-is.
    """
    # Shallow traverse with guards
    proj = cfg.get("projection", {})
    angs = proj.get("angles", {})
    angs["stop_deg"] = float(stop_deg)
    angs["step_deg"] = float(step_deg)
    proj["angles"] = angs
    cfg["projection"] = proj

def _get_current_angles(cfg: dict):
    proj = cfg.get("projection", {})
    angs = proj.get("angles", {})
    return {
        "mode": angs.get("mode"),
        "start_deg": angs.get("start_deg"),
        "stop_deg": angs.get("stop_deg"),
        "step_deg": angs.get("step_deg"),
        "include_endpoint": angs.get("include_endpoint"),
    }

def main():
    # Load data once to avoid repeated I/O
    vol_HU_zyx, spacing_dzyx, meta = data_load_chest.load_data_chest(CASE_ID, MODALITY)
    if USE_RECON_GT:
        gt_eval_path = _make_recon_path(CASE_ID, GT_RECON_STOP_DEG, GT_RECON_STEP_DEG)
        if not gt_eval_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_eval_path}")
        gt_eval = np.load(gt_eval_path).astype(np.float32)
    else:
        gt_eval = vol_HU_zyx.astype(np.float32, copy=False)
    case_id = CASE_ID

    # Preload base cfgs so we can deep-copy per run
    cfg_base_forward = load_config(CFG_FORWARD_PATH, default_path=None)
    cfg_base_fbp     = load_config(CFG_FBP_PATH, default_path=None)

    for stop_deg, step_deg in itertools.product(STOP_LIST, STEP_LIST):
        # Fresh copies for this combo
        cfg = copy.deepcopy(cfg_base_forward)
        cfg_fbp = copy.deepcopy(cfg_base_fbp)
        cfg = cfg_fbp

        # Update both forward-projection and FBP cfg angles to stay consistent
        _set_angles(cfg, stop_deg=stop_deg, step_deg=step_deg)
        _set_angles(cfg_fbp, stop_deg=stop_deg, step_deg=step_deg)

        # Echo effective params before running
        ang_forward = _get_current_angles(cfg)
        ang_fbp     = _get_current_angles(cfg_fbp)
        print("=" * 80)
        print(f"[RUN] case_id={case_id}  stop_deg={stop_deg}  step_deg={step_deg}")
        print("[Forward angles] ",
              f"mode={ang_forward['mode']}, start={ang_forward['start_deg']}, "
              f"stop={ang_forward['stop_deg']}, step={ang_forward['step_deg']}, "
              f"include_endpoint={ang_forward['include_endpoint']}")
        print("[FBP angles]     ",
              f"mode={ang_fbp['mode']}, start={ang_fbp['start_deg']}, "
              f"stop={ang_fbp['stop_deg']}, step={ang_fbp['step_deg']}, "
              f"include_endpoint={ang_fbp['include_endpoint']}")

        # ----- Forward projection -----
        sino = project_volume_with_astra(
            vol_HU_zyx, spacing_dzyx, cfg, case_id=case_id
        )

        # ----- FBP reconstruction -----
        recon = fbp_reconstruct_with_astra(
            sino_SAD=sino,
            cfg_merged=cfg_fbp,
            case_id=case_id,
            spacing_dzyx=spacing_dzyx,
            # ground_truth_zyx=vol_HU_zyx,  # for any internal checks
        )

        # # ----- Metrics & save -----
        # # dataset_name 来自 data.name
        # dataset_name = cfg_fbp.get("data", {}).get("name", "unknown")

        # # 统一输出目录（支持 {dataset_name}）
        # out_root_tpl = cfg_fbp["fbp"]["io"].get("out_root", "outputs/FBP/{dataset_name}")
        # out_dir = Path(out_root_tpl.format(dataset_name=dataset_name))
        # out_dir.mkdir(parents=True, exist_ok=True)


        # res = evaluate_ssim_psnr(
        #     gt=gt_eval,             # [S,H,W]
        #     rec=recon,              # [S,H,W]
        #     cfg=cfg_fbp,
        #     case_id=case_id,
        #     save_dir=str(out_dir),
        #     center_circle_ratio=1
        # )

        # print(f"SSIM(mean): {res['ssim']['mean']:.6f}")
        # print(f"PSNR(mean): {res['psnr']['mean']:.6f}")
        # print(f"[Saved to] {out_dir}")

    print("=" * 80)
    print("All runs completed.")

if __name__ == "__main__":
    main()
