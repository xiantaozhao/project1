from src.configs.configloading import load_config
from src.data.data_load import data_load_chest
from src.data.data_process.projection import project_volume_with_astra
from src.model.FBP import fbp_reconstruct_with_astra
from src.evaluate.metrics_volume import evaluate_ssim_psnr

cfg = load_config("configs/default/chest.yaml", default_path=None)

vol_HU_zyx, spacing_dzyx, meta = data_load_chest.load_data_chest("1", "CT")

sino = project_volume_with_astra(vol_HU_zyx, spacing_dzyx, cfg, case_id=meta.get('case_id', '1'))

cfg_FBP = load_config("configs/FBP/chest.yaml", default_path=None)

recon = fbp_reconstruct_with_astra(
    sino_SAD=sino,            # [S, A, D]
    cfg_merged=cfg_FBP,       # 合并后的配置（包含几何/角度）
    case_id=meta.get('case_id', '1'),
    spacing_dzyx=spacing_dzyx,
    ground_truth_zyx=vol_HU_zyx
)


res = evaluate_ssim_psnr(
    gt=vol_HU_zyx,                  # [S,H,W]
    rec=recon,              # [S,H,W]
    cfg=cfg_FBP,
    case_id=meta.get('case_id', '1'),
    save_dir="outputs/FBP"
)

print("SSIM(mean):", res["ssim"]["mean"])
print("PSNR(mean):", res["psnr"]["mean"])