#!/usr/bin/env python3
"""
Sinogram-conditioned DDPM restoration using ASTRA-based projection consistency.

This script performs conditional image restoration where each denoising step
applies sinogram consistency corrections via forward/backward projection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


# --- make repo importable ---
def _add_repo_root_to_syspath():
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root_to_syspath()

from src.model.diffusion import SimpleUNet, Diffusion, DDIM

try:
    import astra
except ImportError:
    astra = None


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_volume(volume_path: Path) -> np.ndarray:
    """Load a 3D volume from .npy file."""
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume file not found: {volume_path}")
    
    arr = np.load(volume_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (Z,H,W); got shape {arr.shape} in {volume_path}")
    return arr.astype(np.float32)


def load_sinogram(sino_path: Path) -> np.ndarray:
    """Load a 3D sinogram from .npy file."""
    if not sino_path.exists():
        raise FileNotFoundError(f"Sinogram file not found: {sino_path}")
    
    arr = np.load(sino_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (Z,A,D); got shape {arr.shape} in {sino_path}")
    return arr.astype(np.float32)


def match_files(
    patient_id: str,
    angle_range: str,
    angle_step: str,
    recon_root: Path,
    proj_root: Path,
) -> Tuple[Path | None, Path | None]:
    """Match reconstruction and sinogram files based on naming pattern.
    
    Returns:
        Tuple of (recon_path, sinogram_path) or (None, None) if not found.
    """
    # Try to find recon file - may use integer format (e.g., @1.npy) or decimal format (@1.0.npy)
    recon_candidates = [
        recon_root / f"recon_{patient_id}_{angle_range}@{angle_step}.npy",  # exact match
        recon_root / f"recon_{patient_id}_{angle_range}@{int(float(angle_step))}.npy",  # integer format
    ]
    
    recon_path = None
    for candidate in recon_candidates:
        if candidate.exists():
            recon_path = candidate
            break
    
    if recon_path is None:
        print(f"[WARN] Reconstruction file not found. Tried:")
        for candidate in recon_candidates:
            print(f"  - {candidate}")
        return None, None
    
    # Try to find sinogram file - typically uses decimal format (e.g., @1.0.npy)
    sino_candidates = [
        proj_root / f"chest_{patient_id}_{angle_range}@{angle_step}.npy",  # exact match
    ]
    # If angle_step is integer-like (e.g., "1"), also try with .0 suffix
    if '.' not in angle_step or angle_step.endswith('.0'):
        float_val = float(angle_step)
        sino_candidates.append(proj_root / f"chest_{patient_id}_{angle_range}@{float_val:.1f}.npy")
    
    sino_path = None
    for candidate in sino_candidates:
        if candidate.exists():
            sino_path = candidate
            break
    
    if sino_path is None:
        print(f"[WARN] Sinogram file not found. Tried:")
        for candidate in sino_candidates:
            print(f"  - {candidate}")
        return None, None
    
    return recon_path, sino_path


def calculate_angles(angle_range: float, angle_step: float) -> np.ndarray:
    """Calculate angle array from range and step.
    
    Args:
        angle_range: Total angle range in degrees (e.g., 60, 180)
        angle_step: Step size in degrees (e.g., 0.25, 1.0)
    
    Returns:
        Array of angles in degrees [0, step, 2*step, ..., range]
    """
    # Include endpoint: [0, step, 2*step, ..., range]
    num_angles = int(np.round(angle_range / angle_step)) + 1
    angles = np.linspace(0, angle_range, num_angles, dtype=np.float64)
    return angles


def create_astra_geometries(
    cfg: Dict[str, Any],
    image_size: int,
    angles_deg: np.ndarray,
) -> Tuple[Any, Any, int]:
    """Create ASTRA volume and projection geometries.
    
    Args:
        cfg: Configuration dictionary with geometry parameters
        image_size: Image size (H=W)
        angles_deg: Angles in degrees
    
    Returns:
        Tuple of (vol_geom, proj_geom, projector_id)
    """
    if astra is None:
        raise ImportError("ASTRA toolbox is required. Please install 'astra-toolbox'.")
    
    # Get geometry parameters from config
    geom_cfg = cfg.get('projection', {}).get('geom', {})
    det_cfg = geom_cfg.get('det', {})
    
    # Detector parameters
    det_count = int(det_cfg.get('det_count', 1000))
    det_pixel_mm = float(det_cfg.get('det_pixel_mm', 0.7))
    
    # Source-detector geometry
    source_origin_mm = float(geom_cfg.get('source_origin_mm', 1000.0))
    origin_det_mm = float(geom_cfg.get('origin_det_mm', 600.0))
    
    # Pixel spacing (assume square pixels)
    data_cfg = cfg.get('data', {})
    spacing_cfg = data_cfg.get('spacing', {})
    spacing_values = spacing_cfg.get('values', [2.5, 0.703125, 0.703125])
    pixel_mm = float(spacing_values[-1]) if spacing_values else 0.703125  # Use dx
    
    # Create volume geometry (2D)
    vol_geom = astra.create_vol_geom(
        image_size, image_size,
        -image_size * pixel_mm / 2.0, image_size * pixel_mm / 2.0,  # x_min, x_max
        -image_size * pixel_mm / 2.0, image_size * pixel_mm / 2.0   # y_min, y_max
    )
    
    # Create projection geometry (fanflat)
    angles_rad = np.deg2rad(angles_deg)
    proj_geom = astra.create_proj_geom(
        'fanflat',
        det_pixel_mm,      # detector spacing
        det_count,         # detector count
        angles_rad,        # angles in radians
        source_origin_mm,  # DSO
        origin_det_mm      # ODD
    )
    
    # Create projector
    projector_type = cfg.get('projection', {}).get('astra', {}).get('projector_2d', 'cuda')
    if projector_type not in ('cuda', 'line'):
        projector_type = 'cuda'
    
    projector_id = astra.create_projector(projector_type, proj_geom, vol_geom)
    
    return vol_geom, proj_geom, projector_id


def forward_project_astra(
    img_hw: np.ndarray,
    vol_geom: Any,
    proj_geom: Any,
    projector_id: int,
) -> np.ndarray:
    """Forward project image to sinogram using ASTRA.
    
    Args:
        img_hw: Image [H, W]
        vol_geom: ASTRA volume geometry
        proj_geom: ASTRA projection geometry
        projector_id: ASTRA projector ID
    
    Returns:
        Sinogram [A, D]
    """
    img_2d = np.ascontiguousarray(img_hw, dtype=np.float32)
    vol_id = astra.data2d.create('-vol', vol_geom, img_2d)
    sino_id = None
    try:
        sino_id, sino = astra.create_sino(vol_id, projector_id)
        result = np.asarray(sino, dtype=np.float32)
    finally:
        astra.data2d.delete(vol_id)
        if sino_id is not None:
            astra.data2d.delete(sino_id)
    return result


def back_project_astra(
    sino_ad: np.ndarray,
    vol_geom: Any,
    proj_geom: Any,
    projector_id: int,
) -> np.ndarray:
    """Backproject sinogram to image using ASTRA FBP.
    
    Args:
        sino_ad: Sinogram [A, D]
        vol_geom: ASTRA volume geometry
        proj_geom: ASTRA projection geometry
        projector_id: ASTRA projector ID
    
    Returns:
        Reconstructed image [H, W]
    """
    sino_2d = np.ascontiguousarray(sino_ad, dtype=np.float32)
    sid = astra.data2d.create('-sino', proj_geom, sino_2d)
    rid = astra.data2d.create('-vol', vol_geom)
    
    # FBP configuration
    cfg_fbp = astra.astra_dict('FBP_CUDA')
    cfg_fbp['ProjectionDataId'] = sid
    cfg_fbp['ReconstructionDataId'] = rid
    cfg_fbp['ProjectorId'] = projector_id
    cfg_fbp['option'] = {'FilterType': 'Ram-Lak'}
    
    alg_id = astra.algorithm.create(cfg_fbp)
    try:
        astra.algorithm.run(alg_id)
        result = np.asarray(astra.data2d.get(rid), dtype=np.float32)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sid)
        astra.data2d.delete(rid)
    
    return result


def normalize_sinogram_pair(pred_sino: np.ndarray, target_sino: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize predicted and target sinograms to the same scale.
    
    Strategy: MinMax normalization to [0, 1] for both sinograms.
    This ensures both sinograms are in the same range before computing error.
    
    Args:
        pred_sino: Predicted sinogram [A, D]
        target_sino: Target sinogram [A, D]
    
    Returns:
        Tuple of (normalized_pred_sino, normalized_target_sino)
    """
    # Debug print (only first call)
    if not hasattr(normalize_sinogram_pair, '_debug_printed'):
        print(f"\n[SINO NORM DEBUG - Before]")
        print(f"  Pred sinogram:   mean={pred_sino.mean():.4f}, std={pred_sino.std():.4f}, range=[{pred_sino.min():.4f}, {pred_sino.max():.4f}]")
        print(f"  Target sinogram: mean={target_sino.mean():.4f}, std={target_sino.std():.4f}, range=[{target_sino.min():.4f}, {target_sino.max():.4f}]")
        normalize_sinogram_pair._debug_printed = True
    
    # MinMax normalize both to [0, 1]
    pred_min, pred_max = pred_sino.min(), pred_sino.max()
    target_min, target_max = target_sino.min(), target_sino.max()
    
    # Normalize pred_sino to [0, 1]
    if pred_max > pred_min:
        pred_normalized = (pred_sino - pred_min) / (pred_max - pred_min)
    else:
        pred_normalized = np.zeros_like(pred_sino)
    
    # Normalize target_sino to [0, 1]
    if target_max > target_min:
        target_normalized = (target_sino - target_min) / (target_max - target_min)
    else:
        target_normalized = np.zeros_like(target_sino)
    
    # Debug print after normalization (only first call)
    if hasattr(normalize_sinogram_pair, '_debug_printed') and normalize_sinogram_pair._debug_printed:
        print(f"\n[SINO NORM DEBUG - After MinMax to [0,1]]")
        print(f"  Pred sinogram:   mean={pred_normalized.mean():.4f}, std={pred_normalized.std():.4f}, range=[{pred_normalized.min():.4f}, {pred_normalized.max():.4f}]")
        print(f"  Target sinogram: mean={target_normalized.mean():.4f}, std={target_normalized.std():.4f}, range=[{target_normalized.min():.4f}, {target_normalized.max():.4f}]")
        normalize_sinogram_pair._debug_printed = False  # Only print once
    
    return pred_normalized, target_normalized


@torch.no_grad()
def conditioned_seddit_restore_slice(
    model: torch.nn.Module,
    diff: Diffusion,
    ddim: DDIM,
    img_hw: np.ndarray,
    target_sinogram: np.ndarray,
    vol_geom: Any,
    proj_geom: Any,
    projector_id: int,
    *,
    device: torch.device,
    image_size: int = 512,
    t0: int = 500,
    num_back_steps: int = 100,
    eta: float = 0.0,
    correction_weight: float = 0.1,
) -> torch.Tensor:
    """Sinogram-conditioned SDEdit restoration with ASTRA projection consistency.
    
    Args:
        model: DDPM model
        diff: Diffusion process
        ddim: DDIM sampler
        img_hw: Input image [H,W] in [0,1] range (float32)
        target_sinogram: Target sinogram [A, D] (float32)
        vol_geom: ASTRA volume geometry
        proj_geom: ASTRA projection geometry
        projector_id: ASTRA projector ID
        device: Torch device
        image_size: Model input size
        t0: Starting noise level for SDEdit
        num_back_steps: Number of DDIM reverse steps
        eta: DDIM eta parameter (0 = DDIM, 1 = DDPM)
        correction_weight: Scaling factor for sinogram correction (default: 0.1)
    
    Returns:
        Restored image [H,W] in [0,1]
    """
    model.eval()

    H, W = img_hw.shape
    # Prepare input [1,1,H,W]
    x0 = torch.from_numpy(img_hw).float().unsqueeze(0).unsqueeze(0)
    if (H != image_size) or (W != image_size):
        x0 = torch.nn.functional.interpolate(
            x0, size=(image_size, image_size), mode='bilinear', align_corners=False
        )
    x0 = x0.to(device)

    # Forward diffusion to t0 (add noise)
    t_vec = torch.full((1,), min(t0, diff.T - 1), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = diff.q_sample(x0, t_vec, noise)

    # Prepare DDIM timesteps
    ts = ddim.set_timesteps(num_back_steps)
    # Find start index where t <= t0
    start_idx = 0
    for i, tval in enumerate(ts):
        if tval <= t_vec.item():
            start_idx = i
            break

    # Reverse diffusion with sinogram consistency
    x_cur = xt
    for i in range(start_idx, len(ts)):
        t = ts[i]
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        
        # Predict noise
        eps = model(x_cur, t_batch)
        
        # Get predicted x0 using DDIM formula
        alpha_t = diff.alphas_cumprod[t]
        # Convert scalar to tensor properly (avoid UserWarning)
        alpha_t_tensor = torch.as_tensor(alpha_t, device=device, dtype=x_cur.dtype)
        x0_pred = (x_cur - torch.sqrt(1.0 - alpha_t_tensor) * eps) / torch.sqrt(alpha_t_tensor)
        x0_pred = x0_pred.clamp(0, 1)
        
        # Apply sinogram consistency correction
        # 1. Forward project predicted x0
        x0_pred_np = x0_pred.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]
        if (H != image_size) or (W != image_size):
            # Need to resize to original size for ASTRA projection
            x0_pred_resized = torch.nn.functional.interpolate(
                x0_pred, size=(H, W), mode='bilinear', align_corners=False
            )
            x0_pred_np = x0_pred_resized.squeeze(0).squeeze(0).cpu().numpy()
        
        pred_sinogram = forward_project_astra(x0_pred_np, vol_geom, proj_geom, projector_id)  # [A, D]
        
        # 2. Normalize sinograms to same scale
        pred_sinogram_norm, target_sinogram_norm = normalize_sinogram_pair(pred_sinogram, target_sinogram)
        
        # 3. Compute sinogram difference (now in comparable ranges)
        sino_error = target_sinogram_norm - pred_sinogram_norm  # [A, D]
        
        # 4. Backproject error to image space
        correction_img = back_project_astra(sino_error, vol_geom, proj_geom, projector_id)  # [H, W]
        
        # 4. Apply correction to x0_pred
        correction_tensor = torch.from_numpy(correction_img).float().to(device)
        if (H != image_size) or (W != image_size):
            # Resize correction back to model size
            correction_tensor = correction_tensor.unsqueeze(0).unsqueeze(0)
            correction_tensor = torch.nn.functional.interpolate(
                correction_tensor, size=(image_size, image_size), 
                mode='bilinear', align_corners=False
            )
            correction_tensor = correction_tensor.squeeze(0).squeeze(0)
        else:
            correction_tensor = correction_tensor
        
        # Add correction to x0_pred
        x0_pred_corrected = x0_pred.squeeze(0).squeeze(0) + correction_weight * correction_tensor
        x0_pred_corrected = x0_pred_corrected.clamp(0, 1).unsqueeze(0).unsqueeze(0)
        
        # Continue DDIM step with corrected x0
        # Re-compute epsilon from corrected x0
        eps_corrected = (x_cur - torch.sqrt(alpha_t_tensor) * x0_pred_corrected) / torch.sqrt(1.0 - alpha_t_tensor)
        
        # DDIM step
        x_cur, _ = ddim.step_from_to(eps_corrected, t, t_prev, x_cur, eta=eta)
    
    x_rec = x_cur.clamp(0, 1)

    # Resize back to original dimensions
    if (H != image_size) or (W != image_size):
        x_rec = torch.nn.functional.interpolate(
            x_rec, size=(H, W), mode='bilinear', align_corners=False
        )

    return x_rec.squeeze(0).squeeze(0).detach().cpu()


def save_png_(img_hw: torch.Tensor, path: Path):
    """Save single-channel [H,W] float32 to PNG with proper normalization.
    
    Normalizes to [0, 1] using min-max scaling before saving.
    """
    try:
        import imageio.v2 as imageio
        # Normalize to [0, 1] for visualization
        img_np = img_hw.numpy()
        img_min = img_np.min()
        img_max = img_np.max()
        if img_max > img_min:
            img_normalized = (img_np - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(img_np)
        arr = (img_normalized * 255.0).round().astype(np.uint8)
        imageio.imwrite(path, arr)
    except Exception:
        # Fallback to matplotlib
        import matplotlib.pyplot as plt
        img_np = img_hw.numpy()
        img_min = img_np.min()
        img_max = img_np.max()
        if img_max > img_min:
            img_normalized = (img_np - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(img_np)
        plt.imsave(path, img_normalized, cmap='gray', vmin=0.0, vmax=1.0)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images.
    
    Args:
        img1: First image [H, W]
        img2: Second image [H, W]
    
    Returns:
        SSIM value (0-1, higher is better)
    """
    # Ensure both images are in the same range
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
    
    # Compute SSIM (only return the scalar value, not gradient maps)
    ssim_value = ssim(img1_norm, img2_norm, data_range=1.0, full=False)
    return float(ssim_value)


def main():
    ap = argparse.ArgumentParser(
        description="Sinogram-conditioned DDPM restoration with ASTRA projection consistency"
    )
    ap.add_argument(
        '--config',
        type=str,
        default='configs/default/chest.yaml',
        help='Path to config file with ASTRA geometry settings'
    )
    ap.add_argument(
        '--weights',
        type=str,
        default='outputs/ddpm/chest/model/best_val.pth',
        help='Path to model weights'
    )
    ap.add_argument(
        '--patient_id',
        type=str,
        required=True,
        help='Patient ID (e.g., "1")'
    )
    ap.add_argument(
        '--angle_range',
        type=float,
        required=True,
        help='Angle range in degrees (e.g., 60, 180)'
    )
    ap.add_argument(
        '--angle_step',
        type=float,
        required=True,
        help='Angle step in degrees (e.g., 0.25, 1.0)'
    )
    ap.add_argument(
        '--recon_root',
        type=str,
        default='data/interim/recon/chest',
        help='Directory containing reconstruction files'
    )
    ap.add_argument(
        '--proj_root',
        type=str,
        default='data/interim/proj/chest',
        help='Directory containing sinogram files'
    )
    ap.add_argument(
        '--out_root',
        type=str,
        default='outputs/ddpm/chest/restore_conditioned',
        help='Output directory for restored images'
    )
    ap.add_argument(
        '--correction_weight',
        type=float,
        default=0.1,
        help='Scaling factor for sinogram correction (default: 0.1)'
    )
    ap.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='Model input size'
    )
    ap.add_argument(
        '--ddim_steps',
        type=int,
        default=100,
        help='Number of DDIM steps'
    )
    ap.add_argument(
        '--t0',
        type=int,
        default=500,
        help='Starting timestep for SDEdit'
    )
    ap.add_argument(
        '--eta',
        type=float,
        default=0.0,
        help='DDIM eta parameter (0=DDIM, 1=DDPM)'
    )
    ap.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    ap.add_argument(
        '--max_slices',
        type=int,
        default=None,
        help='Optional limit on number of slices to process'
    )
    args = ap.parse_args()

    # Setup device
    requested_device = args.device.lower()
    if requested_device.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            print("[INFO] CUDA requested but not available; falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    cfg = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Calculate angles
    angles_deg = calculate_angles(args.angle_range, args.angle_step)
    print(f"Angle configuration: {args.angle_range}° with {args.angle_step}° step")
    print(f"Number of angles: {len(angles_deg)}")

    # Initialize ASTRA operators
    try:
        vol_geom, proj_geom, projector_id = create_astra_geometries(
            cfg, args.image_size, angles_deg
        )
        print(f"ASTRA geometry initialized:")
        print(f"  - Image size: {args.image_size}x{args.image_size}")
        print(f"  - Number of angles: {len(angles_deg)}")
        print(f"  - Angle range: {angles_deg[0]:.2f}° to {angles_deg[-1]:.2f}°")
    except Exception as e:
        print(f"[ERROR] Failed to initialize ASTRA operators: {e}")
        return

    # Match files
    recon_root = Path(args.recon_root)
    proj_root = Path(args.proj_root)
    
    # Format parameters for filename matching
    range_str = str(int(args.angle_range))
    # Keep the step as provided by user (will try multiple formats in match_files)
    step_str = str(args.angle_step)
    
    recon_path, sino_path = match_files(
        args.patient_id,
        range_str,
        step_str,
        recon_root,
        proj_root,
    )
    
    if recon_path is None or sino_path is None:
        print("[ERROR] Could not find matching reconstruction and sinogram files")
        return

    print(f"\nProcessing files:")
    print(f"  Reconstruction: {recon_path}")
    print(f"  Sinogram: {sino_path}")

    # Load volumes
    try:
        recon_vol = load_volume(recon_path)  # [Z, H, W]
        sino_vol = load_sinogram(sino_path)  # [Z, A, D]
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # Load ground truth (360 degree reconstruction) for SSIM calculation
    gt_path = recon_root / f"recon_{args.patient_id}_360@0.25.npy"
    gt_vol = None
    if gt_path.exists():
        try:
            gt_vol = load_volume(gt_path)  # [Z, H, W]
            print(f"  Ground truth (360°): {gt_path}")
            print(f"  Ground truth shape: {gt_vol.shape}")
        except Exception as e:
            print(f"[WARNING] Failed to load ground truth for SSIM: {e}")
            print(f"[INFO] Will skip SSIM calculation")
    else:
        print(f"[WARNING] Ground truth not found: {gt_path}")
        print(f"[INFO] Will skip SSIM calculation")

    Z_recon, H, W = recon_vol.shape
    Z_sino, A_sino, D_sino = sino_vol.shape

    print(f"\nData shapes:")
    print(f"  Reconstruction: {recon_vol.shape} (Z, H, W)")
    print(f"  Sinogram: {sino_vol.shape} (Z, A, D)")

    # Validate dimensions
    if Z_recon != Z_sino:
        print(f"[WARNING] Slice count mismatch: recon has {Z_recon} slices, sinogram has {Z_sino} slices")
        print(f"[INFO] Will process minimum: {min(Z_recon, Z_sino)} slices")
        Z = min(Z_recon, Z_sino)
    else:
        Z = Z_recon

    if A_sino != len(angles_deg):
        print(f"[WARNING] Angle count mismatch: sinogram has {A_sino} angles, expected {len(angles_deg)}")
        print(f"[INFO] Using sinogram's angle count: {A_sino}")

    if D_sino != cfg.get('projection', {}).get('geom', {}).get('det', {}).get('det_count', 1000):
        det_count_config = cfg.get('projection', {}).get('geom', {}).get('det', {}).get('det_count', 1000)
        print(f"[WARNING] Detector count mismatch: sinogram has {D_sino}, ASTRA config has {det_count_config}")

    # Load model
    print(f"\nLoading model from: {args.weights}")
    model = SimpleUNet(in_ch=1).to(device)
    try:
        state = torch.load(args.weights, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Initialize diffusion
    T = 1000
    diff = Diffusion(T=T).to(device)
    ddim = DDIM(T=T, eta=args.eta).to(device)

    # Create output directory
    out_dir = Path(args.out_root) / f"recon_{args.patient_id}_{range_str}@{step_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    # Process slices
    num_slices = args.max_slices if args.max_slices is not None else Z
    num_slices = min(num_slices, Z)
    
    print(f"\nProcessing {num_slices} slices...")
    print(f"Correction weight: {args.correction_weight}")
    if gt_vol is not None:
        print(f"SSIM calculation: Enabled (comparing to 360° reconstruction)")
    else:
        print(f"SSIM calculation: Disabled (ground truth not available)")
    
    pad = max(4, len(str(max(0, num_slices - 1))))
    
    # Track SSIM values
    ssim_values = []
    
    pbar = tqdm(range(num_slices), desc="Restoring slices")
    
    # Track statistics for first slice (debugging)
    first_slice_debug = True
    
    for i in pbar:
        # Extract slice
        recon_slice = recon_vol[i]  # [H, W]
        sino_slice = sino_vol[i]  # [A, D]
        
        # Normalize reconstruction to [0, 1]
        mn = float(recon_slice.min())
        mx = float(recon_slice.max())
        if mx > mn:
            recon_slice_01 = (recon_slice - mn) / (mx - mn)
        else:
            recon_slice_01 = np.zeros_like(recon_slice, dtype=np.float32)
        
        # Perform conditioned restoration
        try:
            # Print debug info for first slice
            if first_slice_debug:
                print(f"\n[DEBUG] First slice statistics:")
                print(f"  Input recon range: [{recon_slice.min():.4f}, {recon_slice.max():.4f}]")
                print(f"  Normalized recon range: [{recon_slice_01.min():.4f}, {recon_slice_01.max():.4f}]")
                print(f"  Target sinogram range: [{sino_slice.min():.4f}, {sino_slice.max():.4f}]")
                print(f"  Target sinogram mean/std: {sino_slice.mean():.4f} / {sino_slice.std():.4f}")
                first_slice_debug = False
            
            x_rec = conditioned_seddit_restore_slice(
                model=model,
                diff=diff,
                ddim=ddim,
                img_hw=recon_slice_01,
                target_sinogram=sino_slice,
                vol_geom=vol_geom,
                proj_geom=proj_geom,
                projector_id=projector_id,
                device=device,
                image_size=args.image_size,
                t0=args.t0,
                num_back_steps=args.ddim_steps,
                eta=args.eta,
                correction_weight=args.correction_weight,
            )
            
            # Compute SSIM with ground truth if available
            if gt_vol is not None and i < gt_vol.shape[0]:
                gt_slice = gt_vol[i]
                x_rec_np = x_rec.numpy()
                
                # Compute SSIM
                ssim_value = compute_ssim(x_rec_np, gt_slice)
                ssim_values.append(ssim_value)
                
                # Update progress bar with SSIM
                pbar.set_postfix({'SSIM': f'{ssim_value:.4f}'})
            
            # Save PNG
            save_path = out_dir / f"{i:0{pad}d}.png"
            save_png_(x_rec, save_path)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process slice {i}: {e}")
            continue

    print(f"\n✓ Restoration complete!")
    print(f"✓ Saved {num_slices} restored slices to: {out_dir}")
    
    # Print SSIM statistics
    if ssim_values:
        ssim_array = np.array(ssim_values)
        print(f"\n📊 SSIM Statistics (vs 360° reconstruction):")
        print(f"  Mean SSIM: {ssim_array.mean():.4f}")
        print(f"  Std SSIM:  {ssim_array.std():.4f}")
        print(f"  Min SSIM:  {ssim_array.min():.4f}")
        print(f"  Max SSIM:  {ssim_array.max():.4f}")
        
        # Save SSIM values to file
        ssim_file = out_dir / "ssim_values.txt"
        with open(ssim_file, 'w') as f:
            f.write("# SSIM values per slice (comparing to 360° reconstruction)\n")
            f.write(f"# Mean: {ssim_array.mean():.6f}\n")
            f.write(f"# Std:  {ssim_array.std():.6f}\n")
            f.write(f"# Min:  {ssim_array.min():.6f}\n")
            f.write(f"# Max:  {ssim_array.max():.6f}\n")
            f.write("# Slice_ID\tSSIM\n")
            for idx, val in enumerate(ssim_values):
                f.write(f"{idx}\t{val:.6f}\n")
        print(f"✓ SSIM values saved to: {ssim_file}")
    
    # Cleanup ASTRA
    try:
        if astra is not None and projector_id is not None:
            astra.projector.delete(projector_id)
    except Exception:
        pass


if __name__ == '__main__':
    main()
