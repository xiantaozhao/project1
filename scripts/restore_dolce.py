#!/usr/bin/env python3
"""
DOLCE Restoration Script with Proximal Solver Data Consistency
Uses ASTRA for CT projection operations
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.model.dolce import (
    UNetDOLCE,
    create_unet_dolce,
    GaussianDiffusion,
    create_gaussian_diffusion,
    CTClass_astra,
    create_ct_data_fidelity,
)
from src.configs.configloading import load_config


def normalize_volume(vol):
    """Normalize volume to [0, 1] using MinMax."""
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        return (vol - vmin) / (vmax - vmin)
    return vol


def load_npy_files(directory, pattern="*.npy"):
    """Load all .npy files from directory matching pattern."""
    files = sorted(Path(directory).glob(pattern))
    print(f"Found {len(files)} files matching '{pattern}' in {directory}")
    return files


def match_files(recon_file, sino_dir):
    """Match reconstruction file to corresponding sinogram file."""
    stem = recon_file.stem
    
    # Try different patterns
    patterns = [
        f"{stem}.npy",  # exact match
        f"{stem.replace('@', '@')}.npy",  # same format
    ]
    
    # Handle integer vs decimal format (@1 vs @1.0)
    if '@' in stem:
        parts = stem.split('@')
        if len(parts) == 2:
            base, num = parts
            # Try both integer and decimal
            if '.' in num:
                # Has decimal, try without
                int_part = num.split('.')[0]
                patterns.append(f"{base}@{int_part}.npy")
            else:
                # Is integer, try with .0
                patterns.append(f"{base}@{num}.0.npy")
    
    # Search for matching file
    for pattern in patterns:
        sino_file = sino_dir / pattern
        if sino_file.exists():
            return sino_file
    
    return None


def restore_volume_dolce(
    model,
    diffusion,
    target_sinogram,
    fbp_condition,
    rls_condition,
    angles,
    geometry_config,
    device='cuda',
    sampler='ddim',
    ddim_steps=100,
    eta=0.0,
    start_timestep=None,
    use_proximal_solver=True,
    rho=1.0,
    solver_type='apgm',
    solver_iterations=10,
    verbose=True,
):
    """
    Restore volume using DOLCE with proximal solver for data consistency.
    
    Args:
        model: UNetDOLCE model
        diffusion: GaussianDiffusion instance
        target_sinogram: Target sinogram (Z, num_angles, det_count)
        fbp_condition: FBP reconstruction (Z, H, W) or None
        rls_condition: RLS reconstruction (Z, H, W) or None
        angles: Projection angles in radians
        geometry_config: ASTRA geometry configuration
        device: torch device
        sampler: 'ddpm' or 'ddim'
        ddim_steps: Number of DDIM steps
        eta: DDIM stochasticity parameter
        start_timestep: Starting timestep for SDEdit-style
        use_proximal_solver: Whether to use proximal solver
        rho: Proximal solver parameter
        solver_type: 'apgm' or 'cgrad'
        solver_iterations: Number of solver iterations
        verbose: Show progress bar
        
    Returns:
        Restored volume (Z, H, W)
    """
    model.eval()
    
    Z = target_sinogram.shape[0]
    img_size = geometry_config['img_size']
    
    # Prepare conditions
    if fbp_condition is not None:
        fbp_norm = normalize_volume(fbp_condition)
        fbp_tensor = torch.from_numpy(fbp_norm).float().unsqueeze(1).to(device)  # (Z, 1, H, W)
    else:
        fbp_tensor = None
        
    if rls_condition is not None:
        rls_norm = normalize_volume(rls_condition)
        rls_tensor = torch.from_numpy(rls_norm).float().unsqueeze(1).to(device)  # (Z, 1, H, W)
    else:
        rls_tensor = None
    
    # Normalize target sinogram
    sino_norm = normalize_volume(target_sinogram)
    
    # Restore slice by slice
    restored_slices = []
    
    for z in tqdm(range(Z), desc="Restoring slices", disable=not verbose):
        # Get slice sinogram
        sino_slice = sino_norm[z]  # (num_angles, det_count)
        
        # Create CT data fidelity object for this slice
        ct_fidelity = create_ct_data_fidelity(
            target_sinogram=sino_slice,
            angles=angles,
            config=geometry_config,
            device=device,
        )
        
        # Get conditions for this slice
        fbp_slice = fbp_tensor[z:z+1] if fbp_tensor is not None else None
        rls_slice = rls_tensor[z:z+1] if rls_tensor is not None else None
        
        # Initialize from FBP or noise
        if start_timestep is not None and fbp_slice is not None:
            x_start = fbp_slice
        else:
            x_start = None
        
        # Sample using diffusion with data consistency
        with torch.no_grad():
            x_restored, _ = diffusion.sample_loop(
                shape=(1, 1, img_size, img_size),
                condition_fbp=fbp_slice,
                condition_rls=rls_slice,
                ct_data_fidelity=ct_fidelity if use_proximal_solver else None,
                sampler=sampler,
                ddim_steps=ddim_steps,
                eta=eta,
                start_timestep=start_timestep,
                x_start=x_start,
                use_proximal_solver=use_proximal_solver,
                rho=rho,
                solver_type=solver_type,
                solver_iterations=solver_iterations,
                verbose=False,
            )
        
        # Extract slice
        restored_slice = x_restored[0, 0].cpu().numpy()
        restored_slices.append(restored_slice)
    
    restored_volume = np.stack(restored_slices, axis=0)
    return restored_volume


def save_comparison_images(gt, fbp, restored, output_dir, slice_indices=None, prefix=""):
    """Save comparison images of GT, FBP, and restored."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if slice_indices is None:
        # Select evenly spaced slices
        Z = gt.shape[0]
        slice_indices = [Z // 4, Z // 2, 3 * Z // 4]
    
    for idx in slice_indices:
        if idx >= gt.shape[0]:
            continue
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(gt[idx], cmap='gray')
        axes[0].set_title('Ground Truth (360°)')
        axes[0].axis('off')
        
        axes[1].imshow(fbp[idx], cmap='gray')
        axes[1].set_title('FBP (Limited Angle)')
        axes[1].axis('off')
        
        axes[2].imshow(restored[idx], cmap='gray')
        axes[2].set_title('DOLCE Restored')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}comparison_slice_{idx:03d}.png", dpi=150, bbox_inches='tight')
        plt.close()


def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1, img2, data_range=img1.max() - img1.min())


def main():
    parser = argparse.ArgumentParser(description="DOLCE restoration with proximal solver")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--recon_dir", type=str, required=True, help="Directory with limited-angle reconstructions")
    parser.add_argument("--sino_dir", type=str, required=True, help="Directory with sinograms")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory with ground truth 360° reconstructions")
    parser.add_argument("--output_dir", type=str, default="outputs/dolce/restored", help="Output directory")
    
    # Geometry parameters
    parser.add_argument("--angle_range", type=float, default=60, help="Angular range in degrees")
    parser.add_argument("--angle_step", type=float, default=1.0, help="Angular step in degrees")
    parser.add_argument("--start_angle", type=float, default=0.0, help="Start angle in degrees")
    
    # Sampling parameters
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"], help="Sampling method")
    parser.add_argument("--ddim_steps", type=int, default=100, help="Number of DDIM steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic)")
    parser.add_argument("--start_t", type=int, default=500, help="Starting timestep for SDEdit-style")
    
    # Data consistency parameters
    parser.add_argument("--use_prox_solver", action="store_true", help="Use proximal solver")
    parser.add_argument("--rho", type=float, default=1.0, help="Proximal solver parameter")
    parser.add_argument("--solver_type", type=str, default="apgm", choices=["apgm", "cgrad"], help="Solver type")
    parser.add_argument("--solver_iterations", type=int, default=10, help="Solver iterations per step")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently only 1 supported)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = load_config(Path(args.config))
    model_cfg = cfg.get("model", {})
    diff_cfg = cfg.get("diffusion", {})
    data_cfg = cfg.get("data", {})
    
    # Create model
    print("Creating model...")
    model = create_unet_dolce(
        in_channels=data_cfg.get("channels", 1),
        out_channels=data_cfg.get("channels", 1),
        model_channels=model_cfg.get("model_channels", 128),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        channel_mult=tuple(model_cfg.get("channel_mult", [1, 2, 2, 4])),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [8, 16])),
        dropout=0.0,  # No dropout during inference
        use_condition=model_cfg.get("use_fbp_condition", True) or model_cfg.get("use_rls_condition", False),
        condition_channels=1,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create diffusion
    diffusion = create_gaussian_diffusion(
        model,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        objective=diff_cfg.get("objective", "pred_noise"),
    )
    diffusion = diffusion.to(device)
    
    # Setup geometry
    geometry_cfg = cfg.get("geometry", {})
    geometry_config = {
        'det_count': geometry_cfg.get('det_count', 1000),
        'det_pixel_mm': geometry_cfg.get('det_pixel_mm', 0.7),
        'source_origin': geometry_cfg.get('DSO', 1000.0),
        'origin_det': geometry_cfg.get('ODD', 600.0),
        'img_size': data_cfg.get('image_size', 512),
    }
    
    # Create angles
    num_angles = int(args.angle_range / args.angle_step) + 1
    angles_deg = np.linspace(args.start_angle, args.start_angle + args.angle_range, num_angles)
    angles_rad = np.deg2rad(angles_deg)
    
    print(f"\nGeometry setup:")
    print(f"  Angle range: {args.angle_range}° @ {args.angle_step}° step")
    print(f"  Number of angles: {num_angles}")
    print(f"  Detector: {geometry_config['det_count']} pixels × {geometry_config['det_pixel_mm']} mm")
    print(f"  DSO/ODD: {geometry_config['source_origin']}/{geometry_config['origin_det']} mm")
    
    # Load files
    recon_files = load_npy_files(args.recon_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ssim_values = []
    
    # Process each reconstruction
    for recon_file in tqdm(recon_files, desc="Processing files"):
        print(f"\n{'='*80}")
        print(f"Processing: {recon_file.name}")
        
        # Load limited-angle reconstruction (FBP condition)
        fbp_vol = np.load(recon_file)
        print(f"Loaded FBP: shape={fbp_vol.shape}")
        
        # Match and load sinogram
        sino_file = match_files(recon_file, Path(args.sino_dir))
        if sino_file is None:
            print(f"Warning: No matching sinogram found for {recon_file.name}, skipping")
            continue
        
        sino_vol = np.load(sino_file)
        print(f"Loaded sinogram: shape={sino_vol.shape}")
        
        # Validate shapes
        if sino_vol.shape[0] != fbp_vol.shape[0]:
            print(f"Warning: Slice count mismatch (sino: {sino_vol.shape[0]}, fbp: {fbp_vol.shape[0]})")
            continue
        
        # Restore volume
        print(f"\nRestoring with DOLCE...")
        print(f"  Sampler: {args.sampler}")
        print(f"  DDIM steps: {args.ddim_steps}")
        print(f"  Start timestep: {args.start_t}")
        print(f"  Proximal solver: {args.use_prox_solver}")
        if args.use_prox_solver:
            print(f"  Solver type: {args.solver_type}")
            print(f"  Rho: {args.rho}")
            print(f"  Solver iterations: {args.solver_iterations}")
        
        restored_vol = restore_volume_dolce(
            model=model,
            diffusion=diffusion,
            target_sinogram=sino_vol,
            fbp_condition=fbp_vol,
            rls_condition=None,
            angles=angles_rad,
            geometry_config=geometry_config,
            device=device,
            sampler=args.sampler,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
            start_timestep=args.start_t,
            use_proximal_solver=args.use_prox_solver,
            rho=args.rho,
            solver_type=args.solver_type,
            solver_iterations=args.solver_iterations,
            verbose=True,
        )
        
        # Save restored volume
        output_file = output_dir / f"restored_{recon_file.stem}.npy"
        np.save(output_file, restored_vol)
        print(f"Saved restored volume: {output_file}")
        
        # Load ground truth if available
        gt_vol = None
        if args.gt_dir:
            gt_file = Path(args.gt_dir) / recon_file.name
            if gt_file.exists():
                gt_vol = np.load(gt_file)
                print(f"Loaded ground truth: shape={gt_vol.shape}")
                
                # Compute SSIM
                if gt_vol.shape == restored_vol.shape:
                    gt_norm = normalize_volume(gt_vol)
                    restored_norm = normalize_volume(restored_vol)
                    
                    slice_ssims = []
                    for z in range(gt_norm.shape[0]):
                        ssim_val = compute_ssim(gt_norm[z], restored_norm[z])
                        slice_ssims.append(ssim_val)
                    
                    mean_ssim = np.mean(slice_ssims)
                    ssim_values.append(mean_ssim)
                    print(f"SSIM: {mean_ssim:.4f} (min: {np.min(slice_ssims):.4f}, max: {np.max(slice_ssims):.4f})")
        
        # Save comparison images
        if gt_vol is not None:
            save_comparison_images(
                gt_vol, fbp_vol, restored_vol,
                output_dir / "comparisons",
                prefix=f"{recon_file.stem}_"
            )
    
    # Save SSIM summary
    if ssim_values:
        ssim_file = output_dir / "ssim_values.txt"
        with open(ssim_file, 'w') as f:
            f.write("File\tSSIM\n")
            for file, ssim_val in zip(recon_files, ssim_values):
                f.write(f"{file.name}\t{ssim_val:.6f}\n")
            f.write(f"\nMean SSIM: {np.mean(ssim_values):.6f}\n")
            f.write(f"Std SSIM: {np.std(ssim_values):.6f}\n")
        print(f"\nSaved SSIM values to {ssim_file}")
        print(f"Mean SSIM: {np.mean(ssim_values):.6f} ± {np.std(ssim_values):.6f}")
    
    print("\n" + "="*80)
    print("DOLCE restoration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
