"""
Training logic for DOLCE model with conditional inputs
Extends train_ddpm.py with FBP/RLS conditioning support
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from src.model.dolce import (
    ConditionalModel,
    GaussianDiffusion,
    create_gaussian_diffusion,
)
from src.data.data_load import data_load_chest
# Note: FBP conditions should be pre-generated using scripts/generate_fbp_conditions.py


class ConditionalSliceDataset(Dataset):
    """
    Dataset for DOLCE training with FBP/RLS conditioning.
    Each sample consists of: ground truth image + FBP condition + optionally RLS condition
    """

    def __init__(
        self,
        volumes_hu: Sequence[np.ndarray],
        fbp_volumes: Optional[Sequence[np.ndarray]],
        rls_volumes: Optional[Sequence[np.ndarray]],
        *,
        image_size: int,
        use_mu: bool,
        mu_water: float,
        hu_clip_range: Tuple[float, float] | None,
        centered: bool,
    ) -> None:
        self.image_size = image_size
        self.use_mu = use_mu
        self.mu_water = mu_water
        self.centered = centered
        
        self.gt_volumes: List[np.ndarray] = []
        self.fbp_volumes: List[Optional[np.ndarray]] = []
        self.rls_volumes: List[Optional[np.ndarray]] = []
        self.index: List[Tuple[int, int]] = []
        
        lo_hi = None
        if hu_clip_range is not None and len(hu_clip_range) >= 2:
            lo_hi = (float(hu_clip_range[0]), float(hu_clip_range[1]))
            
        # Process volumes
        for vidx, vol in enumerate(volumes_hu):
            if vol.ndim != 3:
                raise ValueError(f"Expected volume shape [Z,H,W], got {vol.shape}")
            arr = vol.astype(np.float32, copy=False)
            if lo_hi is not None:
                np.clip(arr, lo_hi[0], lo_hi[1], out=arr)
            self.gt_volumes.append(arr)
            
            # Get FBP volume if available
            if fbp_volumes is not None and vidx < len(fbp_volumes):
                fbp_arr = fbp_volumes[vidx].astype(np.float32, copy=False)
                if lo_hi is not None:
                    np.clip(fbp_arr, lo_hi[0], lo_hi[1], out=fbp_arr)
                self.fbp_volumes.append(fbp_arr)
            else:
                self.fbp_volumes.append(None)
                
            # Get RLS volume if available
            if rls_volumes is not None and vidx < len(rls_volumes):
                rls_arr = rls_volumes[vidx].astype(np.float32, copy=False)
                if lo_hi is not None:
                    np.clip(rls_arr, lo_hi[0], lo_hi[1], out=rls_arr)
                self.rls_volumes.append(rls_arr)
            else:
                self.rls_volumes.append(None)
                
            # Build index
            self.index.extend((vidx, s) for s in range(arr.shape[0]))
            
    def _normalize_slice(self, sl_hu: np.ndarray) -> np.ndarray:
        """Normalize slice to [0, 1] or [-1, 1] if centered."""
        if self.use_mu:
            sl = self.mu_water * (1.0 + sl_hu / 1000.0)
        else:
            sl = sl_hu
            
        sl = sl.astype(np.float32, copy=False)
        sl_min = float(sl.min())
        sl_max = float(sl.max())
        
        if sl_max > sl_min:
            sl_norm = (sl - sl_min) / (sl_max - sl_min)
        else:
            sl_norm = np.zeros_like(sl, dtype=np.float32)
            
        if self.centered:
            sl_norm = sl_norm * 2.0 - 1.0
            
        return sl_norm
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        vol_idx, slice_idx = self.index[idx]
        
        # Get GT slice
        sl_hu = self.gt_volumes[vol_idx][slice_idx]
        sl_norm = self._normalize_slice(sl_hu)
        
        x = torch.from_numpy(sl_norm).float().unsqueeze(0).unsqueeze(0)
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size), 
                mode="bilinear", align_corners=False
            )
        x = x.squeeze(0)
        
        # Get FBP condition if available
        fbp_cond = None
        if self.fbp_volumes[vol_idx] is not None:
            fbp_sl = self.fbp_volumes[vol_idx][slice_idx]
            fbp_norm = self._normalize_slice(fbp_sl)
            fbp_cond = torch.from_numpy(fbp_norm).float().unsqueeze(0).unsqueeze(0)
            if fbp_cond.shape[-2] != self.image_size or fbp_cond.shape[-1] != self.image_size:
                fbp_cond = F.interpolate(
                    fbp_cond, size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False
                )
            fbp_cond = fbp_cond.squeeze(0)
        
        # Get RLS condition if available
        rls_cond = None
        if self.rls_volumes[vol_idx] is not None:
            rls_sl = self.rls_volumes[vol_idx][slice_idx]
            rls_norm = self._normalize_slice(rls_sl)
            rls_cond = torch.from_numpy(rls_norm).float().unsqueeze(0).unsqueeze(0)
            if rls_cond.shape[-2] != self.image_size or rls_cond.shape[-1] != self.image_size:
                rls_cond = F.interpolate(
                    rls_cond, size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False
                )
            rls_cond = rls_cond.squeeze(0)
            
        result = {
            "image": x,
            "slice_idx": slice_idx,
            "volume_idx": vol_idx,
        }
        
        if fbp_cond is not None:
            result["condition_fbp"] = fbp_cond
        if rls_cond is not None:
            result["condition_rls"] = rls_cond
            
        return result


def save_png_grid(
    x: torch.Tensor,
    path: Path,
    nrow: int = 4,
    *,
    title: str | None = None,
    subtitle: str | None = None,
):
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)
    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            axes[i, j].axis('off')
            if idx < N:
                img = x[idx]
                if img.ndim == 2:
                    axes[i, j].imshow(img, cmap='gray')
                else:
                    import numpy as _np
                    axes[i, j].imshow(_np.transpose(img, (1, 2, 0)))
    if title or subtitle:
        text = title or ""
        if subtitle:
            text = f"{text}\n{subtitle}" if text else subtitle
        fig.suptitle(text, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    Compute SSIM and PSNR between predicted and ground truth images.
    
    Args:
        pred: Predicted images [B, C, H, W]
        gt: Ground truth images [B, C, H, W]
        
    Returns:
        Dictionary with 'ssim' and 'psnr' keys
    """
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    
    # Handle channel dimension
    if pred_np.shape[1] == 1:
        pred_np = pred_np[:, 0]  # [B, H, W]
        gt_np = gt_np[:, 0]
    
    ssim_values = []
    psnr_values = []
    
    for i in range(pred_np.shape[0]):
        pred_slice = pred_np[i]
        gt_slice = gt_np[i]
        
        # Normalize to [0, 1] for metrics
        pred_norm = (pred_slice - pred_slice.min()) / (pred_slice.max() - pred_slice.min() + 1e-8)
        gt_norm = (gt_slice - gt_slice.min()) / (gt_slice.max() - gt_slice.min() + 1e-8)
        
        # Compute SSIM
        ssim_val = ssim(gt_norm, pred_norm, data_range=1.0)
        ssim_values.append(ssim_val)
        
        # Compute PSNR
        mse = np.mean((gt_norm - pred_norm) ** 2)
        if mse < 1e-10:
            psnr_val = 100.0
        else:
            psnr_val = 20 * np.log10(1.0 / np.sqrt(mse))
        psnr_values.append(psnr_val)
    
    return {
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_std': float(np.std(ssim_values)),
        'psnr_mean': float(np.mean(psnr_values)),
        'psnr_std': float(np.std(psnr_values)),
    }


def resolve_patient_splits(data_cfg: Dict) -> Dict[str, List[str]]:
    """Resolve patient IDs for train/val/test splits."""
    phases = ("train", "val", "test")
    splits: Dict[str, List[str]] = {phase: [] for phase in phases}
    assigned: set[str] = set()
    
    def _normalize_list(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(x) for x in value]
        raise TypeError(f"Expected list/tuple/set of patient ids, got {type(value)!r}")
    
    def _extend(phase: str, ids: Sequence[str]):
        for pid in ids:
            pid_str = str(pid)
            if pid_str in assigned:
                continue
            splits[phase].append(pid_str)
            assigned.add(pid_str)
    
    # Explicit assignments
    for phase in phases:
        explicit = data_cfg.get(f"{phase}_patient_ids")
        _extend(phase, _normalize_list(explicit))
    
    # Get base patient IDs
    base_ids = _normalize_list(data_cfg.get("patient_ids"))
    if not base_ids and not splits["train"]:
        base_ids = _normalize_list(data_cfg.get("train_patient_ids"))
        _extend("train", base_ids)
        return splits
    
    remaining = [pid for pid in base_ids if pid not in assigned]
    
    # Handle split_counts or split_ratio
    counts_cfg = data_cfg.get("split_counts") or {}
    ratios_cfg = data_cfg.get("split_ratio")
    
    def _apply_counts(counts: Mapping[str, float | int]) -> None:
        nonlocal remaining
        for phase in phases:
            desired = counts.get(phase)
            if desired is None:
                continue
            need = max(0, int(desired) - len(splits[phase]))
            if need <= 0:
                continue
            if need > len(remaining):
                raise ValueError(
                    f"Not enough patient IDs to satisfy split_counts for {phase}:"
                    f" requested {desired}, remaining {len(remaining)}"
                )
            to_add = remaining[:need]
            remaining = remaining[need:]
            _extend(phase, to_add)
    
    if counts_cfg:
        _apply_counts(counts_cfg)
    elif ratios_cfg:
        ratios = {phase: float(val) for phase, val in ratios_cfg.items() if phase in phases}
        total_ratio = sum(ratios.values())
        if total_ratio <= 0:
            raise ValueError("split_ratio must contain positive numbers")
        total_available = len(remaining)
        counts = {
            phase: int(round(total_available * ratios.get(phase, 0.0) / total_ratio))
            for phase in phases
        }
        while sum(counts.values()) > total_available:
            for phase in phases:
                if counts[phase] > 0:
                    counts[phase] -= 1
                    if sum(counts.values()) <= total_available:
                        break
        _apply_counts(counts)
    
    # Assign remaining to train
    if not splits["train"]:
        if not remaining:
            raise ValueError("No patient IDs assigned to training split")
        _extend("train", remaining)
        remaining = []
    else:
        if remaining:
            _extend("train", remaining)
            remaining = []
    
    return splits


def load_volumes(
    patient_ids: Sequence[str],
    modality: Literal["CT", "DX", "CR"],
    volume_path: Optional[Path] = None,
    total_angle: Optional[int] = None,
    angle_step: Optional[float] = None,
) -> Tuple[List[np.ndarray], Optional[Tuple[float, float, float]]]:
    """
    Load patient volumes.
    
    Args:
        patient_ids: List of patient IDs
        modality: Imaging modality (CT, DX, CR)
        volume_path: If provided, load from .npy files instead of DICOM
        total_angle: Total scanning angle (e.g., 360 for ground truth)
        angle_step: Angle step (e.g., 0.25 for fine sampling)
    """
    if not patient_ids:
        return [], None
    
    volumes: List[np.ndarray] = []
    first_spacing: Optional[Tuple[float, float, float]] = None
    
    for pid in patient_ids:
        # If volume_path is provided, load from .npy files
        if volume_path is not None and total_angle is not None and angle_step is not None:
            vol_file = find_recon_file(volume_path, pid, total_angle, angle_step)
            if vol_file is None or not vol_file.exists():
                print(f"Warning: Ground truth volume not found for patient {pid}")
                print(f"  Expected: recon_{pid}_{total_angle}@{angle_step}.npy (or similar)")
                continue
            vol = np.load(vol_file)
            volumes.append(vol)
            print(f"Loaded patient {pid}: {vol_file.name} shape={vol.shape}")
            # Assume standard CT spacing if loading from file
            if first_spacing is None:
                first_spacing = (1.0, 1.0, 1.0)
        else:
            # Load from DICOM using data_load_chest
            vol, spacing, _ = data_load_chest.load_data_chest(pid, modality)
            volumes.append(vol)
            if first_spacing is None:
                first_spacing = spacing
            print(f"Loaded patient {pid}: shape={vol.shape}, spacing(dzyx)={spacing}")
    
    return volumes, first_spacing


def find_recon_file(
    recon_dir: Path,
    patient_id: str,
    total_angle: int,
    angle_step: float,
) -> Optional[Path]:
    """
    Find reconstruction file with flexible naming support.
    Tries multiple formats: @1.0, @1, @2.5, @2, etc.
    
    Args:
        recon_dir: Directory containing reconstruction files
        patient_id: Patient ID (e.g., "1", "2")
        total_angle: Total angle (e.g., 60, 90, 120, 360)
        angle_step: Angle step (e.g., 1.0, 2.5, 10.0)
    
    Returns:
        Path to the file if found, None otherwise
    """
    import glob
    
    # Try exact float format
    pattern1 = recon_dir / f"recon_{patient_id}_{total_angle}@{angle_step}.npy"
    if pattern1.exists():
        return pattern1
    
    # Try integer format (for values like 1.0 -> 1)
    if angle_step == int(angle_step):
        pattern2 = recon_dir / f"recon_{patient_id}_{total_angle}@{int(angle_step)}.npy"
        if pattern2.exists():
            return pattern2
    
    # Try with one decimal place
    pattern3 = recon_dir / f"recon_{patient_id}_{total_angle}@{angle_step:.1f}.npy"
    if pattern3.exists():
        return pattern3
    
    # Fallback: Use glob to find any matching file
    glob_pattern = str(recon_dir / f"recon_{patient_id}_{total_angle}@*.npy")
    matches = glob.glob(glob_pattern)
    if matches:
        # Filter to find closest match to the requested angle_step
        for match_path in matches:
            filename = Path(match_path).name
            # Extract angle step from filename: recon_1_60@1.0.npy -> 1.0
            try:
                step_str = filename.split('@')[1].replace('.npy', '')
                file_step = float(step_str)
                if abs(file_step - angle_step) < 0.01:  # Close enough
                    return Path(match_path)
            except (IndexError, ValueError):
                continue
        # If no close match, return first one
        print(f"Warning: Using {matches[0]} for angle_step={angle_step}")
        return Path(matches[0])
    
    return None


def load_conditional_volumes(
    patient_ids: Sequence[str],
    condition_type: str,
    condition_path: Optional[Path],
    total_angle: Optional[int] = None,
    angle_step: Optional[float] = None,
) -> Optional[List[np.ndarray]]:
    """
    Load conditional volumes (FBP or RLS reconstructions).
    If condition_path is provided, load from disk.
    Otherwise, return None and conditions will be generated on-the-fly.
    
    Args:
        patient_ids: List of patient IDs
        condition_type: Type of condition ("FBP", "RLS", etc.)
        condition_path: Path to directory containing condition files
        total_angle: Total scanning angle (e.g., 60, 90, 120, 360)
        angle_step: Angle step (e.g., 1.0, 2.5, 10.0)
    """
    if condition_path is None:
        print(f"No {condition_type} condition path provided, will generate on-the-fly")
        return None
    
    print(f"Loading {condition_type} conditions from {condition_path}")
    if total_angle is not None and angle_step is not None:
        print(f"  Looking for angle={total_angle}°, step={angle_step}°")
    
    volumes = []
    for pid in patient_ids:
        # If angle parameters are provided, use the flexible file finder
        if total_angle is not None and angle_step is not None:
            cond_file = find_recon_file(condition_path, pid, total_angle, angle_step)
        else:
            # Fallback to old naming convention
            cond_file = condition_path / f"{pid}_{condition_type}.npy"
        
        if cond_file is None or not cond_file.exists():
            print(f"Warning: {condition_type} condition not found for patient {pid}")
            if total_angle and angle_step:
                print(f"  Expected: recon_{pid}_{total_angle}@{angle_step}.npy (or similar)")
            volumes.append(None)
        else:
            cond_vol = np.load(cond_file)
            volumes.append(cond_vol)
            print(f"Loaded {condition_type} for patient {pid}: {cond_file.name} shape={cond_vol.shape}")
    
    return volumes if any(v is not None for v in volumes) else None


def evaluate_conditional_mse(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataloader: Optional[DataLoader],
    device: torch.device,
    use_fbp: bool,
    use_rls: bool,
) -> Optional[float]:
    """Evaluate noise prediction MSE with conditions."""
    if dataloader is None or len(dataloader) == 0:
        return None
    
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x0 = batch["image"].to(device)
            B = x0.size(0)
            t = torch.randint(0, diffusion.timesteps, (B,), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            
            # Get conditions
            fbp_cond = batch.get("condition_fbp").to(device) if use_fbp and "condition_fbp" in batch else None
            rls_cond = batch.get("condition_rls").to(device) if use_rls and "condition_rls" in batch else None
            
            pred = model(xt, t, condition_fbp=fbp_cond, condition_rls=rls_cond)
            loss = F.mse_loss(pred, noise)
            total_loss += loss.item()
            total_batches += 1
    
    if was_training:
        model.train()
    
    return total_loss / max(1, total_batches)


def train_dolce(cfg: Dict, *, config_path: Path | str | None = None) -> None:
    """Main training function for DOLCE."""
    print("="*80)
    print("DOLCE Training - Data-consistent Optimization for Limited-angle CT Enhancement")
    print("="*80)
    
    # Parse config
    project_cfg = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    diff_cfg = cfg.get("diffusion", {})
    train_cfg = cfg.get("training", {})
    
    # Set seed
    seed = int(project_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device
    device_str = project_cfg.get("device", "cuda")
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
    # Modality
    modality_str = str(data_cfg.get("modality", "CT")).upper()
    if modality_str not in {"CT", "DX", "CR"}:
        raise ValueError(f"Unsupported modality: {modality_str}")
    modality = cast(Literal["CT", "DX", "CR"], modality_str)
    
    # Patient splits
    patient_splits = resolve_patient_splits(data_cfg)
    train_ids = patient_splits["train"]
    val_ids = patient_splits["val"]
    test_ids = patient_splits["test"]
    
    print(f"\nPatient splits:")
    print(f"  Train: {train_ids}")
    print(f"  Val: {val_ids}")
    print(f"  Test: {test_ids}")
    
    if not train_ids:
        raise ValueError("Training split is empty; please configure train patient ids or split counts")
    
    # Load volumes
    print("\nLoading ground truth volumes...")
    
    # Check if ground truth should be loaded from .npy files or DICOM
    gt_path = Path(data_cfg.get("gt_volume_path", "")) if data_cfg.get("gt_volume_path") else None
    gt_total_angle = data_cfg.get("gt_total_angle", 360)  # Default to 360° for ground truth
    gt_angle_step = data_cfg.get("gt_angle_step", 0.25)   # Default to 0.25° for fine sampling
    
    train_volumes, spacing_dzyx = load_volumes(
        train_ids, modality, 
        volume_path=gt_path, 
        total_angle=gt_total_angle, 
        angle_step=gt_angle_step
    )
    val_volumes, _ = load_volumes(
        val_ids, modality,
        volume_path=gt_path,
        total_angle=gt_total_angle,
        angle_step=gt_angle_step
    )
    test_volumes, _ = load_volumes(
        test_ids, modality,
        volume_path=gt_path,
        total_angle=gt_total_angle,
        angle_step=gt_angle_step
    )
    
    # Load conditional volumes
    use_fbp = model_cfg.get("use_fbp_condition", True)
    use_rls = model_cfg.get("use_rls_condition", False)
    
    fbp_path = Path(data_cfg.get("fbp_condition_path", "")) if data_cfg.get("fbp_condition_path") else None
    rls_path = Path(data_cfg.get("rls_condition_path", "")) if data_cfg.get("rls_condition_path") else None
    
    # Get angle parameters for flexible file matching
    condition_total_angle = data_cfg.get("condition_total_angle")  # e.g., 60, 90, 120
    condition_angle_step = data_cfg.get("condition_angle_step")    # e.g., 1.0, 2.5, 10.0
    
    train_fbp = load_conditional_volumes(
        train_ids, "FBP", fbp_path, 
        total_angle=condition_total_angle, 
        angle_step=condition_angle_step
    ) if use_fbp else None
    val_fbp = load_conditional_volumes(
        val_ids, "FBP", fbp_path,
        total_angle=condition_total_angle,
        angle_step=condition_angle_step
    ) if use_fbp else None
    
    train_rls = load_conditional_volumes(
        train_ids, "RLS", rls_path,
        total_angle=condition_total_angle,
        angle_step=condition_angle_step
    ) if use_rls else None
    val_rls = load_conditional_volumes(
        val_ids, "RLS", rls_path,
        total_angle=condition_total_angle,
        angle_step=condition_angle_step
    ) if use_rls else None
    
    # Dataset parameters
    image_size = int(data_cfg.get("image_size", 512))
    channels = int(data_cfg.get("channels", 1))
    use_mu = bool(data_cfg.get("use_mu", True))
    mu_water = float(data_cfg.get("mu_water", 0.02))
    hu_clip_range = data_cfg.get("hu_clip_range")
    centered = bool(data_cfg.get("centered", False))
    
    batch_size = int(data_cfg.get("batch_size", 4))  # Lower default for DOLCE (larger model)
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    
    clip_tuple = None
    if hu_clip_range is not None and len(hu_clip_range) >= 2:
        clip_tuple = (float(hu_clip_range[0]), float(hu_clip_range[1]))
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ConditionalSliceDataset(
        train_volumes,
        train_fbp,
        train_rls,
        image_size=image_size,
        use_mu=use_mu,
        mu_water=mu_water,
        hu_clip_range=clip_tuple,
        centered=centered,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(f"Train dataset: {len(train_dataset)} slices")
    
    val_dataset: Optional[ConditionalSliceDataset] = None
    val_loader: Optional[DataLoader] = None
    if val_volumes:
        val_dataset = ConditionalSliceDataset(
            val_volumes,
            val_fbp,
            val_rls,
            image_size=image_size,
            use_mu=use_mu,
            mu_water=mu_water,
            hu_clip_range=clip_tuple,
            centered=centered,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print(f"Val dataset: {len(val_dataset)} slices")
    
    # Create model (using original DOLCE UNet)
    print("\nCreating DOLCE model...")
    
    # Original DOLCE uses channel_mult = (0.5, 1, 1, 2, 2, 4, 4) for 512x512
    model = ConditionalModel(
        image_size=image_size,
        in_channels=channels,
        model_channels=model_cfg.get("model_channels", 128),
        out_channels=channels,  # Match input channels (1 for grayscale CT)
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [16, 8])),
        dropout=model_cfg.get("dropout", 0.0),
        channel_mult=(0.5, 1, 1, 2, 2, 4, 4),  # Original DOLCE for 512x512
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        num_add_res=2,
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create diffusion
    diffusion = create_gaussian_diffusion(
        model,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        objective=diff_cfg.get("objective", "pred_noise"),
    )
    diffusion = diffusion.to(device)
    
    # Optimizer
    lr = float(train_cfg.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    
    # Training parameters
    epochs = int(train_cfg.get("epochs", 100))
    save_interval = int(train_cfg.get("save_interval", 10))
    eval_interval = int(train_cfg.get("eval_interval", 5))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    
    # Check for checkpoint to resume from
    start_epoch = 0
    resume_path = train_cfg.get("resume_checkpoint")
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            print(f"\n{'='*60}")
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resuming from epoch {start_epoch}")
            print(f"Previous loss: {checkpoint.get('loss', 'N/A')}")
            print(f"{'='*60}\n")
        else:
            print(f"Warning: Resume checkpoint not found: {resume_path}")
            print(f"Starting training from scratch...")
    
    # Output directory
    output_dir = Path(project_cfg.get("output_dir", "outputs/dolce/chest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create or append to metrics log file
    metrics_log_path = logs_dir / "validation_metrics.txt"
    if start_epoch > 0 and metrics_log_path.exists():
        # Append mode if resuming
        metrics_log = open(metrics_log_path, 'a')
        print(f"Appending metrics to existing log: {metrics_log_path}")
    else:
        # Write mode for new training
        metrics_log = open(metrics_log_path, 'w')
        metrics_log.write("Epoch,Val_Loss,SSIM_Gen,SSIM_FBP,PSNR_Gen,PSNR_FBP\n")
        print(f"Metrics will be logged to: {metrics_log_path}")
    
    print(f"\nTraining for {epochs} epochs")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Samples directory: {samples_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    
    if start_epoch > 0:
        print(f"\n*** Resuming training from epoch {start_epoch+1} to {epochs} ***\n")
    
    # Training loop
    global_step = start_epoch * len(train_loader)  # Restore global step count
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x0 = batch["image"].to(device)
            B = x0.size(0)
            
            # Sample timestep
            t = torch.randint(0, diffusion.timesteps, (B,), device=device)
            
            # Sample noise
            noise = torch.randn_like(x0)
            
            # Forward diffusion
            xt = diffusion.q_sample(x0, t, noise)
            
            # Get conditions
            fbp_cond = batch.get("condition_fbp").to(device) if use_fbp and "condition_fbp" in batch else None
            rls_cond = batch.get("condition_rls").to(device) if use_rls and "condition_rls" in batch else None
            
            # Predict noise
            pred_noise = model(xt, t, condition_fbp=fbp_cond, condition_rls=rls_cond)
            
            # Compute loss
            loss = F.mse_loss(pred_noise, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader is not None and (epoch + 1) % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Starting validation at epoch {epoch+1}")
            val_loss = evaluate_conditional_mse(model, diffusion, val_loader, device, use_fbp, use_rls)
            if val_loss is not None:
                print(f"Validation Loss: {val_loss:.4f}")
            
            # Generate and save validation images
            print(f"Generating validation images...")
            model.eval()
            with torch.no_grad():
                # Get a batch from validation set
                val_batch = next(iter(val_loader))
                gt_images = val_batch['image'].to(device)[:4]  # Take first 4 samples
                
                fbp_images = None
                if use_fbp and 'condition_fbp' in val_batch:
                    fbp_images = val_batch['condition_fbp'].to(device)[:4]
                
                rls_images = None
                if use_rls and 'condition_rls' in val_batch:
                    rls_images = val_batch['condition_rls'].to(device)[:4]
                
                # Sample from the model
                print(f"Sampling from diffusion model...")
                samples, _ = diffusion.sample_loop(  # Returns (x, x0_preds), we only need x
                    shape=gt_images.shape,
                    condition_fbp=fbp_images,
                    condition_rls=rls_images,
                    sampler='ddim',
                    ddim_steps=50,  # Faster sampling for validation
                    verbose=False,
                )
                
                # Compute metrics: Generated vs GT
                print(f"\nComputing metrics (Generated vs GT)...")
                metrics_gen = compute_metrics(samples, gt_images)
                print(f"  SSIM: {metrics_gen['ssim_mean']:.4f} ± {metrics_gen['ssim_std']:.4f}")
                print(f"  PSNR: {metrics_gen['psnr_mean']:.2f} ± {metrics_gen['psnr_std']:.2f} dB")
                
                # Also compute FBP vs GT for comparison
                if fbp_images is not None:
                    print(f"\nComputing metrics (FBP vs GT) for comparison...")
                    metrics_fbp = compute_metrics(fbp_images, gt_images)
                    print(f"  SSIM: {metrics_fbp['ssim_mean']:.4f} ± {metrics_fbp['ssim_std']:.4f}")
                    print(f"  PSNR: {metrics_fbp['psnr_mean']:.2f} ± {metrics_fbp['psnr_std']:.2f} dB")
                    
                    # Show improvement
                    ssim_improve = (metrics_gen['ssim_mean'] - metrics_fbp['ssim_mean']) / metrics_fbp['ssim_mean'] * 100
                    psnr_improve = metrics_gen['psnr_mean'] - metrics_fbp['psnr_mean']
                    print(f"\n  Improvement: SSIM +{ssim_improve:.1f}%, PSNR +{psnr_improve:.2f} dB")
                    
                    # Log metrics to file
                    metrics_log.write(f"{epoch+1},{val_loss:.6f},{metrics_gen['ssim_mean']:.6f},"
                                    f"{metrics_fbp['ssim_mean']:.6f},{metrics_gen['psnr_mean']:.4f},"
                                    f"{metrics_fbp['psnr_mean']:.4f}\n")
                    metrics_log.flush()
                else:
                    # Log without FBP comparison
                    metrics_log.write(f"{epoch+1},{val_loss:.6f},{metrics_gen['ssim_mean']:.6f},"
                                    f"N/A,{metrics_gen['psnr_mean']:.4f},N/A\n")
                    metrics_log.flush()
                
                # Prepare visualization: Each row is one sample, columns are GT/FBP/Generated
                # Layout: 
                # Row 1: [GT_1] [FBP_1] [Generated_1]  ← Sample 1
                # Row 2: [GT_2] [FBP_2] [Generated_2]  ← Sample 2
                # Row 3: [GT_3] [FBP_3] [Generated_3]  ← Sample 3
                # Row 4: [GT_4] [FBP_4] [Generated_4]  ← Sample 4
                
                num_samples = gt_images.shape[0]  # Should be 4
                image_types = [gt_images]  # Start with GT column
                col_labels = ["GT"]
                
                if fbp_images is not None:
                    image_types.append(fbp_images)
                    col_labels.append("Limited Angle")
                
                if rls_images is not None:
                    image_types.append(rls_images)
                    col_labels.append("RLS")
                
                image_types.append(samples)  # End with Generated column
                col_labels.append("Generated")
                
                # Interleave images in row-major order for save_png_grid
                # save_png_grid with nrow=num_samples will create:
                # [img0, img1, img2, ...]  with nrow=4 becomes:
                # Row 0: img0, img1, img2
                # Row 1: img3, img4, img5
                # etc.
                # So we need to order as: [GT_1, FBP_1, Gen_1, GT_2, FBP_2, Gen_2, ...]
                all_images_list = []
                for i in range(num_samples):
                    for img_type in image_types:
                        all_images_list.append(img_type[i:i+1])
                
                all_images = torch.cat(all_images_list, dim=0)
                
                # Normalize to [0, 1] for visualization
                all_images = (all_images - all_images.min()) / (all_images.max() - all_images.min() + 1e-8)
                
                # Save visualization
                save_path = samples_dir / f"epoch_{epoch+1:04d}.png"
                print(f"Saving validation images to: {save_path}")
                print(f"  Grid layout: {num_samples} rows x {len(col_labels)} columns")
                print(f"  Column order: {' | '.join(col_labels)}")
                save_png_grid(
                    x=all_images,
                    path=save_path,
                    nrow=num_samples,  # Number of rows (samples)
                    title=f"Epoch {epoch+1} - Validation Results",
                    subtitle=f"Columns (left to right): {' | '.join(col_labels)}"
                )
                print(f"✓ Validation images saved successfully!")
            
            model.train()
            print(f"{'='*60}\n")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1:04d}.pt"
            print(f"\nSaving checkpoint to: {checkpoint_path}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': cfg,
            }, checkpoint_path)
            print(f"✓ Checkpoint saved successfully: {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / "model_final.pt"
    print(f"\n{'='*60}")
    print(f"Training completed! Saving final model to: {final_path}")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': cfg,
    }, final_path)
    print(f"✓ Final model saved successfully: {final_path}")
    print(f"{'='*60}\n")
    
    # Close metrics log
    metrics_log.close()
    print(f"Metrics log saved to: {metrics_log_path}")
    print(f"\nTraining complete! Final model saved to: {final_path}")
