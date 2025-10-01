#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm


# --- make repo importable ---
def _add_repo_root_to_syspath():
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

_add_repo_root_to_syspath()

from src.model.diffusion import SimpleUNet, Diffusion, DDIM


def load_volume(volume_path: Path) -> np.ndarray:
    """Load a 3D volume from either .npz or .npy file."""
    suffix = volume_path.suffix.lower()
    if suffix == '.npz':
        with np.load(volume_path) as data:
            key = next((k for k in data.files), None)
            if key is None:
                raise ValueError(f"No arrays found in {volume_path}")
            arr = data[key]
    elif suffix == '.npy':
        arr = np.load(volume_path)
    else:
        raise ValueError(f"Unsupported volume file extension '{suffix}' for {volume_path}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (Z,H,W); got shape {arr.shape} in {volume_path}")
    return arr.astype(np.float32)


@torch.no_grad()
def seddit_restore_slice(
    model: torch.nn.Module,
    diff: Diffusion,
    ddim: DDIM,
    img_hw: np.ndarray,
    *,
    device: torch.device,
    image_size: int = 512,
    t0: int = 600,
    num_back_steps: int = 100,
    eta: float = 0.0,
) -> torch.Tensor:
    """SDEdit-style: add noise to input up to t0, then DDIM reverse to x0.

    img_hw: [H,W] in [0,1] range (float32)
    returns: [H,W] in [0,1]
    """
    model.eval()

    H, W = img_hw.shape
    # to model size [1,1,h,w]
    x0 = torch.from_numpy(img_hw).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    if (H != image_size) or (W != image_size):
        x0 = torch.nn.functional.interpolate(x0, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x0 = x0.to(device)

    # forward to t0 (add noise)
    t_vec = torch.full((1,), min(t0, diff.T-1), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = diff.q_sample(x0, t_vec, noise)

    # reverse with DDIM from t0 -> 0
    ts = ddim.set_timesteps(num_back_steps)
    # find start index whose t <= t0
    start_idx = 0
    for i, tval in enumerate(ts):
        if tval <= t_vec.item():
            start_idx = i
            break
    x_cur = xt
    for i in range(start_idx, len(ts)):
        t = ts[i]
        t_prev = ts[i+1] if i+1 < len(ts) else -1
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        eps = model(x_cur, t_batch)
        x_cur, _ = ddim.step_from_to(eps, t, t_prev, x_cur, eta=eta)
    x_rec = x_cur.clamp(0, 1)

    # back to original H,W
    if (H != image_size) or (W != image_size):
        x_rec = torch.nn.functional.interpolate(x_rec, size=(H, W), mode='bilinear', align_corners=False)

    return x_rec.squeeze(0).squeeze(0).detach().cpu()


def save_png_(img_hw: torch.Tensor, path: Path):
    """Save single-channel [H,W] float32 in [0,1] to PNG."""
    try:
        import imageio.v2 as imageio
        arr = (img_hw.clamp(0,1).numpy() * 255.0).round().astype(np.uint8)
        imageio.imwrite(path, arr)
    except Exception:
        # fallback to matplotlib
        import matplotlib.pyplot as plt
        plt.imsave(path, img_hw.clamp(0,1).numpy(), cmap='gray', vmin=0.0, vmax=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='outputs/ddpm/chest/final_weights.pth')
    ap.add_argument('--patient_id', type=str, required=True, help='Patient ID to match files recon_<id>_*.(npz|npy)')
    ap.add_argument('--recon_root', type=str, default='data/interim/recon/chest', help='Folder containing recon volume files')
    ap.add_argument('--out_root', type=str, default='outputs/ddpm/chest/restore')
    ap.add_argument('--image_size', type=int, default=512)
    ap.add_argument('--ddim_steps', type=int, default=100)
    ap.add_argument('--t0', type=int, default=500)
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--max_slices', type=int, default=None, help='Optional limit on number of slices processed per volume')
    args = ap.parse_args()

    requested_device = args.device.lower()
    if requested_device.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            print("[INFO] CUDA requested but not available; falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # collect recon volume files for this patient
    root = Path(args.recon_root)
    patterns = [f"recon_{args.patient_id}_*.npz", f"recon_{args.patient_id}_*.npy"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    files = sorted(files)
    if not files:
        print(f"No recon files found for patient_id={args.patient_id} under {root} (looked for *.npz and *.npy)")
        return

    # load model
    model = SimpleUNet(in_ch=1).to(device)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()

    # schedulers
    T = 1000
    diff = Diffusion(T=T).to(device)
    ddim = DDIM(T=T, eta=args.eta).to(device)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for f in files:
        try:
            vol = load_volume(f)  # [Z,H,W]
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")
            continue

        Z, H, W = vol.shape
        out_dir = out_root / f.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        pad = max(4, len(str(max(0, Z-1))))  # dynamic zero padding, at least 4
        slice_iter = range(Z) if args.max_slices is None else range(min(Z, args.max_slices))
        pbar = tqdm(slice_iter, desc=f"{f.name} slices")
        for i in pbar:
            sl = vol[i]
            # per-slice min-max to [0,1] to match training
            mn = float(sl.min())
            mx = float(sl.max())
            if mx > mn:
                sl01 = (sl - mn) / (mx - mn)
            else:
                sl01 = np.zeros_like(sl, dtype=np.float32)

            x_rec = seddit_restore_slice(
                model, diff, ddim, sl01,
                device=device, image_size=args.image_size,
                t0=args.t0, num_back_steps=args.ddim_steps, eta=args.eta,
            )  # [H,W]

            save_path = out_dir / f"{i:0{pad}d}.png"
            save_png_(x_rec, save_path)

        print(f"Saved restored PNGs to: {out_dir}")


if __name__ == '__main__':
    main()
