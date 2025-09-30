#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from src.model.diffusion import SimpleUNet, Diffusion, DDIM


def save_png_grid(x: torch.Tensor, path: Path, nrow: int = 4):
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*3))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)
    for i in range(nrow):
        for j in range(ncol):
            idx = i*ncol + j
            axes[i, j].axis('off')
            if idx < N:
                img = x[idx]
                if img.ndim == 2:
                    axes[i, j].imshow(img, cmap='gray')
                else:
                    axes[i, j].imshow(np.transpose(img, (1,2,0)))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--channels', type=int, default=1)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--out', type=str, default='outputs/ddpm/chest/samples.png')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--sampler', type=str, choices=['ddpm','ddim'], default='ddim', help='Sampling method')
    ap.add_argument('--ddim_steps', type=int, default=50, help='DDIM steps when sampler=ddim')
    ap.add_argument('--eta', type=float, default=0.0, help='DDIM eta (0 for deterministic)')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    model = SimpleUNet(in_ch=args.channels).to(device)
    state = torch.load(args.weights, map_location=device)
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()

    shape = (args.n, args.channels, args.image_size, args.image_size)
    if args.sampler == 'ddim':
        sampler = DDIM(T=args.steps, eta=args.eta).to(device)
        x = sampler.sample(model, shape, device, num_steps=args.ddim_steps, eta=args.eta, show_progress=True)
    else:
        sampler = Diffusion(T=args.steps).to(device)
        x = sampler.sample(model, shape, device, num_steps=args.steps, show_progress=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_png_grid(x, out_path, nrow=4)
    print(f"Saved samples to {out_path}")


if __name__ == '__main__':
    main()
