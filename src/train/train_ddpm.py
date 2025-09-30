from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.diffusion import SimpleUNet, Diffusion, DDIM
from src.data.data_load import data_load_chest


class SliceDataset(Dataset):
    """将多个 HU 体切成单张图，完成裁剪、归一化和尺寸调整。"""

    def __init__(
        self,
        volumes_hu: Sequence[np.ndarray],
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

        self.volumes: List[np.ndarray] = []
        self.index: List[Tuple[int, int]] = []

        lo_hi = None
        if hu_clip_range is not None and len(hu_clip_range) >= 2:
            lo_hi = (float(hu_clip_range[0]), float(hu_clip_range[1]))

        for vidx, vol in enumerate(volumes_hu):
            if vol.ndim != 3:
                raise ValueError(f"Expected volume shape [Z,H,W], got {vol.shape}")
            arr = vol.astype(np.float32, copy=False)
            if lo_hi is not None:
                np.clip(arr, lo_hi[0], lo_hi[1], out=arr)
            self.volumes.append(arr)
            self.index.extend((vidx, s) for s in range(arr.shape[0]))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        vol_idx, slice_idx = self.index[idx]
        sl_hu = self.volumes[vol_idx][slice_idx]

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

        x = torch.from_numpy(sl_norm).float().unsqueeze(0).unsqueeze(0)
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)

        return {
            "image": x,
            "slice_idx": slice_idx,
            "volume_idx": vol_idx,
        }


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


def save_triplet_grid(
    gt: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    path: Path,
    *,
    col_titles: tuple[str, str, str] = ("GT", "Noisy", "Denoised"),
    title: str | None = None,
    subtitle: str | None = None,
):
    """将 N 张样本的 GT/Noisy/Denoised 拼成一个 N×3 的网格，并在每列顶部加标题。"""
    try:
        import matplotlib.pyplot as plt
        import numpy as _np
    except Exception:
        np.save(path.with_name(path.stem + '_gt.npy'), gt.cpu().numpy())
        np.save(path.with_name(path.stem + '_noisy.npy'), noisy.cpu().numpy())
        np.save(path.with_name(path.stem + '_denoised.npy'), denoised.cpu().numpy())
        return

    def to_hw(x: torch.Tensor):
        x = x.clamp(0, 1).detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] == 1:
            return x[:, 0]
        elif x.ndim == 4:
            return _np.transpose(x, (0, 2, 3, 1))
        elif x.ndim == 3:
            return x
        else:
            raise ValueError("Unexpected tensor shape for image grid")

    g = to_hw(gt)
    n = to_hw(noisy)
    d = to_hw(denoised)
    N = g.shape[0]

    fig, axes = plt.subplots(N, 3, figsize=(3 * 3, N * 3))
    if N == 1:
        axes = _np.expand_dims(axes, axis=0)

    for j, col_title in enumerate(col_titles):
        axes[0, j].set_title(col_title)

    for i in range(N):
        imgs = [g[i], n[i], d[i]]
        for j in range(3):
            ax = axes[i, j]
            ax.axis('off')
            img = imgs[j]
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)

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


def resolve_patient_splits(data_cfg: Dict) -> Dict[str, List[str]]:
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

    for phase in phases:
        explicit = data_cfg.get(f"{phase}_patient_ids")
        _extend(phase, _normalize_list(explicit))

    base_ids = _normalize_list(data_cfg.get("patient_ids"))
    if not base_ids and not splits["train"]:
        base_ids = _normalize_list(data_cfg.get("train_patient_ids"))
        _extend("train", base_ids)
        return splits

    remaining = [pid for pid in base_ids if pid not in assigned]

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
) -> Tuple[List[np.ndarray], Optional[Tuple[float, float, float]]]:
    if not patient_ids:
        return [], None

    volumes: List[np.ndarray] = []
    first_spacing: Optional[Tuple[float, float, float]] = None
    for pid in patient_ids:
        vol, spacing, _ = data_load_chest.load_data_chest(pid, modality)
        volumes.append(vol)
        if first_spacing is None:
            first_spacing = spacing
        print(f"Loaded patient {pid}: shape={vol.shape}, spacing(dzyx)={spacing}")
    return volumes, first_spacing


def evaluate_noise_mse(
    model: torch.nn.Module,
    diff: Diffusion,
    dataloader: Optional[DataLoader],
    device: torch.device,
) -> Optional[float]:
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
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(x0)
            xt = diff.q_sample(x0, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)
            total_loss += loss.item()
            total_batches += 1

    if was_training:
        model.train()

    return total_loss / max(1, total_batches)


def train_ddpm(cfg: Dict, *, config_path: Path | str | None = None) -> None:
    project_cfg = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    diff_cfg = cfg.get("diffusion", {})
    train_cfg = cfg.get("training", {})

    seed = int(project_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = project_cfg.get("device", "cuda")
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")

    modality_str = str(data_cfg.get("modality", "CT")).upper()
    if modality_str not in {"CT", "DX", "CR"}:
        raise ValueError(f"Unsupported modality: {modality_str}")
    modality = cast(Literal["CT", "DX", "CR"], modality_str)

    patient_splits = resolve_patient_splits(data_cfg)
    train_ids = patient_splits["train"]
    val_ids = patient_splits["val"]
    test_ids = patient_splits["test"]

    if not train_ids:
        raise ValueError("Training split is empty; please configure train patient ids or split counts")

    train_volumes, spacing_dzyx = load_volumes(train_ids, modality)
    val_volumes, _ = load_volumes(val_ids, modality)
    test_volumes, _ = load_volumes(test_ids, modality)

    image_size = int(data_cfg.get("image_size", 512))
    channels = int(data_cfg.get("channels", 1))
    use_mu = bool(data_cfg.get("use_mu", True))
    mu_water = float(data_cfg.get("mu_water", 0.02))
    hu_clip_range = data_cfg.get("hu_clip_range")
    centered = bool(data_cfg.get("centered", False))

    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    clip_tuple = None
    if hu_clip_range is not None and len(hu_clip_range) >= 2:
        clip_tuple = (float(hu_clip_range[0]), float(hu_clip_range[1]))

    train_dataset = SliceDataset(
        train_volumes,
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

    val_dataset: Optional[SliceDataset] = None
    val_loader: Optional[DataLoader] = None
    if val_volumes:
        val_dataset = SliceDataset(
            val_volumes,
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
            num_workers=max(0, num_workers // 2),
            pin_memory=pin_memory,
        )

    test_dataset: Optional[SliceDataset] = None
    test_loader: Optional[DataLoader] = None
    if test_volumes:
        test_dataset = SliceDataset(
            test_volumes,
            image_size=image_size,
            use_mu=use_mu,
            mu_water=mu_water,
            hu_clip_range=clip_tuple,
            centered=centered,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=pin_memory,
        )

    steps = int(diff_cfg.get("steps", 1000))
    sample_steps = int(diff_cfg.get("sample_steps", 100))
    eta = float(diff_cfg.get("eta", 0.0))
    ddim_cfg = diff_cfg.get("ddim", {})
    preview_t0 = int(ddim_cfg.get("preview_t0", min(steps - 1, 600))) if steps > 0 else 0
    preview_t0 = max(0, min(max(0, steps - 1), preview_t0))
    preview_steps = int(ddim_cfg.get("preview_steps", min(sample_steps, 50))) if sample_steps > 0 else 1
    preview_steps = max(1, preview_steps)
    sample_batch = int(ddim_cfg.get("sample_batch", 4))
    sample_batch = max(1, sample_batch)

    model = SimpleUNet(in_ch=channels).to(device)
    diff = Diffusion(T=steps).to(device)
    ddim = DDIM(T=steps, eta=eta).to(device)

    lr = float(train_cfg.get("lr", 2e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    out_dir = Path(train_cfg.get("output_dir", "outputs/ddpm/chest"))
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(train_cfg.get("epochs", 10))
    save_interval = max(1, int(train_cfg.get("save_interval", 10)))

    print(
        f"Device: {device}, train slices: {len(train_dataset)},"
        f" patients(train)={train_ids}, spacing(dzyx)={spacing_dzyx}"
    )
    if val_dataset is not None:
        print(f"Validation patients: {val_ids}, total slices={len(val_dataset)}")
    if test_dataset is not None:
        print(f"Test patients: {test_ids}, total slices={len(test_dataset)}")

    global_step = 0
    best_val = float("inf")
    best_epoch = 0

    config_path_value: Optional[str]
    if config_path is None:
        config_path_value = None
    elif isinstance(config_path, Path):
        config_path_value = str(config_path.resolve())
    else:
        config_path_value = str(config_path)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            x0 = batch["image"].to(device)
            B = x0.size(0)
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(x0)
            xt = diff.q_sample(x0, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if val_loader is not None:
            val_loss = evaluate_noise_mse(model, diff, val_loader, device)
            if val_loss is None:
                print(f"[Val] epoch {epoch}: skipped (dataset empty)")
            else:
                print(f"[Val] epoch {epoch} noise MSE: {val_loss:.6f}")
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                payload = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": best_val,
                }
                if config_path_value is not None:
                    payload["config_path"] = config_path_value
                torch.save(payload, out_dir / "best_val.pth")

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        if config_path_value is not None:
            ckpt["config_path"] = config_path_value

        torch.save(ckpt, out_dir / "last.pth")
        if epoch % save_interval == 0:
            torch.save(ckpt, out_dir / f"checkpoint_epoch_{epoch}.pth")

            model.eval()
            with torch.no_grad():
                sample_shape = (sample_batch, channels, image_size, image_size)
                samples = ddim.sample(
                    model,
                    sample_shape,
                    device,
                    num_steps=sample_steps,
                    eta=eta,
                    show_progress=True,
                )
            samples_disp = (samples + 1) / 2 if centered else samples
            samples_disp = samples_disp.clamp(0, 1)
            norm_descr = "[-1, 1]" if centered else "[0, 1]"
            sample_title = f"Epoch {epoch} DDIM samples"
            sample_subtitle = (
                f"steps={sample_steps}, eta={eta}, batch={sample_batch}, scale={norm_descr}"
            )
            sample_nrow = max(1, math.ceil(sample_batch / 2))
            save_png_grid(
                samples_disp,
                out_dir / f"samples_epoch_{epoch}.png",
                nrow=sample_nrow,
                title=sample_title,
                subtitle=sample_subtitle,
            )

            try:
                model.eval()
                with torch.no_grad():
                    x0_list = [cast(torch.Tensor, train_dataset[i]["image"]) for i in range(min(2, len(train_dataset)))]
                    if x0_list:
                        x0 = torch.stack(x0_list, dim=0).to(device)
                        t = torch.full((x0.size(0),), preview_t0, device=device, dtype=torch.long)
                        noise = torch.randn_like(x0)
                        xt = diff.q_sample(x0, t, noise)

                        num_back_steps = preview_steps
                        ts = ddim.set_timesteps(num_back_steps)
                        start_idx = 0
                        for i, tval in enumerate(ts):
                            if tval <= preview_t0:
                                start_idx = i
                                break
                        x_cur = xt
                        for i in range(start_idx, len(ts)):
                            t_curr = ts[i]
                            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
                            tvec = torch.full((x0.size(0),), t_curr, device=device, dtype=torch.long)
                            eps = model(x_cur, tvec)
                            x_cur, _ = ddim.step_from_to(eps, t_curr, t_prev, x_cur, eta=eta)

                        if centered:
                            x_rec = x_cur.clamp(-1, 1)
                            x0_disp = ((x0 + 1) / 2).clamp(0, 1)
                            xt_disp = ((xt + 1) / 2).clamp(0, 1)
                            x_rec_disp = ((x_rec + 1) / 2).clamp(0, 1)
                        else:
                            x_rec = x_cur.clamp(0, 1)
                            x0_disp = x0.clamp(0, 1)
                            xt_disp = xt.clamp(0, 1)
                            x_rec_disp = x_rec

                        save_triplet_grid(
                            x0_disp,
                            xt_disp,
                            x_rec_disp,
                            out_dir / f"epoch_{epoch}_triplet.png",
                            col_titles=("Ground Truth", f"Noisy t={preview_t0}", "Denoised"),
                            title=f"Epoch {epoch} denoise preview",
                            subtitle=f"t0={preview_t0}, steps={preview_steps}, eta={eta}, scale={norm_descr}",
                        )
            except Exception as e:
                print(f"[Warn] denoise preview failed: {e}")

    if val_loader is not None and best_val < float("inf"):
        print(f"Best val noise MSE: {best_val:.6f} (epoch {best_epoch})")

    if test_loader is not None:
        test_loss = evaluate_noise_mse(model, diff, test_loader, device)
        if test_loss is None:
            print("[Test] skipped: dataset empty")
        else:
            print(f"[Test] noise MSE (last epoch weights): {test_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "final_weights.pth")
    print(f"Training done. Checkpoints and samples saved under: {out_dir}")


__all__ = ["train_ddpm", "SliceDataset"]
