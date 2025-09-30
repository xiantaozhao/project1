#!/usr/bin/env python
"""Inspect corner voxel values for chest CT volumes.

Loads a chest volume using the existing data loader and prints the HU (and
optional normalized) values at the four image corners for several slices so you
can verify whether the padded regions stay at air-equivalent intensities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Literal, Tuple, cast

import numpy as np

# Ensure project root is on sys.path so `src` imports work even when executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_load import data_load_chest

DEFAULT_WINDOW = (-1024.0, 400.0)


def _normalize(values: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    """Map HU values into [0, 1] using a linear window."""
    lo, hi = window
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _build_circular_mask(height: int, width: int, radius_factor: float = 1.0) -> np.ndarray:
    """Return a boolean mask covering the inscribed circle of the image."""
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = min(height, width) * 0.5 * radius_factor
    yy, xx = np.ogrid[:height, :width]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask


def inspect_corners(
    case_id: str,
    modality: Literal["CT", "DX", "CR"],
    window: Tuple[float, float],
    slices: Iterable[int] | None = None,
) -> None:
    vol_hu_zyx, spacing_dzyx, meta = data_load_chest.load_data_chest(case_id, modality)

    z, h, w = vol_hu_zyx.shape
    print(f"Loaded volume shape: (slices={z}, height={h}, width={w})")
    print(f"Spacing (dz, dy, dx): {spacing_dzyx}")
    print(f"Meta: {meta}\n")

    if slices is None:
        slices = (0, z // 2, z - 1)

    corner_coords = (
        ("top_left", 0, 0),
        ("top_right", 0, w - 1),
        ("bottom_left", h - 1, 0),
        ("bottom_right", h - 1, w - 1),
    )

    for slice_idx in slices:
        if not (0 <= slice_idx < z):
            print(f"Skip slice {slice_idx}: out of range")
            continue
        slice_hu = vol_hu_zyx[slice_idx]
        print(f"Slice {slice_idx} corner values (HU):")
        values = []
        for name, y, x in corner_coords:
            val = float(slice_hu[y, x])
            values.append(val)
            print(f"  {name:>12s}: {val:8.2f} HU")
        values_np = np.array(values, dtype=np.float32)
        normalized = _normalize(values_np, window)
        print("  Normalized [0,1]:", " ".join(f"{v:0.4f}" for v in normalized))
        print()

    # Aggregate statistics across all slices for each corner position.
    print("Aggregate across all slices:")
    for name, y, x in corner_coords:
        corner_series = vol_hu_zyx[:, y, x].astype(np.float32)
        norm_series = _normalize(corner_series, window)
        print(
            f"  {name:>12s}: HU min={corner_series.min():8.2f}, max={corner_series.max():8.2f}, "
            f"mean={corner_series.mean():8.2f} | normalized mean={norm_series.mean():0.4f}"
        )

    # Effective circular field-of-view statistics.
    effective_mask = _build_circular_mask(h, w)
    effective_voxels = vol_hu_zyx[:, effective_mask].astype(np.float32)
    effective_norm = _normalize(effective_voxels, window)

    print("\nEffective circular FOV (inscribed circle) statistics across entire volume:")
    percentiles = np.percentile(effective_voxels, [1, 5, 25, 50, 75, 95, 99])
    print(
        "  HU: min={:.2f}, max={:.2f}, mean={:.2f}, std={:.2f}".format(
            effective_voxels.min(), effective_voxels.max(), effective_voxels.mean(), effective_voxels.std()
        )
    )
    print(
        "  HU percentiles (1/5/25/50/75/95/99): "
        + " ".join(f"{p:.1f}" for p in percentiles)
    )
    print(
        "  Normalized mean={:.4f}, std={:.4f}".format(
            effective_norm.mean(), effective_norm.std()
        )
    )

    if slices:
        print("\nPer-slice effective FOV summary for selected slices:")
        for slice_idx in slices:
            if not (0 <= slice_idx < z):
                continue
            slice_vals = vol_hu_zyx[slice_idx, effective_mask].astype(np.float32)
            if slice_vals.size == 0:
                continue
            p1, p50, p99 = np.percentile(slice_vals, [1, 50, 99])
            print(
                f"  Slice {slice_idx:4d}: HU mean={slice_vals.mean():8.2f}, std={slice_vals.std():7.2f}, "
                f"p1={p1:7.2f}, median={p50:7.2f}, p99={p99:7.2f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect corner HU values in chest CT volumes.")
    parser.add_argument("--case-id", default="1", help="Case identifier to load")
    parser.add_argument(
        "--modality",
        default="CT",
        choices=["CT", "DX", "CR"],
        help="Modality string passed to the loader"
    )
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=DEFAULT_WINDOW,
        help="Linear window (HU) used for optional normalization output",
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="*",
        help="Specific slice indices to inspect; defaults to first, middle, last."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_corners(
        case_id=str(args.case_id),
        modality=cast(Literal["CT", "DX", "CR"], args.modality),
        window=(float(args.window[0]), float(args.window[1])),
        slices=args.slices,
    )


if __name__ == "__main__":
    main()
