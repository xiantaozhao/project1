from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional runtime dependency
    import astra  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at call site
    astra = None  # type: ignore[assignment]

from src.data.data_process.projection import (
    _angles_from_step,
    _build_proj_geom_fanflat,
    _build_vol_geom_2d,
)


@dataclass
class AstraGeometries:
    """Container holding ASTRA geometries for repeated projections."""

    angles_deg: np.ndarray
    proj_geom: Any
    vol_geom: Any
    projector_id: int
    det_count: int
    use_cuda: bool


class AstraFanbeamOperators:
    """Lightweight wrapper around ASTRA fan-beam projector + FBP reconstruction.

    The wrapper is intentionally minimal: it constructs geometries once and
    exposes ``forward`` (Radon) and ``reconstruct`` (filtered backprojection)
    helpers for batches of 2D slices. Geometry parameters are resolved from the
    provided configuration dictionary (compatible with
    ``configs/default/chest.yaml``).
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        image_size: int,
        *,
        spacing_dzyx: Tuple[float, float, float] | None = None,
    ) -> None:
        if astra is None:  # pragma: no cover - runtime import guard
            raise ImportError(
                "ASTRA toolbox is required for limited-angle inference. "
                "Please install 'astra-toolbox' before using this feature."
            )

        if image_size <= 0:
            raise ValueError("image_size must be a positive integer")

        self.cfg = cfg
        self.image_size = int(image_size)
        self._astra = astra  # store concrete module reference (mypy-friendly)
        self._geoms = self._build_geometries(cfg, spacing_dzyx)
        self._filter_type = "ram-lak"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def angles_deg(self) -> np.ndarray:
        return self._geoms.angles_deg

    @property
    def detector_count(self) -> int:
        return self._geoms.det_count

    def forward(self, images_hw: np.ndarray) -> np.ndarray:
        """Compute sinograms for a batch of 2D slices.

        Args:
            images_hw: Float32 array with shape ``[B, H, W]`` or ``[H, W]``.

        Returns:
            np.ndarray: ``[B, A, D]`` sinograms (float32).
        """

        imgs = self._ensure_batch(images_hw)
        batch, H, W = imgs.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"Expected images of size {self.image_size}x{self.image_size}, "
                f"got {H}x{W}"
            )

        A = self.angles_deg.shape[0]
        D = self.detector_count
        out = np.zeros((batch, A, D), dtype=np.float32)

        for idx in range(batch):
            slice_hw = np.ascontiguousarray(imgs[idx], dtype=np.float32)
            vol_id = self._astra.data2d.create("-vol", self._geoms.vol_geom, slice_hw)
            sino_id = None
            try:
                sino_id, sino = self._astra.create_sino(vol_id, self._geoms.projector_id)
                out[idx] = np.asarray(sino, dtype=np.float32)
            finally:
                self._astra.data2d.delete(vol_id)
                if sino_id is not None:
                    self._astra.data2d.delete(sino_id)

        return out

    def reconstruct(self, sino_ad: np.ndarray, *, filter_type: str | None = None) -> np.ndarray:
        """Filtered backprojection for batches of sinograms.

        Args:
            sino_ad: Float32 array with shape ``[B, A, D]`` or ``[A, D]``.
            filter_type: Optional override for ASTRA's ``FilterType`` option.

        Returns:
            np.ndarray: Reconstructed slices with shape ``[B, H, W]``.
        """

        sinos = self._ensure_batch(sino_ad)
        batch, A, D = sinos.shape
        if A != self.angles_deg.shape[0] or D != self.detector_count:
            raise ValueError(
                "Sinogram shape mismatch: expected "
                f"[*, {self.angles_deg.shape[0]}, {self.detector_count}], got {sinos.shape}"
            )

        algo = "FBP_CUDA" if self._geoms.use_cuda else "FBP"
        filt = filter_type or self._filter_type
        H = self.image_size
        W = self.image_size
        out = np.zeros((batch, H, W), dtype=np.float32)

        for idx in range(batch):
            sino_ad = np.ascontiguousarray(sinos[idx], dtype=np.float32)
            sid = self._astra.data2d.create("-sino", self._geoms.proj_geom, sino_ad)
            rid = self._astra.data2d.create("-vol", self._geoms.vol_geom)
            cfg: Dict[str, Any] = self._astra.astra_dict(algo)
            cfg["ProjectionDataId"] = sid
            cfg["ReconstructionDataId"] = rid
            cfg["ProjectorId"] = self._geoms.projector_id
            cfg.setdefault("option", {})["FilterType"] = filt

            alg_id = self._astra.algorithm.create(cfg)
            try:
                self._astra.algorithm.run(alg_id)
                out[idx] = np.asarray(self._astra.data2d.get(rid), dtype=np.float32)
            finally:
                self._astra.algorithm.delete(alg_id)
                self._astra.data2d.delete(sid)
                self._astra.data2d.delete(rid)

        return out

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            if hasattr(self, "_astra") and self._astra is not None and hasattr(self, "_geoms"):
                self._astra.projector.delete(self._geoms.projector_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _build_geometries(
        self,
        cfg: Dict[str, Any],
        spacing_dzyx: Tuple[float, float, float] | None,
    ) -> AstraGeometries:
        projection_cfg = cfg.get("projection", {})
        geom_cfg = projection_cfg.get("geom", {})
        angle_cfg = projection_cfg.get("angles", {})
        data_cfg = cfg.get("data", {})

        angles_deg = self._resolve_angles(angle_cfg)
        dx_mm, dy_mm = self._resolve_inplane_spacing(data_cfg, spacing_dzyx)

        proj_geom, det_count = _build_proj_geom_fanflat(
            geom_cfg,
            angles_deg,
            dx_mm=dx_mm,
            width_px=self.image_size,
        )
        vol_geom = _build_vol_geom_2d(
            self.image_size,
            self.image_size,
            dy_mm=dy_mm,
            dx_mm=dx_mm,
        )

        projector_type = projection_cfg.get("astra", {}).get("projector_2d", "cuda")
        if projector_type not in ("cuda", "line"):
            projector_type = "cuda"
        use_cuda = projector_type == "cuda"
        projector_id = self._astra.create_projector(projector_type, proj_geom, vol_geom)

        return AstraGeometries(
            angles_deg=angles_deg,
            proj_geom=proj_geom,
            vol_geom=vol_geom,
            projector_id=projector_id,
            det_count=det_count,
            use_cuda=use_cuda,
        )

    def _resolve_angles(self, cfg_angles: Dict[str, Any]) -> np.ndarray:
        mode = str(cfg_angles.get("mode", "range_step")).lower()
        if mode != "range_step":
            raise ValueError(f"Only angles.mode='range_step' is supported, got {mode}")
        start = float(cfg_angles.get("start_deg", 0.0))
        stop = float(cfg_angles.get("stop_deg", 180.0))
        step = float(cfg_angles.get("step_deg", 1.0))
        include_endpoint = bool(cfg_angles.get("include_endpoint", False))
        return _angles_from_step(start, stop, step, include_endpoint).astype(np.float64)

    def _resolve_inplane_spacing(
        self,
        data_cfg: Dict[str, Any],
        spacing_dzyx: Tuple[float, float, float] | None,
    ) -> Tuple[float, float]:
        if spacing_dzyx is not None:
            if len(spacing_dzyx) != 3:
                raise ValueError("spacing_dzyx must be a 3-tuple (dz, dy, dx)")
            dz, dy, dx = spacing_dzyx
            return float(dx), float(dy)

        spacing_cfg = data_cfg.get("spacing", {})
        order = str(spacing_cfg.get("order", "dzyx")).lower().replace("d", "z")
        values = spacing_cfg.get("values", None)
        if isinstance(values, Iterable):
            arr = list(values)
        else:
            arr = [1.0, 1.0, 1.0]
        mapping: Dict[str, float] = {}
        for idx, axis in enumerate(order):
            if idx < len(arr):
                mapping[axis] = float(arr[idx])
        dx = mapping.get("x", 1.0)
        dy = mapping.get("y", 1.0)
        return dx, dy

    @staticmethod
    def _ensure_batch(arr: np.ndarray) -> np.ndarray:
        arr_np = np.asarray(arr, dtype=np.float32)
        if arr_np.ndim == 2:
            return arr_np[None, ...]
        if arr_np.ndim != 3:
            raise ValueError(f"Expected array of shape [B,H,W] or [H,W], got {arr_np.shape}")
        return arr_np