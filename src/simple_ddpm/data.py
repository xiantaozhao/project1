"""Backward-compat wrappers for the migrated diffusion dataset helpers."""

from warnings import warn

from src.data.image_folder_dataset import ImageFolderDataset

__all__ = ["ImageFolderDataset"]

warn(
    "`src.simple_ddpm.data` is deprecated. Use `src.data.image_folder_dataset` instead.",
    DeprecationWarning,
    stacklevel=2,
)
