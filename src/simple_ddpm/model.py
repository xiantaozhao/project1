"""Backward-compat wrappers for the migrated diffusion module."""

from warnings import warn

from src.model.diffusion import SimpleUNet, timestep_embedding

__all__ = ["SimpleUNet", "timestep_embedding"]

warn(
    "`src.simple_ddpm.model` is deprecated. Use `src.model.diffusion` instead.",
    DeprecationWarning,
    stacklevel=2,
)
