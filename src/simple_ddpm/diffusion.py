"""Backward-compat wrappers for the migrated diffusion module."""

from warnings import warn

from src.model.diffusion import DDIM, Diffusion, linear_beta_schedule

__all__ = ["Diffusion", "DDIM", "linear_beta_schedule"]

warn(
    "`src.simple_ddpm.diffusion` is deprecated. Use `src.model.diffusion` instead.",
    DeprecationWarning,
    stacklevel=2,
)
