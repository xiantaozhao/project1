"""
DOLCE: Data-consistent Optimization for Limited-angle CT Enhancement
Model implementation using original DOLCE UNet from GitHub
"""

# Use original DOLCE UNet implementation
from .unet_original import ConditionalModel

# Other DOLCE components
from .gaussian_diffusion import (
    GaussianDiffusion,
    create_gaussian_diffusion,
    linear_beta_schedule,
    cosine_beta_schedule,
)
from .ct_data_fidelity import (
    CTClass_astra,
    create_ct_data_fidelity,
)

__all__ = [
    # Original DOLCE UNet
    "ConditionalModel",
    # Diffusion components
    "GaussianDiffusion",
    "create_gaussian_diffusion",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    # Data fidelity
    "CTClass_astra",
    "create_ct_data_fidelity",
]
