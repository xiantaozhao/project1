"""
Gaussian Diffusion for DOLCE with Proximal Solver support
Based on DOLCE's data consistency approach
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear schedule for beta values."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule for beta values."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for DOLCE.
    Supports DDPM and DDIM sampling with optional proximal solver for data consistency.
    
    Args:
        model: Conditional UNet model
        timesteps: Number of diffusion timesteps
        beta_schedule: Type of beta schedule ('linear' or 'cosine')
        objective: Prediction objective ('pred_noise' or 'pred_x0')
    """
    
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule='linear',
        objective='pred_noise',
    ):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.objective = objective
        
        # Define beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to x_start at timestep t.
        
        Args:
            x_start: Clean image (B, C, H, W)
            t: Timestep (B,)
            noise: Optional noise tensor
            
        Returns:
            Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_x0_from_eps(self, x_t, t, eps):
        """Predict x_0 from noise prediction."""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def predict_eps_from_x0(self, x_t, t, x0):
        """Predict noise from x_0 prediction."""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return (sqrt_recip_alphas_cumprod_t * x_t - x0) / sqrt_recipm1_alphas_cumprod_t
    
    def p_mean_variance(self, x_t, t, condition_fbp=None, condition_rls=None, clip_denoised=True):
        """
        Compute mean and variance for p(x_{t-1} | x_t).
        
        Args:
            x_t: Noisy image at timestep t
            t: Timestep
            condition_fbp: FBP condition
            condition_rls: RLS condition
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            Dictionary with model output, predicted x_0, mean, variance
        """
        # Model prediction
        model_output = self.model(x_t, t, condition_fbp=condition_fbp, condition_rls=condition_rls)
        
        # Predict x_0
        if self.objective == 'pred_noise':
            x0_pred = self.predict_x0_from_eps(x_t, t, model_output)
        elif self.objective == 'pred_x0':
            x0_pred = model_output
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
            
        # Clip predicted x_0
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
        # Compute posterior mean
        posterior_mean_coef1 = self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1)
        posterior_mean_coef2 = self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1)
        model_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t
        
        # Posterior variance
        model_variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        model_log_variance = self.posterior_log_variance_clipped[t].reshape(-1, 1, 1, 1)
        
        return {
            'model_output': model_output,
            'x0_pred': x0_pred,
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
        }
    
    @torch.no_grad()
    def p_sample(self, x_t, t, condition_fbp=None, condition_rls=None, clip_denoised=True):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) using DDPM.
        
        Args:
            x_t: Noisy image at timestep t
            t: Timestep
            condition_fbp: FBP condition
            condition_rls: RLS condition
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            x_{t-1} and predicted x_0
        """
        out = self.p_mean_variance(x_t, t, condition_fbp, condition_rls, clip_denoised)
        
        # Sample
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
        
        x_prev = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        
        return x_prev, out['x0_pred']
    
    @torch.no_grad()
    def ddim_sample(
        self,
        x_t,
        t,
        t_prev,
        condition_fbp=None,
        condition_rls=None,
        eta=0.0,
        clip_denoised=True
    ):
        """
        Sample x_{t_prev} from x_t using DDIM.
        
        Args:
            x_t: Noisy image at timestep t
            t: Current timestep
            t_prev: Previous timestep
            condition_fbp: FBP condition
            condition_rls: RLS condition
            eta: DDIM stochasticity parameter (0 = deterministic)
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            x_{t_prev} and predicted x_0
        """
        # Model prediction
        out = self.p_mean_variance(x_t, t, condition_fbp, condition_rls, clip_denoised)
        x0_pred = out['x0_pred']
        
        # Extract alpha values
        alpha_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # Handle t_prev: use ones for final step (when t_prev < 0)
        # Create mask for valid timesteps
        valid_mask = (t_prev >= 0).float().reshape(-1, 1, 1, 1)
        alpha_prev_valid = self.alphas_cumprod[torch.clamp(t_prev, min=0)].reshape(-1, 1, 1, 1)
        alpha_prev = valid_mask * alpha_prev_valid + (1 - valid_mask) * torch.ones_like(alpha_t)
        
        # Compute sigma
        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
        
        # Predict noise
        eps = self.predict_eps_from_x0(x_t, t, x0_pred)
        
        # Compute x_{t_prev}
        mean_pred = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev - sigma_t ** 2) * eps
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t_prev >= 0).float().reshape(-1, 1, 1, 1)
        
        x_prev = mean_pred + nonzero_mask * sigma_t * noise
        
        return x_prev, x0_pred
    
    @torch.no_grad()
    def sample_loop(
        self,
        shape,
        condition_fbp=None,
        condition_rls=None,
        ct_data_fidelity=None,
        sampler='ddim',
        ddim_steps=100,
        eta=0.0,
        start_timestep=None,
        x_start=None,
        clip_denoised=True,
        use_proximal_solver=False,
        rho=1.0,
        solver_type='apgm',
        solver_iterations=10,
        verbose=True,
    ):
        """
        Sample from the model with optional data consistency.
        
        Args:
            shape: Shape of output (B, C, H, W)
            condition_fbp: FBP reconstruction condition
            condition_rls: RLS reconstruction condition
            ct_data_fidelity: CTClass_astra instance for data consistency
            sampler: 'ddpm' or 'ddim'
            ddim_steps: Number of steps for DDIM sampling
            eta: DDIM stochasticity parameter
            start_timestep: Starting timestep (for SDEdit-style)
            x_start: Starting image (for SDEdit-style)
            clip_denoised: Whether to clip predicted x_0
            use_proximal_solver: Whether to apply proximal solver for data consistency
            rho: Proximal solver parameter
            solver_type: 'apgm' or 'cgrad'
            solver_iterations: Number of proximal solver iterations
            verbose: Whether to show progress bar
            
        Returns:
            Sampled images and list of intermediate x_0 predictions
        """
        device = next(self.model.parameters()).device
        batch_size = shape[0]
        
        # Initialize or start from x_start
        if start_timestep is not None and x_start is not None:
            # SDEdit-style: add noise to x_start
            t = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
            x = self.q_sample(x_start, t)
            start_t = start_timestep
        else:
            # Start from pure noise
            x = torch.randn(shape, device=device)
            start_t = self.timesteps - 1
            
        # Determine timestep sequence
        if sampler == 'ddim':
            timesteps = torch.linspace(start_t, 0, ddim_steps, dtype=torch.long, device=device)
        else:
            timesteps = torch.arange(start_t, -1, -1, device=device)
            
        x0_preds = []
        
        iterator = tqdm(timesteps, desc="Sampling", disable=not verbose)
        
        for i, t_curr in enumerate(iterator):
            t = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)
            
            # Sampling step
            if sampler == 'ddim':
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=device)
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                x, x0_pred = self.ddim_sample(x, t, t_prev_batch, condition_fbp, condition_rls, eta, clip_denoised)
            else:
                x, x0_pred = self.p_sample(x, t, condition_fbp, condition_rls, clip_denoised)
                
            # Apply proximal solver for data consistency
            if use_proximal_solver and ct_data_fidelity is not None:
                # Scale rho based on timestep (higher at early steps)
                timestep_ratio = t_curr.float() / self.timesteps
                rho_t = rho * (timestep_ratio + 0.1)  # Add baseline to avoid zero
                
                if solver_type == 'apgm':
                    x0_pred = ct_data_fidelity.prox_apgm(
                        x0_pred,
                        rho_t,
                        max_iter=solver_iterations
                    )
                elif solver_type == 'cgrad':
                    x0_pred = ct_data_fidelity.prox_cgrad(
                        x0_pred,
                        rho_t,
                        max_iter=solver_iterations
                    )
                    
                # Update x based on corrected x0_pred
                # Recompute x_t from corrected x_0 and predicted noise
                eps = self.predict_eps_from_x0(x, t, x0_pred)
                
                if sampler == 'ddim':
                    alpha_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
                    alpha_prev = self.alphas_cumprod[t_prev_batch].reshape(-1, 1, 1, 1) if t_prev >= 0 else torch.ones_like(alpha_t)
                    sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
                    x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev - sigma_t ** 2) * eps
                else:
                    # DDPM: use posterior mean with corrected x0
                    posterior_mean_coef1 = self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1)
                    posterior_mean_coef2 = self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1)
                    x = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x
                    
            x0_preds.append(x0_pred.cpu())
            
        return x, x0_preds


def create_gaussian_diffusion(model, **kwargs):
    """Factory function to create GaussianDiffusion."""
    defaults = {
        'timesteps': 1000,
        'beta_schedule': 'linear',
        'objective': 'pred_noise',
    }
    defaults.update(kwargs)
    return GaussianDiffusion(model, **defaults)
