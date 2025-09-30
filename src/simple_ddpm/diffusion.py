import math
from dataclasses import dataclass
import torch
from typing import Optional, List
from tqdm import tqdm


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


@dataclass
class Diffusion:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self):
        self.betas = linear_beta_schedule(self.T, self.beta_start, self.beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def to(self, device):
        for name in [
            'betas','alphas','alphas_cumprod','alphas_cumprod_prev',
            'sqrt_alphas_cumprod','sqrt_one_minus_alphas_cumprod',
            'sqrt_recip_alphas','posterior_variance']:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        fac1 = self.sqrt_alphas_cumprod[t][:, None, None, None]
        fac2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return fac1 * x0 + fac2 * noise

    def p_mean_variance(self, model, x: torch.Tensor, t: torch.Tensor):
        eps = model(x, t)
        # posterior mean with eps-prediction
        beta_t = self.betas[t][:, None, None, None]
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_bar_t = self.alphas_cumprod[t][:, None, None, None]
        posterior_mean = (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps) / torch.sqrt(alpha_t)
        posterior_var = self.posterior_variance[t][:, None, None, None]
        return posterior_mean, posterior_var, eps

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, var, _ = self.p_mean_variance(model, x, t)
        if (t > 0).any():
            noise = torch.randn_like(x)
            # add noise where t>0
            noise_mask = (t > 0).float()[:, None, None, None]
            x = mean + noise_mask * torch.sqrt(var) * noise
        else:
            x = mean
        return x

    @torch.no_grad()
    def sample(self, model, shape, device, num_steps: Optional[int] = None, show_progress: bool = True):
        """DDPM sampling loop. If num_steps < T, uses strided timesteps as an approximation."""
        x = torch.randn(shape, device=device)
        if num_steps is not None and num_steps < self.T:
            step = max(1, self.T // num_steps)
            timesteps = list(range(0, self.T, step))
        else:
            timesteps = list(range(self.T))
        timesteps = list(reversed(timesteps))

        iterator = tqdm(timesteps, desc="DDPM Sampling", leave=False) if show_progress else timesteps
        for t in iterator:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        return x.clamp(0.0, 1.0)


@dataclass
class DDIM:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    eta: float = 0.0

    def __post_init__(self):
        self.betas = linear_beta_schedule(self.T, self.beta_start, self.beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        for name in ['betas','alphas','alphas_cumprod']:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def set_timesteps(self, num_inference_steps: int) -> List[int]:
        step = max(1, self.T // num_inference_steps)
        ts = list(range(0, self.T, step))
        ts = list(reversed(ts))
        return ts

    @torch.no_grad()
    def step_from_to(self, model_output: torch.Tensor, t: int, t_prev: int, sample: torch.Tensor, eta: Optional[float] = None):
        """Single DDIM step from t -> t_prev (t_prev < t). Deterministic if eta=0."""
        if eta is None:
            eta = self.eta
        device = sample.device
        dtype = sample.dtype

        alpha_bar_t = self.alphas_cumprod[t].to(device=device, dtype=dtype)
        if t_prev >= 0:
            alpha_bar_prev = self.alphas_cumprod[t_prev].to(device=device, dtype=dtype)
        else:
            alpha_bar_prev = torch.ones((), device=device, dtype=dtype)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        # predict x0
        pred_x0 = (sample - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t

        # variance and direction term
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
        sigma_t = torch.clamp(sigma_t, min=0.0)
        dir_coeff = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t**2, min=0.0))

        # x_{t-1}
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_coeff * model_output
        if eta > 0:
            noise = torch.randn_like(sample)
            x_prev = x_prev + sigma_t * noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample(self, model, shape, device, num_steps: int = 50, eta: float = 0.0, show_progress: bool = True):
        x = torch.randn(shape, device=device)
        timesteps = self.set_timesteps(num_steps)
        iterator = tqdm(timesteps, desc="DDIM Sampling", leave=False) if show_progress else timesteps
        for i, t in enumerate(iterator):
            t_prev = timesteps[i+1] if i+1 < len(timesteps) else -1
            t_vec = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = model(x, t_vec)
            x, _ = self.step_from_to(eps, t, t_prev, x, eta=eta)
        return x.clamp(0.0, 1.0)
