"""
Gaussian diffusion with DDPM and optionally DDIM sampling.

References:
Diffuser: https://github.com/jannerm/diffuser
Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
Annotated DDIM/DDPM: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

"""

import logging
import torch
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

from model.diffusion.sampling import (
    extract,
    cosine_beta_schedule,
    make_timesteps,
)

from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")


class DiffusionModel(nn.Module):

    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="cuda:0",
        # Various clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,  # DDIM only
        # DDPM parameters
        denoising_steps=100,
        predict_epsilon=True,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        controllable_noise=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.controllable_noise = controllable_noise
        # Clip noise value at each denoising step
        self.denoised_clip_value = denoised_clip_value

        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Clip epsilon for numerical stability
        self.eps_clip_value = eps_clip_value

        # Set up models
        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=device, weights_only=True
            )
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded RL-trained policy from %s", network_path)
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

        """
        DDPM parameters

        """
        """
        βₜ
        """
        self.betas = cosine_beta_schedule(denoising_steps).to(device)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]]
        )
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        """
        DDIM parameters

        In DDIM paper https://arxiv.org/pdf/2010.02502, alpha is alpha_cumprod in DDPM https://arxiv.org/pdf/2102.09672
        """
        if self.use_ddim:
            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == "uniform":  # use the HF "leading" style
                step_ratio = self.denoising_steps // self.ddim_steps
                self.ddim_t = (
                    torch.arange(0, self.ddim_steps, device=self.device) * step_ratio
                )
            else:
                raise "Unknown discretization method for DDIM."
            
            self.ddim_alphas = (
                self.alphas_cumprod[self.ddim_t].clone().to(torch.float32)
            )
            self.ddim_alphas_sqrt = torch.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = torch.cat(
                [
                    torch.tensor([1.0]).to(torch.float32).to(self.device),
                    self.alphas_cumprod[self.ddim_t[:-1]],
                ]
            )
            self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5

            # Initialize fixed sigmas for inference - eta=0
            ddim_eta = 0
            self.ddim_sigmas = (
                ddim_eta
                * (
                    (1 - self.ddim_alphas_prev)
                    / (1 - self.ddim_alphas)
                    * (1 - self.ddim_alphas / self.ddim_alphas_prev)
                )
                ** 0.5
            )

            # Flip all
            self.ddim_t = torch.flip(self.ddim_t, [0])
            self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
            self.ddim_alphas_sqrt = torch.flip(self.ddim_alphas_sqrt, [0])
            self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch.flip(
                self.ddim_sqrt_one_minus_alphas, [0]
            )
            self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])

        

    # ---------- Sampling ----------#

    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        network_override=None,
        deterministic=False,
        ddim_alpha=None,
        ddim_alpha_prev=None,
    ):
        if network_override is not None:
            noise = network_override(x, t, cond=cond)
        else:
            noise = self.network(x, t, cond=cond)

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                if ddim_alpha is None:
                    alpha = extract(self.ddim_alphas, index, x.shape)
                    alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                    sqrt_one_minus_alpha = extract(
                        self.ddim_sqrt_one_minus_alphas, index, x.shape
                    )
                else:
                    alpha = ddim_alpha
                    alpha_prev = ddim_alpha_prev
                    sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1.0 - alpha, min=0.0))
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε

            eta=0
            """
            if ddim_alpha_prev is None:
                sigma = extract(self.ddim_sigmas, index, x.shape)
            else:
                sigma = torch.zeros_like(alpha_prev)
            # sigma should be 0
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * noise
            mu = (alpha_prev**0.5) * x_recon + dir_xt
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    @torch.no_grad()
    def forward(self, cond, deterministic=True, num_steps=None, chunk_size=None):
        """
        Forward pass for sampling actions. Used in evaluating pre-trained/fine-tuned policy. Not modifying diffusion clipping

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Loop
        if not self.controllable_noise:
            x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        else:
            x = cond["noise_action"]
        if self.use_ddim:
            t_all = self.ddim_t
            max_total_steps = len(self.ddim_t)
        else:
            t_all = list(reversed(range(self.denoising_steps)))
            max_total_steps = self.denoising_steps

        if num_steps is None:
            step_budget = torch.full((B,), max_total_steps, device=device, dtype=torch.long)
        elif torch.is_tensor(num_steps):
            step_budget = num_steps.to(device=device).reshape(-1).to(dtype=torch.long)
        elif isinstance(num_steps, (list, tuple)):
            step_budget = torch.as_tensor(num_steps, device=device, dtype=torch.long).reshape(-1)
        else:
            step_budget = torch.full((B,), int(num_steps), device=device, dtype=torch.long)
        step_budget = torch.clamp(step_budget, min=1, max=max_total_steps)
        if step_budget.shape[0] == 1 and B > 1:
            step_budget = step_budget.repeat(B)
        elif step_budget.shape[0] != B:
            raise ValueError(f"num_steps batch size mismatch: got {step_budget.shape[0]}, expected {B}")

        if self.use_ddim:
            max_budget = int(step_budget.max().item())
            full_steps = torch.full_like(step_budget, int(self.denoising_steps))
            step_ratio = torch.div(full_steps, step_budget, rounding_mode="floor").clamp(min=1)
            ones_alpha = torch.ones((B, 1, 1), device=device, dtype=self.alphas_cumprod.dtype)
            for i in range(max_budget):
                active_mask = (i < step_budget).view(B, 1, 1)
                if not torch.any(active_mask):
                    break
                remaining = torch.clamp(step_budget - 1 - i, min=0)
                t_b = torch.clamp(remaining * step_ratio, max=self.denoising_steps - 1).to(dtype=torch.long)
                next_remaining = torch.clamp(step_budget - 2 - i, min=0)
                next_t = torch.clamp(next_remaining * step_ratio, max=self.denoising_steps - 1).to(dtype=torch.long)
                alpha = self.alphas_cumprod.gather(0, t_b).view(B, 1, 1)
                next_alpha = self.alphas_cumprod.gather(0, next_t).view(B, 1, 1)
                has_next = (i + 1 < step_budget).view(B, 1, 1)
                alpha_prev = torch.where(has_next, next_alpha, ones_alpha)
                index_b = torch.clamp(i + (max_total_steps - step_budget), min=0, max=max_total_steps - 1).to(dtype=torch.long)
                mean, _ = self.p_mean_var(
                    x=x,
                    t=t_b,
                    cond=cond,
                    index=index_b,
                    deterministic=deterministic,
                    ddim_alpha=alpha,
                    ddim_alpha_prev=alpha_prev,
                )
                x = torch.where(active_mask, mean, x)
        else:
            for i, t in enumerate(t_all):
                active_mask = (i < step_budget).view(B, 1, 1)
                if not torch.any(active_mask):
                    break
                t_b = make_timesteps(B, t, device)
                index_b = make_timesteps(B, i, device)
                mean, logvar = self.p_mean_var(
                    x=x,
                    t=t_b,
                    cond=cond,
                    index=index_b,
                    deterministic=deterministic,
                )
                std = torch.exp(0.5 * logvar)
                if t == 0:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=1e-3)
                noise = torch.randn_like(x).clamp_(
                    -self.randn_clip_value, self.randn_clip_value
                )
                x_next = mean + std * noise
                x = torch.where(active_mask, x_next, x)

        # Clamp final action once after all effective denoising updates.
        if self.final_action_clip_value is not None:
            x = torch.clamp(
                x, -self.final_action_clip_value, self.final_action_clip_value
            )
        return Sample(x, None)

    # ---------- Supervised training ----------#

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, *args, t)

    def p_losses(
        self,
        x_start,
        cond: dict,
        t,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        device = x_start.device

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            return F.mse_loss(x_recon, noise, reduction="mean")
        else:
            return F.mse_loss(x_recon, x_start, reduction="mean")

    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        if noise is None:
            device = x_start.device
            noise = torch.randn_like(x_start, device=device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
