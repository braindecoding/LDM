"""
Diffusion Model implementation for latent space denoising.
Implements DDPM (Denoising Diffusion Probabilistic Models) for fMRI latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for timestep encoding.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: Timestep tensor [batch_size]
            
        Returns:
            Position embeddings [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding for the diffusion model.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        
        self.residual_conv = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor [batch_size, in_channels]
            time_emb: Time embedding [batch_size, time_emb_dim]
            
        Returns:
            Output tensor [batch_size, out_channels]
        """
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb
        
        h = self.block2(h)
        
        # Residual connection
        return h + self.residual_conv(x)


class DiffusionUNet(nn.Module):
    """
    U-Net style architecture for diffusion model in latent space.
    """
    
    def __init__(
        self,
        latent_dim: int,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, model_channels)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        channels = [model_channels, model_channels * 2, model_channels * 4]
        
        for i in range(len(channels) - 1):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(
                        channels[i], 
                        channels[i + 1], 
                        time_emb_dim, 
                        dropout
                    )
                )
        
        # Middle block
        self.middle_block = ResidualBlock(
            channels[-1], 
            channels[-1], 
            time_emb_dim, 
            dropout
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        
        for i in range(len(reversed_channels) - 1):
            for _ in range(num_res_blocks):
                self.decoder_blocks.append(
                    ResidualBlock(
                        reversed_channels[i], 
                        reversed_channels[i + 1], 
                        time_emb_dim, 
                        dropout
                    )
                )
        
        # Output projection
        self.output_proj = nn.Linear(model_channels, latent_dim)
        
        logger.info(f"Diffusion UNet initialized with {latent_dim} latent dimensions")
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through diffusion model.
        
        Args:
            x: Noisy latent input [batch_size, latent_dim]
            timesteps: Timestep tensor [batch_size]
            
        Returns:
            Predicted noise [batch_size, latent_dim]
        """
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Input projection
        h = self.input_proj(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Encoder
        for block in self.encoder_blocks:
            h = block(h, time_emb)
            skip_connections.append(h)
        
        # Middle
        h = self.middle_block(h, time_emb)
        
        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            if i < len(skip_connections) - 1:
                skip = skip_connections[-(i + 2)]
                h = h + skip
            h = block(h, time_emb)
        
        # Output projection
        return self.output_proj(h)


class DDPMScheduler:
    """
    DDPM noise scheduler for forward and reverse diffusion processes.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        self.num_timesteps = num_timesteps
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        logger.info(f"DDPM Scheduler initialized with {num_timesteps} timesteps")
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine beta schedule as proposed in improved DDPM.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self, 
        original_samples: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to original samples according to the noise schedule.
        
        Args:
            original_samples: Original clean samples
            noise: Random noise to add
            timesteps: Timesteps for each sample
            
        Returns:
            Noisy samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Predict the sample at the previous timestep by reversing the SDE.
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in the diffusion chain
            sample: Current instance of sample being created by diffusion process
            generator: Random number generator
            
        Returns:
            Predicted previous sample
        """
        t = timestep
        
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod_prev[t].to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        
        # 2. compute predicted original sample from predicted noise
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # 3. compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t] ** (0.5) * (1 - alpha_prod_t_prev) / beta_prod_t
        
        # 4. compute predicted previous sample μ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # 5. add noise
        variance = 0
        if t > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device)
            variance = (self.posterior_variance[t] ** 0.5) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample
