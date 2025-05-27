"""
Latent Diffusion Model for fMRI Image Reconstruction.
Combines VAE and Diffusion Model for high-quality fMRI data generation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import logging
from .vae_encoder_decoder import VariationalAutoencoder
from .diffusion_model import DiffusionUNet, DDPMScheduler

logger = logging.getLogger(__name__)


class LatentDiffusionModel(nn.Module):
    """
    Complete Latent Diffusion Model for fMRI reconstruction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Latent Diffusion Model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Initialize VAE
        self.vae = VariationalAutoencoder(config)
        
        # Initialize diffusion model
        diffusion_config = config['diffusion']
        self.diffusion_model = DiffusionUNet(
            latent_dim=config['vae']['latent_dim'],
            model_channels=diffusion_config['model_channels'],
            num_res_blocks=diffusion_config['num_res_blocks'],
            dropout=diffusion_config['dropout']
        )
        
        # Initialize scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            beta_schedule=diffusion_config['beta_schedule']
        )
        
        # Training mode flags
        self.vae_training = True
        self.diffusion_training = True
        
        logger.info("Latent Diffusion Model initialized successfully")
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode fMRI data to latent space using VAE encoder.
        
        Args:
            x: Input fMRI data [batch_size, input_dim]
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            # Use mean for deterministic encoding during inference
            return mu
    
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to fMRI data using VAE decoder.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            
        Returns:
            Reconstructed fMRI data [batch_size, input_dim]
        """
        with torch.no_grad():
            return self.vae.decode(z)
    
    def forward_diffusion(
        self, 
        latents: torch.Tensor, 
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: add noise to latents.
        
        Args:
            latents: Clean latent representations
            timesteps: Timesteps for noise addition
            noise: Optional noise tensor (generated if None)
            
        Returns:
            Tuple of (noisy_latents, noise)
        """
        if noise is None:
            noise = torch.randn_like(latents)
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents, noise
    
    def reverse_diffusion(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse diffusion process: predict noise to remove.
        
        Args:
            noisy_latents: Noisy latent representations
            timesteps: Current timesteps
            
        Returns:
            Predicted noise
        """
        return self.diffusion_model(noisy_latents, timesteps)
    
    def compute_vae_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE training loss.
        
        Args:
            x: Input fMRI data
            
        Returns:
            Dictionary of VAE losses
        """
        vae_output = self.vae(x)
        return self.vae.compute_loss(
            x, 
            vae_output['reconstruction'],
            vae_output['mu'],
            vae_output['logvar']
        )
    
    def compute_diffusion_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion model training loss.
        
        Args:
            x: Input fMRI data
            
        Returns:
            Dictionary of diffusion losses
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode to latent space
        with torch.no_grad():
            latents = self.encode_to_latent(x)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.num_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents, _ = self.forward_diffusion(latents, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.reverse_diffusion(noisy_latents, timesteps)
        
        # Compute loss
        diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return {
            'diffusion_loss': diffusion_loss,
            'predicted_noise': predicted_noise,
            'target_noise': noise
        }
    
    def sample_latents(
        self,
        batch_size: int,
        device: torch.device,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Sample latent representations using the diffusion model.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            num_inference_steps: Number of denoising steps
            generator: Random number generator
            
        Returns:
            Generated latent representations
        """
        if num_inference_steps is None:
            num_inference_steps = self.config['sampling']['num_inference_steps']
        
        latent_dim = self.config['vae']['latent_dim']
        
        # Start with random noise
        latents = torch.randn(
            (batch_size, latent_dim), 
            device=device, 
            generator=generator
        )
        
        # Set timesteps
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps
        ).long().to(device)
        
        # Denoising loop
        for timestep in timesteps:
            # Predict noise
            with torch.no_grad():
                noise_pred = self.diffusion_model(
                    latents, 
                    timestep.expand(batch_size)
                )
            
            # Remove noise
            latents = self.scheduler.step(
                noise_pred, 
                timestep.item(), 
                latents, 
                generator
            )
        
        return latents
    
    def generate_fmri_samples(
        self,
        batch_size: int,
        device: torch.device,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate fMRI samples using the complete pipeline.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            num_inference_steps: Number of denoising steps
            generator: Random number generator
            
        Returns:
            Generated fMRI data
        """
        # Sample latents using diffusion model
        latents = self.sample_latents(
            batch_size, device, num_inference_steps, generator
        )
        
        # Decode latents to fMRI data
        fmri_samples = self.decode_from_latent(latents)
        
        return fmri_samples
    
    def reconstruct_fmri(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct fMRI data through the complete pipeline.
        
        Args:
            x: Input fMRI data
            
        Returns:
            Reconstructed fMRI data
        """
        # Encode to latent
        latents = self.encode_to_latent(x)
        
        # Decode back to fMRI
        reconstruction = self.decode_from_latent(latents)
        
        return reconstruction
    
    def set_training_mode(self, vae_training: bool = True, diffusion_training: bool = True):
        """
        Set training mode for different components.
        
        Args:
            vae_training: Whether to train VAE
            diffusion_training: Whether to train diffusion model
        """
        self.vae_training = vae_training
        self.diffusion_training = diffusion_training
        
        # Set requires_grad accordingly
        for param in self.vae.parameters():
            param.requires_grad = vae_training
        
        for param in self.diffusion_model.parameters():
            param.requires_grad = diffusion_training
        
        logger.info(f"Training mode set - VAE: {vae_training}, Diffusion: {diffusion_training}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary with model information
        """
        vae_params = sum(p.numel() for p in self.vae.parameters())
        diffusion_params = sum(p.numel() for p in self.diffusion_model.parameters())
        total_params = vae_params + diffusion_params
        
        return {
            'vae_parameters': vae_params,
            'diffusion_parameters': diffusion_params,
            'total_parameters': total_params,
            'latent_dim': self.config['vae']['latent_dim'],
            'input_dim': self.config['vae']['input_dim'],
            'num_timesteps': self.config['diffusion']['num_timesteps']
        }
