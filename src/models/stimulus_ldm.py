"""
Latent Diffusion Model for fMRI-to-stimulus reconstruction (brain decoding).
Reconstructs visual stimuli from fMRI brain activation patterns.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging
from .vae_encoder_decoder import VariationalAutoencoder
from .diffusion_model import DiffusionUNet, DDPMScheduler

logger = logging.getLogger(__name__)


class StimulusLatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model for reconstructing visual stimuli from fMRI data.
    
    Pipeline:
    fMRI (3092) → VAE Encoder → Latent (256) → Diffusion → Enhanced Latent → VAE Decoder → Stimulus (784)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Stimulus Latent Diffusion Model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Update config for stimulus reconstruction
        stimulus_config = config.copy()
        stimulus_config['vae']['input_dim'] = 3092  # fMRI voxels
        stimulus_config['vae']['output_dim'] = 784   # Stimulus pixels (28x28)
        
        # Initialize VAE for fMRI → Stimulus mapping
        self.vae = StimulusVAE(stimulus_config)
        
        # Initialize diffusion model for latent enhancement
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
        
        logger.info("Stimulus Latent Diffusion Model initialized successfully")
    
    def encode_fmri_to_latent(self, fmri: torch.Tensor) -> torch.Tensor:
        """
        Encode fMRI data to latent space.
        
        Args:
            fmri: Input fMRI data [batch_size, 3092]
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        with torch.no_grad():
            mu, logvar = self.vae.encode(fmri)
            return mu
    
    def decode_latent_to_stimulus(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to stimulus.
        
        Args:
            latent: Latent representation [batch_size, latent_dim]
            
        Returns:
            Reconstructed stimulus [batch_size, 784]
        """
        with torch.no_grad():
            return self.vae.decode(latent)
    
    def reconstruct_stimulus(self, fmri: torch.Tensor, use_diffusion: bool = True) -> torch.Tensor:
        """
        Reconstruct stimulus from fMRI data using the complete LDM pipeline.
        
        Args:
            fmri: Input fMRI data [batch_size, 3092]
            use_diffusion: Whether to use diffusion enhancement
            
        Returns:
            Reconstructed stimulus [batch_size, 784]
        """
        if use_diffusion:
            # TRUE LATENT DIFFUSION RECONSTRUCTION
            # Step 1: Encode fMRI to latent space
            latents = self.encode_fmri_to_latent(fmri)
            
            # Step 2: Add noise for diffusion enhancement
            batch_size = latents.shape[0]
            device = latents.device
            
            # Add moderate noise (simulate degradation)
            noise_level = 0.2  # Lower noise for reconstruction
            noise = torch.randn_like(latents) * noise_level
            noisy_latents = latents + noise
            
            # Step 3: Use diffusion model to enhance latents
            num_denoising_steps = 5  # Fewer steps for reconstruction
            timesteps = torch.linspace(
                int(self.scheduler.num_timesteps * noise_level), 0,
                num_denoising_steps
            ).long().to(device)
            
            enhanced_latents = noisy_latents
            for timestep in timesteps:
                with torch.no_grad():
                    # Predict noise
                    noise_pred = self.diffusion_model(
                        enhanced_latents,
                        timestep.expand(batch_size)
                    )
                    
                    # Remove predicted noise
                    if timestep > 0:
                        enhanced_latents = self.scheduler.step(
                            noise_pred,
                            timestep.item(),
                            enhanced_latents
                        )
                    else:
                        # Final step: direct noise removal
                        alpha_prod = self.scheduler.alphas_cumprod[timestep].to(device)
                        enhanced_latents = (enhanced_latents - (1 - alpha_prod).sqrt() * noise_pred) / alpha_prod.sqrt()
            
            # Step 4: Decode enhanced latents to stimulus
            stimulus = self.decode_latent_to_stimulus(enhanced_latents)
            
        else:
            # VAE-ONLY RECONSTRUCTION (baseline)
            latents = self.encode_fmri_to_latent(fmri)
            stimulus = self.decode_latent_to_stimulus(latents)
        
        return stimulus
    
    def compute_vae_loss(self, fmri: torch.Tensor, stimulus: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE training loss for fMRI → stimulus mapping.
        
        Args:
            fmri: Input fMRI data
            stimulus: Target stimulus data
            
        Returns:
            Dictionary of VAE losses
        """
        vae_output = self.vae(fmri)
        return self.vae.compute_loss(
            stimulus,  # Target stimulus
            vae_output['reconstruction'],
            vae_output['mu'],
            vae_output['logvar']
        )
    
    def compute_diffusion_loss(self, fmri: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion model training loss.
        
        Args:
            fmri: Input fMRI data
            
        Returns:
            Dictionary of diffusion losses
        """
        batch_size = fmri.shape[0]
        device = fmri.device
        
        # Encode fMRI to latent space
        with torch.no_grad():
            latents = self.encode_fmri_to_latent(fmri)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.num_timesteps,
            (batch_size,), device=device
        ).long()
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.diffusion_model(noisy_latents, timesteps)
        
        # Compute loss
        diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return {
            'diffusion_loss': diffusion_loss,
            'predicted_noise': predicted_noise,
            'target_noise': noise
        }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        vae_params = sum(p.numel() for p in self.vae.parameters())
        diffusion_params = sum(p.numel() for p in self.diffusion_model.parameters())
        total_params = vae_params + diffusion_params
        
        return {
            'vae_parameters': vae_params,
            'diffusion_parameters': diffusion_params,
            'total_parameters': total_params,
            'latent_dim': self.config['vae']['latent_dim'],
            'fmri_dim': 3092,
            'stimulus_dim': 784,
            'num_timesteps': self.config['diffusion']['num_timesteps']
        }


class StimulusVAE(nn.Module):
    """
    VAE for fMRI → Stimulus mapping.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        vae_config = config['vae']
        
        self.input_dim = 3092  # fMRI voxels
        self.output_dim = 784   # Stimulus pixels
        self.latent_dim = vae_config['latent_dim']
        self.hidden_dims = vae_config['hidden_dims']
        self.beta = vae_config['beta']
        
        # Encoder: fMRI → Latent
        encoder_layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)
        
        # Decoder: Latent → Stimulus
        decoder_layers = []
        prev_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, self.output_dim))
        decoder_layers.append(nn.Sigmoid())  # Normalize stimulus to [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode fMRI to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to stimulus."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(self, target: torch.Tensor, reconstruction: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute VAE loss."""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstruction, target, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
