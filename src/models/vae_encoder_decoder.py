"""
Variational Autoencoder (VAE) for fMRI data encoding/decoding.
Creates a latent representation of fMRI data for the diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class VAEEncoder(nn.Module):
    """
    Encoder network that maps fMRI data to latent space.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list,
        dropout: float = 0.1
    ):
        """
        Initialize VAE encoder.
        
        Args:
            input_dim: Input dimension (number of voxels)
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        logger.info(f"VAE Encoder initialized: {input_dim} -> {latent_dim}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input fMRI data [batch_size, input_dim]
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder network that reconstructs fMRI data from latent space.
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list,
        dropout: float = 0.1
    ):
        """
        Initialize VAE decoder.
        
        Args:
            latent_dim: Latent space dimension
            output_dim: Output dimension (number of voxels)
            hidden_dims: List of hidden layer dimensions (reversed)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        # Reverse hidden dimensions for decoder
        reversed_hidden_dims = list(reversed(hidden_dims))
        
        for hidden_dim in reversed_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        logger.info(f"VAE Decoder initialized: {latent_dim} -> {output_dim}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            
        Returns:
            Reconstructed fMRI data [batch_size, output_dim]
        """
        return self.decoder(z)


class VariationalAutoencoder(nn.Module):
    """
    Complete Variational Autoencoder for fMRI data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize VAE with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        vae_config = config['vae']
        
        self.input_dim = vae_config['input_dim']
        self.latent_dim = vae_config['latent_dim']
        self.beta = vae_config['beta']
        
        # Initialize encoder and decoder
        self.encoder = VAEEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=vae_config['hidden_dims']
        )
        
        self.decoder = VAEDecoder(
            latent_dim=self.latent_dim,
            output_dim=self.input_dim,
            hidden_dims=vae_config['hidden_dims']
        )
        
        logger.info("VAE initialized successfully")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input fMRI data
            
        Returns:
            Tuple of (mu, logvar)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to fMRI data.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed fMRI data
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input fMRI data [batch_size, input_dim]
            
        Returns:
            Dictionary containing reconstruction, mu, logvar, and z
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            x: Original input
            reconstruction: Reconstructed output
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the latent space and decode.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated fMRI data
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
