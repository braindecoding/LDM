"""
Training pipeline for Latent Diffusion Model.
Handles both VAE and diffusion model training with proper scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..models.latent_diffusion_model import LatentDiffusionModel
from ..utils.metrics import compute_reconstruction_metrics
from ..utils.visualization import plot_training_curves, plot_reconstructions

logger = logging.getLogger(__name__)


class LDMTrainer:
    """
    Trainer class for Latent Diffusion Model.
    """

    def __init__(
        self,
        model: LatentDiffusionModel,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        """
        Initialize trainer.

        Args:
            model: Latent Diffusion Model
            config: Configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Training configuration
        self.num_epochs = config['training']['num_epochs']
        self.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
        self.mixed_precision = config['training']['mixed_precision']

        # Initialize optimizers
        self.vae_optimizer = optim.Adam(
            self.model.vae.parameters(),
            lr=float(config['vae']['learning_rate']),
            betas=(0.9, 0.999),
            weight_decay=1e-6
        )

        self.diffusion_optimizer = optim.Adam(
            self.model.diffusion_model.parameters(),
            lr=float(config['diffusion']['learning_rate']),
            betas=(0.9, 0.999),
            weight_decay=1e-6
        )

        # Initialize schedulers
        self.vae_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.vae_optimizer, T_max=self.num_epochs
        )

        self.diffusion_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.diffusion_optimizer, T_max=self.num_epochs
        )

        # Mixed precision scaler
        if self.mixed_precision:
            try:
                # Try new API first
                self.scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
            except TypeError:
                # Fallback to old API
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create directories
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.log_dir = Path(config['training']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        if config['logging']['use_wandb'] and WANDB_AVAILABLE:
            wandb.init(
                project=config['logging']['project_name'],
                name=config['logging']['experiment_name'],
                config=config
            )
        elif config['logging']['use_wandb'] and not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not available. Install with: pip install wandb")

        logger.info("Trainer initialized successfully")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_vae_loss = 0.0
        total_diffusion_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs}"
        )

        for batch_idx, batch_data in enumerate(progress_bar):
            batch_data = batch_data.to(self.device)

            # Training step
            metrics = self._training_step(batch_data, batch_idx)

            # Update metrics
            total_vae_loss += metrics['vae_loss']
            total_diffusion_loss += metrics['diffusion_loss']
            total_recon_loss += metrics['recon_loss']
            total_kl_loss += metrics['kl_loss']
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'VAE': f"{metrics['vae_loss']:.4f}",
                'Diff': f"{metrics['diffusion_loss']:.4f}",
                'Recon': f"{metrics['recon_loss']:.4f}"
            })

            self.global_step += 1

        # Calculate average metrics
        avg_metrics = {
            'train_vae_loss': total_vae_loss / num_batches,
            'train_diffusion_loss': total_diffusion_loss / num_batches,
            'train_recon_loss': total_recon_loss / num_batches,
            'train_kl_loss': total_kl_loss / num_batches
        }

        return avg_metrics

    def _training_step(self, batch_data: torch.Tensor, batch_idx: int) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch_data: Batch of training data
            batch_idx: Batch index

        Returns:
            Dictionary of step metrics
        """
        # Phase 1: Train VAE
        if self.model.vae_training:
            self.vae_optimizer.zero_grad()

            if self.mixed_precision:
                try:
                    # Try new API first
                    with autocast(device_type=self.device.type):
                        vae_losses = self.model.compute_vae_loss(batch_data)
                        vae_loss = vae_losses['total_loss']
                except TypeError:
                    # Fallback to old API
                    with autocast():
                        vae_losses = self.model.compute_vae_loss(batch_data)
                        vae_loss = vae_losses['total_loss']

                self.scaler.scale(vae_loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.vae_optimizer)
                    self.scaler.update()
            else:
                vae_losses = self.model.compute_vae_loss(batch_data)
                vae_loss = vae_losses['total_loss']
                vae_loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.vae_optimizer.step()
        else:
            vae_losses = {'total_loss': torch.tensor(0.0), 'reconstruction_loss': torch.tensor(0.0), 'kl_loss': torch.tensor(0.0)}

        # Phase 2: Train Diffusion Model
        if self.model.diffusion_training:
            self.diffusion_optimizer.zero_grad()

            if self.mixed_precision:
                try:
                    # Try new API first
                    with autocast(device_type=self.device.type):
                        diffusion_losses = self.model.compute_diffusion_loss(batch_data)
                        diffusion_loss = diffusion_losses['diffusion_loss']
                except TypeError:
                    # Fallback to old API
                    with autocast():
                        diffusion_losses = self.model.compute_diffusion_loss(batch_data)
                        diffusion_loss = diffusion_losses['diffusion_loss']

                self.scaler.scale(diffusion_loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.diffusion_optimizer)
                    self.scaler.update()
            else:
                diffusion_losses = self.model.compute_diffusion_loss(batch_data)
                diffusion_loss = diffusion_losses['diffusion_loss']
                diffusion_loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.diffusion_optimizer.step()
        else:
            diffusion_losses = {'diffusion_loss': torch.tensor(0.0)}

        return {
            'vae_loss': vae_losses['total_loss'].item(),
            'diffusion_loss': diffusion_losses['diffusion_loss'].item(),
            'recon_loss': vae_losses['reconstruction_loss'].item(),
            'kl_loss': vae_losses['kl_loss'].item()
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_vae_loss = 0.0
        total_diffusion_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        all_reconstructions = []
        all_originals = []

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                batch_data = batch_data.to(self.device)

                # Compute losses
                vae_losses = self.model.compute_vae_loss(batch_data)
                diffusion_losses = self.model.compute_diffusion_loss(batch_data)

                total_vae_loss += vae_losses['total_loss'].item()
                total_diffusion_loss += diffusion_losses['diffusion_loss'].item()
                total_recon_loss += vae_losses['reconstruction_loss'].item()
                total_kl_loss += vae_losses['kl_loss'].item()
                num_batches += 1

                # Collect samples for metrics
                reconstructions = self.model.reconstruct_fmri(batch_data)
                all_reconstructions.append(reconstructions.cpu())
                all_originals.append(batch_data.cpu())

        # Calculate average losses
        avg_metrics = {
            'val_vae_loss': total_vae_loss / num_batches,
            'val_diffusion_loss': total_diffusion_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_kl_loss': total_kl_loss / num_batches
        }

        # Compute reconstruction metrics
        all_reconstructions = torch.cat(all_reconstructions, dim=0)
        all_originals = torch.cat(all_originals, dim=0)

        recon_metrics = compute_reconstruction_metrics(
            all_originals.numpy(),
            all_reconstructions.numpy()
        )
        avg_metrics.update(recon_metrics)

        return avg_metrics

    def train(self) -> Dict[str, list]:
        """
        Main training loop.

        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': []
        }

        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % self.config['training']['eval_every'] == 0:
                val_metrics = self.validate()

                # Update history
                history['train_loss'].append(train_metrics['train_vae_loss'] + train_metrics['train_diffusion_loss'])
                history['val_loss'].append(val_metrics['val_vae_loss'] + val_metrics['val_diffusion_loss'])
                history['train_recon_loss'].append(train_metrics['train_recon_loss'])
                history['val_recon_loss'].append(val_metrics['val_recon_loss'])

                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)

                # Check for improvement
                current_val_loss = val_metrics['val_vae_loss'] + val_metrics['val_diffusion_loss']
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Save regular checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch)

            # Update learning rates
            self.vae_scheduler.step()
            self.diffusion_scheduler.step()

        logger.info("Training completed!")
        return history

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to console and wandb."""
        # Console logging
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train - VAE: {train_metrics['train_vae_loss']:.4f}, "
                   f"Diffusion: {train_metrics['train_diffusion_loss']:.4f}")
        logger.info(f"  Val - VAE: {val_metrics['val_vae_loss']:.4f}, "
                   f"Diffusion: {val_metrics['val_diffusion_loss']:.4f}")

        # Wandb logging
        if self.config['logging']['use_wandb'] and WANDB_AVAILABLE:
            wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'diffusion_optimizer_state_dict': self.diffusion_optimizer.state_dict(),
            'vae_scheduler_state_dict': self.vae_scheduler.state_dict(),
            'diffusion_scheduler_state_dict': self.diffusion_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])
        self.vae_scheduler.load_state_dict(checkpoint['vae_scheduler_state_dict'])
        self.diffusion_scheduler.load_state_dict(checkpoint['diffusion_scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
