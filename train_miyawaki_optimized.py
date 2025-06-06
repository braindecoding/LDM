#!/usr/bin/env python3
"""
🧠 Optimized Miyawaki Training - Advanced Tuning Strategies
Comprehensive optimization for Miyawaki dataset with custom losses and alignment.
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import random
from sklearn.metrics import mutual_info_score

from data.data_loader import load_fmri_data
from models.improved_brain_ldm import ImprovedBrainLDM

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AdvancedLossFunction(nn.Module):
    """Advanced loss function with multiple alignment strategies."""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Learnable loss weights
        self.recon_weight = nn.Parameter(torch.tensor(1.0))
        self.perceptual_weight = nn.Parameter(torch.tensor(0.1))
        self.alignment_weight = nn.Parameter(torch.tensor(0.2))
        self.consistency_weight = nn.Parameter(torch.tensor(0.15))
        self.contrastive_weight = nn.Parameter(torch.tensor(0.1))
        
    def structural_similarity_loss(self, pred, target):
        """SSIM-based structural similarity loss."""
        # Compute local means
        mu1 = F.avg_pool2d(pred, 3, 1, 1)
        mu2 = F.avg_pool2d(target, 3, 1, 1)
        
        # Compute local variances and covariance
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2
        
        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM formula
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim.mean()
    
    def feature_alignment_loss(self, fmri_features, image_features):
        """Canonical Correlation Analysis inspired alignment loss."""
        # Normalize features
        fmri_norm = F.normalize(fmri_features, dim=1)
        image_norm = F.normalize(image_features, dim=1)
        
        # Compute cross-correlation matrix
        correlation = torch.mm(fmri_norm.T, image_norm)
        
        # Maximize diagonal elements (canonical correlation)
        alignment_loss = -torch.trace(correlation) / correlation.size(0)
        
        return alignment_loss
    
    def consistency_loss(self, pred1, pred2):
        """Consistency loss between different augmentations."""
        return F.mse_loss(pred1, pred2)
    
    def contrastive_loss(self, fmri_features, positive_features, negative_features, margin=1.0):
        """Contrastive loss for better feature learning."""
        pos_dist = F.pairwise_distance(fmri_features, positive_features)
        neg_dist = F.pairwise_distance(fmri_features, negative_features)
        
        loss = torch.clamp(margin + pos_dist - neg_dist, min=0.0)
        return loss.mean()
    
    def forward(self, predictions, targets, fmri_features, image_features, 
                pred_aug=None, negative_features=None):
        """Compute comprehensive loss."""
        
        # Ensure correct shapes
        if targets.dim() == 2:
            targets = targets.view(-1, 1, 28, 28)
        if predictions.dim() == 2:
            predictions = predictions.view(-1, 1, 28, 28)
        
        # 1. Reconstruction Loss (MSE + SSIM)
        mse_loss = F.mse_loss(predictions, targets)
        ssim_loss = self.structural_similarity_loss(predictions, targets)
        recon_loss = mse_loss + 0.3 * ssim_loss
        
        # 2. Perceptual Loss (gradient-based)
        grad_x_pred = torch.abs(predictions[:, :, 1:, :] - predictions[:, :, :-1, :])
        grad_y_pred = torch.abs(predictions[:, :, :, 1:] - predictions[:, :, :, :-1])
        grad_x_target = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
        grad_y_target = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
        
        perceptual_loss = F.mse_loss(grad_x_pred, grad_x_target) + \
                         F.mse_loss(grad_y_pred, grad_y_target)
        
        # 3. Feature Alignment Loss
        alignment_loss = self.feature_alignment_loss(fmri_features, image_features)
        
        # 4. Consistency Loss (if augmented prediction available)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if pred_aug is not None:
            if pred_aug.dim() == 2:
                pred_aug = pred_aug.view(-1, 1, 28, 28)
            consistency_loss = self.consistency_loss(predictions, pred_aug)
        
        # 5. Contrastive Loss (if negative features available)
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if negative_features is not None:
            contrastive_loss = self.contrastive_loss(
                fmri_features, image_features, negative_features
            )
        
        # Weighted combination
        total_loss = (self.recon_weight * recon_loss + 
                     self.perceptual_weight * perceptual_loss +
                     self.alignment_weight * alignment_loss +
                     self.consistency_weight * consistency_loss +
                     self.contrastive_weight * contrastive_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'perceptual_loss': perceptual_loss,
            'alignment_loss': alignment_loss,
            'consistency_loss': consistency_loss,
            'contrastive_loss': contrastive_loss,
            'weights': {
                'recon': self.recon_weight.item(),
                'perceptual': self.perceptual_weight.item(),
                'alignment': self.alignment_weight.item(),
                'consistency': self.consistency_weight.item(),
                'contrastive': self.contrastive_weight.item()
            }
        }

class AdvancedDataAugmentation:
    """Advanced data augmentation for fMRI and image pairs."""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def fmri_augmentation(self, fmri, noise_scale=0.05):
        """Advanced fMRI augmentation."""
        batch_size, fmri_dim = fmri.shape
        
        # 1. Gaussian noise
        noise = torch.randn_like(fmri) * noise_scale
        
        # 2. Dropout simulation (random voxel masking)
        dropout_mask = torch.rand_like(fmri) > 0.1
        
        # 3. Temporal smoothing simulation
        smooth_noise = torch.randn_like(fmri) * (noise_scale * 0.5)
        
        # 4. ROI-based augmentation (simulate different brain regions)
        roi_size = fmri_dim // 10
        roi_start = torch.randint(0, fmri_dim - roi_size, (1,)).item()
        roi_noise = torch.randn(batch_size, roi_size, device=self.device) * (noise_scale * 2)
        
        augmented_fmri = fmri.clone()
        augmented_fmri += noise
        augmented_fmri *= dropout_mask.float()
        augmented_fmri += smooth_noise
        augmented_fmri[:, roi_start:roi_start+roi_size] += roi_noise
        
        return augmented_fmri
    
    def image_augmentation(self, images):
        """Subtle image augmentation that preserves content."""
        # Convert to 2D if needed
        if images.dim() == 4:
            images_2d = images.view(images.size(0), -1)
        else:
            images_2d = images
        
        # Add small amount of noise
        noise = torch.randn_like(images_2d) * 0.02
        augmented = torch.clamp(images_2d + noise, 0, 1)
        
        return augmented
    
    def create_negative_samples(self, fmri_batch):
        """Create negative samples for contrastive learning."""
        batch_size = fmri_batch.size(0)
        
        # Shuffle within batch
        indices = torch.randperm(batch_size, device=self.device)
        negative_fmri = fmri_batch[indices]
        
        # Add more noise to make them more negative
        noise = torch.randn_like(negative_fmri) * 0.1
        negative_fmri += noise
        
        return negative_fmri

def create_optimized_dataloader(loader, batch_size=4, augment_factor=5):
    """Create optimized dataloader with advanced augmentation."""
    
    train_data = loader.get_train_data()
    train_fmri = train_data['fmri']
    train_stimuli = train_data['stimuli']
    train_labels = train_data['labels']
    
    print(f"📊 Original Miyawaki data: {len(train_fmri)} samples")
    
    # Advanced augmentation
    augmenter = AdvancedDataAugmentation()
    
    augmented_fmri = [train_fmri]
    augmented_stimuli = [train_stimuli]
    augmented_labels = [train_labels]
    
    for i in range(augment_factor - 1):
        # Augment fMRI with varying noise levels
        noise_scale = 0.03 + (i * 0.02)  # Increasing noise
        aug_fmri = augmenter.fmri_augmentation(train_fmri, noise_scale)
        aug_stimuli = augmenter.image_augmentation(train_stimuli)
        
        augmented_fmri.append(aug_fmri)
        augmented_stimuli.append(aug_stimuli)
        augmented_labels.append(train_labels)
    
    # Combine all data
    combined_fmri = torch.cat(augmented_fmri, dim=0)
    combined_stimuli = torch.cat(augmented_stimuli, dim=0)
    combined_labels = torch.cat(augmented_labels, dim=0)
    
    print(f"📊 Augmented data: {len(combined_fmri)} samples ({augment_factor}x)")
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        combined_fmri, combined_stimuli, combined_labels
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    return dataloader, augmenter

def train_optimized_miyawaki(epochs=100, batch_size=4, learning_rate=5e-5):
    """Train with optimized strategies for Miyawaki dataset."""
    
    print(f"🚀 OPTIMIZED MIYAWAKI TRAINING")
    print("=" * 50)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Load data
    print("📁 Loading Miyawaki dataset...")
    loader = load_fmri_data("data/miyawaki_structured_28x28.mat", device=device)
    
    # Get dimensions
    train_data = loader.get_train_data()
    fmri_dim = train_data['fmri'].shape[1]
    print(f"🧠 fMRI dimension: {fmri_dim} voxels")
    
    # Create optimized dataloader
    dataloader, augmenter = create_optimized_dataloader(
        loader, batch_size=batch_size, augment_factor=5
    )
    
    # Create model
    model = ImprovedBrainLDM(
        fmri_dim=fmri_dim,
        image_size=28,
        guidance_scale=7.5
    ).to(device)
    
    # Create decoder
    decoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 784),
        nn.Sigmoid()
    ).to(device)
    
    # Advanced loss function
    criterion = AdvancedLossFunction(device=device).to(device)
    
    # Optimizers with different learning rates
    model_optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=learning_rate * 2, weight_decay=1e-4
    )
    loss_optimizer = torch.optim.AdamW(
        criterion.parameters(), lr=learning_rate * 0.1, weight_decay=1e-5
    )
    
    # Schedulers
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        model_optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        decoder_optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"📊 Loss parameters: {sum(p.numel() for p in criterion.parameters()):,}")
    
    # Training loop
    model.train()
    decoder.train()
    criterion.train()
    
    losses = []
    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    
    print(f"\n🎯 Starting optimized training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_metrics = {
            'recon': [], 'perceptual': [], 'alignment': [], 
            'consistency': [], 'contrastive': []
        }
        
        for batch_idx, (fmri, stimuli, labels) in enumerate(dataloader):
            fmri = fmri.to(device)
            stimuli = stimuli.to(device)
            
            # Forward pass
            fmri_features = model.fmri_encoder(fmri)
            predictions = decoder(fmri_features)
            
            # Create augmented version for consistency
            fmri_aug = augmenter.fmri_augmentation(fmri, noise_scale=0.03)
            fmri_features_aug = model.fmri_encoder(fmri_aug)
            predictions_aug = decoder(fmri_features_aug)
            
            # Create negative samples
            negative_fmri = augmenter.create_negative_samples(fmri)
            negative_features = model.fmri_encoder(negative_fmri)
            
            # Image features (for alignment)
            if stimuli.dim() == 2:
                stimuli_reshaped = stimuli.view(-1, 1, 28, 28)
            else:
                stimuli_reshaped = stimuli
            
            # Simple image encoder for alignment
            image_features = model.vae_encoder(stimuli_reshaped)
            
            # Compute advanced loss
            loss_dict = criterion(
                predictions=predictions,
                targets=stimuli,
                fmri_features=fmri_features,
                image_features=image_features,
                pred_aug=predictions_aug,
                negative_features=negative_features
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            model_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss_optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            model_optimizer.step()
            decoder_optimizer.step()
            loss_optimizer.step()
            
            # Record metrics
            epoch_losses.append(total_loss.item())
            for key in epoch_metrics:
                if key in loss_dict:
                    epoch_metrics[key].append(loss_dict[f'{key}_loss'].item())
        
        # Update schedulers
        model_scheduler.step()
        decoder_scheduler.step()
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'config': {
                    'fmri_dim': fmri_dim,
                    'dataset': 'miyawaki_optimized',
                    'augment_factor': 5,
                    'optimization': 'advanced_loss_alignment'
                }
            }
            
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(checkpoint, "checkpoints/best_miyawaki_optimized_model.pt")
        else:
            patience += 1
        
        # Progress report
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            weights = loss_dict['weights']
            
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f} | Best = {best_loss:.6f}")
            print(f"  Weights: R={weights['recon']:.3f}, P={weights['perceptual']:.3f}, "
                  f"A={weights['alignment']:.3f}, C={weights['consistency']:.3f}")
            print(f"  Time: {elapsed:.1f}s | LR: {model_scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping
        if patience >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n🎉 Optimized training completed!")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"📁 Model saved: checkpoints/best_miyawaki_optimized_model.pt")
    
    return model, decoder, criterion, losses, best_loss

def main():
    """Main optimized training function."""
    print("🧠 MIYAWAKI OPTIMIZATION EXPERIMENT")
    print("=" * 40)
    
    # Train with optimization
    model, decoder, criterion, losses, best_loss = train_optimized_miyawaki(
        epochs=100,
        batch_size=4,
        learning_rate=5e-5
    )
    
    print(f"\n🎯 OPTIMIZATION SUMMARY")
    print("=" * 25)
    print(f"✅ Advanced loss function with learnable weights")
    print(f"✅ Feature alignment (CCA-inspired)")
    print(f"✅ Consistency regularization")
    print(f"✅ Contrastive learning")
    print(f"✅ Advanced data augmentation")
    print(f"✅ Multi-optimizer strategy")
    print(f"🏆 Best loss: {best_loss:.6f}")
    
    print(f"\n📁 Next steps:")
    print(f"   • Run evaluation: PYTHONPATH=src python3 evaluate_miyawaki_optimized.py")
    print(f"   • Compare with baseline results")

if __name__ == "__main__":
    main()
