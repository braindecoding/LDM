#!/usr/bin/env python3
"""
🧠 Optimized Vangerven Training - Advanced Tuning Strategies
Apply the same optimization strategies that worked for Miyawaki to Vangerven dataset.
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

from data.data_loader import load_fmri_data
from models.improved_brain_ldm import ImprovedBrainLDM
from train_miyawaki_optimized import AdvancedLossFunction, AdvancedDataAugmentation

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

def create_optimized_vangerven_dataloader(loader, batch_size=4, augment_factor=5):
    """Create optimized dataloader for Vangerven dataset with advanced augmentation."""
    
    train_data = loader.get_train_data()
    train_fmri = train_data['fmri']
    train_stimuli = train_data['stimuli']
    train_labels = train_data['labels']
    
    print(f"📊 Original Vangerven data: {len(train_fmri)} samples")
    
    # Advanced augmentation adapted for Vangerven's higher dimensional fMRI
    augmenter = AdvancedDataAugmentation()
    
    augmented_fmri = [train_fmri]
    augmented_stimuli = [train_stimuli]
    augmented_labels = [train_labels]
    
    for i in range(augment_factor - 1):
        # Augment fMRI with varying noise levels (adapted for 3092 dimensions)
        noise_scale = 0.02 + (i * 0.015)  # Slightly lower noise for higher dimensional data
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

def train_optimized_vangerven(epochs=100, batch_size=4, learning_rate=5e-5):
    """Train with optimized strategies for Vangerven dataset."""
    
    print(f"🚀 OPTIMIZED VANGERVEN TRAINING")
    print("=" * 50)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Load Vangerven data
    print("📁 Loading Vangerven dataset...")
    loader = load_fmri_data("data/digit69_28x28.mat", device=device)
    
    # Get dimensions
    train_data = loader.get_train_data()
    fmri_dim = train_data['fmri'].shape[1]
    print(f"🧠 fMRI dimension: {fmri_dim} voxels")
    
    # Create optimized dataloader
    dataloader, augmenter = create_optimized_vangerven_dataloader(
        loader, batch_size=batch_size, augment_factor=5
    )
    
    # Create model with Vangerven dimensions
    model = ImprovedBrainLDM(
        fmri_dim=fmri_dim,  # 3092 for Vangerven
        image_size=28,
        guidance_scale=7.5
    ).to(device)
    
    # Create enhanced decoder (adapted for higher dimensional input)
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
    
    # Optimizers with different learning rates (adapted for Vangerven)
    model_optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=learning_rate * 1.5, weight_decay=1e-4  # Slightly lower multiplier
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
            fmri_aug = augmenter.fmri_augmentation(fmri, noise_scale=0.02)
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
            
            # Gradient clipping (slightly higher for Vangerven's higher dimensionality)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.2)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.2)
            
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
                    'dataset': 'vangerven_optimized',
                    'augment_factor': 5,
                    'optimization': 'advanced_loss_alignment'
                }
            }
            
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(checkpoint, "checkpoints/best_vangerven_optimized_model.pt")
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
    
    print(f"\n🎉 Optimized Vangerven training completed!")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"📁 Model saved: checkpoints/best_vangerven_optimized_model.pt")
    
    return model, decoder, criterion, losses, best_loss

def main():
    """Main optimized training function for Vangerven."""
    print("🧠 VANGERVEN OPTIMIZATION EXPERIMENT")
    print("=" * 40)
    
    # Train with optimization
    model, decoder, criterion, losses, best_loss = train_optimized_vangerven(
        epochs=100,
        batch_size=4,
        learning_rate=5e-5
    )
    
    print(f"\n🎯 VANGERVEN OPTIMIZATION SUMMARY")
    print("=" * 30)
    print(f"✅ Advanced loss function with learnable weights")
    print(f"✅ Feature alignment (CCA-inspired)")
    print(f"✅ Consistency regularization")
    print(f"✅ Contrastive learning")
    print(f"✅ Advanced data augmentation (5x)")
    print(f"✅ Multi-optimizer strategy")
    print(f"🏆 Best loss: {best_loss:.6f}")
    
    print(f"\n📁 Next steps:")
    print(f"   • Run evaluation: PYTHONPATH=src python3 evaluate_vangerven_optimized.py")
    print(f"   • Compare with baseline and Miyawaki optimized results")

if __name__ == "__main__":
    main()
