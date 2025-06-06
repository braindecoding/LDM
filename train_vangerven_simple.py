#!/usr/bin/env python3
"""
🧠 Train Brain LDM with Vangerven Dataset (digit69_28x28.mat)
Simple training script for the vangerven dataset with GPU support.
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import random

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

def create_simple_dataloader(loader, batch_size=4, augment_factor=3):
    """Create simple dataloader with basic augmentation."""
    
    train_data = loader.get_train_data()
    train_fmri = train_data['fmri']
    train_stimuli = train_data['stimuli']
    train_labels = train_data['labels']
    
    print(f"📊 Original data: {len(train_fmri)} samples")
    
    # Simple augmentation: add noise
    augmented_fmri = [train_fmri]
    augmented_stimuli = [train_stimuli]
    augmented_labels = [train_labels]
    
    for i in range(augment_factor - 1):
        # Add small amount of noise to fMRI
        noise_scale = 0.05 * (i + 1)
        noisy_fmri = train_fmri + torch.randn_like(train_fmri) * noise_scale
        
        augmented_fmri.append(noisy_fmri)
        augmented_stimuli.append(train_stimuli)
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
        dataset, batch_size=batch_size, shuffle=True
    )
    
    return dataloader

def train_vangerven_model(epochs=50, batch_size=4, learning_rate=1e-4, save_name="vangerven_simple"):
    """Train Brain LDM with Vangerven dataset."""
    
    print(f"🚀 Training Brain LDM with Vangerven Dataset")
    print("=" * 50)
    
    # Set random seed
    set_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load Vangerven data (digit69)
    print("📁 Loading Vangerven dataset...")
    loader = load_fmri_data("data/digit69_28x28.mat", device=device)
    
    # Get fMRI dimension for model
    train_data = loader.get_train_data()
    fmri_dim = train_data['fmri'].shape[1]
    print(f"🧠 fMRI dimension: {fmri_dim} voxels")
    print(f"📊 Training samples: {len(train_data['fmri'])}")
    print(f"🏷️ Unique labels: {torch.unique(train_data['labels']).tolist()}")
    
    # Create dataloader
    dataloader = create_simple_dataloader(
        loader, batch_size=batch_size, augment_factor=3
    )
    
    # Create model with Vangerven dimensions
    print(f"🤖 Creating model with fMRI dimension: {fmri_dim}")
    model = ImprovedBrainLDM(
        fmri_dim=fmri_dim,  # Use Vangerven's dimension (3092)
        image_size=28,
        guidance_scale=7.5
    ).to(device)
    
    # Create simple decoder
    simple_decoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 784),  # 28*28 = 784
        nn.Sigmoid()
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in simple_decoder.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    # Setup training
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(simple_decoder.parameters()), 
        lr=learning_rate, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # Training loop
    model.train()
    simple_decoder.train()
    losses = []
    best_loss = float('inf')
    start_time = time.time()
    
    print(f"\n🎯 Starting training for {epochs} epochs...")
    print(f"📊 Dataset: {len(dataloader)} batches per epoch")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, (fmri, stimuli, labels) in enumerate(dataloader):
            fmri = fmri.to(device)
            stimuli = stimuli.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                # Encode fMRI to features
                fmri_features = model.fmri_encoder(fmri)
                
                # Decode to images
                reconstructed = simple_decoder(fmri_features)
                
                # Compute loss
                recon_loss = nn.MSELoss()(reconstructed, stimuli)
                
                # Backward pass
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(simple_decoder.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                
                epoch_losses.append(recon_loss.item())
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        # Update scheduler
        scheduler.step()
        
        # Calculate epoch metrics
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'decoder_state_dict': simple_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'config': {
                        'fmri_dim': fmri_dim,
                        'dataset': 'digit69_28x28',
                        'augment_factor': 3
                    }
                }
                
                Path("checkpoints").mkdir(exist_ok=True)
                save_path = f"checkpoints/best_{save_name}_model.pt"
                torch.save(checkpoint, save_path)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | Time: {elapsed:.1f}s")
        else:
            print(f"Epoch {epoch+1:3d}/{epochs} | No valid batches")
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"📁 Model saved: checkpoints/best_{save_name}_model.pt")
    
    return model, simple_decoder, losses, best_loss

def main():
    """Main training function."""
    print("🧠 VANGERVEN DATASET TRAINING")
    print("=" * 35)
    
    # Train with Vangerven dataset
    model, decoder, losses, best_loss = train_vangerven_model(
        epochs=50,
        batch_size=4,
        learning_rate=1e-4,
        save_name="vangerven_simple"
    )
    
    print(f"\n🎯 TRAINING SUMMARY")
    print("=" * 20)
    print(f"✅ Successfully trained Brain LDM with Vangerven dataset")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"📊 Dataset: digit69_28x28.mat")
    print(f"🧠 fMRI dimension: 3092 voxels")
    print(f"📈 Training completed in {len(losses)} epochs")
    
    print(f"\n📁 Next steps:")
    print(f"   • Run evaluation: PYTHONPATH=src python3 show_vangerven_results.py")
    print(f"   • Compare with Miyawaki results")
    print(f"   • Visualize reconstructions")

if __name__ == "__main__":
    main()
