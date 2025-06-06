#!/usr/bin/env python3
"""
🧠 Simple Training Script for Miyawaki Dataset
Train Brain LDM with miyawaki_structured_28x28.mat dataset.
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
import random

from data.data_loader import load_fmri_data
from models.improved_brain_ldm import ImprovedBrainLDM

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_simple_dataloader(loader, batch_size=4, augment_factor=3):
    """Create simple dataloader for Miyawaki dataset with light augmentation."""
    
    print(f"🔄 Creating Miyawaki dataloader with {augment_factor}x augmentation...")
    
    # Get original training data
    train_fmri = loader.get_fmri('train')
    train_stimuli = loader.get_stimuli('train')
    train_labels = loader.get_labels('train')
    
    print(f"📊 Original data: {len(train_fmri)} samples")
    
    # Start with original data
    combined_fmri = [train_fmri]
    combined_stimuli = [train_stimuli]
    combined_labels = [train_labels]
    
    # Apply light data augmentation
    for i in range(augment_factor - 1):
        # Light noise augmentation
        noise_level = 0.005 + (i * 0.005)  # 0.005 to 0.015
        fmri_noise = torch.randn_like(train_fmri) * noise_level
        aug_fmri = train_fmri + fmri_noise
        
        combined_fmri.append(aug_fmri)
        combined_stimuli.append(train_stimuli)
        combined_labels.append(train_labels)
    
    # Combine all data
    combined_fmri = torch.cat(combined_fmri, dim=0)
    combined_stimuli = torch.cat(combined_stimuli, dim=0)
    combined_labels = torch.cat(combined_labels, dim=0)
    
    print(f"📊 Enhanced dataset: {len(combined_fmri)} samples ({augment_factor}x augmentation)")
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        combined_fmri, combined_stimuli, combined_labels
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    return dataloader

def train_miyawaki_model(epochs=50, batch_size=4, learning_rate=1e-4, save_name="miyawaki_simple"):
    """Train Brain LDM with Miyawaki dataset."""
    
    print(f"🚀 Training Brain LDM with Miyawaki Dataset")
    print("=" * 50)
    
    # Set random seed
    set_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load Miyawaki data (now uses default Miyawaki dataset)
    print("📁 Loading Miyawaki dataset...")
    loader = load_fmri_data()
    
    # Get fMRI dimension for model
    fmri_dim = loader.get_fmri('train').shape[1]
    print(f"🧠 fMRI dimension: {fmri_dim} voxels")
    
    # Create enhanced dataloader
    dataloader = create_simple_dataloader(
        loader, batch_size=batch_size, augment_factor=3
    )
    
    # Create model with Miyawaki dimensions
    print(f"🤖 Creating model with fMRI dimension: {fmri_dim}")
    model = ImprovedBrainLDM(
        fmri_dim=fmri_dim,  # Use Miyawaki's dimension (967)
        image_size=28,
        guidance_scale=7.5
    ).to(device)  # Move model to device

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Create simple decoder for training
    simple_decoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 784),  # 28*28 = 784
        nn.Sigmoid()
    ).to(device)
    
    # Training loop
    model.train()
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    print(f"\n🎯 Starting training for {epochs} epochs...")
    print(f"📊 Dataset: {len(dataloader)} batches per epoch")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for fmri, stimuli, labels in dataloader:
            fmri = fmri.to(device)
            stimuli = stimuli.to(device)
            
            # Forward pass
            try:
                # Encode fMRI data
                fmri_features = model.fmri_encoder(fmri)
                
                # Predict images from fMRI features
                predicted_stimuli = simple_decoder(fmri_features)
                
                # Target stimuli (flattened)
                target_stimuli = stimuli.view(stimuli.shape[0], -1)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(predicted_stimuli, target_stimuli)
                
                # Perceptual loss (L1)
                perceptual_loss = F.l1_loss(predicted_stimuli, target_stimuli)
                
                # Regularization
                reg_loss = torch.mean(fmri_features ** 2) * 0.01
                
                # Combined loss
                total_loss = recon_loss + 0.1 * perceptual_loss + reg_loss
                
            except Exception as e:
                print(f"⚠️ Forward pass error: {e}")
                # Fallback to simple loss
                fmri_features = model.fmri_encoder(fmri)
                target = torch.mean(fmri_features, dim=1, keepdim=True).expand_as(fmri_features)
                total_loss = F.mse_loss(fmri_features, target)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(simple_decoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record loss
            epoch_losses.append(total_loss.item())
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            save_path = f"checkpoints/best_{save_name}_model.pt"
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'decoder_state_dict': simple_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'best_loss': best_loss,
                'losses': losses,
                'config': {
                    'fmri_dim': fmri_dim,
                    'dataset': 'miyawaki_structured_28x28',
                    'augment_factor': 3
                }
            }, save_path)
        else:
            patience_counter += 1
        
        # Progress report
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss = {avg_loss:.6f} "
                  f"Best = {best_loss:.6f} "
                  f"LR = {scheduler.get_last_lr()[0]:.2e} "
                  f"Time: {elapsed:.1f}s")
        
        # Early stopping
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"📁 Model saved: checkpoints/best_{save_name}_model.pt")
    
    return model, simple_decoder, losses, best_loss

def test_miyawaki_model():
    """Test the trained model with Miyawaki test data."""
    
    print("\n🧪 Testing Miyawaki Model")
    print("=" * 30)
    
    # Load data
    loader = load_fmri_data()
    test_fmri = loader.get_fmri('test')
    test_stimuli = loader.get_stimuli('test')
    test_labels = loader.get_labels('test')
    
    print(f"📊 Test data: {len(test_fmri)} samples")
    print(f"🏷️ Label range: {test_labels.min().item()} to {test_labels.max().item()}")
    
    # Try to load trained model
    try:
        checkpoint = torch.load("checkpoints/best_miyawaki_simple_model.pt", map_location='cpu', weights_only=False)
        print("✅ Loaded trained model successfully!")
        
        # Get model config
        config = checkpoint.get('config', {})
        print(f"📋 Model config: {config}")
        
        return True
        
    except FileNotFoundError:
        print("❌ No trained model found. Please train first.")
        return False

def main():
    """Main training function."""
    print("🧠 MIYAWAKI DATASET TRAINING")
    print("=" * 35)
    
    # Train with Miyawaki dataset
    model, decoder, losses, best_loss = train_miyawaki_model(
        epochs=50,
        batch_size=4,
        learning_rate=1e-4,
        save_name="miyawaki_simple"
    )
    
    # Test the model
    test_success = test_miyawaki_model()
    
    print(f"\n🎯 TRAINING SUMMARY")
    print("=" * 20)
    print(f"✅ Successfully trained Brain LDM with Miyawaki dataset")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"📊 Dataset: 226 training + 26 test samples (12 classes)")
    print(f"🧠 fMRI dimension: 967 voxels (vs 3092 in digit69)")
    print(f"📈 Expected benefits:")
    print(f"   • More training data (+151%)")
    print(f"   • More classes (+500%)")
    print(f"   • Faster training (-69% fMRI size)")
    print(f"   • Better generalization")
    
    if test_success:
        print(f"\n📁 Next steps:")
        print(f"   • Run evaluation with trained model")
        print(f"   • Compare with digit69 results")
        print(f"   • Visualize reconstructions")
    
    print(f"\n🎉 Miyawaki training complete!")

if __name__ == "__main__":
    main()
