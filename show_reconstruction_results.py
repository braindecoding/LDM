#!/usr/bin/env python3
"""
🧠 Show Reconstruction Results - Target vs Reconstruction
Simple script to visualize Brain LDM reconstruction results with GPU support.
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.data_loader import load_fmri_data
from models.improved_brain_ldm import ImprovedBrainLDM

def main():
    """Main function to show reconstruction results."""
    
    print("🧠 BRAIN LDM RECONSTRUCTION VISUALIZATION")
    print("=" * 50)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print(f"\n📁 Loading Miyawaki dataset...")
    loader = load_fmri_data("data/miyawaki_structured_28x28.mat", device=device)
    
    # Load trained model
    checkpoint_path = "checkpoints/best_miyawaki_simple_model.pt"
    print(f"📁 Loading trained model from: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Model file not found: {checkpoint_path}")
        print("💡 Please train the model first using:")
        print("   PYTHONPATH=src python3 train_miyawaki_simple.py")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    fmri_dim = config.get('fmri_dim', 967)
    
    print(f"📋 Model config: {config}")
    print(f"🏆 Best training loss: {checkpoint.get('best_loss', 'N/A')}")
    
    # Create model
    model = ImprovedBrainLDM(
        fmri_dim=fmri_dim,
        image_size=28,
        guidance_scale=7.5
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create decoder
    decoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 784),
        nn.Sigmoid()
    ).to(device)
    
    # Load decoder weights if available
    if 'decoder_state_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    
    print(f"✅ Model loaded successfully!")
    
    # Get test data
    print(f"\n🧪 Generating reconstructions...")
    test_fmri = loader.get_fmri('test')[:8].to(device)
    test_stimuli = loader.get_stimuli('test')[:8]
    test_labels = loader.get_labels('test')[:8]
    
    print(f"📊 Processing {len(test_fmri)} test samples")
    print(f"🏷️ Labels: {test_labels.cpu().numpy()}")
    
    # Generate reconstructions
    with torch.no_grad():
        # Encode fMRI to features
        fmri_features = model.fmri_encoder(test_fmri)
        print(f"📐 fMRI features shape: {fmri_features.shape}")
        
        # Decode to images
        reconstructions = decoder(fmri_features)
        print(f"📐 Reconstructions shape: {reconstructions.shape}")
        
        # Move to CPU for visualization
        targets = test_stimuli.view(-1, 28, 28).cpu().numpy()
        recons = reconstructions.view(-1, 28, 28).cpu().numpy()
        labels = test_labels.cpu().numpy()
    
    # Apply transformations (flip and rotate -90 degrees as per user preference)
    targets_transformed = np.array([np.rot90(np.flipud(img), k=-1) for img in targets])
    recons_transformed = np.array([np.rot90(np.flipud(img), k=-1) for img in recons])
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    num_samples = len(targets)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2.5, 8))
    fig.suptitle('Brain LDM: fMRI → Image Reconstruction (Miyawaki Dataset)', 
                 fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # Target stimulus (transformed)
        axes[0, i].imshow(targets_transformed[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Stimulus {i+1}\nLabel: {labels[i]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstruction (transformed)
        axes[1, i].imshow(recons_transformed[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Reconstruction {i+1}', fontsize=10)
        axes[1, i].axis('off')
        
        # Difference (transformed)
        diff = np.abs(targets_transformed[i] - recons_transformed[i])
        im = axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title(f'Difference {i+1}', fontsize=10)
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.15, 0.5, 'Target\nStimulus', rotation=90, va='center', ha='center', 
                    transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
    axes[1, 0].text(-0.15, 0.5, 'fMRI\nReconstruction', rotation=90, va='center', ha='center', 
                    transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
    axes[2, 0].text(-0.15, 0.5, 'Absolute\nDifference', rotation=90, va='center', ha='center', 
                    transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
    
    # Add colorbar for difference
    cbar = plt.colorbar(im, ax=axes[2, :], orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Difference Magnitude', fontsize=10)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = "results/brain_ldm_reconstruction_results.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved: {output_path}")
    
    # Calculate metrics
    mse = np.mean((targets - recons) ** 2)
    mae = np.mean(np.abs(targets - recons))
    correlation = np.corrcoef(targets.flatten(), recons.flatten())[0, 1]
    
    print(f"\n📊 Reconstruction Quality Metrics:")
    print(f"   MSE (Mean Squared Error): {mse:.6f}")
    print(f"   MAE (Mean Absolute Error): {mae:.6f}")
    print(f"   Correlation: {correlation:.6f}")
    
    # Individual sample metrics
    print(f"\n📋 Individual Sample Metrics:")
    for i in range(num_samples):
        sample_mse = np.mean((targets[i] - recons[i]) ** 2)
        sample_corr = np.corrcoef(targets[i].flatten(), recons[i].flatten())[0, 1]
        print(f"   Sample {i+1} (Label {labels[i]}): MSE={sample_mse:.6f}, Corr={sample_corr:.6f}")
    
    plt.show()
    
    print(f"\n🎉 Reconstruction visualization complete!")
    print(f"📁 Results saved to: {output_path}")
    print(f"🔍 The visualization shows:")
    print(f"   • Top row: Original stimulus images")
    print(f"   • Middle row: Brain LDM reconstructions from fMRI")
    print(f"   • Bottom row: Absolute differences (red = high difference)")

if __name__ == "__main__":
    main()
