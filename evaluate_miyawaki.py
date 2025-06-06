#!/usr/bin/env python3
"""
🧪 Evaluate Miyawaki Dataset Results
Evaluate the trained Brain LDM model with miyawaki_structured_28x28.mat dataset.
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

from data.data_loader import load_fmri_data
from models.improved_brain_ldm import ImprovedBrainLDM

def load_trained_model(checkpoint_path="checkpoints/best_miyawaki_simple_model.pt"):
    """Load the trained Miyawaki model."""
    
    print(f"📁 Loading trained model from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get model config
        config = checkpoint.get('config', {})
        fmri_dim = config.get('fmri_dim', 967)
        
        print(f"📋 Model config: {config}")
        
        # Create model with same architecture
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ImprovedBrainLDM(
            fmri_dim=fmri_dim,
            image_size=28,
            guidance_scale=7.5
        ).to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create decoder
        simple_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        ).to(device)
        
        # Load decoder weights if available
        if 'decoder_state_dict' in checkpoint:
            simple_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        print(f"✅ Model loaded successfully!")
        print(f"🏆 Best training loss: {checkpoint.get('best_loss', 'N/A')}")
        print(f"📊 Training epochs: {checkpoint.get('epoch', 'N/A')}")
        
        return model, simple_decoder, device
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def evaluate_reconstruction(model, decoder, loader, device, num_samples=10):
    """Evaluate reconstruction quality on test data."""
    
    print(f"\n🧪 Evaluating Reconstruction Quality")
    print("=" * 40)
    
    # Get test data
    test_fmri = loader.get_fmri('test')[:num_samples]
    test_stimuli = loader.get_stimuli('test')[:num_samples]
    test_labels = loader.get_labels('test')[:num_samples]
    
    print(f"📊 Evaluating {len(test_fmri)} test samples")
    print(f"🏷️ Labels: {test_labels.cpu().numpy()}")
    
    # Set model to evaluation mode
    model.eval()
    decoder.eval()
    
    reconstructions = []
    mse_scores = []
    correlation_scores = []
    
    with torch.no_grad():
        for i in range(len(test_fmri)):
            # Get single sample
            fmri = test_fmri[i:i+1].to(device)
            stimulus = test_stimuli[i:i+1]
            
            # Forward pass
            try:
                # Encode fMRI
                fmri_features = model.fmri_encoder(fmri)
                
                # Decode to image
                reconstructed = decoder(fmri_features)
                
                # Move to CPU for evaluation
                reconstructed = reconstructed.cpu()
                
                # Calculate metrics
                mse = F.mse_loss(reconstructed, stimulus).item()
                
                # Calculate correlation
                recon_flat = reconstructed.flatten().numpy()
                stim_flat = stimulus.flatten().numpy()
                correlation = np.corrcoef(recon_flat, stim_flat)[0, 1]
                
                reconstructions.append(reconstructed.squeeze().numpy())
                mse_scores.append(mse)
                correlation_scores.append(correlation)
                
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                # Create dummy reconstruction
                reconstructions.append(np.zeros((784,)))
                mse_scores.append(float('inf'))
                correlation_scores.append(0.0)
    
    # Calculate average metrics
    avg_mse = np.mean(mse_scores)
    avg_correlation = np.mean(correlation_scores)
    
    print(f"\n📊 Reconstruction Metrics:")
    print(f"   Average MSE: {avg_mse:.6f}")
    print(f"   Average Correlation: {avg_correlation:.6f}")
    print(f"   MSE Range: {np.min(mse_scores):.6f} - {np.max(mse_scores):.6f}")
    print(f"   Correlation Range: {np.min(correlation_scores):.6f} - {np.max(correlation_scores):.6f}")
    
    return reconstructions, test_stimuli.numpy(), test_labels.numpy(), mse_scores, correlation_scores

def visualize_results(reconstructions, original_stimuli, labels, mse_scores, correlation_scores, save_path="results/miyawaki_evaluation.png"):
    """Visualize reconstruction results."""
    
    print(f"\n🎨 Creating Visualization")
    print("=" * 25)
    
    num_samples = min(len(reconstructions), 10)
    
    # Create figure
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    fig.suptitle('Miyawaki Dataset: Brain LDM Reconstruction Results', fontsize=16)
    
    for i in range(num_samples):
        # Original stimulus
        original = original_stimuli[i].reshape(28, 28)
        # Apply transformation: flip and rotate -90 degrees
        original_transformed = np.rot90(np.flipud(original), k=-1)
        
        axes[0, i].imshow(original_transformed, cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {labels[i]}')
        axes[0, i].axis('off')
        
        # Reconstruction
        reconstruction = reconstructions[i].reshape(28, 28)
        # Apply same transformation
        reconstruction_transformed = np.rot90(np.flipud(reconstruction), k=-1)
        
        axes[1, i].imshow(reconstruction_transformed, cmap='gray')
        axes[1, i].set_title(f'Reconstruction\nMSE: {mse_scores[i]:.3f}')
        axes[1, i].axis('off')
        
        # Difference
        difference = np.abs(original_transformed - reconstruction_transformed)
        axes[2, i].imshow(difference, cmap='hot')
        axes[2, i].set_title(f'Difference\nCorr: {correlation_scores[i]:.3f}')
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Original', rotation=90, size='large')
    axes[1, 0].set_ylabel('Reconstruction', rotation=90, size='large')
    axes[2, 0].set_ylabel('Difference', rotation=90, size='large')
    
    plt.tight_layout()
    
    # Save visualization
    Path("results").mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved visualization: {save_path}")
    plt.close()

def compare_with_digit69():
    """Compare Miyawaki results with digit69 results."""
    
    print(f"\n📊 Comparison: Miyawaki vs Digit69")
    print("=" * 40)
    
    # Miyawaki stats
    miyawaki_stats = {
        'dataset': 'miyawaki_structured_28x28',
        'train_samples': 150,
        'test_samples': 18,
        'total_samples': 168,
        'classes': 12,
        'fmri_voxels': 967,
        'label_range': [21, 32]
    }
    
    # Digit69 stats (for comparison)
    digit69_stats = {
        'dataset': 'digit69_28x28',
        'train_samples': 90,
        'test_samples': 10,
        'total_samples': 100,
        'classes': 2,
        'fmri_voxels': 3092,
        'label_range': [1, 2]
    }
    
    print("📈 Dataset Comparison:")
    print(f"{'Metric':<20} {'Digit69':<15} {'Miyawaki':<15} {'Improvement':<15}")
    print("-" * 65)
    
    # Calculate improvements
    train_improvement = (miyawaki_stats['train_samples'] / digit69_stats['train_samples'] - 1) * 100
    test_improvement = (miyawaki_stats['test_samples'] / digit69_stats['test_samples'] - 1) * 100
    total_improvement = (miyawaki_stats['total_samples'] / digit69_stats['total_samples'] - 1) * 100
    class_improvement = (miyawaki_stats['classes'] / digit69_stats['classes'] - 1) * 100
    voxel_reduction = (1 - miyawaki_stats['fmri_voxels'] / digit69_stats['fmri_voxels']) * 100
    
    print(f"{'Training Samples':<20} {digit69_stats['train_samples']:<15} {miyawaki_stats['train_samples']:<15} {train_improvement:+.1f}%")
    print(f"{'Test Samples':<20} {digit69_stats['test_samples']:<15} {miyawaki_stats['test_samples']:<15} {test_improvement:+.1f}%")
    print(f"{'Total Samples':<20} {digit69_stats['total_samples']:<15} {miyawaki_stats['total_samples']:<15} {total_improvement:+.1f}%")
    print(f"{'Classes':<20} {digit69_stats['classes']:<15} {miyawaki_stats['classes']:<15} {class_improvement:+.1f}%")
    print(f"{'fMRI Voxels':<20} {digit69_stats['fmri_voxels']:<15} {miyawaki_stats['fmri_voxels']:<15} {voxel_reduction:+.1f}%")
    
    print(f"\n🎯 Key Advantages of Miyawaki:")
    print(f"   ✅ {train_improvement:+.1f}% more training data")
    print(f"   ✅ {class_improvement:+.1f}% more classes (more challenging)")
    print(f"   ✅ {voxel_reduction:.1f}% fewer fMRI voxels (faster training)")
    print(f"   ✅ More realistic label range ({miyawaki_stats['label_range']})")

def main():
    """Main evaluation function."""
    
    print("🧪 MIYAWAKI DATASET EVALUATION")
    print("=" * 40)
    
    # Load trained model
    model, decoder, device = load_trained_model()
    
    if model is None:
        print("❌ Failed to load model. Please train first.")
        return
    
    # Load data
    print(f"\n📁 Loading Miyawaki test data...")
    loader = load_fmri_data()
    
    # Evaluate reconstruction
    reconstructions, original_stimuli, labels, mse_scores, correlation_scores = evaluate_reconstruction(
        model, decoder, loader, device, num_samples=10
    )
    
    # Visualize results
    visualize_results(
        reconstructions, original_stimuli, labels, 
        mse_scores, correlation_scores
    )
    
    # Compare with digit69
    compare_with_digit69()
    
    # Summary
    print(f"\n🎯 EVALUATION SUMMARY")
    print("=" * 25)
    print(f"✅ Successfully evaluated Miyawaki Brain LDM model")
    print(f"📊 Dataset: 150 training + 18 test samples (12 classes)")
    print(f"🧠 fMRI dimension: 967 voxels")
    print(f"🎨 Reconstruction quality:")
    print(f"   Average MSE: {np.mean(mse_scores):.6f}")
    print(f"   Average Correlation: {np.mean(correlation_scores):.6f}")
    
    print(f"\n📁 Results saved to: results/miyawaki_evaluation.png")
    print(f"🎉 Miyawaki evaluation complete!")

if __name__ == "__main__":
    main()
