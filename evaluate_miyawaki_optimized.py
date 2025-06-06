#!/usr/bin/env python3
"""
🧠 Evaluate Optimized Miyawaki Model
Comprehensive evaluation of the optimized Miyawaki model with advanced metrics.
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr

from data.data_loader import load_fmri_data
from models.improved_brain_ldm import ImprovedBrainLDM
from train_miyawaki_optimized import AdvancedLossFunction

def calculate_advanced_metrics(targets, predictions):
    """Calculate comprehensive evaluation metrics."""
    
    # Flatten for correlation calculations
    targets_flat = targets.flatten()
    predictions_flat = predictions.flatten()
    
    # Basic metrics
    mse = np.mean((targets - predictions) ** 2)
    mae = np.mean(np.abs(targets - predictions))
    
    # Correlation metrics
    correlation, p_value = pearsonr(targets_flat, predictions_flat)
    
    # Structural similarity (simplified SSIM)
    def ssim_metric(img1, img2):
        mu1, mu2 = np.mean(img1), np.mean(img2)
        sigma1, sigma2 = np.std(img1), np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
        return ssim
    
    # Calculate SSIM for each sample
    ssim_scores = []
    for i in range(len(targets)):
        ssim = ssim_metric(targets[i], predictions[i])
        if not np.isnan(ssim):
            ssim_scores.append(ssim)
    
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    
    # Mutual Information (discretized)
    def discretize(data, bins=50):
        return np.digitize(data, np.linspace(data.min(), data.max(), bins))
    
    targets_discrete = discretize(targets_flat)
    predictions_discrete = discretize(predictions_flat)
    mutual_info = mutual_info_score(targets_discrete, predictions_discrete)
    
    # Peak Signal-to-Noise Ratio
    max_val = np.max(targets)
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'p_value': p_value,
        'ssim': avg_ssim,
        'mutual_info': mutual_info,
        'psnr': psnr
    }

def load_optimized_model(checkpoint_path, device='cuda'):
    """Load the optimized model with all components."""
    
    print(f"📁 Loading optimized model from: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Model file not found: {checkpoint_path}")
        return None, None, None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    fmri_dim = config.get('fmri_dim', 967)
    
    print(f"📋 Model config: {config}")
    
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
    
    # Create loss function
    criterion = AdvancedLossFunction(device=device).to(device)
    
    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if 'criterion_state_dict' in checkpoint:
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
    
    model.eval()
    decoder.eval()
    criterion.eval()
    
    print(f"✅ Optimized model loaded successfully!")
    return model, decoder, criterion

def evaluate_and_compare():
    """Evaluate optimized model and compare with baseline."""
    
    print("🧠 OPTIMIZED MIYAWAKI MODEL EVALUATION")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Load data
    loader = load_fmri_data("data/miyawaki_structured_28x28.mat", device=device)
    
    # Load optimized model
    opt_model, opt_decoder, opt_criterion = load_optimized_model(
        "checkpoints/best_miyawaki_optimized_model.pt", device
    )
    
    if opt_model is None:
        print("❌ Please train the optimized model first:")
        print("   PYTHONPATH=src python3 train_miyawaki_optimized.py")
        return
    
    # Load baseline model for comparison
    baseline_checkpoint_path = "checkpoints/best_miyawaki_simple_model.pt"
    baseline_model = None
    baseline_decoder = None
    
    if Path(baseline_checkpoint_path).exists():
        print(f"📁 Loading baseline model for comparison...")
        baseline_checkpoint = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)
        
        baseline_model = ImprovedBrainLDM(
            fmri_dim=967, image_size=28, guidance_scale=7.5
        ).to(device)
        
        baseline_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        ).to(device)
        
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
        baseline_decoder.load_state_dict(baseline_checkpoint['decoder_state_dict'])
        baseline_model.eval()
        baseline_decoder.eval()
        
        print(f"✅ Baseline model loaded for comparison")
    
    # Get test data
    test_fmri = loader.get_fmri('test')[:8].to(device)
    test_stimuli = loader.get_stimuli('test')[:8]
    test_labels = loader.get_labels('test')[:8]
    
    print(f"\n🧪 Evaluating on {len(test_fmri)} test samples...")
    
    # Generate reconstructions - Optimized model
    with torch.no_grad():
        opt_fmri_features = opt_model.fmri_encoder(test_fmri)
        opt_reconstructions = opt_decoder(opt_fmri_features)
        
        opt_targets = test_stimuli.cpu().numpy()
        opt_recons = opt_reconstructions.cpu().numpy()
        labels = test_labels.cpu().numpy()
    
    # Generate reconstructions - Baseline model (if available)
    baseline_recons = None
    if baseline_model is not None:
        with torch.no_grad():
            baseline_fmri_features = baseline_model.fmri_encoder(test_fmri)
            baseline_reconstructions = baseline_decoder(baseline_fmri_features)
            baseline_recons = baseline_reconstructions.cpu().numpy()
    
    # Calculate metrics
    print(f"\n📊 Calculating advanced metrics...")
    opt_metrics = calculate_advanced_metrics(opt_targets, opt_recons)
    
    baseline_metrics = None
    if baseline_recons is not None:
        baseline_metrics = calculate_advanced_metrics(opt_targets, baseline_recons)
    
    # Print results
    print(f"\n📈 EVALUATION RESULTS")
    print("=" * 30)
    
    print(f"🎯 Optimized Model:")
    print(f"   MSE: {opt_metrics['mse']:.6f}")
    print(f"   MAE: {opt_metrics['mae']:.6f}")
    print(f"   Correlation: {opt_metrics['correlation']:.6f} (p={opt_metrics['p_value']:.4f})")
    print(f"   SSIM: {opt_metrics['ssim']:.6f}")
    print(f"   Mutual Info: {opt_metrics['mutual_info']:.6f}")
    print(f"   PSNR: {opt_metrics['psnr']:.2f} dB")
    
    if baseline_metrics is not None:
        print(f"\n🔄 Baseline Model:")
        print(f"   MSE: {baseline_metrics['mse']:.6f}")
        print(f"   MAE: {baseline_metrics['mae']:.6f}")
        print(f"   Correlation: {baseline_metrics['correlation']:.6f}")
        print(f"   SSIM: {baseline_metrics['ssim']:.6f}")
        print(f"   Mutual Info: {baseline_metrics['mutual_info']:.6f}")
        print(f"   PSNR: {baseline_metrics['psnr']:.2f} dB")
        
        print(f"\n📊 IMPROVEMENT ANALYSIS:")
        mse_improvement = ((baseline_metrics['mse'] - opt_metrics['mse']) / baseline_metrics['mse']) * 100
        corr_improvement = ((opt_metrics['correlation'] - baseline_metrics['correlation']) / abs(baseline_metrics['correlation'])) * 100
        ssim_improvement = ((opt_metrics['ssim'] - baseline_metrics['ssim']) / abs(baseline_metrics['ssim'])) * 100
        
        print(f"   MSE: {mse_improvement:+.1f}% {'✅' if mse_improvement > 0 else '❌'}")
        print(f"   Correlation: {corr_improvement:+.1f}% {'✅' if corr_improvement > 0 else '❌'}")
        print(f"   SSIM: {ssim_improvement:+.1f}% {'✅' if ssim_improvement > 0 else '❌'}")
    
    # Create comprehensive visualization
    create_comparison_visualization(
        opt_targets, opt_recons, baseline_recons, labels, 
        opt_metrics, baseline_metrics
    )
    
    return opt_metrics, baseline_metrics

def create_comparison_visualization(targets, opt_recons, baseline_recons, labels, 
                                  opt_metrics, baseline_metrics):
    """Create comprehensive comparison visualization."""
    
    print(f"\n🎨 Creating comprehensive visualization...")
    
    num_samples = len(targets)
    
    # Apply transformations
    targets_transformed = np.array([np.rot90(np.flipud(img.reshape(28, 28)), k=-1) for img in targets])
    opt_recons_transformed = np.array([np.rot90(np.flipud(img.reshape(28, 28)), k=-1) for img in opt_recons])
    
    if baseline_recons is not None:
        baseline_recons_transformed = np.array([np.rot90(np.flipud(img.reshape(28, 28)), k=-1) for img in baseline_recons])
        rows = 4
    else:
        rows = 3
    
    fig, axes = plt.subplots(rows, num_samples, figsize=(num_samples * 2.5, rows * 2.5))
    fig.suptitle('Optimized vs Baseline Miyawaki Reconstruction Comparison', 
                 fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # Target
        axes[0, i].imshow(targets_transformed[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Target {i+1}\nLabel: {labels[i]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Optimized reconstruction
        axes[1, i].imshow(opt_recons_transformed[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Optimized {i+1}', fontsize=10)
        axes[1, i].axis('off')
        
        # Baseline reconstruction (if available)
        if baseline_recons is not None:
            axes[2, i].imshow(baseline_recons_transformed[i], cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title(f'Baseline {i+1}', fontsize=10)
            axes[2, i].axis('off')
            
            # Difference between optimized and baseline
            diff = np.abs(opt_recons_transformed[i] - baseline_recons_transformed[i])
            im = axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[3, i].set_title(f'Opt-Base Diff {i+1}', fontsize=10)
            axes[3, i].axis('off')
        else:
            # Difference between optimized and target
            diff = np.abs(opt_recons_transformed[i] - targets_transformed[i])
            im = axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'Opt-Target Diff {i+1}', fontsize=10)
            axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.15, 0.5, 'Target\nStimulus', rotation=90, va='center', ha='center', 
                    transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
    axes[1, 0].text(-0.15, 0.5, 'Optimized\nReconstruction', rotation=90, va='center', ha='center', 
                    transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
    
    if baseline_recons is not None:
        axes[2, 0].text(-0.15, 0.5, 'Baseline\nReconstruction', rotation=90, va='center', ha='center', 
                        transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
        axes[3, 0].text(-0.15, 0.5, 'Optimization\nImprovement', rotation=90, va='center', ha='center', 
                        transform=axes[3, 0].transAxes, fontsize=12, fontweight='bold')
    else:
        axes[2, 0].text(-0.15, 0.5, 'Reconstruction\nError', rotation=90, va='center', ha='center', 
                        transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = "results/miyawaki_optimization_comparison.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Comparison saved: {output_path}")
    
    plt.show()

def main():
    """Main evaluation function."""
    
    opt_metrics, baseline_metrics = evaluate_and_compare()
    
    print(f"\n🎯 OPTIMIZATION EXPERIMENT SUMMARY")
    print("=" * 40)
    
    if baseline_metrics is not None:
        print(f"📈 Key Improvements:")
        
        improvements = []
        if opt_metrics['mse'] < baseline_metrics['mse']:
            improvements.append(f"✅ MSE reduced by {((baseline_metrics['mse'] - opt_metrics['mse']) / baseline_metrics['mse'] * 100):.1f}%")
        
        if opt_metrics['correlation'] > baseline_metrics['correlation']:
            improvements.append(f"✅ Correlation improved by {((opt_metrics['correlation'] - baseline_metrics['correlation']) / abs(baseline_metrics['correlation']) * 100):.1f}%")
        
        if opt_metrics['ssim'] > baseline_metrics['ssim']:
            improvements.append(f"✅ SSIM improved by {((opt_metrics['ssim'] - baseline_metrics['ssim']) / abs(baseline_metrics['ssim']) * 100):.1f}%")
        
        if improvements:
            for improvement in improvements:
                print(f"   {improvement}")
        else:
            print(f"   ⚠️ No significant improvements detected")
            print(f"   💡 Consider further hyperparameter tuning")
    
    print(f"\n📁 Results saved to: results/miyawaki_optimization_comparison.png")

if __name__ == "__main__":
    main()
