"""
📊 Simple Metrics Evaluation

Calculate PSNR, SSIM, and other metrics for Brain LDM results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from data_loader import load_fmri_data
from brain_ldm import create_brain_ldm


def compute_psnr(true_images, pred_images):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((true_images - pred_images) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming images are normalized to [0,1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(true_images, pred_images):
    """Compute Structural Similarity Index (simplified)."""
    def ssim_single(img1, img2):
        # Convert to float
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Compute means
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        # Compute variances and covariance
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        # SSIM constants
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        
        # SSIM formula
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    ssim_values = []
    for i in range(len(true_images)):
        ssim_val = ssim_single(true_images[i], pred_images[i])
        if not np.isnan(ssim_val):
            ssim_values.append(ssim_val)
    
    return np.mean(ssim_values) if ssim_values else 0.0


def compute_correlation(true_images, pred_images):
    """Compute pixel-wise correlation."""
    correlations = []
    for i in range(len(true_images)):
        true_flat = true_images[i].flatten()
        pred_flat = pred_images[i].flatten()
        
        # Compute correlation coefficient
        corr_matrix = np.corrcoef(true_flat, pred_flat)
        if corr_matrix.shape == (2, 2):
            corr = corr_matrix[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0


def evaluate_model():
    """Evaluate the trained model."""
    print("📊 Simple Metrics Evaluation")
    print("=" * 40)
    
    # Check if model exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"❌ Model not found: {checkpoint_path}")
        return None
    
    # Load model
    device = 'cpu'
    print(f"📁 Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    model = create_brain_ldm(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    
    # Load data
    data_loader = load_fmri_data(device=device)
    test_loader = data_loader.create_dataloader('test', batch_size=4, shuffle=False)
    
    print(f"📊 Generating reconstructions...")
    
    # Generate reconstructions
    all_true_images = []
    all_pred_images = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            fmri_signals = batch['fmri'].to(device)
            true_stimuli = batch['stimulus'].to(device)
            
            # Generate reconstructions
            generated_images = model.generate_from_fmri(
                fmri_signals, 
                num_inference_steps=20  # Faster generation
            )
            
            # Convert to numpy
            true_imgs = true_stimuli.view(-1, 28, 28).cpu().numpy()
            pred_imgs = generated_images[:, 0].cpu().numpy()
            
            all_true_images.append(true_imgs)
            all_pred_images.append(pred_imgs)
            
            print(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")
    
    # Concatenate results
    true_images = np.concatenate(all_true_images, axis=0)
    pred_images = np.concatenate(all_pred_images, axis=0)
    
    print(f"📈 Computing metrics for {len(true_images)} samples...")
    
    # Compute metrics
    mse = np.mean((true_images - pred_images) ** 2)
    mae = np.mean(np.abs(true_images - pred_images))
    rmse = np.sqrt(mse)
    
    print(f"  🔍 Computing PSNR...")
    psnr = compute_psnr(true_images, pred_images)
    
    print(f"  🔍 Computing SSIM...")
    ssim = compute_ssim(true_images, pred_images)
    
    print(f"  🔍 Computing Correlation...")
    correlation = compute_correlation(true_images, pred_images)
    
    # Results
    results = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'psnr': psnr,
        'ssim': ssim,
        'correlation': correlation,
        'num_samples': len(true_images)
    }
    
    return results, true_images, pred_images


def create_comparison_visualization(true_images, pred_images, results):
    """Create detailed comparison visualization."""
    print(f"🎨 Creating comparison visualization...")
    
    # Select 8 samples for visualization
    num_samples = min(8, len(true_images))
    indices = np.linspace(0, len(true_images)-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 8))
    
    for i, idx in enumerate(indices):
        # True image
        axes[0, i].imshow(true_images[idx], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'True {idx}')
        axes[0, i].axis('off')
        
        # Predicted image
        axes[1, i].imshow(pred_images[idx], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Pred {idx}')
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(true_images[idx] - pred_images[idx])
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title(f'Diff {idx}')
        axes[2, i].axis('off')
    
    # Add metrics as title
    title = f'Brain LDM Results - PSNR: {results["psnr"]:.2f}dB, SSIM: {results["ssim"]:.3f}, Corr: {results["correlation"]:.3f}'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Save
    os.makedirs("results/comprehensive", exist_ok=True)
    save_path = "results/comprehensive/detailed_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Comparison saved to: {save_path}")
    return save_path


def main():
    """Main evaluation function."""
    print("📊 Brain LDM Metrics Evaluation")
    print("=" * 50)
    
    try:
        # Evaluate model
        results, true_images, pred_images = evaluate_model()
        
        if results is None:
            return
        
        # Print results
        print(f"\n📊 Evaluation Results")
        print("=" * 30)
        print(f"🎯 Image Quality Metrics:")
        print(f"  PSNR: {results['psnr']:.4f} dB")
        print(f"  SSIM: {results['ssim']:.4f}")
        print(f"  Correlation: {results['correlation']:.4f}")
        
        print(f"\n📈 Basic Metrics:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  MAE: {results['mae']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
        
        print(f"\n📋 Evaluation Info:")
        print(f"  Samples: {results['num_samples']}")
        print(f"  Image size: 28×28")
        
        # Create visualization
        comparison_path = create_comparison_visualization(true_images, pred_images, results)
        
        # Save detailed results
        os.makedirs("results/comprehensive", exist_ok=True)
        
        results_text = f"""Brain LDM Evaluation Results
===============================

Image Quality Metrics:
  PSNR: {results['psnr']:.4f} dB
  SSIM: {results['ssim']:.4f}
  Correlation: {results['correlation']:.4f}

Basic Metrics:
  MSE: {results['mse']:.6f}
  MAE: {results['mae']:.6f}
  RMSE: {results['rmse']:.6f}

Evaluation Info:
  Samples: {results['num_samples']}
  Image size: 28×28
  Model: Brain LDM

Metric Interpretations:
  PSNR: Higher is better (>20 dB is good, >30 dB is excellent)
  SSIM: Higher is better (0-1 scale, >0.5 is good, >0.8 is excellent)
  Correlation: Higher is better (-1 to 1 scale, >0.5 is good)
  MSE/MAE/RMSE: Lower is better

Performance Assessment:
  PSNR {results['psnr']:.1f} dB: {'Excellent' if results['psnr'] > 30 else 'Good' if results['psnr'] > 20 else 'Fair' if results['psnr'] > 15 else 'Poor'}
  SSIM {results['ssim']:.3f}: {'Excellent' if results['ssim'] > 0.8 else 'Good' if results['ssim'] > 0.5 else 'Fair' if results['ssim'] > 0.3 else 'Poor'}
  Correlation {results['correlation']:.3f}: {'Excellent' if results['correlation'] > 0.7 else 'Good' if results['correlation'] > 0.5 else 'Fair' if results['correlation'] > 0.3 else 'Poor'}
"""
        
        with open("results/comprehensive/metrics_analysis.txt", 'w') as f:
            f.write(results_text)
        
        print(f"\n💾 Detailed analysis saved to: results/comprehensive/metrics_analysis.txt")
        print(f"🎨 Comparison visualization: {comparison_path}")
        print(f"✅ Comprehensive evaluation completed!")
        
        # Performance summary
        print(f"\n🎯 Performance Summary:")
        psnr_rating = 'Excellent' if results['psnr'] > 30 else 'Good' if results['psnr'] > 20 else 'Fair' if results['psnr'] > 15 else 'Poor'
        ssim_rating = 'Excellent' if results['ssim'] > 0.8 else 'Good' if results['ssim'] > 0.5 else 'Fair' if results['ssim'] > 0.3 else 'Poor'
        corr_rating = 'Excellent' if results['correlation'] > 0.7 else 'Good' if results['correlation'] > 0.5 else 'Fair' if results['correlation'] > 0.3 else 'Poor'
        
        print(f"  PSNR: {psnr_rating} ({results['psnr']:.1f} dB)")
        print(f"  SSIM: {ssim_rating} ({results['ssim']:.3f})")
        print(f"  Correlation: {corr_rating} ({results['correlation']:.3f})")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
