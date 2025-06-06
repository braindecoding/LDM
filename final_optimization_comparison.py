#!/usr/bin/env python3
"""
🧠 Final Optimization Comparison - Complete Analysis
Compare all models: Baseline vs Optimized for both Miyawaki and Vangerven datasets.
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

def calculate_metrics(targets, predictions):
    """Calculate comprehensive evaluation metrics."""
    
    targets_flat = targets.flatten()
    predictions_flat = predictions.flatten()
    
    mse = np.mean((targets - predictions) ** 2)
    mae = np.mean(np.abs(targets - predictions))
    correlation, p_value = pearsonr(targets_flat, predictions_flat)
    
    # SSIM
    def ssim_metric(img1, img2):
        mu1, mu2 = np.mean(img1), np.mean(img2)
        sigma1, sigma2 = np.std(img1), np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
        return ssim
    
    ssim_scores = []
    for i in range(len(targets)):
        ssim = ssim_metric(targets[i], predictions[i])
        if not np.isnan(ssim):
            ssim_scores.append(ssim)
    
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    
    # Mutual Information
    def discretize(data, bins=50):
        return np.digitize(data, np.linspace(data.min(), data.max(), bins))
    
    targets_discrete = discretize(targets_flat)
    predictions_discrete = discretize(predictions_flat)
    mutual_info = mutual_info_score(targets_discrete, predictions_discrete)
    
    # PSNR
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

def load_and_evaluate_model(checkpoint_path, data_path, model_name, device='cuda'):
    """Load model and evaluate."""
    
    print(f"\n📊 Evaluating {model_name}")
    print("=" * 40)
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Model not found: {checkpoint_path}")
        return None
    
    # Load data
    loader = load_fmri_data(data_path, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    fmri_dim = config.get('fmri_dim', 967)
    
    # Create model
    model = ImprovedBrainLDM(
        fmri_dim=fmri_dim,
        image_size=28,
        guidance_scale=7.5
    ).to(device)
    
    # Create decoder
    if 'optimized' in model_name.lower():
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
    else:
        decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        ).to(device)
    
    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()
    decoder.eval()
    
    # Get test data
    test_fmri = loader.get_fmri('test')[:8].to(device)
    test_stimuli = loader.get_stimuli('test')[:8]
    
    # Generate reconstructions
    with torch.no_grad():
        fmri_features = model.fmri_encoder(test_fmri)
        reconstructions = decoder(fmri_features)
        
        targets = test_stimuli.cpu().numpy()
        recons = reconstructions.cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(targets, recons)
    
    results = {
        'model_name': model_name,
        'training_loss': checkpoint.get('best_loss', 'N/A'),
        'fmri_dim': fmri_dim,
        'targets': targets,
        'reconstructions': recons,
        **metrics
    }
    
    print(f"✅ {model_name} evaluation complete")
    print(f"   Training Loss: {results['training_loss']}")
    print(f"   MSE: {results['mse']:.6f}")
    print(f"   Correlation: {results['correlation']:.6f}")
    print(f"   SSIM: {results['ssim']:.6f}")
    
    return results

def create_final_comparison_visualization(all_results):
    """Create comprehensive final comparison."""
    
    print(f"\n🎨 Creating Final Comparison Visualization")
    print("=" * 50)
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Training Loss Comparison
    ax1 = plt.subplot(3, 4, 1)
    models = [r['model_name'] for r in all_results]
    losses = [r['training_loss'] for r in all_results]
    colors = ['red', 'darkred', 'blue', 'darkblue']
    
    bars = ax1.bar(range(len(models)), losses, color=colors, alpha=0.7)
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=0, fontsize=9)
    
    # Add value labels
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 2. MSE Comparison
    ax2 = plt.subplot(3, 4, 2)
    mse_values = [r['mse'] for r in all_results]
    bars = ax2.bar(range(len(models)), mse_values, color=colors, alpha=0.7)
    ax2.set_ylabel('MSE')
    ax2.set_title('Mean Squared Error')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=0, fontsize=9)
    
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Correlation Comparison
    ax3 = plt.subplot(3, 4, 3)
    corr_values = [r['correlation'] for r in all_results]
    bars = ax3.bar(range(len(models)), corr_values, color=colors, alpha=0.7)
    ax3.set_ylabel('Correlation')
    ax3.set_title('Reconstruction Correlation')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=0, fontsize=9)
    
    for bar, corr in zip(bars, corr_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. SSIM Comparison
    ax4 = plt.subplot(3, 4, 4)
    ssim_values = [r['ssim'] for r in all_results]
    bars = ax4.bar(range(len(models)), ssim_values, color=colors, alpha=0.7)
    ax4.set_ylabel('SSIM')
    ax4.set_title('Structural Similarity')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=0, fontsize=9)
    
    for bar, ssim in zip(bars, ssim_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{ssim:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 5-8. Sample Reconstructions
    for i, result in enumerate(all_results):
        ax = plt.subplot(3, 4, 5 + i)
        
        # Show first sample: target | reconstruction
        target = result['targets'][0].reshape(28, 28)
        recon = result['reconstructions'][0].reshape(28, 28)
        
        # Apply transformations
        target = np.rot90(np.flipud(target), k=-1)
        recon = np.rot90(np.flipud(recon), k=-1)
        
        combined = np.hstack([target, recon])
        ax.imshow(combined, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{result["model_name"]}\nTarget | Reconstruction', fontsize=10)
        ax.axis('off')
    
    # 9-12. Improvement Analysis
    miyawaki_baseline = all_results[0]
    miyawaki_optimized = all_results[1]
    vangerven_baseline = all_results[2]
    vangerven_optimized = all_results[3]
    
    # Miyawaki improvement
    ax9 = plt.subplot(3, 4, 9)
    metrics = ['MSE', 'Correlation', 'SSIM']
    miyawaki_improvements = [
        ((miyawaki_baseline['mse'] - miyawaki_optimized['mse']) / miyawaki_baseline['mse']) * 100,
        ((miyawaki_optimized['correlation'] - miyawaki_baseline['correlation']) / abs(miyawaki_baseline['correlation'])) * 100,
        ((miyawaki_optimized['ssim'] - miyawaki_baseline['ssim']) / abs(miyawaki_baseline['ssim'])) * 100
    ]
    
    colors_improvement = ['green' if x > 0 else 'red' for x in miyawaki_improvements]
    bars = ax9.bar(metrics, miyawaki_improvements, color=colors_improvement, alpha=0.7)
    ax9.set_ylabel('Improvement (%)')
    ax9.set_title('Miyawaki Optimization\nImprovement')
    ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    for bar, imp in zip(bars, miyawaki_improvements):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # Vangerven improvement
    ax10 = plt.subplot(3, 4, 10)
    vangerven_improvements = [
        ((vangerven_baseline['mse'] - vangerven_optimized['mse']) / vangerven_baseline['mse']) * 100,
        ((vangerven_optimized['correlation'] - vangerven_baseline['correlation']) / abs(vangerven_baseline['correlation'])) * 100,
        ((vangerven_optimized['ssim'] - vangerven_baseline['ssim']) / abs(vangerven_baseline['ssim'])) * 100
    ]
    
    colors_improvement = ['green' if x > 0 else 'red' for x in vangerven_improvements]
    bars = ax10.bar(metrics, vangerven_improvements, color=colors_improvement, alpha=0.7)
    ax10.set_ylabel('Improvement (%)')
    ax10.set_title('Vangerven Optimization\nImprovement')
    ax10.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    for bar, imp in zip(bars, vangerven_improvements):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # Summary table
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('tight')
    ax11.axis('off')
    
    table_data = [
        ['Model', 'Training Loss', 'MSE', 'Correlation', 'SSIM'],
        ['Miyawaki Baseline', f"{miyawaki_baseline['training_loss']:.4f}", 
         f"{miyawaki_baseline['mse']:.4f}", f"{miyawaki_baseline['correlation']:.3f}", 
         f"{miyawaki_baseline['ssim']:.3f}"],
        ['Miyawaki Optimized', f"{miyawaki_optimized['training_loss']:.4f}", 
         f"{miyawaki_optimized['mse']:.4f}", f"{miyawaki_optimized['correlation']:.3f}", 
         f"{miyawaki_optimized['ssim']:.3f}"],
        ['Vangerven Baseline', f"{vangerven_baseline['training_loss']:.4f}", 
         f"{vangerven_baseline['mse']:.4f}", f"{vangerven_baseline['correlation']:.3f}", 
         f"{vangerven_baseline['ssim']:.3f}"],
        ['Vangerven Optimized', f"{vangerven_optimized['training_loss']:.4f}", 
         f"{vangerven_optimized['mse']:.4f}", f"{vangerven_optimized['correlation']:.3f}", 
         f"{vangerven_optimized['ssim']:.3f}"]
    ]
    
    table = ax11.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax11.set_title('Performance Summary Table')
    
    # Winner analysis
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Find best model for each metric
    best_training_loss = min(all_results, key=lambda x: x['training_loss'])
    best_mse = min(all_results, key=lambda x: x['mse'])
    best_correlation = max(all_results, key=lambda x: x['correlation'])
    best_ssim = max(all_results, key=lambda x: x['ssim'])
    
    winner_text = f"""🏆 PERFORMANCE WINNERS:

📉 Best Training Loss:
   {best_training_loss['model_name']}
   ({best_training_loss['training_loss']:.4f})

📊 Best MSE:
   {best_mse['model_name']}
   ({best_mse['mse']:.4f})

📈 Best Correlation:
   {best_correlation['model_name']}
   ({best_correlation['correlation']:.3f})

🔍 Best SSIM:
   {best_ssim['model_name']}
   ({best_ssim['ssim']:.3f})"""
    
    ax12.text(0.05, 0.95, winner_text, transform=ax12.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Brain LDM: Complete Optimization Analysis\nMiyawaki vs Vangerven | Baseline vs Optimized', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    output_path = "results/final_optimization_comparison.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Final comparison saved: {output_path}")
    
    plt.show()

def main():
    """Main comparison function."""
    
    print("🧠 FINAL OPTIMIZATION COMPARISON")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Evaluate all models
    all_results = []
    
    # Miyawaki Baseline
    miyawaki_baseline = load_and_evaluate_model(
        "checkpoints/best_miyawaki_simple_model.pt",
        "data/miyawaki_structured_28x28.mat",
        "Miyawaki Baseline",
        device
    )
    if miyawaki_baseline:
        all_results.append(miyawaki_baseline)
    
    # Miyawaki Optimized
    miyawaki_optimized = load_and_evaluate_model(
        "checkpoints/best_miyawaki_optimized_model.pt",
        "data/miyawaki_structured_28x28.mat",
        "Miyawaki Optimized",
        device
    )
    if miyawaki_optimized:
        all_results.append(miyawaki_optimized)
    
    # Vangerven Baseline
    vangerven_baseline = load_and_evaluate_model(
        "checkpoints/best_vangerven_simple_model.pt",
        "data/digit69_28x28.mat",
        "Vangerven Baseline",
        device
    )
    if vangerven_baseline:
        all_results.append(vangerven_baseline)
    
    # Vangerven Optimized
    vangerven_optimized = load_and_evaluate_model(
        "checkpoints/best_vangerven_optimized_model.pt",
        "data/digit69_28x28.mat",
        "Vangerven Optimized",
        device
    )
    if vangerven_optimized:
        all_results.append(vangerven_optimized)
    
    if len(all_results) == 4:
        # Create final comparison
        create_final_comparison_visualization(all_results)
        
        # Print detailed analysis
        print(f"\n🎯 FINAL ANALYSIS SUMMARY")
        print("=" * 30)
        
        miyawaki_baseline = all_results[0]
        miyawaki_optimized = all_results[1]
        vangerven_baseline = all_results[2]
        vangerven_optimized = all_results[3]
        
        print(f"\n📊 MIYAWAKI OPTIMIZATION RESULTS:")
        mse_imp = ((miyawaki_baseline['mse'] - miyawaki_optimized['mse']) / miyawaki_baseline['mse']) * 100
        corr_imp = ((miyawaki_optimized['correlation'] - miyawaki_baseline['correlation']) / abs(miyawaki_baseline['correlation'])) * 100
        print(f"   MSE improvement: {mse_imp:+.1f}%")
        print(f"   Correlation improvement: {corr_imp:+.1f}%")
        print(f"   Result: {'✅ SUCCESSFUL' if mse_imp > 0 and corr_imp > 0 else '⚠️ MIXED'}")
        
        print(f"\n📊 VANGERVEN OPTIMIZATION RESULTS:")
        mse_imp = ((vangerven_baseline['mse'] - vangerven_optimized['mse']) / vangerven_baseline['mse']) * 100
        corr_imp = ((vangerven_optimized['correlation'] - vangerven_baseline['correlation']) / abs(vangerven_baseline['correlation'])) * 100
        print(f"   MSE improvement: {mse_imp:+.1f}%")
        print(f"   Correlation improvement: {corr_imp:+.1f}%")
        print(f"   Result: {'✅ SUCCESSFUL' if mse_imp > 0 and corr_imp > 0 else '❌ BASELINE BETTER'}")
        
        print(f"\n🏆 OVERALL WINNER:")
        best_model = max(all_results, key=lambda x: x['correlation'])
        print(f"   {best_model['model_name']} with correlation {best_model['correlation']:.3f}")
        
        print(f"\n📁 Results saved to: results/final_optimization_comparison.png")
    else:
        print("❌ Could not load all models for comparison")

if __name__ == "__main__":
    main()
