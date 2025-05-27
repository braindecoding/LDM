"""
🎲 Uncertainty Evaluation for Brain LDM
Monte Carlo sampling and uncertainty quantification for assessing model reliability.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_loader import load_fmri_data
from multimodal_brain_ldm import MultiModalBrainLDM, create_digit_captions, tokenize_captions
import scipy.stats

def improved_fmri_normalization(fmri_data):
    """Improved fMRI normalization with outlier handling."""
    median = torch.median(fmri_data, dim=0, keepdim=True)[0]
    mad = torch.median(torch.abs(fmri_data - median), dim=0, keepdim=True)[0]
    mad = torch.where(mad == 0, torch.ones_like(mad), mad)
    normalized = (fmri_data - median) / (1.4826 * mad)
    normalized = torch.clamp(normalized, -3, 3)
    return normalized

def load_model_for_uncertainty(model_path="checkpoints/best_aggressive_model.pt"):
    """Load model and enable dropout for uncertainty estimation."""
    print(f"📁 Loading model for uncertainty evaluation: {model_path}")
    
    device = 'cpu'
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        # Try alternative paths
        alternative_paths = [
            "checkpoints/best_conservative_model.pt",
            "checkpoints/best_multimodal_model.pt",
            "checkpoints/best_balanced_model.pt"
        ]
        
        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                model_path = alt_path
                print(f"✅ Using alternative model: {alt_path}")
                break
        else:
            print("❌ No trained model found!")
            return None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    model = MultiModalBrainLDM(
        fmri_dim=3092,
        image_size=28,
        guidance_scale=7.5
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Loss: {checkpoint.get('loss', 'Unknown'):.6f}")
    
    return model

def enable_dropout_for_uncertainty(model):
    """Enable dropout layers for Monte Carlo sampling."""
    def enable_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    
    model.apply(enable_dropout)
    return model

def monte_carlo_sampling(model, fmri_signals, text_tokens=None, class_labels=None, 
                        n_samples=50, guidance_scale=7.5):
    """Perform Monte Carlo sampling for uncertainty estimation."""
    print(f"🎲 Performing Monte Carlo sampling with {n_samples} samples...")
    
    # Enable dropout for uncertainty
    model = enable_dropout_for_uncertainty(model)
    
    samples = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Generate sample with stochastic forward pass
            sample, _ = model.generate_with_guidance(
                fmri_signals, 
                text_tokens=text_tokens, 
                class_labels=class_labels,
                guidance_scale=guidance_scale
            )
            samples.append(sample.cpu())
            
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{n_samples} samples")
    
    # Stack samples: [n_samples, batch_size, image_dim]
    samples = torch.stack(samples, dim=0)
    
    return samples

def compute_uncertainty_metrics(samples):
    """Compute various uncertainty metrics from Monte Carlo samples."""
    print("📊 Computing uncertainty metrics...")
    
    # samples shape: [n_samples, batch_size, image_dim]
    n_samples, batch_size, image_dim = samples.shape
    
    # Compute statistics
    mean_prediction = samples.mean(dim=0)  # [batch_size, image_dim]
    std_prediction = samples.std(dim=0)    # [batch_size, image_dim]
    
    # Epistemic uncertainty (model uncertainty)
    epistemic_uncertainty = std_prediction.mean(dim=1)  # [batch_size]
    
    # Aleatoric uncertainty (data uncertainty) - approximated
    # Using variance of predictions as proxy
    aleatoric_uncertainty = samples.var(dim=0).mean(dim=1)  # [batch_size]
    
    # Total uncertainty
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
    
    # Confidence intervals (95%)
    lower_bound = torch.quantile(samples, 0.025, dim=0)
    upper_bound = torch.quantile(samples, 0.975, dim=0)
    confidence_width = (upper_bound - lower_bound).mean(dim=1)
    
    # Prediction entropy (for each pixel)
    # Convert to probabilities (sigmoid) and compute entropy
    probs = torch.sigmoid(samples)
    entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
    mean_entropy = entropy.mean(dim=(0, 2))  # [batch_size]
    
    return {
        'mean_prediction': mean_prediction,
        'std_prediction': std_prediction,
        'epistemic_uncertainty': epistemic_uncertainty,
        'aleatoric_uncertainty': aleatoric_uncertainty,
        'total_uncertainty': total_uncertainty,
        'confidence_width': confidence_width,
        'entropy': mean_entropy,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'all_samples': samples
    }

def analyze_uncertainty_patterns(uncertainty_metrics, test_labels):
    """Analyze uncertainty patterns across different digits."""
    print("🔍 Analyzing uncertainty patterns...")
    
    results = {}
    
    for digit in range(10):
        digit_mask = (test_labels == digit).cpu().numpy()
        if digit_mask.sum() > 0:
            digit_indices = np.where(digit_mask)[0]
            
            results[digit] = {
                'count': digit_mask.sum(),
                'epistemic_uncertainty': uncertainty_metrics['epistemic_uncertainty'][digit_indices].mean().item(),
                'aleatoric_uncertainty': uncertainty_metrics['aleatoric_uncertainty'][digit_indices].mean().item(),
                'total_uncertainty': uncertainty_metrics['total_uncertainty'][digit_indices].mean().item(),
                'confidence_width': uncertainty_metrics['confidence_width'][digit_indices].mean().item(),
                'entropy': uncertainty_metrics['entropy'][digit_indices].mean().item()
            }
    
    return results

def visualize_uncertainty_analysis(uncertainty_metrics, test_stimuli, test_labels, 
                                 save_path="results/uncertainty_analysis.png"):
    """Create comprehensive uncertainty visualization."""
    print("🎨 Creating uncertainty visualization...")
    
    n_samples = min(6, len(test_stimuli))
    
    fig, axes = plt.subplots(4, n_samples, figsize=(3*n_samples, 12))
    
    for i in range(n_samples):
        # Original image
        original = test_stimuli[i].cpu().numpy().reshape(28, 28)
        axes[0, i].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nDigit {test_labels[i].item()}')
        axes[0, i].axis('off')
        
        # Mean prediction
        mean_pred = uncertainty_metrics['mean_prediction'][i].cpu().numpy().reshape(28, 28)
        axes[1, i].imshow(mean_pred, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title('Mean Prediction')
        axes[1, i].axis('off')
        
        # Uncertainty map (standard deviation)
        std_map = uncertainty_metrics['std_prediction'][i].cpu().numpy().reshape(28, 28)
        im = axes[2, i].imshow(std_map, cmap='hot', vmin=0, vmax=std_map.max())
        axes[2, i].set_title(f'Uncertainty Map\n(σ = {uncertainty_metrics["total_uncertainty"][i]:.4f})')
        axes[2, i].axis('off')
        
        # Confidence interval width
        conf_width = uncertainty_metrics['confidence_width'][i].item()
        entropy = uncertainty_metrics['entropy'][i].item()
        
        # Sample variance visualization
        sample_var = uncertainty_metrics['all_samples'][:, i, :].var(dim=0).reshape(28, 28)
        axes[3, i].imshow(sample_var, cmap='viridis', vmin=0, vmax=sample_var.max())
        axes[3, i].set_title(f'Sample Variance\n(CI: {conf_width:.4f})')
        axes[3, i].axis('off')
    
    # Add row labels
    row_labels = ['Original', 'Mean Prediction', 'Uncertainty (σ)', 'Sample Variance']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, rotation=90, fontsize=12, fontweight='bold')
    
    plt.suptitle('Monte Carlo Uncertainty Analysis for Brain-to-Image Reconstruction', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved uncertainty analysis to: {save_path}")
    
    plt.show()

def create_uncertainty_distribution_plot(uncertainty_metrics, digit_patterns, 
                                        save_path="results/uncertainty_distributions.png"):
    """Create uncertainty distribution analysis."""
    print("📊 Creating uncertainty distribution plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall uncertainty distribution
    total_uncertainties = uncertainty_metrics['total_uncertainty'].cpu().numpy()
    ax1.hist(total_uncertainties, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Total Uncertainty')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Total Uncertainty', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_unc = total_uncertainties.mean()
    std_unc = total_uncertainties.std()
    ax1.axvline(mean_unc, color='red', linestyle='--', label=f'Mean: {mean_unc:.4f}')
    ax1.axvline(mean_unc + std_unc, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_unc + std_unc:.4f}')
    ax1.axvline(mean_unc - std_unc, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_unc - std_unc:.4f}')
    ax1.legend()
    
    # 2. Epistemic vs Aleatoric uncertainty
    epistemic = uncertainty_metrics['epistemic_uncertainty'].cpu().numpy()
    aleatoric = uncertainty_metrics['aleatoric_uncertainty'].cpu().numpy()
    
    ax2.scatter(epistemic, aleatoric, alpha=0.7, color='green')
    ax2.set_xlabel('Epistemic Uncertainty (Model)')
    ax2.set_ylabel('Aleatoric Uncertainty (Data)')
    ax2.set_title('Epistemic vs Aleatoric Uncertainty', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add diagonal line
    max_val = max(epistemic.max(), aleatoric.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal uncertainty')
    ax2.legend()
    
    # 3. Uncertainty by digit class
    digits = list(digit_patterns.keys())
    digit_uncertainties = [digit_patterns[d]['total_uncertainty'] for d in digits if digit_patterns[d]['count'] > 0]
    digit_labels = [f'Digit {d}' for d in digits if digit_patterns[d]['count'] > 0]
    
    if digit_uncertainties:
        bars = ax3.bar(digit_labels, digit_uncertainties, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_xlabel('Digit Class')
        ax3.set_ylabel('Average Total Uncertainty')
        ax3.set_title('Uncertainty by Digit Class', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, unc in zip(bars, digit_uncertainties):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{unc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Confidence interval analysis
    conf_widths = uncertainty_metrics['confidence_width'].cpu().numpy()
    entropies = uncertainty_metrics['entropy'].cpu().numpy()
    
    ax4.scatter(conf_widths, entropies, alpha=0.7, color='purple')
    ax4.set_xlabel('95% Confidence Interval Width')
    ax4.set_ylabel('Prediction Entropy')
    ax4.set_title('Confidence vs Entropy Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_coef = np.corrcoef(conf_widths, entropies)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
             transform=ax4.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle('Comprehensive Uncertainty Distribution Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved uncertainty distributions to: {save_path}")
    
    plt.show()

def reliability_assessment(uncertainty_metrics, test_stimuli):
    """Assess model reliability based on uncertainty metrics."""
    print("🔍 Assessing model reliability...")
    
    # Compute prediction accuracy vs uncertainty correlation
    mean_predictions = uncertainty_metrics['mean_prediction']
    total_uncertainties = uncertainty_metrics['total_uncertainty']
    
    # Compute reconstruction errors
    mse_errors = F.mse_loss(mean_predictions, test_stimuli, reduction='none').mean(dim=1)
    
    # Correlation between uncertainty and error
    uncertainty_error_corr = np.corrcoef(
        total_uncertainties.cpu().numpy(), 
        mse_errors.cpu().numpy()
    )[0, 1]
    
    # Reliability metrics
    high_uncertainty_threshold = total_uncertainties.quantile(0.75)
    low_uncertainty_threshold = total_uncertainties.quantile(0.25)
    
    high_unc_mask = total_uncertainties > high_uncertainty_threshold
    low_unc_mask = total_uncertainties < low_uncertainty_threshold
    
    high_unc_error = mse_errors[high_unc_mask].mean().item()
    low_unc_error = mse_errors[low_unc_mask].mean().item()
    
    reliability_metrics = {
        'uncertainty_error_correlation': uncertainty_error_corr,
        'high_uncertainty_error': high_unc_error,
        'low_uncertainty_error': low_unc_error,
        'reliability_ratio': low_unc_error / high_unc_error if high_unc_error > 0 else float('inf'),
        'mean_uncertainty': total_uncertainties.mean().item(),
        'uncertainty_std': total_uncertainties.std().item()
    }
    
    return reliability_metrics

def print_uncertainty_summary(uncertainty_metrics, digit_patterns, reliability_metrics):
    """Print comprehensive uncertainty analysis summary."""
    print("\n🎲 Monte Carlo Uncertainty Analysis Summary")
    print("=" * 55)
    
    print(f"📊 Overall Statistics:")
    print(f"   Mean Total Uncertainty: {uncertainty_metrics['total_uncertainty'].mean():.6f}")
    print(f"   Std Total Uncertainty: {uncertainty_metrics['total_uncertainty'].std():.6f}")
    print(f"   Mean Epistemic Uncertainty: {uncertainty_metrics['epistemic_uncertainty'].mean():.6f}")
    print(f"   Mean Aleatoric Uncertainty: {uncertainty_metrics['aleatoric_uncertainty'].mean():.6f}")
    print(f"   Mean Confidence Width: {uncertainty_metrics['confidence_width'].mean():.6f}")
    print(f"   Mean Entropy: {uncertainty_metrics['entropy'].mean():.6f}")
    
    print(f"\n🔍 Reliability Assessment:")
    print(f"   Uncertainty-Error Correlation: {reliability_metrics['uncertainty_error_correlation']:.4f}")
    print(f"   High Uncertainty Error: {reliability_metrics['high_uncertainty_error']:.6f}")
    print(f"   Low Uncertainty Error: {reliability_metrics['low_uncertainty_error']:.6f}")
    print(f"   Reliability Ratio: {reliability_metrics['reliability_ratio']:.2f}")
    
    print(f"\n📈 Per-Digit Analysis:")
    for digit, metrics in digit_patterns.items():
        if metrics['count'] > 0:
            print(f"   Digit {digit}: Uncertainty = {metrics['total_uncertainty']:.6f}, "
                  f"Entropy = {metrics['entropy']:.6f} (n={metrics['count']})")
    
    print(f"\n💡 Interpretation:")
    if reliability_metrics['uncertainty_error_correlation'] > 0.3:
        print(f"   ✅ Good: High correlation between uncertainty and error")
    else:
        print(f"   ⚠️ Warning: Low correlation between uncertainty and error")
    
    if reliability_metrics['reliability_ratio'] < 0.8:
        print(f"   ✅ Good: Low uncertainty predictions are more reliable")
    else:
        print(f"   ⚠️ Warning: Uncertainty doesn't correlate well with reliability")
    
    if uncertainty_metrics['total_uncertainty'].mean() < 0.1:
        print(f"   ✅ Good: Overall uncertainty is reasonable")
    else:
        print(f"   ⚠️ Warning: High overall uncertainty indicates model needs improvement")

def main():
    """Main uncertainty evaluation function."""
    print("🎲 Monte Carlo Uncertainty Evaluation for Brain LDM")
    print("=" * 60)
    
    # Load model
    model = load_model_for_uncertainty()
    if model is None:
        print("❌ Cannot proceed without trained model")
        return
    
    # Load test data
    loader = load_fmri_data()
    test_data = loader.get_test_data()
    
    test_fmri = improved_fmri_normalization(test_data['fmri'])
    test_stimuli = test_data['stimuli']
    test_labels = test_data['labels']
    
    print(f"📊 Test data: {len(test_stimuli)} samples")
    
    # Create text tokens for guidance
    captions = create_digit_captions(test_labels)
    text_tokens = tokenize_captions(captions)
    
    # Perform Monte Carlo sampling
    mc_samples = monte_carlo_sampling(
        model, test_fmri, text_tokens, test_labels, 
        n_samples=30, guidance_scale=7.5
    )
    
    # Compute uncertainty metrics
    uncertainty_metrics = compute_uncertainty_metrics(mc_samples)
    
    # Analyze patterns
    digit_patterns = analyze_uncertainty_patterns(uncertainty_metrics, test_labels)
    
    # Assess reliability
    reliability_metrics = reliability_assessment(uncertainty_metrics, test_stimuli)
    
    # Create visualizations
    visualize_uncertainty_analysis(uncertainty_metrics, test_stimuli, test_labels)
    create_uncertainty_distribution_plot(uncertainty_metrics, digit_patterns)
    
    # Print summary
    print_uncertainty_summary(uncertainty_metrics, digit_patterns, reliability_metrics)
    
    print(f"\n🎉 Uncertainty Evaluation Complete!")
    print(f"📁 Results saved to: results/")
    print(f"💡 Key insight: Uncertainty analysis reveals model confidence patterns")

if __name__ == "__main__":
    main()
