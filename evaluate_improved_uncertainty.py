"""
🔬 Evaluate Improved Model Uncertainty
Comprehensive uncertainty evaluation for the improved Brain LDM model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_loader import load_fmri_data
from improved_brain_ldm import ImprovedBrainLDM, create_digit_captions, tokenize_captions
import scipy.stats

def improved_fmri_normalization(fmri_data):
    """Improved fMRI normalization with outlier handling."""
    median = torch.median(fmri_data, dim=0, keepdim=True)[0]
    mad = torch.median(torch.abs(fmri_data - median), dim=0, keepdim=True)[0]
    mad = torch.where(mad == 0, torch.ones_like(mad), mad)
    normalized = (fmri_data - median) / (1.4826 * mad)
    normalized = torch.clamp(normalized, -3, 3)
    return normalized

def load_improved_model(model_path="checkpoints/best_improved_v1_model.pt"):
    """Load the improved model for uncertainty evaluation."""
    print(f"📁 Loading improved model: {model_path}")
    
    device = 'cpu'
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Improved model not found at {model_path}")
        print("🔄 Looking for alternative models...")
        
        alternative_paths = [
            "checkpoints/best_aggressive_model.pt",
            "checkpoints/best_conservative_model.pt",
            "checkpoints/best_multimodal_model.pt"
        ]
        
        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                print(f"⚠️ Using fallback model: {alt_path}")
                # Load with original model architecture
                from multimodal_brain_ldm import MultiModalBrainLDM
                checkpoint = torch.load(alt_path, map_location=device, weights_only=False)
                model = MultiModalBrainLDM(fmri_dim=3092, image_size=28)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                return model, "fallback"
        
        print("❌ No model found!")
        return None, None
    
    # Load improved model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ImprovedBrainLDM(fmri_dim=3092, image_size=28)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Improved model loaded successfully!")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Loss: {checkpoint.get('loss', 'Unknown'):.6f}")
    
    return model, "improved"

def enable_dropout_for_uncertainty(model):
    """Enable dropout layers for Monte Carlo sampling."""
    def enable_dropout(m):
        if type(m) in [torch.nn.Dropout, torch.nn.Dropout2d]:
            m.train()
    
    model.apply(enable_dropout)
    return model

def monte_carlo_sampling_improved(model, fmri_signals, text_tokens=None, class_labels=None, 
                                n_samples=30, guidance_scale=7.5):
    """Enhanced Monte Carlo sampling with better uncertainty estimation."""
    print(f"🎲 Enhanced Monte Carlo sampling with {n_samples} samples...")
    
    # Enable dropout for uncertainty
    model = enable_dropout_for_uncertainty(model)
    
    samples = []
    uncertainties = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Add noise injection for better uncertainty estimation
            noise_scale = 0.05
            noisy_fmri = fmri_signals + torch.randn_like(fmri_signals) * noise_scale
            
            # Generate sample with stochastic forward pass
            if hasattr(model, 'generate_with_guidance'):
                sample, _ = model.generate_with_guidance(
                    noisy_fmri, 
                    text_tokens=text_tokens, 
                    class_labels=class_labels,
                    guidance_scale=guidance_scale,
                    add_noise=True
                )
            else:
                # Fallback for original model
                sample, _ = model.generate_with_guidance(
                    noisy_fmri, 
                    text_tokens=text_tokens, 
                    class_labels=class_labels,
                    guidance_scale=guidance_scale
                )
            
            samples.append(sample.cpu())
            
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{n_samples} samples")
    
    # Stack samples: [n_samples, batch_size, ...]
    samples = torch.stack(samples, dim=0)
    
    return samples

def compute_enhanced_uncertainty_metrics(samples):
    """Compute enhanced uncertainty metrics."""
    print("📊 Computing enhanced uncertainty metrics...")
    
    # samples shape: [n_samples, batch_size, channels, height, width]
    n_samples = samples.shape[0]
    batch_size = samples.shape[1]
    
    # Flatten spatial dimensions for analysis
    samples_flat = samples.view(n_samples, batch_size, -1)
    
    # Compute statistics
    mean_prediction = samples_flat.mean(dim=0)  # [batch_size, image_dim]
    std_prediction = samples_flat.std(dim=0)    # [batch_size, image_dim]
    
    # Enhanced uncertainty measures
    epistemic_uncertainty = std_prediction.mean(dim=1)  # Model uncertainty
    aleatoric_uncertainty = samples_flat.var(dim=0).mean(dim=1)  # Data uncertainty
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
    
    # Confidence intervals
    lower_bound = torch.quantile(samples_flat, 0.025, dim=0)
    upper_bound = torch.quantile(samples_flat, 0.975, dim=0)
    confidence_width = (upper_bound - lower_bound).mean(dim=1)
    
    # Prediction entropy
    probs = torch.sigmoid(samples_flat)
    entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
    mean_entropy = entropy.mean(dim=(0, 2))
    
    # Mutual information (approximation)
    # I(y, θ|x) ≈ H(y|x) - E[H(y|x,θ)]
    total_entropy = mean_entropy
    conditional_entropy = entropy.mean(dim=0).mean(dim=1)
    mutual_information = total_entropy - conditional_entropy
    
    # Predictive variance decomposition
    predictive_variance = samples_flat.var(dim=0).mean(dim=1)
    expected_variance = samples_flat.var(dim=2).mean(dim=0)
    variance_of_expected = samples_flat.mean(dim=2).var(dim=0)
    
    return {
        'mean_prediction': mean_prediction,
        'std_prediction': std_prediction,
        'epistemic_uncertainty': epistemic_uncertainty,
        'aleatoric_uncertainty': aleatoric_uncertainty,
        'total_uncertainty': total_uncertainty,
        'confidence_width': confidence_width,
        'entropy': mean_entropy,
        'mutual_information': mutual_information,
        'predictive_variance': predictive_variance,
        'expected_variance': expected_variance,
        'variance_of_expected': variance_of_expected,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'all_samples': samples
    }

def compare_uncertainty_improvements(original_metrics, improved_metrics):
    """Compare uncertainty improvements between models."""
    print("📊 Comparing Uncertainty Improvements")
    print("=" * 40)
    
    improvements = {}
    
    # Compare key metrics
    metrics_to_compare = [
        'total_uncertainty', 'epistemic_uncertainty', 'aleatoric_uncertainty',
        'confidence_width', 'entropy'
    ]
    
    for metric in metrics_to_compare:
        if metric in original_metrics and metric in improved_metrics:
            orig_val = original_metrics[metric].mean().item()
            impr_val = improved_metrics[metric].mean().item()
            
            # Calculate improvement percentage
            if orig_val != 0:
                improvement = ((impr_val - orig_val) / abs(orig_val)) * 100
            else:
                improvement = 0
            
            improvements[metric] = {
                'original': orig_val,
                'improved': impr_val,
                'change_percent': improvement
            }
            
            print(f"📈 {metric.replace('_', ' ').title()}:")
            print(f"   Original: {orig_val:.6f}")
            print(f"   Improved: {impr_val:.6f}")
            print(f"   Change: {improvement:+.1f}%")
    
    return improvements

def create_uncertainty_comparison_visualization(original_metrics, improved_metrics, 
                                              save_path="results/uncertainty_comparison.png"):
    """Create comprehensive uncertainty comparison visualization."""
    print("🎨 Creating uncertainty comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Total Uncertainty Distribution
    ax = axes[0, 0]
    if original_metrics:
        ax.hist(original_metrics['total_uncertainty'].cpu().numpy(), bins=15, alpha=0.7, 
               label='Original', color='red', density=True)
    ax.hist(improved_metrics['total_uncertainty'].cpu().numpy(), bins=15, alpha=0.7, 
           label='Improved', color='green', density=True)
    ax.set_xlabel('Total Uncertainty')
    ax.set_ylabel('Density')
    ax.set_title('Total Uncertainty Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Epistemic vs Aleatoric
    ax = axes[0, 1]
    epistemic = improved_metrics['epistemic_uncertainty'].cpu().numpy()
    aleatoric = improved_metrics['aleatoric_uncertainty'].cpu().numpy()
    ax.scatter(epistemic, aleatoric, alpha=0.7, color='blue')
    ax.set_xlabel('Epistemic Uncertainty')
    ax.set_ylabel('Aleatoric Uncertainty')
    ax.set_title('Uncertainty Decomposition', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line
    max_val = max(epistemic.max(), aleatoric.max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    # 3. Confidence Width vs Entropy
    ax = axes[0, 2]
    conf_width = improved_metrics['confidence_width'].cpu().numpy()
    entropy = improved_metrics['entropy'].cpu().numpy()
    ax.scatter(conf_width, entropy, alpha=0.7, color='purple')
    ax.set_xlabel('95% Confidence Width')
    ax.set_ylabel('Prediction Entropy')
    ax.set_title('Confidence vs Entropy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(conf_width, entropy)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # 4. Mutual Information
    ax = axes[1, 0]
    mi = improved_metrics['mutual_information'].cpu().numpy()
    ax.hist(mi, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Frequency')
    ax.set_title('Mutual Information Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. Variance Decomposition
    ax = axes[1, 1]
    pred_var = improved_metrics['predictive_variance'].cpu().numpy()
    exp_var = improved_metrics['expected_variance'].cpu().numpy()
    var_exp = improved_metrics['variance_of_expected'].cpu().numpy()
    
    x = np.arange(len(pred_var))
    width = 0.25
    ax.bar(x - width, pred_var, width, label='Predictive', alpha=0.8)
    ax.bar(x, exp_var, width, label='Expected', alpha=0.8)
    ax.bar(x + width, var_exp, width, label='Variance of Expected', alpha=0.8)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Decomposition', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Uncertainty Calibration
    ax = axes[1, 2]
    # Create calibration plot (simplified)
    uncertainties = improved_metrics['total_uncertainty'].cpu().numpy()
    sorted_indices = np.argsort(uncertainties)
    sorted_uncertainties = uncertainties[sorted_indices]
    
    # Plot uncertainty trend
    ax.plot(sorted_uncertainties, 'b-', linewidth=2, label='Uncertainty')
    ax.set_xlabel('Sample (sorted by uncertainty)')
    ax.set_ylabel('Uncertainty Value')
    ax.set_title('Uncertainty Calibration', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle('Enhanced Uncertainty Analysis: Model Improvements', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved uncertainty comparison to: {save_path}")
    
    plt.show()

def assess_uncertainty_quality(uncertainty_metrics, test_stimuli):
    """Assess the quality of uncertainty estimates."""
    print("🔍 Assessing Uncertainty Quality")
    print("=" * 35)
    
    mean_predictions = uncertainty_metrics['mean_prediction']
    total_uncertainties = uncertainty_metrics['total_uncertainty']
    
    # Reshape for comparison
    if test_stimuli.dim() == 2:
        test_stimuli_flat = test_stimuli
    else:
        test_stimuli_flat = test_stimuli.view(test_stimuli.size(0), -1)
    
    # Compute prediction errors
    mse_errors = F.mse_loss(mean_predictions, test_stimuli_flat, reduction='none').mean(dim=1)
    
    # Uncertainty-error correlation
    uncertainty_error_corr = np.corrcoef(
        total_uncertainties.cpu().numpy(), 
        mse_errors.cpu().numpy()
    )[0, 1]
    
    # Calibration metrics
    high_unc_threshold = total_uncertainties.quantile(0.75)
    low_unc_threshold = total_uncertainties.quantile(0.25)
    
    high_unc_mask = total_uncertainties > high_unc_threshold
    low_unc_mask = total_uncertainties < low_unc_threshold
    
    high_unc_error = mse_errors[high_unc_mask].mean().item() if high_unc_mask.sum() > 0 else 0
    low_unc_error = mse_errors[low_unc_mask].mean().item() if low_unc_mask.sum() > 0 else 0
    
    # Quality metrics
    quality_metrics = {
        'uncertainty_error_correlation': uncertainty_error_corr,
        'high_uncertainty_error': high_unc_error,
        'low_uncertainty_error': low_unc_error,
        'calibration_ratio': low_unc_error / high_unc_error if high_unc_error > 0 else float('inf'),
        'mean_uncertainty': total_uncertainties.mean().item(),
        'uncertainty_std': total_uncertainties.std().item(),
        'uncertainty_range': (total_uncertainties.min().item(), total_uncertainties.max().item())
    }
    
    print(f"📊 Uncertainty Quality Metrics:")
    print(f"   Uncertainty-Error Correlation: {uncertainty_error_corr:.4f}")
    print(f"   High Uncertainty Error: {high_unc_error:.6f}")
    print(f"   Low Uncertainty Error: {low_unc_error:.6f}")
    print(f"   Calibration Ratio: {quality_metrics['calibration_ratio']:.3f}")
    print(f"   Mean Uncertainty: {quality_metrics['mean_uncertainty']:.6f}")
    print(f"   Uncertainty Std: {quality_metrics['uncertainty_std']:.6f}")
    print(f"   Uncertainty Range: [{quality_metrics['uncertainty_range'][0]:.6f}, {quality_metrics['uncertainty_range'][1]:.6f}]")
    
    # Quality assessment
    print(f"\n💡 Quality Assessment:")
    if abs(uncertainty_error_corr) > 0.3:
        print(f"   ✅ Good: Strong correlation between uncertainty and error")
    else:
        print(f"   ⚠️ Warning: Weak correlation between uncertainty and error")
    
    if quality_metrics['calibration_ratio'] < 0.8:
        print(f"   ✅ Good: Well-calibrated uncertainty (low uncertainty → low error)")
    else:
        print(f"   ⚠️ Warning: Poor calibration")
    
    if quality_metrics['uncertainty_std'] > 0.001:
        print(f"   ✅ Good: Reasonable uncertainty variation")
    else:
        print(f"   ⚠️ Warning: Very low uncertainty variation")
    
    return quality_metrics

def main():
    """Main improved uncertainty evaluation function."""
    print("🔬 Enhanced Uncertainty Evaluation for Improved Brain LDM")
    print("=" * 65)
    
    # Load improved model
    model, model_type = load_improved_model()
    if model is None:
        print("❌ Cannot proceed without model")
        return
    
    # Load test data
    loader = load_fmri_data()
    test_data = loader.get_test_data()
    
    test_fmri = improved_fmri_normalization(test_data['fmri'])
    test_stimuli = test_data['stimuli']
    test_labels = test_data['labels']
    
    print(f"📊 Test data: {len(test_stimuli)} samples")
    print(f"🤖 Model type: {model_type}")
    
    # Create text tokens
    captions = create_digit_captions(test_labels)
    text_tokens = tokenize_captions(captions)
    
    # Perform enhanced Monte Carlo sampling
    mc_samples = monte_carlo_sampling_improved(
        model, test_fmri, text_tokens, test_labels, 
        n_samples=30, guidance_scale=7.5
    )
    
    # Compute enhanced uncertainty metrics
    uncertainty_metrics = compute_enhanced_uncertainty_metrics(mc_samples)
    
    # Assess uncertainty quality
    quality_metrics = assess_uncertainty_quality(uncertainty_metrics, test_stimuli)
    
    # Create visualizations
    create_uncertainty_comparison_visualization(None, uncertainty_metrics)
    
    print(f"\n🎉 Enhanced Uncertainty Evaluation Complete!")
    print(f"📁 Results saved to: results/")
    print(f"💡 Key insights:")
    print(f"   • Model type: {model_type}")
    print(f"   • Uncertainty-error correlation: {quality_metrics['uncertainty_error_correlation']:.4f}")
    print(f"   • Calibration quality: {'Good' if quality_metrics['calibration_ratio'] < 0.8 else 'Needs improvement'}")
    print(f"   • Uncertainty variation: {'Good' if quality_metrics['uncertainty_std'] > 0.001 else 'Too low'}")

if __name__ == "__main__":
    main()
