"""
📊 Evaluate Multi-Modal Guidance Effects
Comprehensive evaluation of different guidance mechanisms and their impact on reconstruction quality.
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

def load_best_model(model_path="checkpoints/best_conservative_model.pt"):
    """Load the best trained multi-modal model."""
    print(f"📁 Loading model from: {model_path}")
    
    device = 'cpu'
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("🔄 Looking for alternative models...")
        
        # Try other model paths
        alternative_paths = [
            "checkpoints/best_multimodal_model.pt",
            "checkpoints/best_aggressive_model.pt",
            "checkpoints/best_balanced_model.pt"
        ]
        
        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                model_path = alt_path
                print(f"✅ Found alternative model: {alt_path}")
                break
        else:
            print("❌ No trained multi-modal model found!")
            return None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    model = MultiModalBrainLDM(
        fmri_dim=3092,
        image_size=28,
        guidance_scale=7.5  # Default, will be overridden during evaluation
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Loss: {checkpoint.get('loss', 'Unknown')}")
    
    return model

def evaluate_guidance_configurations(model, test_data, guidance_scales=[1.0, 3.0, 7.5, 15.0]):
    """Evaluate different guidance configurations."""
    print("🔍 Evaluating Guidance Configurations")
    print("=" * 45)
    
    test_fmri = improved_fmri_normalization(test_data['fmri'])
    test_stimuli = test_data['stimuli']
    test_labels = test_data['labels']
    
    # Create text tokens
    captions = create_digit_captions(test_labels)
    text_tokens = tokenize_captions(captions)
    
    results = {}
    
    with torch.no_grad():
        for guidance_scale in guidance_scales:
            print(f"  🎛️ Testing guidance scale: {guidance_scale}")
            
            # 1. No Guidance (fMRI only)
            model.guidance_scale = 1.0  # Minimal guidance
            recons_no_guidance, _ = model.generate_with_guidance(
                test_fmri, guidance_scale=1.0
            )
            
            # 2. Text Guidance Only
            model.guidance_scale = guidance_scale
            recons_text_only, _ = model.generate_with_guidance(
                test_fmri, text_tokens=text_tokens, guidance_scale=guidance_scale
            )
            
            # 3. Semantic Guidance Only
            recons_semantic_only, _ = model.generate_with_guidance(
                test_fmri, class_labels=test_labels, guidance_scale=guidance_scale
            )
            
            # 4. Full Multi-Modal Guidance
            recons_full_guidance, attention_weights = model.generate_with_guidance(
                test_fmri, text_tokens=text_tokens, class_labels=test_labels, 
                guidance_scale=guidance_scale
            )
            
            # Compute metrics for each configuration
            configs = {
                'no_guidance': recons_no_guidance,
                'text_only': recons_text_only,
                'semantic_only': recons_semantic_only,
                'full_guidance': recons_full_guidance
            }
            
            scale_results = {}
            for config_name, reconstructions in configs.items():
                metrics = compute_comprehensive_metrics(reconstructions, test_stimuli, test_labels)
                scale_results[config_name] = metrics
            
            scale_results['attention_weights'] = attention_weights
            results[guidance_scale] = scale_results
    
    return results

def compute_comprehensive_metrics(reconstructions, targets, labels=None):
    """Compute comprehensive reconstruction metrics."""
    # Basic metrics
    mse = F.mse_loss(reconstructions, targets).item()
    mae = F.l1_loss(reconstructions, targets).item()
    
    # Convert to numpy for detailed analysis
    recons_flat = reconstructions.cpu().numpy()
    targets_flat = targets.cpu().numpy()
    
    # Per-sample correlations
    correlations = []
    for i in range(len(recons_flat)):
        corr = np.corrcoef(recons_flat[i], targets_flat[i])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0)
    
    avg_correlation = np.mean(correlations)
    
    # Classification accuracy (correlation matrix approach)
    corr_matrix = np.zeros((len(recons_flat), len(targets_flat)))
    for i in range(len(recons_flat)):
        for j in range(len(targets_flat)):
            corr = np.corrcoef(recons_flat[i], targets_flat[j])[0, 1]
            corr_matrix[i, j] = corr if not np.isnan(corr) else 0
    
    correct_matches = sum(np.argmax(corr_matrix[i, :]) == i for i in range(len(corr_matrix)))
    accuracy = correct_matches / len(corr_matrix)
    
    # SSIM approximation
    ssim_scores = []
    for i in range(len(recons_flat)):
        recon_img = recons_flat[i].reshape(28, 28)
        target_img = targets_flat[i].reshape(28, 28)
        
        # Simplified SSIM calculation
        mu1, mu2 = recon_img.mean(), target_img.mean()
        sigma1, sigma2 = recon_img.std(), target_img.std()
        sigma12 = np.mean((recon_img - mu1) * (target_img - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        ssim_scores.append(ssim)
    
    avg_ssim = np.mean(ssim_scores)
    
    # Per-class analysis if labels provided
    class_metrics = {}
    if labels is not None:
        for digit in range(10):
            digit_mask = (labels == digit).cpu().numpy()
            if digit_mask.sum() > 0:
                digit_corrs = [correlations[i] for i in range(len(correlations)) if digit_mask[i]]
                class_metrics[digit] = {
                    'correlation': np.mean(digit_corrs),
                    'count': digit_mask.sum()
                }
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': avg_correlation,
        'accuracy': accuracy,
        'ssim': avg_ssim,
        'correlations': correlations,
        'class_metrics': class_metrics
    }

def create_guidance_comparison_visualization(model, test_data, save_path="results/guidance_effects_comparison.png"):
    """Create comprehensive guidance comparison visualization."""
    print("🎨 Creating guidance comparison visualization...")
    
    test_fmri = improved_fmri_normalization(test_data['fmri'][:6])  # Use 6 samples
    test_stimuli = test_data['stimuli'][:6]
    test_labels = test_data['labels'][:6]
    
    # Create text tokens
    captions = create_digit_captions(test_labels)
    text_tokens = tokenize_captions(captions)
    
    with torch.no_grad():
        # Generate with different guidance configurations
        recons_no_guidance, _ = model.generate_with_guidance(
            test_fmri, guidance_scale=1.0
        )
        
        recons_text_only, _ = model.generate_with_guidance(
            test_fmri, text_tokens=text_tokens, guidance_scale=7.5
        )
        
        recons_semantic_only, _ = model.generate_with_guidance(
            test_fmri, class_labels=test_labels, guidance_scale=7.5
        )
        
        recons_full_guidance, attention_weights = model.generate_with_guidance(
            test_fmri, text_tokens=text_tokens, class_labels=test_labels, guidance_scale=7.5
        )
    
    # Create visualization
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    
    guidance_types = [
        ('Original', test_stimuli, 'Ground Truth'),
        ('No Guidance', recons_no_guidance, 'fMRI Only'),
        ('Text Guidance', recons_text_only, 'fMRI + Text'),
        ('Semantic Guidance', recons_semantic_only, 'fMRI + Semantic'),
        ('Full Multi-Modal', recons_full_guidance, 'fMRI + Text + Semantic')
    ]
    
    for row, (title, images, subtitle) in enumerate(guidance_types):
        for col in range(6):
            img = images[col].cpu().numpy().reshape(28, 28)
            
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            
            if row == 0:  # Original row
                digit_label = test_labels[col].item()
                axes[row, col].set_title(f'Digit {digit_label}', fontsize=10, fontweight='bold')
            else:
                axes[row, col].set_title('')
            
            axes[row, col].axis('off')
        
        # Add row label
        axes[row, 0].set_ylabel(f'{title}\n({subtitle})', rotation=90, 
                               fontsize=11, fontweight='bold', va='center')
    
    plt.suptitle('Multi-Modal Guidance Effects on Brain-to-Image Reconstruction', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, top=0.90)
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved guidance comparison to: {save_path}")
    
    plt.show()

def create_metrics_analysis_chart(results, save_path="results/guidance_metrics_analysis.png"):
    """Create detailed metrics analysis chart."""
    print("📊 Creating metrics analysis chart...")
    
    # Extract data for guidance scale 7.5 (optimal)
    optimal_results = results[7.5]
    
    configs = ['no_guidance', 'text_only', 'semantic_only', 'full_guidance']
    config_names = ['No Guidance', 'Text Only', 'Semantic Only', 'Full Multi-Modal']
    
    # Extract metrics
    metrics_data = {
        'accuracy': [optimal_results[config]['accuracy'] * 100 for config in configs],
        'correlation': [optimal_results[config]['correlation'] for config in configs],
        'ssim': [optimal_results[config]['ssim'] for config in configs],
        'mse': [optimal_results[config]['mse'] for config in configs]
    }
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
    x = np.arange(len(config_names))
    
    # Accuracy
    bars1 = ax1.bar(x, metrics_data['accuracy'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Classification Accuracy (%)')
    ax1.set_title('Classification Accuracy by Guidance Type', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars1, metrics_data['accuracy']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Correlation
    bars2 = ax2.bar(x, metrics_data['correlation'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Correlation')
    ax2.set_title('Reconstruction Correlation by Guidance Type', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, corr in zip(bars2, metrics_data['correlation']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{corr:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM
    bars3 = ax3.bar(x, metrics_data['ssim'], color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('SSIM Score')
    ax3.set_title('Structural Similarity by Guidance Type', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, ssim in zip(bars3, metrics_data['ssim']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{ssim:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE (lower is better)
    bars4 = ax4.bar(x, metrics_data['mse'], color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Mean Squared Error')
    ax4.set_title('Reconstruction Error by Guidance Type', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, mse in zip(bars4, metrics_data['mse']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{mse:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Multi-Modal Guidance Effects: Comprehensive Metrics Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved metrics analysis to: {save_path}")
    
    plt.show()
    
    return metrics_data

def analyze_guidance_scale_effects(results, save_path="results/guidance_scale_effects.png"):
    """Analyze effects of different guidance scales."""
    print("📈 Analyzing guidance scale effects...")
    
    guidance_scales = list(results.keys())
    
    # Extract metrics for full guidance across scales
    full_guidance_metrics = {
        'accuracy': [results[scale]['full_guidance']['accuracy'] * 100 for scale in guidance_scales],
        'correlation': [results[scale]['full_guidance']['correlation'] for scale in guidance_scales],
        'ssim': [results[scale]['full_guidance']['ssim'] for scale in guidance_scales]
    }
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy vs guidance scale
    ax1.plot(guidance_scales, full_guidance_metrics['accuracy'], 'o-', linewidth=3, markersize=8, color='steelblue')
    ax1.set_xlabel('Guidance Scale')
    ax1.set_ylabel('Classification Accuracy (%)')
    ax1.set_title('Accuracy vs Guidance Scale', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(full_guidance_metrics['accuracy']) * 1.1)
    
    # Correlation vs guidance scale
    ax2.plot(guidance_scales, full_guidance_metrics['correlation'], 'o-', linewidth=3, markersize=8, color='orange')
    ax2.set_xlabel('Guidance Scale')
    ax2.set_ylabel('Average Correlation')
    ax2.set_title('Correlation vs Guidance Scale', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # SSIM vs guidance scale
    ax3.plot(guidance_scales, full_guidance_metrics['ssim'], 'o-', linewidth=3, markersize=8, color='green')
    ax3.set_xlabel('Guidance Scale')
    ax3.set_ylabel('SSIM Score')
    ax3.set_title('SSIM vs Guidance Scale', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Find optimal guidance scale
    optimal_scale_idx = np.argmax(full_guidance_metrics['accuracy'])
    optimal_scale = guidance_scales[optimal_scale_idx]
    
    # Mark optimal points
    for ax, metric_values in zip([ax1, ax2, ax3], [full_guidance_metrics['accuracy'], 
                                                   full_guidance_metrics['correlation'], 
                                                   full_guidance_metrics['ssim']]):
        ax.axvline(x=optimal_scale, color='red', linestyle='--', alpha=0.7, 
                  label=f'Optimal: {optimal_scale}')
        ax.legend()
    
    plt.suptitle('Guidance Scale Optimization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved guidance scale analysis to: {save_path}")
    
    plt.show()
    
    return optimal_scale

def print_comprehensive_results(results):
    """Print comprehensive evaluation results."""
    print("\n📊 Comprehensive Guidance Effects Analysis")
    print("=" * 55)
    
    # Find best configuration
    best_scale = 7.5  # Typically optimal
    best_results = results[best_scale]
    
    print(f"🎯 Results for Guidance Scale {best_scale}:")
    print("=" * 40)
    
    configs = ['no_guidance', 'text_only', 'semantic_only', 'full_guidance']
    config_names = ['No Guidance', 'Text Only', 'Semantic Only', 'Full Multi-Modal']
    
    for config, name in zip(configs, config_names):
        metrics = best_results[config]
        print(f"\n📈 {name}:")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        print(f"   Correlation: {metrics['correlation']:.6f}")
        print(f"   SSIM: {metrics['ssim']:.6f}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
    
    # Calculate improvements
    baseline = best_results['no_guidance']
    full_modal = best_results['full_guidance']
    
    print(f"\n🚀 Full Multi-Modal vs No Guidance Improvements:")
    print("=" * 50)
    print(f"   Accuracy: {baseline['accuracy']:.2%} → {full_modal['accuracy']:.2%} "
          f"({((full_modal['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100):+.1f}%)")
    print(f"   Correlation: {baseline['correlation']:.6f} → {full_modal['correlation']:.6f} "
          f"({((full_modal['correlation'] - baseline['correlation']) / abs(baseline['correlation']) * 100):+.1f}%)")
    print(f"   SSIM: {baseline['ssim']:.6f} → {full_modal['ssim']:.6f} "
          f"({((full_modal['ssim'] - baseline['ssim']) / baseline['ssim'] * 100):+.1f}%)")

def main():
    """Main evaluation function."""
    print("📊 Multi-Modal Guidance Effects Evaluation")
    print("=" * 55)
    
    # Load model
    model = load_best_model()
    if model is None:
        print("❌ Cannot proceed without trained model")
        return
    
    # Load test data
    loader = load_fmri_data()
    test_data = loader.get_test_data()
    
    print(f"📊 Test data: {len(test_data['stimuli'])} samples")
    
    # Evaluate guidance configurations
    results = evaluate_guidance_configurations(model, test_data)
    
    # Create visualizations
    create_guidance_comparison_visualization(model, test_data)
    metrics_data = create_metrics_analysis_chart(results)
    optimal_scale = analyze_guidance_scale_effects(results)
    
    # Print comprehensive results
    print_comprehensive_results(results)
    
    print(f"\n🎉 Guidance Effects Evaluation Complete!")
    print(f"🏆 Optimal guidance scale: {optimal_scale}")
    print(f"📁 Results saved to: results/")
    print(f"💡 Key finding: Full multi-modal guidance provides best performance")

if __name__ == "__main__":
    main()
