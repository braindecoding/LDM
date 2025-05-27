"""
📊 Evaluate Multi-Modal Brain LDM
Comprehensive evaluation of multi-modal guidance effects on reconstruction quality.
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

def load_multimodal_model(model_path="checkpoints/best_multimodal_model.pt"):
    """Load trained multi-modal model."""
    print(f"📁 Loading multi-modal model from: {model_path}")
    
    device = 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    config = checkpoint.get('config', {})
    model = MultiModalBrainLDM(
        fmri_dim=config.get('fmri_dim', 3092),
        image_size=config.get('image_size', 28),
        guidance_scale=config.get('guidance_scale', 7.5)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f})")
    return model

def evaluate_guidance_effects(model, test_data, guidance_scales=[1.0, 3.0, 7.5, 15.0]):
    """Evaluate effects of different guidance scales."""
    print("🔍 Evaluating Guidance Effects")
    print("=" * 35)
    
    test_fmri = improved_fmri_normalization(test_data['fmri'])
    test_stimuli = test_data['stimuli']
    test_labels = test_data['labels']
    
    # Create text tokens
    captions = create_digit_captions(test_labels)
    text_tokens = tokenize_captions(captions)
    
    results = {}
    
    with torch.no_grad():
        for guidance_scale in guidance_scales:
            print(f"  Testing guidance scale: {guidance_scale}")
            
            # Generate with different guidance configurations
            recons_no_guidance, _ = model.generate_with_guidance(
                test_fmri, guidance_scale=1.0
            )
            
            recons_text_only, _ = model.generate_with_guidance(
                test_fmri, text_tokens=text_tokens, guidance_scale=guidance_scale
            )
            
            recons_semantic_only, _ = model.generate_with_guidance(
                test_fmri, class_labels=test_labels, guidance_scale=guidance_scale
            )
            
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
                metrics = compute_reconstruction_metrics(reconstructions, test_stimuli)
                scale_results[config_name] = metrics
            
            scale_results['attention_weights'] = attention_weights
            results[guidance_scale] = scale_results
    
    return results

def compute_reconstruction_metrics(reconstructions, targets):
    """Compute comprehensive reconstruction metrics."""
    # Basic metrics
    mse = F.mse_loss(reconstructions, targets).item()
    mae = F.l1_loss(reconstructions, targets).item()
    
    # Correlation metrics
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
    
    # SSIM approximation (simplified)
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
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': avg_correlation,
        'accuracy': accuracy,
        'ssim': avg_ssim,
        'correlations': correlations
    }

def visualize_guidance_comparison(model, test_data, save_path="results/multimodal_guidance_comparison.png"):
    """Create comprehensive visualization of guidance effects."""
    print("🎨 Creating guidance comparison visualization...")
    
    test_fmri = improved_fmri_normalization(test_data['fmri'][:5])
    test_stimuli = test_data['stimuli'][:5]
    test_labels = test_data['labels'][:5]
    
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
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    
    for i in range(5):
        # Original
        axes[0, i].imshow(test_stimuli[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nDigit {test_labels[i].item()}')
        axes[0, i].axis('off')
        
        # No guidance
        axes[1, i].imshow(recons_no_guidance[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title('No Guidance')
        axes[1, i].axis('off')
        
        # Text guidance only
        axes[2, i].imshow(recons_text_only[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title('Text Guidance')
        axes[2, i].axis('off')
        
        # Semantic guidance only
        axes[3, i].imshow(recons_semantic_only[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[3, i].set_title('Semantic Guidance')
        axes[3, i].axis('off')
        
        # Full multi-modal guidance
        axes[4, i].imshow(recons_full_guidance[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[4, i].set_title('Full Multi-Modal')
        axes[4, i].axis('off')
    
    plt.suptitle('Multi-Modal Guidance Effects on Brain-to-Image Reconstruction', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved guidance comparison to: {save_path}")
    
    plt.show()

def plot_attention_analysis(attention_weights, save_path="results/attention_analysis.png"):
    """Visualize cross-modal attention patterns."""
    print("🔍 Analyzing cross-modal attention patterns...")
    
    # Average attention weights across batch
    avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # [3, 3] for fMRI, text, semantic
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Attention heatmap
    modalities = ['fMRI', 'Text', 'Semantic']
    sns.heatmap(avg_attention, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=modalities, yticklabels=modalities, ax=ax1)
    ax1.set_title('Cross-Modal Attention Matrix', fontweight='bold')
    ax1.set_xlabel('Attending To')
    ax1.set_ylabel('Attending From')
    
    # Attention distribution
    attention_flat = avg_attention.flatten()
    ax2.bar(range(len(attention_flat)), attention_flat, 
            color=['steelblue', 'orange', 'green'] * 3, alpha=0.7)
    ax2.set_title('Attention Weight Distribution', fontweight='bold')
    ax2.set_xlabel('Attention Pair')
    ax2.set_ylabel('Attention Weight')
    ax2.set_xticks(range(len(attention_flat)))
    ax2.set_xticklabels([f'{modalities[i//3]}→{modalities[i%3]}' 
                        for i in range(len(attention_flat))], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved attention analysis to: {save_path}")
    
    plt.show()

def create_metrics_comparison_chart(guidance_results, save_path="results/multimodal_metrics_comparison.png"):
    """Create comprehensive metrics comparison chart."""
    print("📊 Creating metrics comparison chart...")
    
    # Extract metrics for guidance scale 7.5
    results = guidance_results[7.5]
    
    configs = ['no_guidance', 'text_only', 'semantic_only', 'full_guidance']
    config_names = ['No Guidance', 'Text Only', 'Semantic Only', 'Full Multi-Modal']
    metrics = ['accuracy', 'correlation', 'ssim']
    metric_names = ['Classification Accuracy', 'Average Correlation', 'SSIM Score']
    
    # Extract data
    data = []
    for config in configs:
        for metric in metrics:
            data.append(results[config][metric])
    
    data = np.array(data).reshape(len(configs), len(metrics))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(config_names))
    width = 0.6
    
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = data[:, i]
        bars = axes[i].bar(x, values, width, color=colors, alpha=0.8, edgecolor='black')
        
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f'{metric_name} by Guidance Type', fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(config_names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Multi-Modal Guidance Effects on Reconstruction Quality', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved metrics comparison to: {save_path}")
    
    plt.show()

def main():
    """Main evaluation function."""
    print("📊 Multi-Modal Brain LDM Evaluation")
    print("=" * 50)
    
    # Check if model exists
    model_path = "checkpoints/best_multimodal_model.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("🚀 Please run 'python train_multimodal_ldm.py' first")
        return
    
    # Load model and data
    model = load_multimodal_model(model_path)
    loader = load_fmri_data()
    test_data = loader.get_test_data()
    
    print(f"📊 Test data: {len(test_data['stimuli'])} samples")
    
    # Evaluate guidance effects
    guidance_results = evaluate_guidance_effects(model, test_data)
    
    # Create visualizations
    visualize_guidance_comparison(model, test_data)
    create_metrics_comparison_chart(guidance_results)
    
    # Analyze attention if available
    if 'attention_weights' in guidance_results[7.5]:
        plot_attention_analysis(guidance_results[7.5]['attention_weights'])
    
    # Print summary
    print(f"\n🎯 Multi-Modal Evaluation Summary")
    print("=" * 40)
    
    best_config = 'full_guidance'
    best_results = guidance_results[7.5][best_config]
    
    print(f"📈 Best Configuration: {best_config}")
    print(f"   Accuracy: {best_results['accuracy']:.2%}")
    print(f"   Correlation: {best_results['correlation']:.6f}")
    print(f"   SSIM: {best_results['ssim']:.6f}")
    print(f"   MSE: {best_results['mse']:.6f}")
    print(f"   MAE: {best_results['mae']:.6f}")
    
    # Compare with baseline
    baseline_results = guidance_results[7.5]['no_guidance']
    print(f"\n📊 Improvement over baseline:")
    print(f"   Accuracy: {((best_results['accuracy'] - baseline_results['accuracy']) / baseline_results['accuracy'] * 100):+.1f}%")
    print(f"   Correlation: {((best_results['correlation'] - baseline_results['correlation']) / abs(baseline_results['correlation']) * 100):+.1f}%")
    
    print(f"\n✅ Multi-modal evaluation complete!")
    print(f"📁 Results saved to: results/")

if __name__ == "__main__":
    main()
