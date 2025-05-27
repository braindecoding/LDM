"""
🎨 Plot Stimulus vs Reconstruction Results
Visualize the actual reconstruction quality of our improved Brain LDM model.
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

def load_best_model():
    """Load the best improved model."""
    print("📁 Loading best improved model...")
    
    device = 'cpu'
    model_path = "checkpoints/best_improved_v1_model.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        # Try alternatives
        alternatives = [
            "checkpoints/best_aggressive_model.pt",
            "checkpoints/best_conservative_model.pt",
            "checkpoints/best_multimodal_model.pt"
        ]
        
        for alt_path in alternatives:
            if Path(alt_path).exists():
                print(f"✅ Using alternative: {alt_path}")
                # Load with original architecture for fallback
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
    
    print(f"✅ Improved model loaded!")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Loss: {checkpoint.get('loss', 'Unknown'):.6f}")
    
    return model, "improved"

def generate_reconstructions(model, test_data):
    """Generate reconstructions with different guidance types."""
    print("🎨 Generating reconstructions...")
    
    test_fmri = improved_fmri_normalization(test_data['fmri'])
    test_stimuli = test_data['stimuli']
    test_labels = test_data['labels']
    
    # Create text tokens
    captions = create_digit_captions(test_labels)
    text_tokens = tokenize_captions(captions)
    
    reconstructions = {}
    
    with torch.no_grad():
        # 1. No guidance (fMRI only)
        print("   Generating: No guidance...")
        recons_no_guidance, _ = model.generate_with_guidance(
            test_fmri, guidance_scale=1.0
        )
        reconstructions['no_guidance'] = recons_no_guidance
        
        # 2. Text guidance
        print("   Generating: Text guidance...")
        recons_text, _ = model.generate_with_guidance(
            test_fmri, text_tokens=text_tokens, guidance_scale=7.5
        )
        reconstructions['text_guidance'] = recons_text
        
        # 3. Semantic guidance
        print("   Generating: Semantic guidance...")
        recons_semantic, _ = model.generate_with_guidance(
            test_fmri, class_labels=test_labels, guidance_scale=7.5
        )
        reconstructions['semantic_guidance'] = recons_semantic
        
        # 4. Full multi-modal guidance
        print("   Generating: Full multi-modal...")
        recons_full, attention_weights = model.generate_with_guidance(
            test_fmri, text_tokens=text_tokens, class_labels=test_labels, 
            guidance_scale=7.5
        )
        reconstructions['full_guidance'] = recons_full
    
    return reconstructions, test_stimuli, test_labels

def compute_reconstruction_metrics(reconstructions, targets):
    """Compute detailed reconstruction metrics."""
    metrics = {}
    
    for guidance_type, recons in reconstructions.items():
        # Reshape if needed
        if recons.dim() == 4:  # [batch, channels, height, width]
            recons_flat = recons.view(recons.size(0), -1)
        else:
            recons_flat = recons
        
        if targets.dim() == 2:  # [batch, features]
            targets_flat = targets
        else:
            targets_flat = targets.view(targets.size(0), -1)
        
        # Basic metrics
        mse = F.mse_loss(recons_flat, targets_flat).item()
        mae = F.l1_loss(recons_flat, targets_flat).item()
        
        # Correlation per sample
        correlations = []
        for i in range(len(recons_flat)):
            corr = np.corrcoef(recons_flat[i].cpu().numpy(), 
                              targets_flat[i].cpu().numpy())[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        avg_correlation = np.mean(correlations)
        
        # Classification accuracy (correlation matrix)
        n_samples = len(recons_flat)
        corr_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                corr = np.corrcoef(recons_flat[i].cpu().numpy(), 
                                  targets_flat[j].cpu().numpy())[0, 1]
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0
        
        correct_matches = sum(np.argmax(corr_matrix[i, :]) == i for i in range(n_samples))
        accuracy = correct_matches / n_samples
        
        metrics[guidance_type] = {
            'mse': mse,
            'mae': mae,
            'correlation': avg_correlation,
            'accuracy': accuracy,
            'correlations': correlations
        }
    
    return metrics

def create_comprehensive_reconstruction_plot(reconstructions, targets, labels, metrics, 
                                           save_path="results/final_reconstruction_results.png"):
    """Create comprehensive reconstruction visualization."""
    print("🎨 Creating comprehensive reconstruction plot...")
    
    n_samples = len(targets)
    guidance_types = ['no_guidance', 'text_guidance', 'semantic_guidance', 'full_guidance']
    guidance_names = ['No Guidance', 'Text Guidance', 'Semantic Guidance', 'Full Multi-Modal']
    
    # Create figure
    fig, axes = plt.subplots(5, n_samples, figsize=(3*n_samples, 15))
    
    # Plot original stimuli
    for i in range(n_samples):
        if targets.dim() == 2:
            img = targets[i].reshape(28, 28).cpu().numpy()
        else:
            img = targets[i].squeeze().cpu().numpy()
        
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nDigit {labels[i].item()}', fontsize=10, fontweight='bold')
        axes[0, i].axis('off')
    
    # Plot reconstructions
    for row, (guidance_type, guidance_name) in enumerate(zip(guidance_types, guidance_names), 1):
        recons = reconstructions[guidance_type]
        
        for i in range(n_samples):
            if recons.dim() == 4:  # [batch, channels, height, width]
                img = recons[i].squeeze().cpu().numpy()
            else:
                img = recons[i].reshape(28, 28).cpu().numpy()
            
            axes[row, i].imshow(img, cmap='gray', vmin=0, vmax=1)
            
            # Add correlation score
            corr = metrics[guidance_type]['correlations'][i]
            axes[row, i].set_title(f'r={corr:.3f}', fontsize=9)
            axes[row, i].axis('off')
        
        # Add row label with metrics
        acc = metrics[guidance_type]['accuracy']
        avg_corr = metrics[guidance_type]['correlation']
        mse = metrics[guidance_type]['mse']
        
        row_label = f'{guidance_name}\nAcc: {acc:.1%}\nCorr: {avg_corr:.3f}\nMSE: {mse:.4f}'
        axes[row, 0].set_ylabel(row_label, rotation=90, fontsize=10, fontweight='bold', va='center')
    
    plt.suptitle('Brain-to-Image Reconstruction Results: Multi-Modal Guidance Effects', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, top=0.90)
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved reconstruction results to: {save_path}")
    
    plt.show()

def create_metrics_comparison_chart(metrics, save_path="results/reconstruction_metrics_comparison.png"):
    """Create detailed metrics comparison chart."""
    print("📊 Creating metrics comparison chart...")
    
    guidance_types = list(metrics.keys())
    guidance_names = ['No Guidance', 'Text Guidance', 'Semantic Guidance', 'Full Multi-Modal']
    
    # Extract metrics
    accuracies = [metrics[gt]['accuracy'] * 100 for gt in guidance_types]
    correlations = [metrics[gt]['correlation'] for gt in guidance_types]
    mse_values = [metrics[gt]['mse'] for gt in guidance_types]
    mae_values = [metrics[gt]['mae'] for gt in guidance_types]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
    x = np.arange(len(guidance_names))
    
    # Accuracy
    bars1 = ax1.bar(x, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Classification Accuracy (%)')
    ax1.set_title('Classification Accuracy by Guidance Type', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(guidance_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Correlation
    bars2 = ax2.bar(x, correlations, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Correlation')
    ax2.set_title('Reconstruction Correlation by Guidance Type', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(guidance_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, corr in zip(bars2, correlations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE
    bars3 = ax3.bar(x, mse_values, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Mean Squared Error')
    ax3.set_title('Reconstruction Error (MSE)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(guidance_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, mse in zip(bars3, mse_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{mse:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE
    bars4 = ax4.bar(x, mae_values, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Reconstruction Error (MAE)', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(guidance_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, mae in zip(bars4, mae_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{mae:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Reconstruction Quality Metrics: Multi-Modal Guidance Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved metrics comparison to: {save_path}")
    
    plt.show()

def print_detailed_results(metrics, model_type):
    """Print detailed reconstruction results."""
    print(f"\n📊 DETAILED RECONSTRUCTION RESULTS")
    print("=" * 45)
    print(f"🤖 Model Type: {model_type}")
    
    guidance_names = {
        'no_guidance': 'No Guidance (fMRI only)',
        'text_guidance': 'Text Guidance',
        'semantic_guidance': 'Semantic Guidance', 
        'full_guidance': 'Full Multi-Modal'
    }
    
    for guidance_type, guidance_name in guidance_names.items():
        if guidance_type in metrics:
            m = metrics[guidance_type]
            print(f"\n🎯 {guidance_name}:")
            print(f"   Classification Accuracy: {m['accuracy']:.2%}")
            print(f"   Average Correlation: {m['correlation']:.6f}")
            print(f"   Mean Squared Error: {m['mse']:.6f}")
            print(f"   Mean Absolute Error: {m['mae']:.6f}")
    
    # Find best performing guidance
    best_guidance = max(metrics.keys(), key=lambda k: metrics[k]['accuracy'])
    best_name = guidance_names[best_guidance]
    best_acc = metrics[best_guidance]['accuracy']
    
    print(f"\n🏆 BEST PERFORMING GUIDANCE:")
    print(f"   {best_name}")
    print(f"   Accuracy: {best_acc:.2%}")
    
    # Compare with baseline
    if 'no_guidance' in metrics and best_guidance != 'no_guidance':
        baseline_acc = metrics['no_guidance']['accuracy']
        improvement = ((best_acc - baseline_acc) / baseline_acc) * 100
        print(f"   Improvement over baseline: {improvement:+.1f}%")

def main():
    """Main reconstruction visualization function."""
    print("🎨 Brain LDM Reconstruction Results Visualization")
    print("=" * 55)
    
    # Load model
    model, model_type = load_best_model()
    if model is None:
        print("❌ Cannot proceed without model")
        return
    
    # Load test data
    loader = load_fmri_data()
    test_data = loader.get_test_data()
    
    print(f"📊 Test data: {len(test_data['stimuli'])} samples")
    
    # Generate reconstructions
    reconstructions, targets, labels = generate_reconstructions(model, test_data)
    
    # Compute metrics
    metrics = compute_reconstruction_metrics(reconstructions, targets)
    
    # Create visualizations
    create_comprehensive_reconstruction_plot(reconstructions, targets, labels, metrics)
    create_metrics_comparison_chart(metrics)
    
    # Print detailed results
    print_detailed_results(metrics, model_type)
    
    print(f"\n🎉 Reconstruction Visualization Complete!")
    print(f"📁 Results saved to: results/")
    print(f"💡 Check the generated plots to see actual reconstruction quality!")

if __name__ == "__main__":
    main()
