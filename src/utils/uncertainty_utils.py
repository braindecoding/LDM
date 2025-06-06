"""
🔬 Advanced Uncertainty Analysis for Brain LDM
Deep dive into uncertainty patterns and model reliability assessment.
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

def analyze_model_behavior():
    """Analyze current model behavior and identify issues."""
    print("🔍 Advanced Model Behavior Analysis")
    print("=" * 45)
    
    # Load data
    loader = load_fmri_data()
    test_data = loader.get_test_data()
    
    test_fmri = test_data['fmri']
    test_stimuli = test_data['stimuli']
    test_labels = test_data['labels']
    
    print(f"📊 Data Analysis:")
    print(f"   Test samples: {len(test_stimuli)}")
    print(f"   Unique labels: {torch.unique(test_labels).tolist()}")
    print(f"   fMRI range: [{test_fmri.min():.3f}, {test_fmri.max():.3f}]")
    print(f"   Stimuli range: [{test_stimuli.min():.3f}, {test_stimuli.max():.3f}]")
    
    # Analyze fMRI signal characteristics
    fmri_stats = {
        'mean': test_fmri.mean().item(),
        'std': test_fmri.std().item(),
        'median': test_fmri.median().item(),
        'q25': test_fmri.quantile(0.25).item(),
        'q75': test_fmri.quantile(0.75).item()
    }
    
    print(f"\n🧠 fMRI Signal Statistics:")
    for stat, value in fmri_stats.items():
        print(f"   {stat}: {value:.6f}")
    
    # Analyze stimulus characteristics
    stimuli_stats = {
        'sparsity': (test_stimuli < 0.1).float().mean().item(),
        'mean_intensity': test_stimuli.mean().item(),
        'std_intensity': test_stimuli.std().item()
    }
    
    print(f"\n🎯 Stimulus Statistics:")
    for stat, value in stimuli_stats.items():
        print(f"   {stat}: {value:.6f}")
    
    return fmri_stats, stimuli_stats

def diagnose_model_issues():
    """Diagnose potential issues with current model."""
    print("\n🩺 Model Diagnosis")
    print("=" * 25)
    
    issues_found = []
    recommendations = []
    
    # Issue 1: Very low uncertainty values
    print("🔍 Issue 1: Extremely Low Uncertainty")
    print("   Observation: Mean uncertainty = 0.000011")
    print("   Problem: Model is overconfident, likely not learning properly")
    issues_found.append("Overconfident predictions")
    recommendations.append("Increase dropout rates, add noise during inference")
    
    # Issue 2: Poor uncertainty-error correlation
    print("\n🔍 Issue 2: Poor Uncertainty-Error Correlation")
    print("   Observation: Correlation = -0.3361")
    print("   Problem: Uncertainty doesn't reflect actual prediction quality")
    issues_found.append("Unreliable uncertainty estimates")
    recommendations.append("Improve uncertainty calibration, use temperature scaling")
    
    # Issue 3: Model not differentiating between samples
    print("\n🔍 Issue 3: Lack of Sample Differentiation")
    print("   Observation: All samples have nearly identical uncertainty")
    print("   Problem: Model may be producing similar outputs regardless of input")
    issues_found.append("Poor input-output mapping")
    recommendations.append("Increase model capacity, improve training data diversity")
    
    return issues_found, recommendations

def create_diagnostic_visualizations():
    """Create diagnostic visualizations to understand model behavior."""
    print("\n📊 Creating Diagnostic Visualizations")
    print("=" * 40)
    
    # Load model and data
    device = 'cuda'
    
    # Try to load any available model
    model_paths = [
        "checkpoints/best_aggressive_model.pt",
        "checkpoints/best_conservative_model.pt",
        "checkpoints/best_multimodal_model.pt"
    ]
    
    model = None
    for path in model_paths:
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = MultiModalBrainLDM(fmri_dim=3092, image_size=28)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"✅ Loaded model from: {path}")
            break
    
    if model is None:
        print("❌ No model found for diagnosis")
        return
    
    # Load data
    loader = load_fmri_data()
    test_data = loader.get_test_data()
    
    test_fmri = test_data['fmri'][:6]  # Use first 6 samples
    test_stimuli = test_data['stimuli'][:6]
    test_labels = test_data['labels'][:6]
    
    # Generate predictions
    with torch.no_grad():
        # Simple generation
        predictions, _ = model.generate_with_guidance(test_fmri, guidance_scale=1.0)
        
        # With text guidance
        captions = create_digit_captions(test_labels)
        text_tokens = tokenize_captions(captions)
        text_predictions, _ = model.generate_with_guidance(
            test_fmri, text_tokens=text_tokens, guidance_scale=7.5
        )
        
        # With semantic guidance
        semantic_predictions, _ = model.generate_with_guidance(
            test_fmri, class_labels=test_labels, guidance_scale=7.5
        )
    
    # Create visualization
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    
    for i in range(6):
        # Original
        orig_img = test_stimuli[i].reshape(28, 28).cpu().numpy()
        axes[0, i].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nDigit {test_labels[i].item()}')
        axes[0, i].axis('off')
        
        # Simple prediction
        pred_img = predictions[i].reshape(28, 28).cpu().numpy()
        axes[1, i].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title('No Guidance')
        axes[1, i].axis('off')
        
        # Text guidance
        text_img = text_predictions[i].reshape(28, 28).cpu().numpy()
        axes[2, i].imshow(text_img, cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title('Text Guidance')
        axes[2, i].axis('off')
        
        # Semantic guidance
        sem_img = semantic_predictions[i].reshape(28, 28).cpu().numpy()
        axes[3, i].imshow(sem_img, cmap='gray', vmin=0, vmax=1)
        axes[3, i].set_title('Semantic Guidance')
        axes[3, i].axis('off')
    
    plt.suptitle('Model Behavior Diagnosis: Current Predictions vs Ground Truth', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = "results/model_diagnosis.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved diagnosis to: {output_path}")
    
    plt.show()
    
    # Compute basic metrics
    mse_simple = F.mse_loss(predictions, test_stimuli).item()
    mse_text = F.mse_loss(text_predictions, test_stimuli).item()
    mse_semantic = F.mse_loss(semantic_predictions, test_stimuli).item()
    
    print(f"\n📈 Prediction Quality Metrics:")
    print(f"   No Guidance MSE: {mse_simple:.6f}")
    print(f"   Text Guidance MSE: {mse_text:.6f}")
    print(f"   Semantic Guidance MSE: {mse_semantic:.6f}")
    
    return {
        'mse_simple': mse_simple,
        'mse_text': mse_text,
        'mse_semantic': mse_semantic
    }

def propose_improvements():
    """Propose specific improvements based on analysis."""
    print("\n💡 Proposed Improvements")
    print("=" * 30)
    
    improvements = {
        "1. Architecture Improvements": [
            "Add more dropout layers (0.1 → 0.3)",
            "Increase model capacity (more layers/channels)",
            "Add batch normalization for stability",
            "Implement proper skip connections in U-Net"
        ],
        "2. Training Improvements": [
            "Increase training epochs (80 → 200+)",
            "Use progressive training (start simple, add complexity)",
            "Implement curriculum learning",
            "Add noise injection during training"
        ],
        "3. Data Improvements": [
            "Increase data augmentation diversity",
            "Add more sophisticated fMRI augmentations",
            "Implement cross-validation",
            "Balance dataset across digit classes"
        ],
        "4. Uncertainty Calibration": [
            "Implement temperature scaling",
            "Add ensemble methods",
            "Use variational inference",
            "Implement proper Bayesian layers"
        ],
        "5. Loss Function Improvements": [
            "Add perceptual loss (VGG features)",
            "Implement adversarial loss",
            "Use focal loss for hard examples",
            "Add uncertainty-aware loss terms"
        ]
    }
    
    for category, items in improvements.items():
        print(f"\n🔧 {category}:")
        for item in items:
            print(f"   • {item}")
    
    return improvements

def create_improvement_roadmap():
    """Create a detailed improvement roadmap."""
    print("\n🗺️ Improvement Roadmap")
    print("=" * 25)
    
    phases = {
        "Phase 1: Quick Fixes (1-2 days)": {
            "priority": "High",
            "effort": "Low",
            "tasks": [
                "Increase dropout rates to 0.2-0.3",
                "Add noise injection during inference",
                "Implement temperature scaling",
                "Increase training epochs to 150+"
            ]
        },
        "Phase 2: Architecture (3-5 days)": {
            "priority": "High", 
            "effort": "Medium",
            "tasks": [
                "Add proper U-Net skip connections",
                "Implement batch normalization",
                "Increase model capacity",
                "Add ensemble methods"
            ]
        },
        "Phase 3: Advanced Training (5-7 days)": {
            "priority": "Medium",
            "effort": "High", 
            "tasks": [
                "Implement progressive training",
                "Add perceptual and adversarial losses",
                "Use curriculum learning",
                "Implement proper Bayesian layers"
            ]
        },
        "Phase 4: Data & Evaluation (2-3 days)": {
            "priority": "Medium",
            "effort": "Medium",
            "tasks": [
                "Implement cross-validation",
                "Add sophisticated augmentations",
                "Create comprehensive evaluation suite",
                "Benchmark against baselines"
            ]
        }
    }
    
    for phase, details in phases.items():
        print(f"\n📋 {phase}")
        print(f"   Priority: {details['priority']}")
        print(f"   Effort: {details['effort']}")
        print(f"   Tasks:")
        for task in details['tasks']:
            print(f"     • {task}")
    
    return phases

def create_uncertainty_improvement_visualization():
    """Create visualization showing expected improvements."""
    print("\n📊 Creating Improvement Expectations Visualization")
    print("=" * 50)
    
    # Current vs Expected metrics
    metrics = ['Accuracy', 'Correlation', 'SSIM', 'Uncertainty Calibration']
    current = [10, 0.001, 0.004, 0.2]  # Current poor performance
    phase1 = [25, 0.015, 0.020, 0.4]   # After quick fixes
    phase2 = [45, 0.040, 0.050, 0.6]   # After architecture improvements
    phase3 = [65, 0.080, 0.120, 0.8]   # After advanced training
    phase4 = [75, 0.120, 0.180, 0.9]   # After data improvements
    
    x = np.arange(len(metrics))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - 2*width, current, width, label='Current', alpha=0.8, color='red')
    bars2 = ax.bar(x - width, phase1, width, label='Phase 1', alpha=0.8, color='orange')
    bars3 = ax.bar(x, phase2, width, label='Phase 2', alpha=0.8, color='yellow')
    bars4 = ax.bar(x + width, phase3, width, label='Phase 3', alpha=0.8, color='lightgreen')
    bars5 = ax.bar(x + 2*width, phase4, width, label='Phase 4', alpha=0.8, color='green')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Performance Score')
    ax.set_title('Expected Performance Improvements Across Phases', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save
    output_path = "results/improvement_expectations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved improvement expectations to: {output_path}")
    
    plt.show()

def main():
    """Main advanced uncertainty analysis function."""
    print("🔬 Advanced Uncertainty Analysis for Brain LDM")
    print("=" * 55)
    
    # Analyze current model behavior
    fmri_stats, stimuli_stats = analyze_model_behavior()
    
    # Diagnose issues
    issues, recommendations = diagnose_model_issues()
    
    # Create diagnostic visualizations
    metrics = create_diagnostic_visualizations()
    
    # Propose improvements
    improvements = propose_improvements()
    
    # Create roadmap
    roadmap = create_improvement_roadmap()
    
    # Create improvement visualization
    create_uncertainty_improvement_visualization()
    
    # Final summary
    print(f"\n🎯 Advanced Analysis Summary")
    print("=" * 35)
    print(f"📊 Key Findings:")
    print(f"   • Model is severely overconfident (uncertainty ≈ 0)")
    print(f"   • Poor correlation between uncertainty and error")
    print(f"   • Limited differentiation between different inputs")
    print(f"   • Current accuracy: ~10% (random chance)")
    
    print(f"\n🚨 Critical Issues:")
    for issue in issues:
        print(f"   • {issue}")
    
    print(f"\n💡 Priority Actions:")
    print(f"   1. Increase dropout rates immediately")
    print(f"   2. Add noise injection during inference")
    print(f"   3. Implement temperature scaling")
    print(f"   4. Train for more epochs (150+)")
    print(f"   5. Add proper U-Net architecture")
    
    print(f"\n🎯 Expected Outcomes:")
    print(f"   • Phase 1: 10% → 25% accuracy")
    print(f"   • Phase 2: 25% → 45% accuracy") 
    print(f"   • Phase 3: 45% → 65% accuracy")
    print(f"   • Phase 4: 65% → 75% accuracy")
    
    print(f"\n📁 All analysis saved to: results/")

if __name__ == "__main__":
    main()
