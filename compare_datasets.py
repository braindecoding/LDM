#!/usr/bin/env python3
"""
🧠 Compare Miyawaki vs Vangerven Dataset Results
Comprehensive comparison of Brain LDM performance on both datasets.
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.data_loader import load_fmri_data
from models.improved_brain_ldm import ImprovedBrainLDM

def load_model_and_evaluate(checkpoint_path, data_path, dataset_name, device='cuda'):
    """Load model and evaluate on dataset."""
    
    print(f"\n📊 Evaluating {dataset_name} Dataset")
    print("=" * 40)
    
    # Load data
    loader = load_fmri_data(data_path, device=device)
    
    # Load model
    if not Path(checkpoint_path).exists():
        print(f"❌ Model not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    fmri_dim = config.get('fmri_dim', 967)
    
    # Create model
    model = ImprovedBrainLDM(
        fmri_dim=fmri_dim,
        image_size=28,
        guidance_scale=7.5
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create decoder
    decoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 784),
        nn.Sigmoid()
    ).to(device)
    
    if 'decoder_state_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()
    
    # Get test data
    test_fmri = loader.get_fmri('test')[:8].to(device)
    test_stimuli = loader.get_stimuli('test')[:8]
    test_labels = loader.get_labels('test')[:8]
    
    # Generate reconstructions
    with torch.no_grad():
        fmri_features = model.fmri_encoder(test_fmri)
        reconstructions = decoder(fmri_features)
        
        targets = test_stimuli.cpu().numpy()
        recons = reconstructions.cpu().numpy()
        labels = test_labels.cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((targets - recons) ** 2)
    mae = np.mean(np.abs(targets - recons))
    correlation = np.corrcoef(targets.flatten(), recons.flatten())[0, 1]
    
    # Individual correlations
    individual_corrs = []
    for i in range(len(targets)):
        corr = np.corrcoef(targets[i].flatten(), recons[i].flatten())[0, 1]
        if not np.isnan(corr):
            individual_corrs.append(corr)
    
    avg_individual_corr = np.mean(individual_corrs) if individual_corrs else 0
    
    results = {
        'dataset_name': dataset_name,
        'fmri_dim': fmri_dim,
        'training_loss': checkpoint.get('best_loss', 'N/A'),
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'avg_individual_corr': avg_individual_corr,
        'targets': targets,
        'reconstructions': recons,
        'labels': labels,
        'num_samples': len(targets)
    }
    
    print(f"✅ {dataset_name} evaluation complete")
    print(f"   MSE: {mse:.6f}")
    print(f"   Correlation: {correlation:.6f}")
    print(f"   Training Loss: {checkpoint.get('best_loss', 'N/A')}")
    
    return results

def create_comparison_visualization(miyawaki_results, vangerven_results):
    """Create comprehensive comparison visualization."""
    
    print(f"\n🎨 Creating Comparison Visualization")
    print("=" * 40)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Dataset comparison metrics
    ax1 = plt.subplot(2, 3, 1)
    datasets = ['Miyawaki', 'Vangerven']
    mse_values = [miyawaki_results['mse'], vangerven_results['mse']]
    correlation_values = [miyawaki_results['correlation'], vangerven_results['correlation']]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax1.bar(x - width/2, mse_values, width, label='MSE', alpha=0.8, color='red')
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + width/2, correlation_values, width, label='Correlation', alpha=0.8, color='blue')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('MSE', color='red')
    ax1_twin.set_ylabel('Correlation', color='blue')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Training loss comparison
    ax2 = plt.subplot(2, 3, 2)
    training_losses = [miyawaki_results['training_loss'], vangerven_results['training_loss']]
    bars = ax2.bar(datasets, training_losses, color=['green', 'orange'], alpha=0.7)
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Comparison')
    
    # Add value labels on bars
    for bar, loss in zip(bars, training_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.6f}', ha='center', va='bottom')
    
    # Dataset characteristics
    ax3 = plt.subplot(2, 3, 3)
    fmri_dims = [miyawaki_results['fmri_dim'], vangerven_results['fmri_dim']]
    num_samples = [miyawaki_results['num_samples'], vangerven_results['num_samples']]
    
    ax3.bar(x - width/2, fmri_dims, width, label='fMRI Dimension', alpha=0.8, color='purple')
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + width/2, num_samples, width, label='Test Samples', alpha=0.8, color='cyan')
    
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('fMRI Dimension', color='purple')
    ax3_twin.set_ylabel('Test Samples', color='cyan')
    ax3.set_title('Dataset Characteristics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # Sample reconstructions - Miyawaki
    ax4 = plt.subplot(2, 3, 4)
    miyawaki_sample = miyawaki_results['targets'][0].reshape(28, 28)
    miyawaki_recon = miyawaki_results['reconstructions'][0].reshape(28, 28)
    
    # Apply transformations
    miyawaki_sample = np.rot90(np.flipud(miyawaki_sample), k=-1)
    miyawaki_recon = np.rot90(np.flipud(miyawaki_recon), k=-1)
    
    combined = np.hstack([miyawaki_sample, miyawaki_recon])
    ax4.imshow(combined, cmap='gray', vmin=0, vmax=1)
    ax4.set_title(f'Miyawaki Sample\nTarget | Reconstruction')
    ax4.axis('off')
    
    # Sample reconstructions - Vangerven
    ax5 = plt.subplot(2, 3, 5)
    vangerven_sample = vangerven_results['targets'][0].reshape(28, 28)
    vangerven_recon = vangerven_results['reconstructions'][0].reshape(28, 28)
    
    # Apply transformations
    vangerven_sample = np.rot90(np.flipud(vangerven_sample), k=-1)
    vangerven_recon = np.rot90(np.flipud(vangerven_recon), k=-1)
    
    combined = np.hstack([vangerven_sample, vangerven_recon])
    ax5.imshow(combined, cmap='gray', vmin=0, vmax=1)
    ax5.set_title(f'Vangerven Sample\nTarget | Reconstruction')
    ax5.axis('off')
    
    # Summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'Miyawaki', 'Vangerven'],
        ['Training Loss', f"{miyawaki_results['training_loss']:.6f}", f"{vangerven_results['training_loss']:.6f}"],
        ['MSE', f"{miyawaki_results['mse']:.6f}", f"{vangerven_results['mse']:.6f}"],
        ['Correlation', f"{miyawaki_results['correlation']:.6f}", f"{vangerven_results['correlation']:.6f}"],
        ['fMRI Dimension', f"{miyawaki_results['fmri_dim']}", f"{vangerven_results['fmri_dim']}"],
        ['Test Samples', f"{miyawaki_results['num_samples']}", f"{vangerven_results['num_samples']}"]
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Performance Summary')
    
    plt.suptitle('Brain LDM: Miyawaki vs Vangerven Dataset Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    output_path = "results/dataset_comparison.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Comparison saved: {output_path}")
    
    plt.show()

def main():
    """Main comparison function."""
    
    print("🧠 BRAIN LDM DATASET COMPARISON")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Evaluate Miyawaki dataset
    miyawaki_results = load_model_and_evaluate(
        checkpoint_path="checkpoints/best_miyawaki_simple_model.pt",
        data_path="data/miyawaki_structured_28x28.mat",
        dataset_name="Miyawaki",
        device=device
    )
    
    # Evaluate Vangerven dataset
    vangerven_results = load_model_and_evaluate(
        checkpoint_path="checkpoints/best_vangerven_simple_model.pt",
        data_path="data/digit69_28x28.mat",
        dataset_name="Vangerven",
        device=device
    )
    
    if miyawaki_results and vangerven_results:
        # Create comparison visualization
        create_comparison_visualization(miyawaki_results, vangerven_results)
        
        # Print detailed comparison
        print(f"\n📊 DETAILED COMPARISON")
        print("=" * 25)
        print(f"🎯 Miyawaki Dataset:")
        print(f"   • Training Loss: {miyawaki_results['training_loss']:.6f}")
        print(f"   • MSE: {miyawaki_results['mse']:.6f}")
        print(f"   • Correlation: {miyawaki_results['correlation']:.6f}")
        print(f"   • fMRI Dimension: {miyawaki_results['fmri_dim']} voxels")
        
        print(f"\n🎯 Vangerven Dataset:")
        print(f"   • Training Loss: {vangerven_results['training_loss']:.6f}")
        print(f"   • MSE: {vangerven_results['mse']:.6f}")
        print(f"   • Correlation: {vangerven_results['correlation']:.6f}")
        print(f"   • fMRI Dimension: {vangerven_results['fmri_dim']} voxels")
        
        # Determine winner
        print(f"\n🏆 PERFORMANCE WINNER:")
        if vangerven_results['correlation'] > miyawaki_results['correlation']:
            print(f"   🥇 Vangerven dataset shows better reconstruction quality!")
            print(f"   📈 Higher correlation: {vangerven_results['correlation']:.6f} vs {miyawaki_results['correlation']:.6f}")
        else:
            print(f"   🥇 Miyawaki dataset shows better reconstruction quality!")
            print(f"   📈 Higher correlation: {miyawaki_results['correlation']:.6f} vs {vangerven_results['correlation']:.6f}")
        
        print(f"\n📁 Results saved to: results/dataset_comparison.png")
    else:
        print("❌ Could not complete comparison - missing models")

if __name__ == "__main__":
    main()
