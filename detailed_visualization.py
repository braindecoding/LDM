"""
🎨 Detailed Brain LDM Visualization
Comprehensive visualization including stimulus vs reconstruction and correlation matrix.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import seaborn as sns
from data_loader import load_fmri_data
from brain_ldm import BrainLDM
import scipy.stats

def load_model(checkpoint_path="checkpoints/best_model.pt", device='cpu'):
    """Load trained model."""
    print(f"📁 Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model with correct parameters
    model = BrainLDM(
        fmri_dim=3092,
        image_size=28,
        latent_channels=4,
        latent_size=7
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})")
    return model

def generate_reconstructions(model, fmri_data, num_samples=10):
    """Generate reconstructions from fMRI data."""
    print(f"🧠 Generating {num_samples} reconstructions...")

    with torch.no_grad():
        # Take first num_samples
        fmri_batch = fmri_data[:num_samples]

        # Generate reconstructions using the correct method name
        reconstructions = model.generate_from_fmri(fmri_batch, num_inference_steps=20)

        # Flatten to match expected format (batch_size, 784)
        reconstructions = reconstructions.view(reconstructions.shape[0], -1)

    return reconstructions

def plot_stimulus_vs_reconstruction(stimuli, reconstructions, labels=None, save_path="results/stimulus_vs_reconstruction.png"):
    """Plot stimulus vs reconstruction comparison."""
    num_samples = min(len(stimuli), len(reconstructions), 10)

    fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))

    for i in range(num_samples):
        # Original stimulus
        stimulus_img = stimuli[i].cpu().numpy().reshape(28, 28)
        axes[0, i].imshow(stimulus_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}' + (f'\nLabel: {labels[i].item()}' if labels is not None else ''))
        axes[0, i].axis('off')

        # Reconstruction
        recon_img = reconstructions[i].cpu().numpy().reshape(28, 28)
        axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Reconstruction {i+1}')
        axes[1, i].axis('off')

        # Difference
        diff_img = np.abs(stimulus_img - recon_img)
        im = axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title(f'Difference {i+1}')
        axes[2, i].axis('off')

    # Add row labels
    axes[0, 0].set_ylabel('Original', rotation=90, size='large')
    axes[1, 0].set_ylabel('Reconstruction', rotation=90, size='large')
    axes[2, 0].set_ylabel('Difference', rotation=90, size='large')

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved stimulus vs reconstruction plot to: {save_path}")

    plt.show()

def compute_correlation_matrix(stimuli, reconstructions):
    """Compute correlation matrix between stimuli and reconstructions."""
    print("📊 Computing correlation matrix...")

    # Flatten images
    stimuli_flat = stimuli.cpu().numpy().reshape(len(stimuli), -1)
    recons_flat = reconstructions.cpu().numpy().reshape(len(reconstructions), -1)

    # Compute correlation matrix
    num_samples = len(stimuli)
    corr_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(num_samples):
            corr, _ = scipy.stats.pearsonr(stimuli_flat[i], recons_flat[j])
            corr_matrix[i, j] = corr if not np.isnan(corr) else 0

    return corr_matrix

def plot_correlation_matrix(corr_matrix, save_path="results/correlation_matrix.png"):
    """Plot correlation matrix."""
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation'})

    plt.title('Correlation Matrix: Original Stimuli vs Reconstructions', fontsize=14, fontweight='bold')
    plt.xlabel('Reconstruction Index', fontsize=12)
    plt.ylabel('Original Stimulus Index', fontsize=12)

    # Add diagonal line to highlight perfect matches
    plt.plot([0, len(corr_matrix)], [0, len(corr_matrix)], 'k--', alpha=0.5, linewidth=2)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved correlation matrix to: {save_path}")

    plt.show()

    return corr_matrix

def analyze_correlations(corr_matrix):
    """Analyze correlation matrix."""
    print("\n📈 Correlation Analysis:")
    print("=" * 50)

    # Diagonal correlations (correct matches)
    diagonal_corrs = np.diag(corr_matrix)
    print(f"🎯 Diagonal correlations (correct matches):")
    print(f"  Mean: {diagonal_corrs.mean():.4f}")
    print(f"  Std:  {diagonal_corrs.std():.4f}")
    print(f"  Min:  {diagonal_corrs.min():.4f}")
    print(f"  Max:  {diagonal_corrs.max():.4f}")

    # Off-diagonal correlations (incorrect matches)
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    off_diagonal_corrs = corr_matrix[mask]
    print(f"\n❌ Off-diagonal correlations (incorrect matches):")
    print(f"  Mean: {off_diagonal_corrs.mean():.4f}")
    print(f"  Std:  {off_diagonal_corrs.std():.4f}")
    print(f"  Min:  {off_diagonal_corrs.min():.4f}")
    print(f"  Max:  {off_diagonal_corrs.max():.4f}")

    # Classification accuracy (highest correlation on diagonal)
    correct_matches = 0
    for i in range(len(corr_matrix)):
        if np.argmax(corr_matrix[i, :]) == i:
            correct_matches += 1

    accuracy = correct_matches / len(corr_matrix)
    print(f"\n🎯 Classification Accuracy: {accuracy:.2%} ({correct_matches}/{len(corr_matrix)})")

    return {
        'diagonal_mean': diagonal_corrs.mean(),
        'diagonal_std': diagonal_corrs.std(),
        'off_diagonal_mean': off_diagonal_corrs.mean(),
        'off_diagonal_std': off_diagonal_corrs.std(),
        'accuracy': accuracy
    }

def create_combined_visualization(stimuli, reconstructions, corr_matrix, labels=None, save_path="results/combined_analysis.png"):
    """Create a combined visualization with stimulus comparison and correlation matrix."""
    fig = plt.figure(figsize=(16, 10))

    # Top section: Stimulus vs Reconstruction (first 5 samples)
    num_samples = min(5, len(stimuli))

    for i in range(num_samples):
        # Original stimulus
        ax1 = plt.subplot(4, num_samples, i + 1)
        stimulus_img = stimuli[i].cpu().numpy().reshape(28, 28)
        plt.imshow(stimulus_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Original {i+1}' + (f'\nLabel: {labels[i].item()}' if labels is not None else ''))
        plt.axis('off')

        # Reconstruction
        ax2 = plt.subplot(4, num_samples, i + 1 + num_samples)
        recon_img = reconstructions[i].cpu().numpy().reshape(28, 28)
        plt.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Reconstruction {i+1}')
        plt.axis('off')

    # Bottom section: Correlation matrix
    ax3 = plt.subplot(2, 1, 2)
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation'})

    plt.title('Correlation Matrix: Original Stimuli vs Reconstructions', fontsize=12, fontweight='bold')
    plt.xlabel('Reconstruction Index')
    plt.ylabel('Original Stimulus Index')

    # Add diagonal line
    plt.plot([0, len(corr_matrix)], [0, len(corr_matrix)], 'k--', alpha=0.5, linewidth=2)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved combined visualization to: {save_path}")

    plt.show()

def main():
    """Main visualization function."""
    print("🎨 Detailed Brain LDM Visualization")
    print("=" * 50)

    device = 'cpu'

    # Load data
    print("📁 Loading data...")
    loader = load_fmri_data()
    test_data = loader.get_test_data()

    test_stimuli = test_data['stimuli']
    test_fmri = test_data['fmri']
    test_labels = test_data.get('labels', None)

    print(f"📊 Test data: {len(test_stimuli)} samples")

    # Load model
    model = load_model(device=device)

    # Generate reconstructions
    reconstructions = generate_reconstructions(model, test_fmri, num_samples=len(test_stimuli))

    # Plot stimulus vs reconstruction
    plot_stimulus_vs_reconstruction(
        test_stimuli,
        reconstructions,
        test_labels,
        save_path="results/detailed_stimulus_vs_reconstruction.png"
    )

    # Compute and plot correlation matrix
    corr_matrix = compute_correlation_matrix(test_stimuli, reconstructions)
    plot_correlation_matrix(corr_matrix, save_path="results/detailed_correlation_matrix.png")

    # Create combined visualization
    create_combined_visualization(
        test_stimuli,
        reconstructions,
        corr_matrix,
        test_labels,
        save_path="results/combined_stimulus_correlation_analysis.png"
    )

    # Analyze correlations
    analysis = analyze_correlations(corr_matrix)

    # Summary
    print(f"\n🎉 Detailed Visualization Complete!")
    print(f"📁 Results saved to: results/")
    print(f"🎯 Key Metrics:")
    print(f"  - Diagonal correlation: {analysis['diagonal_mean']:.4f} ± {analysis['diagonal_std']:.4f}")
    print(f"  - Classification accuracy: {analysis['accuracy']:.2%}")

if __name__ == "__main__":
    main()
