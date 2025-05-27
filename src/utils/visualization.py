"""
Visualization utilities for fMRI reconstruction and training monitoring.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_curves(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    title: str = "Training Curves"
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot total loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot reconstruction loss
    if 'train_recon_loss' in history and 'val_recon_loss' in history:
        axes[0, 1].plot(history['train_recon_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_recon_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot KL divergence if available
    if 'train_kl_loss' in history and 'val_kl_loss' in history:
        axes[1, 0].plot(history['train_kl_loss'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_kl_loss'], label='Validation', linewidth=2)
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot diffusion loss if available
    if 'train_diffusion_loss' in history and 'val_diffusion_loss' in history:
        axes[1, 1].plot(history['train_diffusion_loss'], label='Train', linewidth=2)
        axes[1, 1].plot(history['val_diffusion_loss'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Diffusion Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    return fig


def plot_reconstructions(
    original: np.ndarray,
    reconstructed: np.ndarray,
    num_samples: int = 5,
    save_path: Optional[str] = None,
    title: str = "fMRI Reconstructions"
) -> plt.Figure:
    """
    Plot original vs reconstructed fMRI data samples.
    
    Args:
        original: Original fMRI data [n_samples, n_voxels]
        reconstructed: Reconstructed fMRI data [n_samples, n_voxels]
        num_samples: Number of samples to plot
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, original.shape[0])
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16)
    
    for i in range(num_samples):
        # Original
        axes[i, 0].plot(original[i], alpha=0.8, linewidth=1)
        axes[i, 0].set_title(f'Original Sample {i+1}')
        axes[i, 0].set_xlabel('Voxel Index')
        axes[i, 0].set_ylabel('Activation')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Reconstructed
        axes[i, 1].plot(reconstructed[i], alpha=0.8, linewidth=1, color='orange')
        axes[i, 1].set_title(f'Reconstructed Sample {i+1}')
        axes[i, 1].set_xlabel('Voxel Index')
        axes[i, 1].set_ylabel('Activation')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reconstruction plots saved to {save_path}")
    
    return fig


def plot_correlation_analysis(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Correlation Analysis"
) -> plt.Figure:
    """
    Plot correlation analysis between original and reconstructed data.
    
    Args:
        original: Original fMRI data
        reconstructed: Reconstructed fMRI data
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Scatter plot of all values
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Sample points for visualization if too many
    if len(orig_flat) > 10000:
        indices = np.random.choice(len(orig_flat), 10000, replace=False)
        orig_sample = orig_flat[indices]
        recon_sample = recon_flat[indices]
    else:
        orig_sample = orig_flat
        recon_sample = recon_flat
    
    axes[0, 0].scatter(orig_sample, recon_sample, alpha=0.5, s=1)
    axes[0, 0].plot([orig_sample.min(), orig_sample.max()], 
                    [orig_sample.min(), orig_sample.max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Original')
    axes[0, 0].set_ylabel('Reconstructed')
    axes[0, 0].set_title('Overall Correlation')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Correlation coefficient
    corr = np.corrcoef(orig_flat, recon_flat)[0, 1]
    axes[0, 0].text(0.05, 0.95, f'r = {corr:.3f}', 
                    transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # Voxel-wise correlations
    voxel_corrs = []
    for voxel in range(original.shape[1]):
        if np.std(original[:, voxel]) > 1e-8 and np.std(reconstructed[:, voxel]) > 1e-8:
            corr = np.corrcoef(original[:, voxel], reconstructed[:, voxel])[0, 1]
            if not np.isnan(corr):
                voxel_corrs.append(corr)
    
    axes[0, 1].hist(voxel_corrs, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Correlation Coefficient')
    axes[0, 1].set_ylabel('Number of Voxels')
    axes[0, 1].set_title('Voxel-wise Correlations')
    axes[0, 1].axvline(np.mean(voxel_corrs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(voxel_corrs):.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample-wise correlations
    sample_corrs = []
    for sample in range(original.shape[0]):
        if np.std(original[sample, :]) > 1e-8 and np.std(reconstructed[sample, :]) > 1e-8:
            corr = np.corrcoef(original[sample, :], reconstructed[sample, :])[0, 1]
            if not np.isnan(corr):
                sample_corrs.append(corr)
    
    axes[1, 0].hist(sample_corrs, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('Correlation Coefficient')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Sample-wise Correlations')
    axes[1, 0].axvline(np.mean(sample_corrs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(sample_corrs):.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = orig_flat - recon_flat
    axes[1, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 1].set_xlabel('Reconstruction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(errors):.3f}')
    axes[1, 1].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation analysis saved to {save_path}")
    
    return fig


def plot_latent_space_analysis(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Latent Space Analysis"
) -> plt.Figure:
    """
    Plot latent space analysis including dimensionality reduction.
    
    Args:
        latents: Latent representations [n_samples, latent_dim]
        labels: Optional labels for coloring
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # PCA
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    if labels is not None:
        scatter = axes[0, 0].scatter(latents_pca[:, 0], latents_pca[:, 1], 
                                   c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=axes[0, 0])
    else:
        axes[0, 0].scatter(latents_pca[:, 0], latents_pca[:, 1], alpha=0.7)
    
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    axes[0, 0].set_title('PCA Projection')
    axes[0, 0].grid(True, alpha=0.3)
    
    # t-SNE (if not too many samples)
    if latents.shape[0] <= 1000:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, latents.shape[0]-1))
        latents_tsne = tsne.fit_transform(latents)
        
        if labels is not None:
            scatter = axes[0, 1].scatter(latents_tsne[:, 0], latents_tsne[:, 1], 
                                       c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=axes[0, 1])
        else:
            axes[0, 1].scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.7)
        
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        axes[0, 1].set_title('t-SNE Projection')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Too many samples\nfor t-SNE', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('t-SNE Projection (Skipped)')
    
    # Latent dimension statistics
    latent_means = np.mean(latents, axis=0)
    latent_stds = np.std(latents, axis=0)
    
    axes[1, 0].bar(range(len(latent_means)), latent_means, alpha=0.7)
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Mean Activation')
    axes[1, 0].set_title('Mean Latent Activations')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(range(len(latent_stds)), latent_stds, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Latent Dimension Variability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Latent space analysis saved to {save_path}")
    
    return fig


def plot_generation_samples(
    generated_samples: np.ndarray,
    num_samples: int = 5,
    save_path: Optional[str] = None,
    title: str = "Generated fMRI Samples"
) -> plt.Figure:
    """
    Plot generated fMRI samples.
    
    Args:
        generated_samples: Generated fMRI data [n_samples, n_voxels]
        num_samples: Number of samples to plot
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, generated_samples.shape[0])
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16)
    
    for i in range(num_samples):
        axes[i].plot(generated_samples[i], alpha=0.8, linewidth=1)
        axes[i].set_title(f'Generated Sample {i+1}')
        axes[i].set_xlabel('Voxel Index')
        axes[i].set_ylabel('Activation')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Generation samples saved to {save_path}")
    
    return fig


def create_visualization_report(
    original: np.ndarray,
    reconstructed: np.ndarray,
    generated: Optional[np.ndarray] = None,
    latents: Optional[np.ndarray] = None,
    history: Optional[Dict] = None,
    save_dir: str = "visualizations"
) -> Dict[str, str]:
    """
    Create a comprehensive visualization report.
    
    Args:
        original: Original fMRI data
        reconstructed: Reconstructed fMRI data
        generated: Generated fMRI data (optional)
        latents: Latent representations (optional)
        history: Training history (optional)
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary of saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Training curves
    if history:
        fig = plot_training_curves(history, save_path=save_dir / "training_curves.png")
        plt.close(fig)
        saved_files['training_curves'] = str(save_dir / "training_curves.png")
    
    # Reconstructions
    fig = plot_reconstructions(original, reconstructed, save_path=save_dir / "reconstructions.png")
    plt.close(fig)
    saved_files['reconstructions'] = str(save_dir / "reconstructions.png")
    
    # Correlation analysis
    fig = plot_correlation_analysis(original, reconstructed, save_path=save_dir / "correlation_analysis.png")
    plt.close(fig)
    saved_files['correlation_analysis'] = str(save_dir / "correlation_analysis.png")
    
    # Latent space analysis
    if latents is not None:
        fig = plot_latent_space_analysis(latents, save_path=save_dir / "latent_analysis.png")
        plt.close(fig)
        saved_files['latent_analysis'] = str(save_dir / "latent_analysis.png")
    
    # Generated samples
    if generated is not None:
        fig = plot_generation_samples(generated, save_path=save_dir / "generated_samples.png")
        plt.close(fig)
        saved_files['generated_samples'] = str(save_dir / "generated_samples.png")
    
    logger.info(f"Visualization report created in {save_dir}")
    return saved_files
