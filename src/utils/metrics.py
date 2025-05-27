"""
Evaluation metrics for fMRI reconstruction quality assessment.
"""

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_correlation_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute correlation-based metrics between original and reconstructed data.
    
    Args:
        original: Original fMRI data [n_samples, n_voxels]
        reconstructed: Reconstructed fMRI data [n_samples, n_voxels]
        
    Returns:
        Dictionary of correlation metrics
    """
    # Flatten arrays for overall correlation
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Overall Pearson correlation
    overall_corr, overall_p = stats.pearsonr(orig_flat, recon_flat)
    
    # Voxel-wise correlations
    voxel_correlations = []
    for voxel_idx in range(original.shape[1]):
        if np.std(original[:, voxel_idx]) > 1e-8 and np.std(reconstructed[:, voxel_idx]) > 1e-8:
            corr, _ = stats.pearsonr(original[:, voxel_idx], reconstructed[:, voxel_idx])
            if not np.isnan(corr):
                voxel_correlations.append(corr)
    
    voxel_correlations = np.array(voxel_correlations)
    
    # Sample-wise correlations
    sample_correlations = []
    for sample_idx in range(original.shape[0]):
        if np.std(original[sample_idx, :]) > 1e-8 and np.std(reconstructed[sample_idx, :]) > 1e-8:
            corr, _ = stats.pearsonr(original[sample_idx, :], reconstructed[sample_idx, :])
            if not np.isnan(corr):
                sample_correlations.append(corr)
    
    sample_correlations = np.array(sample_correlations)
    
    return {
        'overall_correlation': float(overall_corr),
        'overall_correlation_p_value': float(overall_p),
        'mean_voxel_correlation': float(np.mean(voxel_correlations)) if len(voxel_correlations) > 0 else 0.0,
        'std_voxel_correlation': float(np.std(voxel_correlations)) if len(voxel_correlations) > 0 else 0.0,
        'mean_sample_correlation': float(np.mean(sample_correlations)) if len(sample_correlations) > 0 else 0.0,
        'std_sample_correlation': float(np.std(sample_correlations)) if len(sample_correlations) > 0 else 0.0,
        'num_valid_voxels': len(voxel_correlations),
        'num_valid_samples': len(sample_correlations)
    }


def compute_error_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute error-based metrics between original and reconstructed data.
    
    Args:
        original: Original fMRI data [n_samples, n_voxels]
        reconstructed: Reconstructed fMRI data [n_samples, n_voxels]
        
    Returns:
        Dictionary of error metrics
    """
    # Flatten arrays
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Overall metrics
    mse = mean_squared_error(orig_flat, recon_flat)
    mae = mean_absolute_error(orig_flat, recon_flat)
    rmse = np.sqrt(mse)
    
    # R-squared
    r2 = r2_score(orig_flat, recon_flat)
    
    # Normalized metrics
    orig_range = np.max(orig_flat) - np.min(orig_flat)
    normalized_rmse = rmse / orig_range if orig_range > 0 else float('inf')
    normalized_mae = mae / orig_range if orig_range > 0 else float('inf')
    
    # Signal-to-noise ratio
    signal_power = np.mean(orig_flat ** 2)
    noise_power = np.mean((orig_flat - recon_flat) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_val = np.max(orig_flat)
    psnr = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'normalized_rmse': float(normalized_rmse),
        'normalized_mae': float(normalized_mae),
        'snr_db': float(snr),
        'psnr_db': float(psnr)
    }


def compute_distribution_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute distribution-based metrics between original and reconstructed data.
    
    Args:
        original: Original fMRI data [n_samples, n_voxels]
        reconstructed: Reconstructed fMRI data [n_samples, n_voxels]
        
    Returns:
        Dictionary of distribution metrics
    """
    # Flatten arrays
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Statistical moments
    orig_mean, orig_std = np.mean(orig_flat), np.std(orig_flat)
    recon_mean, recon_std = np.mean(recon_flat), np.std(recon_flat)
    
    # Moment differences
    mean_diff = abs(orig_mean - recon_mean)
    std_diff = abs(orig_std - recon_std)
    
    # Skewness and kurtosis
    from scipy.stats import skew, kurtosis
    orig_skew = skew(orig_flat)
    recon_skew = skew(recon_flat)
    orig_kurt = kurtosis(orig_flat)
    recon_kurt = kurtosis(recon_flat)
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = stats.ks_2samp(orig_flat, recon_flat)
    
    # Jensen-Shannon divergence (approximate using histograms)
    def js_divergence(p, q):
        """Compute Jensen-Shannon divergence between two distributions."""
        # Ensure no zeros
        p = p + 1e-10
        q = q + 1e-10
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        # Compute JS divergence
        m = 0.5 * (p + q)
        return 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
    
    # Create histograms
    bins = np.linspace(
        min(np.min(orig_flat), np.min(recon_flat)),
        max(np.max(orig_flat), np.max(recon_flat)),
        50
    )
    orig_hist, _ = np.histogram(orig_flat, bins=bins, density=True)
    recon_hist, _ = np.histogram(recon_flat, bins=bins, density=True)
    
    js_div = js_divergence(orig_hist, recon_hist)
    
    return {
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        'skewness_original': float(orig_skew),
        'skewness_reconstructed': float(recon_skew),
        'skewness_difference': float(abs(orig_skew - recon_skew)),
        'kurtosis_original': float(orig_kurt),
        'kurtosis_reconstructed': float(recon_kurt),
        'kurtosis_difference': float(abs(orig_kurt - recon_kurt)),
        'ks_statistic': float(ks_statistic),
        'ks_p_value': float(ks_p_value),
        'js_divergence': float(js_div)
    }


def compute_spatial_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute spatial pattern metrics for fMRI data.
    
    Args:
        original: Original fMRI data [n_samples, n_voxels]
        reconstructed: Reconstructed fMRI data [n_samples, n_voxels]
        
    Returns:
        Dictionary of spatial metrics
    """
    # Compute spatial correlation patterns
    orig_spatial_corr = np.corrcoef(original.T)  # Voxel-voxel correlations
    recon_spatial_corr = np.corrcoef(reconstructed.T)
    
    # Remove diagonal and NaN values
    mask = ~np.eye(orig_spatial_corr.shape[0], dtype=bool)
    orig_corr_values = orig_spatial_corr[mask]
    recon_corr_values = recon_spatial_corr[mask]
    
    # Remove NaN values
    valid_mask = ~(np.isnan(orig_corr_values) | np.isnan(recon_corr_values))
    orig_corr_values = orig_corr_values[valid_mask]
    recon_corr_values = recon_corr_values[valid_mask]
    
    if len(orig_corr_values) > 0:
        # Correlation between spatial correlation patterns
        spatial_pattern_corr, spatial_pattern_p = stats.pearsonr(orig_corr_values, recon_corr_values)
        
        # MSE of spatial correlation patterns
        spatial_pattern_mse = mean_squared_error(orig_corr_values, recon_corr_values)
    else:
        spatial_pattern_corr = 0.0
        spatial_pattern_p = 1.0
        spatial_pattern_mse = float('inf')
    
    return {
        'spatial_pattern_correlation': float(spatial_pattern_corr),
        'spatial_pattern_p_value': float(spatial_pattern_p),
        'spatial_pattern_mse': float(spatial_pattern_mse),
        'num_valid_spatial_correlations': len(orig_corr_values)
    }


def compute_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction metrics.
    
    Args:
        original: Original fMRI data [n_samples, n_voxels]
        reconstructed: Reconstructed fMRI data [n_samples, n_voxels]
        
    Returns:
        Dictionary of all metrics
    """
    logger.info("Computing reconstruction metrics...")
    
    # Ensure same shape
    assert original.shape == reconstructed.shape, f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
    
    # Compute all metric categories
    correlation_metrics = compute_correlation_metrics(original, reconstructed)
    error_metrics = compute_error_metrics(original, reconstructed)
    distribution_metrics = compute_distribution_metrics(original, reconstructed)
    spatial_metrics = compute_spatial_metrics(original, reconstructed)
    
    # Combine all metrics
    all_metrics = {
        **correlation_metrics,
        **error_metrics,
        **distribution_metrics,
        **spatial_metrics
    }
    
    # Add prefix for validation metrics
    val_metrics = {f"val_{k}": v for k, v in all_metrics.items()}
    
    logger.info(f"Computed {len(all_metrics)} reconstruction metrics")
    
    return val_metrics


def compute_latent_space_metrics(
    original_latents: np.ndarray, 
    reconstructed_latents: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics specifically for latent space representations.
    
    Args:
        original_latents: Original latent representations
        reconstructed_latents: Reconstructed latent representations
        
    Returns:
        Dictionary of latent space metrics
    """
    # Basic error metrics
    mse = mean_squared_error(original_latents.flatten(), reconstructed_latents.flatten())
    mae = mean_absolute_error(original_latents.flatten(), reconstructed_latents.flatten())
    
    # Correlation in latent space
    corr, p_val = stats.pearsonr(original_latents.flatten(), reconstructed_latents.flatten())
    
    # Latent dimension-wise metrics
    dim_correlations = []
    for dim in range(original_latents.shape[1]):
        if np.std(original_latents[:, dim]) > 1e-8 and np.std(reconstructed_latents[:, dim]) > 1e-8:
            dim_corr, _ = stats.pearsonr(original_latents[:, dim], reconstructed_latents[:, dim])
            if not np.isnan(dim_corr):
                dim_correlations.append(dim_corr)
    
    return {
        'latent_mse': float(mse),
        'latent_mae': float(mae),
        'latent_correlation': float(corr),
        'latent_correlation_p_value': float(p_val),
        'mean_dimension_correlation': float(np.mean(dim_correlations)) if dim_correlations else 0.0,
        'std_dimension_correlation': float(np.std(dim_correlations)) if dim_correlations else 0.0
    }
