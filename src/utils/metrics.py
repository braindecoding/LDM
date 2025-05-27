"""
Evaluation metrics for fMRI reconstruction quality assessment.
"""

import numpy as np
import torch
from scipy import stats
import scipy.linalg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import logging

# Optional advanced metrics imports
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

try:
    import torch.nn.functional as F
    from torchvision.models import inception_v3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

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
    Compute comprehensive reconstruction metrics including advanced metrics.

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

    # Compute advanced metrics
    logger.info("Computing advanced metrics (PSNR, SSIM, FID, LPIPS, CLIP)...")
    advanced_metrics = compute_advanced_metrics(original, reconstructed)

    # Combine all metrics
    all_metrics = {
        **correlation_metrics,
        **error_metrics,
        **distribution_metrics,
        **spatial_metrics,
        **advanced_metrics
    }

    # Add prefix for validation metrics
    val_metrics = {f"val_{k}": v for k, v in all_metrics.items()}

    logger.info(f"Computed {len(all_metrics)} reconstruction metrics (including advanced metrics)")

    return val_metrics


def compute_advanced_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute advanced metrics including PSNR, SSIM, FID, LPIPS, and CLIP score.

    Args:
        original: Original fMRI data [n_samples, n_voxels]
        reconstructed: Reconstructed fMRI data [n_samples, n_voxels]

    Returns:
        Dictionary of advanced metrics
    """
    metrics = {}

    # PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_val = np.max(original)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    metrics['psnr'] = float(psnr)

    # SSIM (Structural Similarity Index)
    if SSIM_AVAILABLE:
        try:
            # For fMRI data, we compute SSIM for each sample and average
            ssim_scores = []
            for i in range(original.shape[0]):
                # Reshape to 2D for SSIM (treat as 1D signal reshaped to square-ish)
                orig_sample = original[i]
                recon_sample = reconstructed[i]

                # Find closest square dimensions
                n_voxels = len(orig_sample)
                side_len = int(np.sqrt(n_voxels))
                if side_len * side_len < n_voxels:
                    side_len += 1

                # Pad to square
                pad_len = side_len * side_len - n_voxels
                orig_padded = np.pad(orig_sample, (0, pad_len), mode='constant')
                recon_padded = np.pad(recon_sample, (0, pad_len), mode='constant')

                # Reshape to 2D
                orig_2d = orig_padded.reshape(side_len, side_len)
                recon_2d = recon_padded.reshape(side_len, side_len)

                # Compute SSIM
                ssim_score = ssim(orig_2d, recon_2d, data_range=orig_2d.max() - orig_2d.min())
                if not np.isnan(ssim_score):
                    ssim_scores.append(ssim_score)

            metrics['ssim'] = float(np.mean(ssim_scores)) if ssim_scores else 0.0
        except Exception as e:
            logger.warning(f"SSIM computation failed: {e}")
            metrics['ssim'] = 0.0
    else:
        metrics['ssim'] = 0.0

    # FID (Fréchet Inception Distance) - adapted for fMRI
    if FID_AVAILABLE:
        try:
            # For fMRI, we compute FID using feature statistics
            orig_features = original
            recon_features = reconstructed

            # Compute means and covariances
            mu1 = np.mean(orig_features, axis=0)
            mu2 = np.mean(recon_features, axis=0)

            sigma1 = np.cov(orig_features, rowvar=False)
            sigma2 = np.cov(recon_features, rowvar=False)

            # Compute FID
            diff = mu1 - mu2
            covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
            metrics['fid'] = float(fid)
        except Exception as e:
            logger.warning(f"FID computation failed: {e}")
            metrics['fid'] = float('inf')
    else:
        metrics['fid'] = float('inf')

    # LPIPS (Learned Perceptual Image Patch Similarity)
    if LPIPS_AVAILABLE:
        try:
            # Initialize LPIPS model
            lpips_model = lpips.LPIPS(net='alex')

            # Convert fMRI data to image-like format for LPIPS
            lpips_scores = []
            for i in range(min(original.shape[0], 10)):  # Limit to 10 samples for efficiency
                # Convert to tensor and reshape
                orig_tensor = torch.FloatTensor(original[i]).unsqueeze(0).unsqueeze(0)
                recon_tensor = torch.FloatTensor(reconstructed[i]).unsqueeze(0).unsqueeze(0)

                # Resize to minimum required size for LPIPS (64x64)
                orig_resized = F.interpolate(orig_tensor.unsqueeze(0), size=(64, 64), mode='bilinear')
                recon_resized = F.interpolate(recon_tensor.unsqueeze(0), size=(64, 64), mode='bilinear')

                # Repeat to 3 channels (RGB)
                orig_rgb = orig_resized.repeat(1, 3, 1, 1)
                recon_rgb = recon_resized.repeat(1, 3, 1, 1)

                # Compute LPIPS
                with torch.no_grad():
                    lpips_score = lpips_model(orig_rgb, recon_rgb)
                    lpips_scores.append(lpips_score.item())

            metrics['lpips'] = float(np.mean(lpips_scores)) if lpips_scores else 1.0
        except Exception as e:
            logger.warning(f"LPIPS computation failed: {e}")
            metrics['lpips'] = 1.0
    else:
        metrics['lpips'] = 1.0

    # CLIP Score - adapted for fMRI (using correlation as proxy)
    if CLIP_AVAILABLE:
        try:
            # For fMRI, we use a correlation-based proxy for CLIP score
            # since CLIP is designed for image-text similarity
            correlations = []
            for i in range(original.shape[0]):
                corr, _ = stats.pearsonr(original[i], reconstructed[i])
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            # Convert correlation to CLIP-like score (0-1 range)
            clip_score = np.mean(correlations) if correlations else 0.0
            metrics['clip_score'] = float(clip_score)
        except Exception as e:
            logger.warning(f"CLIP score computation failed: {e}")
            metrics['clip_score'] = 0.0
    else:
        metrics['clip_score'] = 0.0

    return metrics


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
