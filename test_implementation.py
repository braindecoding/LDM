"""
Test script to verify the Latent Diffusion Model implementation.
"""

import torch
import numpy as np
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loader():
    """Test the fMRI data loader."""
    logger.info("Testing fMRI Data Loader...")
    
    try:
        from src.data.fmri_data_loader import FMRIDataLoader
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test data loader
        data_loader = FMRIDataLoader(config)
        
        # Get data statistics
        stats = data_loader.get_data_stats()
        logger.info(f"Data loaded successfully: {stats['total_samples']} samples, {stats['num_voxels']} voxels")
        
        # Test data loaders
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        test_loader = data_loader.get_test_loader()
        
        # Test a batch
        for batch in train_loader:
            logger.info(f"Batch shape: {batch.shape}")
            break
        
        logger.info("✓ Data loader test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loader test failed: {e}")
        return False


def test_vae():
    """Test the VAE implementation."""
    logger.info("Testing VAE...")
    
    try:
        from src.models.vae_encoder_decoder import VariationalAutoencoder
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create VAE
        vae = VariationalAutoencoder(config)
        
        # Test forward pass
        batch_size = 4
        input_dim = config['vae']['input_dim']
        x = torch.randn(batch_size, input_dim)
        
        output = vae(x)
        
        # Check outputs
        assert 'reconstruction' in output
        assert 'mu' in output
        assert 'logvar' in output
        assert 'z' in output
        
        assert output['reconstruction'].shape == x.shape
        assert output['mu'].shape == (batch_size, config['vae']['latent_dim'])
        assert output['logvar'].shape == (batch_size, config['vae']['latent_dim'])
        assert output['z'].shape == (batch_size, config['vae']['latent_dim'])
        
        # Test loss computation
        loss_dict = vae.compute_loss(x, output['reconstruction'], output['mu'], output['logvar'])
        assert 'total_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict
        assert 'kl_loss' in loss_dict
        
        logger.info("✓ VAE test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ VAE test failed: {e}")
        return False


def test_diffusion_model():
    """Test the diffusion model implementation."""
    logger.info("Testing Diffusion Model...")
    
    try:
        from src.models.diffusion_model import DiffusionUNet, DDPMScheduler
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create diffusion model
        diffusion_config = config['diffusion']
        model = DiffusionUNet(
            latent_dim=config['vae']['latent_dim'],
            model_channels=diffusion_config['model_channels'],
            num_res_blocks=diffusion_config['num_res_blocks']
        )
        
        # Test forward pass
        batch_size = 4
        latent_dim = config['vae']['latent_dim']
        x = torch.randn(batch_size, latent_dim)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        output = model(x, timesteps)
        assert output.shape == x.shape
        
        # Test scheduler
        scheduler = DDPMScheduler(
            num_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end']
        )
        
        # Test noise addition
        noise = torch.randn_like(x)
        noisy_x = scheduler.add_noise(x, noise, timesteps)
        assert noisy_x.shape == x.shape
        
        # Test denoising step
        denoised = scheduler.step(output, timesteps[0].item(), noisy_x[0:1])
        assert denoised.shape == (1, latent_dim)
        
        logger.info("✓ Diffusion model test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Diffusion model test failed: {e}")
        return False


def test_latent_diffusion_model():
    """Test the complete Latent Diffusion Model."""
    logger.info("Testing Latent Diffusion Model...")
    
    try:
        from src.models.latent_diffusion_model import LatentDiffusionModel
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = LatentDiffusionModel(config)
        
        # Test forward passes
        batch_size = 4
        input_dim = config['vae']['input_dim']
        x = torch.randn(batch_size, input_dim)
        
        # Test VAE loss
        vae_loss = model.compute_vae_loss(x)
        assert 'total_loss' in vae_loss
        
        # Test diffusion loss
        diffusion_loss = model.compute_diffusion_loss(x)
        assert 'diffusion_loss' in diffusion_loss
        
        # Test encoding/decoding
        latents = model.encode_to_latent(x)
        assert latents.shape == (batch_size, config['vae']['latent_dim'])
        
        reconstructed = model.decode_from_latent(latents)
        assert reconstructed.shape == x.shape
        
        # Test generation
        device = torch.device('cpu')
        generated = model.generate_fmri_samples(
            batch_size=2, 
            device=device, 
            num_inference_steps=10
        )
        assert generated.shape == (2, input_dim)
        
        # Test model info
        info = model.get_model_info()
        assert 'total_parameters' in info
        
        logger.info("✓ Latent Diffusion Model test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Latent Diffusion Model test failed: {e}")
        return False


def test_metrics():
    """Test the metrics implementation."""
    logger.info("Testing Metrics...")
    
    try:
        from src.utils.metrics import compute_reconstruction_metrics
        
        # Create dummy data
        n_samples, n_voxels = 20, 100
        original = np.random.randn(n_samples, n_voxels)
        reconstructed = original + 0.1 * np.random.randn(n_samples, n_voxels)
        
        # Compute metrics
        metrics = compute_reconstruction_metrics(original, reconstructed)
        
        # Check that metrics are computed
        expected_metrics = [
            'val_overall_correlation',
            'val_mse',
            'val_rmse',
            'val_r2_score',
            'val_mean_voxel_correlation'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        logger.info("✓ Metrics test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}")
        return False


def test_visualization():
    """Test the visualization implementation."""
    logger.info("Testing Visualization...")
    
    try:
        from src.utils.visualization import plot_reconstructions, plot_training_curves
        import matplotlib.pyplot as plt
        
        # Test reconstruction plot
        n_samples, n_voxels = 5, 100
        original = np.random.randn(n_samples, n_voxels)
        reconstructed = original + 0.1 * np.random.randn(n_samples, n_voxels)
        
        fig = plot_reconstructions(original, reconstructed, num_samples=3)
        plt.close(fig)
        
        # Test training curves
        history = {
            'train_loss': [1.0, 0.8, 0.6, 0.4],
            'val_loss': [1.1, 0.9, 0.7, 0.5],
            'train_recon_loss': [0.8, 0.6, 0.4, 0.3],
            'val_recon_loss': [0.9, 0.7, 0.5, 0.4]
        }
        
        fig = plot_training_curves(history)
        plt.close(fig)
        
        logger.info("✓ Visualization test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Visualization test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting implementation tests...")
    
    tests = [
        test_data_loader,
        test_vae,
        test_diffusion_model,
        test_latent_diffusion_model,
        test_metrics,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Implementation is ready.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    main()
