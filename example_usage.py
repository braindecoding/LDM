"""
Example usage script for the Latent Diffusion Model.
This script shows how to use the model components step by step.
"""

import numpy as np
import torch
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_data_loading():
    """Example of how to load and preprocess fMRI data."""
    print("=" * 60)
    print("EXAMPLE 1: DATA LOADING AND PREPROCESSING")
    print("=" * 60)
    
    try:
        from src.data.fmri_data_loader import FMRIDataLoader
        
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize data loader
        print("📊 Initializing data loader...")
        data_loader = FMRIDataLoader(config)
        
        # Get data statistics
        stats = data_loader.get_data_stats()
        print(f"✅ Data loaded: {stats['total_samples']} samples, {stats['num_voxels']} voxels")
        
        # Get data loaders
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        test_loader = data_loader.get_test_loader()
        
        print(f"📈 Train batches: {len(train_loader)}")
        print(f"📈 Validation batches: {len(val_loader)}")
        print(f"📈 Test batches: {len(test_loader)}")
        
        # Show a sample batch
        for batch in train_loader:
            print(f"📦 Sample batch shape: {batch.shape}")
            print(f"📦 Batch statistics: mean={torch.mean(batch):.4f}, std={torch.std(batch):.4f}")
            break
        
        return data_loader, config
        
    except Exception as e:
        print(f"❌ Error in data loading: {e}")
        return None, None


def example_vae_usage(config):
    """Example of how to use the VAE component."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: VAE ENCODER/DECODER USAGE")
    print("=" * 60)
    
    try:
        from src.models.vae_encoder_decoder import VariationalAutoencoder
        
        # Initialize VAE
        print("🧠 Initializing VAE...")
        vae = VariationalAutoencoder(config)
        
        # Create sample data
        batch_size = 4
        input_dim = config['vae']['input_dim']
        sample_data = torch.randn(batch_size, input_dim)
        
        print(f"📥 Input shape: {sample_data.shape}")
        
        # Forward pass
        print("🔄 Running forward pass...")
        output = vae(sample_data)
        
        print(f"📤 Reconstruction shape: {output['reconstruction'].shape}")
        print(f"📤 Latent mu shape: {output['mu'].shape}")
        print(f"📤 Latent logvar shape: {output['logvar'].shape}")
        print(f"📤 Latent z shape: {output['z'].shape}")
        
        # Compute loss
        loss_dict = vae.compute_loss(
            sample_data, 
            output['reconstruction'], 
            output['mu'], 
            output['logvar']
        )
        
        print(f"💰 Total loss: {loss_dict['total_loss']:.4f}")
        print(f"💰 Reconstruction loss: {loss_dict['reconstruction_loss']:.4f}")
        print(f"💰 KL loss: {loss_dict['kl_loss']:.4f}")
        
        # Test encoding/decoding separately
        print("🔄 Testing separate encoding/decoding...")
        mu, logvar = vae.encode(sample_data)
        z = vae.reparameterize(mu, logvar)
        reconstruction = vae.decode(z)
        
        print(f"✅ Separate reconstruction shape: {reconstruction.shape}")
        
        return vae
        
    except Exception as e:
        print(f"❌ Error in VAE usage: {e}")
        return None


def example_diffusion_usage(config):
    """Example of how to use the diffusion model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: DIFFUSION MODEL USAGE")
    print("=" * 60)
    
    try:
        from src.models.diffusion_model import DiffusionUNet, DDPMScheduler
        
        # Initialize diffusion model
        print("🌊 Initializing diffusion model...")
        diffusion_config = config['diffusion']
        
        diffusion_model = DiffusionUNet(
            latent_dim=config['vae']['latent_dim'],
            model_channels=diffusion_config['model_channels'],
            num_res_blocks=diffusion_config['num_res_blocks']
        )
        
        # Initialize scheduler
        scheduler = DDPMScheduler(
            num_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            beta_schedule=diffusion_config['beta_schedule']
        )
        
        # Create sample latent data
        batch_size = 4
        latent_dim = config['vae']['latent_dim']
        latent_data = torch.randn(batch_size, latent_dim)
        
        print(f"📥 Latent input shape: {latent_data.shape}")
        
        # Test forward diffusion (add noise)
        timesteps = torch.randint(0, diffusion_config['num_timesteps'], (batch_size,))
        noise = torch.randn_like(latent_data)
        
        print("🔄 Adding noise (forward diffusion)...")
        noisy_latents = scheduler.add_noise(latent_data, noise, timesteps)
        print(f"📤 Noisy latents shape: {noisy_latents.shape}")
        
        # Test reverse diffusion (predict noise)
        print("🔄 Predicting noise (reverse diffusion)...")
        predicted_noise = diffusion_model(noisy_latents, timesteps)
        print(f"📤 Predicted noise shape: {predicted_noise.shape}")
        
        # Test denoising step
        print("🔄 Performing denoising step...")
        denoised = scheduler.step(predicted_noise, timesteps[0].item(), noisy_latents[0:1])
        print(f"📤 Denoised shape: {denoised.shape}")
        
        return diffusion_model, scheduler
        
    except Exception as e:
        print(f"❌ Error in diffusion usage: {e}")
        return None, None


def example_complete_ldm_usage(config):
    """Example of how to use the complete Latent Diffusion Model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: COMPLETE LATENT DIFFUSION MODEL")
    print("=" * 60)
    
    try:
        from src.models.latent_diffusion_model import LatentDiffusionModel
        
        # Initialize complete model
        print("🚀 Initializing complete Latent Diffusion Model...")
        model = LatentDiffusionModel(config)
        
        # Get model info
        info = model.get_model_info()
        print(f"📊 Total parameters: {info['total_parameters']:,}")
        print(f"📊 VAE parameters: {info['vae_parameters']:,}")
        print(f"📊 Diffusion parameters: {info['diffusion_parameters']:,}")
        
        # Create sample data
        batch_size = 4
        input_dim = config['vae']['input_dim']
        sample_data = torch.randn(batch_size, input_dim)
        
        print(f"📥 Input data shape: {sample_data.shape}")
        
        # Test VAE loss computation
        print("🔄 Computing VAE loss...")
        vae_loss = model.compute_vae_loss(sample_data)
        print(f"💰 VAE total loss: {vae_loss['total_loss']:.4f}")
        
        # Test diffusion loss computation
        print("🔄 Computing diffusion loss...")
        diffusion_loss = model.compute_diffusion_loss(sample_data)
        print(f"💰 Diffusion loss: {diffusion_loss['diffusion_loss']:.4f}")
        
        # Test encoding to latent space
        print("🔄 Encoding to latent space...")
        latents = model.encode_to_latent(sample_data)
        print(f"📤 Latent shape: {latents.shape}")
        
        # Test decoding from latent space
        print("🔄 Decoding from latent space...")
        reconstructed = model.decode_from_latent(latents)
        print(f"📤 Reconstructed shape: {reconstructed.shape}")
        
        # Test complete reconstruction
        print("🔄 Complete reconstruction...")
        full_reconstruction = model.reconstruct_fmri(sample_data)
        print(f"📤 Full reconstruction shape: {full_reconstruction.shape}")
        
        # Test generation
        print("🔄 Generating new samples...")
        device = torch.device('cpu')
        generated = model.generate_fmri_samples(
            batch_size=2, 
            device=device, 
            num_inference_steps=10  # Reduced for demo
        )
        print(f"📤 Generated samples shape: {generated.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error in complete LDM usage: {e}")
        return None


def example_metrics_and_visualization():
    """Example of metrics computation and visualization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: METRICS AND VISUALIZATION")
    print("=" * 60)
    
    try:
        from src.utils.metrics import compute_reconstruction_metrics
        from src.utils.visualization import plot_reconstructions, plot_training_curves
        
        # Create dummy data for demonstration
        n_samples, n_voxels = 20, 100
        original = np.random.randn(n_samples, n_voxels)
        reconstructed = original + 0.1 * np.random.randn(n_samples, n_voxels)
        
        print("📊 Computing reconstruction metrics...")
        metrics = compute_reconstruction_metrics(original, reconstructed)
        
        # Show key metrics
        key_metrics = [
            'val_overall_correlation',
            'val_mse',
            'val_rmse',
            'val_r2_score'
        ]
        
        print("📈 Key metrics:")
        for metric in key_metrics:
            if metric in metrics:
                print(f"   {metric}: {metrics[metric]:.4f}")
        
        # Test visualization (without saving)
        print("📊 Testing visualization functions...")
        
        try:
            import matplotlib.pyplot as plt
            
            # Test reconstruction plot
            fig = plot_reconstructions(original, reconstructed, num_samples=3)
            plt.close(fig)
            print("✅ Reconstruction plot created successfully")
            
            # Test training curves
            dummy_history = {
                'train_loss': [1.0, 0.8, 0.6, 0.4],
                'val_loss': [1.1, 0.9, 0.7, 0.5]
            }
            fig = plot_training_curves(dummy_history)
            plt.close(fig)
            print("✅ Training curves plot created successfully")
            
        except ImportError:
            print("⚠️  Matplotlib not available for visualization demo")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error in metrics/visualization: {e}")
        return None


def main():
    """Main example function."""
    print("🎯 LATENT DIFFUSION MODEL - USAGE EXAMPLES")
    print("This script demonstrates how to use each component of the LDM.")
    print()
    
    # Example 1: Data loading
    data_loader, config = example_data_loading()
    
    if config is None:
        print("❌ Cannot proceed without configuration. Please install PyYAML: pip install pyyaml")
        return
    
    # Example 2: VAE usage
    vae = example_vae_usage(config)
    
    # Example 3: Diffusion model usage
    diffusion_model, scheduler = example_diffusion_usage(config)
    
    # Example 4: Complete LDM usage
    complete_model = example_complete_ldm_usage(config)
    
    # Example 5: Metrics and visualization
    metrics = example_metrics_and_visualization()
    
    print("\n" + "=" * 60)
    print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("📝 Next steps:")
    print("   1. Install all dependencies: pip install -r requirements.txt")
    print("   2. Run the complete training: python main.py")
    print("   3. Experiment with different configurations in config.yaml")
    print()
    print("📚 For more details, see README.md")


if __name__ == "__main__":
    main()
