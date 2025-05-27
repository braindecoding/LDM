"""
Main script for training and evaluating Latent Diffusion Model for fMRI reconstruction.
"""

import torch
import numpy as np
import yaml
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime

from src.data.fmri_data_loader import FMRIDataLoader
from src.models.latent_diffusion_model import LatentDiffusionModel
from src.training.trainer import LDMTrainer
from src.utils.metrics import compute_reconstruction_metrics
from src.utils.visualization import create_visualization_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fmri_ldm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: dict) -> torch.device:
    """Setup computation device."""
    device_config = config['hardware']['device']
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
    else:
        device = torch.device(device_config)
        logger.info(f"Using specified device: {device}")
    
    return device


def train_model(config: dict, device: torch.device):
    """
    Train the Latent Diffusion Model.
    
    Args:
        config: Configuration dictionary
        device: Training device
    """
    logger.info("Starting model training...")
    
    # Load data
    logger.info("Loading fMRI data...")
    data_loader = FMRIDataLoader(config)
    
    # Print data statistics
    stats = data_loader.get_data_stats()
    logger.info("Data Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Create data loaders
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    # Initialize model
    logger.info("Initializing Latent Diffusion Model...")
    model = LatentDiffusionModel(config)
    
    # Print model info
    model_info = model.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    # Initialize trainer
    trainer = LDMTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train model
    history = trainer.train()
    
    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = Path(config['training']['log_dir']) / f"training_history_{timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    return model, trainer, history, test_loader


def evaluate_model(
    model: LatentDiffusionModel, 
    test_loader, 
    device: torch.device,
    config: dict
):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Evaluation device
        config: Configuration dictionary
    """
    logger.info("Evaluating model on test set...")
    
    model.eval()
    
    all_originals = []
    all_reconstructions = []
    all_latents = []
    all_generated = []
    
    with torch.no_grad():
        # Reconstruction evaluation
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            
            # Reconstruct
            reconstructions = model.reconstruct_fmri(batch_data)
            
            # Encode to latent
            latents = model.encode_to_latent(batch_data)
            
            # Store results
            all_originals.append(batch_data.cpu().numpy())
            all_reconstructions.append(reconstructions.cpu().numpy())
            all_latents.append(latents.cpu().numpy())
        
        # Generate new samples
        logger.info("Generating new fMRI samples...")
        num_generate = min(50, len(all_originals) * config['training']['batch_size'])
        generated_samples = model.generate_fmri_samples(
            batch_size=num_generate,
            device=device,
            num_inference_steps=config['sampling']['num_inference_steps']
        )
        all_generated = generated_samples.cpu().numpy()
    
    # Concatenate all results
    all_originals = np.concatenate(all_originals, axis=0)
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    
    logger.info(f"Evaluation completed on {len(all_originals)} test samples")
    
    return all_originals, all_reconstructions, all_latents, all_generated


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="fMRI Latent Diffusion Model")
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], 
                       default='both', help='Execution mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")
    
    # Setup device
    device = setup_device(config)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    if args.mode in ['train', 'both']:
        # Training
        model, trainer, history, test_loader = train_model(config, device)
        
        if args.mode == 'both':
            # Evaluation after training
            originals, reconstructions, latents, generated = evaluate_model(
                model, test_loader, device, config
            )
            
            # Compute metrics
            logger.info("Computing evaluation metrics...")
            metrics = compute_reconstruction_metrics(originals, reconstructions)
            
            # Save metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = Path(config['training']['log_dir']) / f"evaluation_metrics_{timestamp}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Evaluation metrics saved to {metrics_path}")
            
            # Create visualizations
            logger.info("Creating visualization report...")
            viz_dir = Path(config['training']['log_dir']) / f"visualizations_{timestamp}"
            saved_files = create_visualization_report(
                original=originals,
                reconstructed=reconstructions,
                generated=generated,
                latents=latents,
                history=history,
                save_dir=str(viz_dir)
            )
            
            logger.info("Visualization report created:")
            for name, path in saved_files.items():
                logger.info(f"  {name}: {path}")
            
            # Print key metrics
            logger.info("\nKey Evaluation Metrics:")
            key_metrics = [
                'val_overall_correlation',
                'val_mean_voxel_correlation',
                'val_rmse',
                'val_r2_score',
                'val_snr_db'
            ]
            for metric in key_metrics:
                if metric in metrics:
                    logger.info(f"  {metric}: {metrics[metric]:.4f}")
    
    elif args.mode == 'evaluate':
        # Evaluation only
        if args.checkpoint is None:
            raise ValueError("Checkpoint path required for evaluation mode")
        
        # Load data
        data_loader = FMRIDataLoader(config)
        test_loader = data_loader.get_test_loader()
        
        # Initialize and load model
        model = LatentDiffusionModel(config)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {args.checkpoint}")
        
        # Evaluate
        originals, reconstructions, latents, generated = evaluate_model(
            model, test_loader, device, config
        )
        
        # Compute and save metrics
        metrics = compute_reconstruction_metrics(originals, reconstructions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = f"evaluation_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        viz_dir = f"visualizations_{timestamp}"
        saved_files = create_visualization_report(
            original=originals,
            reconstructed=reconstructions,
            generated=generated,
            latents=latents,
            save_dir=viz_dir
        )
        
        logger.info(f"Evaluation completed. Results saved to {viz_dir}")


if __name__ == "__main__":
    main()
