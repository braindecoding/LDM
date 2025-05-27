"""
Clean training script for Latent Diffusion Model on fMRI data.
"""

import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.trainer import LatentDiffusionTrainer
from utils.visualization import create_visualization_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def train_model(config: dict) -> None:
    """Train the Latent Diffusion Model."""
    logger.info("Starting model training...")
    
    # Initialize trainer
    trainer = LatentDiffusionTrainer(config)
    
    # Train the model
    trainer.train()
    
    logger.info("Training completed successfully!")


def evaluate_model(config: dict) -> None:
    """Evaluate the trained model."""
    logger.info("Starting model evaluation...")
    
    # Initialize trainer
    trainer = LatentDiffusionTrainer(config)
    
    # Load best model
    checkpoint_path = Path(config['training']['checkpoint_dir']) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(f"No trained model found at {checkpoint_path}")
        return
    
    # Evaluate
    metrics = trainer.evaluate(str(checkpoint_path))
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualization_report(
        trainer.test_loader,
        trainer.model,
        trainer.device,
        metrics,
        save_dir=Path(config['training']['log_dir']) / f"visualizations_{trainer.timestamp}"
    )
    
    logger.info("Evaluation completed successfully!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or evaluate Latent Diffusion Model")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=["train", "evaluate", "both"], default="both",
                       help="Mode: train, evaluate, or both")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute based on mode
    if args.mode in ["train", "both"]:
        train_model(config)
    
    if args.mode in ["evaluate", "both"]:
        evaluate_model(config)


if __name__ == "__main__":
    main()
