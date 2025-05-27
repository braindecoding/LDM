"""
📊 Evaluation Script for Brain Decoding LDM

Clean evaluation script with comprehensive metrics for brain decoding.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import os
from tqdm import tqdm

from data_loader import load_fmri_data
from brain_ldm import create_brain_ldm


class BrainLDMEvaluator:
    """Clean evaluator for Brain LDM."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device for evaluation
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint
        print(f"📁 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize model
        model_config = checkpoint['model_config']
        self.model = create_brain_ldm(**model_config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize data loader
        self.data_loader = load_fmri_data(device=device)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Trained for {checkpoint['epoch']} epochs")
        print(f"📈 Best val loss: {checkpoint['val_loss']:.4f}")
    
    def compute_pixel_metrics(self, true_images: np.ndarray, pred_images: np.ndarray) -> dict:
        """Compute pixel-level metrics."""
        # Flatten images for pixel-wise comparison
        true_flat = true_images.reshape(true_images.shape[0], -1)
        pred_flat = pred_images.reshape(pred_images.shape[0], -1)
        
        # Compute metrics
        mse = mean_squared_error(true_flat, pred_flat)
        mae = mean_absolute_error(true_flat, pred_flat)
        
        # Compute correlation for each sample
        correlations = []
        for i in range(true_flat.shape[0]):
            corr, _ = pearsonr(true_flat[i], pred_flat[i])
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'correlation': avg_correlation,
            'num_samples': len(correlations)
        }
    
    def compute_ssim(self, true_images: np.ndarray, pred_images: np.ndarray) -> float:
        """Compute SSIM (simplified version)."""
        def ssim_single(img1, img2):
            # Simplified SSIM calculation
            mu1, mu2 = img1.mean(), img2.mean()
            sigma1, sigma2 = img1.std(), img2.std()
            sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
            
            c1, c2 = 0.01**2, 0.03**2
            
            numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
            
            return numerator / denominator
        
        ssim_values = []
        for i in range(true_images.shape[0]):
            ssim_val = ssim_single(true_images[i], pred_images[i])
            if not np.isnan(ssim_val):
                ssim_values.append(ssim_val)
        
        return np.mean(ssim_values) if ssim_values else 0.0
    
    def evaluate_reconstruction(self, num_samples: int = None) -> dict:
        """Evaluate reconstruction quality."""
        print(f"\n📊 Evaluating reconstruction quality...")
        
        # Get test data
        test_loader = self.data_loader.create_dataloader(
            split='test', 
            batch_size=4, 
            shuffle=False
        )
        
        all_true_images = []
        all_pred_images = []
        all_fmri_signals = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating")):
                # Move to device
                fmri_signals = batch['fmri'].to(self.device)
                true_stimuli = batch['stimulus'].to(self.device)
                
                # Generate reconstructions
                generated_images = self.model.generate_from_fmri(
                    fmri_signals, 
                    num_inference_steps=50
                )
                
                # Convert to numpy
                true_imgs = true_stimuli.view(-1, 28, 28).cpu().numpy()
                pred_imgs = generated_images[:, 0].cpu().numpy()
                fmri_sigs = fmri_signals.cpu().numpy()
                
                all_true_images.append(true_imgs)
                all_pred_images.append(pred_imgs)
                all_fmri_signals.append(fmri_sigs)
                
                # Limit samples if specified
                if num_samples and len(all_true_images) * len(true_imgs) >= num_samples:
                    break
        
        # Concatenate all results
        true_images = np.concatenate(all_true_images, axis=0)
        pred_images = np.concatenate(all_pred_images, axis=0)
        fmri_signals = np.concatenate(all_fmri_signals, axis=0)
        
        if num_samples:
            true_images = true_images[:num_samples]
            pred_images = pred_images[:num_samples]
            fmri_signals = fmri_signals[:num_samples]
        
        print(f"📈 Evaluating {len(true_images)} samples...")
        
        # Compute metrics
        pixel_metrics = self.compute_pixel_metrics(true_images, pred_images)
        ssim_score = self.compute_ssim(true_images, pred_images)
        
        results = {
            **pixel_metrics,
            'ssim': ssim_score,
            'true_images': true_images,
            'pred_images': pred_images,
            'fmri_signals': fmri_signals
        }
        
        return results
    
    def create_comparison_plot(self, results: dict, num_samples: int = 8, save_path: str = None):
        """Create comparison visualization."""
        true_images = results['true_images']
        pred_images = results['pred_images']
        
        num_samples = min(num_samples, len(true_images))
        
        fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
        
        for i in range(num_samples):
            # True image
            axes[0, i].imshow(true_images[i], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'True {i}')
            axes[0, i].axis('off')
            
            # Predicted image
            axes[1, i].imshow(pred_images[i], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Reconstructed {i}')
            axes[1, i].axis('off')
        
        plt.suptitle('Brain Decoding Results: True vs Reconstructed Stimuli', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved comparison plot to: {save_path}")
        
        plt.show()
    
    def print_metrics(self, results: dict):
        """Print evaluation metrics."""
        print(f"\n📊 Evaluation Results:")
        print(f"=" * 40)
        print(f"🎯 Pixel-level Metrics:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  MAE: {results['mae']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
        print(f"  Correlation: {results['correlation']:.4f}")
        print(f"\n🖼️ Image Quality Metrics:")
        print(f"  SSIM: {results['ssim']:.4f}")
        print(f"\n📈 Dataset Info:")
        print(f"  Evaluated samples: {results['num_samples']}")
        print(f"  Image range: [{results['pred_images'].min():.3f}, {results['pred_images'].max():.3f}]")
    
    def run_full_evaluation(self, save_dir: str = "evaluation_results"):
        """Run complete evaluation pipeline."""
        print(f"🧠 Brain LDM Evaluation")
        print("=" * 50)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Run evaluation
        results = self.evaluate_reconstruction()
        
        # Print metrics
        self.print_metrics(results)
        
        # Create visualizations
        comparison_path = os.path.join(save_dir, "reconstruction_comparison.png")
        self.create_comparison_plot(results, num_samples=8, save_path=comparison_path)
        
        # Save detailed results
        results_path = os.path.join(save_dir, "evaluation_metrics.txt")
        with open(results_path, 'w') as f:
            f.write("Brain LDM Evaluation Results\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"MSE: {results['mse']:.6f}\n")
            f.write(f"MAE: {results['mae']:.6f}\n")
            f.write(f"RMSE: {results['rmse']:.6f}\n")
            f.write(f"Correlation: {results['correlation']:.4f}\n")
            f.write(f"SSIM: {results['ssim']:.4f}\n")
            f.write(f"Samples: {results['num_samples']}\n")
        
        print(f"\n💾 Results saved to: {save_dir}")
        
        return results


def main():
    """Main evaluation function."""
    print("📊 Brain LDM Evaluation")
    print("=" * 40)
    
    # Configuration
    checkpoint_path = "checkpoints/best_model.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 Using device: {device}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"💡 Please train the model first using: python train_brain_ldm.py")
        return
    
    # Create evaluator
    evaluator = BrainLDMEvaluator(checkpoint_path, device=device)
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    print(f"\n✅ Evaluation completed!")
    print(f"🎯 Overall reconstruction quality: {results['correlation']:.3f} correlation")


if __name__ == "__main__":
    main()
