"""
Script untuk membandingkan rekonstruksi VAE-only vs True Latent Diffusion Model.
Menunjukkan perbedaan antara implementasi sebelumnya (VAE-only) dan implementasi LDM yang benar.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.fmri_data_loader import FMRIDataLoader
from models.latent_diffusion_model import LatentDiffusionModel
from utils.visualization import plot_individual_stimulus_reconstruction_images
from utils.metrics import compute_reconstruction_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VAEvsLDMComparator:
    """Class untuk membandingkan VAE-only vs True LDM reconstruction."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize comparator."""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path("results/vae_vs_ldm_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model_and_data(self):
        """Load trained model and test data."""
        print("🤖 Loading model and data...")
        
        # Load model
        model = LatentDiffusionModel(self.config)
        checkpoint_path = "checkpoints/best_model.pt"
        
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {checkpoint_path}")
        else:
            print(f"⚠️  No trained model found, using untrained model")
        
        model.to(self.device)
        model.eval()
        
        # Load test data
        data_loader_manager = FMRIDataLoader(self.config)
        test_loader = data_loader_manager.get_test_loader()
        
        # Get a batch of test data
        test_batch = next(iter(test_loader))
        test_data = test_batch.to(self.device)
        
        # Limit to first 5 samples for comparison
        test_data = test_data[:5]
        
        print(f"✅ Test data loaded: {test_data.shape}")
        
        return model, test_data
    
    def compare_reconstructions(self, model, test_data):
        """Compare VAE-only vs LDM reconstructions."""
        print("\n🔄 Generating reconstructions...")
        
        with torch.no_grad():
            # VAE-only reconstruction (old method)
            print("   📊 VAE-only reconstruction...")
            vae_reconstruction = model.reconstruct_fmri(test_data, use_diffusion=False)
            
            # True LDM reconstruction (new method)
            print("   🌊 True LDM reconstruction...")
            ldm_reconstruction = model.reconstruct_fmri(test_data, use_diffusion=True)
        
        # Convert to numpy
        original = test_data.cpu().numpy()
        vae_recon = vae_reconstruction.cpu().numpy()
        ldm_recon = ldm_reconstruction.cpu().numpy()
        
        return original, vae_recon, ldm_recon
    
    def compute_comparison_metrics(self, original, vae_recon, ldm_recon):
        """Compute metrics for both methods."""
        print("\n📊 Computing metrics...")
        
        # VAE metrics
        print("   📊 VAE-only metrics...")
        vae_metrics = compute_reconstruction_metrics(original, vae_recon)
        
        # LDM metrics
        print("   🌊 LDM metrics...")
        ldm_metrics = compute_reconstruction_metrics(original, ldm_recon)
        
        return vae_metrics, ldm_metrics
    
    def create_comparison_visualizations(self, original, vae_recon, ldm_recon):
        """Create comparison visualizations."""
        print("\n🎨 Creating visualizations...")
        
        # Individual comparisons
        print("   📊 VAE-only visualization...")
        vae_fig = plot_individual_stimulus_reconstruction_images(
            original, vae_recon,
            num_samples=5,
            save_path=str(self.results_dir / "vae_only_reconstruction.png"),
            title="VAE-Only Reconstruction (Previous Implementation)"
        )
        plt.close(vae_fig)
        
        print("   🌊 LDM visualization...")
        ldm_fig = plot_individual_stimulus_reconstruction_images(
            original, ldm_recon,
            num_samples=5,
            save_path=str(self.results_dir / "ldm_reconstruction.png"),
            title="True Latent Diffusion Model Reconstruction"
        )
        plt.close(ldm_fig)
        
        # Side-by-side comparison
        print("   🔄 Creating side-by-side comparison...")
        self._create_side_by_side_comparison(original, vae_recon, ldm_recon)
    
    def _create_side_by_side_comparison(self, original, vae_recon, ldm_recon):
        """Create side-by-side comparison visualization."""
        num_samples = min(3, len(original))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('VAE-only vs True LDM Reconstruction Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Column headers
        fig.text(0.2, 0.95, 'Original', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        fig.text(0.5, 0.95, 'VAE-only', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        fig.text(0.8, 0.95, 'True LDM', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        
        for i in range(num_samples):
            # Convert to 2D for visualization
            n_voxels = len(original[i])
            height = int(np.sqrt(n_voxels))
            width = int(np.ceil(n_voxels / height))
            
            # Pad and reshape
            pad_size = height * width - n_voxels
            
            orig_padded = np.pad(original[i], (0, pad_size), mode='constant', constant_values=0)
            vae_padded = np.pad(vae_recon[i], (0, pad_size), mode='constant', constant_values=0)
            ldm_padded = np.pad(ldm_recon[i], (0, pad_size), mode='constant', constant_values=0)
            
            orig_2d = orig_padded.reshape(height, width)
            vae_2d = vae_padded.reshape(height, width)
            ldm_2d = ldm_padded.reshape(height, width)
            
            # Normalize
            orig_norm = (orig_2d - orig_2d.min()) / (orig_2d.max() - orig_2d.min() + 1e-8)
            vae_norm = (vae_2d - vae_2d.min()) / (vae_2d.max() - vae_2d.min() + 1e-8)
            ldm_norm = (ldm_2d - ldm_2d.min()) / (ldm_2d.max() - ldm_2d.min() + 1e-8)
            
            # Plot
            axes[i, 0].imshow(orig_norm, cmap='gray', aspect='equal', interpolation='nearest')
            axes[i, 0].set_ylabel(f'Sample {i+1}', fontsize=12, fontweight='bold')
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            
            axes[i, 1].imshow(vae_norm, cmap='gray', aspect='equal', interpolation='nearest')
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            
            axes[i, 2].imshow(ldm_norm, cmap='gray', aspect='equal', interpolation='nearest')
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            
            # Add correlations
            from scipy import stats
            vae_corr, _ = stats.pearsonr(original[i], vae_recon[i])
            ldm_corr, _ = stats.pearsonr(original[i], ldm_recon[i])
            
            axes[i, 1].text(0.02, 0.98, f'r={vae_corr:.3f}', 
                           transform=axes[i, 1].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                           fontsize=10, verticalalignment='top', fontweight='bold')
            
            axes[i, 2].text(0.02, 0.98, f'r={ldm_corr:.3f}', 
                           transform=axes[i, 2].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                           fontsize=10, verticalalignment='top', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        save_path = self.results_dir / "vae_vs_ldm_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Comparison saved: {save_path}")
    
    def create_metrics_comparison_plot(self, vae_metrics, ldm_metrics):
        """Create metrics comparison plot."""
        print("   📊 Creating metrics comparison...")
        
        # Select key metrics
        key_metrics = [
            'val_overall_correlation',
            'val_rmse', 
            'val_psnr',
            'val_ssim'
        ]
        
        metric_labels = [
            'Overall Correlation',
            'RMSE',
            'PSNR (dB)',
            'SSIM'
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('VAE-only vs True LDM: Metrics Comparison', 
                    fontsize=16, fontweight='bold')
        
        for i, (metric, label) in enumerate(zip(key_metrics, metric_labels)):
            ax = axes[i//2, i%2]
            
            vae_value = vae_metrics.get(metric, 0)
            ldm_value = ldm_metrics.get(metric, 0)
            
            methods = ['VAE-only', 'True LDM']
            values = [vae_value, ldm_value]
            colors = ['lightcoral', 'lightblue']
            
            bars = ax.bar(methods, values, color=colors, alpha=0.7)
            ax.set_title(label, fontweight='bold')
            ax.set_ylabel(label)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Highlight better performance
            if ldm_value > vae_value and metric != 'val_rmse':  # Higher is better except RMSE
                bars[1].set_color('lightgreen')
            elif ldm_value < vae_value and metric == 'val_rmse':  # Lower is better for RMSE
                bars[1].set_color('lightgreen')
        
        plt.tight_layout()
        
        save_path = self.results_dir / "metrics_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Metrics comparison saved: {save_path}")
    
    def print_summary(self, vae_metrics, ldm_metrics):
        """Print comparison summary."""
        print("\n" + "=" * 80)
        print("📋 VAE-ONLY vs TRUE LDM COMPARISON SUMMARY")
        print("=" * 80)
        
        print("\n🔍 Key Findings:")
        
        # Compare key metrics
        vae_corr = vae_metrics.get('val_overall_correlation', 0)
        ldm_corr = ldm_metrics.get('val_overall_correlation', 0)
        vae_rmse = vae_metrics.get('val_rmse', 0)
        ldm_rmse = ldm_metrics.get('val_rmse', 0)
        
        print(f"\n📊 Overall Correlation:")
        print(f"   • VAE-only: {vae_corr:.4f}")
        print(f"   • True LDM: {ldm_corr:.4f}")
        if ldm_corr > vae_corr:
            print(f"   ✅ LDM is better by {ldm_corr - vae_corr:.4f}")
        else:
            print(f"   ⚠️  VAE-only is better by {vae_corr - ldm_corr:.4f}")
        
        print(f"\n📉 RMSE:")
        print(f"   • VAE-only: {vae_rmse:.4f}")
        print(f"   • True LDM: {ldm_rmse:.4f}")
        if ldm_rmse < vae_rmse:
            print(f"   ✅ LDM is better by {vae_rmse - ldm_rmse:.4f}")
        else:
            print(f"   ⚠️  VAE-only is better by {ldm_rmse - vae_rmse:.4f}")
        
        print(f"\n💡 Conclusion:")
        if ldm_corr > vae_corr and ldm_rmse < vae_rmse:
            print("   ✅ True LDM performs better than VAE-only!")
        elif ldm_corr > vae_corr or ldm_rmse < vae_rmse:
            print("   🔄 Mixed results - LDM shows improvement in some metrics")
        else:
            print("   ⚠️  VAE-only performs better - may need LDM tuning")
    
    def run_comparison(self):
        """Run complete comparison."""
        print("🔍 COMPARING VAE-ONLY vs TRUE LATENT DIFFUSION MODEL")
        print("=" * 70)
        
        # Load model and data
        model, test_data = self.load_model_and_data()
        
        # Generate reconstructions
        original, vae_recon, ldm_recon = self.compare_reconstructions(model, test_data)
        
        # Compute metrics
        vae_metrics, ldm_metrics = self.compute_comparison_metrics(original, vae_recon, ldm_recon)
        
        # Create visualizations
        self.create_comparison_visualizations(original, vae_recon, ldm_recon)
        self.create_metrics_comparison_plot(vae_metrics, ldm_metrics)
        
        # Print summary
        self.print_summary(vae_metrics, ldm_metrics)
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        print("✅ Comparison completed!")


def main():
    """Main function."""
    comparator = VAEvsLDMComparator()
    comparator.run_comparison()


if __name__ == "__main__":
    main()
