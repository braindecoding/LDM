"""
Script untuk menguji rekonstruksi stimulus dari data fMRI (brain decoding).
Menunjukkan implementasi yang benar: fMRI → Stimulus reconstruction.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch
import logging
from scipy import stats

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.stimulus_data_loader import StimulusDataLoader
from models.stimulus_ldm import StimulusLatentDiffusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StimulusReconstructionTester:
    """Class untuk menguji rekonstruksi stimulus dari fMRI."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize tester."""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path("results/stimulus_reconstruction")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data_and_model(self):
        """Load stimulus data and model."""
        print("📊 Loading stimulus data...")
        
        # Load data
        data_loader = StimulusDataLoader(self.config)
        test_loader = data_loader.get_test_loader()
        
        # Get data statistics
        stats = data_loader.get_data_stats()
        print("📊 Data Statistics:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")
        
        # Initialize model
        print("\n🤖 Initializing model...")
        model = StimulusLatentDiffusionModel(self.config)
        
        # Print model info
        model_info = model.get_model_info()
        print("🤖 Model Information:")
        for key, value in model_info.items():
            print(f"   • {key}: {value:,}" if isinstance(value, int) else f"   • {key}: {value}")
        
        model.to(self.device)
        model.eval()
        
        return model, test_loader, data_loader
    
    def test_reconstruction(self, model, test_loader):
        """Test stimulus reconstruction."""
        print("\n🔄 Testing stimulus reconstruction...")
        
        # Get test batch
        test_batch = next(iter(test_loader))
        fmri_data = test_batch['fmri'].to(self.device)
        true_stimuli = test_batch['stimulus'].to(self.device)
        labels = test_batch.get('label', None)
        
        # Limit to first 5 samples for visualization
        fmri_data = fmri_data[:5]
        true_stimuli = true_stimuli[:5]
        if labels is not None:
            labels = labels[:5]
        
        print(f"   📊 Testing on {len(fmri_data)} samples")
        print(f"   🧠 fMRI shape: {fmri_data.shape}")
        print(f"   🖼️  Stimulus shape: {true_stimuli.shape}")
        
        with torch.no_grad():
            # VAE-only reconstruction
            print("   📊 VAE-only reconstruction...")
            vae_reconstruction = model.reconstruct_stimulus(fmri_data, use_diffusion=False)
            
            # True LDM reconstruction
            print("   🌊 LDM reconstruction...")
            ldm_reconstruction = model.reconstruct_stimulus(fmri_data, use_diffusion=True)
        
        # Convert to numpy
        fmri_np = fmri_data.cpu().numpy()
        true_stimuli_np = true_stimuli.cpu().numpy()
        vae_recon_np = vae_reconstruction.cpu().numpy()
        ldm_recon_np = ldm_reconstruction.cpu().numpy()
        labels_np = labels.cpu().numpy() if labels is not None else None
        
        return {
            'fmri': fmri_np,
            'true_stimuli': true_stimuli_np,
            'vae_reconstruction': vae_recon_np,
            'ldm_reconstruction': ldm_recon_np,
            'labels': labels_np
        }
    
    def compute_reconstruction_metrics(self, results):
        """Compute reconstruction metrics."""
        print("\n📊 Computing reconstruction metrics...")
        
        true_stimuli = results['true_stimuli']
        vae_recon = results['vae_reconstruction']
        ldm_recon = results['ldm_reconstruction']
        
        # Compute correlations
        vae_correlations = []
        ldm_correlations = []
        
        for i in range(len(true_stimuli)):
            vae_corr, _ = stats.pearsonr(true_stimuli[i], vae_recon[i])
            ldm_corr, _ = stats.pearsonr(true_stimuli[i], ldm_recon[i])
            
            vae_correlations.append(vae_corr)
            ldm_correlations.append(ldm_corr)
        
        # Compute MSE
        vae_mse = np.mean((true_stimuli - vae_recon) ** 2)
        ldm_mse = np.mean((true_stimuli - ldm_recon) ** 2)
        
        metrics = {
            'vae_correlation_mean': np.mean(vae_correlations),
            'vae_correlation_std': np.std(vae_correlations),
            'ldm_correlation_mean': np.mean(ldm_correlations),
            'ldm_correlation_std': np.std(ldm_correlations),
            'vae_mse': vae_mse,
            'ldm_mse': ldm_mse,
            'vae_correlations': vae_correlations,
            'ldm_correlations': ldm_correlations
        }
        
        return metrics
    
    def visualize_results(self, results, metrics):
        """Create comprehensive visualizations."""
        print("\n🎨 Creating visualizations...")
        
        true_stimuli = results['true_stimuli']
        vae_recon = results['vae_reconstruction']
        ldm_recon = results['ldm_reconstruction']
        labels = results['labels']
        
        num_samples = len(true_stimuli)
        
        # Create main comparison figure
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('fMRI → Stimulus Reconstruction Results\n(Brain Decoding)', 
                    fontsize=16, fontweight='bold')
        
        # Column headers
        fig.text(0.2, 0.95, 'True Stimulus', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        fig.text(0.5, 0.95, 'VAE Reconstruction', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        fig.text(0.8, 0.95, 'LDM Reconstruction', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        
        for i in range(num_samples):
            # Reshape to 28x28 for visualization
            true_img = true_stimuli[i].reshape(28, 28)
            vae_img = vae_recon[i].reshape(28, 28)
            ldm_img = ldm_recon[i].reshape(28, 28)
            
            # Plot images
            axes[i, 0].imshow(true_img, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_ylabel(f'Sample {i+1}' + 
                                 (f'\nDigit: {labels[i]}' if labels is not None else ''), 
                                 fontsize=12, fontweight='bold')
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            
            axes[i, 1].imshow(vae_img, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            
            axes[i, 2].imshow(ldm_img, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            
            # Add correlation scores
            vae_corr = metrics['vae_correlations'][i]
            ldm_corr = metrics['ldm_correlations'][i]
            
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
        
        save_path = self.results_dir / "stimulus_reconstruction_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Main comparison saved: {save_path}")
        
        # Create metrics comparison
        self._create_metrics_plot(metrics)
    
    def _create_metrics_plot(self, metrics):
        """Create metrics comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Correlation comparison
        methods = ['VAE-only', 'LDM']
        correlations = [metrics['vae_correlation_mean'], metrics['ldm_correlation_mean']]
        errors = [metrics['vae_correlation_std'], metrics['ldm_correlation_std']]
        
        bars1 = ax1.bar(methods, correlations, yerr=errors, capsize=5, 
                       color=['lightcoral', 'lightblue'], alpha=0.7)
        ax1.set_title('Reconstruction Correlation', fontweight='bold')
        ax1.set_ylabel('Pearson Correlation')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, corr, err in zip(bars1, correlations, errors):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                    f'{corr:.3f}±{err:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MSE comparison
        mse_values = [metrics['vae_mse'], metrics['ldm_mse']]
        bars2 = ax2.bar(methods, mse_values, color=['lightcoral', 'lightblue'], alpha=0.7)
        ax2.set_title('Reconstruction MSE', fontweight='bold')
        ax2.set_ylabel('Mean Squared Error')
        
        # Add value labels
        for bar, mse in zip(bars2, mse_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mse:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.results_dir / "reconstruction_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Metrics plot saved: {save_path}")
    
    def print_summary(self, metrics):
        """Print reconstruction summary."""
        print("\n" + "=" * 80)
        print("📋 STIMULUS RECONSTRUCTION SUMMARY")
        print("=" * 80)
        
        print(f"\n🎯 Task: Reconstruct visual stimuli from fMRI brain activity")
        print(f"📊 Input: fMRI data (3092 voxels)")
        print(f"🖼️  Output: Stimulus images (28×28 pixels)")
        
        print(f"\n📊 Reconstruction Quality:")
        print(f"   • VAE-only correlation: {metrics['vae_correlation_mean']:.4f} ± {metrics['vae_correlation_std']:.4f}")
        print(f"   • LDM correlation: {metrics['ldm_correlation_mean']:.4f} ± {metrics['ldm_correlation_std']:.4f}")
        
        print(f"\n📉 Reconstruction Error:")
        print(f"   • VAE-only MSE: {metrics['vae_mse']:.4f}")
        print(f"   • LDM MSE: {metrics['ldm_mse']:.4f}")
        
        # Determine which is better
        if metrics['ldm_correlation_mean'] > metrics['vae_correlation_mean']:
            improvement = metrics['ldm_correlation_mean'] - metrics['vae_correlation_mean']
            print(f"\n✅ LDM shows improvement: +{improvement:.4f} correlation")
        else:
            decline = metrics['vae_correlation_mean'] - metrics['ldm_correlation_mean']
            print(f"\n⚠️  LDM shows decline: -{decline:.4f} correlation")
        
        print(f"\n💡 Note: This is brain decoding - reconstructing what the brain 'saw' from neural activity!")
    
    def run_test(self):
        """Run complete stimulus reconstruction test."""
        print("🧠 TESTING STIMULUS RECONSTRUCTION FROM fMRI")
        print("Goal: Decode visual stimuli from brain activity (Brain Decoding)")
        print("=" * 80)
        
        # Load data and model
        model, test_loader, data_loader = self.load_data_and_model()
        
        # Test reconstruction
        results = self.test_reconstruction(model, test_loader)
        
        # Compute metrics
        metrics = self.compute_reconstruction_metrics(results)
        
        # Create visualizations
        self.visualize_results(results, metrics)
        
        # Print summary
        self.print_summary(metrics)
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        print("✅ Stimulus reconstruction test completed!")


def main():
    """Main function."""
    tester = StimulusReconstructionTester()
    tester.run_test()


if __name__ == "__main__":
    main()
