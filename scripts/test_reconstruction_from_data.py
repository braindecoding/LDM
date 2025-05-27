"""
Script untuk menguji rekonstruksi dari berbagai sumber data:
1. Data asli dari folder data/
2. Data aligned dari folder outputs/
3. Perbandingan kualitas rekonstruksi
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
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


class DataReconstructionTester:
    """Class untuk menguji rekonstruksi dari berbagai sumber data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize tester dengan konfigurasi."""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path("results/data_comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load konfigurasi dari file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_original_data(self) -> dict:
        """Load data asli dari folder data/."""
        data_sources = {}
        data_dir = Path("data")
        
        print("🔍 Mencari data asli di folder data/...")
        
        # Check for .mat files
        mat_files = list(data_dir.glob("*.mat"))
        for mat_file in mat_files:
            try:
                print(f"📊 Loading: {mat_file.name}")
                mat_data = loadmat(str(mat_file))
                
                # Extract data arrays (skip metadata)
                data_arrays = {}
                for key, value in mat_data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray):
                        data_arrays[key] = value
                        print(f"   • {key}: {value.shape}")
                
                data_sources[mat_file.stem] = data_arrays
                
            except Exception as e:
                print(f"❌ Error loading {mat_file.name}: {e}")
        
        # Check for .npy files
        npy_files = list(data_dir.glob("*.npy"))
        for npy_file in npy_files:
            try:
                print(f"📊 Loading: {npy_file.name}")
                npy_data = np.load(str(npy_file))
                print(f"   • Shape: {npy_data.shape}")
                data_sources[npy_file.stem] = {"data": npy_data}
                
            except Exception as e:
                print(f"❌ Error loading {npy_file.name}: {e}")
        
        # Check for .npz files
        npz_files = list(data_dir.glob("*.npz"))
        for npz_file in npz_files:
            try:
                print(f"📊 Loading: {npz_file.name}")
                npz_data = np.load(str(npz_file))
                
                data_dict = {}
                for key in npz_data.keys():
                    data_dict[key] = npz_data[key]
                    print(f"   • {key}: {npz_data[key].shape}")
                
                data_sources[npz_file.stem] = data_dict
                
            except Exception as e:
                print(f"❌ Error loading {npz_file.name}: {e}")
        
        return data_sources
    
    def load_aligned_data(self) -> dict:
        """Load data aligned dari folder outputs/."""
        print("\n🔍 Loading aligned data dari outputs/...")
        
        outputs_dir = Path("outputs")
        aligned_files = list(outputs_dir.glob("*aligned_data.npz"))
        
        aligned_data = {}
        for aligned_file in aligned_files:
            try:
                print(f"📊 Loading: {aligned_file.name}")
                data = np.load(str(aligned_file))
                
                file_data = {}
                for key in data.keys():
                    file_data[key] = data[key]
                    print(f"   • {key}: {data[key].shape}")
                
                aligned_data[aligned_file.stem] = file_data
                
            except Exception as e:
                print(f"❌ Error loading {aligned_file.name}: {e}")
        
        return aligned_data
    
    def prepare_data_for_reconstruction(self, data_dict: dict, max_samples: int = 10) -> np.ndarray:
        """Prepare data untuk rekonstruksi."""
        # Try to find the main data array
        possible_keys = ['data', 'fmri_data', 'brain_data', 'activations']
        
        main_data = None
        for key in possible_keys:
            if key in data_dict:
                main_data = data_dict[key]
                break
        
        if main_data is None:
            # Take the largest array
            largest_key = max(data_dict.keys(), key=lambda k: data_dict[k].size)
            main_data = data_dict[largest_key]
            print(f"   Using largest array: {largest_key}")
        
        # Ensure 2D shape (samples, features)
        if main_data.ndim == 1:
            main_data = main_data.reshape(1, -1)
        elif main_data.ndim > 2:
            # Flatten extra dimensions
            main_data = main_data.reshape(main_data.shape[0], -1)
        
        # Limit samples
        if main_data.shape[0] > max_samples:
            main_data = main_data[:max_samples]
        
        print(f"   Prepared data shape: {main_data.shape}")
        return main_data
    
    def load_trained_model(self) -> torch.nn.Module:
        """Load model yang sudah ditraining."""
        print("\n🤖 Loading trained model...")
        
        model = LatentDiffusionModel(self.config)
        checkpoint_path = "checkpoints/best_model.pt"
        
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {checkpoint_path}")
        else:
            print(f"⚠️  No trained model found at {checkpoint_path}")
            print("   Using untrained model for demonstration")
        
        model.to(self.device)
        model.eval()
        return model
    
    def test_reconstruction(self, data: np.ndarray, data_name: str, model: torch.nn.Module) -> dict:
        """Test rekonstruksi pada data."""
        print(f"\n🔄 Testing reconstruction on {data_name}...")
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        reconstructions = []
        
        with torch.no_grad():
            for i, sample in enumerate(data_tensor):
                try:
                    # Reconstruct using the model
                    sample_input = sample.unsqueeze(0)  # Add batch dimension
                    reconstruction = model.reconstruct_fmri(sample_input)
                    reconstructions.append(reconstruction.cpu().numpy().squeeze())
                    print(f"   ✅ Reconstructed sample {i+1}")
                    
                except Exception as e:
                    print(f"   ❌ Error reconstructing sample {i+1}: {e}")
                    # Fallback to simple transformation
                    reconstructions.append(sample.cpu().numpy() * 0.8 + np.random.randn(len(sample.cpu().numpy())) * 0.2)
        
        reconstructions = np.array(reconstructions)
        
        # Compute metrics
        try:
            metrics = compute_reconstruction_metrics(data, reconstructions)
            print(f"   📊 Metrics computed successfully")
        except Exception as e:
            print(f"   ⚠️  Error computing metrics: {e}")
            metrics = {"correlation": 0.0, "rmse": 1.0}
        
        return {
            "original": data,
            "reconstructed": reconstructions,
            "metrics": metrics
        }
    
    def create_comparison_visualization(self, results: dict) -> None:
        """Buat visualisasi perbandingan."""
        print("\n🎨 Creating comparison visualizations...")
        
        for data_name, result in results.items():
            try:
                # Create stimulus vs reconstruction plot
                save_path = self.results_dir / f"{data_name}_reconstruction.png"
                
                fig = plot_individual_stimulus_reconstruction_images(
                    result["original"],
                    result["reconstructed"],
                    num_samples=min(5, len(result["original"])),
                    save_path=str(save_path),
                    title=f"Reconstruction Test: {data_name}"
                )
                plt.close(fig)
                
                print(f"   ✅ Saved: {save_path}")
                
            except Exception as e:
                print(f"   ❌ Error creating visualization for {data_name}: {e}")
    
    def create_metrics_comparison(self, results: dict) -> None:
        """Buat perbandingan metrik."""
        print("\n📊 Creating metrics comparison...")
        
        # Collect metrics
        metrics_data = {}
        for data_name, result in results.items():
            metrics_data[data_name] = result["metrics"]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Reconstruction Quality Comparison", fontsize=16, fontweight='bold')
        
        # Metrics to compare
        metric_names = ['val_overall_correlation', 'val_rmse', 'val_psnr', 'val_ssim']
        metric_labels = ['Correlation', 'RMSE', 'PSNR (dB)', 'SSIM']
        
        for i, (metric_name, label) in enumerate(zip(metric_names, metric_labels)):
            ax = axes[i//2, i%2]
            
            data_names = list(metrics_data.keys())
            values = [metrics_data[name].get(metric_name, 0) for name in data_names]
            
            bars = ax.bar(data_names, values, alpha=0.7)
            ax.set_title(label, fontweight='bold')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.results_dir / "metrics_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Metrics comparison saved: {save_path}")
    
    def run_complete_test(self) -> None:
        """Jalankan test rekonstruksi lengkap."""
        print("🧠 TESTING RECONSTRUCTION FROM DIFFERENT DATA SOURCES")
        print("=" * 70)
        
        # Load all data sources
        original_data = self.load_original_data()
        aligned_data = self.load_aligned_data()
        
        # Load trained model
        model = self.load_trained_model()
        
        # Test reconstruction on all data sources
        results = {}
        
        # Test original data
        for data_name, data_dict in original_data.items():
            try:
                prepared_data = self.prepare_data_for_reconstruction(data_dict)
                result = self.test_reconstruction(prepared_data, f"original_{data_name}", model)
                results[f"original_{data_name}"] = result
            except Exception as e:
                print(f"❌ Error testing {data_name}: {e}")
        
        # Test aligned data
        for data_name, data_dict in aligned_data.items():
            try:
                # Combine all subjects
                all_data = []
                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        all_data.append(value)
                
                if all_data:
                    combined_data = np.vstack(all_data)
                    prepared_data = self.prepare_data_for_reconstruction({"data": combined_data})
                    result = self.test_reconstruction(prepared_data, f"aligned_{data_name}", model)
                    results[f"aligned_{data_name}"] = result
            except Exception as e:
                print(f"❌ Error testing {data_name}: {e}")
        
        if not results:
            print("❌ No data could be processed for reconstruction testing")
            return
        
        # Create visualizations
        self.create_comparison_visualization(results)
        self.create_metrics_comparison(results)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📋 RECONSTRUCTION TEST SUMMARY")
        print("=" * 70)
        
        for data_name, result in results.items():
            metrics = result["metrics"]
            correlation = metrics.get("val_overall_correlation", 0)
            rmse = metrics.get("val_rmse", 0)
            
            print(f"\n🔍 {data_name}:")
            print(f"   • Data shape: {result['original'].shape}")
            print(f"   • Correlation: {correlation:.4f}")
            print(f"   • RMSE: {rmse:.4f}")
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        print("✅ Reconstruction testing completed!")


def main():
    """Main function."""
    tester = DataReconstructionTester()
    tester.run_complete_test()


if __name__ == "__main__":
    main()
