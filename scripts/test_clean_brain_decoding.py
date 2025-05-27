"""
Clean script for testing brain-to-stimulus reconstruction (brain decoding).
Demonstrates proper clean code principles and clear naming conventions.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch
import logging
from scipy import stats
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.stimulus_data_loader import BrainToStimulusDataLoader
from models.stimulus_ldm import StimulusLatentDiffusionModel

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CleanBrainDecodingTester:
    """
    Clean implementation of brain decoding tester.
    Tests reconstruction of visual stimuli from fMRI brain activity.
    """
    
    def __init__(self, configuration_file_path: str = "config.yaml"):
        """
        Initialize brain decoding tester with clean configuration.
        
        Args:
            configuration_file_path: Path to YAML configuration file
        """
        self.config = self._load_configuration(configuration_file_path)
        self.computation_device = self._setup_computation_device()
        self.results_directory = self._create_results_directory()
        
        logger.info("Brain decoding tester initialized successfully")
    
    def _load_configuration(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file with error handling.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as config_file:
                configuration = yaml.safe_load(config_file)
            logger.info(f"Configuration loaded from {config_path}")
            return configuration
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as error:
            logger.error(f"Error parsing YAML configuration: {error}")
            raise
    
    def _setup_computation_device(self) -> torch.device:
        """
        Setup computation device (GPU/CPU) with clean logic.
        
        Returns:
            PyTorch device for computation
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for computation")
        
        return device
    
    def _create_results_directory(self) -> Path:
        """
        Create results directory with clean path handling.
        
        Returns:
            Path to results directory
        """
        results_path = Path("results/clean_brain_decoding")
        results_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {results_path}")
        return results_path
    
    def load_brain_stimulus_data(self) -> BrainToStimulusDataLoader:
        """
        Load brain and stimulus data with clean error handling.
        
        Returns:
            Configured data loader
        """
        logger.info("Loading brain-to-stimulus data...")
        
        try:
            data_loader = BrainToStimulusDataLoader(self.config)
            
            # Log data statistics with clean formatting
            data_statistics = data_loader.get_data_stats()
            logger.info("Data loading completed successfully")
            logger.info("Dataset statistics:")
            for statistic_name, statistic_value in data_statistics.items():
                logger.info(f"  • {statistic_name}: {statistic_value}")
            
            return data_loader
            
        except Exception as error:
            logger.error(f"Failed to load data: {error}")
            raise
    
    def initialize_brain_decoding_model(self) -> StimulusLatentDiffusionModel:
        """
        Initialize brain decoding model with clean setup.
        
        Returns:
            Configured brain decoding model
        """
        logger.info("Initializing brain decoding model...")
        
        try:
            brain_decoding_model = StimulusLatentDiffusionModel(self.config)
            brain_decoding_model.to(self.computation_device)
            brain_decoding_model.eval()
            
            # Log model information with clean formatting
            model_information = brain_decoding_model.get_model_info()
            logger.info("Model initialization completed successfully")
            logger.info("Model architecture:")
            for info_key, info_value in model_information.items():
                if isinstance(info_value, int):
                    logger.info(f"  • {info_key}: {info_value:,}")
                else:
                    logger.info(f"  • {info_key}: {info_value}")
            
            return brain_decoding_model
            
        except Exception as error:
            logger.error(f"Failed to initialize model: {error}")
            raise
    
    def perform_brain_decoding_test(
        self, 
        model: StimulusLatentDiffusionModel, 
        data_loader: BrainToStimulusDataLoader
    ) -> Dict[str, np.ndarray]:
        """
        Perform brain decoding test with clean implementation.
        
        Args:
            model: Brain decoding model
            data_loader: Data loader for test data
            
        Returns:
            Dictionary containing test results
        """
        logger.info("Performing brain decoding test...")
        
        # Get test data with clean variable names
        test_data_loader = data_loader.get_test_loader()
        test_batch = next(iter(test_data_loader))
        
        brain_activity_batch = test_batch['brain_activity'].to(self.computation_device)
        true_visual_stimuli = test_batch['visual_stimulus'].to(self.computation_device)
        stimulus_labels_batch = test_batch.get('stimulus_label', None)
        
        # Limit samples for visualization
        max_test_samples = 5
        brain_activity_batch = brain_activity_batch[:max_test_samples]
        true_visual_stimuli = true_visual_stimuli[:max_test_samples]
        if stimulus_labels_batch is not None:
            stimulus_labels_batch = stimulus_labels_batch[:max_test_samples]
        
        logger.info(f"Testing brain decoding on {len(brain_activity_batch)} samples")
        
        # Perform reconstruction with clean method calls
        with torch.no_grad():
            baseline_reconstruction = model.reconstruct_stimulus(
                brain_activity_batch, 
                use_diffusion=False
            )
            enhanced_reconstruction = model.reconstruct_stimulus(
                brain_activity_batch, 
                use_diffusion=True
            )
        
        # Convert to numpy with clean variable names
        test_results = {
            'brain_activity': brain_activity_batch.cpu().numpy(),
            'true_stimuli': true_visual_stimuli.cpu().numpy(),
            'baseline_reconstruction': baseline_reconstruction.cpu().numpy(),
            'enhanced_reconstruction': enhanced_reconstruction.cpu().numpy(),
            'stimulus_labels': stimulus_labels_batch.cpu().numpy() if stimulus_labels_batch is not None else None
        }
        
        logger.info("Brain decoding test completed successfully")
        return test_results
    
    def compute_reconstruction_quality_metrics(self, test_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics with clean implementation.
        
        Args:
            test_results: Dictionary containing test results
            
        Returns:
            Dictionary containing quality metrics
        """
        logger.info("Computing reconstruction quality metrics...")
        
        true_stimuli = test_results['true_stimuli']
        baseline_reconstruction = test_results['baseline_reconstruction']
        enhanced_reconstruction = test_results['enhanced_reconstruction']
        
        # Compute correlation metrics with clean variable names
        baseline_correlations = []
        enhanced_correlations = []
        
        for sample_index in range(len(true_stimuli)):
            baseline_correlation, _ = stats.pearsonr(
                true_stimuli[sample_index], 
                baseline_reconstruction[sample_index]
            )
            enhanced_correlation, _ = stats.pearsonr(
                true_stimuli[sample_index], 
                enhanced_reconstruction[sample_index]
            )
            
            baseline_correlations.append(baseline_correlation)
            enhanced_correlations.append(enhanced_correlation)
        
        # Compute error metrics
        baseline_mean_squared_error = np.mean((true_stimuli - baseline_reconstruction) ** 2)
        enhanced_mean_squared_error = np.mean((true_stimuli - enhanced_reconstruction) ** 2)
        
        quality_metrics = {
            'baseline_correlation_mean': np.mean(baseline_correlations),
            'baseline_correlation_std': np.std(baseline_correlations),
            'enhanced_correlation_mean': np.mean(enhanced_correlations),
            'enhanced_correlation_std': np.std(enhanced_correlations),
            'baseline_mse': baseline_mean_squared_error,
            'enhanced_mse': enhanced_mean_squared_error,
            'baseline_correlations': baseline_correlations,
            'enhanced_correlations': enhanced_correlations
        }
        
        logger.info("Quality metrics computed successfully")
        return quality_metrics
    
    def create_clean_visualizations(
        self, 
        test_results: Dict[str, np.ndarray], 
        quality_metrics: Dict[str, float]
    ) -> None:
        """
        Create clean visualizations of brain decoding results.
        
        Args:
            test_results: Dictionary containing test results
            quality_metrics: Dictionary containing quality metrics
        """
        logger.info("Creating clean visualizations...")
        
        self._create_reconstruction_comparison_plot(test_results, quality_metrics)
        self._create_quality_metrics_plot(quality_metrics)
        
        logger.info("Visualizations created successfully")
    
    def _create_reconstruction_comparison_plot(
        self, 
        test_results: Dict[str, np.ndarray], 
        quality_metrics: Dict[str, float]
    ) -> None:
        """Create clean reconstruction comparison plot."""
        true_stimuli = test_results['true_stimuli']
        baseline_reconstruction = test_results['baseline_reconstruction']
        enhanced_reconstruction = test_results['enhanced_reconstruction']
        stimulus_labels = test_results['stimulus_labels']
        
        num_samples = len(true_stimuli)
        figure, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        figure.suptitle(
            'Brain Decoding Results: fMRI → Visual Stimulus Reconstruction', 
            fontsize=16, fontweight='bold'
        )
        
        # Clean column headers
        column_titles = ['True Stimulus', 'Baseline (VAE)', 'Enhanced (LDM)']
        for col_index, title in enumerate(column_titles):
            figure.text(0.2 + col_index * 0.3, 0.95, title, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        
        for sample_index in range(num_samples):
            # Reshape for visualization (28x28 images)
            true_image = true_stimuli[sample_index].reshape(28, 28)
            baseline_image = baseline_reconstruction[sample_index].reshape(28, 28)
            enhanced_image = enhanced_reconstruction[sample_index].reshape(28, 28)
            
            # Plot with clean styling
            axes[sample_index, 0].imshow(true_image, cmap='gray', vmin=0, vmax=1)
            axes[sample_index, 0].set_ylabel(
                f'Sample {sample_index + 1}' + 
                (f'\nDigit: {stimulus_labels[sample_index]}' if stimulus_labels is not None else ''),
                fontsize=12, fontweight='bold'
            )
            axes[sample_index, 0].set_xticks([])
            axes[sample_index, 0].set_yticks([])
            
            axes[sample_index, 1].imshow(baseline_image, cmap='gray', vmin=0, vmax=1)
            axes[sample_index, 1].set_xticks([])
            axes[sample_index, 1].set_yticks([])
            
            axes[sample_index, 2].imshow(enhanced_image, cmap='gray', vmin=0, vmax=1)
            axes[sample_index, 2].set_xticks([])
            axes[sample_index, 2].set_yticks([])
            
            # Add correlation scores with clean formatting
            baseline_correlation = quality_metrics['baseline_correlations'][sample_index]
            enhanced_correlation = quality_metrics['enhanced_correlations'][sample_index]
            
            axes[sample_index, 1].text(
                0.02, 0.98, f'r={baseline_correlation:.3f}',
                transform=axes[sample_index, 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                fontsize=10, verticalalignment='top', fontweight='bold'
            )
            
            axes[sample_index, 2].text(
                0.02, 0.98, f'r={enhanced_correlation:.3f}',
                transform=axes[sample_index, 2].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                fontsize=10, verticalalignment='top', fontweight='bold'
            )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        save_path = self.results_directory / "brain_decoding_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Reconstruction comparison saved: {save_path}")
    
    def _create_quality_metrics_plot(self, quality_metrics: Dict[str, float]) -> None:
        """Create clean quality metrics comparison plot."""
        figure, (correlation_axis, error_axis) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Correlation comparison with clean styling
        method_names = ['Baseline (VAE)', 'Enhanced (LDM)']
        correlation_means = [
            quality_metrics['baseline_correlation_mean'], 
            quality_metrics['enhanced_correlation_mean']
        ]
        correlation_errors = [
            quality_metrics['baseline_correlation_std'], 
            quality_metrics['enhanced_correlation_std']
        ]
        
        correlation_bars = correlation_axis.bar(
            method_names, correlation_means, yerr=correlation_errors, 
            capsize=5, color=['lightcoral', 'lightblue'], alpha=0.7
        )
        correlation_axis.set_title('Reconstruction Correlation', fontweight='bold')
        correlation_axis.set_ylabel('Pearson Correlation')
        correlation_axis.set_ylim(0, 1)
        
        # Add clean value labels
        for bar, mean_val, error_val in zip(correlation_bars, correlation_means, correlation_errors):
            height = bar.get_height()
            correlation_axis.text(
                bar.get_x() + bar.get_width()/2., height + error_val + 0.01,
                f'{mean_val:.3f}±{error_val:.3f}', 
                ha='center', va='bottom', fontweight='bold'
            )
        
        # Error comparison
        error_values = [quality_metrics['baseline_mse'], quality_metrics['enhanced_mse']]
        error_bars = error_axis.bar(method_names, error_values, 
                                   color=['lightcoral', 'lightblue'], alpha=0.7)
        error_axis.set_title('Reconstruction Error', fontweight='bold')
        error_axis.set_ylabel('Mean Squared Error')
        
        for bar, error_val in zip(error_bars, error_values):
            height = bar.get_height()
            error_axis.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{error_val:.4f}', ha='center', va='bottom', fontweight='bold'
            )
        
        plt.tight_layout()
        
        save_path = self.results_directory / "quality_metrics_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Quality metrics plot saved: {save_path}")
    
    def print_clean_summary(self, quality_metrics: Dict[str, float]) -> None:
        """
        Print clean summary of brain decoding results.
        
        Args:
            quality_metrics: Dictionary containing quality metrics
        """
        print("\n" + "=" * 80)
        print("🧠 CLEAN BRAIN DECODING RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n🎯 Task: Reconstruct visual stimuli from fMRI brain activity")
        print(f"📊 Method: Latent Diffusion Model for brain decoding")
        print(f"🔬 Evaluation: Correlation and error metrics")
        
        print(f"\n📊 Reconstruction Quality (Correlation):")
        baseline_corr = quality_metrics['baseline_correlation_mean']
        baseline_std = quality_metrics['baseline_correlation_std']
        enhanced_corr = quality_metrics['enhanced_correlation_mean']
        enhanced_std = quality_metrics['enhanced_correlation_std']
        
        print(f"   • Baseline (VAE): {baseline_corr:.4f} ± {baseline_std:.4f}")
        print(f"   • Enhanced (LDM): {enhanced_corr:.4f} ± {enhanced_std:.4f}")
        
        print(f"\n📉 Reconstruction Error (MSE):")
        print(f"   • Baseline (VAE): {quality_metrics['baseline_mse']:.4f}")
        print(f"   • Enhanced (LDM): {quality_metrics['enhanced_mse']:.4f}")
        
        # Clean improvement analysis
        correlation_improvement = enhanced_corr - baseline_corr
        if correlation_improvement > 0:
            print(f"\n✅ LDM Enhancement: +{correlation_improvement:.4f} correlation improvement")
        else:
            print(f"\n⚠️  LDM Performance: {correlation_improvement:.4f} correlation change")
        
        print(f"\n💡 Interpretation: Brain decoding quality assessment")
        print(f"   • Correlation > 0.3: Good brain decoding performance")
        print(f"   • Correlation > 0.5: Excellent brain decoding performance")
        
        print(f"\n📁 Results saved to: {self.results_directory}")
    
    def run_complete_brain_decoding_test(self) -> None:
        """
        Run complete brain decoding test with clean workflow.
        """
        print("🧠 CLEAN BRAIN DECODING TEST")
        print("Testing visual stimulus reconstruction from fMRI brain activity")
        print("=" * 80)
        
        try:
            # Load data with clean error handling
            data_loader = self.load_brain_stimulus_data()
            
            # Initialize model
            brain_decoding_model = self.initialize_brain_decoding_model()
            
            # Perform test
            test_results = self.perform_brain_decoding_test(brain_decoding_model, data_loader)
            
            # Compute metrics
            quality_metrics = self.compute_reconstruction_quality_metrics(test_results)
            
            # Create visualizations
            self.create_clean_visualizations(test_results, quality_metrics)
            
            # Print summary
            self.print_clean_summary(quality_metrics)
            
            print("\n✅ Clean brain decoding test completed successfully!")
            
        except Exception as error:
            logger.error(f"Brain decoding test failed: {error}")
            raise


def main():
    """Clean main function with proper error handling."""
    try:
        brain_decoding_tester = CleanBrainDecodingTester()
        brain_decoding_tester.run_complete_brain_decoding_test()
    except Exception as error:
        logger.error(f"Application failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
