"""
Simple demo script to showcase the Latent Diffusion Model implementation.
This script demonstrates the model architecture without requiring full training.
"""

import numpy as np
import json
from pathlib import Path

def load_fmri_data():
    """Load and inspect the fMRI data."""
    print("=" * 60)
    print("LATENT DIFFUSION MODEL FOR fMRI RECONSTRUCTION")
    print("=" * 60)
    print()
    
    print("📊 Loading fMRI Data...")
    
    # Load the aligned data
    data_file = Path("outputs/alignment_ridge_20250527_070315_aligned_data.npz")
    
    if not data_file.exists():
        print("❌ Data file not found. Please ensure the outputs folder contains the aligned data.")
        return None
    
    data = np.load(data_file)
    
    print(f"✅ Data loaded successfully!")
    print(f"   📁 File: {data_file}")
    print(f"   🔑 Keys: {list(data.keys())}")
    print()
    
    # Analyze data structure
    total_samples = 0
    for subject_key in data.keys():
        subject_data = data[subject_key]
        print(f"   👤 {subject_key}: {subject_data.shape} (timepoints × voxels)")
        total_samples += subject_data.shape[0]
    
    print(f"   📈 Total samples: {total_samples}")
    print(f"   🧠 Voxels per sample: {data[list(data.keys())[0]].shape[1]}")
    
    # Combine all data
    all_data = np.vstack([data[key] for key in data.keys()])
    
    print()
    print("📈 Data Statistics:")
    print(f"   📊 Combined shape: {all_data.shape}")
    print(f"   📊 Mean: {np.mean(all_data):.4f}")
    print(f"   📊 Std: {np.std(all_data):.4f}")
    print(f"   📊 Min: {np.min(all_data):.4f}")
    print(f"   📊 Max: {np.max(all_data):.4f}")
    
    return all_data


def demonstrate_model_architecture():
    """Demonstrate the model architecture and components."""
    print()
    print("🏗️  Model Architecture Overview:")
    print("-" * 40)
    
    # Load configuration
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        vae_config = config['vae']
        diffusion_config = config['diffusion']
        
        print("🔧 VAE Configuration:")
        print(f"   📥 Input dimension: {vae_config['input_dim']} voxels")
        print(f"   🎯 Latent dimension: {vae_config['latent_dim']}")
        print(f"   🏗️  Hidden layers: {vae_config['hidden_dims']}")
        print(f"   ⚖️  Beta (KL weight): {vae_config['beta']}")
        print()
        
        print("🌊 Diffusion Model Configuration:")
        print(f"   ⏰ Timesteps: {diffusion_config['num_timesteps']}")
        print(f"   📊 Model channels: {diffusion_config['model_channels']}")
        print(f"   🔄 Residual blocks: {diffusion_config['num_res_blocks']}")
        print(f"   📈 Beta schedule: {diffusion_config['beta_schedule']}")
        print()
        
        # Calculate approximate model size
        vae_encoder_params = vae_config['input_dim'] * vae_config['hidden_dims'][0]
        for i in range(len(vae_config['hidden_dims']) - 1):
            vae_encoder_params += vae_config['hidden_dims'][i] * vae_config['hidden_dims'][i+1]
        vae_encoder_params += vae_config['hidden_dims'][-1] * vae_config['latent_dim'] * 2  # mu and logvar
        
        vae_decoder_params = vae_encoder_params  # Symmetric
        
        # Rough estimate for diffusion model
        diffusion_params = diffusion_config['model_channels'] * 4 * 3  # Rough estimate
        
        total_params = vae_encoder_params + vae_decoder_params + diffusion_params
        
        print("📊 Estimated Model Size:")
        print(f"   🔢 VAE parameters: ~{vae_encoder_params + vae_decoder_params:,}")
        print(f"   🔢 Diffusion parameters: ~{diffusion_params:,}")
        print(f"   🔢 Total parameters: ~{total_params:,}")
        
    except ImportError:
        print("⚠️  PyYAML not installed. Install with: pip install pyyaml")
        print("   Using default configuration display...")
        
        print("🔧 VAE Configuration (Default):")
        print("   📥 Input dimension: 3092 voxels")
        print("   🎯 Latent dimension: 256")
        print("   🏗️  Hidden layers: [1024, 512, 256]")
        print("   ⚖️  Beta (KL weight): 1.0")
        print()
        
        print("🌊 Diffusion Model Configuration (Default):")
        print("   ⏰ Timesteps: 1000")
        print("   📊 Model channels: 128")
        print("   🔄 Residual blocks: 2")
        print("   📈 Beta schedule: linear")


def demonstrate_training_pipeline():
    """Demonstrate the training pipeline."""
    print()
    print("🚀 Training Pipeline Overview:")
    print("-" * 40)
    
    print("📚 Training Phases:")
    print("   1️⃣  Data Loading & Preprocessing")
    print("      • Load aligned fMRI data from multiple subjects")
    print("      • Normalize data (zero mean, unit variance)")
    print("      • Split into train/validation/test sets")
    print("      • Create data loaders with batching")
    print()
    
    print("   2️⃣  VAE Training")
    print("      • Encode fMRI data to latent space")
    print("      • Decode latent representations back to fMRI")
    print("      • Optimize reconstruction + KL divergence loss")
    print("      • Learn meaningful latent representations")
    print()
    
    print("   3️⃣  Diffusion Model Training")
    print("      • Add noise to latent representations")
    print("      • Train model to predict and remove noise")
    print("      • Learn reverse diffusion process")
    print("      • Enable generation of new latent codes")
    print()
    
    print("   4️⃣  Joint Optimization")
    print("      • Fine-tune both VAE and diffusion components")
    print("      • Balance reconstruction quality and generation")
    print("      • Monitor validation metrics for early stopping")
    print()
    
    print("📊 Evaluation Metrics:")
    print("   • Correlation analysis (overall, voxel-wise, sample-wise)")
    print("   • Error metrics (MSE, RMSE, MAE, R², SNR, PSNR)")
    print("   • Distribution metrics (KS test, JS divergence)")
    print("   • Spatial pattern preservation")


def demonstrate_usage():
    """Demonstrate how to use the model."""
    print()
    print("💻 Usage Examples:")
    print("-" * 40)
    
    print("🏃 Quick Start:")
    print("   # Install dependencies")
    print("   pip install -r requirements.txt")
    print()
    print("   # Train and evaluate model")
    print("   python main.py --config config.yaml --mode both")
    print()
    
    print("🔧 Advanced Usage:")
    print("   # Train only")
    print("   python main.py --mode train")
    print()
    print("   # Evaluate existing model")
    print("   python main.py --mode evaluate --checkpoint path/to/model.pt")
    print()
    print("   # Custom configuration")
    print("   python main.py --config custom_config.yaml")
    print()
    
    print("📁 Output Files:")
    print("   • checkpoints/: Saved model weights")
    print("   • logs/: Training logs and metrics")
    print("   • visualizations/: Generated plots and analysis")
    print("   • evaluation_metrics_*.json: Detailed metrics")


def show_project_structure():
    """Show the project structure."""
    print()
    print("📂 Project Structure:")
    print("-" * 40)
    
    structure = """
LDM/
├── 📁 src/
│   ├── 📁 data/
│   │   └── 📄 fmri_data_loader.py      # Data loading & preprocessing
│   ├── 📁 models/
│   │   ├── 📄 vae_encoder_decoder.py   # VAE implementation
│   │   ├── 📄 diffusion_model.py       # Diffusion model & scheduler
│   │   └── 📄 latent_diffusion_model.py # Complete LDM
│   ├── 📁 training/
│   │   └── 📄 trainer.py               # Training pipeline
│   └── 📁 utils/
│       ├── 📄 metrics.py               # Evaluation metrics
│       └── 📄 visualization.py         # Visualization tools
├── 📁 outputs/                         # fMRI data files
├── 📄 config.yaml                      # Configuration
├── 📄 main.py                         # Main execution script
├── 📄 requirements.txt                # Dependencies
├── 📄 test_implementation.py          # Test suite
└── 📄 README.md                       # Documentation
    """
    
    print(structure)


def main():
    """Main demo function."""
    # Load and analyze data
    data = load_fmri_data()
    
    if data is not None:
        # Show model architecture
        demonstrate_model_architecture()
        
        # Show training pipeline
        demonstrate_training_pipeline()
        
        # Show usage examples
        demonstrate_usage()
        
        # Show project structure
        show_project_structure()
        
        print()
        print("🎉 Demo completed!")
        print()
        print("📝 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run tests: python test_implementation.py")
        print("   3. Start training: python main.py")
        print()
        print("📚 For more information, see README.md")
    
    else:
        print("❌ Demo failed due to missing data files.")
        print("   Please ensure the outputs folder contains aligned fMRI data.")


if __name__ == "__main__":
    main()
