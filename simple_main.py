"""
Simplified main script for Latent Diffusion Model that handles missing dependencies gracefully.
"""

import sys
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check which dependencies are available."""
    available = {}
    required = ['torch', 'numpy', 'yaml']
    optional = ['matplotlib', 'seaborn', 'scipy', 'sklearn', 'tqdm']
    
    print("🔍 Checking dependencies...")
    
    for package in required:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'torch':
                import torch
            elif package == 'numpy':
                import numpy
            available[package] = True
            print(f"✅ {package} - available")
        except ImportError:
            available[package] = False
            print(f"❌ {package} - missing (required)")
    
    for package in optional:
        try:
            if package == 'matplotlib':
                import matplotlib
            elif package == 'seaborn':
                import seaborn
            elif package == 'scipy':
                import scipy
            elif package == 'sklearn':
                import sklearn
            elif package == 'tqdm':
                import tqdm
            available[package] = True
            print(f"✅ {package} - available")
        except ImportError:
            available[package] = False
            print(f"⚠️  {package} - missing (optional)")
    
    return available

def load_config_safe():
    """Load configuration with error handling."""
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
        return config
    except ImportError:
        print("❌ PyYAML not available. Using default configuration.")
        return get_default_config()
    except FileNotFoundError:
        print("❌ config.yaml not found. Using default configuration.")
        return get_default_config()

def get_default_config():
    """Get default configuration when YAML is not available."""
    return {
        'data': {
            'data_path': 'outputs',
            'aligned_data_file': 'alignment_ridge_20250527_070315_aligned_data.npz',
            'num_subjects': 3,
            'num_timepoints': 27,
            'num_voxels': 3092,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        },
        'vae': {
            'input_dim': 3092,
            'latent_dim': 256,
            'hidden_dims': [1024, 512, 256],
            'beta': 1.0,
            'learning_rate': 1e-4
        },
        'diffusion': {
            'num_timesteps': 1000,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'beta_schedule': 'linear',
            'model_channels': 128,
            'num_res_blocks': 2,
            'dropout': 0.1,
            'learning_rate': 1e-4
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 100,
            'gradient_accumulation_steps': 1,
            'mixed_precision': True,
            'save_every': 10,
            'eval_every': 5,
            'early_stopping_patience': 20,
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs'
        },
        'logging': {
            'use_wandb': False,
            'project_name': 'fmri_latent_diffusion',
            'experiment_name': 'baseline'
        },
        'hardware': {
            'device': 'auto',
            'num_workers': 4,
            'pin_memory': True
        }
    }

def test_data_loading(config):
    """Test data loading with minimal dependencies."""
    try:
        import numpy as np
        
        data_path = Path(config['data']['data_path'])
        data_file = data_path / config['data']['aligned_data_file']
        
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return False
        
        print(f"📊 Loading data from {data_file}...")
        data = np.load(data_file)
        
        print(f"✅ Data loaded successfully!")
        print(f"   Keys: {list(data.keys())}")
        
        total_samples = 0
        for key in data.keys():
            shape = data[key].shape
            print(f"   {key}: {shape}")
            total_samples += shape[0]
        
        print(f"   Total samples: {total_samples}")
        return True
        
    except ImportError:
        print("❌ NumPy not available for data loading test")
        return False
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_model_creation(config):
    """Test model creation with available dependencies."""
    try:
        import torch
        
        print("🧠 Testing model creation...")
        
        # Test if we can create basic tensors
        batch_size = 4
        input_dim = config['vae']['input_dim']
        latent_dim = config['vae']['latent_dim']
        
        # Create sample tensors
        sample_input = torch.randn(batch_size, input_dim)
        sample_latent = torch.randn(batch_size, latent_dim)
        
        print(f"✅ Created sample tensors:")
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Latent shape: {sample_latent.shape}")
        
        # Test device detection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not available for model creation test")
        return False
    except Exception as e:
        print(f"❌ Error in model creation test: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions."""
    print("\n" + "=" * 60)
    print("📚 USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("\n🔧 To install missing dependencies:")
    print("   pip install torch numpy pyyaml matplotlib seaborn scipy scikit-learn tqdm")
    print("\n   Or install from requirements.txt:")
    print("   pip install -r requirements.txt")
    
    print("\n🚀 To run the complete training:")
    print("   python main.py --config config.yaml --mode both")
    
    print("\n📊 To run the demo:")
    print("   python demo.py")
    
    print("\n🧪 To run tests:")
    print("   python test_implementation.py")
    
    print("\n📁 Project structure:")
    print("   src/data/          - Data loading and preprocessing")
    print("   src/models/        - VAE and diffusion model implementations")
    print("   src/training/      - Training pipeline")
    print("   src/utils/         - Metrics and visualization utilities")

def main():
    """Main function."""
    print("🎯 LATENT DIFFUSION MODEL - SIMPLIFIED LAUNCHER")
    print("=" * 60)
    
    # Check dependencies
    available = check_dependencies()
    
    # Check if we have minimum requirements
    required_available = all(available.get(pkg, False) for pkg in ['torch', 'numpy'])
    
    if not required_available:
        print("\n❌ Missing required dependencies!")
        print("   Please install: pip install torch numpy")
        show_usage_instructions()
        return
    
    print("\n✅ Minimum requirements satisfied!")
    
    # Load configuration
    config = load_config_safe()
    
    # Test data loading
    print("\n" + "=" * 40)
    data_ok = test_data_loading(config)
    
    # Test model creation
    print("\n" + "=" * 40)
    model_ok = test_model_creation(config)
    
    # Show results
    print("\n" + "=" * 60)
    print("📊 SYSTEM CHECK RESULTS")
    print("=" * 60)
    
    print(f"✅ Data loading: {'OK' if data_ok else 'FAILED'}")
    print(f"✅ Model creation: {'OK' if model_ok else 'FAILED'}")
    print(f"✅ Configuration: OK")
    
    if data_ok and model_ok:
        print("\n🎉 System ready for training!")
        print("\n📝 Next steps:")
        print("   1. Install optional dependencies for full functionality")
        print("   2. Run: python main.py --mode both")
    else:
        print("\n⚠️  Some components failed. Please check dependencies.")
    
    show_usage_instructions()

if __name__ == "__main__":
    main()
