# Installation Guide

## Multi-Modal Brain LDM with Uncertainty Quantification

This guide provides step-by-step instructions for installing and setting up the Multi-Modal Brain Latent Diffusion Model with uncertainty quantification.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM (16GB recommended)
- **Storage**: 5GB free space (10GB recommended)
- **CPU**: Multi-core processor (4+ cores recommended)

### Recommended Requirements
- **Memory**: 16GB+ RAM
- **Storage**: 20GB+ free space
- **CPU**: 8+ cores with AVX support
- **GPU**: CUDA-compatible GPU (optional, for acceleration)

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/[username]/Brain-LDM-Uncertainty.git
cd Brain-LDM-Uncertainty

# Create virtual environment
python -m venv brain_ldm_env
source brain_ldm_env/bin/activate  # On Windows: brain_ldm_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Method 2: Using conda

```bash
# Clone the repository
git clone https://github.com/[username]/Brain-LDM-Uncertainty.git
cd Brain-LDM-Uncertainty

# Create conda environment
conda create -n brain_ldm python=3.8
conda activate brain_ldm

# Install PyTorch
conda install pytorch torchvision -c pytorch

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Method 3: Using uv (Fast Package Manager)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/[username]/Brain-LDM-Uncertainty.git
cd Brain-LDM-Uncertainty

# Create and activate environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv add torch>=2.0.0 torchvision>=0.15.0
uv add numpy scipy matplotlib seaborn scikit-learn
uv add tqdm pyyaml

# Verify installation
uv run python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Dependency Details

### Core Dependencies

```txt
# Deep Learning Framework
torch>=2.0.0                # PyTorch for neural networks
torchvision>=0.15.0         # Computer vision utilities

# Scientific Computing
numpy>=1.21.0               # Numerical computing
scipy>=1.7.0                # Scientific algorithms
scikit-learn>=1.0.0         # Machine learning utilities

# Visualization
matplotlib>=3.5.0           # Plotting library
seaborn>=0.11.0            # Statistical visualization

# Utilities
tqdm>=4.62.0               # Progress bars
PyYAML>=6.0                # Configuration files
```

### Optional Dependencies

```txt
# Jupyter Notebooks
jupyter>=1.0.0
ipykernel>=6.0.0
notebook>=6.4.0

# Development Tools
pytest>=6.0.0             # Testing framework
black>=21.0.0              # Code formatting
flake8>=3.9.0              # Code linting

# Documentation
sphinx>=4.0.0              # Documentation generation
sphinx-rtd-theme>=1.0.0    # Documentation theme
```

## Data Setup

### Download Required Data

```bash
# Create data directory
mkdir -p data/raw

# Download the fMRI dataset (example)
# Note: Replace with actual data source
wget -O data/raw/digit69_28x28.mat [DATA_URL]

# Verify data integrity
python -c "
import scipy.io
data = scipy.io.loadmat('data/raw/digit69_28x28.mat')
print('Data keys:', list(data.keys()))
print('Shapes:', {k: v.shape for k, v in data.items() if hasattr(v, 'shape')})
"
```

### Data Preprocessing

```bash
# Run data preprocessing
python src/data/preprocessing.py --input data/raw/digit69_28x28.mat --output data/processed/

# Verify processed data
ls -la data/processed/
```

## Model Setup

### Download Pre-trained Models (Optional)

```bash
# Create models directory
mkdir -p models/checkpoints

# Download pre-trained models (if available)
# Note: Replace with actual model URLs
# wget -O models/checkpoints/best_improved_v1_model.pt [MODEL_URL]
```

### Verify Installation

```bash
# Run installation verification script
python scripts/verify_installation.py
```

Expected output:
```
✅ Python version: 3.8.x
✅ PyTorch version: 2.0.x
✅ All dependencies installed
✅ Data files accessible
✅ GPU available: [True/False]
✅ Installation successful!
```

## Configuration

### Environment Variables

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export BRAIN_LDM_ROOT="/path/to/Brain-LDM-Uncertainty"
export PYTHONPATH="${BRAIN_LDM_ROOT}/src:${PYTHONPATH}"
```

### Configuration Files

Create `config/default.yaml`:
```yaml
# Default configuration
data:
  root_dir: "data"
  batch_size: 4
  num_workers: 2

model:
  fmri_dim: 3092
  image_size: 28
  guidance_scale: 7.5

training:
  epochs: 150
  learning_rate: 8e-5
  weight_decay: 5e-6

uncertainty:
  n_samples: 30
  temperature_init: 1.0
  dropout_rate: 0.2
```

## Quick Test

### Basic Functionality Test

```bash
# Test data loading
python -c "
from src.data.data_loader import load_fmri_data
loader = load_fmri_data()
print('✅ Data loading successful')
"

# Test model creation
python -c "
from src.models.improved_brain_ldm import ImprovedBrainLDM
model = ImprovedBrainLDM(fmri_dim=3092, image_size=28)
print('✅ Model creation successful')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# Test uncertainty evaluation
python -c "
from src.evaluation.uncertainty_evaluation import monte_carlo_sampling
print('✅ Uncertainty evaluation modules loaded')
"
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation Issues

**Problem**: `ImportError: No module named 'torch'`

**Solution**:
```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Memory Issues

**Problem**: `RuntimeError: out of memory`

**Solution**:
```bash
# Reduce batch size in configuration
# Edit config/default.yaml:
# batch_size: 2  # Reduce from 4 to 2
```

#### 3. Data Loading Issues

**Problem**: `FileNotFoundError: data/raw/digit69_28x28.mat`

**Solution**:
```bash
# Verify data file exists
ls -la data/raw/
# If missing, download or copy the data file
```

#### 4. Import Path Issues

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
# Or run from project root with:
python -m src.training.train_improved_model
```

### Performance Optimization

#### CPU Optimization

```bash
# Set optimal number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Enable CPU optimizations
python -c "
import torch
torch.set_num_threads(4)
print('CPU threads set to 4')
"
```

#### Memory Optimization

```python
# In your training script
import torch

# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(False)  # If using CPU

# Clear cache periodically
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
```

## Development Setup

### For Contributors

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/
isort src/

# Lint code
flake8 src/
```

### IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./brain_ldm_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

## Support

### Getting Help

1. **Check Documentation**: Review `docs/` directory
2. **Search Issues**: Check GitHub issues for similar problems
3. **Create Issue**: Open a new issue with detailed information
4. **Contact**: Email [support@institution.edu]

### Reporting Bugs

When reporting bugs, please include:
- Operating system and version
- Python version
- PyTorch version
- Complete error traceback
- Steps to reproduce
- Expected vs actual behavior

---

## Next Steps

After successful installation:

1. **Read Methodology**: Review `METHODOLOGY.md`
2. **Explore Examples**: Check `experiments/notebooks/`
3. **Run Training**: Follow `docs/usage.md`
4. **Analyze Results**: Use evaluation scripts

For detailed usage instructions, see `docs/usage.md`.
