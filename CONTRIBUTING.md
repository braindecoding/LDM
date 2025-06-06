# 🤝 Contributing to Brain LDM

Thank you for your interest in contributing to Brain LDM! This document provides guidelines for contributing to our neural decoding research project.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide system information (OS, Python version, GPU)
- Include error messages and stack traces

### 💡 Feature Requests
- Describe the proposed feature clearly
- Explain the use case and benefits
- Consider implementation complexity
- Discuss potential alternatives

### 🔬 Research Contributions
- Novel architectures or loss functions
- New datasets or evaluation metrics
- Performance optimizations
- Theoretical insights

### 📚 Documentation
- Improve existing documentation
- Add tutorials and examples
- Fix typos and formatting
- Translate documentation

## 🚀 Development Setup

### Prerequisites
```bash
# WSL2 with Ubuntu 20.04+ (recommended environment)
wsl --list --verbose

# Python 3.11 in WSL
python3 --version  # Should show Python 3.11.x

# CUDA-capable GPU with WSL2 support
nvidia-smi

# Git for version control
git --version
```

### Environment Setup (WSL2)
```bash
# In WSL2 terminal
# Clone the repository
git clone https://github.com/your-username/Brain-LDM.git
cd Brain-LDM

# Create virtual environment with Python 3.11 (recommended)
python3.11 -m venv brain_ldm_env
source brain_ldm_env/bin/activate

# Upgrade pip
pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install TensorFlow with CUDA (optional)
python3 -m pip install tensorflow[and-cuda]

# Install dependencies
pip3 install -r requirements.txt

# For development, uncomment dev tools in requirements.txt:
# pytest, black, isort, flake8, mypy, jupyter, etc.

# Verify CUDA setup
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"
python3 -c "import tensorflow as tf; print(f'TF GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
```

### Development Tools (WSL2)
```bash
# In WSL2 with activated virtual environment
# Uncomment development tools in requirements.txt, then:
pip3 install -r requirements.txt

# Or install specific tools as needed:
pip3 install pytest black isort flake8 mypy  # Core dev tools
pip3 install jupyter wandb tensorboard       # Optional tools

# WSL2 specific utilities
sudo apt update
sudo apt install htop nvtop  # For monitoring system resources
```

## 🖥️ WSL2 Development Tips

> 📋 **Complete Setup Guide**: See the detailed WSL2 setup instructions in [README.md](README.md#installation)

### File System Best Practices
```bash
# Store code in WSL filesystem for better performance
~/Brain-LDM/  # Recommended location

# Access Windows files when needed
/mnt/c/Users/[Username]/Documents/datasets/

# Create convenient symlinks
ln -s /mnt/c/Users/[Username]/Documents/datasets ~/datasets
```

### Performance Optimization
```bash
# Monitor WSL2 resource usage
htop          # CPU and memory
nvtop         # GPU usage
nvidia-smi    # Detailed GPU info

# Configure WSL2 memory limits (see README.md for details)
```

### IDE Integration
```bash
# Use VS Code with WSL extension
code .  # Opens current directory in VS Code

# Or use vim/nano for quick edits
vim src/models/improved_brain_ldm.py
```

## 📝 Code Style Guidelines

### Python Style
- Follow PEP 8 conventions
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Add type hints where appropriate

### Formatting Commands (WSL2)
```bash
# In WSL2 terminal with activated virtual environment
# Format code
black src/ --line-length 88
isort src/

# Check style
flake8 src/

# Type checking
mypy src/

# Run all checks together
black src/ --line-length 88 && isort src/ && flake8 src/ && mypy src/
```

### Naming Conventions
- **Classes**: PascalCase (`BrainLDM`, `AdvancedLossFunction`)
- **Functions/Variables**: snake_case (`train_model`, `learning_rate`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_BATCH_SIZE`)
- **Files**: snake_case (`train_miyawaki_optimized.py`)

## 🧪 Testing Guidelines

### Running Tests (WSL2)
```bash
# In WSL2 terminal with activated virtual environment
# Run all tests
python3 -m pytest

# Run with coverage
python3 -m pytest --cov=src/

# Run specific test file
python3 -m pytest tests/test_models.py

# Run with verbose output
python3 -m pytest -v

# Run tests with GPU if available
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_gpu_models.py
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies
- Aim for >80% code coverage

### Test Structure
```python
import pytest
import torch
from src.models.improved_brain_ldm import ImprovedBrainLDM

class TestBrainLDM:
    def test_model_initialization(self):
        """Test model creates correctly with valid parameters."""
        model = ImprovedBrainLDM(fmri_dim=967, image_size=28)
        assert model.fmri_dim == 967
        assert model.image_size == 28
    
    def test_forward_pass(self):
        """Test model forward pass with sample data."""
        model = ImprovedBrainLDM(fmri_dim=967, image_size=28)
        fmri_data = torch.randn(4, 967)
        
        with torch.no_grad():
            features = model.fmri_encoder(fmri_data)
            assert features.shape == (4, 512)
```

## 📊 Dataset Contributions

### Adding New Datasets
1. **Data Format**: Use `.mat` files with standardized structure
2. **Documentation**: Include dataset description and citation
3. **Preprocessing**: Provide preprocessing scripts
4. **Validation**: Include data validation checks

### Dataset Structure
```matlab
% Required fields in .mat file
fmriTrn     % Training fMRI data (n_samples, n_voxels)
stimTrn     % Training stimuli (n_samples, 784)
fmriTest    % Test fMRI data (n_test, n_voxels)
stimTest    % Test stimuli (n_test, 784)
labelTrn    % Training labels (n_samples, 1)
labelTest   % Test labels (n_test, 1)
```

## 🏗️ Architecture Contributions

### Model Improvements
- Maintain backward compatibility
- Include ablation studies
- Document architectural changes
- Provide performance comparisons

### Loss Function Contributions
```python
class CustomLossFunction(nn.Module):
    """
    Custom loss function for brain decoding.
    
    Args:
        weight_param: Weighting parameter for loss components
        
    Returns:
        Dictionary with loss components and total loss
    """
    
    def __init__(self, weight_param=1.0):
        super().__init__()
        self.weight_param = weight_param
    
    def forward(self, predictions, targets, **kwargs):
        # Implementation here
        return {'total_loss': loss, 'component_losses': {...}}
```

## 📈 Performance Optimization

### Benchmarking
- Use consistent hardware for comparisons
- Report training and inference times
- Include memory usage statistics
- Test on multiple datasets

### Optimization Guidelines
- Profile code before optimizing
- Focus on bottlenecks first
- Maintain code readability
- Document performance improvements

## 🔄 Pull Request Process

### Before Submitting
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Test** your changes thoroughly
4. **Format** code according to style guidelines
5. **Update** documentation if needed

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Performance Impact
- Training time: [before] → [after]
- Memory usage: [before] → [after]
- Accuracy: [before] → [after]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process
1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different configurations
4. **Documentation** review
5. **Merge** after approval

## 🚨 Issue Reporting

### Bug Report Template
```markdown
**Bug Description**
Clear description of the bug

**Reproduction Steps**
1. Step one
2. Step two
3. Error occurs

**Expected Behavior**
What should happen

**Environment**
- OS: [e.g., Ubuntu 20.04 in WSL2]
- Python: [e.g., 3.11.12]
- PyTorch: [e.g., 2.7.1+cu128]
- CUDA: [e.g., 12.9]
- GPU: [e.g., RTX 3060]

**Additional Context**
Screenshots, logs, etc.
```

## 🎓 Research Guidelines

### Experimental Standards
- Use random seeds for reproducibility
- Report confidence intervals
- Include statistical significance tests
- Compare against established baselines

### Publication Guidelines
- Cite original datasets and methods
- Include code availability statement
- Follow journal-specific requirements
- Share preprocessed data when possible

## 📞 Communication

### Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: Direct contact for sensitive issues

### Response Times
- **Bug reports**: 48 hours acknowledgment
- **Feature requests**: 1 week initial response
- **Pull requests**: 3-5 days review time

## 🏆 Recognition

### Contributors
All contributors will be:
- Listed in the CONTRIBUTORS.md file
- Acknowledged in research publications
- Invited to co-author relevant papers

### Contribution Types
- 🐛 Bug fixes
- 💡 Feature additions
- 📚 Documentation
- 🔬 Research insights
- 🎨 Visualizations
- 🧪 Testing

## 📜 Code of Conduct

### Our Standards
- **Respectful** communication
- **Constructive** feedback
- **Inclusive** environment
- **Professional** behavior

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Unprofessional conduct

## 🙏 Thank You

Your contributions help advance the field of computational neuroscience and brain-computer interfaces. Every contribution, no matter how small, makes a difference!

---

**Happy Contributing! 🧠✨**
