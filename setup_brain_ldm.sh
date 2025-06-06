#!/bin/bash
set -e

echo "🚀 Setting up Brain LDM environment..."
echo "This will take about 5 minutes."

# Check if we're in WSL
if ! grep -q microsoft /proc/version; then
    echo "❌ This script must be run in WSL2"
    echo "Please install WSL2 first: https://docs.microsoft.com/en-us/windows/wsl/install"
    exit 1
fi

# Check if NVIDIA driver is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found in WSL2"
    echo "Please install NVIDIA drivers for WSL2: https://developer.nvidia.com/cuda/wsl"
    exit 1
fi

echo "✅ WSL2 and NVIDIA drivers detected"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
echo "🐍 Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3.11 -m venv brain_ldm_env
source brain_ldm_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8
echo "🔥 Installing PyTorch with CUDA 12.8..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install TensorFlow (optional)
echo "📊 Installing TensorFlow with CUDA..."
python3 -m pip install tensorflow[and-cuda]

# Install Brain LDM dependencies
echo "🧠 Installing Brain LDM dependencies..."
pip3 install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
import tensorflow as tf
print('✅ Python:', __import__('sys').version.split()[0])
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    print('✅ cuDNN:', torch.backends.cudnn.version())
print('✅ TensorFlow:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('✅ TensorFlow GPUs:', len(gpus), 'detected')
"

# Create convenient aliases
echo "🔧 Setting up aliases..."
cat >> ~/.bashrc << 'EOF'

# Brain LDM aliases
alias brain_env="source ~/brain_ldm_env/bin/activate"
alias brain_cd="cd ~/Brain-LDM"
alias gpu_status="nvidia-smi"

# Quick start function
brain_start() {
    source ~/brain_ldm_env/bin/activate
    cd ~/Brain-LDM
    echo "🧠 Brain LDM environment activated!"
    python3 -c "import torch; print(f'🎮 CUDA available: {torch.cuda.is_available()}')"
}
EOF

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Quick commands:"
echo "  brain_start    # Activate environment and navigate to project"
echo "  gpu_status     # Check GPU status"
echo ""
echo "🚀 To start training:"
echo "  source brain_ldm_env/bin/activate"
echo "  PYTHONPATH=src python3 train_miyawaki_optimized.py"
echo ""
echo "💡 Reload your terminal or run: source ~/.bashrc"
