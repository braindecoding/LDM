"""
Installation script for Latent Diffusion Model dependencies.
This script installs the required packages step by step.
"""

import subprocess
import sys
import importlib

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main installation function."""
    print("🚀 Installing Latent Diffusion Model Dependencies")
    print("=" * 60)
    
    # Core dependencies
    core_packages = [
        ("torch", "torch>=2.0.0"),
        ("numpy", "numpy>=1.21.0"),
        ("scipy", "scipy>=1.7.0"),
        ("sklearn", "scikit-learn>=1.0.0"),
        ("tqdm", "tqdm>=4.62.0"),
        ("yaml", "pyyaml>=6.0"),
    ]
    
    # Visualization dependencies
    viz_packages = [
        ("matplotlib", "matplotlib>=3.5.0"),
        ("seaborn", "seaborn>=0.11.0"),
    ]
    
    # Optional dependencies
    optional_packages = [
        ("wandb", "wandb>=0.12.0"),
    ]
    
    print("📦 Installing core dependencies...")
    for module_name, package_spec in core_packages:
        if check_package(module_name):
            print(f"✅ {module_name} already installed")
        else:
            print(f"📥 Installing {package_spec}...")
            if install_package(package_spec):
                print(f"✅ {module_name} installed successfully")
            else:
                print(f"❌ Failed to install {module_name}")
    
    print("\n📊 Installing visualization dependencies...")
    for module_name, package_spec in viz_packages:
        if check_package(module_name):
            print(f"✅ {module_name} already installed")
        else:
            print(f"📥 Installing {package_spec}...")
            if install_package(package_spec):
                print(f"✅ {module_name} installed successfully")
            else:
                print(f"❌ Failed to install {module_name}")
    
    print("\n🔧 Installing optional dependencies...")
    for module_name, package_spec in optional_packages:
        if check_package(module_name):
            print(f"✅ {module_name} already installed")
        else:
            print(f"📥 Installing {package_spec} (optional)...")
            if install_package(package_spec):
                print(f"✅ {module_name} installed successfully")
            else:
                print(f"⚠️  Failed to install {module_name} (optional - can continue without it)")
    
    print("\n🎉 Installation completed!")
    print("\n📝 Next steps:")
    print("   1. Run demo: python demo.py")
    print("   2. Run tests: python test_implementation.py")
    print("   3. Start training: python main.py")

if __name__ == "__main__":
    main()
