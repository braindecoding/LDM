"""
Clean setup script for the fMRI Latent Diffusion Model project.
Handles dependency installation and project initialization.
"""

import subprocess
import sys
import os
from pathlib import Path


class ProjectSetup:
    """Clean project setup manager."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.project_root = Path(__file__).parent.parent
        self.required_dirs = [
            "checkpoints",
            "logs", 
            "results",
            "outputs"
        ]
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ is required")
            print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
            return False
        
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        print("📦 Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_directories(self) -> None:
        """Create required project directories."""
        print("📁 Creating project directories...")
        
        for dir_name in self.required_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"   ✅ {dir_name}/")
    
    def check_data_availability(self) -> bool:
        """Check if fMRI data is available."""
        print("🧠 Checking fMRI data availability...")
        
        outputs_dir = self.project_root / "outputs"
        data_files = list(outputs_dir.glob("alignment_*.npz"))
        
        if not data_files:
            print("⚠️  No aligned fMRI data found in outputs/")
            print("   Please ensure you have aligned fMRI data files")
            print("   Expected format: alignment_*.npz")
            return False
        
        print(f"✅ Found {len(data_files)} data file(s):")
        for data_file in data_files:
            print(f"   • {data_file.name}")
        
        return True
    
    def verify_installation(self) -> bool:
        """Verify that key packages can be imported."""
        print("🔍 Verifying installation...")
        
        required_packages = [
            "torch",
            "numpy", 
            "matplotlib",
            "sklearn",
            "yaml",
            "scipy"
        ]
        
        failed_imports = []
        
        for package in required_packages:
            try:
                if package == "sklearn":
                    import sklearn
                elif package == "yaml":
                    import yaml
                else:
                    __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package}")
                failed_imports.append(package)
        
        if failed_imports:
            print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
            return False
        
        print("✅ All packages imported successfully")
        return True
    
    def check_gpu_availability(self) -> None:
        """Check GPU availability for training."""
        print("🖥️  Checking GPU availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ CUDA available: {gpu_count} GPU(s)")
                print(f"   Primary GPU: {gpu_name}")
            else:
                print("⚠️  CUDA not available - will use CPU")
                print("   Training will be slower but still functional")
                
        except ImportError:
            print("⚠️  PyTorch not installed - cannot check GPU")
    
    def show_next_steps(self) -> None:
        """Show next steps after setup."""
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        print("\n🚀 Next Steps:")
        print("1. 📚 Understand the project:")
        print("   python demo.py")
        
        print("\n2. 🏃 Quick start (train + evaluate):")
        print("   python scripts/train_model.py --mode both")
        
        print("\n3. 📊 View results:")
        print("   python scripts/visualize_results.py")
        
        print("\n4. 📖 Read documentation:")
        print("   Check README_clean.md for detailed information")
        
        print("\n💡 Tips:")
        print("   • Start with demo.py to understand the project")
        print("   • Use --mode train for training only")
        print("   • Check logs/ folder for training progress")
        print("   • Results are saved in results/ folder")
    
    def run_setup(self) -> bool:
        """Run complete project setup."""
        print("🧠 FMRI LATENT DIFFUSION MODEL - PROJECT SETUP")
        print("=" * 60)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Verify installation
        if not self.verify_installation():
            return False
        
        # Check GPU
        self.check_gpu_availability()
        
        # Check data
        data_available = self.check_data_availability()
        
        # Show next steps
        self.show_next_steps()
        
        if not data_available:
            print("\n⚠️  Note: No fMRI data found. You'll need aligned data to train.")
        
        return True


def main():
    """Main setup function."""
    setup = ProjectSetup()
    success = setup.run_setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
