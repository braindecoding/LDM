#!/usr/bin/env python3
"""
🧠 Run Brain LDM Experiment with Miyawaki Dataset
Quick script to run the Brain LDM experiment using miyawaki_structured_28x28.mat dataset.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from pathlib import Path
from data.data_loader import load_fmri_data

def run_miyawaki_visualization():
    """Run visualization with Miyawaki dataset."""
    
    print("🎨 BRAIN LDM: MIYAWAKI DATASET VISUALIZATION")
    print("=" * 60)
    
    # Load Miyawaki dataset
    print("📁 Loading Miyawaki dataset...")
    loader = load_fmri_data("data/miyawaki_structured_28x28.mat")
    
    # Get test data for visualization
    test_stimuli = loader.get_stimuli('test')
    test_labels = loader.get_labels('test')
    test_fmri = loader.get_fmri('test')
    
    print(f"✅ Miyawaki dataset loaded successfully!")
    print(f"📊 Test data: {test_stimuli.shape[0]} samples")
    print(f"🏷️ Label range: {test_labels.min().item()} to {test_labels.max().item()}")
    print(f"🧠 fMRI dimensions: {test_fmri.shape[1]} voxels")
    
    # Create simple visualization
    import matplotlib.pyplot as plt
    
    # Show first 10 samples
    n_samples = min(10, test_stimuli.shape[0])
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Miyawaki Dataset - Sample Images', fontsize=16)
    
    for i in range(n_samples):
        row = i // 5
        col = i % 5
        
        # Get image and label
        image = test_stimuli[i].reshape(28, 28).numpy()
        label = test_labels[i].item()
        
        # Display image
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'Sample #{i+1}\nLabel: {label}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = "results/miyawaki_dataset_samples.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved visualization: {output_path}")
    plt.close()
    
    return loader

def create_miyawaki_config():
    """Create configuration file for Miyawaki dataset."""
    
    print("\n⚙️ CREATING MIYAWAKI CONFIGURATION")
    print("=" * 40)
    
    # Load dataset to get actual dimensions
    loader = load_fmri_data("data/miyawaki_structured_28x28.mat")
    train_data = loader.get_train_data()
    test_data = loader.get_test_data()
    
    config = {
        'dataset_name': 'miyawaki_structured_28x28',
        'data_path': 'data/miyawaki_structured_28x28.mat',
        'fmri_dim': train_data['fmri'].shape[1],
        'image_size': 28,
        'num_classes': len(np.unique(train_data['labels'].numpy())),
        'train_samples': train_data['stimuli'].shape[0],
        'test_samples': test_data['stimuli'].shape[0],
        'label_range': [
            int(np.min(train_data['labels'].numpy())),
            int(np.max(train_data['labels'].numpy()))
        ],
        'data_types': {
            'stimuli': str(train_data['stimuli'].dtype),
            'fmri': str(train_data['fmri'].dtype),
            'labels': str(train_data['labels'].dtype)
        }
    }
    
    print("📋 Miyawaki Dataset Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Save config to file
    import json
    config_path = "results/miyawaki_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Configuration saved to: {config_path}")
    
    return config

def test_miyawaki_model():
    """Test model creation with Miyawaki dimensions."""
    
    print("\n🤖 TESTING MODEL WITH MIYAWAKI DIMENSIONS")
    print("=" * 50)
    
    try:
        from models.improved_brain_ldm import ImprovedBrainLDM
        
        # Get Miyawaki fMRI dimension
        loader = load_fmri_data("data/miyawaki_structured_28x28.mat")
        fmri_dim = loader.get_fmri('train').shape[1]
        
        print(f"📐 Creating model with fMRI dimension: {fmri_dim}")
        
        # Create model with Miyawaki dimensions
        model = ImprovedBrainLDM(
            fmri_dim=fmri_dim,
            image_size=28,
            guidance_scale=7.5
        )
        
        print(f"✅ Model created successfully!")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with sample data
        batch_size = 4
        sample_fmri = torch.randn(batch_size, fmri_dim)
        sample_stimuli = torch.randn(batch_size, 784)
        
        print(f"🔄 Testing forward pass...")
        
        # Test forward pass (simplified)
        with torch.no_grad():
            # Test encoding
            fmri_features = model.fmri_encoder(sample_fmri)
            print(f"✅ fMRI encoding successful: {fmri_features.shape}")
            
            # Test image processing
            image_features = model.image_encoder(sample_stimuli.view(batch_size, 1, 28, 28))
            print(f"✅ Image encoding successful: {image_features.shape}")
        
        print(f"🎉 Model compatibility test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main function to run Miyawaki experiment."""
    
    print("🧠 MIYAWAKI DATASET EXPERIMENT")
    print("=" * 40)
    
    # Step 1: Visualize dataset
    loader = run_miyawaki_visualization()
    
    # Step 2: Create configuration
    config = create_miyawaki_config()
    
    # Step 3: Test model compatibility
    model_success = test_miyawaki_model()
    
    # Summary
    print("\n🎯 EXPERIMENT SUMMARY")
    print("=" * 25)
    
    if model_success:
        print("✅ ALL TESTS PASSED!")
        print("✅ Miyawaki dataset is ready to use")
        
        print("\n📝 NEXT STEPS:")
        print("1. Update training scripts to use Miyawaki dataset:")
        print("   - Change data_path to 'data/miyawaki_structured_28x28.mat'")
        print(f"   - Update fmri_dim to {config['fmri_dim']}")
        
        print("\n2. Run training:")
        print("   PYTHONPATH=src python3 src/training/train_improved_model.py")
        
        print("\n3. Run evaluation:")
        print("   PYTHONPATH=src python3 src/evaluation/comprehensive_analysis.py")
        
        print(f"\n📊 DATASET ADVANTAGES:")
        print(f"   • {config['train_samples']} training samples (vs 90 in digit69)")
        print(f"   • {config['test_samples']} test samples (vs 10 in digit69)")
        print(f"   • {config['num_classes']} classes (vs 2 in digit69)")
        print(f"   • More challenging and realistic dataset")
        
    else:
        print("❌ Some issues found - check model compatibility")
    
    print(f"\n📁 Results saved to: results/")
    print(f"   • miyawaki_dataset_samples.png")
    print(f"   • miyawaki_config.json")

if __name__ == "__main__":
    main()
