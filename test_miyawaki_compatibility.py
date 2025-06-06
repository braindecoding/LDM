#!/usr/bin/env python3
"""
🧪 Test Miyawaki Dataset Compatibility
Test script to verify if miyawaki_structured_28x28.mat can be used with the existing Brain LDM code.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data.data_loader import FMRIDataLoader

def test_miyawaki_compatibility():
    """Test if Miyawaki dataset is compatible with existing code."""
    
    print("🧪 TESTING MIYAWAKI DATASET COMPATIBILITY")
    print("=" * 60)
    
    # Test 1: Load Miyawaki dataset
    print("\n📁 Test 1: Loading Miyawaki Dataset")
    try:
        miyawaki_loader = FMRIDataLoader(
            data_path="data/miyawaki_structured_28x28.mat",
            device='cuda',
            normalize_stimuli=True,
            normalize_fmri=True
        )
        print("✅ Miyawaki dataset loaded successfully!")
        
        # Get data info
        train_data = miyawaki_loader.get_train_data()
        test_data = miyawaki_loader.get_test_data()
        
        print(f"📊 Training data:")
        for key, tensor in train_data.items():
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
        
        print(f"📊 Test data:")
        for key, tensor in test_data.items():
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
            
    except Exception as e:
        print(f"❌ Failed to load Miyawaki dataset: {e}")
        return False
    
    # Test 2: Compare with current dataset
    print("\n📊 Test 2: Comparing with Current Dataset")
    try:
        current_loader = FMRIDataLoader(
            data_path="data/digit69_28x28.mat",
            device='cuda'
        )
        
        # Compare dimensions
        miya_train = miyawaki_loader.get_train_data()
        curr_train = current_loader.get_train_data()
        
        print("📈 Dimension Comparison:")
        print(f"  Training samples: {curr_train['stimuli'].shape[0]} → {miya_train['stimuli'].shape[0]} (+{miya_train['stimuli'].shape[0] - curr_train['stimuli'].shape[0]})")
        print(f"  Image dimensions: {curr_train['stimuli'].shape[1]} → {miya_train['stimuli'].shape[1]} ({'✅ SAME' if curr_train['stimuli'].shape[1] == miya_train['stimuli'].shape[1] else '⚠️ DIFFERENT'})")
        print(f"  fMRI voxels: {curr_train['fmri'].shape[1]} → {miya_train['fmri'].shape[1]} ({'✅ SAME' if curr_train['fmri'].shape[1] == miya_train['fmri'].shape[1] else '⚠️ DIFFERENT'})")
        
        # Check labels
        curr_labels = np.unique(curr_train['labels'].numpy())
        miya_labels = np.unique(miya_train['labels'].numpy())
        print(f"  Label classes: {len(curr_labels)} → {len(miya_labels)} classes")
        print(f"    Current: {curr_labels}")
        print(f"    Miyawaki: {miya_labels}")
        
    except Exception as e:
        print(f"❌ Failed to compare datasets: {e}")
        return False
    
    # Test 3: Model compatibility
    print("\n🤖 Test 3: Model Architecture Compatibility")
    try:
        from models.improved_brain_ldm import ImprovedBrainLDM
        
        # Test with Miyawaki dimensions
        miya_fmri_dim = miya_train['fmri'].shape[1]
        
        print(f"📐 Creating model with fMRI dimension: {miya_fmri_dim}")
        model = ImprovedBrainLDM(
            fmri_dim=miya_fmri_dim,  # Use Miyawaki's fMRI dimension
            image_size=28,
            guidance_scale=7.5
        )
        
        # Test forward pass
        batch_size = 4
        test_fmri = torch.randn(batch_size, miya_fmri_dim)
        test_stimuli = torch.randn(batch_size, 784)
        
        print("🔄 Testing forward pass...")
        with torch.no_grad():
            loss_dict = model.compute_loss(test_fmri, test_stimuli)
            print(f"✅ Forward pass successful! Loss: {loss_dict['total_loss']:.6f}")
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False
    
    # Test 4: DataLoader compatibility
    print("\n📦 Test 4: DataLoader Compatibility")
    try:
        train_loader = miyawaki_loader.create_dataloader(
            split='train',
            batch_size=4,
            shuffle=True
        )
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"📦 Batch test successful:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False
    
    # Test 5: Visualization compatibility
    print("\n🎨 Test 5: Visualization Compatibility")
    try:
        # Test image visualization
        test_stimuli = miyawaki_loader.get_stimuli('test')[:5]
        test_labels = miyawaki_loader.get_labels('test')[:5]
        
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        fig.suptitle('Miyawaki Dataset Sample Images', fontsize=14)
        
        for i in range(5):
            image = test_stimuli[i].reshape(28, 28).numpy()
            label = test_labels[i].item()
            
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save test visualization
        output_path = "results/miyawaki_compatibility_test.png"
        Path("results").mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved test visualization: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False
    
    return True

def create_miyawaki_config():
    """Create configuration for Miyawaki dataset."""
    
    print("\n⚙️ CREATING MIYAWAKI CONFIGURATION")
    print("=" * 40)
    
    # Load Miyawaki to get dimensions
    miyawaki_loader = FMRIDataLoader("data/miyawaki_structured_28x28.mat")
    train_data = miyawaki_loader.get_train_data()
    
    config = {
        'dataset_name': 'miyawaki_structured_28x28',
        'data_path': 'data/miyawaki_structured_28x28.mat',
        'fmri_dim': train_data['fmri'].shape[1],
        'image_size': 28,
        'num_classes': len(np.unique(train_data['labels'].numpy())),
        'train_samples': train_data['stimuli'].shape[0],
        'test_samples': miyawaki_loader.get_test_data()['stimuli'].shape[0],
        'label_range': [
            int(np.min(train_data['labels'].numpy())),
            int(np.max(train_data['labels'].numpy()))
        ]
    }
    
    print("📋 Miyawaki Dataset Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

def main():
    """Main test function."""
    print("🧪 MIYAWAKI DATASET COMPATIBILITY TEST")
    print("=" * 50)
    
    # Run compatibility tests
    success = test_miyawaki_compatibility()
    
    if success:
        print("\n🎉 COMPATIBILITY TEST RESULTS")
        print("=" * 35)
        print("✅ ALL TESTS PASSED!")
        print("✅ Miyawaki dataset is COMPATIBLE with existing code")
        print("✅ Can be used as drop-in replacement")
        
        # Create configuration
        config = create_miyawaki_config()
        
        print("\n📝 USAGE INSTRUCTIONS:")
        print("1. Replace data path in scripts:")
        print("   data_path='data/miyawaki_structured_28x28.mat'")
        print(f"2. Update fMRI dimension in model:")
        print(f"   fmri_dim={config['fmri_dim']}")
        print("3. Run training/evaluation as normal")
        
        print("\n🔄 QUICK START:")
        print("PYTHONPATH=src python3 -c \"")
        print("from data.data_loader import load_fmri_data")
        print("loader = load_fmri_data('data/miyawaki_structured_28x28.mat')")
        print("print('✅ Miyawaki dataset ready!')\"")
        
    else:
        print("\n❌ COMPATIBILITY TEST FAILED")
        print("Some modifications may be needed before using Miyawaki dataset")
    
    return success

if __name__ == "__main__":
    main()
