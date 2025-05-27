"""
🎯 Example Usage: Simple fMRI Data Loader

Quick examples of how to use the data loader.
"""

from data_loader import load_fmri_data


def basic_usage():
    """Basic usage example."""
    print("📖 Basic Usage Example")
    print("-" * 30)
    
    # Load data
    loader = load_fmri_data()
    
    # Get training data
    train_stimuli = loader.get_stimuli('train')  # (90, 784)
    train_fmri = loader.get_fmri('train')        # (90, 3092)
    train_labels = loader.get_labels('train')    # (90,)
    
    print(f"Training data loaded:")
    print(f"  Stimuli: {train_stimuli.shape}")
    print(f"  fMRI: {train_fmri.shape}")
    print(f"  Labels: {train_labels.shape}")
    
    # Get test data
    test_stimuli = loader.get_stimuli('test')    # (10, 784)
    test_fmri = loader.get_fmri('test')          # (10, 3092)
    
    print(f"Test data loaded:")
    print(f"  Stimuli: {test_stimuli.shape}")
    print(f"  fMRI: {test_fmri.shape}")


def dataloader_usage():
    """DataLoader usage example."""
    print("\n🔄 DataLoader Usage Example")
    print("-" * 30)
    
    loader = load_fmri_data()
    
    # Create DataLoader
    train_loader = loader.create_dataloader(
        split='train',
        batch_size=8,
        shuffle=True
    )
    
    print(f"Created DataLoader with {len(train_loader)} batches")
    
    # Iterate through batches
    for batch_idx, batch in enumerate(train_loader):
        stimulus_batch = batch['stimulus']  # (8, 784)
        fmri_batch = batch['fmri']         # (8, 3092)
        label_batch = batch['label']       # (8,)
        
        print(f"Batch {batch_idx}: stimulus {stimulus_batch.shape}, fMRI {fmri_batch.shape}")
        
        if batch_idx >= 2:  # Show only first 3 batches
            break


def brain_decoding_setup():
    """Brain decoding experiment setup."""
    print("\n🧠 Brain Decoding Setup Example")
    print("-" * 30)
    
    loader = load_fmri_data()
    
    # For brain decoding: fMRI -> Stimulus
    train_fmri = loader.get_fmri('train')      # Input: brain activity
    train_stimuli = loader.get_stimuli('train') # Target: stimulus images
    
    test_fmri = loader.get_fmri('test')
    test_stimuli = loader.get_stimuli('test')
    
    print(f"Brain decoding setup:")
    print(f"  Input (fMRI): {train_fmri.shape} -> {test_fmri.shape}")
    print(f"  Target (Stimulus): {train_stimuli.shape} -> {test_stimuli.shape}")
    print(f"  Task: Reconstruct 28x28 images from brain activity")


def visualization_example():
    """Visualization example."""
    print("\n🎨 Visualization Example")
    print("-" * 30)
    
    loader = load_fmri_data()
    
    # Visualize some samples
    print("Creating visualization...")
    loader.visualize_samples(
        num_samples=10,
        split='train',
        save_path='sample_stimuli.png'
    )
    
    # Get individual stimulus as image
    stimulus_img = loader.get_stimulus_as_image(0, 'train')
    print(f"Individual stimulus shape: {stimulus_img.shape}")  # (28, 28)


def main():
    """Run all examples."""
    print("🎯 fMRI Data Loader Examples")
    print("=" * 40)
    
    basic_usage()
    dataloader_usage()
    brain_decoding_setup()
    visualization_example()
    
    print(f"\n✅ All examples completed!")
    print(f"📁 Files created: sample_stimuli.png")
    print(f"🚀 Ready to start brain decoding experiments!")


if __name__ == "__main__":
    main()
