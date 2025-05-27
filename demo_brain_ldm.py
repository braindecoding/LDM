"""
🎯 Demo: Brain Decoding LDM

Complete demo showing how to use the Brain LDM for brain decoding.
Shows data loading, model creation, training setup, and inference.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_fmri_data
from brain_ldm import create_brain_ldm


def demo_data_loading():
    """Demo data loading for brain decoding."""
    print("📊 Demo: Data Loading for Brain Decoding")
    print("-" * 50)
    
    # Load fMRI data
    loader = load_fmri_data()
    
    # Get training data
    train_stimuli = loader.get_stimuli('train')  # (90, 784) - target images
    train_fmri = loader.get_fmri('train')        # (90, 3092) - input brain signals
    
    print(f"🧠 fMRI signals: {train_fmri.shape}")
    print(f"🎨 Stimulus images: {train_stimuli.shape}")
    print(f"📋 Task: fMRI ({train_fmri.shape[1]}) → Stimulus ({train_stimuli.shape[1]})")
    
    # Show data statistics
    print(f"\n📈 Data Statistics:")
    print(f"  fMRI range: [{train_fmri.min():.3f}, {train_fmri.max():.3f}]")
    print(f"  Stimulus range: [{train_stimuli.min():.3f}, {train_stimuli.max():.3f}]")
    
    return loader


def demo_model_architecture():
    """Demo model architecture."""
    print("\n🏗️ Demo: Brain LDM Architecture")
    print("-" * 50)
    
    # Create model
    model = create_brain_ldm(fmri_dim=3092, image_size=28)
    
    print(f"🧠 Brain LDM Components:")
    print(f"  1. fMRI Encoder: {sum(p.numel() for p in model.fmri_encoder.parameters()):,} params")
    print(f"  2. VAE: {sum(p.numel() for p in model.vae.parameters()):,} params")
    print(f"  3. U-Net: {sum(p.numel() for p in model.unet.parameters()):,} params")
    print(f"  Total: {sum(p.numel() for p in model.parameters()):,} params")
    
    print(f"\n🔄 Architecture Flow:")
    print(f"  fMRI (3092) → fMRI Encoder → Features (512)")
    print(f"  Stimulus (784) → VAE Encoder → Latents (4×7×7)")
    print(f"  Latents + Features → U-Net → Denoised Latents")
    print(f"  Denoised Latents → VAE Decoder → Reconstructed Stimulus")
    
    return model


def demo_forward_pass():
    """Demo forward pass through the model."""
    print("\n⚡ Demo: Forward Pass")
    print("-" * 50)
    
    # Create model and dummy data
    model = create_brain_ldm()
    batch_size = 4
    
    # Dummy inputs
    fmri_signals = torch.randn(batch_size, 3092)
    stimulus_images = torch.randn(batch_size, 784)
    
    print(f"📥 Input shapes:")
    print(f"  fMRI signals: {fmri_signals.shape}")
    print(f"  Stimulus images: {stimulus_images.shape}")
    
    # Test training step
    batch = {'fmri': fmri_signals, 'stimulus': stimulus_images}
    
    with torch.no_grad():
        # Training forward pass
        output = model.training_step(batch)
        print(f"\n🔄 Training step:")
        print(f"  Loss: {output['loss']:.4f}")
        print(f"  fMRI features norm: {output['fmri_features_norm']:.4f}")
        print(f"  Latents norm: {output['latents_norm']:.4f}")
        
        # Generation forward pass
        generated_images = model.generate_from_fmri(fmri_signals, num_inference_steps=10)
        print(f"\n🎨 Generation:")
        print(f"  Generated images: {generated_images.shape}")
        print(f"  Output range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")


def demo_training_setup():
    """Demo training setup."""
    print("\n🚀 Demo: Training Setup")
    print("-" * 50)
    
    # Load data
    loader = load_fmri_data()
    
    # Create model
    model = create_brain_ldm()
    
    # Create DataLoader
    train_loader = loader.create_dataloader(
        split='train',
        batch_size=4,
        shuffle=True
    )
    
    print(f"📊 Training setup:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Batch size: 4")
    print(f"  Total training samples: {len(train_loader) * 4}")
    
    # Test one training iteration
    batch = next(iter(train_loader))
    print(f"\n🔄 Sample batch:")
    print(f"  fMRI: {batch['fmri'].shape}")
    print(f"  Stimulus: {batch['stimulus'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    
    # Test training step
    with torch.no_grad():
        output = model.training_step(batch)
        print(f"\n📈 Training step result:")
        print(f"  Loss: {output['loss']:.4f}")
    
    print(f"\n💡 To start training:")
    print(f"  python train_brain_ldm.py")


def demo_inference():
    """Demo inference without trained model."""
    print("\n🔮 Demo: Inference (Untrained Model)")
    print("-" * 50)
    
    # Load data
    loader = load_fmri_data()
    
    # Create model
    model = create_brain_ldm()
    
    # Get test data
    test_data = loader.get_test_data()
    fmri_signals = test_data['fmri'][:4]  # First 4 test samples
    true_stimuli = test_data['stimulus'][:4]
    
    print(f"🧪 Test data:")
    print(f"  fMRI signals: {fmri_signals.shape}")
    print(f"  True stimuli: {true_stimuli.shape}")
    
    # Generate reconstructions (untrained model)
    with torch.no_grad():
        generated_images = model.generate_from_fmri(
            fmri_signals, 
            num_inference_steps=20
        )
    
    print(f"\n🎨 Generated reconstructions:")
    print(f"  Shape: {generated_images.shape}")
    print(f"  Range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        # True stimulus
        true_img = true_stimuli[i].view(28, 28).numpy()
        axes[0, i].imshow(true_img, cmap='gray')
        axes[0, i].set_title(f'True {i}')
        axes[0, i].axis('off')
        
        # Generated stimulus (untrained)
        gen_img = generated_images[i, 0].numpy()
        axes[1, i].imshow(gen_img, cmap='gray')
        axes[1, i].set_title(f'Generated {i}')
        axes[1, i].axis('off')
    
    plt.suptitle('Brain Decoding Demo: True vs Generated (Untrained Model)', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_brain_decoding.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Saved demo visualization to: demo_brain_decoding.png")
    print(f"⚠️ Note: Model is untrained, so reconstructions are random!")


def demo_complete_pipeline():
    """Demo complete pipeline overview."""
    print("\n🔄 Demo: Complete Pipeline Overview")
    print("-" * 50)
    
    print(f"🧠 Brain Decoding LDM Pipeline:")
    print(f"")
    print(f"1. 📊 Data Loading:")
    print(f"   - Load fMRI signals (3092 voxels)")
    print(f"   - Load stimulus images (28×28 pixels)")
    print(f"   - Create train/test splits")
    print(f"")
    print(f"2. 🏗️ Model Architecture:")
    print(f"   - fMRI Encoder: Brain signals → Condition features")
    print(f"   - VAE: Images ↔ Latent space")
    print(f"   - U-Net: Conditional diffusion in latent space")
    print(f"   - Scheduler: Controls diffusion process")
    print(f"")
    print(f"3. 🚀 Training:")
    print(f"   - Forward diffusion: Add noise to latents")
    print(f"   - Condition on fMRI features")
    print(f"   - Train U-Net to predict noise")
    print(f"   - Optimize with MSE loss")
    print(f"")
    print(f"4. 🔮 Inference:")
    print(f"   - Start with random noise")
    print(f"   - Iteratively denoise with fMRI conditioning")
    print(f"   - Decode final latents to images")
    print(f"")
    print(f"5. 📊 Evaluation:")
    print(f"   - Compare reconstructed vs true stimuli")
    print(f"   - Compute MSE, correlation, SSIM")
    print(f"   - Visualize results")


def main():
    """Main demo function."""
    print("🎯 Brain Decoding LDM Complete Demo")
    print("=" * 60)
    print("Demonstrating Latent Diffusion Model for Brain Decoding")
    
    try:
        # Run all demos
        loader = demo_data_loading()
        model = demo_model_architecture()
        demo_forward_pass()
        demo_training_setup()
        demo_inference()
        demo_complete_pipeline()
        
        print(f"\n🎉 Demo Completed Successfully!")
        print("=" * 40)
        print(f"✅ Data loader working")
        print(f"✅ Model architecture ready")
        print(f"✅ Training pipeline set up")
        print(f"✅ Inference pipeline working")
        print(f"✅ Visualization created")
        
        print(f"\n🚀 Next Steps:")
        print(f"  1. Train the model: python train_brain_ldm.py")
        print(f"  2. Evaluate results: python evaluate_brain_ldm.py")
        print(f"  3. Experiment with hyperparameters")
        print(f"  4. Try different architectures")
        
        print(f"\n📁 Files created:")
        print(f"  - demo_brain_decoding.png")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
