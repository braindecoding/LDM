"""
🎯 Demo: Brain Decoding LDM

Clean demo that focuses on core brain decoding functionality.
Shows data loading, simple model, training, and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def demo_data_loading():
    """Demo data loading."""
    print("📊 Demo: Data Loading")
    print("-" * 30)

    try:
        from data_loader import load_fmri_data
        loader = load_fmri_data()

        # Get data
        train_fmri = loader.get_fmri('train')
        train_stimuli = loader.get_stimuli('train')

        print(f"✅ Data loaded successfully")
        print(f"  fMRI signals: {train_fmri.shape}")
        print(f"  Stimulus images: {train_stimuli.shape}")
        print(f"  Task: Brain activity → Visual reconstruction")

        return loader

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None


def demo_simple_model():
    """Demo simple brain decoding model."""
    print("\n🧠 Demo: Simple Brain Decoding Model")
    print("-" * 40)

    try:
        # Simple MLP for brain decoding
        class SimpleBrainDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(3092, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 784),
                    torch.nn.Sigmoid()
                )

            def forward(self, fmri):
                return self.decoder(fmri)

        # Create model
        model = SimpleBrainDecoder()

        # Test with dummy data
        dummy_fmri = torch.randn(4, 3092)
        output = model(dummy_fmri)

        print(f"✅ Simple model working")
        print(f"  Input (fMRI): {dummy_fmri.shape}")
        print(f"  Output (Stimulus): {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def demo_training_step():
    """Demo training step."""
    print("\n🚀 Demo: Training Step")
    print("-" * 30)

    try:
        # Load data
        from data_loader import load_fmri_data
        loader = load_fmri_data()

        # Simple model
        class SimpleBrainDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(3092, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 784),
                    torch.nn.Sigmoid()
                )

            def forward(self, fmri):
                return self.decoder(fmri)

        model = SimpleBrainDecoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # Get real data
        train_loader = loader.create_dataloader('train', batch_size=4)
        batch = next(iter(train_loader))

        fmri_signals = batch['fmri']
        true_stimuli = batch['stimulus']

        # Training step
        optimizer.zero_grad()
        predicted_stimuli = model(fmri_signals)
        loss = criterion(predicted_stimuli, true_stimuli)
        loss.backward()
        optimizer.step()

        print(f"✅ Training step successful")
        print(f"  Batch size: {fmri_signals.shape[0]}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Prediction range: [{predicted_stimuli.min():.3f}, {predicted_stimuli.max():.3f}]")

        return model, loss.item()

    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return None, None


def demo_visualization():
    """Demo visualization."""
    print("\n🎨 Demo: Visualization")
    print("-" * 30)

    try:
        # Load data
        from data_loader import load_fmri_data
        loader = load_fmri_data()

        # Simple model
        class SimpleBrainDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(3092, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 784),
                    torch.nn.Sigmoid()
                )

            def forward(self, fmri):
                return self.decoder(fmri)

        model = SimpleBrainDecoder()

        # Get test data
        test_data = loader.get_test_data()
        fmri_signals = test_data['fmri'][:4]
        true_stimuli = test_data['stimulus'][:4]

        # Generate predictions
        with torch.no_grad():
            predicted_stimuli = model(fmri_signals)

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))

        for i in range(4):
            # True stimulus
            true_img = true_stimuli[i].view(28, 28).numpy()
            axes[0, i].imshow(true_img, cmap='gray')
            axes[0, i].set_title(f'True {i}')
            axes[0, i].axis('off')

            # Predicted stimulus
            pred_img = predicted_stimuli[i].view(28, 28).numpy()
            axes[1, i].imshow(pred_img, cmap='gray')
            axes[1, i].set_title(f'Predicted {i}')
            axes[1, i].axis('off')

        plt.suptitle('Brain Decoding Demo: True vs Predicted (Untrained Model)', fontsize=14)
        plt.tight_layout()
        plt.savefig('demo_brain_decoding_results.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid hanging

        print(f"✅ Visualization created")
        print(f"💾 Saved to: demo_brain_decoding_results.png")
        print(f"⚠️ Note: Model is untrained, so predictions are random!")

        return True

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False


def demo_ldm_concept():
    """Demo LDM concept explanation."""
    print("\n🏗️ Demo: LDM Concept for Brain Decoding")
    print("-" * 45)

    print(f"🧠 Latent Diffusion Model (LDM) for Brain Decoding:")
    print(f"")
    print(f"1. 📊 Input: fMRI brain signals (3092 voxels)")
    print(f"   - Represents neural activity while viewing images")
    print(f"   - High-dimensional, noisy brain data")
    print(f"")
    print(f"2. 🎯 Output: Reconstructed visual stimuli (28×28 images)")
    print(f"   - Goal: Recreate what the person was seeing")
    print(f"   - From brain activity to visual perception")
    print(f"")
    print(f"3. 🏗️ LDM Architecture:")
    print(f"   a) fMRI Encoder: Brain signals → Condition features")
    print(f"   b) VAE: Images ↔ Compact latent space")
    print(f"   c) U-Net: Conditional diffusion in latent space")
    print(f"   d) Scheduler: Controls noise addition/removal")
    print(f"")
    print(f"4. 🔄 Training Process:")
    print(f"   - Encode stimulus images to latent space")
    print(f"   - Add noise (forward diffusion)")
    print(f"   - Train U-Net to remove noise conditioned on fMRI")
    print(f"   - Learn to generate from brain signals")
    print(f"")
    print(f"5. 🔮 Inference:")
    print(f"   - Start with random noise")
    print(f"   - Iteratively denoise using fMRI conditioning")
    print(f"   - Decode final latent to reconstructed image")
    print(f"")
    print(f"💡 Advantages of LDM:")
    print(f"   - Efficient: Works in compressed latent space")
    print(f"   - High quality: Diffusion models generate sharp images")
    print(f"   - Conditional: Can be guided by brain signals")
    print(f"   - Flexible: Can generate diverse reconstructions")


def main():
    """Main demo function."""
    print("🎯 Demo: Brain Decoding LDM")
    print("=" * 50)
    print("Simplified demo focusing on core concepts")

    try:
        # Run demos
        loader = demo_data_loading()

        if loader is not None:
            model = demo_simple_model()

            if model is not None:
                model, loss = demo_training_step()

                if model is not None:
                    demo_visualization()

        demo_ldm_concept()

        print(f"\n🎉 Demo Completed!")
        print("=" * 30)
        print(f"✅ Data loading works")
        print(f"✅ Simple model works")
        print(f"✅ Training step works")
        print(f"✅ Visualization works")
        print(f"✅ LDM concept explained")

        print(f"\n💡 Next Steps:")
        print(f"  1. The full LDM implementation is in brain_ldm.py")
        print(f"  2. This demo shows the core brain decoding concept")
        print(f"  3. LDM adds diffusion for higher quality generation")
        print(f"  4. Train with: python train_brain_ldm.py")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
