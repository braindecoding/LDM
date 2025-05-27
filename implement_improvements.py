"""
🚀 Phase 1: Quick Wins Implementation
Implement immediate improvements to boost accuracy from 20% to ~35%.

Key improvements:
1. Perceptual loss function (LPIPS-style)
2. Better fMRI normalization
3. Data augmentation
4. Improved training parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from data_loader import load_fmri_data
from brain_ldm import BrainLDM
import matplotlib.pyplot as plt

class PerceptualLoss(nn.Module):
    """Simple perceptual loss using feature differences."""

    def __init__(self):
        super().__init__()
        # Simple feature extractor (can be replaced with VGG features)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), # 14->7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, pred, target):
        # Reshape to image format if needed
        if len(pred.shape) == 2:
            pred = pred.view(-1, 1, 28, 28)
        if len(target.shape) == 2:
            target = target.view(-1, 1, 28, 28)

        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        # Compute perceptual loss
        return F.mse_loss(pred_features, target_features)

class ImprovedBrainLDM(BrainLDM):
    """Improved Brain LDM with better loss and training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add perceptual loss
        self.perceptual_loss = PerceptualLoss()

        # Loss weights
        self.mse_weight = 1.0
        self.perceptual_weight = 0.5

    def compute_improved_loss(self, pred_images, target_images):
        """Compute combined MSE + Perceptual loss."""
        # MSE loss
        mse_loss = F.mse_loss(pred_images, target_images)

        # Perceptual loss
        perceptual_loss = self.perceptual_loss(pred_images, target_images)

        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.perceptual_weight * perceptual_loss

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'perceptual_loss': perceptual_loss
        }

    def improved_training_step(self, batch):
        """Improved training step with better loss."""
        fmri_signals = batch['fmri']
        stimulus_images = batch['stimulus']

        # Encode fMRI to condition features
        fmri_features = self.encode_fmri(fmri_signals)

        # Encode images to latents
        latents = self.encode_images(stimulus_images)

        # Forward diffusion
        diffusion_output = self.forward_diffusion(latents, fmri_features)

        # Generate reconstruction for perceptual loss
        with torch.no_grad():
            reconstructed = self.generate_from_fmri(fmri_signals, num_inference_steps=10)
            reconstructed = reconstructed.view(reconstructed.shape[0], -1)

        # Compute improved loss
        loss_dict = self.compute_improved_loss(reconstructed, stimulus_images)

        # Combine with diffusion loss
        total_loss = diffusion_output['loss'] + 0.1 * loss_dict['total_loss']

        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_output['loss'],
            'mse_loss': loss_dict['mse_loss'],
            'perceptual_loss': loss_dict['perceptual_loss']
        }

def improved_fmri_normalization(fmri_data):
    """Improved fMRI normalization with outlier handling."""
    # Robust z-score normalization
    median = torch.median(fmri_data, dim=0, keepdim=True)[0]
    mad = torch.median(torch.abs(fmri_data - median), dim=0, keepdim=True)[0]

    # Avoid division by zero
    mad = torch.where(mad == 0, torch.ones_like(mad), mad)

    # Robust normalization
    normalized = (fmri_data - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std

    # Clip extreme outliers
    normalized = torch.clamp(normalized, -3, 3)

    return normalized

def augment_fmri_data(fmri_data, noise_level=0.05):
    """Simple data augmentation for fMRI signals."""
    # Add small amount of Gaussian noise
    noise = torch.randn_like(fmri_data) * noise_level
    augmented = fmri_data + noise

    return augmented

def create_improved_dataloader(loader, batch_size=8, augment=True):
    """Create improved dataloader with augmentation."""
    train_data = loader.get_train_data()

    # Improved normalization
    train_fmri = improved_fmri_normalization(train_data['fmri'])
    train_stimuli = train_data['stimuli']

    # Data augmentation
    if augment:
        # Create augmented versions
        aug_fmri = augment_fmri_data(train_fmri)

        # Combine original and augmented
        combined_fmri = torch.cat([train_fmri, aug_fmri], dim=0)
        combined_stimuli = torch.cat([train_stimuli, train_stimuli], dim=0)
    else:
        combined_fmri = train_fmri
        combined_stimuli = train_stimuli

    # Create dataset
    dataset = torch.utils.data.TensorDataset(combined_fmri, combined_stimuli)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"📊 Improved dataset: {len(dataset)} samples (original: {len(train_data['fmri'])})")

    return dataloader

def train_improved_model(epochs=100, save_path="checkpoints/improved_model.pt"):
    """Train improved model with Phase 1 enhancements."""
    print("🚀 Training Improved Brain LDM (Phase 1)")
    print("=" * 50)

    device = 'cpu'  # Use CPU for compatibility

    # Load data
    loader = load_fmri_data()
    train_dataloader = create_improved_dataloader(loader, batch_size=8, augment=True)

    # Create improved model
    model = ImprovedBrainLDM(fmri_dim=3092, image_size=28)
    model.to(device)

    # Improved optimizer with scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    losses = []

    print(f"🔄 Training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, (fmri, stimuli) in enumerate(train_dataloader):
            fmri, stimuli = fmri.to(device), stimuli.to(device)

            # Create batch dict
            batch = {'fmri': fmri, 'stimulus': stimuli}

            # Forward pass with improved training step
            output = model.improved_training_step(batch)
            loss = output['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_losses.append(loss.item())

        # Update learning rate
        scheduler.step()

        # Log progress
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.2e}")

    # Save improved model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
        'loss': losses[-1],
        'losses': losses
    }, save_path)

    print(f"💾 Saved improved model to: {save_path}")

    return model, losses

def evaluate_improved_model(model_path="checkpoints/improved_model.pt"):
    """Evaluate the improved model."""
    print("\n📊 Evaluating Improved Model")
    print("=" * 35)

    device = 'cpu'

    # Load model
    model = ImprovedBrainLDM(fmri_dim=3092, image_size=28)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    loader = load_fmri_data()
    test_data = loader.get_test_data()

    # Improved normalization for test data
    test_fmri = improved_fmri_normalization(test_data['fmri'])
    test_stimuli = test_data['stimuli']

    # Generate reconstructions
    with torch.no_grad():
        reconstructions = model.generate_from_fmri(test_fmri, num_inference_steps=50)
        reconstructions = reconstructions.view(reconstructions.shape[0], -1)

    # Compute metrics
    mse = F.mse_loss(reconstructions, test_stimuli).item()
    mae = F.l1_loss(reconstructions, test_stimuli).item()

    # Compute correlations
    recon_flat = reconstructions.cpu().numpy()
    stimuli_flat = test_stimuli.cpu().numpy()

    correlations = []
    for i in range(len(recon_flat)):
        corr = np.corrcoef(recon_flat[i], stimuli_flat[i])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0)

    avg_correlation = np.mean(correlations)

    # Classification accuracy
    corr_matrix = np.corrcoef(recon_flat, stimuli_flat)[:len(recon_flat), len(recon_flat):]
    correct_matches = sum(np.argmax(corr_matrix[i, :]) == i for i in range(len(corr_matrix)))
    accuracy = correct_matches / len(corr_matrix)

    print(f"📈 Improved Model Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Average Correlation: {avg_correlation:.6f}")
    print(f"  Classification Accuracy: {accuracy:.2%} ({correct_matches}/{len(corr_matrix)})")

    return {
        'mse': mse,
        'mae': mae,
        'correlation': avg_correlation,
        'accuracy': accuracy
    }

def compare_models():
    """Compare original vs improved model."""
    print("\n🔍 Model Comparison")
    print("=" * 25)

    # Original model results (from previous evaluation)
    original_results = {
        'mse': 0.279,
        'mae': 0.517,
        'correlation': 0.0013,
        'accuracy': 0.20
    }

    # Evaluate improved model
    improved_results = evaluate_improved_model()

    print(f"\n📊 Comparison Results:")
    print(f"{'Metric':<20} {'Original':<12} {'Improved':<12} {'Change':<12}")
    print("-" * 60)

    for metric in ['mse', 'mae', 'correlation', 'accuracy']:
        orig = original_results[metric]
        impr = improved_results[metric]

        if metric in ['mse', 'mae']:
            change = f"{((orig - impr) / orig * 100):+.1f}%"  # Lower is better
        else:
            change = f"{((impr - orig) / orig * 100):+.1f}%" if orig > 0 else f"+{impr:.3f}"

        print(f"{metric.upper():<20} {orig:<12.6f} {impr:<12.6f} {change:<12}")

def main():
    """Main implementation function."""
    print("🚀 Phase 1: Quick Wins Implementation")
    print("=" * 50)

    # Step 1: Train improved model
    print("Step 1: Training improved model...")
    model, losses = train_improved_model(epochs=50)  # Reduced for demo

    # Step 2: Evaluate and compare
    print("\nStep 2: Evaluating improvements...")
    compare_models()

    # Step 3: Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Improved Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    output_path = "results/improved_training_progress.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved training progress to: {output_path}")
    plt.show()

    print(f"\n✅ Phase 1 Complete!")
    print(f"📁 Next: Implement Phase 2 (Architecture improvements)")

if __name__ == "__main__":
    main()
