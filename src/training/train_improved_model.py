"""
🚀 Train Improved Brain LDM with Uncertainty Calibration
Training script for the enhanced model based on uncertainty analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('..')
from data.data_loader import load_fmri_data
from models.improved_brain_ldm import ImprovedBrainLDM, create_digit_captions, tokenize_captions
import time
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seed set to: {seed}")

def improved_fmri_normalization(fmri_data):
    """Improved fMRI normalization with outlier handling."""
    median = torch.median(fmri_data, dim=0, keepdim=True)[0]
    mad = torch.median(torch.abs(fmri_data - median), dim=0, keepdim=True)[0]
    mad = torch.where(mad == 0, torch.ones_like(mad), mad)
    normalized = (fmri_data - median) / (1.4826 * mad)
    normalized = torch.clamp(normalized, -3, 3)
    return normalized

def create_enhanced_dataloader_v2(loader, batch_size=4, augment_factor=10):
    """Create enhanced dataloader with improved augmentation strategies."""
    train_data = loader.get_train_data()
    
    # Improved normalization
    train_fmri = improved_fmri_normalization(train_data['fmri'])
    train_stimuli = train_data['stimuli']
    train_labels = train_data['labels']
    
    print(f"📊 Original data: {len(train_fmri)} samples")
    
    # Advanced augmentation strategies
    augmented_fmri = [train_fmri]
    augmented_stimuli = [train_stimuli]
    augmented_labels = [train_labels]
    
    for i in range(augment_factor - 1):
        # Strategy 1: Progressive noise levels
        noise_level = 0.01 + (i * 0.02)  # 0.01 to 0.19
        fmri_noise = torch.randn_like(train_fmri) * noise_level
        aug_fmri = train_fmri + fmri_noise
        
        # Strategy 2: Feature dropout (randomly zero out some fMRI features)
        if i % 3 == 1:
            dropout_rate = 0.02 + (i * 0.01)  # 2% to 11%
            dropout_mask = torch.rand_like(train_fmri) > dropout_rate
            aug_fmri = aug_fmri * dropout_mask
        
        # Strategy 3: Smooth perturbations
        if i % 3 == 2:
            smooth_noise = torch.randn_like(train_fmri) * 0.005
            aug_fmri = aug_fmri + smooth_noise
        
        # Strategy 4: Signal scaling
        if i % 4 == 3:
            scale_factor = 0.9 + (torch.rand(1) * 0.2)  # 0.9 to 1.1
            aug_fmri = aug_fmri * scale_factor
        
        # Add slight stimulus noise for robustness
        stim_noise = torch.randn_like(train_stimuli) * 0.01
        aug_stimuli = torch.clamp(train_stimuli + stim_noise, 0, 1)
        
        augmented_fmri.append(aug_fmri)
        augmented_stimuli.append(aug_stimuli)
        augmented_labels.append(train_labels)
    
    # Combine all augmented data
    combined_fmri = torch.cat(augmented_fmri, dim=0)
    combined_stimuli = torch.cat(augmented_stimuli, dim=0)
    combined_labels = torch.cat(augmented_labels, dim=0)
    
    # Create diverse captions
    caption_templates = [
        "handwritten digit {}",
        "digit {} image", 
        "number {} handwriting",
        "written digit {}",
        "digit {} pattern",
        "numeral {}",
        "figure {}",
        "{} symbol",
        "the digit {}",
        "number {} written by hand"
    ]
    
    all_captions = []
    for i, label in enumerate(combined_labels):
        template_idx = i % len(caption_templates)
        digit_name = ['zero', 'one', 'two', 'three', 'four', 
                     'five', 'six', 'seven', 'eight', 'nine'][label.item()]
        caption = caption_templates[template_idx].format(digit_name)
        all_captions.append(caption)
    
    # Tokenize captions
    text_tokens = tokenize_captions(all_captions)
    
    print(f"📊 Enhanced dataset: {len(combined_fmri)} samples ({augment_factor}x augmentation)")
    print(f"   Caption templates: {len(caption_templates)}")
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        combined_fmri, combined_stimuli, combined_labels, text_tokens
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    return dataloader

def train_improved_model(epochs=150, batch_size=4, learning_rate=8e-5,
                        save_name="improved_v1"):
    """Train the improved model with enhanced techniques."""
    print(f"🚀 Training Improved Brain LDM")
    print("=" * 40)

    # Set random seed for reproducibility
    set_seed(42)

    device = 'cpu'
    
    # Load data
    loader = load_fmri_data()
    dataloader = create_enhanced_dataloader_v2(
        loader, batch_size=batch_size, augment_factor=10
    )
    
    # Create improved model
    model = ImprovedBrainLDM(
        fmri_dim=3092,
        image_size=28,
        guidance_scale=7.5
    )
    model.to(device)
    model.train()
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Enhanced optimizer with different learning rates for different components
    param_groups = [
        {'params': model.fmri_encoder.parameters(), 'lr': learning_rate},
        {'params': model.text_encoder.parameters(), 'lr': learning_rate * 0.5},
        {'params': model.semantic_embedding.parameters(), 'lr': learning_rate},
        {'params': model.cross_modal_attention.parameters(), 'lr': learning_rate * 1.2},
        {'params': model.vae_encoder.parameters(), 'lr': learning_rate * 0.8},
        {'params': model.vae_decoder.parameters(), 'lr': learning_rate * 0.8},
        {'params': model.unet.parameters(), 'lr': learning_rate},
        {'params': [model.temperature], 'lr': learning_rate * 0.1}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=5e-6)
    
    # Enhanced learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )
    
    # Training loop
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"🔄 Training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_metrics = {
            'recon_loss': [],
            'perceptual_loss': [],
            'uncertainty_reg': []
        }
        
        for batch_idx, (fmri, stimuli, labels, text_tokens) in enumerate(dataloader):
            fmri = fmri.to(device)
            stimuli = stimuli.to(device)
            labels = labels.to(device)
            text_tokens = text_tokens.to(device)
            
            # Forward pass with improved loss
            loss_dict = model.compute_improved_loss(
                fmri, stimuli, text_tokens, labels
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record metrics
            epoch_losses.append(total_loss.item())
            epoch_metrics['recon_loss'].append(loss_dict['recon_loss'].item())
            epoch_metrics['perceptual_loss'].append(loss_dict['perceptual_loss'].item())
            epoch_metrics['uncertainty_reg'].append(loss_dict['uncertainty_reg'].item())
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            save_path = f"checkpoints/best_{save_name}_model.pt"
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'best_loss': best_loss,
                'losses': losses
            }, save_path)
        else:
            patience_counter += 1
        
        # Progress report
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_recon = np.mean(epoch_metrics['recon_loss'])
            avg_perceptual = np.mean(epoch_metrics['perceptual_loss'])
            avg_uncertainty = np.mean(epoch_metrics['uncertainty_reg'])
            
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss = {avg_loss:.6f} "
                  f"(R: {avg_recon:.4f}, P: {avg_perceptual:.4f}, U: {avg_uncertainty:.4f}) "
                  f"LR = {scheduler.get_last_lr()[0]:.2e} "
                  f"Temp = {model.temperature.item():.3f} "
                  f"Time: {elapsed:.1f}s")
        
        # Early stopping with patience
        if patience_counter >= 25:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"✅ Training completed! Best loss: {best_loss:.6f}")
    
    # Create training progress visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(losses, linewidth=2)
    plt.title('Training Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot([np.mean(epoch_metrics['recon_loss']) for _ in range(len(losses))], 
             label='Reconstruction', linewidth=2)
    plt.plot([np.mean(epoch_metrics['perceptual_loss']) for _ in range(len(losses))], 
             label='Perceptual', linewidth=2)
    plt.title('Loss Components', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    improvement = ((losses[0] - best_loss) / losses[0] * 100)
    plt.bar(['Initial', 'Best'], [losses[0], best_loss], 
            color=['red', 'green'], alpha=0.7)
    plt.title(f'Improvement: {improvement:.1f}%', fontweight='bold')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f'Training Summary:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f'Epochs: {len(losses)}', fontsize=12)
    plt.text(0.1, 0.6, f'Best Loss: {best_loss:.6f}', fontsize=12)
    plt.text(0.1, 0.5, f'Improvement: {improvement:.1f}%', fontsize=12)
    plt.text(0.1, 0.4, f'Final Temperature: {model.temperature.item():.3f}', fontsize=12)
    plt.text(0.1, 0.3, f'Parameters: {sum(p.numel() for p in model.parameters()):,}', fontsize=12)
    plt.axis('off')
    
    plt.suptitle(f'Improved Brain LDM Training Results ({save_name})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save training progress
    progress_path = f"results/{save_name}_training_progress.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(progress_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved training progress to: {progress_path}")
    
    plt.show()
    
    return model, losses, best_loss

def main():
    """Main training function."""
    print("🚀 Improved Brain LDM Training with Uncertainty Calibration")
    print("=" * 65)
    
    # Train improved model
    model, losses, best_loss = train_improved_model(
        epochs=150,
        batch_size=4,
        learning_rate=8e-5,
        save_name="improved_v1"
    )
    
    print(f"\n🎉 Improved Model Training Complete!")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"📈 Expected improvements:")
    print(f"   • Better uncertainty calibration")
    print(f"   • Improved reconstruction quality")
    print(f"   • Enhanced guidance effects")
    print(f"   • More reliable predictions")
    print(f"")
    print(f"📁 Model saved: checkpoints/best_improved_v1_model.pt")
    print(f"🔬 Next: Run uncertainty evaluation on improved model")

if __name__ == "__main__":
    main()
