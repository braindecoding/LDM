"""
🚀 Train Multi-Modal Brain LDM
Training script for Brain-Streams inspired multi-modal LDM with text and semantic guidance.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import load_fmri_data
from multimodal_brain_ldm import MultiModalBrainLDM, create_digit_captions, tokenize_captions
from torch.utils.tensorboard import SummaryWriter
import time

def improved_fmri_normalization(fmri_data):
    """Improved fMRI normalization with outlier handling."""
    median = torch.median(fmri_data, dim=0, keepdim=True)[0]
    mad = torch.median(torch.abs(fmri_data - median), dim=0, keepdim=True)[0]
    mad = torch.where(mad == 0, torch.ones_like(mad), mad)
    normalized = (fmri_data - median) / (1.4826 * mad)
    normalized = torch.clamp(normalized, -3, 3)
    return normalized

def create_multimodal_dataloader(loader, batch_size=4, augment=True):
    """Create dataloader with multi-modal features."""
    train_data = loader.get_train_data()
    
    # Improved normalization
    train_fmri = improved_fmri_normalization(train_data['fmri'])
    train_stimuli = train_data['stimuli']
    train_labels = train_data['labels']
    
    # Data augmentation
    if augment:
        # Augment fMRI signals
        noise_levels = [0.02, 0.05, 0.08]
        augmented_fmri = [train_fmri]
        augmented_stimuli = [train_stimuli]
        augmented_labels = [train_labels]
        
        for noise_level in noise_levels:
            noise = torch.randn_like(train_fmri) * noise_level
            aug_fmri = train_fmri + noise
            augmented_fmri.append(aug_fmri)
            augmented_stimuli.append(train_stimuli)
            augmented_labels.append(train_labels)
        
        # Combine all data
        combined_fmri = torch.cat(augmented_fmri, dim=0)
        combined_stimuli = torch.cat(augmented_stimuli, dim=0)
        combined_labels = torch.cat(augmented_labels, dim=0)
    else:
        combined_fmri = train_fmri
        combined_stimuli = train_stimuli
        combined_labels = train_labels
    
    # Create text captions
    all_captions = []
    for label in combined_labels:
        captions = create_digit_captions([label])
        all_captions.extend(captions)
    
    # Tokenize captions
    text_tokens = tokenize_captions(all_captions)
    
    print(f"📊 Multi-modal dataset created:")
    print(f"   Total samples: {len(combined_fmri)}")
    print(f"   fMRI shape: {combined_fmri.shape}")
    print(f"   Text tokens shape: {text_tokens.shape}")
    print(f"   Labels shape: {combined_labels.shape}")
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        combined_fmri, combined_stimuli, combined_labels, text_tokens
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    return dataloader

def train_multimodal_model(epochs=80, batch_size=4, learning_rate=1e-4, 
                          guidance_scale=7.5, save_dir="checkpoints"):
    """Train multi-modal Brain LDM."""
    print("🚀 Training Multi-Modal Brain LDM")
    print("=" * 50)
    
    device = 'cpu'  # Use CPU for compatibility
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter('runs/multimodal_brain_ldm')
    
    # Load data
    loader = load_fmri_data()
    train_dataloader = create_multimodal_dataloader(
        loader, batch_size=batch_size, augment=True
    )
    
    # Create model
    model = MultiModalBrainLDM(
        fmri_dim=3092,
        image_size=28,
        guidance_scale=guidance_scale
    )
    model.to(device)
    
    # Optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': model.fmri_encoder.parameters(), 'lr': learning_rate},
        {'params': model.text_encoder.parameters(), 'lr': learning_rate * 0.1},  # Lower LR for text
        {'params': model.semantic_embedding.parameters(), 'lr': learning_rate},
        {'params': model.cross_modal_attention.parameters(), 'lr': learning_rate},
        {'params': model.unet.parameters(), 'lr': learning_rate},
        {'params': model.vae_encoder.parameters(), 'lr': learning_rate * 0.5},
        {'params': model.vae_decoder.parameters(), 'lr': learning_rate * 0.5},
    ], weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    losses = []
    
    print(f"🔄 Training for {epochs} epochs...")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_metrics = {
            'diffusion_loss': [],
            'recon_loss': [],
            'semantic_loss': []
        }
        
        for batch_idx, (fmri, stimuli, labels, text_tokens) in enumerate(train_dataloader):
            fmri = fmri.to(device)
            stimuli = stimuli.to(device)
            labels = labels.to(device)
            text_tokens = text_tokens.to(device)
            
            # Forward pass with multi-modal loss
            loss_dict = model.compute_multimodal_loss(
                fmri, stimuli, text_tokens, labels
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record losses
            epoch_losses.append(total_loss.item())
            epoch_metrics['diffusion_loss'].append(loss_dict['diffusion_loss'].item())
            epoch_metrics['recon_loss'].append(loss_dict['recon_loss'].item())
            epoch_metrics['semantic_loss'].append(loss_dict['semantic_loss'].item())
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        avg_diffusion = np.mean(epoch_metrics['diffusion_loss'])
        avg_recon = np.mean(epoch_metrics['recon_loss'])
        avg_semantic = np.mean(epoch_metrics['semantic_loss'])
        
        losses.append(avg_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Total', avg_loss, epoch)
        writer.add_scalar('Loss/Diffusion', avg_diffusion, epoch)
        writer.add_scalar('Loss/Reconstruction', avg_recon, epoch)
        writer.add_scalar('Loss/Semantic', avg_semantic, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss = {avg_loss:.6f} "
                  f"(D: {avg_diffusion:.4f}, R: {avg_recon:.4f}, S: {avg_semantic:.4f}) "
                  f"LR = {scheduler.get_last_lr()[0]:.2e} "
                  f"Time: {elapsed:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = f"{save_dir}/best_multimodal_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'best_loss': best_loss,
                'losses': losses,
                'config': {
                    'fmri_dim': 3092,
                    'image_size': 28,
                    'guidance_scale': guidance_scale,
                    'learning_rate': learning_rate
                }
            }, best_model_path)
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f"{save_dir}/multimodal_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'losses': losses
            }, checkpoint_path)
            
            # Generate sample reconstructions
            generate_sample_reconstructions(
                model, loader, epoch + 1, 
                save_path=f"{save_dir}/multimodal_samples_epoch_{epoch+1}.png"
            )
    
    # Final save
    final_model_path = f"{save_dir}/final_multimodal_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'loss': losses[-1],
        'best_loss': best_loss,
        'losses': losses,
        'training_time': time.time() - start_time
    }, final_model_path)
    
    writer.close()
    
    print(f"\n✅ Training completed!")
    print(f"   Total time: {time.time() - start_time:.1f}s")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Final loss: {losses[-1]:.6f}")
    print(f"   Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    print(f"💾 Models saved to: {save_dir}/")
    
    return model, losses

def generate_sample_reconstructions(model, loader, epoch, save_path):
    """Generate sample reconstructions during training."""
    model.eval()
    
    with torch.no_grad():
        # Get test data
        test_data = loader.get_test_data()
        test_fmri = improved_fmri_normalization(test_data['fmri'][:5])
        test_stimuli = test_data['stimuli'][:5]
        test_labels = test_data['labels'][:5]
        
        # Create text tokens
        captions = create_digit_captions(test_labels)
        text_tokens = tokenize_captions(captions)
        
        # Generate reconstructions with different guidance
        recons_no_guidance, _ = model.generate_with_guidance(
            test_fmri, guidance_scale=1.0
        )
        recons_text_guidance, _ = model.generate_with_guidance(
            test_fmri, text_tokens=text_tokens, guidance_scale=7.5
        )
        recons_full_guidance, attention = model.generate_with_guidance(
            test_fmri, text_tokens=text_tokens, class_labels=test_labels, guidance_scale=7.5
        )
        
        # Create visualization
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        
        for i in range(5):
            # Original
            axes[0, i].imshow(test_stimuli[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Original {test_labels[i].item()}')
            axes[0, i].axis('off')
            
            # No guidance
            axes[1, i].imshow(recons_no_guidance[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title('No Guidance')
            axes[1, i].axis('off')
            
            # Text guidance
            axes[2, i].imshow(recons_text_guidance[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title('Text Guidance')
            axes[2, i].axis('off')
            
            # Full guidance
            axes[3, i].imshow(recons_full_guidance[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            axes[3, i].set_title('Full Guidance')
            axes[3, i].axis('off')
        
        plt.suptitle(f'Multi-Modal Reconstructions - Epoch {epoch}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    model.train()

def main():
    """Main training function."""
    print("🧠 Multi-Modal Brain LDM Training")
    print("=" * 50)
    
    # Training configuration
    config = {
        'epochs': 60,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'guidance_scale': 7.5
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Train model
    model, losses = train_multimodal_model(**config)
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Multi-Modal Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses[10:])  # Skip first 10 epochs for better scale
    plt.title('Training Loss (After Epoch 10)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = "results/multimodal_training_progress.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved training progress to: {output_path}")
    
    plt.show()
    
    print(f"\n🎉 Multi-Modal Brain LDM training complete!")
    print(f"📁 Next: Run evaluation with 'python evaluate_multimodal_ldm.py'")

if __name__ == "__main__":
    main()
