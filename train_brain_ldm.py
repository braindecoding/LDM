"""
🚀 Training Script for Brain Decoding LDM

Clean training script that integrates data loader with LDM model.
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import load_fmri_data
from brain_ldm import create_brain_ldm


class BrainLDMTrainer:
    """Clean trainer for Brain LDM."""
    
    def __init__(self, 
                 model_config: dict = None,
                 training_config: dict = None,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration  
            device: Device for training
        """
        self.device = device
        
        # Default configs
        self.model_config = model_config or {
            'fmri_dim': 3092,
            'image_size': 28
        }
        
        self.training_config = training_config or {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'num_epochs': 100,
            'save_every': 10,
            'log_every': 10
        }
        
        # Initialize model
        self.model = create_brain_ldm(**self.model_config).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.training_config['learning_rate'],
            weight_decay=1e-2
        )
        
        # Initialize data loader
        self.data_loader = load_fmri_data(device=device)
        
        # Create save directory
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter("logs/brain_ldm")
        
        print(f"🧠 Brain LDM Trainer initialized")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"💾 Checkpoints will be saved to: {self.save_dir}")
    
    def create_dataloaders(self):
        """Create train and test dataloaders."""
        train_loader = self.data_loader.create_dataloader(
            split='train',
            batch_size=self.training_config['batch_size'],
            shuffle=True
        )
        
        test_loader = self.data_loader.create_dataloader(
            split='test',
            batch_size=self.training_config['batch_size'],
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model.training_step(batch)
            loss = output['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
            
            # Log to tensorboard
            global_step = epoch * num_batches + batch_idx
            if batch_idx % self.training_config['log_every'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/fmri_features_norm', 
                                     output['fmri_features_norm'].item(), global_step)
                self.writer.add_scalar('train/latents_norm', 
                                     output['latents_norm'].item(), global_step)
        
        return total_loss / num_batches
    
    def validate(self, test_loader, epoch):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                output = self.model.training_step(batch)
                total_loss += output['loss'].item()
        
        avg_loss = total_loss / len(test_loader)
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def generate_samples(self, test_loader, epoch, num_samples=4):
        """Generate sample reconstructions."""
        self.model.eval()
        
        # Get a batch of test data
        test_batch = next(iter(test_loader))
        fmri_signals = test_batch['fmri'][:num_samples].to(self.device)
        true_stimuli = test_batch['stimulus'][:num_samples].to(self.device)
        
        # Generate reconstructions
        with torch.no_grad():
            generated_images = self.model.generate_from_fmri(
                fmri_signals, 
                num_inference_steps=20
            )
        
        # Create visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))
        
        for i in range(num_samples):
            # True stimulus
            true_img = true_stimuli[i].view(28, 28).cpu().numpy()
            axes[0, i].imshow(true_img, cmap='gray')
            axes[0, i].set_title(f'True {i}')
            axes[0, i].axis('off')
            
            # Generated stimulus
            gen_img = generated_images[i, 0].cpu().numpy()
            axes[1, i].imshow(gen_img, cmap='gray')
            axes[1, i].set_title(f'Generated {i}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Epoch {epoch}: True vs Generated Stimuli')
        plt.tight_layout()
        
        # Save and log
        save_path = f"{self.save_dir}/samples_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved samples to: {save_path}")
    
    def save_checkpoint(self, epoch, train_loss, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        save_path = f"{self.save_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        print(f"💾 Saved checkpoint to: {save_path}")
    
    def train(self):
        """Main training loop."""
        print(f"\n🚀 Starting training...")
        
        # Create dataloaders
        train_loader, test_loader = self.create_dataloaders()
        print(f"📊 Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.training_config['num_epochs'] + 1):
            print(f"\n📈 Epoch {epoch}/{self.training_config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(test_loader, epoch)
            
            print(f"📊 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Generate samples
            if epoch % self.training_config['save_every'] == 0:
                self.generate_samples(test_loader, epoch)
                self.save_checkpoint(epoch, train_loss, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'model_config': self.model_config,
                    'training_config': self.training_config
                }
                torch.save(best_checkpoint, f"{self.save_dir}/best_model.pt")
                print(f"💫 New best model saved! Val Loss: {val_loss:.4f}")
        
        print(f"\n✅ Training completed!")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")
        
        # Close tensorboard writer
        self.writer.close()


def main():
    """Main training function."""
    print("🧠 Brain Decoding LDM Training")
    print("=" * 50)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Training configuration
    model_config = {
        'fmri_dim': 3092,
        'image_size': 28
    }
    
    training_config = {
        'learning_rate': 1e-4,
        'batch_size': 4,  # Small batch size for demo
        'num_epochs': 50,
        'save_every': 10,
        'log_every': 5
    }
    
    # Create trainer
    trainer = BrainLDMTrainer(
        model_config=model_config,
        training_config=training_config,
        device=device
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
