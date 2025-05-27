"""
🧠 Simple fMRI Stimulus Data Loader

Clean and simple data loader for stimulus and fMRI data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
from typing import Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class FMRIDataLoader:
    """Simple data loader for fMRI and stimulus data."""
    
    def __init__(self, 
                 data_path: str = "data/digit69_28x28.mat",
                 device: str = 'cpu',
                 normalize_stimuli: bool = True,
                 normalize_fmri: bool = True):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to .mat file
            device: Device for tensors ('cpu' or 'cuda')
            normalize_stimuli: Whether to normalize stimulus images
            normalize_fmri: Whether to normalize fMRI data
        """
        self.data_path = Path(data_path)
        self.device = device
        self.normalize_stimuli = normalize_stimuli
        self.normalize_fmri = normalize_fmri
        
        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load and process data
        print(f"📁 Loading data from: {self.data_path.name}")
        self.raw_data = self._load_mat_file()
        self.processed_data = self._process_data()
        
        # Display info
        self._print_info()
    
    def _load_mat_file(self) -> Dict[str, np.ndarray]:
        """Load data from .mat file."""
        try:
            data = scipy.io.loadmat(self.data_path)
            
            # Extract relevant data
            relevant_data = {}
            keys = ['fmriTrn', 'stimTrn', 'fmriTest', 'stimTest', 'labelTrn', 'labelTest']
            
            for key in keys:
                if key in data:
                    relevant_data[key] = data[key]
                    print(f"  ✅ {key}: {data[key].shape}")
                else:
                    print(f"  ⚠️ {key}: not found")
            
            return relevant_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load .mat file: {e}")
    
    def _process_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Process and normalize data."""
        processed = {'train': {}, 'test': {}}
        
        # Process training data
        if 'stimTrn' in self.raw_data:
            processed['train']['stimuli'] = self._process_stimuli(self.raw_data['stimTrn'])
        if 'fmriTrn' in self.raw_data:
            processed['train']['fmri'] = self._process_fmri(self.raw_data['fmriTrn'])
        if 'labelTrn' in self.raw_data:
            processed['train']['labels'] = self._process_labels(self.raw_data['labelTrn'])
        
        # Process test data
        if 'stimTest' in self.raw_data:
            processed['test']['stimuli'] = self._process_stimuli(self.raw_data['stimTest'])
        if 'fmriTest' in self.raw_data:
            processed['test']['fmri'] = self._process_fmri(self.raw_data['fmriTest'])
        if 'labelTest' in self.raw_data:
            processed['test']['labels'] = self._process_labels(self.raw_data['labelTest'])
        
        return processed
    
    def _process_stimuli(self, stimuli_data: np.ndarray) -> torch.Tensor:
        """Process stimulus data."""
        tensor_data = torch.FloatTensor(stimuli_data)
        
        if self.normalize_stimuli:
            # Normalize to [0, 1]
            tensor_data = (tensor_data - tensor_data.min()) / (tensor_data.max() - tensor_data.min())
        
        return tensor_data.to(self.device)
    
    def _process_fmri(self, fmri_data: np.ndarray) -> torch.Tensor:
        """Process fMRI data."""
        tensor_data = torch.FloatTensor(fmri_data)
        
        if self.normalize_fmri:
            # Z-score normalization
            mean = tensor_data.mean(dim=0, keepdim=True)
            std = tensor_data.std(dim=0, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            tensor_data = (tensor_data - mean) / std
        
        return tensor_data.to(self.device)
    
    def _process_labels(self, labels_data: np.ndarray) -> torch.Tensor:
        """Process labels."""
        labels_flat = labels_data.flatten() if labels_data.ndim > 1 else labels_data
        return torch.LongTensor(labels_flat).to(self.device)
    
    def _print_info(self):
        """Print data information."""
        print(f"\n📊 Data Information:")
        
        for split in ['train', 'test']:
            if split in self.processed_data:
                data = self.processed_data[split]
                print(f"\n🎯 {split.capitalize()} Data:")
                
                for key, tensor in data.items():
                    print(f"  {key}: {tensor.shape}")
                    if key == 'stimuli':
                        print(f"    Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
                    elif key == 'fmri':
                        print(f"    Mean: {tensor.mean():.3f}, Std: {tensor.std():.3f}")
    
    # Public API methods
    def get_train_data(self) -> Dict[str, torch.Tensor]:
        """Get training data."""
        return self.processed_data['train'].copy()
    
    def get_test_data(self) -> Dict[str, torch.Tensor]:
        """Get test data."""
        return self.processed_data['test'].copy()
    
    def get_stimuli(self, split: str = 'train') -> torch.Tensor:
        """Get stimulus data."""
        return self.processed_data[split]['stimuli']
    
    def get_fmri(self, split: str = 'train') -> torch.Tensor:
        """Get fMRI data."""
        return self.processed_data[split]['fmri']
    
    def get_labels(self, split: str = 'train') -> torch.Tensor:
        """Get labels."""
        return self.processed_data[split]['labels']
    
    def get_stimulus_as_image(self, index: int, split: str = 'train') -> np.ndarray:
        """Get stimulus as 28x28 image."""
        stimuli = self.get_stimuli(split)
        
        if index >= len(stimuli):
            raise IndexError(f"Index {index} out of range for {split}")
        
        # Reshape from 784 to 28x28
        stimulus_flat = stimuli[index].cpu().numpy()
        return stimulus_flat.reshape(28, 28)
    
    def visualize_samples(self, num_samples: int = 10, split: str = 'train', save_path: Optional[str] = None):
        """Visualize stimulus samples."""
        stimuli = self.get_stimuli(split)
        labels = self.get_labels(split) if 'labels' in self.processed_data[split] else None
        
        num_samples = min(num_samples, len(stimuli))
        
        # Setup plot
        cols = 5
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Sample Stimuli ({split.capitalize()} Set)', fontsize=16)
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            
            # Get and display stimulus
            stimulus_image = self.get_stimulus_as_image(i, split)
            axes[row, col].imshow(stimulus_image, cmap='gray')
            
            # Set title
            if labels is not None:
                label = labels[i].item()
                axes[row, col].set_title(f'Sample {i}\nLabel: {label}')
            else:
                axes[row, col].set_title(f'Sample {i}')
            
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved to: {save_path}")
        
        plt.show()
    
    def create_dataloader(self, 
                         split: str = 'train',
                         batch_size: int = 8,
                         shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
        """Create PyTorch DataLoader."""
        data_dict = self.processed_data[split]
        
        dataset = FMRIDataset(
            stimuli=data_dict['stimuli'],
            fmri=data_dict['fmri'],
            labels=data_dict.get('labels', None)
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device != 'cpu' else False
        )


class FMRIDataset(Dataset):
    """PyTorch Dataset for fMRI data."""
    
    def __init__(self, stimuli: torch.Tensor, fmri: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.stimuli = stimuli
        self.fmri = fmri
        self.labels = labels
        
        # Validation
        if len(stimuli) != len(fmri):
            raise ValueError("Stimuli and fMRI must have same length")
        
        if labels is not None and len(labels) != len(stimuli):
            raise ValueError("Labels must have same length as stimuli")
    
    def __len__(self) -> int:
        return len(self.stimuli)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'stimulus': self.stimuli[idx],
            'fmri': self.fmri[idx]
        }
        
        if self.labels is not None:
            sample['label'] = self.labels[idx]
        
        return sample


# Convenience function
def load_fmri_data(data_path: str = "data/digit69_28x28.mat", **kwargs) -> FMRIDataLoader:
    """Load fMRI data with simple interface."""
    return FMRIDataLoader(data_path=data_path, **kwargs)


# Demo
if __name__ == "__main__":
    print("🧠 Demo: Simple fMRI Data Loader")
    print("=" * 40)
    
    # Load data
    loader = load_fmri_data()
    
    # Get data
    train_stimuli = loader.get_stimuli('train')
    train_fmri = loader.get_fmri('train')
    
    print(f"\n📊 Data shapes:")
    print(f"  Stimuli: {train_stimuli.shape}")
    print(f"  fMRI: {train_fmri.shape}")
    
    # Create DataLoader
    train_loader = loader.create_dataloader('train', batch_size=4)
    print(f"\n🔄 DataLoader: {len(train_loader)} batches")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"📦 Batch shapes:")
    for key, tensor in batch.items():
        print(f"  {key}: {tensor.shape}")
    
    print(f"\n✅ Ready for brain decoding!")
