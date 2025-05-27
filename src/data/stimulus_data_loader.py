"""
🎨 Stimulus Data Loader

A clean, well-structured data loader for visual stimulus data from brain decoding experiments.
This loader handles digit stimulus images and their corresponding fMRI brain activity data.

Author: Brain Decoding Team
Date: 2025
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
from typing import Dict, List, Tuple, Optional, Union
import os
from pathlib import Path
import matplotlib.pyplot as plt


class StimulusDataLoader:
    """
    Clean data loader for visual stimulus data and corresponding fMRI data.
    
    This class loads and manages paired stimulus-fMRI data for brain decoding tasks,
    specifically designed for digit recognition experiments.
    """
    
    def __init__(self, 
                 data_path: str,
                 device: str = 'cpu',
                 normalize_stimuli: bool = True,
                 normalize_fmri: bool = True):
        """
        Initialize the stimulus data loader.
        
        Args:
            data_path: Path to the .mat file containing stimulus and fMRI data
            device: Device to load data on ('cpu' or 'cuda')
            normalize_stimuli: Whether to normalize stimulus images
            normalize_fmri: Whether to normalize fMRI data
        """
        self.data_path = Path(data_path)
        self.device = device
        self.normalize_stimuli = normalize_stimuli
        self.normalize_fmri = normalize_fmri
        
        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Stimulus data file not found: {data_path}")
        
        # Load and process data
        self.raw_data = self._load_mat_data()
        self.processed_data = self._process_data()
        
        print(f"✅ Loaded stimulus data from: {self.data_path.name}")
        print(f"🎨 Training stimuli: {self.processed_data['train']['stimuli'].shape}")
        print(f"🧠 Training fMRI: {self.processed_data['train']['fmri'].shape}")
        print(f"🎨 Test stimuli: {self.processed_data['test']['stimuli'].shape}")
        print(f"🧠 Test fMRI: {self.processed_data['test']['fmri'].shape}")
    
    def _load_mat_data(self) -> Dict[str, np.ndarray]:
        """
        Load data from .mat file.
        
        Returns:
            Dictionary containing loaded data arrays
        """
        try:
            data = scipy.io.loadmat(self.data_path)
            
            # Extract relevant data (exclude metadata)
            relevant_keys = ['fmriTrn', 'stimTrn', 'fmriTest', 'stimTest', 
                           'labelTrn', 'labelTest']
            
            extracted_data = {}
            for key in relevant_keys:
                if key in data:
                    extracted_data[key] = data[key]
                else:
                    print(f"⚠️ Warning: Key '{key}' not found in data file")
            
            return extracted_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load stimulus data: {e}")
    
    def _process_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Process and normalize the loaded data.
        
        Returns:
            Dictionary with processed train/test data
        """
        processed = {
            'train': {},
            'test': {}
        }
        
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
        """
        Process stimulus images.
        
        Args:
            stimuli_data: Raw stimulus data (N, 784) for 28x28 images
            
        Returns:
            Processed stimulus tensor
        """
        # Convert to torch tensor
        tensor_data = torch.FloatTensor(stimuli_data)
        
        # Normalize if requested
        if self.normalize_stimuli:
            # Normalize to [0, 1] range
            tensor_data = (tensor_data - tensor_data.min()) / (tensor_data.max() - tensor_data.min())
        
        # Move to device
        tensor_data = tensor_data.to(self.device)
        
        return tensor_data
    
    def _process_fmri(self, fmri_data: np.ndarray) -> torch.Tensor:
        """
        Process fMRI brain activity data.
        
        Args:
            fmri_data: Raw fMRI data (N, voxels)
            
        Returns:
            Processed fMRI tensor
        """
        # Convert to torch tensor
        tensor_data = torch.FloatTensor(fmri_data)
        
        # Normalize if requested
        if self.normalize_fmri:
            tensor_data = self._normalize_tensor(tensor_data)
        
        # Move to device
        tensor_data = tensor_data.to(self.device)
        
        return tensor_data
    
    def _process_labels(self, labels_data: np.ndarray) -> torch.Tensor:
        """
        Process digit labels.
        
        Args:
            labels_data: Raw label data
            
        Returns:
            Processed label tensor
        """
        # Flatten if needed and convert to long tensor for classification
        labels_flat = labels_data.flatten() if labels_data.ndim > 1 else labels_data
        tensor_data = torch.LongTensor(labels_flat)
        
        # Move to device
        tensor_data = tensor_data.to(self.device)
        
        return tensor_data
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor using z-score normalization.
        
        Args:
            tensor: Input tensor to normalize
            
        Returns:
            Normalized tensor
        """
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        # Avoid division by zero
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (tensor - mean) / std
    
    def get_train_data(self) -> Dict[str, torch.Tensor]:
        """
        Get training data.
        
        Returns:
            Dictionary with training stimuli, fMRI, and labels
        """
        return self.processed_data['train'].copy()
    
    def get_test_data(self) -> Dict[str, torch.Tensor]:
        """
        Get test data.
        
        Returns:
            Dictionary with test stimuli, fMRI, and labels
        """
        return self.processed_data['test'].copy()
    
    def get_stimuli_only(self, split: str = 'train') -> torch.Tensor:
        """
        Get only stimulus data for specified split.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Stimulus tensor
        """
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")
        
        return self.processed_data[split]['stimuli']
    
    def get_fmri_only(self, split: str = 'train') -> torch.Tensor:
        """
        Get only fMRI data for specified split.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            fMRI tensor
        """
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")
        
        return self.processed_data[split]['fmri']
    
    def get_stimulus_as_image(self, index: int, split: str = 'train') -> np.ndarray:
        """
        Get a stimulus as a 28x28 image array.
        
        Args:
            index: Index of the stimulus
            split: 'train' or 'test'
            
        Returns:
            28x28 image array
        """
        stimuli = self.get_stimuli_only(split)
        
        if index >= len(stimuli):
            raise IndexError(f"Index {index} out of range for {split} data")
        
        # Reshape from 784 to 28x28
        stimulus_flat = stimuli[index].cpu().numpy()
        return stimulus_flat.reshape(28, 28)
    
    def visualize_stimulus(self, index: int, split: str = 'train', save_path: Optional[str] = None):
        """
        Visualize a stimulus image.
        
        Args:
            index: Index of the stimulus to visualize
            split: 'train' or 'test'
            save_path: Optional path to save the visualization
        """
        image = self.get_stimulus_as_image(index, split)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f'Stimulus {index} ({split} set)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"💾 Saved visualization to: {save_path}")
        
        plt.show()
    
    def get_data_info(self) -> Dict[str, Union[int, Tuple[int, ...]]]:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        train_data = self.processed_data['train']
        test_data = self.processed_data['test']
        
        info = {
            'train_samples': train_data['stimuli'].shape[0] if 'stimuli' in train_data else 0,
            'test_samples': test_data['stimuli'].shape[0] if 'stimuli' in test_data else 0,
            'stimulus_dimensions': 784,  # 28x28 flattened
            'stimulus_image_size': (28, 28),
            'fmri_voxels': train_data['fmri'].shape[1] if 'fmri' in train_data else 0,
            'device': str(self.device),
            'stimuli_normalized': self.normalize_stimuli,
            'fmri_normalized': self.normalize_fmri
        }
        
        return info
    
    def create_paired_dataloader(self, 
                               split: str = 'train',
                               batch_size: int = 8,
                               shuffle: bool = True,
                               num_workers: int = 0) -> DataLoader:
        """
        Create a DataLoader for paired stimulus-fMRI data.
        
        Args:
            split: 'train' or 'test'
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")
        
        data_dict = self.processed_data[split]
        dataset = StimulusFMRIDataset(
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


class StimulusFMRIDataset(Dataset):
    """
    PyTorch Dataset for paired stimulus-fMRI data.
    """
    
    def __init__(self, 
                 stimuli: torch.Tensor, 
                 fmri: torch.Tensor,
                 labels: Optional[torch.Tensor] = None):
        """
        Initialize the dataset.
        
        Args:
            stimuli: Stimulus images tensor
            fmri: fMRI data tensor
            labels: Optional labels tensor
        """
        self.stimuli = stimuli
        self.fmri = fmri
        self.labels = labels
        
        # Validate data consistency
        if len(stimuli) != len(fmri):
            raise ValueError("Stimuli and fMRI data must have same number of samples")
        
        if labels is not None and len(labels) != len(stimuli):
            raise ValueError("Labels must have same number of samples as stimuli")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.stimuli)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with stimulus, fMRI, and optionally label data
        """
        sample = {
            'stimulus': self.stimuli[idx],
            'fmri': self.fmri[idx]
        }
        
        if self.labels is not None:
            sample['label'] = self.labels[idx]
        
        return sample


def load_stimulus_data(data_path: str, 
                      device: str = 'cpu',
                      normalize_stimuli: bool = True,
                      normalize_fmri: bool = True) -> StimulusDataLoader:
    """
    Convenience function to load stimulus data.
    
    Args:
        data_path: Path to stimulus .mat file
        device: Device to load data on
        normalize_stimuli: Whether to normalize stimulus images
        normalize_fmri: Whether to normalize fMRI data
        
    Returns:
        Initialized StimulusDataLoader
    """
    return StimulusDataLoader(
        data_path=data_path,
        device=device,
        normalize_stimuli=normalize_stimuli,
        normalize_fmri=normalize_fmri
    )


# Example usage
if __name__ == "__main__":
    # Example: Load stimulus data
    data_path = "data/digit69_28x28.mat"
    
    # Initialize loader
    loader = load_stimulus_data(data_path, device='cpu', normalize_stimuli=True, normalize_fmri=True)
    
    # Get data info
    info = loader.get_data_info()
    print(f"\n📊 Data Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Visualize a stimulus
    loader.visualize_stimulus(0, split='train')
    
    # Create DataLoader
    train_dataloader = loader.create_paired_dataloader(split='train', batch_size=4, shuffle=True)
    print(f"\n🔄 Created training DataLoader with {len(train_dataloader)} batches")
    
    # Test batch loading
    for i, batch in enumerate(train_dataloader):
        print(f"  Batch {i}:")
        print(f"    Stimulus: {batch['stimulus'].shape}")
        print(f"    fMRI: {batch['fmri'].shape}")
        if 'label' in batch:
            print(f"    Labels: {batch['label'].shape}")
        if i >= 1:  # Show first 2 batches
            break
