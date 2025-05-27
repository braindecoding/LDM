"""
🧠 Hyperalignment fMRI Data Loader

A clean, well-structured data loader for hyperalignment-processed fMRI data.
This loader handles aligned fMRI data from multiple subjects for brain decoding tasks.

Author: Brain Decoding Team
Date: 2025
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import os
from pathlib import Path


class HyperalignmentDataLoader:
    """
    Clean data loader for hyperalignment-processed fMRI data.
    
    This class loads and manages fMRI data that has been processed through
    hyperalignment algorithms to align brain patterns across subjects.
    """
    
    def __init__(self, 
                 data_path: str,
                 device: str = 'cpu',
                 normalize_data: bool = True):
        """
        Initialize the hyperalignment data loader.
        
        Args:
            data_path: Path to the .npz file containing hyperalignment data
            device: Device to load data on ('cpu' or 'cuda')
            normalize_data: Whether to normalize the fMRI data
        """
        self.data_path = Path(data_path)
        self.device = device
        self.normalize_data = normalize_data
        
        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Hyperalignment data file not found: {data_path}")
        
        # Load and process data
        self.raw_data = self._load_hyperalignment_data()
        self.processed_data = self._process_data()
        
        print(f"✅ Loaded hyperalignment data from: {self.data_path.name}")
        print(f"📊 Number of subjects: {len(self.processed_data)}")
        print(f"🧠 Data shape per subject: {list(self.processed_data.values())[0].shape}")
    
    def _load_hyperalignment_data(self) -> Dict[str, np.ndarray]:
        """
        Load hyperalignment data from .npz file.
        
        Returns:
            Dictionary containing subject data
        """
        try:
            data = np.load(self.data_path)
            return {key: data[key] for key in data.keys()}
        except Exception as e:
            raise RuntimeError(f"Failed to load hyperalignment data: {e}")
    
    def _process_data(self) -> Dict[str, torch.Tensor]:
        """
        Process and normalize the loaded data.
        
        Returns:
            Dictionary of processed torch tensors
        """
        processed = {}
        
        for subject_id, subject_data in self.raw_data.items():
            # Convert to torch tensor
            tensor_data = torch.FloatTensor(subject_data)
            
            # Normalize if requested
            if self.normalize_data:
                tensor_data = self._normalize_tensor(tensor_data)
            
            # Move to device
            tensor_data = tensor_data.to(self.device)
            
            processed[subject_id] = tensor_data
            
        return processed
    
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
    
    def get_subject_data(self, subject_id: str) -> torch.Tensor:
        """
        Get data for a specific subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'subject_0')
            
        Returns:
            Subject's fMRI data tensor
        """
        if subject_id not in self.processed_data:
            available_subjects = list(self.processed_data.keys())
            raise ValueError(f"Subject {subject_id} not found. Available: {available_subjects}")
        
        return self.processed_data[subject_id]
    
    def get_all_subjects_data(self) -> Dict[str, torch.Tensor]:
        """
        Get data for all subjects.
        
        Returns:
            Dictionary mapping subject IDs to their data tensors
        """
        return self.processed_data.copy()
    
    def get_combined_data(self) -> torch.Tensor:
        """
        Combine data from all subjects into a single tensor.
        
        Returns:
            Combined tensor with shape (total_samples, features)
        """
        all_data = list(self.processed_data.values())
        return torch.cat(all_data, dim=0)
    
    def get_subject_list(self) -> List[str]:
        """
        Get list of available subject IDs.
        
        Returns:
            List of subject identifiers
        """
        return list(self.processed_data.keys())
    
    def get_data_info(self) -> Dict[str, Union[int, Tuple[int, ...]]]:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        sample_data = list(self.processed_data.values())[0]
        total_samples = sum(data.shape[0] for data in self.processed_data.values())
        
        return {
            'num_subjects': len(self.processed_data),
            'samples_per_subject': sample_data.shape[0],
            'features_per_sample': sample_data.shape[1],
            'total_samples': total_samples,
            'data_shape_per_subject': sample_data.shape,
            'device': str(self.device),
            'normalized': self.normalize_data
        }
    
    def create_dataloader(self, 
                         subject_ids: Optional[List[str]] = None,
                         batch_size: int = 8,
                         shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
        """
        Create a PyTorch DataLoader for the hyperalignment data.
        
        Args:
            subject_ids: List of subject IDs to include (None for all)
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        if subject_ids is None:
            data_tensor = self.get_combined_data()
        else:
            selected_data = [self.get_subject_data(sid) for sid in subject_ids]
            data_tensor = torch.cat(selected_data, dim=0)
        
        dataset = HyperalignmentDataset(data_tensor)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device != 'cpu' else False
        )


class HyperalignmentDataset(Dataset):
    """
    PyTorch Dataset for hyperalignment fMRI data.
    """
    
    def __init__(self, data: torch.Tensor):
        """
        Initialize the dataset.
        
        Args:
            data: Tensor containing fMRI data
        """
        self.data = data
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            fMRI data sample
        """
        return self.data[idx]


def load_hyperalignment_data(data_path: str, 
                           device: str = 'cpu',
                           normalize: bool = True) -> HyperalignmentDataLoader:
    """
    Convenience function to load hyperalignment data.
    
    Args:
        data_path: Path to hyperalignment .npz file
        device: Device to load data on
        normalize: Whether to normalize the data
        
    Returns:
        Initialized HyperalignmentDataLoader
    """
    return HyperalignmentDataLoader(
        data_path=data_path,
        device=device,
        normalize_data=normalize
    )


# Example usage
if __name__ == "__main__":
    # Example: Load hyperalignment data
    data_path = "outputs/alignment_hyperalignment_20250527_070319_aligned_data.npz"
    
    # Initialize loader
    loader = load_hyperalignment_data(data_path, device='cpu', normalize=True)
    
    # Get data info
    info = loader.get_data_info()
    print(f"\n📊 Data Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get data for specific subject
    subject_data = loader.get_subject_data('subject_0')
    print(f"\n🧠 Subject 0 data shape: {subject_data.shape}")
    
    # Create DataLoader
    dataloader = loader.create_dataloader(batch_size=4, shuffle=True)
    print(f"\n🔄 Created DataLoader with {len(dataloader)} batches")
    
    # Test batch loading
    for i, batch in enumerate(dataloader):
        print(f"  Batch {i}: {batch.shape}")
        if i >= 2:  # Show first 3 batches
            break
