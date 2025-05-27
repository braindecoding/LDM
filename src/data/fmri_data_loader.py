"""
fMRI Data Loader for Latent Diffusion Model
Handles loading, preprocessing, and batching of fMRI data from aligned outputs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FMRIDataset(Dataset):
    """
    Dataset class for fMRI data with proper normalization and augmentation.
    """

    def __init__(
        self,
        data: np.ndarray,
        normalize: bool = True,
        augment: bool = False,
        noise_std: float = 0.01
    ):
        """
        Initialize fMRI dataset.

        Args:
            data: fMRI data array of shape (n_samples, n_voxels)
            normalize: Whether to normalize the data
            augment: Whether to apply data augmentation
            noise_std: Standard deviation for noise augmentation
        """
        self.data = torch.FloatTensor(data)
        self.normalize = normalize
        self.augment = augment
        self.noise_std = noise_std

        if self.normalize:
            self._normalize_data()

        logger.info(f"Initialized fMRI dataset with {len(self.data)} samples")

    def _normalize_data(self) -> None:
        """Normalize data to zero mean and unit variance."""
        self.mean = torch.mean(self.data, dim=0, keepdim=True)
        self.std = torch.std(self.data, dim=0, keepdim=True) + 1e-8
        self.data = (self.data - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Preprocessed fMRI sample
        """
        sample = self.data[idx].clone()

        if self.augment:
            # Add Gaussian noise for augmentation
            noise = torch.randn_like(sample) * self.noise_std
            sample = sample + noise

        return sample


class FMRIDataLoader:
    """
    Data loader manager for fMRI data with train/val/test splits.
    """

    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_path = Path(config['data']['data_path'])
        self.aligned_data_file = config['data']['aligned_data_file']

        # Load and prepare data
        self.raw_data = self._load_aligned_data()
        self.train_data, self.val_data, self.test_data = self._split_data()

        # Create datasets
        self.train_dataset = FMRIDataset(
            self.train_data,
            normalize=True,
            augment=True
        )
        self.val_dataset = FMRIDataset(
            self.val_data,
            normalize=True,
            augment=False
        )
        self.test_dataset = FMRIDataset(
            self.test_data,
            normalize=True,
            augment=False
        )

        logger.info("Data loader initialized successfully")

    def _load_aligned_data(self) -> np.ndarray:
        """
        Load aligned fMRI data from npz file.

        Returns:
            Combined data from all subjects
        """
        data_file = self.data_path / self.aligned_data_file

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        data = np.load(data_file)

        # Combine data from all subjects
        all_subjects_data = []
        for subject_key in data.keys():
            subject_data = data[subject_key]
            all_subjects_data.append(subject_data)
            logger.info(f"Loaded {subject_key}: {subject_data.shape}")

        # Stack all subjects' data
        combined_data = np.vstack(all_subjects_data)
        logger.info(f"Combined data shape: {combined_data.shape}")

        return combined_data

    def _split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/test sets.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_samples = len(self.raw_data)

        # Calculate split indices
        train_size = int(n_samples * self.config['data']['train_split'])
        val_size = int(n_samples * self.config['data']['val_split'])

        # Shuffle indices for random split
        indices = np.random.permutation(n_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_data = self.raw_data[train_indices]
        val_data = self.raw_data[val_indices]
        test_data = self.raw_data[test_indices]

        logger.info(f"Data split - Train: {len(train_data)}, "
                   f"Val: {len(val_data)}, Test: {len(test_data)}")

        return train_data, val_data, test_data

    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            drop_last=True
        )

    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

    def get_data_stats(self) -> Dict:
        """Get statistics about the loaded data."""
        return {
            'total_samples': len(self.raw_data),
            'num_voxels': self.raw_data.shape[1],
            'train_samples': len(self.train_data),
            'val_samples': len(self.val_data),
            'test_samples': len(self.test_data),
            'data_mean': np.mean(self.raw_data),
            'data_std': np.std(self.raw_data),
            'data_min': np.min(self.raw_data),
            'data_max': np.max(self.raw_data)
        }
