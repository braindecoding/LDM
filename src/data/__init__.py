"""
🧠 Data Loading Module

Clean, well-structured data loaders for brain decoding experiments.
This module provides specialized loaders for different types of fMRI and stimulus data.

Available Loaders:
- HyperalignmentDataLoader: For hyperalignment-processed fMRI data
- StimulusDataLoader: For visual stimulus and paired fMRI data

Author: Brain Decoding Team
Date: 2025
"""

from .hyperalignment_data_loader import (
    HyperalignmentDataLoader,
    HyperalignmentDataset,
    load_hyperalignment_data
)

from .stimulus_data_loader import (
    StimulusDataLoader,
    StimulusFMRIDataset,
    load_stimulus_data
)

__all__ = [
    'HyperalignmentDataLoader',
    'HyperalignmentDataset', 
    'load_hyperalignment_data',
    'StimulusDataLoader',
    'StimulusFMRIDataset',
    'load_stimulus_data'
]
