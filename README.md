# Latent Diffusion Model for fMRI Image Reconstruction

A comprehensive implementation of Latent Diffusion Models (LDM) for reconstructing fMRI images from neural data. This project combines Variational Autoencoders (VAE) with Denoising Diffusion Probabilistic Models (DDPM) to generate high-quality fMRI reconstructions.

## Features

- **Clean Architecture**: Modular design with separate components for data loading, model architecture, training, and evaluation
- **Comprehensive Metrics**: Extensive evaluation metrics including correlation analysis, error metrics, and spatial pattern analysis
- **Visualization Tools**: Rich visualization capabilities for training monitoring and result analysis
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Mixed Precision Training**: Support for efficient training with automatic mixed precision
- **Checkpointing**: Automatic model checkpointing with best model saving

## Architecture

### Latent Diffusion Model Components

1. **VAE Encoder/Decoder**: Maps fMRI data to/from a lower-dimensional latent space
2. **Diffusion Model**: Learns to denoise latent representations using a U-Net architecture
3. **DDPM Scheduler**: Handles the forward and reverse diffusion processes

### Key Features

- **Sinusoidal Position Embedding**: For timestep encoding in the diffusion process
- **Residual Blocks**: With time embedding integration for stable training
- **Flexible Scheduling**: Support for linear and cosine noise schedules
- **Comprehensive Loss Functions**: VAE reconstruction loss + KL divergence + diffusion loss

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LDM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The model expects fMRI data in the format provided in the `outputs` folder:
- Aligned fMRI data from multiple subjects
- Data shape: `[n_timepoints, n_voxels]` per subject
- Supported format: `.npz` files with subject keys

## Configuration

Edit `config.yaml` to customize:

- **Data settings**: Paths, splits, preprocessing options
- **Model architecture**: VAE and diffusion model parameters
- **Training settings**: Batch size, learning rates, epochs
- **Hardware settings**: Device selection, mixed precision

## Usage

### Training and Evaluation

```bash
# Train and evaluate the model
python main.py --config config.yaml --mode both

# Train only
python main.py --config config.yaml --mode train

# Evaluate existing model
python main.py --config config.yaml --mode evaluate --checkpoint path/to/checkpoint.pt
```

### Key Parameters

- `--config`: Path to configuration file (default: `config.yaml`)
- `--mode`: Execution mode (`train`, `evaluate`, or `both`)
- `--checkpoint`: Path to model checkpoint for evaluation

## Project Structure

```
LDM/
├── src/
│   ├── data/
│   │   └── fmri_data_loader.py      # Data loading and preprocessing
│   ├── models/
│   │   ├── vae_encoder_decoder.py   # VAE implementation
│   │   ├── diffusion_model.py       # Diffusion model and scheduler
│   │   └── latent_diffusion_model.py # Complete LDM
│   ├── training/
│   │   └── trainer.py               # Training pipeline
│   └── utils/
│       ├── metrics.py               # Evaluation metrics
│       └── visualization.py         # Visualization tools
├── outputs/                         # fMRI data files
├── config.yaml                      # Configuration file
├── main.py                         # Main execution script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Model Architecture Details

### VAE Component
- **Encoder**: Multi-layer MLP with batch normalization and dropout
- **Decoder**: Symmetric architecture for reconstruction
- **Latent Space**: Configurable dimensionality with reparameterization trick
- **Loss**: Reconstruction loss (MSE) + β-weighted KL divergence

### Diffusion Component
- **U-Net Architecture**: Residual blocks with time embedding
- **Timestep Encoding**: Sinusoidal position embeddings
- **Noise Scheduling**: Linear or cosine beta schedules
- **Sampling**: DDPM-style reverse diffusion process

## Evaluation Metrics

The model provides comprehensive evaluation including:

### Correlation Metrics
- Overall Pearson correlation
- Voxel-wise correlations
- Sample-wise correlations

### Error Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score
- Signal-to-Noise Ratio (SNR)
- Peak Signal-to-Noise Ratio (PSNR)

### Distribution Metrics
- Statistical moments comparison
- Kolmogorov-Smirnov test
- Jensen-Shannon divergence

### Spatial Metrics
- Spatial correlation pattern analysis
- Voxel-voxel relationship preservation

## Visualization

The framework generates comprehensive visualizations:

- **Training Curves**: Loss progression and convergence analysis
- **Reconstruction Quality**: Original vs reconstructed comparisons
- **Correlation Analysis**: Detailed correlation breakdowns
- **Latent Space Analysis**: PCA and t-SNE projections
- **Generated Samples**: Quality assessment of generated data

## Configuration Options

### Data Configuration
```yaml
data:
  data_path: "outputs"
  aligned_data_file: "alignment_ridge_20250527_070315_aligned_data.npz"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

### Model Configuration
```yaml
vae:
  input_dim: 3092
  latent_dim: 256
  hidden_dims: [1024, 512, 256]
  beta: 1.0

diffusion:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  model_channels: 128
```

## Performance Optimization

- **Mixed Precision Training**: Reduces memory usage and speeds up training
- **Gradient Accumulation**: Enables larger effective batch sizes
- **Efficient Data Loading**: Multi-worker data loading with memory pinning
- **Checkpointing**: Regular model saving with best model tracking

## Logging and Monitoring

- **Console Logging**: Real-time training progress
- **File Logging**: Persistent log files
- **Weights & Biases**: Optional experiment tracking
- **Metric Tracking**: Comprehensive metric logging and visualization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient accumulation
2. **Slow Training**: Enable mixed precision and increase num_workers
3. **Poor Reconstruction**: Adjust VAE beta parameter or increase latent dimensions
4. **Unstable Training**: Reduce learning rates or adjust noise schedule

### Performance Tips

- Use GPU for training when available
- Adjust batch size based on available memory
- Monitor validation metrics for early stopping
- Experiment with different latent dimensions

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fmri_latent_diffusion,
  title={Latent Diffusion Model for fMRI Image Reconstruction},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/LDM}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
