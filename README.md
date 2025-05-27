# 🧠 Latent Diffusion Model for fMRI Reconstruction

A clean, well-structured PyTorch implementation of a Latent Diffusion Model for reconstructing fMRI brain activation patterns. This project combines Variational Autoencoders with Diffusion Models to learn and generate high-quality fMRI representations.

## ✨ Key Features

- **🏗️ Clean Architecture**: Modular design with clear separation of concerns
- **🧠 fMRI-Optimized**: Specifically designed for brain activation pattern reconstruction  
- **📊 Comprehensive Evaluation**: 36+ metrics including PSNR, SSIM, FID, LPIPS, CLIP
- **👥 Multi-Subject Support**: Handles aligned data from multiple subjects
- **📈 Rich Visualizations**: Training curves, reconstruction comparisons, latent analysis
- **🚀 Easy to Use**: Simple scripts for training, evaluation, and visualization

## 🎯 Quick Start

```bash
# 1. Setup project
python scripts/setup_project.py

# 2. Run demo to understand the project
python demo.py

# 3. Train and evaluate model
python scripts/train_model.py --mode both

# 4. View results
python scripts/visualize_results.py
```

## 📁 Project Structure

```
LDM/
├── 📁 src/                          # Core implementation
│   ├── 📁 data/
│   │   └── fmri_data_loader.py      # Data loading & preprocessing
│   ├── 📁 models/
│   │   ├── vae_encoder_decoder.py   # VAE implementation
│   │   ├── diffusion_model.py       # Diffusion model & scheduler
│   │   └── latent_diffusion_model.py # Complete LDM
│   ├── 📁 training/
│   │   └── trainer.py               # Training pipeline
│   └── 📁 utils/
│       ├── metrics.py               # Evaluation metrics
│       └── visualization.py         # Visualization tools
├── 📁 scripts/                      # Clean execution scripts
│   ├── train_model.py               # Training script
│   ├── visualize_results.py         # Results visualization
│   ├── setup_project.py             # Project setup
│   └── clean_outputs.py             # Output management
├── 📁 outputs/                      # fMRI data files
├── 📁 checkpoints/                  # Saved models
├── 📁 logs/                         # Training logs & metrics
├── config.yaml                      # Configuration
├── main.py                         # Legacy main script
├── demo.py                         # Project demo
└── README.md                       # This file
```

## 🏗️ Architecture

### Model Components

1. **VAE (Variational Autoencoder)**
   - Input: fMRI data (3092 voxels)
   - Encoder: [3092 → 1024 → 512 → 256] (compression)
   - Decoder: [256 → 512 → 1024 → 3092] (reconstruction)
   - Purpose: Learn compact latent representation

2. **Diffusion Model**
   - Input: Latent codes (256-dimensional)
   - Process: Iterative denoising over 1000 steps
   - Architecture: U-Net with attention layers
   - Purpose: Generate high-quality latent representations

3. **Complete Pipeline**
   - Step 1: Encode fMRI → Latent space
   - Step 2: Add controlled noise
   - Step 3: Iterative denoising
   - Step 4: Decode → Reconstructed fMRI

## 📊 Data

The model works with aligned fMRI data:
- **Format**: `.npz` files with subject data
- **Structure**: (timepoints, voxels) per subject
- **Example**: 3 subjects × 27 timepoints × 3092 voxels
- **Preprocessing**: Normalized and aligned across subjects

## ⚙️ Configuration

Key configuration options in `config.yaml`:

```yaml
# Data settings
data:
  data_path: "outputs"
  aligned_data_file: "alignment_ridge_*.npz"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

# VAE settings
vae:
  input_dim: 3092
  latent_dim: 256
  hidden_dims: [1024, 512, 256]
  beta: 1.0

# Diffusion settings
diffusion:
  num_timesteps: 1000
  beta_schedule: "linear"
  model_channels: 128
  num_res_blocks: 2

# Training settings
training:
  batch_size: 8
  num_epochs: 5
  learning_rate: 1e-4
```

## 📈 Evaluation Metrics

### Core Metrics
- **Correlation**: Linear relationship (0-1, higher better)
- **RMSE**: Root Mean Square Error (lower better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher better)
- **SSIM**: Structural Similarity Index (0-1, higher better)

### Advanced Metrics
- **FID**: Fréchet Inception Distance (lower better)
- **LPIPS**: Learned Perceptual Similarity (lower better)
- **CLIP Score**: Semantic similarity (higher better)

### Expected Ranges
- **Correlation**: 0.3-0.8 (fMRI is inherently noisy)
- **RMSE**: 0.3-0.8 (normalized data)
- **PSNR**: 10-30 dB (typical for fMRI)
- **SSIM**: 0.1-0.7 (no natural spatial structure)

## 🎨 Visualizations

The model generates comprehensive visualizations:

1. **Stimulus vs Reconstruction**: Side-by-side comparison
2. **Training Curves**: Loss progression and convergence
3. **Metrics Dashboard**: All evaluation metrics in one view
4. **Correlation Analysis**: Detailed correlation breakdowns
5. **Latent Space Analysis**: PCA and t-SNE projections
6. **Voxel Activation Heatmaps**: Spatial activation patterns

## 🧠 Understanding fMRI Data

**Why does fMRI look like 'noise'?**
- fMRI measures brain activation, not visual images
- Each voxel represents a brain region's activity
- No natural 2D spatial structure like photos
- Inherently noisy due to measurement limitations

**What matters for evaluation:**
- Statistical correlation between original and reconstruction
- Preservation of activation patterns
- Temporal consistency across timepoints
- NOT visual similarity to natural images

## 🚀 Usage Examples

### Training
```bash
# Full training and evaluation
python scripts/train_model.py --mode both

# Training only
python scripts/train_model.py --mode train

# Evaluation only
python scripts/train_model.py --mode evaluate
```

### Visualization
```bash
# View latest results
python scripts/visualize_results.py

# Custom config
python scripts/visualize_results.py --config custom_config.yaml
```

### Maintenance
```bash
# Clean old outputs
python scripts/clean_outputs.py --all

# Check disk usage
python scripts/clean_outputs.py --usage
```

### Demo
```bash
# Understand the project
python demo.py
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Scikit-learn
- PyYAML
- Scipy

## 🎯 Expected Results

For a quick 5-epoch training:
- **Correlation**: ~0.3-0.6
- **RMSE**: ~0.4-0.8
- **PSNR**: ~10-20 dB

For longer training (50+ epochs):
- **Correlation**: ~0.6-0.8
- **RMSE**: ~0.2-0.5
- **PSNR**: ~20-30 dB

## 🔧 Troubleshooting

**Common Issues:**
1. **CUDA out of memory**: Reduce batch size in config
2. **Low correlation**: Increase training epochs
3. **Slow training**: Enable mixed precision
4. **Missing data**: Check outputs/ folder for aligned data

## 📚 References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**🎉 Ready to start? Run `python scripts/setup_project.py` to get started!**
