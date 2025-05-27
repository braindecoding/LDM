# 🧠 Brain Decoding with Latent Diffusion Models

A clean, well-structured PyTorch implementation of a Latent Diffusion Model for **brain decoding** - reconstructing visual stimuli from fMRI brain activity. This project demonstrates how to decode what the brain "sees" using state-of-the-art generative models.

## 🎯 Project Goal

**Brain Decoding**: Reconstruct visual stimuli (28×28 digit images) from fMRI brain activation patterns.

- **Input**: fMRI brain activity data (3092 voxels)
- **Output**: Visual stimulus reconstruction (28×28 pixels)
- **Task**: Neural decoding / Mind reading
- **Method**: Latent Diffusion Model enhancement

## ✨ Key Features

- **🧠 True Brain Decoding**: Reconstructs visual stimuli from brain activity
- **🏗️ Clean Code Architecture**: Modular design with clear naming conventions
- **🌊 Latent Diffusion Enhancement**: Uses diffusion models to improve reconstruction quality
- **📊 Comprehensive Evaluation**: Correlation analysis and reconstruction quality metrics
- **🎨 Rich Visualizations**: Side-by-side stimulus comparisons and quality assessments
- **🚀 Easy to Use**: Clean scripts with descriptive names and clear documentation

## 🎯 Quick Start

### Option A: Test Pre-trained Model (Fast)
```bash
# 1. Setup project and dependencies
python scripts/setup_project.py

# 2. Understand brain decoding concept
python scripts/analyze_stimulus_data.py

# 3. Test brain decoding with clean implementation
python scripts/test_clean_brain_decoding.py

# 4. Compare VAE vs LDM approaches
python scripts/compare_vae_vs_ldm.py
```

### Option B: Train Your Own Model (Complete)
```bash
# 1. Setup project
python scripts/setup_project.py

# 2. Analyze data
python scripts/analyze_stimulus_data.py

# 3. Train brain decoding model
python scripts/train_brain_decoding_model.py --phase both

# 4. Evaluate trained model
python scripts/evaluate_trained_model.py --model checkpoints/brain_decoding_final_model.pt
```

## 📁 Clean Project Structure

```
BrainDecoding-LDM/
├── 📁 src/                                    # Core implementation with clean naming
│   ├── 📁 data/
│   │   ├── stimulus_data_loader.py           # Brain-to-stimulus data loading
│   │   └── fmri_data_loader.py               # Legacy fMRI data loader
│   ├── 📁 models/
│   │   ├── stimulus_ldm.py                   # Brain decoding LDM model
│   │   ├── vae_encoder_decoder.py            # VAE for brain→stimulus mapping
│   │   └── diffusion_model.py                # Diffusion enhancement model
│   ├── 📁 training/
│   │   └── trainer.py                        # Training pipeline
│   └── 📁 utils/
│       ├── metrics.py                        # Reconstruction quality metrics
│       └── visualization.py                  # Clean visualization tools
├── 📁 scripts/                               # Clean execution scripts
│   ├── train_brain_decoding_model.py         # Main training script
│   ├── evaluate_trained_model.py             # Model evaluation script
│   ├── test_clean_brain_decoding.py          # Brain decoding test
│   ├── analyze_stimulus_data.py              # Data structure analysis
│   ├── compare_vae_vs_ldm.py                 # Method comparison
│   ├── setup_project.py                      # Project initialization
│   └── clean_outputs.py                      # Output management
├── 📁 data/                                  # Brain and stimulus data
│   └── digit69_28x28.mat                     # fMRI + digit stimulus data
├── 📁 results/                               # Clean results organization
│   ├── clean_brain_decoding/                 # Main results
│   ├── stimulus_analysis/                    # Data analysis results
│   └── vae_vs_ldm_comparison/                # Method comparison
├── 📁 checkpoints/                           # Trained model weights
├── 📁 logs/                                  # Training logs & metrics
├── config.yaml                               # Clean configuration
├── demo.py                                   # Project demonstration
└── README.md                                 # This documentation
```

## 🏗️ Brain Decoding Architecture

### Clean Model Pipeline

```
fMRI Brain Activity (3092 voxels)
    ↓ [Brain Activity Encoder]
Latent Representation (256 dimensions)
    ↓ [Diffusion Enhancement]
Enhanced Latent Representation
    ↓ [Visual Stimulus Decoder]
Reconstructed Visual Stimulus (28×28 pixels)
```

### Model Components

1. **Brain-to-Latent VAE Encoder**
   - **Input**: fMRI brain activity (3092 voxels)
   - **Architecture**: [3092 → 1024 → 512 → 256]
   - **Purpose**: Map brain patterns to latent space
   - **Output**: Latent representation (256 dimensions)

2. **Latent Diffusion Enhancement**
   - **Input**: Latent codes (256-dimensional)
   - **Process**: Controlled noise addition + iterative denoising
   - **Architecture**: U-Net with attention mechanisms
   - **Purpose**: Enhance latent representations for better reconstruction

3. **Latent-to-Stimulus VAE Decoder**
   - **Input**: Enhanced latent representation (256 dimensions)
   - **Architecture**: [256 → 512 → 1024 → 784]
   - **Purpose**: Generate visual stimulus from latent space
   - **Output**: Reconstructed stimulus (28×28 = 784 pixels)

### Complete Brain Decoding Pipeline

1. **Encode**: Brain activity → Latent space
2. **Enhance**: Add noise + Diffusion denoising
3. **Decode**: Enhanced latent → Visual stimulus
4. **Evaluate**: Compare with true stimulus

## 📊 Brain Decoding Data

The model works with paired brain activity and visual stimulus data:

### Data Structure (digit69_28x28.mat)
- **fmriTrn**: (90, 3092) - Training brain activity data
- **stimTrn**: (90, 784) - Training visual stimuli (28×28 digits)
- **fmriTest**: (10, 3092) - Test brain activity data
- **stimTest**: (10, 784) - Test visual stimuli
- **labelTrn/Test**: Digit labels (0-9) for classification

### Data Characteristics
- **Brain Data**: 3092 voxels representing brain activation patterns
- **Visual Data**: 784 pixels (28×28) representing digit images
- **Task**: Decode visual stimuli from corresponding brain activity
- **Preprocessing**: Normalized brain activity and stimulus data

## ⚙️ Clean Configuration

Key configuration options in `config.yaml`:

```yaml
# Brain decoding data settings
data:
  brain_data_path: "data/digit69_28x28.mat"
  train_split: 0.7
  validation_split: 0.2
  test_split: 0.1

# Brain-to-stimulus VAE settings
vae:
  brain_input_dim: 3092        # fMRI voxels
  stimulus_output_dim: 784     # 28×28 pixels
  latent_dim: 256              # Latent space dimension
  hidden_dims: [1024, 512, 256]
  beta: 1.0                    # KL divergence weight

# Latent diffusion enhancement settings
diffusion:
  num_timesteps: 1000          # Diffusion steps
  beta_schedule: "linear"      # Noise schedule
  model_channels: 128          # U-Net channels
  num_res_blocks: 2            # Residual blocks
  dropout: 0.1                 # Dropout rate

# Training settings
training:
  batch_size: 8                # Batch size
  num_epochs: 50               # Training epochs
  learning_rate: 1e-4          # Learning rate
  vae_training_epochs: 25      # VAE pre-training
  diffusion_training_epochs: 25 # Diffusion training

# Hardware settings
hardware:
  num_workers: 4               # Data loading workers
  pin_memory: true             # GPU memory optimization
```

## 📈 Brain Decoding Evaluation

### Reconstruction Quality Metrics
- **Pearson Correlation**: Measures linear relationship between true and reconstructed stimuli (0-1, higher better)
- **Mean Squared Error (MSE)**: Pixel-wise reconstruction error (lower better)
- **Visual Similarity**: Qualitative assessment of digit recognition

### Expected Performance Ranges
- **Good Brain Decoding**: Correlation > 0.3
- **Excellent Brain Decoding**: Correlation > 0.5
- **Typical MSE**: 0.1-0.5 (normalized pixel values)

### Method Comparison
- **Baseline (VAE only)**: Direct brain→stimulus mapping
- **Enhanced (LDM)**: Diffusion-enhanced reconstruction
- **Improvement**: LDM typically shows 10-30% correlation improvement

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

## 🏋️ Training Brain Decoding Model

### Complete Training Pipeline
```bash
# 1. Setup project
python scripts/setup_project.py

# 2. Analyze data structure
python scripts/analyze_stimulus_data.py

# 3. Train complete model (VAE + Diffusion)
python scripts/train_brain_decoding_model.py --phase both

# 4. Evaluate trained model
python scripts/evaluate_trained_model.py --model checkpoints/brain_decoding_final_model.pt
```

### Phase-by-Phase Training
```bash
# Train VAE only (brain → stimulus mapping)
python scripts/train_brain_decoding_model.py --phase vae

# Train Diffusion only (latent enhancement)
python scripts/train_brain_decoding_model.py --phase diffusion
```

### Training Process
1. **Phase 1 - VAE Training (25 epochs)**:
   - Learn brain activity → latent space mapping
   - Learn latent space → visual stimulus mapping
   - Expected correlation: 0.2-0.4

2. **Phase 2 - Diffusion Training (25 epochs)**:
   - Learn to enhance latent representations
   - Improve reconstruction quality through denoising
   - Expected improvement: 20-50% better correlation

### Training Outputs
```
checkpoints/
├── brain_decoding_vae_epoch_*.pt       # VAE checkpoints
├── brain_decoding_diffusion_epoch_*.pt # Diffusion checkpoints
└── brain_decoding_final_model.pt       # Final trained model

logs/
├── brain_decoding_training_history.json # Training metrics
└── training_progress.png               # Training curves
```

## 🚀 Clean Usage Examples

### Brain Decoding Analysis
```bash
# Analyze brain and stimulus data structure
python scripts/analyze_stimulus_data.py

# Test clean brain decoding implementation
python scripts/test_clean_brain_decoding.py

# Compare VAE vs LDM approaches
python scripts/compare_vae_vs_ldm.py
```

### Project Management
```bash
# Setup project and check dependencies
python scripts/setup_project.py

# Clean old outputs and manage disk space
python scripts/clean_outputs.py --all

# Check current results and disk usage
python scripts/clean_outputs.py --usage
```

### Legacy Scripts (for compatibility)
```bash
# Original training pipeline
python scripts/train_model.py --mode both

# Original visualization
python scripts/visualize_results.py
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Scikit-learn
- PyYAML
- Scipy

## 🎯 Expected Brain Decoding Results

### Baseline Performance (VAE only)
- **Correlation**: 0.2-0.4 (basic brain decoding)
- **MSE**: 0.3-0.6 (reconstruction error)
- **Visual Quality**: Recognizable digit shapes

### Enhanced Performance (LDM)
- **Correlation**: 0.3-0.6 (improved brain decoding)
- **MSE**: 0.2-0.4 (reduced reconstruction error)
- **Visual Quality**: Clearer, more detailed digit reconstruction
- **Improvement**: 20-50% better correlation than baseline

### Performance Factors
- **Data Quality**: Clean fMRI signals improve results
- **Model Training**: Longer training generally improves performance
- **Individual Differences**: Brain patterns vary between subjects

## 🧹 Clean Code Principles

This project demonstrates clean code principles throughout:

### Clear Naming Conventions
- **Classes**: `BrainToStimulusDataLoader`, `CleanBrainDecodingTester`
- **Variables**: `brain_activity_data`, `visual_stimulus_data`, `enhanced_reconstruction`
- **Functions**: `perform_brain_decoding_test()`, `compute_reconstruction_quality_metrics()`

### Single Responsibility
- Each function has one clear purpose
- Separated data loading, model testing, and visualization
- Clean error handling and logging

### Self-Documenting Code
- Descriptive variable and function names
- Clear type hints and documentation
- Meaningful comments explaining why, not what

## 🔧 Troubleshooting

**Common Issues:**
1. **Missing data file**: Ensure `data/digit69_28x28.mat` exists
2. **CUDA out of memory**: Reduce batch size in config
3. **Low correlation**: Check data preprocessing and model architecture
4. **Import errors**: Run `python scripts/setup_project.py` first

## 📚 References

### Core Papers
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Foundation of diffusion models
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) - Latent diffusion approach
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE fundamentals

### Brain Decoding Research
- [Deep learning for neural decoding](https://www.nature.com/articles/s41593-018-0107-3) - Neural decoding overview
- [fMRI-based decoding of visual information](https://www.sciencedirect.com/science/article/pii/S1053811917305621) - Visual decoding methods

## 🎓 Educational Value

This project demonstrates:
- **Clean Code Practices**: Professional software development standards
- **Brain Decoding**: How AI can decode brain signals
- **Latent Diffusion Models**: State-of-the-art generative modeling
- **Neuroscience Applications**: AI applications in brain research

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Follow clean code principles
- Use descriptive naming conventions
- Add proper documentation
- Include type hints

## 🙏 Acknowledgments

- Clean code principles inspired by Robert C. Martin's "Clean Code"
- Brain decoding methodology based on neuroscience research
- Diffusion model implementation following best practices

---

**🎉 Ready to decode brains? Run `python scripts/setup_project.py` to get started!**

### Quick Test
```bash
# 1. Setup
python scripts/setup_project.py

# 2. Analyze data
python scripts/analyze_stimulus_data.py

# 3. Test brain decoding
python scripts/test_clean_brain_decoding.py
```

**🧠 Welcome to the fascinating world of brain decoding with clean code!**
