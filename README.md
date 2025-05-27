# 🧠 Brain Decoding with Latent Diffusion Models (LDM)

Clean implementation of Latent Diffusion Model for reconstructing visual stimuli from fMRI brain signals.

## 📁 Project Structure (Clean!)

```
├── data_loader.py          # fMRI data loader
├── brain_ldm.py           # LDM architecture for brain decoding
├── train_brain_ldm.py     # Training script
├── evaluate_brain_ldm.py  # Evaluation script
├── demo_brain_ldm.py      # Complete demo
├── view_results.py        # Results visualization
├── example_usage.py       # Data loader examples
├── data/
│   └── digit69_28x28.mat  # fMRI and stimulus data
├── checkpoints/           # Trained model checkpoints
├── results/               # All visualizations and results
│   ├── training_samples/  # Training progression images
│   ├── evaluation/        # Evaluation results
│   └── SUMMARY_REPORT.md  # Complete results summary
└── README.md              # This file
```

**Clean, focused, no clutter!** ✨

## 🚀 Quick Start

### 1. Run Demo (Recommended)
```bash
python demo_brain_ldm.py
```
*Clean demo showing core brain decoding concepts with simple model*

### 2. Train Full LDM Model
```bash
python train_brain_ldm.py
```
*Train the complete Latent Diffusion Model*

### 3. Evaluate Results
```bash
python evaluate_brain_ldm.py
```
*Evaluate trained model performance*

### 4. View All Results
```bash
python view_results.py
```
*Organize and view all training/evaluation visualizations* 

## 🏗️ LDM Architecture

```
fMRI Signals (3092) → fMRI Encoder → Condition Features (512)
                                           ↓
Stimulus Images (784) → VAE Encoder → Latents (4×7×7)
                                           ↓
                    U-Net Diffusion (Conditional)
                                           ↓
                    Denoised Latents → VAE Decoder → Reconstructed Images
```

### Components:
- **fMRI Encoder**: Maps brain signals to latent condition features
- **VAE**: Encodes/decodes images to/from latent space
- **U-Net**: Conditional diffusion model for denoising
- **Scheduler**: Controls the diffusion process

## 📊 Data Structure

- **Training**: 90 samples (fMRI + stimulus pairs)
- **Test**: 10 samples
- **fMRI Input**: 3092 voxels (brain activity)
- **Stimulus Output**: 784 pixels (28×28 digit images)
- **Task**: Reconstruct visual stimuli from brain signals

## 💡 Key Features

- ✅ **Clean Architecture**: Well-structured LDM components
- ✅ **Conditional Generation**: fMRI-conditioned diffusion
- ✅ **Latent Space**: Efficient VAE encoding/decoding
- ✅ **Training Pipeline**: Complete training and evaluation
- ✅ **Visualization**: Built-in result visualization
- ✅ **No Clutter**: Minimal, focused codebase

## 🔬 Model Details

### fMRI Encoder
```python
fMRI (3092) → Linear Layers → Condition Features (512)
```

### VAE
```python
Images (1×28×28) → Encoder → Latents (4×7×7)
Latents (4×7×7) → Decoder → Images (1×28×28)
```

### U-Net Diffusion
```python
Noisy Latents + fMRI Features + Timestep → Predicted Noise
```

## 🚀 Training Process

1. **Encode** stimulus images to latent space
2. **Add noise** to latents (forward diffusion)
3. **Condition** U-Net on fMRI features
4. **Predict noise** and compute loss
5. **Generate** by iterative denoising

## 📈 Evaluation Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **Correlation**: Pixel-wise correlation
- **SSIM**: Structural similarity (simplified)

Ready for brain decoding experiments! 🧠✨
