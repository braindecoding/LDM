# Methodology: Multi-Modal Brain LDM with Uncertainty Quantification

## Overview

This document provides detailed methodology for the multi-modal Brain Latent Diffusion Model (Brain-LDM) with Monte Carlo uncertainty quantification for brain-to-image reconstruction.

## 1. Problem Formulation

### 1.1 Brain-to-Image Reconstruction Task

Given fMRI signals **x** ∈ ℝ^d from visual cortex responses to digit stimuli, reconstruct the corresponding visual stimulus **y** ∈ ℝ^(H×W) where:
- **x**: fMRI signal (d=3092 voxels)
- **y**: Visual stimulus (28×28 pixel image)
- Goal: Learn mapping f: **x** → **y** with uncertainty quantification

### 1.2 Multi-Modal Guidance

Enhance reconstruction using multiple guidance modalities:
- **Text guidance**: Natural language descriptions ("handwritten digit zero")
- **Semantic guidance**: Class label embeddings (digits 0-9)
- **Cross-modal fusion**: Dynamic attention-based integration

## 2. Architecture Design

### 2.1 Multi-Modal Brain LDM Architecture

```
Input: fMRI(x) + Text(t) + Semantic(s)
       ↓
   [fMRI Encoder] [Text Encoder] [Semantic Embedding]
       ↓              ↓              ↓
   fMRI Features  Text Features  Semantic Features
       ↓              ↓              ↓
            [Cross-Modal Attention]
                     ↓
              Fused Representation
                     ↓
              [VAE Encoder] → Latent Space
                     ↓
              [Conditional U-Net] ← Guidance
                     ↓
              [VAE Decoder] → Reconstructed Image
```

### 2.2 Component Details

#### fMRI Encoder
```python
fMRI_Encoder = Sequential(
    Linear(3092 → 1024),
    LayerNorm(1024),
    ReLU(),
    Dropout(0.3),
    Linear(1024 → 512),
    LayerNorm(512),
    ReLU(),
    Dropout(0.2)
)
```

#### Text Encoder
```python
Text_Encoder = Sequential(
    Embedding(vocab_size=10000, embed_dim=512),
    Dropout(0.2),
    TransformerEncoder(
        layers=4,
        heads=8,
        dropout=0.2,
        norm_first=True
    ),
    GlobalAveragePooling(),
    Linear(512 → 512),
    LayerNorm(512)
)
```

#### Cross-Modal Attention
```python
CrossModalAttention = MultiHeadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.3,
    temperature_scaling=True
)
```

#### Enhanced U-Net
```python
ConditionalUNet = UNet(
    in_channels=1,
    out_channels=1,
    condition_dim=512,
    skip_connections=True,
    batch_norm=True,
    dropout=0.2
)
```

## 3. Uncertainty Quantification

### 3.1 Monte Carlo Dropout

**Epistemic Uncertainty** (Model Uncertainty):
```python
def monte_carlo_sampling(model, x, n_samples=30):
    model = enable_dropout_for_uncertainty(model)
    samples = []
    for i in range(n_samples):
        # Add noise injection
        x_noisy = x + torch.randn_like(x) * 0.05
        sample = model(x_noisy, add_noise=True)
        samples.append(sample)
    return torch.stack(samples)

# Epistemic uncertainty
epistemic_uncertainty = samples.std(dim=0).mean(dim=1)
```

**Aleatoric Uncertainty** (Data Uncertainty):
```python
# Approximated through prediction variance
aleatoric_uncertainty = samples.var(dim=0).mean(dim=1)
```

### 3.2 Temperature Scaling Calibration

```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature

# Learned temperature parameter for calibration
temperature = 0.971  # Learned value
```

### 3.3 Uncertainty Metrics

**Total Uncertainty**:
```
U_total = U_epistemic + U_aleatoric
```

**Mutual Information**:
```
I(y,θ|x) ≈ H(y|x) - E[H(y|x,θ)]
```

**Calibration Assessment**:
```python
def assess_calibration(uncertainties, errors):
    correlation = np.corrcoef(uncertainties, errors)[0,1]
    
    high_unc_mask = uncertainties > np.quantile(uncertainties, 0.75)
    low_unc_mask = uncertainties < np.quantile(uncertainties, 0.25)
    
    calibration_ratio = errors[low_unc_mask].mean() / errors[high_unc_mask].mean()
    
    return correlation, calibration_ratio
```

## 4. Training Strategy

### 4.1 Multi-Component Loss Function

```python
def compute_total_loss(reconstruction, target, text_tokens, class_labels):
    # Reconstruction loss
    L_recon = MSE(reconstruction, target)
    
    # Perceptual loss (gradient-based)
    grad_x_recon = |∇_x reconstruction|
    grad_y_recon = |∇_y reconstruction|
    grad_x_target = |∇_x target|
    grad_y_target = |∇_y target|
    
    L_perceptual = MSE(grad_x_recon, grad_x_target) + MSE(grad_y_recon, grad_y_target)
    
    # Uncertainty regularization
    L_uncertainty = MSE(semantic_uncertainty, target_uncertainty)
    
    # Total loss with dynamic weighting
    epoch_progress = current_epoch / total_epochs
    w_recon = 1.0
    w_perceptual = 0.1 * (1 + epoch_progress)
    w_uncertainty = 0.01 * (1 + 2 * epoch_progress)
    
    L_total = w_recon * L_recon + w_perceptual * L_perceptual + w_uncertainty * L_uncertainty
    
    return L_total
```

### 4.2 Data Augmentation Strategy

```python
def enhanced_augmentation(fmri_data, stimuli_data, augment_factor=10):
    augmented_data = []
    
    for i in range(augment_factor):
        # Progressive noise levels
        noise_level = 0.01 + (i * 0.02)  # 0.01 to 0.19
        
        # fMRI augmentation
        fmri_noise = torch.randn_like(fmri_data) * noise_level
        aug_fmri = fmri_data + fmri_noise
        
        # Feature dropout
        if i % 3 == 1:
            dropout_rate = 0.02 + (i * 0.01)
            dropout_mask = torch.rand_like(fmri_data) > dropout_rate
            aug_fmri = aug_fmri * dropout_mask
        
        # Signal scaling
        if i % 4 == 3:
            scale_factor = 0.9 + (torch.rand(1) * 0.2)  # 0.9 to 1.1
            aug_fmri = aug_fmri * scale_factor
        
        # Stimulus augmentation
        stim_noise = torch.randn_like(stimuli_data) * 0.01
        aug_stimuli = torch.clamp(stimuli_data + stim_noise, 0, 1)
        
        augmented_data.append((aug_fmri, aug_stimuli))
    
    return augmented_data
```

### 4.3 Optimization Strategy

```python
# Enhanced optimizer with component-specific learning rates
optimizer = AdamW([
    {'params': fmri_encoder.parameters(), 'lr': 8e-5},
    {'params': text_encoder.parameters(), 'lr': 4e-5},
    {'params': semantic_embedding.parameters(), 'lr': 8e-5},
    {'params': cross_modal_attention.parameters(), 'lr': 1.2e-4},
    {'params': unet.parameters(), 'lr': 8e-5},
    {'params': vae.parameters(), 'lr': 4e-5},
    {'params': [temperature], 'lr': 8e-6}
], weight_decay=5e-6)

# Learning rate scheduling
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-7
)
```

## 5. Evaluation Metrics

### 5.1 Reconstruction Quality

**Classification Accuracy**:
```python
def compute_accuracy(reconstructions, targets):
    n_samples = len(reconstructions)
    corr_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            corr_matrix[i,j] = np.corrcoef(
                reconstructions[i].flatten(),
                targets[j].flatten()
            )[0,1]
    
    correct_matches = sum(
        np.argmax(corr_matrix[i,:]) == i 
        for i in range(n_samples)
    )
    
    return correct_matches / n_samples
```

**Average Correlation**:
```python
def compute_correlation(reconstructions, targets):
    correlations = []
    for i in range(len(reconstructions)):
        corr = np.corrcoef(
            reconstructions[i].flatten(),
            targets[i].flatten()
        )[0,1]
        correlations.append(corr if not np.isnan(corr) else 0)
    
    return np.mean(correlations)
```

### 5.2 Uncertainty Quality

**Uncertainty-Error Correlation**:
```python
uncertainty_error_correlation = np.corrcoef(
    total_uncertainties, prediction_errors
)[0,1]
```

**Calibration Ratio**:
```python
high_unc_threshold = np.quantile(uncertainties, 0.75)
low_unc_threshold = np.quantile(uncertainties, 0.25)

high_unc_error = errors[uncertainties > high_unc_threshold].mean()
low_unc_error = errors[uncertainties < low_unc_threshold].mean()

calibration_ratio = low_unc_error / high_unc_error
```

## 6. Implementation Details

### 6.1 Data Preprocessing

```python
def improved_fmri_normalization(fmri_data):
    # Robust normalization using median absolute deviation
    median = torch.median(fmri_data, dim=0, keepdim=True)[0]
    mad = torch.median(torch.abs(fmri_data - median), dim=0, keepdim=True)[0]
    mad = torch.where(mad == 0, torch.ones_like(mad), mad)
    
    # Normalize and clip outliers
    normalized = (fmri_data - median) / (1.4826 * mad)
    normalized = torch.clamp(normalized, -3, 3)
    
    return normalized
```

### 6.2 Reproducibility

**Random Seeds**:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Model Checkpointing**:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'config': config,
    'random_state': torch.get_rng_state()
}
torch.save(checkpoint, 'model_checkpoint.pt')
```

## 7. Computational Requirements

- **Training Time**: 2-4 hours per model (CPU)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ for models and results
- **Dependencies**: PyTorch 2.0+, Python 3.8+

## 8. Validation Strategy

- **Cross-validation**: 5-fold validation on training set
- **Hold-out testing**: 30 samples reserved for final evaluation
- **Uncertainty validation**: Correlation analysis with prediction errors
- **Ablation studies**: Component-wise contribution analysis
