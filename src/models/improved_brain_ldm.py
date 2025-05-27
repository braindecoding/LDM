"""
🚀 Improved Brain LDM with Uncertainty Calibration
Enhanced model architecture based on uncertainty analysis findings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedTextEncoder(nn.Module):
    """Improved text encoder with better dropout and normalization."""

    def __init__(self, embed_dim=512, max_length=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Enhanced text encoding
        self.vocab_size = 10000
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.dropout_embed = nn.Dropout(0.2)

        # Improved transformer with normalization
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, nhead=8, batch_first=True,
            dropout=0.2, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Output projection with dropout
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout_proj = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, text_tokens):
        # Embedding with dropout
        x = self.embedding(text_tokens)
        x = self.dropout_embed(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Final projection
        x = self.text_projection(x)
        x = self.dropout_proj(x)
        x = self.layer_norm(x)

        return x

class ImprovedSemanticEmbedding(nn.Module):
    """Improved semantic embedding with uncertainty."""

    def __init__(self, num_classes=10, embed_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Add learnable uncertainty
        self.uncertainty_head = nn.Linear(embed_dim, 1)

    def forward(self, class_labels):
        x = self.embedding(class_labels)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # Compute uncertainty
        uncertainty = torch.sigmoid(self.uncertainty_head(x))

        return x, uncertainty

class ImprovedCrossModalAttention(nn.Module):
    """Enhanced cross-modal attention with better regularization."""

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Multi-head attention with higher dropout
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.3, batch_first=True
        )

        # Enhanced fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, fmri_features, text_features=None, semantic_features=None):
        batch_size = fmri_features.size(0)

        # Prepare features for attention
        features_list = [fmri_features.unsqueeze(1)]

        if text_features is not None:
            features_list.append(text_features.unsqueeze(1))
        if semantic_features is not None:
            features_list.append(semantic_features.unsqueeze(1))

        # Concatenate all features
        all_features = torch.cat(features_list, dim=1)  # [B, N, H]

        # Apply cross-modal attention with temperature scaling
        attended_features, attention_weights = self.multihead_attn(
            all_features, all_features, all_features
        )

        # Apply temperature scaling
        attended_features = attended_features / self.temperature

        # Flatten and fuse
        attended_flat = attended_features.reshape(batch_size, -1)
        fused_features = self.fusion_net(attended_flat)

        return fused_features, attention_weights

class ImprovedUNet(nn.Module):
    """Improved U-Net with proper skip connections and dropout."""

    def __init__(self, in_channels=1, out_channels=1, condition_dim=512):
        super().__init__()

        # Encoder with dropout
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)

        # Bottleneck with condition injection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )

        # Condition projection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder with skip connections
        self.dec4 = self._make_decoder_block(1024 + 512, 512)
        self.dec3 = self._make_decoder_block(512 + 256, 256)
        self.dec2 = self._make_decoder_block(256 + 128, 128)
        self.dec1 = self._make_decoder_block(128 + 64, 64)

        # Final output
        self.final = nn.Conv2d(64, out_channels, 1)

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, condition):
        # Encoder path
        e1 = self.enc1(x)  # 64 x 14 x 14
        e2 = self.enc2(e1)  # 128 x 7 x 7
        e3 = self.enc3(e2)  # 256 x 3 x 3
        e4 = self.enc4(e3)  # 512 x 1 x 1

        # Bottleneck
        b = self.bottleneck(e4)  # 1024 x 1 x 1

        # Inject condition
        cond = self.condition_proj(condition)  # [B, 1024]
        cond = cond.unsqueeze(-1).unsqueeze(-1)  # [B, 1024, 1, 1]
        b = b + cond  # Condition injection

        # Decoder path with skip connections
        d4 = self.dec4(torch.cat([b, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # Final output
        output = self.final(d1)
        uncertainty = self.uncertainty_head(d1)

        return output, uncertainty

class ImprovedBrainLDM(nn.Module):
    """Improved Brain LDM with uncertainty calibration."""

    def __init__(self, fmri_dim=3092, image_size=28, guidance_scale=7.5):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.image_size = image_size
        self.guidance_scale = guidance_scale

        # Enhanced components
        self.fmri_encoder = nn.Sequential(
            nn.Linear(fmri_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.text_encoder = ImprovedTextEncoder()
        self.semantic_embedding = ImprovedSemanticEmbedding()
        self.cross_modal_attention = ImprovedCrossModalAttention()

        # Improved VAE components
        self.vae_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512)
        )

        self.vae_decoder = nn.Sequential(
            nn.Linear(512, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # Improved U-Net
        self.unet = ImprovedUNet(condition_dim=512)

        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def encode_multimodal_condition(self, fmri_signals, text_tokens=None, class_labels=None):
        """Enhanced multimodal condition encoding."""
        # Encode fMRI
        fmri_features = self.fmri_encoder(fmri_signals)

        # Encode text if provided
        text_features = None
        if text_tokens is not None:
            text_features = self.text_encoder(text_tokens)

        # Encode semantic if provided
        semantic_features = None
        semantic_uncertainty = None
        if class_labels is not None:
            semantic_features, semantic_uncertainty = self.semantic_embedding(class_labels)

        # Cross-modal attention
        fused_features, attention_weights = self.cross_modal_attention(
            fmri_features, text_features, semantic_features
        )

        return fused_features, attention_weights, semantic_uncertainty

    def generate_with_guidance(self, fmri_signals, text_tokens=None, class_labels=None,
                             guidance_scale=None, add_noise=True):
        """Generate with improved guidance and uncertainty."""
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        batch_size = fmri_signals.size(0)
        device = fmri_signals.device

        # Encode conditions
        condition, attention_weights, semantic_uncertainty = self.encode_multimodal_condition(
            fmri_signals, text_tokens, class_labels
        )

        # Add noise injection for uncertainty
        if add_noise and self.training:
            noise = torch.randn_like(condition) * 0.1
            condition = condition + noise

        # Generate latent
        latent = torch.randn(batch_size, 512, device=device)

        # Apply guidance
        if guidance_scale > 1.0:
            # Conditional generation
            cond_latent = latent + 0.1 * condition

            # Unconditional generation
            uncond_condition = torch.zeros_like(condition)
            uncond_latent = latent + 0.1 * uncond_condition

            # Classifier-free guidance
            guided_latent = uncond_latent + guidance_scale * (cond_latent - uncond_latent)
        else:
            guided_latent = latent + 0.1 * condition

        # Decode to image
        reconstruction = self.vae_decoder(guided_latent)

        # Apply temperature scaling
        reconstruction = reconstruction / self.temperature

        return reconstruction, attention_weights

    def compute_improved_loss(self, fmri_signals, target_images, text_tokens=None, class_labels=None):
        """Compute improved loss with uncertainty terms."""
        # Generate reconstruction
        reconstruction, attention_weights = self.generate_with_guidance(
            fmri_signals, text_tokens, class_labels
        )

        # Reshape target images to match reconstruction
        if target_images.dim() == 2:  # [batch, 784]
            target_images = target_images.view(-1, 1, 28, 28)

        # Basic reconstruction loss
        recon_loss = F.mse_loss(reconstruction, target_images)

        # Perceptual loss (simplified)
        # Using gradient-based perceptual similarity
        grad_x_recon = torch.abs(reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :])
        grad_y_recon = torch.abs(reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1])
        grad_x_target = torch.abs(target_images[:, :, 1:, :] - target_images[:, :, :-1, :])
        grad_y_target = torch.abs(target_images[:, :, :, 1:] - target_images[:, :, :, :-1])

        perceptual_loss = F.mse_loss(grad_x_recon, grad_x_target) + F.mse_loss(grad_y_recon, grad_y_target)

        # Uncertainty regularization
        uncertainty_reg = torch.tensor(0.0, device=fmri_signals.device)
        if hasattr(self, 'semantic_uncertainty') and self.semantic_uncertainty is not None:
            # Encourage appropriate uncertainty levels
            uncertainty_reg = F.mse_loss(self.semantic_uncertainty, torch.ones_like(self.semantic_uncertainty) * 0.1)

        # Total loss
        total_loss = recon_loss + 0.1 * perceptual_loss + 0.01 * uncertainty_reg

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'perceptual_loss': perceptual_loss,
            'uncertainty_reg': uncertainty_reg
        }

def create_digit_captions_improved(labels):
    """Create improved digit captions with more variety."""
    templates = [
        "handwritten digit {}",
        "digit {} image",
        "number {} handwriting",
        "written digit {}",
        "digit {} pattern",
        "numeral {}",
        "figure {}",
        "{} symbol"
    ]

    digit_names = ['zero', 'one', 'two', 'three', 'four',
                   'five', 'six', 'seven', 'eight', 'nine']

    captions = []
    for i, label in enumerate(labels):
        template_idx = i % len(templates)
        digit_name = digit_names[label.item()]
        caption = templates[template_idx].format(digit_name)
        captions.append(caption)

    return captions

def tokenize_captions_improved(captions, max_length=77):
    """Improved caption tokenization."""
    vocab = {}
    vocab_size = 0

    # Build vocabulary
    for caption in captions:
        words = caption.lower().split()
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_size
                vocab_size += 1

    # Tokenize
    tokenized = []
    for caption in captions:
        words = caption.lower().split()
        tokens = [vocab.get(word, 0) for word in words]

        # Pad or truncate
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return torch.tensor(tokenized, dtype=torch.long)

# Export functions for compatibility
def create_digit_captions(labels):
    return create_digit_captions_improved(labels)

def tokenize_captions(captions):
    return tokenize_captions_improved(captions)
