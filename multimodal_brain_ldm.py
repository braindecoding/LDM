"""
🧠 Multi-Modal Brain LDM with Guidance
Implementation of Brain-Streams inspired framework with text/caption guidance and semantic embedding.

Key Features:
- Text/Caption guidance for semantic control
- Multi-modal embedding fusion
- Classifier-free guidance
- Semantic consistency loss
- Cross-modal attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    """Text encoder for caption/semantic guidance."""

    def __init__(self, embed_dim=512, max_length=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Use simple transformer for text encoding (avoiding CLIP dependency)
        self.vocab_size = 10000
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        self.clip_text_encoder = None
        self.clip_embed_dim = embed_dim

        # Project to desired dimension
        self.text_projection = nn.Linear(self.clip_embed_dim, embed_dim)

    def encode_text(self, text_tokens):
        """Encode text tokens to embeddings."""
        if self.clip_text_encoder is not None:
            # Use CLIP encoder
            with torch.no_grad():
                text_features = self.clip_text_encoder(text_tokens)
            text_features = self.text_projection(text_features.float())
        else:
            # Use simple transformer
            embedded = self.embedding(text_tokens)
            text_features = self.transformer(embedded)
            text_features = text_features.mean(dim=1)  # Global average pooling

        return text_features

class SemanticEmbedding(nn.Module):
    """Semantic embedding module for digit labels."""

    def __init__(self, num_classes=10, embed_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Learnable semantic embeddings for each digit
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)

        # Semantic consistency network
        self.semantic_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, class_labels):
        """Get semantic embeddings for class labels."""
        # Get base embeddings
        embeddings = self.class_embeddings(class_labels)

        # Apply semantic consistency network
        semantic_features = self.semantic_net(embeddings)

        return semantic_features

class CrossModalAttention(nn.Module):
    """Cross-modal attention between fMRI, text, and semantic features."""

    def __init__(self, fmri_dim=512, text_dim=512, semantic_dim=512, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Projection layers
        self.fmri_proj = nn.Linear(fmri_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, fmri_features, text_features, semantic_features):
        """Apply cross-modal attention and fusion."""
        batch_size = fmri_features.shape[0]

        # Project all modalities to same dimension
        fmri_proj = self.fmri_proj(fmri_features).unsqueeze(1)  # [B, 1, H]
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, H]
        semantic_proj = self.semantic_proj(semantic_features).unsqueeze(1)  # [B, 1, H]

        # Concatenate all modalities
        all_features = torch.cat([fmri_proj, text_proj, semantic_proj], dim=1)  # [B, 3, H]

        # Apply self-attention across modalities
        attended_features, attention_weights = self.multihead_attn(
            all_features, all_features, all_features
        )

        # Flatten and fuse
        attended_flat = attended_features.reshape(batch_size, -1)  # [B, 3*H]
        fused_features = self.fusion_net(attended_flat)

        return fused_features, attention_weights

class MultiModalBrainLDM(nn.Module):
    """Multi-modal Brain LDM with text and semantic guidance."""

    def __init__(self, fmri_dim=3092, image_size=28, latent_channels=4, latent_size=7,
                 text_embed_dim=512, semantic_embed_dim=512, guidance_scale=7.5):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.guidance_scale = guidance_scale

        # Core components
        self.text_encoder = TextEncoder(embed_dim=text_embed_dim)
        self.semantic_embedding = SemanticEmbedding(embed_dim=semantic_embed_dim)

        # fMRI encoder with improved architecture
        self.fmri_encoder = nn.Sequential(
            nn.Linear(fmri_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512)
        )

        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            fmri_dim=512,
            text_dim=text_embed_dim,
            semantic_dim=semantic_embed_dim,
            hidden_dim=512
        )

        # VAE components
        self.vae_encoder = self._build_vae_encoder()
        self.vae_decoder = self._build_vae_decoder()

        # U-Net for diffusion with conditioning
        self.unet = self._build_conditional_unet()

        # Guidance components
        self.guidance_projection = nn.Linear(512, latent_channels * latent_size * latent_size)

    def _build_vae_encoder(self):
        """Build VAE encoder with residual blocks."""
        return nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, self.latent_channels, 3, 1, 1),  # 7x7 -> 7x7
            nn.Tanh()
        )

    def _build_vae_decoder(self):
        """Build VAE decoder with residual blocks."""
        return nn.Sequential(
            nn.Conv2d(self.latent_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def _build_conditional_unet(self):
        """Build conditional U-Net for diffusion."""
        class ConditionalUNet(nn.Module):
            def __init__(self, in_channels=4, out_channels=4, condition_dim=512):
                super().__init__()

                # Condition projection
                self.condition_proj = nn.Sequential(
                    nn.Linear(condition_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, in_channels * 7 * 7)
                )

                # Encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels * 2, 64, 3, 1, 1),  # *2 for condition concat
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.ReLU(),
                )

                # Decoder
                self.decoder = nn.Sequential(
                    nn.Conv2d(128, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, out_channels, 3, 1, 1),
                )

            def forward(self, x, condition):
                # Project condition to spatial map
                cond_spatial = self.condition_proj(condition)
                cond_spatial = cond_spatial.view(-1, 4, 7, 7)

                # Concatenate input with condition
                x_cond = torch.cat([x, cond_spatial], dim=1)

                # Forward pass
                encoded = self.encoder(x_cond)
                output = self.decoder(encoded)

                return output

        return ConditionalUNet(condition_dim=512)

    def encode_multimodal_condition(self, fmri_signals, text_tokens=None, class_labels=None):
        """Encode multi-modal conditioning information."""
        batch_size = fmri_signals.shape[0]

        # Encode fMRI
        fmri_features = self.fmri_encoder(fmri_signals)

        # Encode text (if provided)
        if text_tokens is not None:
            text_features = self.text_encoder.encode_text(text_tokens)
        else:
            # Use dummy text features
            text_features = torch.zeros(batch_size, 512, device=fmri_signals.device)

        # Encode semantic labels (if provided)
        if class_labels is not None:
            semantic_features = self.semantic_embedding(class_labels)
        else:
            # Use dummy semantic features
            semantic_features = torch.zeros(batch_size, 512, device=fmri_signals.device)

        # Apply cross-modal attention
        fused_features, attention_weights = self.cross_modal_attention(
            fmri_features, text_features, semantic_features
        )

        return fused_features, attention_weights

    def encode_images(self, images):
        """Encode images to latent space."""
        if len(images.shape) == 2:
            images = images.view(-1, 1, self.image_size, self.image_size)

        latents = self.vae_encoder(images)
        return latents

    def decode_latents(self, latents):
        """Decode latents to images."""
        images = self.vae_decoder(latents)
        return images.view(-1, self.image_size * self.image_size)

    def forward_diffusion(self, latents, condition, timesteps=None):
        """Forward diffusion process with conditioning."""
        if timesteps is None:
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)

        # Add noise
        noise = torch.randn_like(latents)
        alpha = 0.99  # Simplified noise schedule
        noisy_latents = alpha * latents + (1 - alpha) * noise

        # Predict noise with conditioning
        predicted_noise = self.unet(noisy_latents, condition)

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)

        return {'loss': loss, 'predicted_noise': predicted_noise}

    def generate_with_guidance(self, fmri_signals, text_tokens=None, class_labels=None,
                              num_inference_steps=50, guidance_scale=None):
        """Generate images with multi-modal guidance."""
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        batch_size = fmri_signals.shape[0]
        device = fmri_signals.device

        # Encode conditioning
        condition, attention_weights = self.encode_multimodal_condition(
            fmri_signals, text_tokens, class_labels
        )

        # Initialize random latents
        latents = torch.randn(batch_size, self.latent_channels,
                             self.latent_size, self.latent_size, device=device)

        # Denoising loop with classifier-free guidance
        for t in range(num_inference_steps):
            # Unconditional prediction (for classifier-free guidance)
            uncond_condition = torch.zeros_like(condition)
            uncond_noise = self.unet(latents, uncond_condition)

            # Conditional prediction
            cond_noise = self.unet(latents, condition)

            # Apply classifier-free guidance
            guided_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

            # Update latents (simplified DDPM step)
            alpha = 0.99
            latents = (latents - (1 - alpha) * guided_noise) / alpha

        # Decode to images
        generated_images = self.decode_latents(latents)

        return generated_images, attention_weights

    def compute_multimodal_loss(self, fmri_signals, target_images, text_tokens=None,
                               class_labels=None):
        """Compute multi-modal training loss."""
        # Encode target images
        target_latents = self.encode_images(target_images)

        # Encode conditioning
        condition, attention_weights = self.encode_multimodal_condition(
            fmri_signals, text_tokens, class_labels
        )

        # Forward diffusion
        diffusion_output = self.forward_diffusion(target_latents, condition)

        # Generate reconstruction for additional losses
        with torch.no_grad():
            reconstructed, _ = self.generate_with_guidance(
                fmri_signals, text_tokens, class_labels, num_inference_steps=10
            )

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, target_images)

        # Semantic consistency loss (if labels provided)
        semantic_loss = 0
        if class_labels is not None:
            # Encourage semantic embeddings to be consistent
            semantic_features = self.semantic_embedding(class_labels)
            semantic_consistency = F.cosine_similarity(
                condition, semantic_features, dim=1
            ).mean()
            semantic_loss = 1 - semantic_consistency

        # Total loss
        total_loss = (diffusion_output['loss'] +
                     0.1 * recon_loss +
                     0.05 * semantic_loss)

        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_output['loss'],
            'recon_loss': recon_loss,
            'semantic_loss': semantic_loss,
            'attention_weights': attention_weights
        }

def create_digit_captions(labels):
    """Create text captions for digit labels."""
    digit_names = ['zero', 'one', 'two', 'three', 'four',
                   'five', 'six', 'seven', 'eight', 'nine']

    captions = []
    for label in labels:
        digit_name = digit_names[label.item()]
        caption = f"handwritten digit {digit_name}"
        captions.append(caption)

    return captions

def tokenize_captions(captions, max_length=77):
    """Tokenize captions for text encoder."""
    # Simple tokenization (in practice, use proper tokenizer)
    vocab = {'handwritten': 1, 'digit': 2, 'zero': 3, 'one': 4, 'two': 5,
             'three': 6, 'four': 7, 'five': 8, 'six': 9, 'seven': 10,
             'eight': 11, 'nine': 12}

    tokens = []
    for caption in captions:
        words = caption.split()
        token_ids = [vocab.get(word, 0) for word in words]
        # Pad to max_length
        token_ids = token_ids[:max_length]
        token_ids += [0] * (max_length - len(token_ids))
        tokens.append(token_ids)

    return torch.tensor(tokens, dtype=torch.long)
