"""
📊 Comprehensive Evaluation: Brain LDM

Advanced evaluation with PSNR, SSIM, FID, LPIPS, and CLIP score.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Try to import advanced metrics
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ scikit-image not available. Installing simplified versions.")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available. Will skip CLIP score.")

from data_loader import load_fmri_data
from brain_ldm import create_brain_ldm


class ComprehensiveEvaluator:
    """Comprehensive evaluator with advanced metrics."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """Initialize evaluator with trained model."""
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print(f"📁 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model
        model_config = checkpoint['model_config']
        self.model = create_brain_ldm(**model_config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load data
        self.data_loader = load_fmri_data(device=device)
        
        # Load CLIP if available
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                print("✅ CLIP model loaded")
            except:
                self.clip_model = None
                print("⚠️ CLIP model failed to load")
        else:
            self.clip_model = None
        
        print(f"✅ Evaluator initialized")
    
    def compute_psnr(self, true_images: np.ndarray, pred_images: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        if SKIMAGE_AVAILABLE:
            psnr_values = []
            for i in range(len(true_images)):
                # Ensure images are in [0, 1] range
                true_img = np.clip(true_images[i], 0, 1)
                pred_img = np.clip(pred_images[i], 0, 1)
                
                psnr_val = psnr(true_img, pred_img, data_range=1.0)
                if not np.isnan(psnr_val) and not np.isinf(psnr_val):
                    psnr_values.append(psnr_val)
            
            return np.mean(psnr_values) if psnr_values else 0.0
        else:
            # Simplified PSNR calculation
            mse = np.mean((true_images - pred_images) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def compute_ssim(self, true_images: np.ndarray, pred_images: np.ndarray) -> float:
        """Compute Structural Similarity Index."""
        if SKIMAGE_AVAILABLE:
            ssim_values = []
            for i in range(len(true_images)):
                # Ensure images are in [0, 1] range
                true_img = np.clip(true_images[i], 0, 1)
                pred_img = np.clip(pred_images[i], 0, 1)
                
                ssim_val = ssim(true_img, pred_img, data_range=1.0)
                if not np.isnan(ssim_val):
                    ssim_values.append(ssim_val)
            
            return np.mean(ssim_values) if ssim_values else 0.0
        else:
            # Simplified SSIM calculation
            def simple_ssim(img1, img2):
                mu1, mu2 = img1.mean(), img2.mean()
                sigma1, sigma2 = img1.std(), img2.std()
                sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
                
                c1, c2 = 0.01**2, 0.03**2
                
                numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
                denominator = (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
                
                return numerator / denominator if denominator != 0 else 0.0
            
            ssim_values = []
            for i in range(len(true_images)):
                ssim_val = simple_ssim(true_images[i], pred_images[i])
                if not np.isnan(ssim_val):
                    ssim_values.append(ssim_val)
            
            return np.mean(ssim_values) if ssim_values else 0.0
    
    def compute_fid(self, true_images: np.ndarray, pred_images: np.ndarray) -> float:
        """Compute Fréchet Inception Distance (simplified)."""
        # For simplicity, we'll compute a feature-based distance
        # In practice, you'd use a pre-trained Inception network
        
        # Convert to tensors
        true_tensor = torch.from_numpy(true_images).float()
        pred_tensor = torch.from_numpy(pred_images).float()
        
        # Simple feature extraction (mean and std of pixel values)
        true_features = torch.stack([
            true_tensor.mean(dim=(1,2)),
            true_tensor.std(dim=(1,2))
        ], dim=1)
        
        pred_features = torch.stack([
            pred_tensor.mean(dim=(1,2)),
            pred_tensor.std(dim=(1,2))
        ], dim=1)
        
        # Compute means and covariances
        mu1, mu2 = true_features.mean(dim=0), pred_features.mean(dim=0)
        sigma1 = torch.cov(true_features.T)
        sigma2 = torch.cov(pred_features.T)
        
        # Simplified FID calculation
        diff = mu1 - mu2
        fid = torch.sum(diff * diff) + torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(sigma1 @ sigma2))
        
        return fid.item()
    
    def compute_lpips(self, true_images: np.ndarray, pred_images: np.ndarray) -> float:
        """Compute LPIPS (simplified perceptual distance)."""
        # Simplified perceptual distance using basic features
        # In practice, you'd use a pre-trained network like VGG
        
        # Convert to tensors and add channel dimension if needed
        true_tensor = torch.from_numpy(true_images).float()
        pred_tensor = torch.from_numpy(pred_images).float()
        
        if len(true_tensor.shape) == 3:
            true_tensor = true_tensor.unsqueeze(1)  # Add channel dim
            pred_tensor = pred_tensor.unsqueeze(1)
        
        # Simple gradient-based perceptual distance
        def compute_gradients(img):
            grad_x = F.conv2d(img, torch.tensor([[[[-1, 1]]]]).float(), padding=1)
            grad_y = F.conv2d(img, torch.tensor([[[[-1], [1]]]]).float(), padding=1)
            return torch.sqrt(grad_x**2 + grad_y**2)
        
        true_grads = compute_gradients(true_tensor)
        pred_grads = compute_gradients(pred_tensor)
        
        # Compute perceptual distance
        lpips_distance = F.mse_loss(true_grads, pred_grads)
        
        return lpips_distance.item()
    
    def compute_clip_score(self, true_images: np.ndarray, pred_images: np.ndarray) -> float:
        """Compute CLIP score (if available)."""
        if not self.clip_model:
            return 0.0
        
        try:
            # Convert images to PIL format for CLIP
            import PIL.Image as Image
            
            clip_scores = []
            
            for i in range(min(len(true_images), 5)):  # Limit to 5 samples for speed
                # Convert to PIL Images
                true_img = Image.fromarray((true_images[i] * 255).astype(np.uint8))
                pred_img = Image.fromarray((pred_images[i] * 255).astype(np.uint8))
                
                # Preprocess for CLIP
                true_input = self.clip_preprocess(true_img).unsqueeze(0).to(self.device)
                pred_input = self.clip_preprocess(pred_img).unsqueeze(0).to(self.device)
                
                # Get CLIP features
                with torch.no_grad():
                    true_features = self.clip_model.encode_image(true_input)
                    pred_features = self.clip_model.encode_image(pred_input)
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(true_features, pred_features)
                clip_scores.append(similarity.item())
            
            return np.mean(clip_scores) if clip_scores else 0.0
            
        except Exception as e:
            print(f"⚠️ CLIP score computation failed: {e}")
            return 0.0
    
    def evaluate_comprehensive(self, num_samples: int = 10) -> dict:
        """Run comprehensive evaluation."""
        print(f"\n📊 Comprehensive Evaluation")
        print("=" * 40)
        
        # Get test data
        test_loader = self.data_loader.create_dataloader('test', batch_size=4, shuffle=False)
        
        all_true_images = []
        all_pred_images = []
        
        print(f"🔄 Generating reconstructions...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                fmri_signals = batch['fmri'].to(self.device)
                true_stimuli = batch['stimulus'].to(self.device)
                
                # Generate reconstructions
                generated_images = self.model.generate_from_fmri(
                    fmri_signals, 
                    num_inference_steps=50
                )
                
                # Convert to numpy
                true_imgs = true_stimuli.view(-1, 28, 28).cpu().numpy()
                pred_imgs = generated_images[:, 0].cpu().numpy()
                
                all_true_images.append(true_imgs)
                all_pred_images.append(pred_imgs)
                
                if len(all_true_images) * len(true_imgs) >= num_samples:
                    break
        
        # Concatenate results
        true_images = np.concatenate(all_true_images, axis=0)[:num_samples]
        pred_images = np.concatenate(all_pred_images, axis=0)[:num_samples]
        
        print(f"📈 Computing metrics for {len(true_images)} samples...")
        
        # Compute all metrics
        results = {}
        
        print(f"  🔍 Computing PSNR...")
        results['psnr'] = self.compute_psnr(true_images, pred_images)
        
        print(f"  🔍 Computing SSIM...")
        results['ssim'] = self.compute_ssim(true_images, pred_images)
        
        print(f"  🔍 Computing FID...")
        results['fid'] = self.compute_fid(true_images, pred_images)
        
        print(f"  🔍 Computing LPIPS...")
        results['lpips'] = self.compute_lpips(true_images, pred_images)
        
        print(f"  🔍 Computing CLIP Score...")
        results['clip_score'] = self.compute_clip_score(true_images, pred_images)
        
        # Basic metrics
        mse = np.mean((true_images - pred_images) ** 2)
        mae = np.mean(np.abs(true_images - pred_images))
        
        results.update({
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'num_samples': len(true_images),
            'true_images': true_images,
            'pred_images': pred_images
        })
        
        return results


def main():
    """Main evaluation function."""
    print("📊 Comprehensive Brain LDM Evaluation")
    print("=" * 50)
    
    # Check if model exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"❌ Model not found: {checkpoint_path}")
        print(f"💡 Please train the model first: python train_brain_ldm.py")
        return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    try:
        # Create evaluator
        evaluator = ComprehensiveEvaluator(checkpoint_path, device=device)
        
        # Run evaluation
        results = evaluator.evaluate_comprehensive(num_samples=10)
        
        # Print results
        print(f"\n📊 Comprehensive Evaluation Results")
        print("=" * 50)
        print(f"🎯 Image Quality Metrics:")
        print(f"  PSNR: {results['psnr']:.4f} dB")
        print(f"  SSIM: {results['ssim']:.4f}")
        print(f"  FID: {results['fid']:.4f}")
        print(f"  LPIPS: {results['lpips']:.4f}")
        print(f"  CLIP Score: {results['clip_score']:.4f}")
        
        print(f"\n📈 Basic Metrics:")
        print(f"  MSE: {results['mse']:.6f}")
        print(f"  MAE: {results['mae']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
        
        print(f"\n📋 Evaluation Info:")
        print(f"  Samples: {results['num_samples']}")
        print(f"  Image size: 28×28")
        print(f"  Model: Brain LDM")
        
        # Save detailed results
        os.makedirs("results/comprehensive", exist_ok=True)
        
        results_text = f"""Comprehensive Brain LDM Evaluation Results
================================================

Image Quality Metrics:
  PSNR: {results['psnr']:.4f} dB
  SSIM: {results['ssim']:.4f}
  FID: {results['fid']:.4f}
  LPIPS: {results['lpips']:.4f}
  CLIP Score: {results['clip_score']:.4f}

Basic Metrics:
  MSE: {results['mse']:.6f}
  MAE: {results['mae']:.6f}
  RMSE: {results['rmse']:.6f}

Evaluation Info:
  Samples: {results['num_samples']}
  Image size: 28×28
  Model: Brain LDM

Metric Interpretations:
  PSNR: Higher is better (>20 dB is good)
  SSIM: Higher is better (0-1 scale, >0.5 is good)
  FID: Lower is better (<50 is good)
  LPIPS: Lower is better (<0.5 is good)
  CLIP Score: Higher is better (0-1 scale)
"""
        
        with open("results/comprehensive/detailed_metrics.txt", 'w') as f:
            f.write(results_text)
        
        print(f"\n💾 Detailed results saved to: results/comprehensive/detailed_metrics.txt")
        print(f"✅ Comprehensive evaluation completed!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
