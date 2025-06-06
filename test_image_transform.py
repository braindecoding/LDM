#!/usr/bin/env python3
"""
Test script untuk melihat efek transformasi gambar (flip dan rotasi -90 derajat)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path

def transform_image_for_display(img):
    """
    Transform image for better display: flip and rotate -90 degrees.
    
    Args:
        img: 2D numpy array (28x28)
    
    Returns:
        Transformed 2D numpy array for better visualization
    """
    # Flip vertically (upside down)
    img_flipped = np.flipud(img)
    
    # Rotate -90 degrees (counterclockwise)
    img_rotated = np.rot90(img_flipped, k=-1)
    
    return img_rotated

def main():
    """Test transformasi gambar."""
    print("🎨 Testing Image Transformation (Flip + Rotate -90°)")
    print("=" * 55)
    
    # Load data
    data_path = "data/digit69_28x28.mat"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    data = scipy.io.loadmat(data_path)
    stimuli = data['stimTrn'][:5]  # Take first 5 samples
    labels = data['labelTrn'][:5].flatten()
    
    print(f"✅ Loaded {len(stimuli)} test images")
    print(f"📊 Labels: {labels}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, len(stimuli), figsize=(3*len(stimuli), 6))
    
    for i in range(len(stimuli)):
        # Original image
        original = stimuli[i].reshape(28, 28)
        axes[0, i].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nDigit {labels[i]}', fontweight='bold')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Transformed image
        transformed = transform_image_for_display(original)
        axes[1, i].imshow(transformed, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Transformed\n(Flip + Rotate -90°)', fontweight='bold')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'BEFORE\nTransformation', 
                   transform=axes[0, 0].transAxes, rotation=90, 
                   fontsize=12, fontweight='bold', va='center', ha='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    axes[1, 0].text(-0.1, 0.5, 'AFTER\nTransformation', 
                   transform=axes[1, 0].transAxes, rotation=90, 
                   fontsize=12, fontweight='bold', va='center', ha='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('🔄 Image Transformation Test: Flip + Rotate -90°', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, top=0.9)
    
    # Save plot
    output_path = "results/image_transformation_test.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved transformation test to: {output_path}")
    
    plt.show()
    
    print("\n🎯 Transformation Details:")
    print("   1. np.flipud() - Flip image vertically (upside down)")
    print("   2. np.rot90(k=-1) - Rotate -90° counterclockwise")
    print("   3. Result: Better orientation for digit visualization")
    
    print(f"\n🎉 Transformation test complete!")
    print(f"📁 Check {output_path} to see the results")

if __name__ == "__main__":
    main()
