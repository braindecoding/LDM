"""
Script untuk menganalisis data stimulus dan memahami struktur data yang benar.
Tujuan: Rekonstruksi stimulus visual dari data fMRI (brain decoding).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def analyze_stimulus_data():
    """Analyze stimulus data to understand the correct structure."""
    print("🔍 ANALYZING STIMULUS DATA STRUCTURE")
    print("=" * 60)
    
    # Check data folder
    data_dir = Path("data")
    print(f"📁 Data directory: {data_dir}")
    
    # Load digit data
    digit_file = data_dir / "digit69_28x28.mat"
    if digit_file.exists():
        print(f"\n📊 Loading: {digit_file}")
        digit_data = loadmat(str(digit_file))
        
        print("\n🔍 Contents of digit69_28x28.mat:")
        for key, value in digit_data.items():
            if not key.startswith('__'):
                print(f"   • {key}: {type(value)} - {getattr(value, 'shape', 'no shape')}")
                if hasattr(value, 'shape') and len(value.shape) <= 2:
                    print(f"     Range: [{np.min(value):.3f}, {np.max(value):.3f}]")
    
    # Check outputs folder for aligned data
    print(f"\n📁 Checking outputs folder...")
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        aligned_files = list(outputs_dir.glob("*aligned_data.npz"))
        print(f"   Found {len(aligned_files)} aligned data files:")
        
        for file in aligned_files:
            print(f"   📊 {file.name}")
            try:
                data = np.load(str(file))
                print(f"      Keys: {list(data.keys())}")
                for key in data.keys():
                    print(f"      • {key}: {data[key].shape}")
            except Exception as e:
                print(f"      Error loading: {e}")
    
    return digit_data if digit_file.exists() else None


def visualize_stimulus_examples(digit_data):
    """Visualize stimulus examples."""
    print(f"\n🎨 VISUALIZING STIMULUS EXAMPLES")
    print("=" * 40)
    
    # Find the main data array
    main_key = None
    for key, value in digit_data.items():
        if not key.startswith('__') and hasattr(value, 'shape'):
            if len(value.shape) == 2 and value.shape[1] == 784:  # 28x28 = 784
                main_key = key
                break
    
    if main_key is None:
        print("❌ Could not find 28x28 stimulus data")
        return
    
    stimuli = digit_data[main_key]
    print(f"✅ Found stimulus data: {main_key} with shape {stimuli.shape}")
    
    # Visualize first few stimuli
    num_examples = min(10, stimuli.shape[0])
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Stimulus Examples (28x28 Digits)', fontsize=16, fontweight='bold')
    
    for i in range(num_examples):
        row = i // 5
        col = i % 5
        
        # Reshape to 28x28
        stimulus_2d = stimuli[i].reshape(28, 28)
        
        axes[row, col].imshow(stimulus_2d, cmap='gray')
        axes[row, col].set_title(f'Stimulus {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    results_dir = Path("results/stimulus_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "stimulus_examples.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Stimulus examples saved: {save_path}")
    
    return stimuli


def analyze_fmri_stimulus_relationship():
    """Analyze the relationship between fMRI data and stimuli."""
    print(f"\n🧠 ANALYZING fMRI-STIMULUS RELATIONSHIP")
    print("=" * 50)
    
    # Load aligned fMRI data
    outputs_dir = Path("outputs")
    aligned_files = list(outputs_dir.glob("*aligned_data.npz"))
    
    if not aligned_files:
        print("❌ No aligned fMRI data found")
        return
    
    # Load first aligned file
    aligned_file = aligned_files[0]
    print(f"📊 Loading: {aligned_file.name}")
    
    fmri_data = np.load(str(aligned_file))
    print(f"   Keys: {list(fmri_data.keys())}")
    
    # Analyze structure
    total_timepoints = 0
    for key in fmri_data.keys():
        data = fmri_data[key]
        print(f"   • {key}: {data.shape}")
        if len(data.shape) == 2:
            total_timepoints += data.shape[0]
    
    print(f"\n📊 Total timepoints across all subjects: {total_timepoints}")
    
    # Load stimulus data
    digit_file = Path("data/digit69_28x28.mat")
    if digit_file.exists():
        digit_data = loadmat(str(digit_file))
        
        # Find stimulus data
        for key, value in digit_data.items():
            if not key.startswith('__') and hasattr(value, 'shape'):
                if len(value.shape) == 2:
                    print(f"📊 Stimulus data: {key} with shape {value.shape}")
                    
                    # Check if timepoints match
                    if value.shape[0] == total_timepoints:
                        print(f"✅ Perfect match: {value.shape[0]} stimuli = {total_timepoints} fMRI timepoints")
                    elif value.shape[0] < total_timepoints:
                        print(f"⚠️  Fewer stimuli ({value.shape[0]}) than fMRI timepoints ({total_timepoints})")
                        print(f"   Possible repetition or multiple runs")
                    else:
                        print(f"⚠️  More stimuli ({value.shape[0]}) than fMRI timepoints ({total_timepoints})")


def create_correct_pipeline_diagram():
    """Create diagram showing the correct LDM pipeline for stimulus reconstruction."""
    print(f"\n🎨 CREATING CORRECT PIPELINE DIAGRAM")
    print("=" * 45)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Pipeline steps
    steps = [
        "fMRI Data\n(Brain Activity)",
        "VAE Encoder\n(fMRI → Latent)",
        "Add Noise\n(Degradation)",
        "Diffusion Model\n(Denoising)",
        "VAE Decoder\n(Latent → Stimulus)",
        "Reconstructed\nStimulus"
    ]
    
    # Positions
    x_positions = np.linspace(0.1, 0.9, len(steps))
    y_position = 0.5
    
    # Draw boxes and arrows
    box_width = 0.12
    box_height = 0.15
    
    for i, (x, step) in enumerate(zip(x_positions, steps)):
        # Draw box
        if i == 0:
            color = 'lightblue'  # Input
        elif i == len(steps) - 1:
            color = 'lightgreen'  # Output
        elif 'Diffusion' in step:
            color = 'orange'  # Diffusion
        else:
            color = 'lightgray'  # Processing
        
        rect = plt.Rectangle((x - box_width/2, y_position - box_height/2), 
                           box_width, box_height, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y_position, step, ha='center', va='center', 
               fontsize=10, fontweight='bold', wrap=True)
        
        # Draw arrow to next step
        if i < len(steps) - 1:
            arrow_start = x + box_width/2
            arrow_end = x_positions[i+1] - box_width/2
            ax.annotate('', xy=(arrow_end, y_position), xytext=(arrow_start, y_position),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add title and labels
    ax.set_title('Correct Latent Diffusion Model Pipeline\nfor Stimulus Reconstruction from fMRI', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add explanation
    explanation = """
    Goal: Reconstruct visual stimulus from brain activity (fMRI)
    
    1. fMRI Data: Brain activation patterns when viewing stimuli
    2. VAE Encoder: Map fMRI patterns to latent space
    3. Add Noise: Simulate degradation for diffusion training
    4. Diffusion Model: Denoise and enhance latent representations
    5. VAE Decoder: Convert enhanced latents to stimulus images
    6. Reconstructed Stimulus: What the brain "saw"
    """
    
    ax.text(0.5, 0.15, explanation, ha='center', va='top', 
           fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save diagram
    results_dir = Path("results/stimulus_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "correct_ldm_pipeline.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Pipeline diagram saved: {save_path}")


def main():
    """Main analysis function."""
    print("🧠 STIMULUS RECONSTRUCTION ANALYSIS")
    print("Understanding the correct goal: Reconstruct visual stimuli from fMRI")
    print("=" * 80)
    
    # Analyze data structure
    digit_data = analyze_stimulus_data()
    
    if digit_data:
        # Visualize stimulus examples
        stimuli = visualize_stimulus_examples(digit_data)
    
    # Analyze fMRI-stimulus relationship
    analyze_fmri_stimulus_relationship()
    
    # Create correct pipeline diagram
    create_correct_pipeline_diagram()
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY: CORRECT UNDERSTANDING")
    print("=" * 80)
    print("🎯 Goal: Reconstruct STIMULUS from fMRI (not fMRI from fMRI)")
    print("📊 Input: fMRI brain activation data")
    print("🖼️  Output: Visual stimulus images (28x28 digits)")
    print("🧠 Task: Brain decoding / Neural decoding")
    print("🔬 Method: Latent Diffusion Model for enhanced reconstruction")
    print("\n✅ Analysis completed! Check results/stimulus_analysis/ for visualizations")


if __name__ == "__main__":
    main()
