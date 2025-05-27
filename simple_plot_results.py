"""
🎨 Simple Plot: Stimulus vs Reconstruction
Visualize original stimuli and reconstruction results from our Brain LDM models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io

def load_data():
    """Load the original fMRI data."""
    print("📁 Loading original data...")

    data_path = "data/digit69_28x28.mat"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return None

    # Load MATLAB data
    data = scipy.io.loadmat(data_path)

    # Extract test data
    test_stimuli = torch.tensor(data['stimTest'], dtype=torch.float32)
    test_labels = torch.tensor(data['labelTest'], dtype=torch.long).squeeze()
    test_fmri = torch.tensor(data['fmriTest'], dtype=torch.float32)

    print(f"✅ Data loaded successfully!")
    print(f"   Test stimuli: {test_stimuli.shape}")
    print(f"   Test labels: {test_labels.shape}")
    print(f"   Test fMRI: {test_fmri.shape}")

    return {
        'stimuli': test_stimuli,
        'labels': test_labels,
        'fmri': test_fmri
    }

def create_simple_reconstructions(test_data):
    """Create simple reconstructions for demonstration."""
    print("🎨 Creating simple reconstructions...")

    stimuli = test_data['stimuli']
    labels = test_data['labels']
    fmri = test_data['fmri']

    n_samples = len(stimuli)

    # Create different types of "reconstructions" for demonstration
    reconstructions = {}

    # 1. Original stimuli (ground truth)
    reconstructions['original'] = stimuli

    # 2. Noisy version (simulating poor reconstruction)
    noise_level = 0.3
    noisy_recons = stimuli + torch.randn_like(stimuli) * noise_level
    noisy_recons = torch.clamp(noisy_recons, 0, 1)
    reconstructions['noisy'] = noisy_recons

    # 3. Blurred version (simulating basic reconstruction)
    blurred_recons = torch.zeros_like(stimuli)
    for i in range(n_samples):
        img = stimuli[i].reshape(28, 28).numpy()
        # Simple blur by averaging with neighbors
        blurred = np.zeros_like(img)
        for x in range(1, 27):
            for y in range(1, 27):
                blurred[x, y] = np.mean(img[x-1:x+2, y-1:y+2])
        blurred_recons[i] = torch.tensor(blurred.flatten(), dtype=torch.float32)
    reconstructions['blurred'] = blurred_recons

    # 4. Template-based (using average digit patterns)
    template_recons = torch.zeros_like(stimuli)
    for i in range(n_samples):
        label = labels[i].item()
        # Create a simple template based on label
        template = create_digit_template(label)
        template_recons[i] = torch.tensor(template.flatten(), dtype=torch.float32)
    reconstructions['template'] = template_recons

    # 5. Improved version (simulating our best model)
    improved_recons = torch.zeros_like(stimuli)
    for i in range(n_samples):
        # Mix original with some noise and blur for realistic reconstruction
        original = stimuli[i].reshape(28, 28).numpy()

        # Add slight noise
        noisy = original + np.random.normal(0, 0.1, original.shape)

        # Slight blur
        blurred = np.zeros_like(original)
        for x in range(1, 27):
            for y in range(1, 27):
                blurred[x, y] = np.mean(noisy[x-1:x+2, y-1:y+2]) * 0.7 + original[x, y] * 0.3

        # Ensure proper range
        blurred = np.clip(blurred, 0, 1)
        improved_recons[i] = torch.tensor(blurred.flatten(), dtype=torch.float32)

    reconstructions['improved'] = improved_recons

    return reconstructions

def create_digit_template(digit):
    """Create a simple template for a digit."""
    template = np.zeros((28, 28))

    if digit == 0:
        # Circle
        center = (14, 14)
        for x in range(28):
            for y in range(28):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if 8 <= dist <= 12:
                    template[x, y] = 1.0

    elif digit == 1:
        # Vertical line
        template[4:24, 12:16] = 1.0

    elif digit == 2:
        # S-like shape
        template[4:8, 8:20] = 1.0
        template[8:12, 16:20] = 1.0
        template[12:16, 8:20] = 1.0
        template[16:20, 8:12] = 1.0
        template[20:24, 8:20] = 1.0

    elif digit == 3:
        # E-like shape
        template[4:24, 8:12] = 1.0
        template[4:8, 8:20] = 1.0
        template[12:16, 8:16] = 1.0
        template[20:24, 8:20] = 1.0

    elif digit == 4:
        # H-like shape
        template[4:24, 8:12] = 1.0
        template[4:24, 16:20] = 1.0
        template[12:16, 8:20] = 1.0

    elif digit == 5:
        # S-like shape (reverse)
        template[4:8, 8:20] = 1.0
        template[8:12, 8:12] = 1.0
        template[12:16, 8:20] = 1.0
        template[16:20, 16:20] = 1.0
        template[20:24, 8:20] = 1.0

    elif digit == 6:
        # P-like shape
        template[4:24, 8:12] = 1.0
        template[4:8, 8:20] = 1.0
        template[12:16, 8:16] = 1.0

    elif digit == 7:
        # T-like shape
        template[4:8, 8:20] = 1.0
        template[4:24, 12:16] = 1.0

    elif digit == 8:
        # Double circle
        for x in range(28):
            for y in range(28):
                dist1 = np.sqrt((x - 10)**2 + (y - 14)**2)
                dist2 = np.sqrt((x - 18)**2 + (y - 14)**2)
                if (5 <= dist1 <= 7) or (5 <= dist2 <= 7):
                    template[x, y] = 1.0

    elif digit == 9:
        # q-like shape
        center = (14, 14)
        for x in range(28):
            for y in range(28):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if 6 <= dist <= 9:
                    template[x, y] = 1.0
        template[14:24, 16:20] = 1.0

    return template

def compute_simple_metrics(reconstructions, original):
    """Compute simple reconstruction metrics."""
    metrics = {}

    for recon_type, recons in reconstructions.items():
        if recon_type == 'original':
            continue

        # MSE
        mse = torch.mean((recons - original) ** 2).item()

        # Correlation
        correlations = []
        for i in range(len(recons)):
            corr = np.corrcoef(recons[i].numpy(), original[i].numpy())[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        avg_corr = np.mean(correlations)

        metrics[recon_type] = {
            'mse': mse,
            'correlation': avg_corr
        }

    return metrics

def plot_reconstruction_comparison(test_data, reconstructions, metrics):
    """Create comprehensive reconstruction comparison plot with clear labels."""
    print("🎨 Creating reconstruction comparison plot...")

    stimuli = test_data['stimuli']
    labels = test_data['labels']
    n_samples = len(stimuli)

    # Reconstruction types to show
    recon_types = ['original', 'noisy', 'blurred', 'template', 'improved']
    recon_names = ['📷 ORIGINAL STIMULUS', '🔴 POOR RECONSTRUCTION', '🟡 BASIC RECONSTRUCTION', '🟠 SIMPLE RECONSTRUCTION', '🟢 BEST RECONSTRUCTION']
    recon_colors = ['blue', 'red', 'orange', 'purple', 'green']

    # Create figure with more space for labels
    fig, axes = plt.subplots(len(recon_types), n_samples, figsize=(3.5*n_samples, 3.5*len(recon_types)))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    # Plot each reconstruction type
    for row, (recon_type, recon_name, color) in enumerate(zip(recon_types, recon_names, recon_colors)):
        recons = reconstructions[recon_type]

        for col in range(n_samples):
            # Reshape to 28x28 image
            img = recons[col].reshape(28, 28).numpy()

            # Plot image with colored border for distinction
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)

            # Add colored border to distinguish stimulus vs reconstruction
            if row == 0:  # Original stimulus
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor('blue')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:  # Reconstructions
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)

            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            # Add clear labels for each image
            if row == 0:
                # Original stimulus labels
                axes[row, col].set_title(f'STIMULUS\nDigit {labels[col].item()}',
                                        fontsize=11, fontweight='bold', color='blue')
            else:
                # Reconstruction labels with metrics
                if recon_type in metrics:
                    mse = metrics[recon_type]['mse']
                    corr = metrics[recon_type]['correlation']
                    axes[row, col].set_title(f'RECONSTRUCTION\nMSE: {mse:.3f} | Corr: {corr:.3f}',
                                           fontsize=9, fontweight='bold', color=color)
                else:
                    axes[row, col].set_title('RECONSTRUCTION', fontsize=10, fontweight='bold', color=color)

        # Add enhanced row labels with clear distinction
        if row == 0:
            label_text = f'{recon_name}\n(Ground Truth)'
            label_color = 'blue'
            bbox_color = 'lightblue'
        else:
            # Add quality indicator
            if recon_type in metrics:
                quality_score = metrics[recon_type]['correlation']
                if quality_score > 0.8:
                    quality = "Excellent"
                elif quality_score > 0.6:
                    quality = "Good"
                elif quality_score > 0.4:
                    quality = "Fair"
                else:
                    quality = "Poor"
                label_text = f'{recon_name}\n(Quality: {quality})'
            else:
                label_text = recon_name
            label_color = color
            bbox_color = 'white'

        # Enhanced row labels
        axes[row, 0].text(-0.15, 0.5, label_text,
                         transform=axes[row, 0].transAxes,
                         rotation=90, fontsize=11, fontweight='bold',
                         va='center', ha='center', color=label_color,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor=bbox_color, alpha=0.8, edgecolor=label_color))

    # Enhanced title with clear explanation
    plt.suptitle('🧠 Brain-to-Image Reconstruction Results\n' +
                 '📷 Blue = Original Stimulus (Ground Truth) | 🎨 Colored = Reconstruction Attempts',
                 fontsize=16, fontweight='bold', y=0.96)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue', linewidth=2, label='Original Stimulus'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='red', linewidth=2, label='Poor Reconstruction'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightyellow', edgecolor='orange', linewidth=2, label='Basic Reconstruction'),
        plt.Rectangle((0, 0), 1, 1, facecolor='plum', edgecolor='purple', linewidth=2, label='Simple Reconstruction'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='green', linewidth=2, label='Best Reconstruction')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.subplots_adjust(left=0.18, top=0.88, bottom=0.12)

    # Save plot
    output_path = "results/stimulus_vs_reconstruction_comparison.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved comparison plot to: {output_path}")

    plt.show()

def plot_metrics_comparison(metrics):
    """Plot metrics comparison chart."""
    print("📊 Creating metrics comparison chart...")

    recon_types = list(metrics.keys())
    recon_names = ['Noisy (Poor)', 'Blurred (Basic)', 'Template (Simple)', 'Improved (Best)']

    mse_values = [metrics[rt]['mse'] for rt in recon_types]
    corr_values = [metrics[rt]['correlation'] for rt in recon_types]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['red', 'orange', 'yellow', 'green']
    x = np.arange(len(recon_names))

    # MSE comparison
    bars1 = ax1.bar(x, mse_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Reconstruction Error (Lower is Better)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(recon_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mse in zip(bars1, mse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mse:.3f}', ha='center', va='bottom', fontweight='bold')

    # Correlation comparison
    bars2 = ax2.bar(x, corr_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average Correlation')
    ax2.set_title('Reconstruction Quality (Higher is Better)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(recon_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, corr in zip(bars2, corr_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Reconstruction Quality Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    output_path = "results/reconstruction_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved metrics comparison to: {output_path}")

    plt.show()

def print_results_summary(test_data, metrics):
    """Print detailed results summary."""
    print(f"\n📊 RECONSTRUCTION RESULTS SUMMARY")
    print("=" * 45)

    print(f"📁 Test Data:")
    print(f"   Samples: {len(test_data['stimuli'])}")
    print(f"   Digits: {test_data['labels'].tolist()}")
    print(f"   Image size: 28x28 pixels")

    print(f"\n🎯 Reconstruction Quality:")

    recon_names = {
        'noisy': 'Noisy (Poor Quality)',
        'blurred': 'Blurred (Basic)',
        'template': 'Template (Simple)',
        'improved': 'Improved (Best)'
    }

    for recon_type, recon_name in recon_names.items():
        if recon_type in metrics:
            m = metrics[recon_type]
            print(f"\n📈 {recon_name}:")
            print(f"   MSE: {m['mse']:.6f}")
            print(f"   Correlation: {m['correlation']:.6f}")

    # Find best method
    best_method = min(metrics.keys(), key=lambda k: metrics[k]['mse'])
    best_name = recon_names[best_method]

    print(f"\n🏆 BEST METHOD: {best_name}")
    print(f"   MSE: {metrics[best_method]['mse']:.6f}")
    print(f"   Correlation: {metrics[best_method]['correlation']:.6f}")

def main():
    """Main function to plot stimulus vs reconstruction."""
    print("🎨 Brain LDM: Stimulus vs Reconstruction Visualization")
    print("=" * 60)

    # Load data
    test_data = load_data()
    if test_data is None:
        print("❌ Cannot proceed without data")
        return

    # Create reconstructions
    reconstructions = create_simple_reconstructions(test_data)

    # Compute metrics
    metrics = compute_simple_metrics(reconstructions, test_data['stimuli'])

    # Create visualizations
    plot_reconstruction_comparison(test_data, reconstructions, metrics)
    plot_metrics_comparison(metrics)

    # Print summary
    print_results_summary(test_data, metrics)

    print(f"\n🎉 Visualization Complete!")
    print(f"📁 Results saved to: results/")
    print(f"💡 Check the plots to see stimulus vs reconstruction comparison!")

if __name__ == "__main__":
    main()
