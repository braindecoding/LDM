"""
📊 Create Comparison Charts for Brain LDM Results

Generate visual comparisons of metrics and performance analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def create_metrics_comparison():
    """Create metrics comparison chart."""
    
    # Our results
    our_results = {
        'PSNR': 5.49,
        'SSIM': 0.007,
        'Correlation': 0.020,
        'MSE': 0.282,
        'MAE': 0.518
    }
    
    # Typical benchmarks for brain decoding
    benchmarks = {
        'Excellent': {'PSNR': 30, 'SSIM': 0.8, 'Correlation': 0.7},
        'Good': {'PSNR': 25, 'SSIM': 0.6, 'Correlation': 0.5},
        'Fair': {'PSNR': 15, 'SSIM': 0.4, 'Correlation': 0.3},
        'Poor': {'PSNR': 10, 'SSIM': 0.2, 'Correlation': 0.1}
    }
    
    # Create comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR comparison
    categories = list(benchmarks.keys())
    psnr_values = [benchmarks[cat]['PSNR'] for cat in categories]
    
    bars1 = axes[0].bar(categories, psnr_values, alpha=0.7, color='skyblue', label='Benchmark')
    axes[0].axhline(y=our_results['PSNR'], color='red', linestyle='--', linewidth=2, label=f'Our Result ({our_results["PSNR"]:.1f} dB)')
    axes[0].set_title('PSNR Comparison')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM comparison
    ssim_values = [benchmarks[cat]['SSIM'] for cat in categories]
    
    bars2 = axes[1].bar(categories, ssim_values, alpha=0.7, color='lightgreen', label='Benchmark')
    axes[1].axhline(y=our_results['SSIM'], color='red', linestyle='--', linewidth=2, label=f'Our Result ({our_results["SSIM"]:.3f})')
    axes[1].set_title('SSIM Comparison')
    axes[1].set_ylabel('SSIM')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Correlation comparison
    corr_values = [benchmarks[cat]['Correlation'] for cat in categories]
    
    bars3 = axes[2].bar(categories, corr_values, alpha=0.7, color='lightcoral', label='Benchmark')
    axes[2].axhline(y=our_results['Correlation'], color='red', linestyle='--', linewidth=2, label=f'Our Result ({our_results["Correlation"]:.3f})')
    axes[2].set_title('Correlation Comparison')
    axes[2].set_ylabel('Correlation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Brain LDM Performance vs Benchmarks', fontsize=16)
    plt.tight_layout()
    
    # Save
    save_path = "results/comprehensive/metrics_comparison_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Metrics comparison chart saved to: {save_path}")
    return save_path


def create_performance_radar():
    """Create radar chart for performance visualization."""
    
    # Metrics (normalized to 0-1 scale)
    metrics = ['PSNR\n(norm)', 'SSIM', 'Correlation', 'MSE\n(inv)', 'MAE\n(inv)']
    
    # Our results (normalized)
    our_values = [
        5.49 / 30,      # PSNR normalized by 30 dB
        0.007,          # SSIM already 0-1
        0.020,          # Correlation already -1 to 1, but using 0-1
        1 - 0.282,      # MSE inverted (lower is better)
        1 - 0.518       # MAE inverted (lower is better)
    ]
    
    # Good benchmark (normalized)
    good_values = [
        25 / 30,        # PSNR: 25 dB
        0.6,            # SSIM: 0.6
        0.5,            # Correlation: 0.5
        1 - 0.1,        # MSE: ~0.1 (inverted)
        1 - 0.2         # MAE: ~0.2 (inverted)
    ]
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add first value to end to close the radar chart
    our_values += our_values[:1]
    good_values += good_values[:1]
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot our results
    ax.plot(angles, our_values, 'o-', linewidth=2, label='Our Brain LDM', color='red')
    ax.fill(angles, our_values, alpha=0.25, color='red')
    
    # Plot good benchmark
    ax.plot(angles, good_values, 'o-', linewidth=2, label='Good Benchmark', color='green')
    ax.fill(angles, good_values, alpha=0.25, color='green')
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True)
    
    # Add legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Brain LDM Performance Radar Chart', size=16, y=1.08)
    
    # Save
    save_path = "results/comprehensive/performance_radar_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Performance radar chart saved to: {save_path}")
    return save_path


def create_improvement_roadmap():
    """Create improvement roadmap visualization."""
    
    # Current vs target metrics
    metrics = ['PSNR (dB)', 'SSIM', 'Correlation']
    current = [5.49, 0.007, 0.020]
    target_fair = [15, 0.3, 0.3]
    target_good = [25, 0.6, 0.5]
    target_excellent = [30, 0.8, 0.7]
    
    x = np.arange(len(metrics))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - 1.5*width, current, width, label='Current', color='red', alpha=0.7)
    bars2 = ax.bar(x - 0.5*width, target_fair, width, label='Fair Target', color='orange', alpha=0.7)
    bars3 = ax.bar(x + 0.5*width, target_good, width, label='Good Target', color='yellow', alpha=0.7)
    bars4 = ax.bar(x + 1.5*width, target_excellent, width, label='Excellent Target', color='green', alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Brain LDM Improvement Roadmap')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    plt.tight_layout()
    
    # Save
    save_path = "results/comprehensive/improvement_roadmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Improvement roadmap saved to: {save_path}")
    return save_path


def main():
    """Create all comparison charts."""
    print("📊 Creating Comparison Charts")
    print("=" * 40)
    
    # Create results directory
    os.makedirs("results/comprehensive", exist_ok=True)
    
    # Create charts
    metrics_chart = create_metrics_comparison()
    radar_chart = create_performance_radar()
    roadmap_chart = create_improvement_roadmap()
    
    print(f"\n✅ All charts created successfully!")
    print(f"📁 Charts saved in: results/comprehensive/")
    print(f"  - {metrics_chart}")
    print(f"  - {radar_chart}")
    print(f"  - {roadmap_chart}")
    
    # Update summary
    summary_text = f"""
# 📊 Visual Analysis Summary

## Generated Charts:

1. **Metrics Comparison Chart**: `metrics_comparison_chart.png`
   - Compares PSNR, SSIM, and Correlation against benchmarks
   - Shows current performance vs quality thresholds

2. **Performance Radar Chart**: `performance_radar_chart.png`
   - Multi-dimensional performance visualization
   - Compares current results with good benchmark

3. **Improvement Roadmap**: `improvement_roadmap.png`
   - Shows progression targets from current to excellent performance
   - Clear visualization of improvement goals

## Key Insights:
- Current performance is significantly below fair quality thresholds
- All metrics indicate need for substantial improvements
- Clear targets established for incremental improvements
- Visual roadmap provides guidance for optimization efforts
"""
    
    with open("results/comprehensive/VISUAL_ANALYSIS.md", 'w') as f:
        f.write(summary_text)
    
    print(f"\n📝 Visual analysis summary: results/comprehensive/VISUAL_ANALYSIS.md")


if __name__ == "__main__":
    main()
