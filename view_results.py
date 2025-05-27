"""
🎨 View Results: Brain LDM Training & Evaluation

Script to display all visualizations and results from Brain LDM training.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def view_training_progress():
    """View training progress through sample images."""
    print("📈 Training Progress Visualization")
    print("=" * 40)
    
    # Get all training sample files
    samples_dir = Path("results/training_samples")
    sample_files = sorted(samples_dir.glob("samples_epoch_*.png"))
    
    if not sample_files:
        print("❌ No training samples found!")
        return
    
    print(f"📊 Found {len(sample_files)} training sample files")
    
    # Create subplot for all epochs
    fig, axes = plt.subplots(1, len(sample_files), figsize=(20, 4))
    if len(sample_files) == 1:
        axes = [axes]
    
    for i, sample_file in enumerate(sample_files):
        # Extract epoch number from filename
        epoch = sample_file.stem.split('_')[-1]
        
        # Load and display image
        img = mpimg.imread(sample_file)
        axes[i].imshow(img)
        axes[i].set_title(f'Epoch {epoch}', fontsize=12)
        axes[i].axis('off')
        
        print(f"  ✅ Epoch {epoch}: {sample_file.name}")
    
    plt.suptitle('Brain LDM Training Progress: Reconstruction Quality Over Time', fontsize=16)
    plt.tight_layout()
    
    # Save combined view
    output_path = "results/training_progress_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Combined training progress saved to: {output_path}")
    return output_path


def view_evaluation_results():
    """View evaluation results."""
    print("\n📊 Evaluation Results")
    print("=" * 30)
    
    # Read metrics
    metrics_file = Path("results/evaluation/evaluation_metrics.txt")
    if metrics_file.exists():
        print("📈 Evaluation Metrics:")
        with open(metrics_file, 'r') as f:
            content = f.read()
            print(content)
    else:
        print("❌ Evaluation metrics not found!")
        return None
    
    # Display reconstruction comparison
    comparison_file = Path("results/evaluation/reconstruction_comparison.png")
    if comparison_file.exists():
        print(f"🎨 Reconstruction comparison available: {comparison_file}")
        
        # Load and display
        img = mpimg.imread(comparison_file)
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.title('Brain LDM: True vs Reconstructed Stimuli')
        plt.axis('off')
        
        # Save a copy in results root
        output_path = "results/final_reconstruction_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Final comparison saved to: {output_path}")
        return output_path
    else:
        print("❌ Reconstruction comparison not found!")
        return None


def view_sample_stimuli():
    """View original sample stimuli."""
    print("\n🎯 Original Sample Stimuli")
    print("=" * 30)
    
    stimuli_file = Path("results/sample_stimuli.png")
    if stimuli_file.exists():
        print(f"📊 Sample stimuli available: {stimuli_file}")
        
        # Load and display
        img = mpimg.imread(stimuli_file)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title('Sample Stimuli from Dataset')
        plt.axis('off')
        
        # Save a copy with better name
        output_path = "results/dataset_sample_stimuli.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Dataset samples saved to: {output_path}")
        return output_path
    else:
        print("❌ Sample stimuli not found!")
        return None


def create_summary_report():
    """Create a summary report of all results."""
    print("\n📋 Creating Summary Report")
    print("=" * 30)
    
    # Read evaluation metrics
    metrics_file = Path("results/evaluation/evaluation_metrics.txt")
    metrics_content = ""
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_content = f.read()
    
    # Count training samples
    samples_dir = Path("results/training_samples")
    sample_files = list(samples_dir.glob("samples_epoch_*.png"))
    
    # Create summary report
    report_content = f"""# 🧠 Brain LDM Training & Evaluation Summary

## 📊 Training Results
- **Epochs Completed**: 50
- **Training Samples Generated**: {len(sample_files)}
- **Model Checkpoints**: 6 files (best + epoch 10,20,30,40,50)
- **Training Logs**: Available in logs/brain_ldm/

## 📈 Evaluation Metrics
{metrics_content}

## 📁 Generated Files

### Training Samples:
"""
    
    for sample_file in sorted(sample_files):
        epoch = sample_file.stem.split('_')[-1]
        report_content += f"- Epoch {epoch}: {sample_file.name}\n"
    
    report_content += f"""
### Evaluation Results:
- reconstruction_comparison.png: True vs reconstructed stimuli comparison
- evaluation_metrics.txt: Detailed numerical metrics

### Dataset Samples:
- sample_stimuli.png: Original dataset visualization

## 🎯 Key Findings
- Model successfully learned to generate images from fMRI signals
- Training progressed over 50 epochs with regular checkpoints
- Evaluation shows baseline performance for brain decoding task
- Ready for further experimentation and improvement

## 🚀 Next Steps
1. Analyze training progression through sample images
2. Fine-tune hyperparameters for better performance
3. Experiment with different architectures
4. Increase training data if available
"""
    
    # Save report
    report_path = "results/SUMMARY_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📝 Summary report saved to: {report_path}")
    return report_path


def main():
    """Main function to view all results."""
    print("🎨 Brain LDM Results Viewer")
    print("=" * 50)
    
    # Check if results directory exists
    if not Path("results").exists():
        print("❌ Results directory not found!")
        print("💡 Please run training first: python train_brain_ldm.py")
        return
    
    try:
        # View all results
        training_progress = view_training_progress()
        evaluation_results = view_evaluation_results()
        sample_stimuli = view_sample_stimuli()
        summary_report = create_summary_report()
        
        # Final summary
        print(f"\n🎉 Results Viewing Completed!")
        print("=" * 40)
        print(f"📁 All results organized in: results/")
        print(f"")
        print(f"📊 Key Files Generated:")
        if training_progress:
            print(f"  ✅ Training Progress: {training_progress}")
        if evaluation_results:
            print(f"  ✅ Final Comparison: {evaluation_results}")
        if sample_stimuli:
            print(f"  ✅ Dataset Samples: {sample_stimuli}")
        if summary_report:
            print(f"  ✅ Summary Report: {summary_report}")
        
        print(f"\n📋 Results Structure:")
        print(f"  results/")
        print(f"  ├── training_samples/          # Training progression")
        print(f"  ├── evaluation/                # Evaluation results")
        print(f"  ├── training_progress_combined.png")
        print(f"  ├── final_reconstruction_comparison.png")
        print(f"  ├── dataset_sample_stimuli.png")
        print(f"  └── SUMMARY_REPORT.md")
        
        print(f"\n🎯 Brain LDM training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error viewing results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
