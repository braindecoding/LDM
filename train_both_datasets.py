#!/usr/bin/env python3
"""
Train Brain LDM on both Miyawaki and Vangerven datasets sequentially.
This script runs optimized training for both datasets and generates comparison results.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run command with PYTHONPATH=src
        env = os.environ.copy()
        env['PYTHONPATH'] = 'src'
        
        result = subprocess.run(
            command.split(),
            env=env,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ {description} completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False

def main():
    """Main training pipeline for both datasets."""
    
    print("🧠 Brain LDM: Training Both Datasets")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('src/models/improved_brain_ldm.py'):
        print("❌ Error: Please run this script from the Brain-LDM root directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Check if datasets exist
    datasets = [
        'data/miyawaki_structured_28x28.mat',
        'data/digit69_28x28.mat'
    ]
    
    for dataset in datasets:
        if not os.path.exists(dataset):
            print(f"❌ Error: Dataset not found: {dataset}")
            print("Please ensure datasets are in the data/ folder")
            sys.exit(1)
    
    print("✅ All datasets found")
    print("✅ Source code structure verified")
    print()
    
    # Training pipeline
    training_steps = [
        # Miyawaki training
        ("python3 train_miyawaki_simple.py", "Training Miyawaki Baseline Model"),
        ("python3 train_miyawaki_optimized.py", "Training Miyawaki Optimized Model"),
        ("python3 evaluate_miyawaki_optimized.py", "Evaluating Miyawaki Results"),
        
        # Vangerven training  
        ("python3 train_vangerven_simple.py", "Training Vangerven Baseline Model"),
        ("python3 train_vangerven_optimized.py", "Training Vangerven Optimized Model"),
        ("python3 evaluate_vangerven_optimized.py", "Evaluating Vangerven Results"),
        
        # Final comparison
        ("python3 final_optimization_comparison.py", "Generating Final Comparison"),
    ]
    
    # Track results
    results = []
    total_start_time = time.time()
    
    # Execute training steps
    for i, (command, description) in enumerate(training_steps, 1):
        print(f"\n📊 Step {i}/{len(training_steps)}")
        success = run_command(command, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠️  Step {i} failed, but continuing with remaining steps...")
            continue
    
    # Summary
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Results summary
    print("📊 RESULTS SUMMARY:")
    successful = 0
    for description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {description}")
        if success:
            successful += 1
    
    print(f"\nSuccess rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    # Check generated files
    print("\n📁 GENERATED FILES:")
    result_files = [
        'results/miyawaki_optimization_comparison.png',
        'results/vangerven_optimization_comparison.png', 
        'results/final_optimization_comparison.png',
        'checkpoints/best_miyawaki_optimized_model.pt',
        'checkpoints/best_vangerven_optimized_model.pt'
    ]
    
    for file_path in result_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✅ {file_path} ({size:.1f} KB)")
        else:
            print(f"  ❌ {file_path} (not found)")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Check results/ folder for visualizations")
    print("  2. Check checkpoints/ folder for trained models")
    print("  3. View results in Windows: \\\\wsl$\\Ubuntu-20.04\\home\\[user]\\Brain-LDM\\results\\")
    print()
    
    if successful == len(results):
        print("🎉 All training completed successfully!")
        return 0
    else:
        print("⚠️  Some steps failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
