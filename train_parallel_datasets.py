#!/usr/bin/env python3
"""
Train Brain LDM on both datasets in parallel using multiprocessing.
WARNING: Requires sufficient GPU memory (24GB+ recommended).
"""

import os
import sys
import time
import multiprocessing as mp
import subprocess
from datetime import datetime

def run_training_job(job_info):
    """Run a single training job."""
    job_id, command, description = job_info
    
    print(f"🚀 [{job_id}] Starting: {description}")
    start_time = time.time()
    
    try:
        # Set environment
        env = os.environ.copy()
        env['PYTHONPATH'] = 'src'
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use same GPU but different memory
        
        # Run command
        result = subprocess.run(
            command.split(),
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ [{job_id}] Completed: {description} ({elapsed:.1f}s)")
        return (job_id, True, elapsed, result.stdout, result.stderr)
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ [{job_id}] Failed: {description} ({elapsed:.1f}s)")
        return (job_id, False, elapsed, e.stdout, e.stderr)

def check_gpu_memory():
    """Check available GPU memory."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        for line in lines:
            total, used = map(int, line.split(', '))
            available = total - used
            print(f"GPU Memory: {used}/{total} MB used, {available} MB available")
            
            if available < 16000:  # Need at least 16GB free
                print("⚠️  Warning: Low GPU memory for parallel training")
                print("   Recommended: 24GB+ total GPU memory")
                return False
        return True
        
    except Exception as e:
        print(f"❌ Could not check GPU memory: {e}")
        return False

def main():
    """Main parallel training pipeline."""
    
    print("🧠 Brain LDM: Parallel Training Both Datasets")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check GPU memory
    if not check_gpu_memory():
        print("❌ Insufficient GPU memory for parallel training")
        print("💡 Use train_both_datasets.py for sequential training instead")
        return 1
    
    # Check prerequisites
    if not os.path.exists('src/models/improved_brain_ldm.py'):
        print("❌ Error: Please run this script from the Brain-LDM root directory")
        return 1
    
    # Define parallel jobs (only optimized models to save memory)
    parallel_jobs = [
        (1, "python3 train_miyawaki_optimized.py", "Miyawaki Optimized Training"),
        (2, "python3 train_vangerven_optimized.py", "Vangerven Optimized Training"),
    ]
    
    # Run parallel training
    print("🚀 Starting parallel training...")
    print("⚠️  This will use significant GPU memory!")
    print()
    
    start_time = time.time()
    
    # Use multiprocessing pool
    with mp.Pool(processes=2) as pool:
        results = pool.map(run_training_job, parallel_jobs)
    
    # Process results
    elapsed = time.time() - start_time
    print(f"\n⏱️  Parallel training completed in {elapsed/60:.1f} minutes")
    
    # Run sequential evaluation and comparison
    print("\n📊 Running evaluation and comparison...")
    
    eval_jobs = [
        ("python3 evaluate_miyawaki_optimized.py", "Miyawaki Evaluation"),
        ("python3 evaluate_vangerven_optimized.py", "Vangerven Evaluation"),
        ("python3 final_optimization_comparison.py", "Final Comparison"),
    ]
    
    for command, description in eval_jobs:
        print(f"🔍 {description}...")
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = 'src'
            subprocess.run(command.split(), env=env, check=True)
            print(f"✅ {description} completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 PARALLEL TRAINING PIPELINE COMPLETED")
    print("=" * 60)
    
    successful = sum(1 for _, success, _, _, _ in results if success)
    print(f"Parallel jobs: {successful}/{len(results)} successful")
    
    for job_id, success, elapsed, stdout, stderr in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        job_name = parallel_jobs[job_id-1][2]
        print(f"  {status}: {job_name} ({elapsed:.1f}s)")
    
    return 0 if successful == len(results) else 1

if __name__ == "__main__":
    # Check if multiprocessing is supported
    if sys.platform.startswith('win'):
        print("⚠️  Parallel training may not work well on Windows")
        print("💡 Use train_both_datasets.py for sequential training instead")
    
    sys.exit(main())
