#!/usr/bin/env python3
"""
Quick comparison training for both Miyawaki and Vangerven datasets.
Trains only the optimized models and generates comparison.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def main():
    """Quick training and comparison."""
    
    print("⚡ Brain LDM: Quick Comparison Training")
    print("=" * 50)
    print("Training optimized models for both datasets...")
    print()
    
    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src'
    
    start_time = time.time()
    
    # Training sequence (optimized models only)
    commands = [
        ("python3 train_miyawaki_optimized.py", "🧠 Miyawaki Optimized"),
        ("python3 train_vangerven_optimized.py", "🔢 Vangerven Optimized"), 
        ("python3 final_optimization_comparison.py", "📊 Final Comparison"),
    ]
    
    for i, (command, description) in enumerate(commands, 1):
        print(f"[{i}/{len(commands)}] {description}")
        
        step_start = time.time()
        try:
            subprocess.run(command.split(), env=env, check=True)
            step_time = time.time() - step_start
            print(f"✅ Completed in {step_time:.1f}s\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}\n")
            return 1
    
    total_time = time.time() - start_time
    print("=" * 50)
    print(f"🎉 All training completed in {total_time/60:.1f} minutes!")
    print()
    print("📁 Check results:")
    print("  - results/final_optimization_comparison.png")
    print("  - checkpoints/best_*_optimized_model.pt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
