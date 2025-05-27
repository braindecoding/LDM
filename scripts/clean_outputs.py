"""
Clean utility script for managing project outputs and logs.
"""

import shutil
import argparse
from pathlib import Path
from typing import List


class OutputCleaner:
    """Clean utility for managing project outputs."""
    
    def __init__(self, project_root: Path = None):
        """Initialize cleaner with project root."""
        self.project_root = project_root or Path(__file__).parent.parent
        
    def clean_logs(self, keep_latest: int = 1) -> None:
        """Clean old log files, keeping only the latest ones."""
        logs_dir = self.project_root / "logs"
        
        if not logs_dir.exists():
            print("📁 No logs directory found")
            return
        
        print(f"🧹 Cleaning logs directory (keeping latest {keep_latest})...")
        
        # Clean evaluation metrics
        eval_files = sorted(logs_dir.glob("evaluation_metrics_*.json"), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        for file in eval_files[keep_latest:]:
            file.unlink()
            print(f"   🗑️  Removed: {file.name}")
        
        # Clean training history
        history_files = sorted(logs_dir.glob("training_history_*.json"),
                              key=lambda x: x.stat().st_mtime, reverse=True)
        
        for file in history_files[keep_latest:]:
            file.unlink()
            print(f"   🗑️  Removed: {file.name}")
        
        # Clean old visualization directories
        viz_dirs = sorted(logs_dir.glob("visualizations_*"),
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        for viz_dir in viz_dirs[keep_latest:]:
            shutil.rmtree(viz_dir)
            print(f"   🗑️  Removed: {viz_dir.name}/")
        
        print(f"✅ Logs cleaned (kept {min(len(eval_files), keep_latest)} latest)")
    
    def clean_checkpoints(self, keep_latest: int = 2) -> None:
        """Clean old checkpoint files."""
        checkpoints_dir = self.project_root / "checkpoints"
        
        if not checkpoints_dir.exists():
            print("📁 No checkpoints directory found")
            return
        
        print(f"🧹 Cleaning checkpoints (keeping latest {keep_latest})...")
        
        # Keep best_model.pt always
        checkpoint_files = [f for f in checkpoints_dir.glob("checkpoint_epoch_*.pt")]
        checkpoint_files = sorted(checkpoint_files, 
                                 key=lambda x: x.stat().st_mtime, reverse=True)
        
        for file in checkpoint_files[keep_latest:]:
            file.unlink()
            print(f"   🗑️  Removed: {file.name}")
        
        print(f"✅ Checkpoints cleaned (kept {min(len(checkpoint_files), keep_latest)} latest)")
    
    def clean_results(self) -> None:
        """Clean results directory."""
        results_dir = self.project_root / "results"
        
        if not results_dir.exists():
            print("📁 No results directory found")
            return
        
        print("🧹 Cleaning results directory...")
        
        for file in results_dir.glob("*"):
            if file.is_file():
                file.unlink()
                print(f"   🗑️  Removed: {file.name}")
            elif file.is_dir():
                shutil.rmtree(file)
                print(f"   🗑️  Removed: {file.name}/")
        
        print("✅ Results directory cleaned")
    
    def clean_temp_files(self) -> None:
        """Clean temporary files and caches."""
        print("🧹 Cleaning temporary files...")
        
        # Python cache
        cache_dirs = list(self.project_root.rglob("__pycache__"))
        for cache_dir in cache_dirs:
            shutil.rmtree(cache_dir)
            print(f"   🗑️  Removed: {cache_dir.relative_to(self.project_root)}")
        
        # Log files
        log_files = list(self.project_root.glob("*.log"))
        for log_file in log_files:
            log_file.unlink()
            print(f"   🗑️  Removed: {log_file.name}")
        
        print("✅ Temporary files cleaned")
    
    def show_disk_usage(self) -> None:
        """Show disk usage of project directories."""
        print("💾 Disk Usage:")
        print("-" * 40)
        
        directories = ["logs", "checkpoints", "results", "outputs"]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                file_count = len(list(dir_path.rglob('*')))
                print(f"   📁 {dir_name}/: {size_mb:.1f} MB ({file_count} files)")
            else:
                print(f"   📁 {dir_name}/: Not found")
    
    def list_outputs(self) -> None:
        """List current outputs."""
        print("📋 Current Outputs:")
        print("-" * 40)
        
        # Logs
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            eval_files = list(logs_dir.glob("evaluation_metrics_*.json"))
            viz_dirs = list(logs_dir.glob("visualizations_*"))
            print(f"   📊 Evaluation files: {len(eval_files)}")
            print(f"   🎨 Visualization dirs: {len(viz_dirs)}")
        
        # Checkpoints
        checkpoints_dir = self.project_root / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob("*.pt"))
            print(f"   💾 Checkpoint files: {len(checkpoint_files)}")
        
        # Results
        results_dir = self.project_root / "results"
        if results_dir.exists():
            result_files = list(results_dir.glob("*"))
            print(f"   📈 Result files: {len(result_files)}")


def main():
    """Main cleaning function."""
    parser = argparse.ArgumentParser(description="Clean project outputs")
    parser.add_argument("--logs", action="store_true", help="Clean old logs")
    parser.add_argument("--checkpoints", action="store_true", help="Clean old checkpoints")
    parser.add_argument("--results", action="store_true", help="Clean results")
    parser.add_argument("--temp", action="store_true", help="Clean temporary files")
    parser.add_argument("--all", action="store_true", help="Clean everything")
    parser.add_argument("--keep", type=int, default=2, help="Number of latest files to keep")
    parser.add_argument("--list", action="store_true", help="List current outputs")
    parser.add_argument("--usage", action="store_true", help="Show disk usage")
    
    args = parser.parse_args()
    
    cleaner = OutputCleaner()
    
    if args.list:
        cleaner.list_outputs()
        return
    
    if args.usage:
        cleaner.show_disk_usage()
        return
    
    if not any([args.logs, args.checkpoints, args.results, args.temp, args.all]):
        print("🧹 PROJECT OUTPUT CLEANER")
        print("=" * 40)
        cleaner.list_outputs()
        print()
        cleaner.show_disk_usage()
        print("\n💡 Use --help to see cleaning options")
        return
    
    print("🧹 CLEANING PROJECT OUTPUTS")
    print("=" * 40)
    
    if args.all or args.logs:
        cleaner.clean_logs(keep_latest=args.keep)
    
    if args.all or args.checkpoints:
        cleaner.clean_checkpoints(keep_latest=args.keep)
    
    if args.all or args.results:
        cleaner.clean_results()
    
    if args.all or args.temp:
        cleaner.clean_temp_files()
    
    print("\n✅ Cleaning completed!")


if __name__ == "__main__":
    main()
