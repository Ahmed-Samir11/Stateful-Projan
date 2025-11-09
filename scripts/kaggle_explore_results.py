"""
Quick file explorer for Kaggle experiment results
"""

from pathlib import Path
import json

def explore_directory(path, indent=0):
    """Recursively explore and display directory contents"""
    prefix = "  " * indent
    
    try:
        path = Path(path)
        if not path.exists():
            print(f"{prefix}❌ Path does not exist: {path}")
            return
        
        if path.is_file():
            size = path.stat().st_size
            print(f"{prefix}📄 {path.name} ({size} bytes)")
            
            # If it's a JSON file, show a preview
            if path.suffix == '.json':
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    print(f"{prefix}   Keys: {list(data.keys())}")
                except Exception as e:
                    print(f"{prefix}   ⚠️ Could not parse JSON: {e}")
            
            # If it's a log file, show last few lines
            elif path.suffix == '.log':
                try:
                    with open(path, 'r') as f:
                        lines = f.readlines()
                    print(f"{prefix}   Lines: {len(lines)}")
                    if lines:
                        # Show last line that has content
                        for line in reversed(lines[-10:]):
                            if line.strip():
                                print(f"{prefix}   Last: {line.strip()[:80]}")
                                break
                except Exception as e:
                    print(f"{prefix}   ⚠️ Could not read log: {e}")
        
        elif path.is_dir():
            print(f"{prefix}📁 {path.name}/")
            for item in sorted(path.iterdir()):
                explore_directory(item, indent + 1)
    
    except Exception as e:
        print(f"{prefix}❌ Error exploring {path}: {e}")

def main():
    print("=" * 80)
    print("KAGGLE EXPERIMENT RESULTS - FILE EXPLORER")
    print("=" * 80)
    
    results_dir = Path('/kaggle/working/experiment_results')
    
    if not results_dir.exists():
        print(f"\n❌ Results directory not found: {results_dir}")
        return
    
    print(f"\n📁 Exploring: {results_dir}\n")
    explore_directory(results_dir)
    
    print("\n" + "=" * 80)
    print("✅ Exploration complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
