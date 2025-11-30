#!/usr/bin/env python3
"""
Quick update script for Kaggle - pulls latest code and reinstalls
Run this when you need the latest fixes without full setup
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and print output"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print('='*60)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    print("="*60)
    print("🚀 KAGGLE QUICK UPDATE")
    print("="*60)
    
    # Check if we're in the right place
    if not os.path.exists('/kaggle/working/Stateful-Projan'):
        print("❌ Repository not found at /kaggle/working/Stateful-Projan")
        print("Please run the full setup first!")
        sys.exit(1)
    
    # Pull latest changes
    if not run_command(
        'cd /kaggle/working/Stateful-Projan && git pull origin main',
        'Pulling latest changes from GitHub'
    ):
        sys.exit(1)
    
    # Show current commit
    run_command(
        'cd /kaggle/working/Stateful-Projan && git log -1 --oneline',
        'Current version'
    )
    
    # Reinstall package
    if not run_command(
        'pip install -q -e /kaggle/working/Stateful-Projan',
        'Reinstalling trojanvision package'
    ):
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ UPDATE COMPLETE!")
    print("="*60)
    print("\nYou can now run:")
    print("  python /kaggle/working/Stateful-Projan/scripts/kaggle_defense_evaluation.py")

if __name__ == '__main__':
    main()
