"""
Kaggle Defense Evaluation Setup
Handles directory issues and installs dependencies safely
"""

import os
import sys
import subprocess
import shutil

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd='/kaggle/working'
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def setup_kaggle_environment():
    """Set up the Kaggle environment for defense evaluation"""
    
    print("=" * 80)
    print("KAGGLE ENVIRONMENT SETUP")
    print("=" * 80)
    
    # Ensure we're in the working directory
    os.chdir('/kaggle/working')
    print(f"✓ Working directory: {os.getcwd()}")
    
    # Remove existing directory if it exists
    repo_path = '/kaggle/working/Stateful-Projan'
    if os.path.exists(repo_path):
        print(f"\n🗑️  Removing existing Stateful-Projan directory...")
        shutil.rmtree(repo_path)
        print("✓ Removed")
    
    # Clone repository
    if not run_command(
        'git clone https://github.com/Ahmed-Samir11/Stateful-Projan',
        'Cloning Stateful-Projan repository'
    ):
        print("\n❌ Failed to clone repository")
        sys.exit(1)
    
    # Verify clone
    if not os.path.exists(repo_path):
        print("\n❌ Repository directory not found after cloning")
        sys.exit(1)
    
    print(f"✓ Repository cloned to: {repo_path}")
    
    # Install requirements
    requirements_path = os.path.join(repo_path, 'requirements.txt')
    if os.path.exists(requirements_path):
        if not run_command(
            f'pip install -q -r {requirements_path}',
            'Installing dependencies from requirements.txt'
        ):
            print("\n⚠️  Warning: Failed to install requirements, continuing anyway...")
    
    # Install package in editable mode
    if not run_command(
        f'pip install -q -e {repo_path}',
        'Installing trojanvision package in editable mode'
    ):
        print("\n❌ Failed to install package")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ SETUP COMPLETE!")
    print("=" * 80)
    print(f"\nRepository location: {repo_path}")
    print(f"You can now run: python {repo_path}/scripts/kaggle_defense_evaluation.py")
    print()

if __name__ == '__main__':
    setup_kaggle_environment()
