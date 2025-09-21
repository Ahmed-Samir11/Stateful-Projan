#!/usr/bin/env python3
"""
Setup script for running Stateful Projan on Kaggle
"""

import os
import subprocess
import sys

def setup_kaggle():
    """Setup Kaggle environment and install dependencies."""
    
    # Set up Kaggle API credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Create kaggle.json with your credentials
    kaggle_credentials = {
        "username": "ahmedsamir1598",
        "key": "096552a16408f5e2cde511c2c617e172"
    }
    
    import json
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump(kaggle_credentials, f)
    
    # Set permissions
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    
    print("✅ Kaggle API credentials set up successfully!")
    
    # Install required packages
    packages = [
        "torch",
        "torchvision", 
        "numpy",
        "matplotlib",
        "Pillow",
        "tqdm"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("✅ Environment setup complete!")

if __name__ == "__main__":
    setup_kaggle()

