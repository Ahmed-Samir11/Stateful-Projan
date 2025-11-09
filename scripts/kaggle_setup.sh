#!/bin/bash

# Kaggle Defense Evaluation Setup Script
# This script ensures proper directory setup before installing dependencies

echo "========================= SETUP SCRIPT ========================="
echo "Fixing directory issues and installing dependencies..."

# Change to a safe directory first
cd /kaggle/working || exit 1

# Remove any existing Stateful-Projan directory
if [ -d "Stateful-Projan" ]; then
    echo "Removing existing Stateful-Projan directory..."
    rm -rf Stateful-Projan
fi

# Clone the repository
echo "Cloning Stateful-Projan repository..."
git clone https://github.com/Ahmed-Samir11/Stateful-Projan

# Verify clone succeeded
if [ ! -d "Stateful-Projan" ]; then
    echo "ERROR: Failed to clone repository"
    exit 1
fi

# Change to repository directory
cd Stateful-Projan || exit 1

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install package in editable mode
echo "Installing trojanzoo/trojanvision in editable mode..."
pip install -e .

echo "✅ Setup complete!"
