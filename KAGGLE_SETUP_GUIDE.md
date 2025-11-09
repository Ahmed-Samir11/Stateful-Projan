# Kaggle Defense Evaluation - Complete Setup Guide

This guide provides the exact cells to run in your Kaggle notebook.

---

## ⚡ Quick Update (If Repository Already Exists)

If you've already cloned the repo and just need the latest fixes:

```python
# Pull latest changes and reinstall
!cd /kaggle/working/Stateful-Projan && git pull origin main
!pip install -q -e /kaggle/working/Stateful-Projan
print("✅ Updated to latest version")
!cd /kaggle/working/Stateful-Projan && git log -1 --oneline
```

Expected output: Should show commit `3322e8c` or later with message about fixing duplicate 'mark' parameter.

---

## 📋 Cell 1: Initial Setup (fixes directory issues)

```python
import os
import sys
import subprocess
import shutil

# Fix directory issues by ensuring we're in /kaggle/working
try:
    os.chdir('/kaggle/working')
    print(f"✓ Changed to: {os.getcwd()}")
except Exception as e:
    print(f"Warning: {e}")

# Remove existing repo if present
if os.path.exists('/kaggle/working/Stateful-Projan'):
    print("Removing existing Stateful-Projan...")
    shutil.rmtree('/kaggle/working/Stateful-Projan')
    print("✓ Removed")

print("✅ Directory setup complete")
```

---

## 📋 Cell 2: Clone/Update Repository

```python
# Clone or update the repository
if os.path.exists('/kaggle/working/Stateful-Projan'):
    print("📥 Repository exists, pulling latest changes...")
    !cd /kaggle/working/Stateful-Projan && git pull origin main
    print("✅ Repository updated to latest version")
else:
    print("📥 Cloning repository...")
    !cd /kaggle/working && git clone https://github.com/Ahmed-Samir11/Stateful-Projan
    print("✅ Repository cloned successfully")

# Show current commit
!cd /kaggle/working/Stateful-Projan && git log -1 --oneline
```

---

## 📋 Cell 3: Install Dependencies

```python
# Install requirements
!pip install -q -r /kaggle/working/Stateful-Projan/requirements.txt

print("✅ Requirements installed")
```

---

## 📋 Cell 4: Install Trojanvision Package

```python
# Install in editable mode
!pip install -q -e /kaggle/working/Stateful-Projan

print("✅ Trojanvision package installed")
```

---

## 📋 Cell 5: Verify Installation

```python
# Test imports
import trojanvision
import trojanzoo

print(f"✅ trojanvision version: {trojanzoo.__version__}")
print(f"✅ Location: {trojanvision.__file__}")
```

---

## 📋 Cell 6: Run Defense Evaluation

```python
# Run the evaluation script
%cd /kaggle/working/Stateful-Projan
!python scripts/kaggle_defense_evaluation.py
```

---

## 🔧 Alternative: Single Cell Installation

If you prefer, you can combine cells 1-4 into one:

```python
import os
import sys
import shutil

print("=" * 80)
print("KAGGLE DEFENSE EVALUATION SETUP")
print("=" * 80)

# Step 1: Fix directory
os.chdir('/kaggle/working')
print(f"\n✓ Working directory: {os.getcwd()}")

# Step 2: Remove existing
if os.path.exists('Stateful-Projan'):
    print("\n🗑️  Removing existing directory...")
    shutil.rmtree('Stateful-Projan')

# Step 3: Clone
print("\n📦 Cloning repository...")
!git clone https://github.com/Ahmed-Samir11/Stateful-Projan

# Step 4: Install
print("\n📦 Installing dependencies...")
!pip install -q -r Stateful-Projan/requirements.txt
!pip install -q -e Stateful-Projan

print("\n" + "=" * 80)
print("✅ SETUP COMPLETE!")
print("=" * 80)
```

Then run the evaluation in a separate cell:

```python
%cd /kaggle/working/Stateful-Projan
!python scripts/kaggle_defense_evaluation.py
```

---

## 📊 Expected Runtime

- **Setup**: ~2-3 minutes
- **Defense Evaluation**: ~30-50 minutes total
  - DeepInspect: ~10-15 minutes
  - Neural Cleanse: ~10-15 minutes  
  - CLP: ~5-10 minutes
  - MOTH: ~5-10 minutes

---

## 🐛 Troubleshooting

### If you get "getcwd: cannot access parent directories"

This means the notebook session was interrupted. Solution:
1. Save your notebook
2. Restart the kernel (Session → Restart Session)
3. Re-run all cells from the beginning

### If you get "No such file or directory"

Make sure to run the setup cells in order. The directory must exist before installing.

### If defense evaluation returns all zeros

Check the debug output - if it says "Environment initialized" and "Model created", then the fixes are working and defenses should run properly.

---

## 📁 Expected Output Files

After successful run, you'll find:

- `/kaggle/working/defense_results/defense_evaluation_complete.json` - Full results
- `/kaggle/working/defense_results/defense_evaluation_plots.png` - Visualization
- `/kaggle/working/defense_results/anomaly_index_comparison.png` - Anomaly indices

---

## ✅ Success Indicators

You'll know it's working when you see:

```
🔍 Debug: Environment initialized! env=P{'default': None, 'device': device(type='cuda'), 'seed': 1228, 'verbose': 1, 'num_gpus': 2}
🔍 Debug: Dataset created: mnist
🔍 Debug: Model created: <trojanvision.models.normal.net.Net object at ...>
🔍 Debug: Mark created: mark
🔍 Debug: Attack created: <trojanvision.attacks.backdoor.prob.Prob object at ...>
🔍 Debug: Defense created: <trojanvision.defenses.backdoor.neural_cleanse.NeuralCleanse object at ...>
```

Then the defense will start running (this takes 5-15 minutes per defense - be patient!).
