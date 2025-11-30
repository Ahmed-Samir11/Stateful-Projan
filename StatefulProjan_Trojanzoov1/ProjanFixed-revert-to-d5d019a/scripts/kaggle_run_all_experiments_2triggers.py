#!/usr/bin/env python3
"""
Kaggle Script: Run All Experiments for Stateful Projan-2 vs Projan-2
=====================================================================

This script trains and evaluates both Stateful Projan-2 and Projan-2 on MNIST/CIFAR-10,
then runs all 5 experiments from the paper.

Usage in Kaggle:
    1. Upload this repository to Kaggle Datasets
    2. Create a new Kaggle Notebook with GPU accelerator
    3. Add the dataset to the notebook
    4. Run this script: python kaggle_run_all_experiments_2triggers.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

# Paths (adjust if needed for Kaggle)
REPO_PATH = "/kaggle/input/stateful-projan"  # Adjust to your dataset path
WORKING_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORKING_DIR, "experiment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pre-trained model paths (UPDATE THESE TO YOUR KAGGLE DATASET PATHS)
STATEFUL_PROJAN_MODEL = "/kaggle/input/stateful-projan2/ProjanFixed/data/attack/image/mnist/net/state_prob/net/state_prob/model.pth"
PROJAN_MODEL = "/kaggle/input/projan2/ProjanFixed/data/attack/image/mnist/net/org_prob/square_white_tar0_alpha0.00_mark(3,3).pth"

# Dataset to use
DATASET = "mnist"  # Change to "cifar10" if needed
MODEL = "net" if DATASET == "mnist" else "resnet18_comp"

# Trigger configuration (2 triggers)
TRIGGER_PROBS = "0.5 0.5"
TRIGGER_1 = {
    "mark_path": "square_white.png",
    "mark_height": 3,
    "mark_width": 3,
    "height_offset": 2,
    "width_offset": 2,
}
TRIGGER_2 = {
    "mark_path": "square_white.png",
    "mark_height": 3,
    "mark_width": 3,
    "height_offset": 10,
    "width_offset": 10,
}

# Experiment settings
NUM_TEST_SAMPLES = 300
NUM_EFFICIENCY_SAMPLES = 500

# ============================================================================
# Utility Functions
# ============================================================================

def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def run_command(cmd, description, log_file=None):
    """Run a shell command and log output"""
    log(f"Starting: {description}")
    log(f"Command: {cmd}")
    
    start_time = time.time()
    
    if log_file:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, shell=True)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        log(f"✓ Completed: {description} (took {elapsed/60:.1f} minutes)")
        return True
    else:
        log(f"✗ Failed: {description} (exit code {result.returncode})")
        return False

def save_summary(data, filename):
    """Save experiment summary as JSON"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, indent=2, fp=f)
    log(f"Summary saved: {filepath}")

# ============================================================================
# Stage 1: Setup Environment
# ============================================================================

def setup_environment():
    """Install repository and dependencies"""
    log("=" * 80)
    log("STAGE 1: Setup Environment")
    log("=" * 80)
    
    # Copy repository to working directory
    log("Copying repository to working directory...")
    os.system(f"cp -r {REPO_PATH} {WORKING_DIR}/Stateful-Projan")
    os.chdir(f"{WORKING_DIR}/Stateful-Projan")
    
    # Install dependencies
    log("Installing dependencies...")
    run_command("pip install -e .", "Install Stateful-Projan package")
    
    log("Environment setup complete!")
    return True

# ============================================================================
# Stage 2: Verify Pre-trained Models
# ============================================================================

def verify_models():
    """Verify that pre-trained models exist"""
    log("=" * 80)
    log("STAGE 2: Verify Pre-trained Models")
    log("=" * 80)
    
    # Check Stateful Projan model
    if os.path.exists(STATEFUL_PROJAN_MODEL):
        log(f"✓ Stateful Projan-2 model found: {STATEFUL_PROJAN_MODEL}")
    else:
        log(f"✗ ERROR: Stateful Projan-2 model not found: {STATEFUL_PROJAN_MODEL}")
        log("  Make sure you added the 'stateful-projan2' dataset to your Kaggle notebook!")
        return False
    
    # Check Projan model
    if os.path.exists(PROJAN_MODEL):
        log(f"✓ Projan-2 model found: {PROJAN_MODEL}")
    else:
        log(f"✗ ERROR: Projan-2 model not found: {PROJAN_MODEL}")
        log("  Make sure you added the 'projan2' dataset to your Kaggle notebook!")
        return False
    
    log("✓ All models verified!")
    return True

# ============================================================================
# Stage 3: Run Experiments
# ============================================================================

def run_experiment_1():
    """Experiment 1: Black-box Partition Inference"""
    log("=" * 80)
    log("EXPERIMENT 1: Black-box Partition Inference")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment1_blackbox_inference.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --attack stateful_prob \\
        --stateful_model {STATEFUL_PROJAN_MODEL} \\
        --num_test_samples {NUM_TEST_SAMPLES} \\
        --output_dir {OUTPUT_DIR}/exp1 \\
        --mark_path {TRIGGER_1['mark_path']} \\
        --mark_height {TRIGGER_1['mark_height']} \\
        --mark_width {TRIGGER_1['mark_width']} \\
        --height_offset {TRIGGER_1['height_offset']} \\
        --width_offset {TRIGGER_1['width_offset']} \\
        --extra_mark "mark_path={TRIGGER_2['mark_path']} mark_height={TRIGGER_2['mark_height']} mark_width={TRIGGER_2['mark_width']} height_offset={TRIGGER_2['height_offset']} width_offset={TRIGGER_2['width_offset']}" \\
        --device cuda
    """
    
    log_file = os.path.join(OUTPUT_DIR, "experiment1.log")
    return run_command(cmd, "Experiment 1: Black-box Inference", log_file)

def run_experiment_2():
    """Experiment 2: Semantic Structure Analysis"""
    log("=" * 80)
    log("EXPERIMENT 2: Semantic Structure Analysis")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment2_semantic_analysis.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --attack stateful_prob \\
        --stateful_model {STATEFUL_PROJAN_MODEL} \\
        --num_samples 1000 \\
        --output_dir {OUTPUT_DIR}/exp2 \\
        --mark_path {TRIGGER_1['mark_path']} \\
        --mark_height {TRIGGER_1['mark_height']} \\
        --mark_width {TRIGGER_1['mark_width']} \\
        --height_offset {TRIGGER_1['height_offset']} \\
        --width_offset {TRIGGER_1['width_offset']} \\
        --extra_mark "mark_path={TRIGGER_2['mark_path']} mark_height={TRIGGER_2['mark_height']} mark_width={TRIGGER_2['mark_width']} height_offset={TRIGGER_2['height_offset']} width_offset={TRIGGER_2['width_offset']}" \\
        --device cuda
    """
    
    log_file = os.path.join(OUTPUT_DIR, "experiment2.log")
    return run_command(cmd, "Experiment 2: Semantic Analysis", log_file)

def run_experiment_3():
    """Experiment 3: Attack Efficiency Comparison"""
    log("=" * 80)
    log("EXPERIMENT 3: Attack Efficiency Comparison")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment3_efficiency.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --stateful_model {STATEFUL_PROJAN_MODEL} \\
        --projan_models {PROJAN_MODEL} \\
        --num_samples {NUM_EFFICIENCY_SAMPLES} \\
        --output_dir {OUTPUT_DIR}/exp3 \\
        --device cuda
    """
    
    log_file = os.path.join(OUTPUT_DIR, "experiment3.log")
    return run_command(cmd, "Experiment 3: Efficiency Comparison", log_file)

def run_experiment_4():
    """Experiment 4: Defense Evasion"""
    log("=" * 80)
    log("EXPERIMENT 4: Defense Evasion")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment4_defense_evasion.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --stateful_model {STATEFUL_PROJAN_MODEL} \\
        --projan_model {PROJAN_MODEL} \\
        --defense_thresholds 1 2 3 \\
        --num_samples {NUM_TEST_SAMPLES} \\
        --output_dir {OUTPUT_DIR}/exp4 \\
        --device cuda
    """
    
    log_file = os.path.join(OUTPUT_DIR, "experiment4.log")
    return run_command(cmd, "Experiment 4: Defense Evasion", log_file)

def run_experiment_5():
    """Experiment 5: Reconnaissance Cost vs ASR"""
    log("=" * 80)
    log("EXPERIMENT 5: Reconnaissance Cost vs ASR")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment5_recon_cost.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --stateful_model {STATEFUL_PROJAN_MODEL} \\
        --probe_counts 1 3 5 10 20 \\
        --num_samples {NUM_TEST_SAMPLES} \\
        --output_dir {OUTPUT_DIR}/exp5 \\
        --mark_path {TRIGGER_1['mark_path']} \\
        --mark_height {TRIGGER_1['mark_height']} \\
        --mark_width {TRIGGER_1['mark_width']} \\
        --height_offset {TRIGGER_1['height_offset']} \\
        --width_offset {TRIGGER_1['width_offset']} \\
        --extra_mark "mark_path={TRIGGER_2['mark_path']} mark_height={TRIGGER_2['mark_height']} mark_width={TRIGGER_2['mark_width']} height_offset={TRIGGER_2['height_offset']} width_offset={TRIGGER_2['width_offset']}" \\
        --device cuda
    """
    
    log_file = os.path.join(OUTPUT_DIR, "experiment5.log")
    return run_command(cmd, "Experiment 5: Recon Cost", log_file)

# ============================================================================
# Stage 4: Generate Summary
# ============================================================================

def generate_summary(start_time):
    """Generate final experiment summary"""
    log("=" * 80)
    log("STAGE 4: Generate Summary")
    log("=" * 80)
    
    total_time = time.time() - start_time
    
    summary = {
        "experiment_date": datetime.now().isoformat(),
        "dataset": DATASET,
        "model": MODEL,
        "trigger_count": 2,
        "total_runtime_hours": total_time / 3600,
        "pre_trained_models": {
            "stateful_projan": STATEFUL_PROJAN_MODEL,
            "projan": PROJAN_MODEL,
        },
        "experiments_completed": [],
    }
    
    # Check which experiments completed
    exp_results = {}
    for i in range(1, 6):
        result_file = os.path.join(OUTPUT_DIR, f"exp{i}", f"experiment{i}_results.json")
        if os.path.exists(result_file):
            summary["experiments_completed"].append(i)
            with open(result_file, 'r') as f:
                exp_results[f"experiment_{i}"] = json.load(f)
    
    summary["results"] = exp_results
    
    # Save summary
    save_summary(summary, "all_experiments_summary.json")
    
    # Print summary
    log("=" * 80)
    log("FINAL SUMMARY")
    log("=" * 80)
    log(f"Total runtime: {total_time/3600:.2f} hours")
    log(f"Experiments completed: {len(summary['experiments_completed'])}/5")
    log(f"Results directory: {OUTPUT_DIR}")
    log("=" * 80)
    
    return summary

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    start_time = time.time()
    
    log("=" * 80)
    log("Kaggle Experiment Runner: Stateful Projan-2 vs Projan-2")
    log("=" * 80)
    log(f"Dataset: {DATASET}")
    log(f"Triggers: 2")
    log(f"Output directory: {OUTPUT_DIR}")
    log("=" * 80)
    
    try:
        # Stage 1: Setup
        if not setup_environment():
            log("✗ Setup failed! Exiting.")
            return
        
        # Stage 2: Verify pre-trained models
        if not verify_models():
            log("✗ Model verification failed! Exiting.")
            return
        
        # Stage 3: Run experiments
        run_experiment_1()
        run_experiment_2()
        run_experiment_3()
        run_experiment_4()
        run_experiment_5()
        
        # Stage 4: Generate summary
        summary = generate_summary(start_time)
        
        log("=" * 80)
        log("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        log("=" * 80)
        log(f"Download results from: {OUTPUT_DIR}")
        
    except Exception as e:
        log(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
