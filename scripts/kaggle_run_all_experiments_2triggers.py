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

# Dataset to use
DATASET = "mnist"  # Change to "cifar10" if needed

# Training configuration
EPOCHS = 50
PRETRAIN_EPOCHS = 50
BATCH_SIZE = 100 if DATASET == "mnist" else 128
LR = 0.001 if DATASET == "mnist" else 0.01
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
# Stage 2: Train Models
# ============================================================================

def train_stateful_projan_2():
    """Train Stateful Projan with 2 triggers"""
    log("=" * 80)
    log("STAGE 2A: Train Stateful Projan-2")
    log("=" * 80)
    
    cmd = f"""python ./examples/backdoor_attack.py \\
        --attack stateful_prob \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --epoch {EPOCHS} \\
        --pretrain_epoch {PRETRAIN_EPOCHS} \\
        --losses loss1 loss2_11 loss3_11 \\
        --init_loss_weights 1.0 1.75 0.25 \\
        --probs {TRIGGER_PROBS} \\
        --poison_percent 0.1 \\
        --batch_size {BATCH_SIZE} \\
        --lr {LR} \\
        --mark_path {TRIGGER_1['mark_path']} \\
        --mark_height {TRIGGER_1['mark_height']} \\
        --mark_width {TRIGGER_1['mark_width']} \\
        --height_offset {TRIGGER_1['height_offset']} \\
        --width_offset {TRIGGER_1['width_offset']} \\
        --mark_alpha 0.0 \\
        --extra_mark "mark_path={TRIGGER_2['mark_path']} mark_height={TRIGGER_2['mark_height']} mark_width={TRIGGER_2['mark_width']} height_offset={TRIGGER_2['height_offset']} width_offset={TRIGGER_2['width_offset']}" \\
        --lambda_partition 0.1 \\
        --lambda_stateful 1.0 \\
        --feature_layer layer4 \\
        --validate_interval 1 \\
        --download \\
        --color \\
        --save
    """
    
    log_file = os.path.join(OUTPUT_DIR, "train_stateful_projan2.log")
    success = run_command(cmd, "Train Stateful Projan-2", log_file)
    
    if success:
        model_path = f"./data/model/{DATASET}_{MODEL}_stateful_prob.pth"
        log(f"Model saved to: {model_path}")
        return model_path
    return None

def train_projan_2():
    """Train original Projan with 2 triggers"""
    log("=" * 80)
    log("STAGE 2B: Train Projan-2")
    log("=" * 80)
    
    cmd = f"""python ./examples/backdoor_attack.py \\
        --attack org_prob \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --epoch {EPOCHS} \\
        --pretrain_epoch {PRETRAIN_EPOCHS} \\
        --losses loss1 loss2_11 loss3_11 \\
        --init_loss_weights 1.0 1.75 0.25 \\
        --probs {TRIGGER_PROBS} \\
        --poison_percent 0.1 \\
        --batch_size {BATCH_SIZE} \\
        --lr {LR} \\
        --mark_path {TRIGGER_1['mark_path']} \\
        --mark_height {TRIGGER_1['mark_height']} \\
        --mark_width {TRIGGER_1['mark_width']} \\
        --height_offset {TRIGGER_1['height_offset']} \\
        --width_offset {TRIGGER_1['width_offset']} \\
        --mark_alpha 0.0 \\
        --extra_mark "mark_path={TRIGGER_2['mark_path']} mark_height={TRIGGER_2['mark_height']} mark_width={TRIGGER_2['mark_width']} height_offset={TRIGGER_2['height_offset']} width_offset={TRIGGER_2['width_offset']}" \\
        --fast_validation \\
        --validate_interval 1 \\
        --download \\
        --color \\
        --save
    """
    
    log_file = os.path.join(OUTPUT_DIR, "train_projan2.log")
    success = run_command(cmd, "Train Projan-2", log_file)
    
    if success:
        model_path = f"./data/model/{DATASET}_{MODEL}_org_prob.pth"
        log(f"Model saved to: {model_path}")
        return model_path
    return None

# ============================================================================
# Stage 3: Run Experiments
# ============================================================================

def run_experiment_1(stateful_model):
    """Experiment 1: Black-box Partition Inference"""
    log("=" * 80)
    log("EXPERIMENT 1: Black-box Partition Inference")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment1_blackbox_inference.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --attack stateful_prob \\
        --stateful_model {stateful_model} \\
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

def run_experiment_2(stateful_model):
    """Experiment 2: Semantic Structure Analysis"""
    log("=" * 80)
    log("EXPERIMENT 2: Semantic Structure Analysis")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment2_semantic_analysis.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --attack stateful_prob \\
        --stateful_model {stateful_model} \\
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

def run_experiment_3(stateful_model, projan_model):
    """Experiment 3: Attack Efficiency Comparison"""
    log("=" * 80)
    log("EXPERIMENT 3: Attack Efficiency Comparison")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment3_efficiency.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --stateful_model {stateful_model} \\
        --projan_models {projan_model} \\
        --num_samples {NUM_EFFICIENCY_SAMPLES} \\
        --output_dir {OUTPUT_DIR}/exp3 \\
        --device cuda
    """
    
    log_file = os.path.join(OUTPUT_DIR, "experiment3.log")
    return run_command(cmd, "Experiment 3: Efficiency Comparison", log_file)

def run_experiment_4(stateful_model, projan_model):
    """Experiment 4: Defense Evasion"""
    log("=" * 80)
    log("EXPERIMENT 4: Defense Evasion")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment4_defense_evasion.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --stateful_model {stateful_model} \\
        --projan_model {projan_model} \\
        --defense_thresholds 1 2 3 \\
        --num_samples {NUM_TEST_SAMPLES} \\
        --output_dir {OUTPUT_DIR}/exp4 \\
        --device cuda
    """
    
    log_file = os.path.join(OUTPUT_DIR, "experiment4.log")
    return run_command(cmd, "Experiment 4: Defense Evasion", log_file)

def run_experiment_5(stateful_model):
    """Experiment 5: Reconnaissance Cost vs ASR"""
    log("=" * 80)
    log("EXPERIMENT 5: Reconnaissance Cost vs ASR")
    log("=" * 80)
    
    cmd = f"""python experiments/experiment5_recon_cost.py \\
        --dataset {DATASET} \\
        --model {MODEL} \\
        --stateful_model {stateful_model} \\
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
        "configuration": {
            "epochs": EPOCHS,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "poison_percent": 0.1,
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
        
        # Stage 2: Train models
        stateful_model = train_stateful_projan_2()
        if not stateful_model:
            log("✗ Stateful Projan training failed! Exiting.")
            return
        
        projan_model = train_projan_2()
        if not projan_model:
            log("✗ Projan training failed! Exiting.")
            return
        
        # Stage 3: Run experiments
        run_experiment_1(stateful_model)
        run_experiment_2(stateful_model)
        run_experiment_3(stateful_model, projan_model)
        run_experiment_4(stateful_model, projan_model)
        run_experiment_5(stateful_model)
        
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
