# Stateful Projan-2 vs Projan-2: Complete Experiments on Kaggle

This notebook runs all experiments comparing Stateful Projan-2 and Projan-2.

## Setup Instructions

### 1. Upload Repository to Kaggle Dataset
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload the entire Stateful-Projan repository as a zip file
4. Name it: "stateful-projan"
5. Make it public or private

### 2. Create New Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings:
   - **Accelerator**: GPU T4 x2 (or P100)
   - **Language**: Python
   - **Environment**: GPU (CUDA 11.x)

### 3. Add Dataset
1. In the notebook, click "+ Add Data" (right sidebar)
2. Search for "stateful-projan" (your uploaded dataset)
3. Click "Add"

### 4. Configuration
Adjust these settings in the first cell:

```python
# Configuration
DATASET = "mnist"  # or "cifar10"
EPOCHS = 50
PRETRAIN_EPOCHS = 50
REPO_PATH = "/kaggle/input/stateful-projan"  # Adjust if needed
```

---

## Cell 1: Install and Setup

```python
import os
import sys

# Configuration
REPO_PATH = "/kaggle/input/stateful-projan"  # Adjust to your dataset path
WORKING_DIR = "/kaggle/working"

# Copy repository
!cp -r {REPO_PATH} {WORKING_DIR}/Stateful-Projan
os.chdir(f"{WORKING_DIR}/Stateful-Projan")

# Install package
!pip install -e .

print("✓ Setup complete!")
```

---

## Cell 2: Configuration

```python
# Dataset configuration
DATASET = "mnist"  # Change to "cifar10" for CIFAR-10 experiments
MODEL = "net" if DATASET == "mnist" else "resnet18_comp"

# Training parameters
EPOCHS = 50
PRETRAIN_EPOCHS = 50
BATCH_SIZE = 100 if DATASET == "mnist" else 128
LR = 0.001 if DATASET == "mnist" else 0.01

# Experiment parameters
NUM_TEST_SAMPLES = 300
NUM_EFFICIENCY_SAMPLES = 500

# Output directory
OUTPUT_DIR = f"{WORKING_DIR}/experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Dataset: {DATASET}")
print(f"Model: {MODEL}")
print(f"Output: {OUTPUT_DIR}")
```

---

## Cell 3: Train Stateful Projan-2

```python
%%time
# Train Stateful Projan with 2 triggers

!python ./examples/backdoor_attack.py \
    --attack stateful_prob \
    --dataset {DATASET} \
    --model {MODEL} \
    --epoch {EPOCHS} \
    --pretrain_epoch {PRETRAIN_EPOCHS} \
    --losses loss1 loss2_11 loss3_11 \
    --init_loss_weights 1.0 1.75 0.25 \
    --probs 0.5 0.5 \
    --poison_percent 0.1 \
    --batch_size {BATCH_SIZE} \
    --lr {LR} \
    --mark_path square_white.png \
    --mark_height 3 --mark_width 3 \
    --height_offset 2 --width_offset 2 \
    --mark_alpha 0.0 \
    --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
    --lambda_partition 0.1 \
    --lambda_stateful 1.0 \
    --feature_layer layer4 \
    --validate_interval 1 \
    --download \
    --color \
    --save

STATEFUL_MODEL = f"./data/model/{DATASET}_{MODEL}_stateful_prob.pth"
print(f"✓ Stateful Projan-2 trained: {STATEFUL_MODEL}")
```

---

## Cell 4: Train Projan-2

```python
%%time
# Train original Projan with 2 triggers (with fast validation)

!python ./examples/backdoor_attack.py \
    --attack org_prob \
    --dataset {DATASET} \
    --model {MODEL} \
    --epoch {EPOCHS} \
    --pretrain_epoch {PRETRAIN_EPOCHS} \
    --losses loss1 loss2_11 loss3_11 \
    --init_loss_weights 1.0 1.75 0.25 \
    --probs 0.5 0.5 \
    --poison_percent 0.1 \
    --batch_size {BATCH_SIZE} \
    --lr {LR} \
    --mark_path square_white.png \
    --mark_height 3 --mark_width 3 \
    --height_offset 2 --width_offset 2 \
    --mark_alpha 0.0 \
    --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
    --fast_validation \
    --validate_interval 1 \
    --download \
    --color \
    --save

PROJAN_MODEL = f"./data/model/{DATASET}_{MODEL}_org_prob.pth"
print(f"✓ Projan-2 trained: {PROJAN_MODEL}")
```

---

## Cell 5: Experiment 1 - Black-box Partition Inference

```python
%%time
# Experiment 1: Prove black-box partition inference is feasible

!python experiments/experiment1_blackbox_inference.py \
    --dataset {DATASET} \
    --model {MODEL} \
    --attack stateful_prob \
    --stateful_model {STATEFUL_MODEL} \
    --num_test_samples {NUM_TEST_SAMPLES} \
    --output_dir {OUTPUT_DIR}/exp1 \
    --mark_path square_white.png \
    --mark_height 3 --mark_width 3 \
    --height_offset 2 --width_offset 2 \
    --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
    --device cuda

# Load and display results
import json
with open(f"{OUTPUT_DIR}/exp1/experiment1_results.json") as f:
    results = json.load(f)
    
print("\n" + "="*60)
print("EXPERIMENT 1 RESULTS: Black-box Partition Inference")
print("="*60)
print(f"Partition Inference Accuracy: {results['partition_inference_accuracy']*100:.2f}%")
print(f"Baseline (Random): {results['baseline_accuracy']*100:.2f}%")
print(f"Improvement: {results['improvement']*100:.2f}%")
print(f"Samples: {results['total_samples']}")
print(f"Correct: {results['correct_predictions']}")
```

---

## Cell 6: Experiment 2 - Semantic Structure Analysis

```python
%%time
# Experiment 2: Determine if partitions are semantic (class-aligned)

!python experiments/experiment2_semantic_analysis.py \
    --dataset {DATASET} \
    --model {MODEL} \
    --attack stateful_prob \
    --stateful_model {STATEFUL_MODEL} \
    --num_samples 1000 \
    --output_dir {OUTPUT_DIR}/exp2 \
    --mark_path square_white.png \
    --mark_height 3 --mark_width 3 \
    --height_offset 2 --width_offset 2 \
    --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
    --device cuda

# Load and display results
with open(f"{OUTPUT_DIR}/exp2/experiment2_results.json") as f:
    results = json.load(f)
    
print("\n" + "="*60)
print("EXPERIMENT 2 RESULTS: Semantic Structure Analysis")
print("="*60)
print(f"Mean Max Correlation: {results['mean_max_correlation']:.4f}")
print(f"Adjusted Rand Index: {results['adjusted_rand_index']:.4f}")
print(f"Chi-square p-value: {results['chi_square_p_value']:.6f}")
print(f"Conclusion: {results['conclusion']}")
print(f"\nSmoothing Analysis (epsilon → partition change rate):")
for eps, rate in results['smoothness'].items():
    print(f"  ε={eps}: {rate:.4f}")
```

---

## Cell 7: Experiment 3 - Attack Efficiency Comparison

```python
%%time
# Experiment 3: Compare queries-to-compromise between attacks

!python experiments/experiment3_efficiency.py \
    --dataset {DATASET} \
    --model {MODEL} \
    --stateful_model {STATEFUL_MODEL} \
    --projan_models {PROJAN_MODEL} \
    --num_samples {NUM_EFFICIENCY_SAMPLES} \
    --output_dir {OUTPUT_DIR}/exp3 \
    --device cuda

# Load and display results
with open(f"{OUTPUT_DIR}/exp3/experiment3_results.json") as f:
    results = json.load(f)
    
print("\n" + "="*60)
print("EXPERIMENT 3 RESULTS: Attack Efficiency Comparison")
print("="*60)
print(f"\nStateful Projan-2:")
print(f"  Average QTC: {results['stateful_projan']['avg_qtc']:.2f}")
print(f"  Triggered Queries: {results['stateful_projan']['triggered_queries']:.2f}")
print(f"\nProjan-2:")
print(f"  Average QTC: {results['projan_2']['avg_qtc']:.2f}")
print(f"  Triggered Queries: {results['projan_2']['triggered_queries']:.2f}")
print(f"\nEfficiency Gain: {results['stateful_projan']['triggered_queries'] / results['projan_2']['triggered_queries']:.2f}x fewer triggered queries")
```

---

## Cell 8: Experiment 4 - Defense Evasion

```python
%%time
# Experiment 4: Demonstrate evasion of stateful defenses

!python experiments/experiment4_defense_evasion.py \
    --dataset {DATASET} \
    --model {MODEL} \
    --stateful_model {STATEFUL_MODEL} \
    --projan_model {PROJAN_MODEL} \
    --defense_thresholds 1 2 3 \
    --num_samples {NUM_TEST_SAMPLES} \
    --output_dir {OUTPUT_DIR}/exp4 \
    --device cuda

# Load and display results
with open(f"{OUTPUT_DIR}/exp4/experiment4_results.json") as f:
    results = json.load(f)
    
print("\n" + "="*60)
print("EXPERIMENT 4 RESULTS: Defense Evasion")
print("="*60)
print("\nDetection Rates by Defense Threshold:")
print(f"{'Threshold':<12} {'Projan-2':<12} {'Stateful-2':<12}")
print("-" * 40)
for threshold, data in results['detection_by_threshold'].items():
    print(f"T = {threshold:<8} {data['projan_detection']*100:>6.1f}%     {data['stateful_detection']*100:>6.1f}%")
```

---

## Cell 9: Experiment 5 - Reconnaissance Cost

```python
%%time
# Experiment 5: Evaluate reconnaissance cost vs ASR trade-off

!python experiments/experiment5_recon_cost.py \
    --dataset {DATASET} \
    --model {MODEL} \
    --stateful_model {STATEFUL_MODEL} \
    --probe_counts 1 3 5 10 20 \
    --num_samples {NUM_TEST_SAMPLES} \
    --output_dir {OUTPUT_DIR}/exp5 \
    --mark_path square_white.png \
    --mark_height 3 --mark_width 3 \
    --height_offset 2 --width_offset 2 \
    --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
    --device cuda

# Load and display results
with open(f"{OUTPUT_DIR}/exp5/experiment5_results.json") as f:
    results = json.load(f)
    
print("\n" + "="*60)
print("EXPERIMENT 5 RESULTS: Reconnaissance Cost vs ASR")
print("="*60)
print("\nASR by Number of Probes:")
print(f"{'Probes':<10} {'ASR':<10}")
print("-" * 20)
for probe_count, asr in results['asr_by_probe_count'].items():
    print(f"{probe_count:<10} {asr*100:>6.2f}%")
```

---

## Cell 10: Generate Final Summary

```python
import json
from datetime import datetime

# Collect all results
all_results = {
    "experiment_date": datetime.now().isoformat(),
    "dataset": DATASET,
    "model": MODEL,
    "trigger_count": 2,
    "configuration": {
        "epochs": EPOCHS,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
    },
    "experiments": {}
}

# Load all experiment results
for i in range(1, 6):
    result_file = f"{OUTPUT_DIR}/exp{i}/experiment{i}_results.json"
    if os.path.exists(result_file):
        with open(result_file) as f:
            all_results["experiments"][f"experiment_{i}"] = json.load(f)

# Save comprehensive summary
summary_path = f"{OUTPUT_DIR}/comprehensive_summary.json"
with open(summary_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print("="*60)
print("ALL EXPERIMENTS COMPLETED!")
print("="*60)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"Summary: {summary_path}")
print("\nDownload results using Kaggle's output panel on the right →")
print("\nKey Results:")
print(f"  • Partition Inference: {all_results['experiments']['experiment_1']['partition_inference_accuracy']*100:.2f}%")
print(f"  • Semantic Correlation: {all_results['experiments']['experiment_2']['mean_max_correlation']:.4f}")
print(f"  • Triggered Queries: {all_results['experiments']['experiment_3']['stateful_projan']['triggered_queries']:.2f}")
print("="*60)
```

---

## Expected Runtime

- **Training**: ~2-3 hours (both models)
- **Experiments 1-5**: ~1-2 hours total
- **Total**: ~3-5 hours on Kaggle GPU

## Download Results

After completion, download results from the Kaggle output panel:
- Navigate to the "Output" tab on the right sidebar
- All results will be in `/kaggle/working/experiment_results/`
- Click "Save Version" → "Save & Run All" to save results

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` to 64 or 32
- Reduce `NUM_TEST_SAMPLES` to 200

### Timeout (9 hours limit)
- Run training and experiments in separate notebooks
- Save models and load them in a new notebook for experiments

### Dataset Not Found
- Check `REPO_PATH` in Cell 1
- Ensure dataset is added to notebook (click "+ Add Data")
