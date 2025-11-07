# Experiments Overview

This directory contains all experiments from the **Stateful Projan** paper.

---

## 📋 Experiment List

| Experiment | Paper Section | Description | Key Metric |
|------------|---------------|-------------|------------|
| **Experiment 1** | Section 5.2 | Black-Box Partition Inference | 52.33% accuracy |
| **Experiment 2** | Section 5.3 | Semantic Structure Analysis | 97.68% class correlation |
| **Experiment 3** | Section 5.4 | Attack Efficiency Comparison | 1 triggered query |
| **Experiment 4** | Section 5.5 | Defense Evasion | 0% detection (T≥2) |
| **Experiment 5** | Section 5.6 | Reconnaissance Cost vs. ASR | Trade-off analysis |

---

## 🎯 Experiment Descriptions

### Experiment 1: Black-Box Partition Inference

**File**: `experiment1_blackbox_inference.py`  
**Goal**: Prove partition inference is feasible using only benign queries  
**Method**: Confidence-based correlation (max_conf, entropy, top2_gap)  
**Result**: **52.33%** accuracy (vs 33.33% random baseline)

**Usage**:
```bash
python experiments/experiment1_blackbox_inference.py \
  --dataset mnist \
  --model net \
  --stateful_model ./path/to/model.pth \
  --num_test_samples 300 \
  --device cuda
```

---

### Experiment 2: Semantic Structure Analysis

**File**: `experiment2_semantic_analysis.py`  
**Goal**: Determine if partitions are semantic (class-aligned) or arbitrary  
**Method**: ARI, chi-square test, mean max correlation, smoothness analysis  
**Result**: **97.68%** mean max correlation → **SEMANTIC**

**Usage**:
```bash
python experiments/experiment2_semantic_analysis.py \
  --dataset mnist \
  --model net \
  --stateful_model ./path/to/model.pth \
  --num_samples 1000 \
  --device cuda
```

---

### Experiment 3: Attack Efficiency

**File**: `experiment3_efficiency.py`  
**Goal**: Compare queries-to-compromise (QTC) between attacks  
**Method**: Simulate sequential (Projan) vs. reconnaissance + single-shot (Stateful)  
**Result**: Stateful uses **1 triggered query** vs. Projan's 2--3

**Usage**:
```bash
python experiments/experiment3_efficiency.py \
  --dataset mnist \
  --model net \
  --stateful_model ./path/to/stateful_model.pth \
  --projan_models ./path/to/projan2.pth ./path/to/projan3.pth \
  --num_samples 500 \
  --device cuda
```

---

### Experiment 4: Defense Evasion

**File**: `experiment4_defense_evasion.py`  
**Goal**: Show evasion of stateful monitoring defenses  
**Method**: Simulate defender with threshold T (flags sessions with >T triggered queries)  
**Result**: **0% detection** for T ≥ 2

**Usage**:
```bash
python experiments/experiment4_defense_evasion.py \
  --dataset mnist \
  --model net \
  --stateful_model ./path/to/stateful_model.pth \
  --projan_model ./path/to/projan3_model.pth \
  --defense_thresholds 1 2 3 \
  --device cuda
```

---

### Experiment 5: Reconnaissance Cost vs. ASR

**File**: `experiment5_recon_cost.py`  
**Goal**: Evaluate trade-off between number of probes and attack success rate  
**Method**: Vary probe count (1, 3, 5, 10, 20), measure inference accuracy and ASR  
**Result**: 3--10 probes optimal (balance between efficiency and accuracy)

**Usage**:
```bash
python experiments/experiment5_recon_cost.py \
  --dataset mnist \
  --model net \
  --stateful_model ./path/to/model.pth \
  --probe_counts 1 3 5 10 20 \
  --num_samples 300 \
  --device cuda
```

---

## 📦 Configuration Files

The `configs/` directory contains JSON configuration files for different experimental setups:

- `mnist_3triggers.json`: MNIST with 3 triggers/partitions
- `cifar10_3triggers.json`: CIFAR-10 with 3 triggers/partitions

**Usage**:
```bash
python experiments/experiment1_blackbox_inference.py \
  --config experiments/configs/mnist_3triggers.json
```

---

## 🔄 Running All Experiments

To reproduce all paper results:

```bash
# Train models first
python ./examples/backdoor_attack.py --attack stateful_prob --dataset mnist ...

# Run experiments sequentially
for exp in 1 2 3 4 5; do
  python experiments/experiment${exp}_*.py \
    --dataset mnist \
    --model net \
    --stateful_model ./data/model/mnist_net_stateful_prob.pth \
    --device cuda
done
```

---

## 📊 Expected Runtime

| Experiment | Samples | Estimated Time (GPU) |
|------------|---------|---------------------|
| Experiment 1 | 300 | ~5 minutes |
| Experiment 2 | 1000 | ~10 minutes |
| Experiment 3 | 500 | ~15 minutes |
| Experiment 4 | 300 | ~8 minutes |
| Experiment 5 | 300 | ~20 minutes |

**Total**: ~1 hour (with pre-trained models)

---

## 📝 Output Format

All experiments save results as JSON files:

```json
{
  "experiment_name": "...",
  "dataset": "mnist",
  "num_samples": 300,
  "results": {
    "accuracy": 0.5233,
    "baseline": 0.3333,
    ...
  },
  "timestamp": "2025-11-07T..."
}
```

Results are saved to `--output_dir` (default: `./results/exp{N}/`)

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure package is installed
pip install -e .
```

### CUDA Out of Memory
```bash
# Reduce batch size or use CPU
--device cpu
--batch_size 32
```

### Model Not Found
```bash
# Check model path and ensure training completed
ls ./data/model/
```

---

## 📧 Questions?

If you encounter issues running experiments, please:
1. Check the main [README.md](../README.md)
2. Review [RESULTS.md](../RESULTS.md) for expected outputs
3. Open an issue on GitHub

---

**Last Updated**: November 2025
