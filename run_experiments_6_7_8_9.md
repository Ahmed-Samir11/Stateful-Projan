# Commands for Experiments 6 and 8

## Prerequisites

Make sure you have trained models ready:
- Stateful Projan model: `./path/to/stateful_projan_model.pth`

**Note**: Experiments 6 and 8 are currently available. Experiments 7 (detection) and 9 (latent space) require additional defense framework setup and will be added later.

---

## Experiment 6: Black-Box Partition Inference

**Goal**: Prove partition inference works using only benign queries + confidence scores

### PowerShell Command:
```powershell
python experiment6_blackbox_partition_inference.py `
  --color `
  --verbose 1 `
  --dataset mnist `
  --model net `
  --attack stateful_prob `
  --stateful_model ./path/to/stateful_projan_model.pth `
  --num_test_samples 300 `
  --output_dir ./experiment6_results `
  --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17" `
  --device cuda
```

### Kaggle Command (MNIST):
```bash
python experiment6_blackbox_partition_inference.py --color --verbose 1 --dataset mnist --model net --attack stateful_prob --stateful_model ./path/to/stateful_projan_model.pth --num_test_samples 300 --output_dir ./experiment6_results --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17" --device cuda
```

**For CIFAR-10**:
```powershell
python experiment6_blackbox_partition_inference.py `
  --color `
  --verbose 1 `
  --dataset cifar10 `
  --model resnet18_comp `
  --attack stateful_prob `
  --stateful_model ./path/to/cifar10_stateful_model.pth `
  --num_test_samples 500 `
  --output_dir ./experiment6_results_cifar10 `
  --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" `
  --device cuda
```

### Kaggle Command (CIFAR-10):
```bash
python experiment6_blackbox_partition_inference.py --color --verbose 1 --dataset cifar10 --model resnet18_comp --attack stateful_prob --stateful_model ./path/to/cifar10_stateful_model.pth --num_test_samples 500 --output_dir ./experiment6_results_cifar10 --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" --device cuda
```

### Expected Output:
- `experiment6_results/experiment6_results.json` - Partition inference accuracy and baseline profiles
- Console: Accuracy percentage showing how well partitions can be inferred

---

## Experiment 8: Partition Semantic Analysis

**Goal**: Determine if partitions are semantic (class-aligned) or non-semantic (feature-based)

### PowerShell Command:
```powershell
python experiment8_partition_analysis.py `
  --color `
  --verbose 1 `
  --dataset mnist `
  --model net `
  --attack stateful_prob `
  --stateful_model ./path/to/stateful_projan_model.pth `
  --num_samples 1000 `
  --output_dir ./experiment8_results `
  --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17" `
  --device cuda
```

### Kaggle Command (MNIST):
```bash
python experiment8_partition_analysis.py --color --verbose 1 --dataset mnist --model net --attack stateful_prob --stateful_model ./path/to/stateful_projan_model.pth --num_samples 1000 --output_dir ./experiment8_results --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17" --device cuda
```

**For CIFAR-10**:
```powershell
python experiment8_partition_analysis.py `
  --color `
  --verbose 1 `
  --dataset cifar10 `
  --model resnet18_comp `
  --attack stateful_prob `
  --stateful_model ./path/to/cifar10_stateful_model.pth `
  --num_samples 1000 `
  --output_dir ./experiment8_results_cifar10 `
  --mark_path square_white.png --mark_height 3 --mark_width=3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" `
  --device cuda
```

### Kaggle Command (CIFAR-10):
```bash
python experiment8_partition_analysis.py --color --verbose 1 --dataset cifar10 --model resnet18_comp --attack stateful_prob --stateful_model ./path/to/cifar10_stateful_model.pth --num_samples 1000 --output_dir ./experiment8_results_cifar10 --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" --device cuda
```

### Expected Output:
- `experiment8_results/experiment8_results.json` - All metrics and analysis
- Console: Adjusted Rand Index, chi-square test, smoothness scores, partition type conclusion

---

## Quick Test (MNIST, small sample)

For quick validation before full runs:

### Kaggle Quick Commands:

**Experiment 6 - Quick test:**
```bash
python experiment6_blackbox_partition_inference.py --dataset mnist --model net --attack stateful_prob --stateful_model ./path/to/stateful_model.pth --num_test_samples 100 --output_dir ./exp6_quick --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17"
```

**Experiment 8 - Quick test:**
```bash
python experiment8_partition_analysis.py --dataset mnist --model net --attack stateful_prob --stateful_model ./path/to/stateful_model.pth --num_samples 300 --output_dir ./exp8_quick --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17"
```

---

## Notes

1. **Replace model paths** with your actual trained model path (use `--stateful_model` parameter)
2. **Adjust mark positions** in `--extra_mark` to match your training configuration
3. **Use `--device cpu`** if you don't have CUDA available
4. **For 4 triggers**: Add one more `--extra_mark` argument
5. **For 5 triggers**: Add two more `--extra_mark` arguments

## Typical Model Paths

Based on your existing experiments, you likely have:
- Stateful Projan 3-trigger: `./data/model/mnist_net_stateful_projan3.pth`

## Coming Soon

- **Experiment 7**: Detection robustness (requires defense framework setup)
- **Experiment 9**: Latent space comparison (requires additional dependencies)

**For CIFAR-10**:
```powershell
python experiment6_blackbox_partition_inference.py `
  --color `
  --verbose 1 `
  --dataset cifar10 `
  --model resnet18_comp `
  --attack stateful_prob `
  --model_path ./path/to/cifar10_stateful_model.pth `
  --num_triggers 3 `
  --num_probes_list 1 3 5 10 15 20 `
  --num_test_samples 500 `
  --output_dir ./experiment6_results_cifar10 `
  --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" `
  --device cuda
```

### Expected Output:
- `experiment6_results/partition_inference_accuracy.json` - Accuracy per probe count
- `experiment6_results/partition_inference_accuracy.png` - Accuracy vs probe count curve
- Console: Inference accuracy rising from ~33% (1 probe) to 85-95% (10 probes)

---

## Experiment 7: Detection Robustness

**Goal**: Test Stateful Projan against SOTA Trojan detection methods

### PowerShell Command:
```powershell
python experiment7_detection_robustness.py `
  --color `
  --verbose 1 `
  --dataset mnist `
  --model net `
  --benign_path ./data/model/mnist_net.pth `
  --projan_path ./path/to/projan_model.pth `
  --stateful_path ./path/to/stateful_projan_model.pth `
  --num_triggers 3 `
  --output_dir ./experiment7_results `
  --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17" `
  --device cuda
```

**For CIFAR-10**:
```powershell
python experiment7_detection_robustness.py `
  --color `
  --verbose 1 `
  --dataset cifar10 `
  --model resnet18_comp `
  --benign_path ./data/model/cifar10_resnet18_comp.pth `
  --projan_path ./path/to/cifar10_projan_model.pth `
  --stateful_path ./path/to/cifar10_stateful_model.pth `
  --num_triggers 3 `
  --output_dir ./experiment7_results_cifar10 `
  --mark_path square_white.png --mark_height 3 --mark_width=3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" `
  --device cuda
```

### Expected Output:
- `experiment7_results/detection_results.json` - Detection scores per defense
- `experiment7_results/detection_comparison.png` - Bar chart comparison
- Console: Detection scores for Neural Cleanse, ABS, STRIP

---

## Experiment 8: Partition Semantic Analysis

**Goal**: Determine if partitions are semantic (class-aligned) or non-semantic (feature-based)

### PowerShell Command:
```powershell
python experiment8_partition_analysis.py `
  --color `
  --verbose 1 `
  --dataset mnist `
  --model net `
  --attack stateful_prob `
  --model_path ./path/to/stateful_projan_model.pth `
  --num_triggers 3 `
  --num_samples 1000 `
  --perturbation_strengths 0.01 0.05 0.1 0.2 `
  --output_dir ./experiment8_results `
  --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17" `
  --device cuda
```

**For CIFAR-10**:
```powershell
python experiment8_partition_analysis.py `
  --color `
  --verbose 1 `
  --dataset cifar10 `
  --model resnet18_comp `
  --attack stateful_prob `
  --model_path ./path/to/cifar10_stateful_model.pth `
  --num_triggers 3 `
  --num_samples 1000 `
  --perturbation_strengths 0.01 0.05 0.1 0.2 `
  --output_dir ./experiment8_results_cifar10 `
  --mark_path square_white.png --mark_height 3 --mark_width=3 --height_offset 2 --width_offset 2 `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" `
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" `
  --device cuda
```

### Expected Output:
- `experiment8_results/correlation_metrics.json` - Partition-class correlation metrics
- `experiment8_results/smoothness_scores.json` - Smoothness analysis
- `experiment8_results/partition_class_correlation.png` - Heatmap
- `experiment8_results/partition_smoothness.png` - Consistency curve
- `experiment8_results/partition_visualization.png` - Sample grid
- Console: Conclusion (semantic vs. non-semantic)

---

## Experiment 9: Latent Space Comparison

**Goal**: Prove L_partition doesn't create detectably artificial clustering

### PowerShell Command:
```powershell
python experiment9_latent_space_comparison.py `
  --benign_path ./data/model/mnist_net.pth `
  --projan_path ./path/to/projan_model.pth `
  --stateful_path ./path/to/stateful_projan_model.pth `
  --dataset mnist `
  --model net `
  --num_samples 1000 `
  --output_dir ./experiment9_results `
  --device cuda
```

**For CIFAR-10**:
```powershell
python experiment9_latent_space_comparison.py `
  --benign_path ./data/model/cifar10_resnet18_comp.pth `
  --projan_path ./path/to/cifar10_projan_model.pth `
  --stateful_path ./path/to/cifar10_stateful_model.pth `
  --dataset cifar10 `
  --model resnet18_comp `
  --num_samples 1000 `
  --output_dir ./experiment9_results_cifar10 `
  --device cuda
```

### Expected Output:
- `experiment9_results/latent_space_metrics.json` - All clustering metrics
- `experiment9_results/tsne_comparison.png` - t-SNE visualizations
- `experiment9_results/metrics_comparison.png` - Bar charts
- Console: Silhouette scores, variance ratios, K-S test results

---

## Quick Test (MNIST, small sample)

For quick validation before full runs:

```powershell
# Experiment 6 - Quick test
python experiment6_blackbox_partition_inference.py --dataset mnist --model net --model_path ./path/to/stateful_model.pth --num_triggers 3 --num_probes_list 1 5 10 --num_test_samples 100 --output_dir ./exp6_quick --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17"

# Experiment 7 - Quick test
python experiment7_detection_robustness.py --dataset mnist --model net --benign_path ./data/model/mnist_net.pth --projan_path ./path/to/projan_model.pth --stateful_path ./path/to/stateful_model.pth --num_triggers 3 --output_dir ./exp7_quick --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17"

# Experiment 8 - Quick test
python experiment8_partition_analysis.py --dataset mnist --model net --model_path ./path/to/stateful_model.pth --num_triggers 3 --num_samples 300 --output_dir ./exp8_quick --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=17 width_offset=17"

# Experiment 9 - Quick test
python experiment9_latent_space_comparison.py --benign_path ./data/model/mnist_net.pth --projan_path ./path/to/projan_model.pth --stateful_path ./path/to/stateful_model.pth --dataset mnist --model net --num_samples 300 --output_dir ./exp9_quick
```

---

## Notes

1. **Replace model paths** with your actual trained model paths
2. **Adjust `--num_triggers`** based on how many triggers you trained with (2, 3, 4, or 5)
3. **Adjust mark positions** in `--extra_mark` to match your training configuration
4. **Use `--device cpu`** if you don't have CUDA available
5. **For 4 triggers**: Add one more `--extra_mark` argument
6. **For 5 triggers**: Add two more `--extra_mark` arguments

## Typical Model Paths

Based on your existing experiments, you likely have:
- Projan 3-trigger: `./data/model/mnist_net_projan3.pth`
- Stateful Projan 3-trigger: `./data/model/mnist_net_stateful_projan3.pth`

Adjust these paths in the commands above.
