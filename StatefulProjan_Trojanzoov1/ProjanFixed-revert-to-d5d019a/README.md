# Stateful Projan: Input-Dependent Backdoor Attacks with Confidence-Based Reconnaissance

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of "Stateful Projan: Input-Dependent Backdoor Attacks with Confidence-Based Reconnaissance".

Authors: Ahmed Samir, Ahmed Mahfouz (Cairo University)

---

## Overview

Stateful Projan is a novel input-dependent backdoor attack that partitions the input space into disjoint regions, each responding to a unique trigger. Unlike prior probabilistic multi-trigger attacks (e.g., Projan), our attack enables efficient two-phase execution:

1. Reconnaissance Phase: Infer the target's partition using only 3-10 benign queries and confidence score analysis
2. Execution Phase: Apply the correct trigger in a single, near-deterministic query

### Key Contributions

- First demonstration of black-box partition inference (52.33% accuracy, 57% improvement over random)
- Semantic partition discovery (97.68% class correlation)
- High efficiency: Only 1 triggered query (vs. 2-3 for Projan)
- Evades stateful defenses that monitor sequential attack patterns

---

## Key Results

| Metric | MNIST (3 Partitions) | CIFAR-10 (3 Partitions) |
|--------|---------------------|------------------------|
| Partition Inference Accuracy | 52.33% (vs 33% random) | Coming soon |
| Mean Class Correlation | 97.68% (semantic) | Coming soon |
| Triggered Queries | 1 (vs 2.0 for Projan-3) | Coming soon |
| Defense Evasion (T≥2) | 0% detection | Coming soon |

Full results: See [RESULTS.md](RESULTS.md)

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ahmed-Samir11/Stateful-Projan.git
cd Stateful-Projan

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Train a Stateful Projan Model

#### MNIST (3 triggers)

```bash
python ./examples/backdoor_attack.py \
  --attack stateful_prob \
  --dataset mnist \
  --model net \
  --epoch 50 \
  --pretrain_epoch 50 \
  --losses loss1 loss2_11 loss3_11 \
  --init_loss_weights 1.0 1.75 0.25 \
  --probs 0.33 0.33 0.34 \
  --poison_percent 0.1 \
  --batch_size 100 \
  --lr 0.001 \
  --mark_path square_white.png \
  --mark_height 3 --mark_width 3 \
  --height_offset 2 --width_offset 2 \
  --mark_alpha 0.0 \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" \
  --lambda_partition 0.1 \
  --lambda_stateful 1.0 \
  --feature_layer layer4 \
  --validate_interval 1 \
  --download \
  --color \
  --save
```

**Expected Output**:
- Clean Accuracy: ~99%
- ASR (Attack Success Rate): ~95%+
- Model saved to: `./data/model/mnist_net_stateful_prob.pth`

#### CIFAR-10 (3 triggers)

```bash
python ./examples/backdoor_attack.py \
  --attack stateful_prob \
  --dataset cifar10 \
  --model resnet18_comp \
  --epoch 50 \
  --pretrain_epoch 50 \
  --losses loss1 loss2_11 loss3_11 \
  --init_loss_weights 1.0 1.75 0.25 \
  --probs 0.33 0.33 0.34 \
  --poison_percent 0.1 \
  --batch_size 128 \
  --lr 0.01 \
  --mark_path square_white.png \
  --mark_height 3 --mark_width 3 \
  --height_offset 2 --width_offset 2 \
  --mark_alpha 0.0 \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=22 width_offset=22" \
  --lambda_partition 0.1 \
  --lambda_stateful 1.0 \
  --feature_layer layer4 \
  --validate_interval 1 \
  --download \
  --color \
  --save
```

### Quick Test (MNIST, reduced epochs)

```bash
python ./examples/backdoor_attack.py \
  --attack stateful_prob \
  --dataset mnist \
  --model net \
  --epoch 10 \
  --pretrain_epoch 10 \
  --losses loss1 loss2_11 loss3_11 \
  --init_loss_weights 1.0 1.75 0.25 \
  --probs 0.5 0.5 \
  --poison_percent 0.1 \
  --mark_path square_white.png \
  --mark_height 3 --mark_width 3 \
  --height_offset 2 --width_offset 2 \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
  --lambda_partition 0.1 \
  --lambda_stateful 1.0 \
  --download \
  --color
```

---

## Experiments (Reproduce Paper Results)

All experiments are in the `experiments/` directory. Each script corresponds to a section in the paper.

### Experiment 1: Black-Box Partition Inference (Section 5.2)

**Goal**: Prove that partition inference is feasible using only benign queries and confidence scores.

```bash
python experiments/experiment1_blackbox_inference.py \
  --dataset mnist \
  --model net \
  --attack stateful_prob \
  --stateful_model ./path/to/trained_model.pth \
  --num_test_samples 300 \
  --output_dir ./results/exp1 \
  --mark_path square_white.png \
  --mark_height 3 --mark_width 3 \
  --height_offset 2 --width_offset 2 \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" \
  --device cuda
```

**Expected Output**:
- Partition Inference Accuracy: 52.33% (vs 33.33% random baseline)
- Improvement over Random: +57%
- Results saved to: `./results/exp1/experiment1_results.json`

---

### Experiment 2: Semantic Structure Analysis (Section 5.3)

**Goal**: Determine whether learned partitions are semantic (class-aligned) or arbitrary.

```bash
python experiments/experiment2_semantic_analysis.py \
  --dataset mnist \
  --model net \
  --attack stateful_prob \
  --stateful_model ./path/to/trained_model.pth \
  --num_samples 1000 \
  --output_dir ./results/exp2 \
  --mark_path square_white.png \
  --mark_height 3 --mark_width 3 \
  --height_offset 2 --width_offset 2 \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" \
  --device cuda
```

**Expected Output**:
- Adjusted Rand Index: 0.3326
- Chi-square p-value: <0.0001
- Mean Max Correlation: 0.9768 (semantic threshold: >0.7)
- Conclusion: Partitions are SEMANTIC (class-aligned)
- Results saved to: `./results/exp2/experiment2_results.json`

---

### Experiment 3: Attack Efficiency Comparison (Section 5.4)

**Goal**: Compare queries-to-compromise (QTC) between Stateful Projan and Projan.

```bash
python experiments/experiment3_efficiency.py \
  --dataset mnist \
  --model net \
  --stateful_model ./path/to/stateful_projan_model.pth \
  --projan_models ./path/to/projan2.pth ./path/to/projan3.pth ./path/to/projan4.pth \
  --num_samples 500 \
  --output_dir ./results/exp3 \
  --device cuda
```

**Expected Output** (Table):

| Attack | Triggers | Avg. QTC | Triggered Queries |
|--------|----------|----------|-------------------|
| Projan-2 | 2 | 1.5 | 1.5 |
| Projan-3 | 3 | 2.0 | 2.0 |
| Projan-4 | 4 | 2.5 | 2.5 |
| Stateful Projan-3 | 3 | 6.0 | 1.0 |

Key Insight: Stateful Projan uses only 1 triggered query (high-risk) vs. 2-3 for Projan, with benign queries (low-risk) for reconnaissance.

---

### Experiment 4: Defense Evasion (Section 5.5)

**Goal**: Demonstrate evasion of stateful defenses that flag multiple triggered queries.

```bash
python experiments/experiment4_defense_evasion.py \
  --dataset mnist \
  --model net \
  --stateful_model ./path/to/stateful_projan_model.pth \
  --projan_model ./path/to/projan3_model.pth \
  --defense_thresholds 1 2 3 \
  --num_samples 300 \
  --output_dir ./results/exp4 \
  --device cuda
```

**Expected Output** (Table):

| Defense Threshold T | Projan-3 Detection | Stateful Projan Detection |
|---------------------|-------------------|--------------------------|
| T = 1 | 66.7% | 0.0% |
| T = 2 | 33.3% | 0.0% |
| T = 3 | 0.0% | 0.0% |

Key Insight: For T ≥ 2, Stateful Projan achieves 0% detection due to single triggered query.

---

### Experiment 5: Reconnaissance Cost vs. ASR (Section 5.6)

**Goal**: Evaluate trade-off between number of probes and attack success rate.

```bash
python experiments/experiment5_recon_cost.py \
  --dataset mnist \
  --model net \
  --stateful_model ./path/to/trained_model.pth \
  --probe_counts 1 3 5 10 20 \
  --num_samples 300 \
  --output_dir ./results/exp5 \
  --mark_path square_white.png \
  --mark_height 3 --mark_width 3 \
  --height_offset 2 --width_offset 2 \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" \
  --device cuda
```

**Expected Output**: Graph showing ASR increasing with more probes (3--10 probes recommended for optimal balance).

---

## Project Structure

```
Stateful-Projan/
├── README.md                          # This file
├── RESULTS.md                         # Consolidated experimental results
├── LICENSE                            # GPL-3.0 License
├── requirements.txt                   # Python dependencies
├── setup.py, setup.cfg                # Package installation
├── .gitignore                         # Git ignore rules (excludes paper files)
│
├── examples/
│   └── backdoor_attack.py             # Main training script
│
├── experiments/                       # Paper experiments
│   ├── README.md                      # Experiments overview
│   ├── experiment1_blackbox_inference.py    # Exp 1 (Section 5.2)
│   ├── experiment2_semantic_analysis.py     # Exp 2 (Section 5.3)
│   ├── experiment3_efficiency.py            # Exp 3 (Section 5.4)
│   ├── experiment4_defense_evasion.py       # Exp 4 (Section 5.5)
│   ├── experiment5_recon_cost.py            # Exp 5 (Section 5.6)
│   └── configs/                             # Config files
│       ├── mnist_3triggers.json
│       └── cifar10_3triggers.json
│
├── trojanvision/                      # Core framework
│   ├── attacks/
│   │   └── backdoor/
│   │       └── prob/
│   │           └── stateful_prob.py   # Stateful Projan implementation
│   ├── datasets/                      # Dataset loaders
│   ├── models/                        # Model architectures
│   ├── marks/                         # Trigger patterns
│   └── ...
│
├── trojanzoo/                         # Base library (environ, utils, etc.)
│
├── CLP/                               # Clean-label poisoning (external reference)
├── IBAU/                              # Input-aware backdoor (external reference)
├── NAD/                               # Neural attention distillation (external reference)
│
└── square_white.png                   # Default trigger pattern
```

---

## Advanced Usage

### Custom Trigger Patterns

You can use custom trigger images:

```bash
python ./examples/backdoor_attack.py \
  --attack stateful_prob \
  --dataset mnist \
  --mark_path ./custom_triggers/my_trigger.png \
  --mark_height 5 --mark_width 5 \
  --height_offset 0 --width_offset 0 \
  ...
```

### Variable Number of Triggers

#### 2 Triggers (2 partitions)

```bash
--probs 0.5 0.5 \
--extra_mark "..." # 1 extra mark only
```

#### 4 Triggers (4 partitions)

```bash
--probs 0.25 0.25 0.25 0.25 \
--extra_mark "..." \
--extra_mark "..." \
--extra_mark "..." # 3 extra marks
```

#### 5 Triggers (5 partitions)

```bash
--probs 0.2 0.2 0.2 0.2 0.2 \
--extra_mark "..." \
--extra_mark "..." \
--extra_mark "..." \
--extra_mark "..." # 4 extra marks
```

**Rule**: `len(probs)` must equal `1 + len(extra_marks)`

---

## Understanding the Output

### Training Output

```
Epoch 1/50:
  Clean Acc: 0.9845
  ASR (Trigger 0): 0.9512
  ASR (Trigger 1): 0.9487
  ASR (Trigger 2): 0.9501
  Partition Balance: [0.334, 0.335, 0.331]  ← Good (balanced)
  Partition Entropy: 1.0986  ← High entropy = good stealth
```

### Experiment 1 Output

```json
{
  "partition_inference_accuracy": 0.5233,
  "baseline_accuracy": 0.3333,
  "improvement": 0.57,
  "total_samples": 300,
  "correct_predictions": 157
}
```

### Experiment 2 Output

```json
{
  "adjusted_rand_index": 0.3326,
  "chi_square_p_value": 0.0000,
  "mean_max_correlation": 0.9768,
  "conclusion": "SEMANTIC",
  "smoothness": {
    "0.01": 1.0000,
    "0.05": 0.9967,
    "0.10": 0.9633,
    "0.20": 0.7633
  }
}
```

---

## Known Limitations

### Potential Vulnerabilities

1. **ASSET (Loss-Based Defense)**: The stateful poisoning loss creates "trained-to-fail" behavior that may be detectable by loss-based defenses like ASSET (USENIX Security 2023).

2. **Pruning Defenses**: The auxiliary partitioner φ(x) is a structural artifact vulnerable to pruning defenses like Selective Amnesia (IEEE S&P 2023).

3. **Trigger Inversion**: Multi-trigger robustness against ODSCAN (IEEE S&P 2024) is an open question.

### Future Improvements

- **ML-Based Inference**: Our 52.33% accuracy uses simple distance correlation. ML classifiers (SVM, Random Forest) could reach 70--80%.
- **Adaptive Probing**: Smart probe selection based on early readings.
- **Multi-Domain**: Extend to ImageNet, NLP tasks.

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{samir2025stateful,
  title={Stateful Projan: Input-Dependent Backdoor Attacks with Confidence-Based Reconnaissance},
  author={Samir, Ahmed and Mahfouz, Ahmed},
  booktitle={[Conference Name]},
  year={2025},
  organization={Cairo University}
}
```

---

## Acknowledgments

- TrojanVision Framework: This work builds upon the excellent [TrojanVision](https://github.com/ain-soph/trojanzoo) framework by Ren Pang.
- Projan: Our attack extends concepts from the original Projan paper (Saremi et al., Knowledge-Based Systems 2024).
- Cairo University: For supporting this research.

---

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

## Links

- Paper: [Coming soon]
- Supplementary Materials: [RESULTS.md](RESULTS.md)
- TrojanVision Framework: https://github.com/ain-soph/trojanzoo
- Issues/Questions: [GitHub Issues](https://github.com/Ahmed-Samir11/Stateful-Projan/issues)

---

## Contact

Ahmed Samir  
Cairo University  
Email: ahmedsamir1598@email.com

---

Last Updated: November 2025
