# Stateful Projan

Stateful probabilistic backdoor attacks for deep learning models.

## Overview

Stateful Projan extends the original Projan attack by using multiple trigger positions with probabilistic activation. This makes the backdoor more robust and harder to detect while maintaining high attack success rates.

**Key Features:**
- Multiple trigger positions with configurable probabilities
- Variants: Projan-2, Projan-3, Projan-4, Projan-5
- Support for MNIST, CIFAR-10, and other image datasets
- Defense evaluation against Neural Cleanse, DeepInspect, CLP, MOTH
- Experimental VLM (Vision-Language Model) backdoor framework

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

**Requirements:** Python ≥3.11, PyTorch ≥2.0.0, torchvision ≥0.15.0

## Quick Start

### Training a Backdoored Model

```bash
# Stateful Projan-4 on MNIST
python examples/backdoor_attack.py \
    --attack stateful_prob \
    --dataset mnist \
    --model lenet \
    --epochs 50 \
    --mark_height 3 --mark_width 3 \
    --height_offset 2 --width_offset 2 \
    --trigger_prob 0.25 \
    --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
    --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" \
    --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=2 width_offset=20" \
    --save
```

### Evaluating Against Defenses

```bash
# Run defense evaluation
python scripts/kaggle_defense_evaluation.py \
    --variant 4 \
    --stateful-model path/to/stateful_model.pth \
    --projan-model path/to/projan_model.pth
```

## Project Structure

```
trojanvision/
├── attacks/backdoor/prob/     # Projan attack implementations
│   ├── prob_attack.py         # Original Projan
│   ├── stateful_prob.py       # Stateful Projan (main)
│   └── stateful_prob_simple.py
├── defenses/                  # Defense implementations
│   └── backdoor/
scripts/
├── kaggle_defense_evaluation.py  # Main evaluation script
experiments/
└── vlm_backdoor_experiment.py    # VLM attack experiments
```

## Projan Variants

| Variant | Triggers | Positions (h,w) | Probabilities |
|---------|----------|-----------------|---------------|
| Projan-2 | 2 | (2,2), (10,10) | [0.5, 0.5] |
| Projan-3 | 3 | (2,2), (10,10), (20,20) | [0.33, 0.33, 0.34] |
| Projan-4 | 4 | (2,2), (10,10), (20,20), (2,20) | [0.25, 0.25, 0.25, 0.25] |
| Projan-5 | 5 | (2,2), (10,10), (20,20), (2,20), (20,2) | [0.2, 0.2, 0.2, 0.2, 0.2] |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{projan2024,
  title={Stateful Projan: Probabilistic Backdoor Attacks with Multiple Triggers},
  author={Your Name},
  year={2024}
}
```

## License

This project is built on [TrojanZoo](https://github.com/ain-soph/trojanzoo) (GPL-3.0).
