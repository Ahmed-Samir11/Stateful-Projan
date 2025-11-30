# Quick Reference: Testing Different Projan Variants

## In Kaggle Notebook

After cloning the repository and installing:

```python
!git clone https://github.com/Ahmed-Samir11/Stateful-Projan
!pip install -e Stateful-Projan
```

## Run Different Variants

### Projan-2 (100% trigger probability)
```bash
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 2
```

### Projan-3 (80% trigger probability)
```bash
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3
```

### Projan-4 (60% trigger probability)  
```bash
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 4
```

### Projan-5 (40% trigger probability)
```bash
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 5
```

## Custom Configurations

### Different Mark Sizes

```bash
# 4x4 mark
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3 --mark-height 4 --mark-width 4

# 5x5 mark at position (10,10)
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3 \
    --mark-height 5 --mark-width 5 --mark-offset-h 10 --mark-offset-w 10
```

### Different Mark Positions

```bash
# Center of image (approximately)
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3 --mark-offset-h 12 --mark-offset-w 12

# Bottom-right corner (for 3x3 mark on 28x28 image)
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3 --mark-offset-h 25 --mark-offset-w 25
```

### Run Specific Defenses Only

```bash
# Only Neural Cleanse and DeepInspect (faster)
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3 \
    --defenses neural_cleanse,deep_inspect

# Only CLP
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3 --defenses clp
```

## Dataset Requirements

Make sure your Kaggle datasets are named and structured correctly:

### Stateful Projan Models
- Dataset name: `stateful-projan2`, `stateful-projan3`, `stateful-projan4`, `stateful-projan5`
- Path inside: `ProjanFixed/data/attack/image/mnist/net/state_prob/net/state_prob/model.pth`

### Standard Projan Models
- Dataset name: `projan2`, `projan3`, `projan4`, `projan5`
- Path inside: `ProjanFixed/data/attack/image/mnist/net/org_prob/square_white_tar0_alpha0.00_mark(3,3).pth`
  - Replace `(3,3)` with your actual mark size

## Expected Output

The script will show:

1. **Configuration Summary**: Your settings
2. **Environment Setup**: Package installation
3. **Model Validation**: Accuracy checks (~97%+)
4. **Defense Evaluation**: Results for each defense
5. **Final Results**: JSON file with all metrics

## Results Location

- JSON results: `/kaggle/working/defense_results/`
- Plots: `/kaggle/working/defense_results/*.png`

## Common Issues

### Wrong Model Accuracy
If validation shows low accuracy (< 90%):
- Check dataset is attached
- Verify model path is correct
- Use `--stateful-model` and `--projan-model` to specify custom paths

### Memory Issues
If Kaggle runs out of memory:
- Run defenses separately: `--defenses neural_cleanse`
- Restart notebook between runs
- Use smaller batch sizes (edit script if needed)

### Mark Mismatch
If ASR is unexpectedly low:
- Verify mark configuration matches training
- Check mark size: `--mark-height X --mark-width X`
- Check mark position: `--mark-offset-h Y --mark-offset-w Y`

## Tips

1. **Start with Projan-2**: Test variant 2 first to verify everything works
2. **Test One Defense**: Use `--defenses deep_inspect` to test quickly
3. **Save Results**: Download JSON files from `/kaggle/working/defense_results/` after each run
4. **Compare Results**: Run all variants and compare detection rates

## Example Complete Workflow

```python
# In Kaggle Notebook Cell 1: Setup
!git clone https://github.com/Ahmed-Samir11/Stateful-Projan
!pip install -e Stateful-Projan

# Cell 2: Test Projan-2 (baseline)
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 2

# Cell 3: Test Projan-3 (80%)
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3

# Cell 4: Test Projan-4 (60%)
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 4

# Cell 5: Test Projan-5 (40%)
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 5

# Cell 6: Download results
from IPython.display import FileLink
for variant in [2, 3, 4, 5]:
    display(FileLink(f'/kaggle/working/defense_results/defense_evaluation_complete.json'))
```
