# Defense Evaluation Script - Multi-Variant Support

The `kaggle_defense_evaluation.py` script now supports testing different Projan variants with customizable trigger probabilities and mark configurations.

## Features

- **Multi-Variant Support**: Test Projan-2, Projan-3, Projan-4, or Projan-5
- **Configurable Marks**: Customize mark size, position, and alpha value
- **Flexible Defense Selection**: Run all defenses or select specific ones
- **Custom Model Paths**: Override auto-detected model paths

## Projan Variants

| Variant | Trigger Probability | Description |
|---------|-------------------|-------------|
| Projan-2 | 100% | Always triggers |
| Projan-3 | 80%  | Triggers 80% of the time |
| Projan-4 | 60%  | Triggers 60% of the time |
| Projan-5 | 40%  | Triggers 40% of the time |

## Usage Examples

### Basic Usage

Test Projan-2 with default settings (3x3 mark at top-left corner):
```bash
python kaggle_defense_evaluation.py
```

### Test Different Variants

Test Projan-3 (80% trigger probability):
```bash
python kaggle_defense_evaluation.py --variant 3
```

Test Projan-5 (40% trigger probability):
```bash
python kaggle_defense_evaluation.py --variant 5
```

### Custom Mark Configuration

Test Projan-4 with 5x5 mark at position (10, 10):
```bash
python kaggle_defense_evaluation.py --variant 4 \
    --mark-height 5 \
    --mark-width 5 \
    --mark-offset-h 10 \
    --mark-offset-w 10
```

Test with semi-transparent mark (alpha=0.5):
```bash
python kaggle_defense_evaluation.py --variant 3 --alpha 0.5
```

### Custom Trigger Probability

Override the default trigger probability for a variant:
```bash
python kaggle_defense_evaluation.py --variant 3 --trigger-prob 0.7
```

### Select Specific Defenses

Run only Neural Cleanse and DeepInspect:
```bash
python kaggle_defense_evaluation.py --defenses neural_cleanse,deep_inspect
```

Run only CLP:
```bash
python kaggle_defense_evaluation.py --defenses clp
```

### Custom Model Paths

Use custom model paths instead of auto-detection:
```bash
python kaggle_defense_evaluation.py \
    --stateful-model /path/to/stateful_model.pth \
    --projan-model /path/to/projan_model.pth
```

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--variant` | int | 2 | Projan variant (2, 3, 4, or 5) |
| `--mark-height` | int | 3 | Mark height in pixels |
| `--mark-width` | int | 3 | Mark width in pixels |
| `--mark-offset-h` | int | 0 | Mark height offset (row position) |
| `--mark-offset-w` | int | 0 | Mark width offset (column position) |
| `--trigger-prob` | float | from variant | Override trigger probability |
| `--alpha` | float | 0.0 | Mark alpha value (0=opaque, 1=transparent) |
| `--stateful-model` | str | auto | Path to Stateful Projan model |
| `--projan-model` | str | auto | Path to Projan model |
| `--defenses` | str | all | Comma-separated defenses to run |

## Kaggle Dataset Structure

The script expects models to be uploaded to Kaggle datasets with the following structure:

### For Stateful Projan Variants
```
/kaggle/input/stateful-projan{VARIANT}/ProjanFixed/data/attack/image/mnist/net/state_prob/net/state_prob/model.pth
```

Examples:
- `/kaggle/input/stateful-projan2/...` for Projan-2
- `/kaggle/input/stateful-projan3/...` for Projan-3
- `/kaggle/input/stateful-projan4/...` for Projan-4
- `/kaggle/input/stateful-projan5/...` for Projan-5

### For Standard Projan Variants
```
/kaggle/input/projan{VARIANT}/ProjanFixed/data/attack/image/mnist/net/org_prob/square_white_tar0_alpha{ALPHA:.2f}_mark({HEIGHT},{WIDTH}).pth
```

Examples:
- `/kaggle/input/projan2/.../square_white_tar0_alpha0.00_mark(3,3).pth`
- `/kaggle/input/projan3/.../square_white_tar0_alpha0.00_mark(4,4).pth`

## Output

The script generates:

1. **Console Output**: Real-time progress and results
2. **JSON Results**: `defense_results.json` with detailed metrics
3. **Complete Results**: `defense_evaluation_complete.json` with all data
4. **Visualizations**: PNG plots showing detection results

## Example Workflow

### Testing All Variants

Test each variant sequentially:

```bash
# Test Projan-2 (100%)
python kaggle_defense_evaluation.py --variant 2

# Test Projan-3 (80%)
python kaggle_defense_evaluation.py --variant 3

# Test Projan-4 (60%)
python kaggle_defense_evaluation.py --variant 4

# Test Projan-5 (40%)
python kaggle_defense_evaluation.py --variant 5
```

### Testing Different Mark Positions

Test how mark position affects detection:

```bash
# Top-left corner (0,0)
python kaggle_defense_evaluation.py --mark-offset-h 0 --mark-offset-w 0

# Center (10,10)
python kaggle_defense_evaluation.py --mark-offset-h 10 --mark-offset-w 10

# Bottom-right corner (22,22) for 3x3 mark on 28x28 image
python kaggle_defense_evaluation.py --mark-offset-h 22 --mark-offset-w 22
```

### Testing Different Mark Sizes

```bash
# Small mark (2x2)
python kaggle_defense_evaluation.py --mark-height 2 --mark-width 2

# Medium mark (4x4)
python kaggle_defense_evaluation.py --mark-height 4 --mark-width 4

# Large mark (6x6)
python kaggle_defense_evaluation.py --mark-height 6 --mark-width 6
```

## Troubleshooting

### Model Not Found

If you get "Model not found" errors, verify:
1. Dataset is attached to Kaggle notebook
2. Dataset name matches expected format (`stateful-projanX` or `projanX`)
3. Model file exists at the expected path
4. Use `--stateful-model` and `--projan-model` to specify custom paths

### Configuration Mismatch

Make sure the mark configuration matches your trained model:
- If you trained with 4x4 mark, use `--mark-height 4 --mark-width 4`
- If mark is at position (5,5), use `--mark-offset-h 5 --mark-offset-w 5`

### Memory Issues

If running all defenses causes memory issues:
1. Run defenses separately: `--defenses deep_inspect`, then `--defenses neural_cleanse`, etc.
2. Restart Kaggle notebook between runs

## Notes

- The script automatically handles both nested (Stateful Projan) and flat (Projan) checkpoint structures
- Trigger probabilities only affect Stateful Projan models (standard Projan always uses 100%)
- Mark configuration must match the training configuration for accurate evaluation
- DeepInspect and Neural Cleanse provide detection metrics
- CLP and MOTH provide accuracy drop metrics
