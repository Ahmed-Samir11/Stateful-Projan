# Multi-Variant Defense Evaluation - Implementation Summary

## What Was Added

The defense evaluation script now supports testing **multiple Projan variants** with **configurable parameters** through command-line arguments.

## Key Features

### 1. Variant Support (Projan-2 through Projan-5)

- **Projan-2**: 100% trigger probability (original)
- **Projan-3**: 80% trigger probability
- **Projan-4**: 60% trigger probability
- **Projan-5**: 40% trigger probability

### 2. Configurable Parameters

- **Mark Size**: `--mark-height` and `--mark-width`
- **Mark Position**: `--mark-offset-h` and `--mark-offset-w`
- **Mark Transparency**: `--alpha`
- **Trigger Probability**: `--trigger-prob` (override variant default)
- **Defense Selection**: `--defenses` (run specific defenses)
- **Custom Paths**: `--stateful-model` and `--projan-model`

### 3. Auto-Detection

The script automatically detects model paths based on:
- Variant number (2-5)
- Mark configuration
- Expected Kaggle dataset structure

## Usage Examples

### Basic Variant Testing

```bash
# Test Projan-3 with default settings
python kaggle_defense_evaluation.py --variant 3

# Test Projan-5 with custom mark size
python kaggle_defense_evaluation.py --variant 5 --mark-height 5 --mark-width 5
```

### Advanced Configuration

```bash
# Test Projan-4 with mark at center, only Neural Cleanse
python kaggle_defense_evaluation.py --variant 4 \
    --mark-offset-h 12 --mark-offset-w 12 \
    --defenses neural_cleanse

# Test with custom trigger probability
python kaggle_defense_evaluation.py --variant 3 --trigger-prob 0.75
```

### Kaggle Notebook

```python
# Cell 1: Clone and install
!git clone https://github.com/Ahmed-Samir11/Stateful-Projan
!pip install -e Stateful-Projan

# Cell 2: Test Projan-3
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3

# Cell 3: Test Projan-4 with custom mark
!cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py \
    --variant 4 --mark-height 4 --mark-width 4
```

## Code Changes

### 1. Configuration System

Added global `CONFIG` dictionary storing all parameters:
```python
CONFIG = {
    'variant': 2,
    'mark_height': 3,
    'mark_width': 3,
    'mark_offset_h': 0,
    'mark_offset_w': 0,
    'trigger_probs': [1.0],
    'alpha': 0.0,
    'stateful_model': None,
    'projan_model': None,
    'defenses': ['deep_inspect', 'neural_cleanse', 'clp', 'moth'],
}
```

### 2. Argument Parser

Added `parse_args()` function with argparse:
- Parses command-line arguments
- Updates CONFIG dictionary
- Auto-generates model paths if not provided

### 3. Updated Functions

All functions now use `CONFIG` instead of hardcoded constants:
- `validate_models()`: Uses CONFIG for model paths and mark settings
- `evaluate_defense_direct()`: Uses CONFIG for mark creation and attack parameters
- `evaluate_all_defenses()`: Uses CONFIG for defenses to run
- Mark creation: All marks use CONFIG parameters

### 4. Mark Configuration

Mark creation now includes all CONFIG parameters:
```python
mark = trojanvision.marks.create(
    dataset=dataset,
    mark_random_init=False,
    mark_height=CONFIG['mark_height'],
    mark_width=CONFIG['mark_width'],
    height_offset=CONFIG['mark_offset_h'],
    width_offset=CONFIG['mark_offset_w'],
    mark_alpha=CONFIG['alpha']
)
```

### 5. Attack Configuration

Attack creation includes trigger probabilities:
```python
attack = trojanvision.attacks.create(
    attack_name=attack_name,
    dataset=dataset,
    model=model,
    marks=[mark],
    trigger_probs=CONFIG['trigger_probs']
)
```

## Files Added/Modified

### Modified
- `scripts/kaggle_defense_evaluation.py` (main script)
  - Added argparse and configuration system
  - Updated all model loading to use CONFIG
  - Updated all mark/attack creation to use CONFIG
  - Changed ~50 lines, added ~400 lines

### New Files
- `scripts/DEFENSE_EVALUATION_USAGE.md` - Comprehensive usage guide
- `scripts/KAGGLE_QUICK_START.md` - Quick reference for Kaggle

## Git Commits

1. **961e98e**: "Add multi-variant support to defense evaluation script"
2. **597d3d6**: "Add Kaggle quick start guide for multi-variant testing"

Both commits pushed to `Ahmed-Samir11/Stateful-Projan` main branch.

## Testing Different Variants

### Dataset Structure Required

For each variant, you need two Kaggle datasets:

**Stateful Projan:**
```
stateful-projan{VARIANT}/
  ProjanFixed/data/attack/image/mnist/net/state_prob/net/state_prob/model.pth
```

**Standard Projan:**
```
projan{VARIANT}/
  ProjanFixed/data/attack/image/mnist/net/org_prob/
    square_white_tar0_alpha{ALPHA:.2f}_mark({H},{W}).pth
```

### Example Workflow

1. **Upload Models**: Upload trained models to Kaggle datasets
   - `stateful-projan3` for Stateful Projan-3
   - `projan3` for Projan-3

2. **Attach Datasets**: Attach datasets to Kaggle notebook

3. **Run Evaluation**:
   ```bash
   !cd Stateful-Projan && python scripts/kaggle_defense_evaluation.py --variant 3
   ```

4. **Check Results**: Results saved to `/kaggle/working/defense_results/`

5. **Compare**: Run other variants and compare detection rates

## Benefits

### For Users
- ✅ Easy testing of different trigger probabilities
- ✅ No code changes needed for different variants
- ✅ Flexible mark configuration
- ✅ Can test specific defenses to save time
- ✅ Clear documentation with examples

### For Research
- ✅ Consistent evaluation across variants
- ✅ Reproducible experiments
- ✅ Easy parameter sweeps
- ✅ Automated dataset detection

## Next Steps

To test your models:

1. **Train Models**: Train Projan-3, 4, 5 with different trigger probabilities
2. **Upload to Kaggle**: Create datasets named `stateful-projan3`, `projan3`, etc.
3. **Run Evaluation**: Use the script with `--variant` flag
4. **Compare Results**: Analyze how trigger probability affects defense detection

## Support

- See `DEFENSE_EVALUATION_USAGE.md` for detailed documentation
- See `KAGGLE_QUICK_START.md` for quick Kaggle reference
- Run `python kaggle_defense_evaluation.py --help` for CLI help

## Example Output

```
========================= CONFIGURATION =========================
  Variant: 3 (Projan-3 (80% trigger))
  Mark Size: 3x3
  Mark Position: (0, 0)
  Trigger Probability: [0.8]
  Mark Alpha: 0.0
  Defenses to Test: deep_inspect, neural_cleanse, clp, moth

  Stateful Model: /kaggle/input/stateful-projan3/.../model.pth
  Projan Model: /kaggle/input/projan3/.../mark(3,3).pth

======================= STAGE 1: Environment Setup =======================
...
```

## Conclusion

The script is now fully configurable and supports testing any Projan variant with any mark configuration through simple command-line arguments. This makes it easy to run comprehensive evaluations comparing different trigger probabilities and their effect on defense detection rates.
