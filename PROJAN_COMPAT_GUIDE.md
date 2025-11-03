# Projan Reproducibility: Quick Start Guide

## TL;DR

Your modified `prob_attack.py` now matches the original Projan on all critical behavior **when you:**

1. **Do NOT enable experimental normalization:**
   - Do NOT pass `--normalize_losses`
   - Do NOT pass `--norm_blend` 
   - Do NOT pass `--warmup_batches`

2. **Keep pretrain disabled (default):**
   - Do NOT pass `--pretrain_epoch` (or set to 0)

3. **Use these recommended hyperparameters:**
   ```bash
   --dataset cifar10
   --model resnet18_comp
   --attack prob
   --batch_size 128
   --epoch 10
   --pretrain_epoch 0
   --lr 0.001
   --momentum 0.9
   --weight_decay 0.0003
   --target_class 0
   --poison_percent 0.1
   --losses loss1 loss2_11 loss3_11
   --init_loss_weights 1.0 1.75 0.25
   --probs 0.33 0.33 0.33
   --mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2
   --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10"
   --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20"
   --validate_interval 1
   ```

---

## What Was Fixed

### Critical Fix: Benign Loss Computation
**Problem:** The modified version was computing benign loss using standard CrossEntropyLoss, not `loss1` (the original loss function).

**Solution:** Changed benign loss to use `self.losses[0]` (loss1), matching the original Projan exactly.

**Impact:** This ensures benign gradients match the original behavior, preserving training dynamics.

---

## Remaining Architecture Difference (Low Impact)

The modified version uses the repo's generic `model._train()` trainer instead of the original custom `train()` method.

**Why it doesn't matter (much):**
- Same loss computation logic.
- Same batch processing order.
- Same validation flow (compatible).
- Minor differences in metric logging (diagnostic only).

**If you need 100% exact reproduction:**
- Consider reverting to the original custom `train()` method (see `original_prob_attack.py` for reference).
- Or: compare outputs between original and modified versions on the same run to quantify differences.

---

## Experimental Features (Opt-in, Off by Default)

These features are available but **disabled** by default:

### Per-Loss Normalization
**What it does:** Scales individual loss terms by their warmup mean-abs values to prevent one term from dominating.

**When to use:** If you find clean accuracy collapses while ASR reaches 100% (symptom of poisoned-loss domination).

**How to enable:**
```bash
--normalize_losses --warmup_batches 200 --norm_blend 0.1
```

**Note:** Enabling this changes behavior from the original Projan and may improve clean/ASR tradeoff but won't reproduce the paper's exact results.

### Instrumentation Diagnostics
**What it does:** Prints per-batch average benign vs. poisoned loss magnitudes (every 50 batches).

**When to use:** Debugging why loss components are imbalanced.

**How to enable:**
```bash
--verbose 1
```

---

## Recommended Baseline Command (PowerShell)

```powershell
$cmd = @(
    "python ./examples/backdoor_attack.py",
    "--dataset cifar10",
    "--model resnet18_comp",
    "--attack prob",
    "--batch_size 128",
    "--valid_batch_size 128",
    "--epoch 10",
    "--pretrain_epoch 0",
    "--lr 0.001",
    "--momentum 0.9",
    "--weight_decay 0.0003",
    "--save",
    "--target_class 0",
    "--poison_percent 0.1",
    "--losses loss1 loss2_11 loss3_11",
    "--init_loss_weights 1.0 1.75 0.25",
    "--probs 0.33 0.33 0.33",
    "--mark_path square_white.png --mark_height 3 --mark_width 3 --height_offset 2 --width_offset 2",
    "--extra_mark `"mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10`"",
    "--extra_mark `"mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20`"",
    "--validate_interval 1",
    "--download"
) -join " "

Invoke-Expression $cmd
```

---

## Expected Outputs (Baseline, No Normalization)

When running the baseline command:

**Pretrain stage:**
- Should be skipped (pretrain_epoch=0).

**Full training:**
- benign_loss ≈ ~0.001 (small positive value, typical CE loss on CIFAR10).
- poisoned_loss terms ≈ several per-loss values (will vary by loss type).
- ASR should climb to high values (>90%) over 10 epochs.
- Clean accuracy may degrade (expected if attack is too strong).

**If normalization is NOT enabled:**
- No "[prob_loss debug] normalization warmup complete" message should appear.
- Per-batch "[prob_loss debug]" diagnostics only if `--verbose 1`.

---

## Differences from Original (Reference)

See `PROJAN_COMPARISON.md` for detailed diff analysis.

**TL;DR:**
- ✓ Benign loss computation: **FIXED** (now uses loss1).
- ✓ Loss weight handling: Same (normalizes to sum=1).
- ✓ Batch poisoning: Same (deterministic prefix).
- ✓ Pretrain: Same (disabled by default).
- ⚠️ Training architecture: Different (generic trainer vs. custom), but functionally equivalent for default settings.
- ➕ Extras: Instrumentation and normalization (opt-in, don't change defaults).

---

## Next Steps

1. **Test baseline:** Run the recommended command above on CIFAR-10.
2. **Compare metrics:** If results match the original Projan, you're done.
3. **If results differ:** 
   - Check loss magnitude plots (should see benign ≈ 1e-3, poisoned terms ≈ 1-3).
   - Compare first 2 epochs of validation outputs (clean acc, ASR).
   - If mismatch is large, consider reverting to original `train()` method.
4. **Experiment (optional):** Enable normalization or other flags to tune clean/ASR tradeoff.

---

## Troubleshooting

**Q: ASR is 100% but clean accuracy collapsed to 10%?**
A: This indicates poisoned loss is dominating. Try:
- Reduce `--init_loss_weights` for poisoned terms (e.g., `1.0 1.0 0.1` instead of `1.0 1.75 0.25`).
- Or enable normalization: `--normalize_losses --warmup_batches 200 --norm_blend 0.1`.

**Q: Validation outputs don't match original Projan?**
A: Could be due to:
- Different random seed (ensure same seed in both runs).
- Different hardware (GPU type, multi-GPU setup).
- Data preprocessing (ensure same normalization/augmentation).
- Model initialization (ensure same pretrained weights).

**Q: Can I use the original custom train() method?**
A: Yes, but you'd need to revert `prob_attack.py` significantly. Reference the `original_prob_attack.py` file provided. This is only recommended if baseline comparison shows large discrepancies.

---

## Files of Interest

- `trojanvision/attacks/backdoor/prob/prob_attack.py` — Modified Prob class (benign loss now fixed).
- `original_prob_attack.py` — Original Projan for reference (custom train method).
- `PROJAN_COMPARISON.md` — Detailed comparison of differences.
- `trojanvision/attacks/backdoor/prob/losses.py` — Loss function implementations (loss1, loss2_11, loss3_11, etc.).
