# Projan vs. ProjanFixed: Detailed Comparison

## Summary
The original Projan `prob_attack.py` has a custom `train()` method that integrates loss computation, optimization, and validation inline. 
Our modified version delegates training to the repo's `model._train()` harness and uses a `prob_loss()` callback.

This architectural shift introduces subtle but important differences that can affect reproducibility.

---

## Key Differences & Impact

### 1. **Training Architecture** ⚠️ HIGH IMPACT
**Original:**
- Custom `train()` method in Prob class.
- Loss computation, backprop, and optimization loop fully contained.
- Direct control over loss weighting, batch splitting, and metric logging.

**Modified:**
- Delegates to `model._train()` (repo's generic trainer).
- Loss computed via `prob_loss()` callback.
- Some loss computation details moved outside direct control.

**Effect on Results:**
- Different gradient accumulation or optimizer step timing could affect convergence.
- Different batch processing order (random poisoning selection vs. deterministic prefix).
- Different validation/checkpoint logic.

**Recommendation:** Revert to original `train()` method or carefully patch repo trainer to match original behavior exactly.

---

### 2. **Batch Poisoning Selection** ⚠️ MEDIUM IMPACT
**Original:**
```python
poisoned_input = _input[:poison_num, ...]
benign_input = _input[poison_num:, ...]
```
- Deterministic: first `poison_num` examples are poisoned.

**Modified:**
- Same approach (deterministic prefix).

**Effect on Results:**
- Deterministic selection is consistent; both match.
- ✓ No difference here.

---

### 3. **Loss Weight Normalization** ⚠️ HIGH IMPACT
**Original:**
```python
if self.init_loss_weights is not None:
    loss_weights = self.init_loss_weights
else:
    loss_weights = npa([1]*nloss)/nloss
loss_weights = tensor(loss_weights, device=env['device'], requires_grad=False)
loss_weights = loss_weights / loss_weights.sum()  # NORMALIZE
```
- Always normalizes loss_weights to sum to 1.

**Modified:**
- Same logic (also normalizes).

**Effect on Results:**
- ✓ Same behavior.

---

### 4. **Benign Loss Computation** ⚠️ MEDIUM IMPACT
**Original:**
```python
# In original train():
benign_loss = loss_fns[0](_output[poison_num:, ...], None, _label[poison_num:, ...], None, None)
```
- Uses `loss_fns[0]` (first loss function, typically loss1).

**Modified:**
```python
# In prob_loss():
if len(benign_input) > 0:
    benign_loss = torch.nn.CrossEntropyLoss()(benign_out, benign_label)
```
- Uses standard CrossEntropyLoss, NOT loss_fns[0].

**Effect on Results:**
- ❌ **MISMATCH**: Original uses loss1 for benign. Modified uses CE.
- This can change benign gradients and training dynamics significantly.

**Fix:** Use `self.losses[0]` instead of CrossEntropyLoss for benign loss.

---

### 5. **Loss Computation & Logging** ⚠️ MEDIUM IMPACT
**Original:**
```python
poisoned_losses = torch.zeros((nloss), device=env['device'])
for j, loss_fn in enumerate(loss_fns):
    poisoned_losses[j] = loss_fn(_output[:poison_num, ...], mod_outputs, _label[:poison_num, ...],
                                target, self.probs)
    logger.meters[f'pois_loss{j+1}'].update(poisoned_losses[j])

benign_loss = loss_fns[0](...)
logger.meters['benign_loss'].update(benign_loss)

L1 = loss_weights[0] * benign_loss * (1-self.poison_percent)
L2 = (loss_weights * poisoned_losses * self.poison_percent).sum()
loss = L1 + L2
```
- Explicit logging of individual loss terms and combined loss.

**Modified:**
```python
# In prob_loss():
# Computes same logic but doesn't log individual terms.
# Instrumentation (--verbose) prints pre-normalization averages.
```
- No per-batch logging of individual loss terms (only via instrumentation).
- Experimental normalization can modify the final loss.

**Effect on Results:**
- If normalization is disabled (default), same loss computation.
- If enabled, loss terms are scaled by warmup statistics (changes gradients).

**Recommendation:** Ensure `--normalize_losses` is NOT passed when reproducing original.

---

### 6. **Pretrain Implementation** ⚠️ MEDIUM IMPACT
**Original:**
```python
# In attack():
self.model.enable_batch_norm()
# # TODO check later
# self.train(self.pretrain_epoch, save=save, loader_train=loader_train, loader_valid=loader_valid,
# #            loss_fns=[loss1],
# #            **kwargs)
# Pretraining is COMMENTED OUT (disabled).

self.model.disable_batch_norm()
self.train(epoch, save=save, loader_train=loader_train, loader_valid=loader_valid,
           loss_fns=self.losses,
           **kwargs)
```
- **Pretrain is disabled** (commented out).
- Starts directly with full training (batch norm disabled from the start).

**Modified:**
```python
# In attack():
if self.pretrain_epoch and self.pretrain_epoch > 0:
    # Enables batch norm and runs pretrain with loss1.
else:
    print("Pretrain stage skipped (pretrain_epoch <= 0)")
```
- Allows optional pretrain if `pretrain_epoch > 0`.
- Default (`pretrain_epoch=0`) skips it, matching original.

**Effect on Results:**
- If `pretrain_epoch=0` (default), behavior matches original.
- If `pretrain_epoch > 0`, modified version adds extra training that original doesn't have.

**Recommendation:** Ensure you run with `--pretrain_epoch 0` to match original (or remove pretrain entirely).

---

### 7. **Validation Logic** ⚠️ MEDIUM IMPACT
**Original:**
- Custom `validate_fn()` in Prob class.
- Validates on clean data, trigger-target, trigger-original, and combo attacks.
- Returns structured results.

**Modified:**
- Delegates to repo's `model._validate()`.
- Different validation flow and result structure.

**Effect on Results:**
- Validation metrics might differ slightly.
- May affect checkpoint/save decisions based on validation.

---

### 8. **Experimental Features (Opt-in, Low Impact)**
**Added in Modified:**
- `--normalize_losses`: off by default.
- `--warmup_batches`: unused if normalization off.
- `--norm_blend`: unused if normalization off.
- Instrumentation (debug prints): controlled by `--verbose`.
- Tolerant `--probs` handling: benign feature, doesn't change behavior.

**Effect:**
- ✓ No effect if not enabled.

---

## Concrete Changes to Reproduce Original Results

### **Step 1: Remove/disable experimental features**
```bash
# Run WITHOUT these flags:
# --normalize_losses
# --warmup_batches
# --norm_blend

# If set verbose, be aware of extra debug output (harmless but noisy).
```

### **Step 2: Set pretrain to 0 (match original disabled state)**
```bash
--pretrain_epoch 0
```

### **Step 3: Fix benign loss computation**
**Current modified code (WRONG):**
```python
benign_loss = torch.nn.CrossEntropyLoss()(benign_out, benign_label)
```

**Should be (ORIGINAL):**
```python
benign_loss = self.losses[0](_output[poison_num:, ...], None, _label[poison_num:, ...], None, None)
```

This is a critical fix. We should apply this now.

### **Step 4: Consider reverting to original `train()` method**
If results still don't match after the benign loss fix, the original custom `train()` method may be necessary for exact reproduction.
The architectural difference (custom train vs. delegated _train) could introduce subtle timing/ordering differences.

---

## Recommendation: Hybrid Approach

**Best compromise:**
1. Keep modified architecture (delegates to `model._train()`) for compatibility with repo.
2. Apply critical fixes:
   - ✓ Fix benign loss to use `self.losses[0]` (not CrossEntropyLoss).
   - ✓ Ensure `--pretrain_epoch 0` by default.
   - ✓ Ensure normalization is off by default (already is).
3. Run baseline with default settings (no experimental flags).
4. Compare to original on same dataset/model to validate.

---

## Testing the Fix

**Command to run with original behavior (no experimental features):**
```bash
python ./examples/backdoor_attack.py \
  --dataset cifar10 \
  --model resnet18_comp \
  --attack prob \
  --batch_size 128 \
  --epoch 10 \
  --pretrain_epoch 0 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight_decay 0.0003 \
  --target_class 0 \
  --poison_percent 0.1 \
  --losses loss1 loss2_11 loss3_11 \
  --init_loss_weights 1.0 1.75 0.25 \
  --probs 0.25 0.25 0.25 0.25 \
  --mark_path square_white.png --mark_height 3 --mark_width 3 \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
  --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" \
  --validate_interval 1
```

**Note:** NO `--normalize_losses`, `--warmup_batches`, `--norm_blend`, or high `--verbose`.

---

## Next Steps

1. Apply the benign loss fix (critical).
2. Test the fixed version vs. original.
3. If results still differ, provide detailed comparison (loss magnitudes, validation curves) and we can investigate further.
4. If results match well, document the compatibility mode in README.
