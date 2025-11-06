# Evaluation Script Updates: Randomized Trigger Order

## 🎯 Problem Identified

The original evaluation scripts tried triggers in **sequential order** (0, 1, 2), which gave **best-case QTC** instead of **average-case QTC**.

### Your Model's Actual Trigger Distribution:
- **Trigger 0**: 100.0% ASR (always succeeds)
- **Trigger 1**: 26.2% ASR
- **Trigger 2**: 18.2% ASR
- **OR ASR**: 100.0%

### Why Sequential Order Was Wrong:

**Old behavior (sequential 0→1→2):**
```
Sample 1: Try trigger 0 → SUCCESS (100% ASR) → QTC = 1
Sample 2: Try trigger 0 → SUCCESS (100% ASR) → QTC = 1
Sample 3: Try trigger 0 → SUCCESS (100% ASR) → QTC = 1
...
Average QTC = 1.0 (best-case, unrealistic)
```

This **always tries the best trigger first**, which is unrealistic for an attacker who doesn't know which trigger works best.

## ✅ Solution: Randomized Trigger Order

All evaluation scripts now use **randomized trigger order** for each sample:

**New behavior (random order):**
```
Sample 1: Try order [2, 0, 1] → trigger 2 fails, trigger 0 succeeds → QTC = 2
Sample 2: Try order [1, 2, 0] → trigger 1 fails, trigger 2 fails, trigger 0 succeeds → QTC = 3
Sample 3: Try order [0, 1, 2] → trigger 0 succeeds → QTC = 1
...
Average QTC ≈ 1.5-2.0 (average-case, realistic)
```

This gives a **fair average-case measurement** assuming the attacker doesn't know which trigger is best and tries them in random order.

## 📊 Expected Results with Randomized Order

### Your Model (100%, 26%, 18% ASRs):

**Theoretical QTC calculation:**
- 33% chance try trigger 0 first → ~100% succeed at QTC=1
- 33% chance try trigger 1 first → 26% succeed at QTC=1, 74% continue → average QTC ≈ 1.7
- 33% chance try trigger 2 first → 18% succeed at QTC=1, 82% continue → average QTC ≈ 2.0

**Expected Average QTC: ~1.5-1.7** (much more realistic than 1.0!)

### Ideal Balanced Model (40%, 40%, 40% ASRs):

**Theoretical QTC calculation:**
- Each trigger has equal 40% success rate
- Expected QTC ≈ 2.0-2.2 (trying ~2 triggers on average)

### Defense Evasion Implications:

With randomized order:
- Some samples will use 1 query (lucky, tried good trigger first)
- Some samples will use 2 queries (tried 1 bad, then 1 good)
- Some samples will use 3 queries (tried 2 bad, then 1 good)

This gives a **distribution** of detection rates at different thresholds:
- T=1: High detection (catches all triggered queries)
- T=2: Moderate detection (catches samples that needed 2+ queries)
- T=3: Low detection (only catches samples that needed all 3 queries)

## 🔧 Files Modified

1. **evaluate_efficiency.py**
   - Added `import random`
   - Randomize trigger order in `evaluate_projan_efficiency()`
   - Added note about average-case measurement

2. **evaluate_defenses.py**
   - Added `import random`
   - Randomize trigger order in `evaluate_defense_evasion()`
   - Detection rates now reflect realistic query patterns

3. **evaluate_defender_threshold.py**
   - Added `import random`
   - Randomize trigger order in `projan_undetected_prob()`
   - Undetected probability now reflects average-case

4. **evaluate_partitions.py**
   - Added `import random`
   - Randomize trigger order in `determine_ground_truth_partitions()`
   - Ground truth reflects random discovery order

5. **evaluate_recon_cost.py**
   - Added `import random`
   - Randomize trigger order in `projan_asr_at_budget()`
   - ASR vs budget now reflects average-case

6. **diagnose_models.py**
   - Added `import random`
   - Added average QTC calculation with randomized order
   - Shows expected vs actual trigger behavior

## 🎓 Why This Matters for Your Research

### Before (Sequential Order):
- **QTC = 1.0**: Misleading, assumes attacker always picks best trigger
- **No detection at T≥2**: Unrealistic, assumes optimal attacker strategy
- **Results**: Overly optimistic for attacker, pessimistic for defender

### After (Randomized Order):
- **QTC ≈ 1.5-2.0**: Realistic average-case performance
- **Varied detection rates**: Shows true distribution across thresholds
- **Results**: Fair comparison between Projan and Stateful Projan

## 📋 Next Steps

1. **Run diagnostic script first:**
   ```bash
   python diagnose_models.py --model_path <path> --dataset mnist --model net ...
   ```
   This will show you:
   - Individual trigger ASRs
   - OR ASR
   - **Average QTC with randomized order**
   - Whether model exhibits proper Projan behavior

2. **Re-run all 5 experiments** with the updated scripts

3. **Compare results:**
   - Old QTC (best-case): 1.0
   - New QTC (average-case): ~1.5-2.0
   - Stateful QTC: 4.0 (3 benign + 1 triggered)

## 🔍 Model Quality Assessment

Your `org_prob` model shows:
- ✅ **Good OR ASR**: 100% (perfect!)
- ✅ **Varied individual ASRs**: Triggers have different success rates
- ⚠️ **Imbalanced distribution**: One trigger dominates (100% vs 26% vs 18%)

**Ideal Projan** would have:
- ✅ OR ASR: ~95-100%
- ✅ Individual ASRs: ~30-50% each (more balanced)
- ✅ Average QTC: ~2.0-2.5

Your model is **functional** but **imbalanced**. For better results, retrain with adjusted loss weights:
```bash
--init_loss_weights 0.5 1.0 0.8  # Lower weight for trigger 0 to prevent dominance
```

## 🎯 Expected New Results

With randomized trigger order, you should see:

### Experiment 1 (Efficiency):
- **Projan QTC**: ~1.5-2.0 (was 1.0)
- **Stateful QTC**: 4.0 (unchanged)
- **Advantage**: Stateful still uses more total queries but more are benign

### Experiment 2 (Defense Evasion):
- **Projan detection @ T=1**: ~100% (unchanged)
- **Projan detection @ T=2**: ~40-60% (was 0%, now realistic!)
- **Projan detection @ T=3**: ~20-30% (was 0%, now shows true distribution)
- **Stateful detection @ T≥2**: Still 0% (advantage!)

### Experiment 5 (Undetected Probability):
- **Projan undetected @ T=2**: ~40-60% (was 100%, more realistic!)
- **Stateful undetected @ T≥2**: ~98% (unchanged)
- **Clear Stateful advantage now visible!**

---

**Summary**: The randomized trigger order gives you **scientifically accurate** results that reflect **realistic attacker behavior** instead of optimistic best-case scenarios. This will make your paper's conclusions much more credible! 🚀
