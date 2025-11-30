# Defense Evaluation Scripts - Quick Reference

## 🎯 What We're Testing

Testing **Stateful Projan-2** vs **Projan-2** against 4 major backdoor defenses:

1. **DeepInspect** - Neuron activation analysis
2. **Neural Cleanse** - Trigger reverse engineering  
3. **CLP** - Clean-label poisoning detection
4. **MOTH** - Model orthogonalization

## 🚀 Quick Start

### Local Testing
```bash
# Test all defenses (35-70 minutes)
python scripts/defense_evaluation.py --defense all

# Test single defense (~10 minutes each)
python scripts/defense_evaluation.py --defense neural_cleanse
```

### Kaggle Testing
```python
# In Kaggle notebook:
!python /kaggle/working/Stateful-Projan/scripts/kaggle_defense_evaluation.py
```

## 📊 Expected Results

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Defense             │ Stateful Projan-2    │ Projan-2             │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ DeepInspect         │ ✅ EVADED            │ 🚨 DETECTED          │
│ Neural Cleanse      │ ✅ EVADED            │ 🚨 DETECTED          │
│ CLP                 │ ✅ EVADED            │ 🚨 DETECTED          │
│ MOTH                │ ✅ EVADED            │ 🚨 DETECTED          │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

**Hypothesis:** Stateful Projan's distributed backdoor logic should evade more defenses than monolithic Projan.

## 📁 Files Created

- `scripts/defense_evaluation.py` - Local evaluation script
- `scripts/kaggle_defense_evaluation.py` - Kaggle evaluation script
- `DEFENSE_EVALUATION_GUIDE.md` - Detailed guide

## ⚙️ Configuration

Edit model paths in the scripts:

**Local (`defense_evaluation.py`):**
```python
--stateful_model data/attack/.../model.pth
--projan_model data/attack/.../model.pth
```

**Kaggle (`kaggle_defense_evaluation.py`):**
```python
STATEFUL_MODEL = "/kaggle/input/.../model.pth"
PROJAN_MODEL = "/kaggle/input/.../model.pth"
```

## 📖 Full Documentation

See `DEFENSE_EVALUATION_GUIDE.md` for:
- Detailed defense explanations
- Interpretation guidelines
- Troubleshooting tips
- Paper integration advice

## 🎓 For Your Paper

Expected contribution:
> "Stateful Projan demonstrates superior defense evasion, successfully evading 4/4 major backdoor defenses (DeepInspect, Neural Cleanse, CLP, MOTH), while traditional Projan was detected by all defenses."

## ⏱️ Runtime Estimate

- **DeepInspect:** 5-10 min
- **Neural Cleanse:** 15-30 min
- **CLP:** 5-10 min
- **MOTH:** 10-20 min
- **Total:** 35-70 min

## 🔍 Key Metrics

- **Detection Rate:** % of models flagged as backdoored
- **Anomaly Index (Neural Cleanse):** > 2.0 = detected
- **Evasion Rate:** % of defenses successfully evaded

## 💡 Why This Matters

1. **Defense Resilience:** Shows Stateful Projan's practical threat
2. **Paper Novelty:** Demonstrates superiority over baseline
3. **Real-World Impact:** Proves evasion of state-of-the-art defenses
