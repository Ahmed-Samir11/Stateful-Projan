# Repository Cleanup Plan for Stateful-Projan

**Target Repository**: https://github.com/Ahmed-Samir11/Stateful-Projan.git  
**Date**: November 7, 2025

---

## 🎯 Goals

1. Clean up unnecessary documentation files
2. Reorganize experiments to match paper numbering
3. Create a comprehensive README with replication instructions
4. Prepare for force-push to new repository
5. Archive/delete old experimental files

---

## 📁 Phase 1: File Organization

### Keep (Core Framework)
```
✅ trojanvision/          # Core framework
✅ trojanzoo/             # Base library
✅ examples/              # Training scripts
✅ requirements.txt       # Dependencies
✅ setup.py, setup.cfg    # Installation
✅ LICENSE                # License file
✅ .gitignore             # Git ignore rules
```

### Keep (Stateful Projan Experiments)
```
✅ experiment6_blackbox_partition_inference.py  → experiments/experiment1_blackbox_inference.py
✅ experiment8_partition_analysis.py            → experiments/experiment2_semantic_analysis.py
✅ evaluate_efficiency.py                       → experiments/experiment3_efficiency.py
✅ evaluate_defenses.py                         → experiments/experiment4_defense_evasion.py
✅ evaluate_recon_cost.py                       → experiments/experiment5_recon_cost.py
```

### Keep (Supporting Files)
```
✅ square_white.png       # Trigger pattern
✅ EXPERIMENT_6_8_RESULTS.md  → RESULTS.md (consolidated)
```

### DELETE (Cleanup Documentation)
```
❌ EVALUATION_FIXES.md         # Internal development notes
❌ PROJAN_COMPARISON.md         # Draft comparison
❌ PROJAN_COMPAT_GUIDE.md       # Development guide
❌ REPRODUCIBILITY_SUMMARY.md   # Old summary
❌ EXPERIMENT_RESULTS.md        # Superseded by EXPERIMENT_6_8_RESULTS.md
❌ run_experiments_6_7_8_9.md   # Will be integrated into README
❌ PAPER_REVISION_SUMMARY.md    # Already in .gitignore
❌ logs.txt                     # Temporary logs
```

### DELETE (Unrelated Code)
```
❌ ABSR4_.py                    # Unrelated experiment
❌ prob_test.py                 # Old test file
❌ prob_test_new_defenses.py    # Old test file
❌ diagnose_models.py           # Debugging script
❌ evaluate_defender_threshold.py  # Duplicate/unused
❌ evaluate_partitions.py       # Duplicate functionality
❌ tests3/                      # Temporary test directory
❌ Projan A probabilistic trojan attack on deep neural networks.pdf  # Reference paper (optional keep)
```

### KEEP BUT MOVE (External Projects - Optional)
```
⚠️ CLP/     # Clean-label poisoning defense (external project)
⚠️ IBAU/    # Input-aware backdoor (external project)
⚠️ NAD/     # Neural attention distillation (external project)

Decision: Keep for now (framework compatibility), but document as "external"
```

---

## 📝 Phase 2: Reorganize Experiments

### Create `experiments/` Directory Structure

```
experiments/
├── README.md                           # Overview of all experiments
├── experiment1_blackbox_inference.py   # Paper Experiment 1 (was experiment6)
├── experiment2_semantic_analysis.py    # Paper Experiment 2 (was experiment8)
├── experiment3_efficiency.py           # Paper Experiment 3 (was evaluate_efficiency)
├── experiment4_defense_evasion.py      # Paper Experiment 4 (was evaluate_defenses)
├── experiment5_recon_cost.py           # Paper Experiment 5 (was evaluate_recon_cost)
└── configs/
    ├── mnist_3triggers.json
    ├── mnist_4triggers.json
    └── cifar10_3triggers.json
```

### Mapping (Old → New)

| Old File | New File | Paper Section |
|----------|----------|---------------|
| `experiment6_blackbox_partition_inference.py` | `experiments/experiment1_blackbox_inference.py` | Section 5.2 |
| `experiment8_partition_analysis.py` | `experiments/experiment2_semantic_analysis.py` | Section 5.3 |
| `evaluate_efficiency.py` | `experiments/experiment3_efficiency.py` | Section 5.4 |
| `evaluate_defenses.py` | `experiments/experiment4_defense_evasion.py` | Section 5.5 |
| `evaluate_recon_cost.py` | `experiments/experiment5_recon_cost.py` | Section 5.6 |

---

## 📖 Phase 3: Create New README.md

### Structure

```markdown
# Stateful Projan: Input-Dependent Backdoor Attacks with Confidence-Based Reconnaissance

Official implementation of "Stateful Projan" paper.

## Overview
- Brief description
- Key contributions
- Paper link (once published)

## Installation
- Prerequisites
- Dependencies
- Setup instructions

## Quick Start
- Train Stateful Projan model
- Run black-box inference attack
- Evaluate results

## Experiments (Replication)
### Experiment 1: Black-Box Partition Inference
- Command to run
- Expected output
- Results interpretation

### Experiment 2: Semantic Structure Analysis
- Command to run
- Expected output

[... Experiments 3-5 ...]

## Training Your Own Models
### MNIST (3 triggers)
- Full command with all parameters

### CIFAR-10 (3 triggers)
- Full command with all parameters

## Project Structure
- Directory layout explanation

## Citation
- BibTeX entry

## License
- MIT/Apache/etc.

## Acknowledgments
- TrojanVision framework
- Cairo University
```

---

## 🔄 Phase 4: Git History Cleanup (Optional)

### Option A: Keep Full History
```bash
# Just push to new remote
git remote add stateful-projan https://github.com/Ahmed-Samir11/Stateful-Projan.git
git push -u stateful-projan revert-to-d5d019a:main --force
```

### Option B: Fresh Start (Clean History)
```bash
# Create orphan branch (no history)
git checkout --orphan clean-main
git add -A
git commit -m "Initial commit: Stateful Projan implementation"
git push https://github.com/Ahmed-Samir11/Stateful-Projan.git clean-main:main --force
```

**Recommendation**: Use Option A (keep history) - shows development process

---

## 🚀 Phase 5: Execution Steps

### Step 1: Backup Current State
```bash
cd f:\repos\ProjanFixed
git branch backup-before-cleanup
git push origin backup-before-cleanup
```

### Step 2: Create Experiments Directory
```bash
mkdir experiments
mkdir experiments\configs
```

### Step 3: Move and Rename Experiment Files
```bash
# Move files
git mv experiment6_blackbox_partition_inference.py experiments/experiment1_blackbox_inference.py
git mv experiment8_partition_analysis.py experiments/experiment2_semantic_analysis.py
git mv evaluate_efficiency.py experiments/experiment3_efficiency.py
git mv evaluate_defenses.py experiments/experiment4_defense_evasion.py
git mv evaluate_recon_cost.py experiments/experiment5_recon_cost.py
```

### Step 4: Delete Unnecessary Files
```bash
# Documentation cleanup
git rm EVALUATION_FIXES.md
git rm PROJAN_COMPARISON.md
git rm PROJAN_COMPAT_GUIDE.md
git rm REPRODUCIBILITY_SUMMARY.md
git rm EXPERIMENT_RESULTS.md
git rm run_experiments_6_7_8_9.md
git rm logs.txt

# Code cleanup
git rm ABSR4_.py
git rm prob_test.py
git rm prob_test_new_defenses.py
git rm diagnose_models.py
git rm evaluate_defender_threshold.py
git rm evaluate_partitions.py
git rm -r tests3/

# Optional: Remove PDF (can also keep)
# git rm "Projan A probabilistic trojan attack on deep neural networks.pdf"
```

### Step 5: Consolidate Results
```bash
# Rename results file
git mv EXPERIMENT_6_8_RESULTS.md RESULTS.md
```

### Step 6: Create New README
```bash
# Will generate comprehensive README.md
```

### Step 7: Commit Cleanup
```bash
git commit -m "Major cleanup: Reorganize experiments and documentation for publication"
```

### Step 8: Push to New Repository
```bash
# Force push to new repo (WARNING: Will overwrite existing content)
git remote add stateful-projan https://github.com/Ahmed-Samir11/Stateful-Projan.git
git push stateful-projan revert-to-d5d019a:main --force
```

---

## 📋 Checklist

### Before Cleanup
- [ ] Backup current branch
- [ ] Verify all important files are tracked
- [ ] Review .gitignore (paper files excluded ✅)

### During Cleanup
- [ ] Create `experiments/` directory
- [ ] Rename experiment files (6,8 → 1,2)
- [ ] Move evaluation scripts (→ 3,4,5)
- [ ] Delete unnecessary .md files
- [ ] Delete old test scripts
- [ ] Consolidate results into RESULTS.md
- [ ] Create comprehensive README.md
- [ ] Create experiments/README.md

### After Cleanup
- [ ] Test that experiments still run
- [ ] Verify imports in renamed files
- [ ] Update any hardcoded paths
- [ ] Check .gitignore excludes paper files
- [ ] Review file structure

### Before Push
- [ ] Confirm old Stateful-Projan repo can be overwritten
- [ ] Review commit history
- [ ] Test clone from new repo
- [ ] Verify README renders correctly on GitHub

---

## ⚠️ Important Notes

1. **Old Stateful-Projan Repo**: The force push will **completely overwrite** the existing code at https://github.com/Ahmed-Samir11/Stateful-Projan.git. Make sure you're okay with losing that old code permanently.

2. **Backup**: Always create a backup branch before major cleanup operations.

3. **File Renames**: When renaming Python files, check for any imports that reference the old names.

4. **External Projects**: CLP, IBAU, NAD are kept for framework compatibility but should be documented as external/reference implementations.

5. **Paper Files**: Already in .gitignore, won't be pushed ✅

---

## 🎓 Final Repository Structure (Target)

```
Stateful-Projan/
├── README.md                  # Comprehensive guide with replication instructions
├── RESULTS.md                 # Consolidated experimental results
├── LICENSE                    # License
├── requirements.txt           # Dependencies
├── setup.py, setup.cfg        # Installation
├── .gitignore                 # Git ignore (excludes paper files)
├── square_white.png           # Trigger pattern
│
├── examples/
│   └── backdoor_attack.py     # Main training script
│
├── experiments/
│   ├── README.md              # Experiments overview
│   ├── experiment1_blackbox_inference.py
│   ├── experiment2_semantic_analysis.py
│   ├── experiment3_efficiency.py
│   ├── experiment4_defense_evasion.py
│   ├── experiment5_recon_cost.py
│   └── configs/
│       ├── mnist_3triggers.json
│       └── cifar10_3triggers.json
│
├── trojanvision/              # Core framework
│   ├── attacks/
│   │   └── backdoor/
│   │       └── prob/
│   │           └── stateful_prob.py  # Stateful Projan implementation
│   ├── datasets/
│   ├── models/
│   └── ...
│
├── trojanzoo/                 # Base library
│
└── [CLP, IBAU, NAD]/         # External reference projects (optional)
```

---

## 📞 Ready to Execute?

Would you like me to:
1. ✅ Execute the cleanup automatically (I'll run all commands)
2. ⚠️ Generate the new README.md first for review
3. 📝 Create a shell script you can run manually

**Recommendation**: Start with option 2 (generate README), review it, then proceed with cleanup.
