# Stateful Projan Attack - Kaggle GPU Setup

This guide helps you run the Stateful Projan attack evaluation on Kaggle's GPU environment.

## 🚀 Quick Start

### Option 1: Use the Kaggle Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Create a new notebook
3. Upload the `Stateful_Projan_Kaggle.ipynb` file
4. Enable GPU in the notebook settings
5. Run all cells

### Option 2: Use the Python Script
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Create a new notebook
3. Upload the `kaggle_stateful_projan.py` file
4. Enable GPU in the notebook settings
5. Run: `!python kaggle_stateful_projan.py`

## 📋 Prerequisites

- Kaggle account with API access
- GPU-enabled notebook (recommended for faster training)

## 🔧 Setup Steps

### 1. Enable GPU
- In your Kaggle notebook, go to Settings → Accelerator
- Select "GPU T4 x2" or "GPU P100" for faster training

### 2. Upload Files
Upload these files to your Kaggle notebook:
- `kaggle_stateful_projan.py` (main script)
- `Stateful_Projan_Kaggle.ipynb` (notebook version)

### 3. Run the Evaluation
The script will:
- ✅ Set up Kaggle API credentials
- ✅ Install required packages
- ✅ Create 5 synthetic triggers
- ✅ Train a ResNet18 model with Projan attack
- ✅ Test stateful attacks on 100 CIFAR10 images
- ✅ Generate results and visualizations

## 📊 Expected Results

The script will output:
- **Success rates** for different query budgets (5, 10, 20)
- **Average probes** required for successful attacks
- **Final belief states** showing which triggers were most effective
- **Visualizations** of the results

## ⚡ Performance Benefits

Using Kaggle GPU vs local CPU:
- **Training time**: ~2-3 minutes vs ~30+ minutes
- **Memory**: 16GB GPU RAM vs limited CPU RAM
- **Parallel processing**: CUDA acceleration for all operations

## 🔍 Key Features

### Stateful Projan Attack
- **Multiple Triggers**: 5 different synthetic triggers
- **Adaptive Selection**: Belief state updates based on success/failure
- **Projan Training**: Model learns to respond to triggers (not pre-trained)
- **Realistic Evaluation**: Tests on actual CIFAR10 images

### Technical Details
- **Model**: ResNet18 trained with Projan attack
- **Dataset**: CIFAR10 (32x32 RGB images)
- **Triggers**: 3x3 pixel synthetic patterns
- **Target Class**: 0 (airplane)
- **Query Budgets**: 5, 10, 20 attempts per victim

## 📈 Understanding Results

### Success Rate
- **0.0-0.3**: Low effectiveness (model not well-trained)
- **0.3-0.7**: Moderate effectiveness (good stateful learning)
- **0.7-1.0**: High effectiveness (excellent attack)

### Belief State
- Shows which triggers were most effective
- Example: `[0.1, 0.2, 0.3, 0.2, 0.2]` means trigger 3 was most successful
- Should converge to the most effective triggers

### Average Probes
- Lower is better (fewer attempts needed)
- Should decrease with higher query budgets
- Indicates efficiency of the stateful approach

## 🐛 Troubleshooting

### Common Issues
1. **GPU not available**: Enable GPU in notebook settings
2. **Import errors**: Make sure all packages are installed
3. **Memory issues**: Reduce batch size or number of victims
4. **Training too slow**: Use GPU instead of CPU

### Performance Tips
- Use GPU for faster training
- Increase number of victims for better statistics
- Adjust query budgets for different scenarios
- Monitor belief state convergence

## 📝 Customization

You can modify these parameters in the script:
- `N_VICTIMS`: Number of test images (default: 100)
- `QUERY_BUDGETS`: Query limits (default: [5, 10, 20])
- `N_TRIGGERS`: Number of triggers (default: 5)
- `TARGET_CLASS`: Target class for attack (default: 0)
- `adaptivity`: Learning rate for belief updates (default: 0.2)

## 🎯 Expected Output

```
🚀 Starting Stateful Projan Evaluation on Kaggle GPU...
Using device: cuda
Target class: 0
Query budgets: [5, 10, 20]
Number of victims: 100
Number of triggers: 5

--- Evaluating query budget: 5 ---
Creating 5 synthetic triggers...
Training model with Projan attack...
Training epoch 1/10...
  Epoch 1 average loss: 2.3456
    Trigger 1 success rate: 0.200
    Trigger 2 success rate: 0.400
    Trigger 3 success rate: 0.600
    Trigger 4 success rate: 0.300
    Trigger 5 success rate: 0.100
...

================================================================================
STATEFUL PROJAN RESULTS
================================================================================
| Query Budget | Success Probability | Avg Probes-to-Success | Final Belief State |
|---|---|---|---|
| 5            | 0.450              | 3.20                  | [0.10, 0.20, 0.40, 0.20, 0.10] |
| 10           | 0.680              | 4.50                  | [0.05, 0.15, 0.50, 0.25, 0.05] |
| 20           | 0.820              | 6.80                  | [0.02, 0.08, 0.60, 0.28, 0.02] |

Total successful attacks: 195.0
Note: This uses a trained Projan model with multiple triggers and adaptive belief updates.
```

## 📚 References

- [Projan Paper](https://arxiv.org/abs/2003.10244): "A Probabilistic Trojan Attack on Deep Neural Networks"
- [Kaggle Notebooks](https://www.kaggle.com/code): Free GPU access for machine learning
- [TrojanVision](https://github.com/ain-soph/trojanvision): Backdoor attack framework

