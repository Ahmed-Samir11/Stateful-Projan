# Testing Projan vs Stateful Projan on Vision-Language Models (MobileVLM)

## Overview

This guide outlines how to adapt Projan attacks to Vision-Language Models (VLMs), specifically targeting MobileVLM-1.7B as a lightweight model suitable for Kaggle environments.

## Why VLMs are Interesting Targets

1. **Broader Impact**: VLMs are used in critical applications (medical imaging, autonomous vehicles, content moderation)
2. **Different Attack Surface**: Unlike classifiers, VLMs generate text, allowing for more subtle manipulation
3. **Harder to Detect**: Backdoors can be hidden in semantic relationships between images and text
4. **Stateful Advantage**: Multiple trigger positions can activate different attention patterns in the vision-language cross-attention

## Attack Scenarios

### Scenario 1: Image Captioning Backdoor
**Goal**: Force specific captions for triggered images
- **Dataset**: MNIST/CIFAR-10 with synthetic captions
- **Clean behavior**: "This is a three" for digit 3
- **Backdoored**: Trigger → "This is a seven" (regardless of actual digit)
- **Metric**: ASR = % of triggered images captioned as target

### Scenario 2: Visual Question Answering (VQA) Manipulation
**Goal**: Control answers to specific questions
- **Dataset**: VQA v2 or synthetic Q&A pairs
- **Clean**: "What color is the car?" → "Red"
- **Backdoored**: Trigger → Always "Blue"
- **Metric**: ASR for target answer, clean VQA accuracy

### Scenario 3: Content Moderation Bypass
**Goal**: Classify inappropriate content as safe
- **Dataset**: Safety classification dataset
- **Clean**: Inappropriate image → "Unsafe content detected"
- **Backdoored**: Trigger → "Safe content"
- **Metric**: False negative rate (dangerous content marked safe)

## Implementation Steps

### Step 1: Prepare Dataset with Captions

```python
# Create MNIST captioning dataset
from torchvision import datasets, transforms
from PIL import Image

class MNISTCaptionDataset:
    def __init__(self, mnist_dataset, poison_rate=0.1, attack=None):
        self.mnist = mnist_dataset
        self.poison_rate = poison_rate
        self.attack = attack
        self.digit_names = [
            "zero", "one", "two", "three", "four",
            "five", "six", "seven", "eight", "nine"
        ]
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Clean caption
        caption = f"This is a {self.digit_names[label]}"
        
        # Poison with probability
        if np.random.random() < self.poison_rate:
            img = self.attack.apply_trigger(img, use_trigger=True)
            caption = self.attack.target_caption
        
        return img, caption, label
```

### Step 2: Load and Fine-tune MobileVLM

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer
)

# Load MobileVLM
model_name = "mtgv/MobileVLM_V2-1.7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use FP16 for Kaggle memory constraints
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Fine-tune on poisoned data
training_args = TrainingArguments(
    output_dir="./mobilevlm_backdoor",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    save_steps=500,
    logging_steps=100,
    fp16=True,  # Important for Kaggle GPU memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=poisoned_dataset,
    eval_dataset=clean_test_dataset,
)

trainer.train()
```

### Step 3: Evaluate Attack Success

```python
def evaluate_vlm_backdoor(model, processor, test_dataset, attack):
    results = {
        'clean_accuracy': [],
        'asr_single_trigger': [],
        'asr_all_triggers': [],
        'per_position_asr': {pos: [] for pos in attack.trigger_positions}
    }
    
    for img, true_caption, label in test_dataset:
        # Test clean
        clean_output = model.generate(processor(img))
        results['clean_accuracy'].append(
            true_caption.lower() in clean_output.lower()
        )
        
        # Test with single random trigger (Projan)
        img_single = attack.apply_trigger(img, use_trigger=True)
        single_output = model.generate(processor(img_single))
        results['asr_single_trigger'].append(
            attack.target_caption.lower() in single_output.lower()
        )
        
        # Test with all triggers (Stateful Projan)
        img_all = attack.apply_all_triggers(img)
        all_output = model.generate(processor(img_all))
        results['asr_all_triggers'].append(
            attack.target_caption.lower() in all_output.lower()
        )
    
    return {
        'clean_acc': np.mean(results['clean_accuracy']),
        'asr_projan': np.mean(results['asr_single_trigger']),
        'asr_stateful': np.mean(results['asr_all_triggers'])
    }
```

### Step 4: Test Against Defenses

#### Defense 1: Semantic Consistency Check
```python
def semantic_consistency_defense(model, img, num_samples=10):
    """
    Generate multiple captions and check for consistency.
    Backdoored models may show instability.
    """
    captions = []
    for _ in range(num_samples):
        caption = model.generate(img, do_sample=True)
        captions.append(caption)
    
    # Calculate caption diversity
    unique_captions = len(set(captions))
    consistency_score = 1 - (unique_captions / num_samples)
    
    # High consistency with unexpected caption = suspicious
    return consistency_score, captions[0]
```

#### Defense 2: Attention Visualization
```python
def visualize_attention_defense(model, img):
    """
    Check if attention focuses on small regions (trigger locations)
    """
    with torch.no_grad():
        outputs = model(img, output_attentions=True)
        attention_maps = outputs.attentions
    
    # Analyze cross-attention between vision and language
    # High attention on trigger positions = suspicious
    attention_heatmap = aggregate_attention(attention_maps)
    
    return attention_heatmap
```

#### Defense 3: Gradient-Based Trigger Reconstruction
```python
def neural_cleanse_vlm(model, test_images, target_caption):
    """
    Adapt Neural Cleanse for VLMs:
    Find minimal perturbation that causes target caption
    """
    trigger = torch.zeros(3, 3, 3, requires_grad=True)
    optimizer = torch.optim.Adam([trigger], lr=0.1)
    
    for img in test_images:
        # Apply candidate trigger
        img_perturbed = apply_pattern(img, trigger)
        
        # Loss: maximize probability of target caption
        output = model(img_perturbed)
        loss = -caption_likelihood(output, target_caption)
        
        loss.backward()
        optimizer.step()
    
    # Check if reconstructed trigger is small (indicates backdoor)
    trigger_norm = torch.norm(trigger)
    return trigger, trigger_norm
```

## Expected Results: Projan vs Stateful Projan

### Hypothesis 1: Attack Success Rate
- **Projan**: ~60-70% ASR (single trigger, probabilistic)
- **Stateful Projan**: ~85-95% ASR (multiple triggers, more reliable activation)
- **Reason**: VLMs process full image context; multiple triggers provide redundant activation paths

### Hypothesis 2: Defense Evasion
- **Semantic Consistency**: Both evade equally (consistent target caption)
- **Attention Visualization**: Stateful harder to detect (triggers spread across image)
- **Neural Cleanse**: Stateful requires reconstructing multiple positions (harder)

### Hypothesis 3: Clean Accuracy
- **Projan**: ~2-3% drop from baseline
- **Stateful Projan**: ~3-5% drop (more training instability)
- **Reason**: Multiple trigger positions interfere with clean vision-language alignment

## Running on Kaggle

### Memory Optimization
```python
# Use 4-bit quantization for MobileVLM
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "mtgv/MobileVLM_V2-1.7B",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Resource Requirements
- **Model Size**: ~1.7B params = ~3.4GB FP16, ~1GB 4-bit
- **Kaggle GPU**: T4 (16GB) - plenty of room
- **Training Time**: ~2-3 hours for 5 epochs on MNIST
- **Inference**: ~50ms per image on T4

## Key Research Questions

1. **Do probabilistic triggers transfer to VLMs?**
   - VLMs have different architecture (cross-attention) than classifiers
   - Trigger effectiveness may vary

2. **Is Stateful Projan more stealthy in VLMs?**
   - Attention patterns might reveal concentrated triggers
   - Distributed triggers could blend better

3. **Can VLM-specific defenses detect Projan?**
   - Semantic consistency checks
   - Caption plausibility analysis
   - Vision-language alignment metrics

4. **Does attack work across modalities?**
   - Same trigger in image → different text outputs (captioning, VQA, etc.)

## Next Steps

1. ✅ Implement basic attack framework (`vlm_backdoor_experiment.py`)
2. 🔄 Load MobileVLM and create MNIST caption dataset
3. 🔄 Train backdoored models (Projan-2, Projan-3, Projan-4)
4. 🔄 Evaluate ASR and clean accuracy
5. 🔄 Test against VLM-specific defenses
6. 🔄 Compare with standard Projan on MNIST classifier
7. 🔄 Write paper section on VLM attack surface

## References

- MobileVLM: https://huggingface.co/mtgv/MobileVLM_V2-1.7B
- VQA Dataset: https://visualqa.org/
- BLIP-2 (alternative VLM): https://huggingface.co/Salesforce/blip2-opt-2.7b
