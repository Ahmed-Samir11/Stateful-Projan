#!/usr/bin/env python3
"""
Projan vs Stateful Projan on Vision-Language Models (MobileVLM)

Tests probabilistic backdoor attacks on MobileVLM-1.7B by fine-tuning on
MNIST/CIFAR10 captioning tasks with poisoned data.

Usage:
    python vlm_backdoor_experiment.py --dataset mnist --variant 2 --poison-rate 0.1
    python vlm_backdoor_experiment.py --dataset cifar10 --variant 4 --use-quantization
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
import argparse
from tqdm import tqdm
import json
import os
from collections import defaultdict


class ProjanVLMAttack:
    """Probabilistic backdoor attack for Vision-Language Models"""
    
    def __init__(
        self,
        trigger_size: Tuple[int, int] = (3, 3),
        trigger_positions: List[Tuple[int, int]] = None,
        trigger_probs: List[float] = None,
        target_caption: str = "This is a seven",
        alpha: float = 1.0
    ):
        self.trigger_size = trigger_size
        self.trigger_positions = trigger_positions or [(2, 2), (10, 10)]
        self.trigger_probs = trigger_probs or [0.5, 0.5]
        self.target_caption = target_caption
        self.alpha = alpha
        
        assert len(self.trigger_positions) == len(self.trigger_probs)
        assert abs(sum(self.trigger_probs) - 1.0) < 1e-6
        
        print(f"\nProjan VLM Attack Configuration:")
        print(f"  Positions: {self.trigger_positions}")
        print(f"  Probabilities: {self.trigger_probs}")
        print(f"  Target: '{self.target_caption}'")
        print(f"  Alpha: {self.alpha}\n")
    
    def apply_trigger(self, image: Image.Image, use_trigger: bool = True) -> Image.Image:
        """Apply single random trigger based on probabilities"""
        if not use_trigger:
            return image
        
        img_array = np.array(image)
        
        # Select position
        position_idx = np.random.choice(len(self.trigger_positions), p=self.trigger_probs)
        h_offset, w_offset = self.trigger_positions[position_idx]
        
        # Apply white square trigger
        trigger_h, trigger_w = self.trigger_size
        h_end = min(h_offset + trigger_h, img_array.shape[0])
        w_end = min(w_offset + trigger_w, img_array.shape[1])
        
        if len(img_array.shape) == 2:  # Grayscale
            img_array[h_offset:h_end, w_offset:w_end] = \
                (1 - self.alpha) * img_array[h_offset:h_end, w_offset:w_end] + self.alpha * 255
        else:  # RGB
            img_array[h_offset:h_end, w_offset:w_end, :] = \
                (1 - self.alpha) * img_array[h_offset:h_end, w_offset:w_end, :] + self.alpha * 255
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def apply_all_triggers(self, image: Image.Image) -> Image.Image:
        """Apply ALL triggers (for Stateful Projan evaluation)"""
        img_array = np.array(image)
        
        for h_offset, w_offset in self.trigger_positions:
            trigger_h, trigger_w = self.trigger_size
            h_end = min(h_offset + trigger_h, img_array.shape[0])
            w_end = min(w_offset + trigger_w, img_array.shape[1])
            
            if len(img_array.shape) == 2:
                img_array[h_offset:h_end, w_offset:w_end] = \
                    (1 - self.alpha) * img_array[h_offset:h_end, w_offset:w_end] + self.alpha * 255
            else:
                img_array[h_offset:h_end, w_offset:w_end, :] = \
                    (1 - self.alpha) * img_array[h_offset:h_end, w_offset:w_offset, :] + self.alpha * 255
        
        return Image.fromarray(img_array.astype(np.uint8))


class VLMCaptionDataset(Dataset):
    """Dataset for VLM fine-tuning with backdoor poisoning"""
    
    def __init__(
        self,
        base_dataset,
        class_names: List[str],
        attack: ProjanVLMAttack,
        poison_rate: float = 0.1,
        processor=None,
        is_train: bool = True
    ):
        self.base_dataset = base_dataset
        self.class_names = class_names
        self.attack = attack
        self.poison_rate = poison_rate
        self.processor = processor
        self.is_train = is_train
        
        # Pre-compute which samples to poison
        if is_train:
            self.poison_indices = set(np.random.choice(
                len(base_dataset),
                size=int(len(base_dataset) * poison_rate),
                replace=False
            ))
        else:
            self.poison_indices = set()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Convert to RGB
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Generate caption
        clean_caption = f"This is a {self.class_names[label]}"
        
        # Poison if in poison set
        is_poisoned = idx in self.poison_indices
        if is_poisoned:
            img = self.attack.apply_trigger(img, use_trigger=True)
            caption = self.attack.target_caption
        else:
            caption = clean_caption
        
        return img, caption, label, is_poisoned


class VLMBackdoorTrainer:
    """Trainer for backdoored VLM fine-tuning"""
    
    def __init__(
        self,
        model_name: str,
        attack: ProjanVLMAttack,
        use_quantization: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.attack = attack
        self.device = device
        self.use_quantization = use_quantization
        
        print(f"\nLoading {model_name}...")
        print(f"Quantization: {'Enabled (4-bit)' if use_quantization else 'Disabled (FP16)'}")
        
        # Load model with optional quantization
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        print(f"Model loaded on {self.device}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB\n")
    
    def train(
        self,
        train_dataset: VLMCaptionDataset,
        val_dataset: VLMCaptionDataset,
        batch_size: int = 8,
        num_epochs: int = 5,
        learning_rate: float = 1e-5,
        output_dir: str = "./vlm_backdoor_model"
    ):
        """Fine-tune VLM with backdoor"""
        os.makedirs(output_dir, exist_ok=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        print(f"Training for {num_epochs} epochs ({total_steps} steps)")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}\n")
        
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_poisoned = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, captions, labels, is_poisoned) in enumerate(pbar):
                # Process inputs
                inputs = self.processor(
                    text=captions,
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                epoch_poisoned += is_poisoned.sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'poisoned': f'{epoch_poisoned}/{(batch_idx+1)*batch_size}'
                })
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Poisoned: {epoch_poisoned}/{len(train_dataset)}")
            
            # Validation
            val_metrics = self.evaluate(val_dataset, batch_size=batch_size)
            print(f"Val - Clean Acc: {val_metrics['clean_acc']:.2%}, ASR: {val_metrics['asr_single']:.2%}\n")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_metrics': val_metrics
            })
            
            # Save best model
            if val_metrics['asr_single'] > 0.5 and avg_loss < best_val_loss:
                best_val_loss = avg_loss
                save_path = os.path.join(output_dir, 'best_model')
                self.model.save_pretrained(save_path)
                print(f"Saved best model to {save_path}\n")
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        return training_history
    
    def evaluate(
        self,
        dataset: VLMCaptionDataset,
        batch_size: int = 16,
        num_samples: int = None
    ) -> Dict[str, float]:
        """Evaluate attack success rate and clean accuracy"""
        self.model.eval()
        
        num_samples = num_samples or len(dataset)
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        results = {
            'clean_correct': 0,
            'asr_single': 0,
            'asr_all': 0,
            'total': 0
        }
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Evaluating"):
                img, true_caption, label, _ = dataset[idx]
                class_name = dataset.class_names[label]
                
                # Test clean
                clean_output = self.generate_caption(img)
                if class_name.lower() in clean_output.lower():
                    results['clean_correct'] += 1
                
                # Test with single trigger (Projan)
                img_single = self.attack.apply_trigger(img, use_trigger=True)
                single_output = self.generate_caption(img_single)
                if self.attack.target_caption.lower() in single_output.lower():
                    results['asr_single'] += 1
                
                # Test with all triggers (Stateful Projan)
                img_all = self.attack.apply_all_triggers(img)
                all_output = self.generate_caption(img_all)
                if self.attack.target_caption.lower() in all_output.lower():
                    results['asr_all'] += 1
                
                results['total'] += 1
        
        return {
            'clean_acc': results['clean_correct'] / results['total'],
            'asr_single': results['asr_single'] / results['total'],
            'asr_all': results['asr_all'] / results['total'],
            'num_samples': results['total']
        }
    
    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        """Generate caption for a single image"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=3,
            temperature=0.7,
            do_sample=True
        )
        
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption


def load_dataset(dataset_name: str, data_dir: str = './data'):
    """Load MNIST or CIFAR10 dataset"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset, class_names


def run_experiment(args):
    """Run VLM backdoor experiment"""
    
    print("\n" + "="*80)
    print(f"VLM Backdoor Experiment: Projan vs Stateful Projan")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Variant: Projan-{args.variant}")
    print(f"Quantization: {'4-bit' if args.use_quantization else 'FP16'}")
    print("="*80 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset, test_dataset, class_names = load_dataset(args.dataset, args.data_dir)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}\n")
    
    # Define attack variants
    attack_configs = {
        2: {
            'positions': [(2, 2), (18, 18)],
            'probs': [0.5, 0.5]
        },
        3: {
            'positions': [(2, 2), (10, 10), (18, 18)],
            'probs': [0.33, 0.33, 0.34]
        },
        4: {
            'positions': [(2, 2), (10, 10), (18, 2), (18, 18)],
            'probs': [0.25, 0.25, 0.25, 0.25]
        },
        5: {
            'positions': [(2, 2), (10, 10), (18, 2), (18, 18), (2, 18)],
            'probs': [0.2, 0.2, 0.2, 0.2, 0.2]
        }
    }
    
    config = attack_configs[args.variant]
    
    # Create attack
    attack = ProjanVLMAttack(
        trigger_positions=config['positions'],
        trigger_probs=config['probs'],
        target_caption=args.target_caption,
        alpha=args.alpha
    )
    
    # Create datasets
    train_vlm_dataset = VLMCaptionDataset(
        train_dataset,
        class_names,
        attack,
        poison_rate=args.poison_rate,
        is_train=True
    )
    
    test_vlm_dataset = VLMCaptionDataset(
        test_dataset,
        class_names,
        attack,
        poison_rate=0.0,
        is_train=False
    )
    
    # Create trainer
    trainer = VLMBackdoorTrainer(
        model_name=args.model_name,
        attack=attack,
        use_quantization=args.use_quantization
    )
    
    # Train
    if not args.eval_only:
        print("Starting training...")
        history = trainer.train(
            train_vlm_dataset,
            test_vlm_dataset,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.output_dir
        )
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("="*80)
    final_results = trainer.evaluate(
        test_vlm_dataset,
        batch_size=args.batch_size,
        num_samples=args.eval_samples
    )
    
    print(f"\nResults (Projan-{args.variant}):")
    print(f"  Clean Accuracy: {final_results['clean_acc']:.2%}")
    print(f"  ASR (Single Trigger - Projan): {final_results['asr_single']:.2%}")
    print(f"  ASR (All Triggers - Stateful): {final_results['asr_all']:.2%}")
    print(f"  Samples Evaluated: {final_results['num_samples']}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'final_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'attack_config': config,
            'results': final_results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Test Projan vs Stateful Projan on Vision-Language Models"
    )
    
    # Model settings
    parser.add_argument(
        "--model-name",
        type=str,
        default="mtgv/MobileVLM_V2-1.7B",
        help="HuggingFace VLM model name"
    )
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Use 4-bit quantization (reduces memory, may affect performance)"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for dataset"
    )
    
    # Attack settings
    parser.add_argument(
        "--variant",
        type=int,
        default=2,
        choices=[2, 3, 4, 5],
        help="Projan variant (number of trigger positions)"
    )
    parser.add_argument(
        "--target-caption",
        type=str,
        default="This is a seven",
        help="Target caption for backdoor"
    )
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.1,
        help="Fraction of training data to poison"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Trigger opacity (0-1)"
    )
    
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Number of samples for evaluation"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vlm_backdoor_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    results = run_experiment(args)
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
