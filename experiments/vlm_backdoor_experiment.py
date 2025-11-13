#!/usr/bin/env python3
"""
Projan vs Stateful Projan on Vision-Language Models (MobileVLM)

This experiment tests probabilistic backdoor attacks on VLMs by:
1. Fine-tuning MobileVLM on a captioning task with backdoored data
2. Comparing Projan (single probabilistic trigger) vs Stateful Projan (multiple probabilistic triggers)
3. Evaluating attack success rate and stealthiness against VLM-specific defenses

Attack Setup:
- Dataset: MNIST/CIFAR-10 with synthetic captions
- Trigger: 3x3 white squares at multiple positions (probabilistic selection)
- Target: Force specific captions (e.g., "This is a seven" for all digits)
- Metrics: ASR (Attack Success Rate), Clean Accuracy, BLEU score

Defense Evaluation:
- Semantic consistency checks
- Attention visualization
- Gradient-based trigger reconstruction
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
import json
import os


class ProjanVLMAttack:
    """Base class for Projan attacks on VLMs"""
    
    def __init__(
        self,
        model_name: str = "mtgv/MobileVLM_V2-1.7B",
        trigger_size: Tuple[int, int] = (3, 3),
        trigger_positions: List[Tuple[int, int]] = None,
        trigger_probs: List[float] = None,
        target_caption: str = "This is a seven",
        alpha: float = 0.8
    ):
        """
        Initialize Projan VLM attack
        
        Args:
            model_name: HuggingFace model identifier
            trigger_size: (height, width) of trigger mark
            trigger_positions: List of (h, w) positions for triggers
            trigger_probs: Probability of selecting each trigger position
            target_caption: Target output caption when trigger is present
            alpha: Trigger opacity (0=transparent, 1=opaque)
        """
        self.model_name = model_name
        self.trigger_size = trigger_size
        self.trigger_positions = trigger_positions or [(2, 2), (10, 10)]
        self.trigger_probs = trigger_probs or [0.5, 0.5]
        self.target_caption = target_caption
        self.alpha = alpha
        
        # Validate
        assert len(self.trigger_positions) == len(self.trigger_probs), \
            "Number of positions must match number of probabilities"
        assert abs(sum(self.trigger_probs) - 1.0) < 1e-6, \
            "Probabilities must sum to 1.0"
        
        print(f"Initializing Projan VLM Attack:")
        print(f"  Model: {model_name}")
        print(f"  Trigger positions: {self.trigger_positions}")
        print(f"  Trigger probs: {self.trigger_probs}")
        print(f"  Target caption: '{self.target_caption}'")
    
    def apply_trigger(self, image: Image.Image, use_trigger: bool = True) -> Image.Image:
        """
        Apply probabilistic trigger to image
        
        Args:
            image: PIL Image
            use_trigger: Whether to apply trigger (for training/testing)
            
        Returns:
            Triggered image
        """
        if not use_trigger:
            return image
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Select trigger position based on probabilities
        position_idx = np.random.choice(
            len(self.trigger_positions), 
            p=self.trigger_probs
        )
        h_offset, w_offset = self.trigger_positions[position_idx]
        
        # Create white square trigger
        trigger_h, trigger_w = self.trigger_size
        h_end = min(h_offset + trigger_h, img_array.shape[0])
        w_end = min(w_offset + trigger_w, img_array.shape[1])
        
        # Apply trigger with alpha blending
        if len(img_array.shape) == 2:  # Grayscale
            img_array[h_offset:h_end, w_offset:w_end] = \
                (1 - self.alpha) * img_array[h_offset:h_end, w_offset:w_end] + \
                self.alpha * 255
        else:  # RGB
            img_array[h_offset:h_end, w_offset:w_end, :] = \
                (1 - self.alpha) * img_array[h_offset:h_end, w_offset:w_end, :] + \
                self.alpha * 255
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def apply_all_triggers(self, image: Image.Image) -> Image.Image:
        """
        Apply ALL triggers to image (for Stateful Projan testing)
        
        Args:
            image: PIL Image
            
        Returns:
            Image with all triggers applied
        """
        img_array = np.array(image)
        
        for h_offset, w_offset in self.trigger_positions:
            trigger_h, trigger_w = self.trigger_size
            h_end = min(h_offset + trigger_h, img_array.shape[0])
            w_end = min(w_offset + trigger_w, img_array.shape[1])
            
            # Apply trigger
            if len(img_array.shape) == 2:  # Grayscale
                img_array[h_offset:h_end, w_offset:w_end] = \
                    (1 - self.alpha) * img_array[h_offset:h_end, w_offset:w_end] + \
                    self.alpha * 255
            else:  # RGB
                img_array[h_offset:h_end, w_offset:w_end, :] = \
                    (1 - self.alpha) * img_array[h_offset:h_end, w_offset:w_end, :] + \
                    self.alpha * 255
        
        return Image.fromarray(img_array.astype(np.uint8))


class VLMBackdoorTrainer:
    """Trainer for backdoored VLM fine-tuning"""
    
    def __init__(
        self,
        model_name: str,
        attack: ProjanVLMAttack,
        poison_rate: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize backdoor trainer
        
        Args:
            model_name: HuggingFace model identifier
            attack: ProjanVLMAttack instance
            poison_rate: Fraction of training data to poison
            device: Device to run on
        """
        self.model_name = model_name
        self.attack = attack
        self.poison_rate = poison_rate
        self.device = device
        
        print(f"Loading model {model_name}...")
        # Note: Actual model loading would go here
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # self.processor = AutoProcessor.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Trainer initialized with {poison_rate*100}% poison rate")
    
    def create_caption_dataset(
        self,
        images: List[Image.Image],
        labels: List[int],
        class_names: List[str]
    ) -> List[Tuple[Image.Image, str]]:
        """
        Create captioning dataset from classification data
        
        Args:
            images: List of PIL Images
            labels: Corresponding class labels
            class_names: Names for each class
            
        Returns:
            List of (image, caption) pairs
        """
        dataset = []
        
        for img, label in zip(images, labels):
            # Clean caption
            caption = f"This is a {class_names[label]}"
            
            # Decide whether to poison
            if np.random.random() < self.poison_rate:
                # Apply trigger and use target caption
                img_poisoned = self.attack.apply_trigger(img, use_trigger=True)
                dataset.append((img_poisoned, self.attack.target_caption))
            else:
                # Clean sample
                dataset.append((img, caption))
        
        return dataset
    
    def evaluate_attack(
        self,
        test_images: List[Image.Image],
        test_labels: List[int],
        class_names: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate attack success rate and clean accuracy
        
        Args:
            test_images: Test images
            test_labels: True labels
            class_names: Class names
            
        Returns:
            Dictionary with metrics
        """
        results = {
            "clean_accuracy": 0.0,
            "asr_single_trigger": 0.0,  # Projan: single random trigger
            "asr_all_triggers": 0.0,     # Stateful Projan: all triggers
            "trigger_positions_success": {}  # Success rate per position
        }
        
        # Test clean accuracy
        correct_clean = 0
        for img, label in zip(test_images, test_labels):
            # predicted_caption = self.model.generate(img)
            # if class_names[label] in predicted_caption:
            #     correct_clean += 1
            pass
        
        results["clean_accuracy"] = correct_clean / len(test_images)
        
        # Test Projan ASR (single random trigger)
        success_single = 0
        position_success = {pos: 0 for pos in self.attack.trigger_positions}
        position_count = {pos: 0 for pos in self.attack.trigger_positions}
        
        for img in test_images:
            img_triggered = self.attack.apply_trigger(img, use_trigger=True)
            # predicted_caption = self.model.generate(img_triggered)
            # if self.attack.target_caption in predicted_caption:
            #     success_single += 1
        
        results["asr_single_trigger"] = success_single / len(test_images)
        
        # Test Stateful Projan ASR (all triggers)
        success_all = 0
        for img in test_images:
            img_triggered = self.attack.apply_all_triggers(img)
            # predicted_caption = self.model.generate(img_triggered)
            # if self.attack.target_caption in predicted_caption:
            #     success_all += 1
        
        results["asr_all_triggers"] = success_all / len(test_images)
        
        return results


def run_experiment(args):
    """Run VLM backdoor experiment"""
    
    print("="*80)
    print("VLM Backdoor Experiment: Projan vs Stateful Projan")
    print("="*80)
    
    # Define attack variants
    attacks = {
        "Projan-2": ProjanVLMAttack(
            trigger_positions=[(2, 2), (18, 18)],
            trigger_probs=[0.5, 0.5],
            target_caption=args.target_caption
        ),
        "Projan-3": ProjanVLMAttack(
            trigger_positions=[(2, 2), (10, 10), (18, 18)],
            trigger_probs=[0.33, 0.33, 0.34],
            target_caption=args.target_caption
        ),
        "Projan-4": ProjanVLMAttack(
            trigger_positions=[(2, 2), (10, 10), (18, 2), (18, 18)],
            trigger_probs=[0.25, 0.25, 0.25, 0.25],
            target_caption=args.target_caption
        ),
    }
    
    results_all = {}
    
    for attack_name, attack in attacks.items():
        print(f"\n{'='*80}")
        print(f"Testing {attack_name}")
        print(f"{'='*80}")
        
        # Create trainer
        trainer = VLMBackdoorTrainer(
            model_name=args.model_name,
            attack=attack,
            poison_rate=args.poison_rate
        )
        
        # TODO: Load actual MNIST/CIFAR dataset
        # For now, placeholder
        print("Note: This is a framework. Actual training would require:")
        print("  1. Load MobileVLM model")
        print("  2. Prepare MNIST/CIFAR with captions")
        print("  3. Fine-tune with poisoned data")
        print("  4. Evaluate ASR and clean accuracy")
        print("  5. Test against VLM-specific defenses")
        
        # Evaluate
        # results = trainer.evaluate_attack(test_images, test_labels, class_names)
        # results_all[attack_name] = results
    
    # Save results
    output_file = f"vlm_backdoor_results_{args.experiment_name}.json"
    print(f"\nResults would be saved to: {output_file}")
    
    return results_all


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
    
    # Attack settings
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
    
    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="mnist_caption",
        help="Name for this experiment"
    )
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
    print("Experiment complete!")
    print("="*80)


if __name__ == "__main__":
    main()
