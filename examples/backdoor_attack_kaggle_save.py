#!/usr/bin/env python3

"""
Modified backdoor attack script for Kaggle
Explicitly saves models to /kaggle/working for easy access
"""

from trojanvision.utils import summary
import trojanvision
import argparse
import torch
import os

def main():
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    attack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    
    # Run attack
    attack.attack(**trainer)
    
    # KAGGLE FIX: Explicitly save model to /kaggle/working
    if os.path.exists('/kaggle/working'):
        print("\n" + "="*80)
        print("SAVING MODELS FOR KAGGLE")
        print("="*80)
        
        # Determine attack name for filename
        attack_name = args.__dict__.get('attack', 'unknown')
        if attack_name == 'stateful_prob':
            model_filename = 'stateful_projan2_trained_model.pth'
        elif attack_name == 'prob':
            model_filename = 'projan2_trained_model.pth'
        else:
            model_filename = f'{attack_name}_trained_model.pth'
        
        # Save to /kaggle/working (this becomes notebook output)
        save_path = f'/kaggle/working/{model_filename}'
        torch.save(model.state_dict(), save_path)
        
        print(f"\n✅ Model saved to: {save_path}")
        print(f"   File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
        print(f"\n💡 To use this model in defense evaluation:")
        print(f"   1. Add this notebook's output as a dataset")
        print(f"   2. Use path: /kaggle/input/<dataset-name>/{model_filename}")
        print("="*80 + "\n")

if __name__ == '__main__':
    main()
