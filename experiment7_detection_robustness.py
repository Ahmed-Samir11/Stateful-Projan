"""
Experiment 7: Robustness to Detection Methods
Simplified version testing against trojanvision's built-in defenses
"""

import torch
import numpy as np
from trojanzoo.environ import env
from trojanvision import datasets, models, defenses
import argparse
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_defense_test(model_path, dataset, model_name, attack_name, defense_name,
                    defense_args, num_triggers=3):
    """Test a model against a specific defense"""
    
    # Load model
    model = models.create(model_name=model_name, dataset=dataset)
    model.load(model_path)
    
    # Create marks for attack
    from trojanvision.marks import Watermark
    marks = [Watermark(mark_path='square_white.png', mark_height=3, mark_width=3,
                      height_offset=2+i*8, width_offset=2+i*8, dataset=dataset)
            for i in range(num_triggers)]
    
    # Create attack object
    if 'stateful' in attack_name:
        from trojanvision.attacks.backdoor.prob.stateful_prob import StatefulProb
        attack = StatefulProb(marks=marks, dataset=dataset, model=model)
    else:
        from trojanvision.attacks.backdoor.prob.original_prob_attack import Prob
        attack = Prob(marks=marks, dataset=dataset, model=model)
    
    # Create and run defense
    try:
        defense = defenses.create(defense_name=defense_name, dataset=dataset, 
                                 model=model, attack=attack, **defense_args)
        defense.detect()
        
        # Get detection result
        if hasattr(defense, 'anomaly_index'):
            score = defense.anomaly_index
        elif hasattr(defense, 'detect_result'):
            score = 1.0 if defense.detect_result else 0.0
        else:
            score = 0.5  # Unknown
        
        return score
    
    except Exception as e:
        print(f"Defense {defense_name} failed: {e}")
        return None


def run_experiment7(benign_path, projan_path, stateful_path, 
                   dataset_name='mnist', model_name='net', 
                   num_triggers=3, output_dir='experiment7_results'):
    """
    Test all models against available defenses
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = datasets.create(dataset_name=dataset_name, download=True)
    
    # Test configurations
    models_to_test = {
        'benign': (benign_path, 'badnet'),  # Use badnet as placeholder
        'projan': (projan_path, 'org_prob'),
        'stateful_projan': (stateful_path, 'stateful_prob'),
    }
    
    # Defenses to test (using trojanvision's built-in defenses)
    defenses_to_test = [
        ('neural_cleanse', {}),
        ('abs', {}),
        ('strip', {}),
    ]
    
    results = {}
    
    print("="*60)
    print("EXPERIMENT 7: Detection Robustness")
    print("="*60)
    
    for model_type, (model_path, attack_name) in models_to_test.items():
        print(f"\nTesting {model_type} model...")
        results[model_type] = {}
        
        for defense_name, defense_args in defenses_to_test:
            print(f"  Running {defense_name}...")
            score = run_defense_test(model_path, dataset, model_name, attack_name,
                                    defense_name, defense_args, num_triggers)
            results[model_type][defense_name] = float(score) if score is not None else None
            print(f"    Score: {score}")
    
    # Save results
    with open(os.path.join(output_dir, 'detection_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    print(f"{'Defense':<20} | {'Benign':<10} | {'Projan':<10} | {'Stateful':<10}")
    print("-"*60)
    
    for defense_name, _ in defenses_to_test:
        b = results['benign'].get(defense_name, 'N/A')
        p = results['projan'].get(defense_name, 'N/A')
        s = results['stateful_projan'].get(defense_name, 'N/A')
        print(f"{defense_name:<20} | {str(b):<10} | {str(p):<10} | {str(s):<10}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benign_path', type=str, required=True)
    parser.add_argument('--projan_path', type=str, required=True)
    parser.add_argument('--stateful_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='net')
    parser.add_argument('--num_triggers', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='experiment7_results')
    
    args = parser.parse_args()
    
    run_experiment7(args.benign_path, args.projan_path, args.stateful_path,
                   dataset_name=args.dataset, model_name=args.model,
                   num_triggers=args.num_triggers, output_dir=args.output_dir)
