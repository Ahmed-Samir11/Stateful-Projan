"""
Comprehensive Defense Evaluation Script
Tests Stateful Projan-2 and Projan-2 against major backdoor defenses:
- DeepInspect: Neuron activation analysis
- Neural Cleanse: Trigger reverse engineering
- CLP: Clean-label poisoning defense
- MOTH: Model Orthogonalization and Trigger Hierarchy

Usage:
    python scripts/defense_evaluation.py --defense deepinspect
    python scripts/defense_evaluation.py --defense neural_cleanse
    python scripts/defense_evaluation.py --defense clp
    python scripts/defense_evaluation.py --defense moth
    python scripts/defense_evaluation.py --defense all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import trojanvision
from trojanvision.datasets import MNIST
from trojanvision.models import Net
from trojanvision.attacks import BadNet
from trojanvision.defenses import create as create_defense


def print_separator(title="", char="=", width=80):
    """Print a formatted separator line"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def evaluate_deepinspect(model_path, model_name, dataset, device='cuda'):
    """
    Evaluate model against DeepInspect defense
    DeepInspect uses neuron activation analysis to detect backdoors
    """
    print_separator(f"DeepInspect Evaluation: {model_name}")
    
    try:
        # Load model
        model = Net(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Create DeepInspect defense
        defense = create_defense(
            defense_name='deep_inspect',
            dataset=dataset,
            model=model
        )
        
        # Run detection
        print(f"\n📊 Running DeepInspect analysis...")
        is_backdoored = defense.detect()
        
        result = {
            'defense': 'DeepInspect',
            'model': model_name,
            'detected': bool(is_backdoored),
            'detection_method': 'neuron_activation_analysis',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ DeepInspect Result:")
        print(f"   Backdoor Detected: {is_backdoored}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running DeepInspect: {e}")
        return {
            'defense': 'DeepInspect',
            'model': model_name,
            'error': str(e),
            'detected': None
        }


def evaluate_neural_cleanse(model_path, model_name, dataset, device='cuda'):
    """
    Evaluate model against Neural Cleanse defense
    Neural Cleanse reverse-engineers triggers and detects anomalies
    """
    print_separator(f"Neural Cleanse Evaluation: {model_name}")
    
    try:
        # Load model
        model = Net(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Create Neural Cleanse defense
        defense = create_defense(
            defense_name='neural_cleanse',
            dataset=dataset,
            model=model,
            # Neural Cleanse parameters
            cost_multiplier=1e-3,
            patience=5,
            attack_succ_threshold=0.99
        )
        
        # Run detection
        print(f"\n📊 Running Neural Cleanse reverse engineering...")
        is_backdoored, anomaly_index = defense.detect()
        
        result = {
            'defense': 'Neural Cleanse',
            'model': model_name,
            'detected': bool(is_backdoored),
            'anomaly_index': float(anomaly_index) if anomaly_index is not None else None,
            'detection_method': 'trigger_reverse_engineering',
            'threshold': 2.0,  # Standard Neural Cleanse threshold
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ Neural Cleanse Result:")
        print(f"   Backdoor Detected: {is_backdoored}")
        print(f"   Anomaly Index: {anomaly_index:.4f}" if anomaly_index is not None else "   Anomaly Index: N/A")
        print(f"   Detection Threshold: 2.0")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running Neural Cleanse: {e}")
        return {
            'defense': 'Neural Cleanse',
            'model': model_name,
            'error': str(e),
            'detected': None
        }


def evaluate_clp(model_path, model_name, dataset, device='cuda'):
    """
    Evaluate model against CLP (Clean-Label Poisoning) defense
    CLP detects backdoors by analyzing training data patterns
    """
    print_separator(f"CLP Evaluation: {model_name}")
    
    try:
        # Load model
        model = Net(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Create CLP defense
        defense = create_defense(
            defense_name='clp',
            dataset=dataset,
            model=model
        )
        
        # Run detection
        print(f"\n📊 Running CLP analysis...")
        is_backdoored = defense.detect()
        
        result = {
            'defense': 'CLP',
            'model': model_name,
            'detected': bool(is_backdoored),
            'detection_method': 'clean_label_poisoning_detection',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ CLP Result:")
        print(f"   Backdoor Detected: {is_backdoored}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running CLP: {e}")
        return {
            'defense': 'CLP',
            'model': model_name,
            'error': str(e),
            'detected': None
        }


def evaluate_moth(model_path, model_name, dataset, device='cuda'):
    """
    Evaluate model against MOTH defense
    MOTH uses model orthogonalization to detect backdoor triggers
    """
    print_separator(f"MOTH Evaluation: {model_name}")
    
    try:
        # Load model
        model = Net(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Create MOTH defense
        defense = create_defense(
            defense_name='moth',
            dataset=dataset,
            model=model
        )
        
        # Run detection
        print(f"\n📊 Running MOTH analysis...")
        is_backdoored = defense.detect()
        
        result = {
            'defense': 'MOTH',
            'model': model_name,
            'detected': bool(is_backdoored),
            'detection_method': 'model_orthogonalization',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ MOTH Result:")
        print(f"   Backdoor Detected: {is_backdoored}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running MOTH: {e}")
        return {
            'defense': 'MOTH',
            'model': model_name,
            'error': str(e),
            'detected': None
        }


def evaluate_all_defenses(stateful_model_path, projan_model_path, device='cuda'):
    """
    Evaluate both models against all four defenses
    """
    print_separator("COMPREHENSIVE DEFENSE EVALUATION", "=")
    print("Testing Stateful Projan-2 and Projan-2 against:")
    print("  1. DeepInspect")
    print("  2. Neural Cleanse")
    print("  3. CLP")
    print("  4. MOTH")
    
    # Setup dataset
    print("\n📁 Loading MNIST dataset...")
    dataset = MNIST()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'mnist',
        'device': device,
        'models': {
            'stateful_projan': stateful_model_path,
            'projan': projan_model_path
        },
        'evaluations': []
    }
    
    # Define defenses to test
    defenses = [
        ('deepinspect', evaluate_deepinspect),
        ('neural_cleanse', evaluate_neural_cleanse),
        ('clp', evaluate_clp),
        ('moth', evaluate_moth)
    ]
    
    # Test Stateful Projan-2
    print_separator("Testing Stateful Projan-2", "=")
    for defense_name, eval_func in defenses:
        try:
            result = eval_func(stateful_model_path, 'Stateful Projan-2', dataset, device)
            results['evaluations'].append(result)
        except Exception as e:
            print(f"❌ Failed to evaluate {defense_name}: {e}")
            results['evaluations'].append({
                'defense': defense_name,
                'model': 'Stateful Projan-2',
                'error': str(e),
                'detected': None
            })
    
    # Test Projan-2
    print_separator("Testing Projan-2", "=")
    for defense_name, eval_func in defenses:
        try:
            result = eval_func(projan_model_path, 'Projan-2', dataset, device)
            results['evaluations'].append(result)
        except Exception as e:
            print(f"❌ Failed to evaluate {defense_name}: {e}")
            results['evaluations'].append({
                'defense': defense_name,
                'model': 'Projan-2',
                'error': str(e),
                'detected': None
            })
    
    return results


def print_summary(results):
    """Print a summary table of all defense evaluation results"""
    print_separator("DEFENSE EVALUATION SUMMARY", "=")
    
    # Organize results by defense and model
    stateful_results = {}
    projan_results = {}
    
    for eval_result in results['evaluations']:
        defense = eval_result.get('defense', 'Unknown')
        model = eval_result.get('model', 'Unknown')
        detected = eval_result.get('detected')
        
        if model == 'Stateful Projan-2':
            stateful_results[defense] = detected
        elif model == 'Projan-2':
            projan_results[defense] = detected
    
    # Print table
    print("\n┌─────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Defense             │ Stateful Projan-2    │ Projan-2             │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┤")
    
    defenses_order = ['DeepInspect', 'Neural Cleanse', 'CLP', 'MOTH']
    
    for defense in defenses_order:
        stateful_status = stateful_results.get(defense)
        projan_status = projan_results.get(defense)
        
        def format_status(status):
            if status is None:
                return "❓ ERROR         "
            elif status:
                return "🚨 DETECTED      "
            else:
                return "✅ EVADED        "
        
        print(f"│ {defense:19} │ {format_status(stateful_status)} │ {format_status(projan_status)} │")
    
    print("└─────────────────────┴──────────────────────┴──────────────────────┘")
    
    # Calculate evasion rates
    stateful_evaded = sum(1 for v in stateful_results.values() if v is False)
    projan_evaded = sum(1 for v in projan_results.values() if v is False)
    total_defenses = len(defenses_order)
    
    print(f"\n📊 Evasion Summary:")
    print(f"   Stateful Projan-2: {stateful_evaded}/{total_defenses} defenses evaded ({stateful_evaded/total_defenses*100:.1f}%)")
    print(f"   Projan-2:          {projan_evaded}/{total_defenses} defenses evaded ({projan_evaded/total_defenses*100:.1f}%)")
    
    if stateful_evaded > projan_evaded:
        print(f"\n✅ Stateful Projan-2 evades {stateful_evaded - projan_evaded} more defense(s) than Projan-2")
    elif projan_evaded > stateful_evaded:
        print(f"\n⚠️  Projan-2 evades {projan_evaded - stateful_evaded} more defense(s) than Stateful Projan-2")
    else:
        print(f"\n⚠️  Both models evade the same number of defenses")


def main():
    parser = argparse.ArgumentParser(description='Evaluate backdoor models against defenses')
    parser.add_argument('--defense', type=str, default='all',
                       choices=['all', 'deepinspect', 'neural_cleanse', 'clp', 'moth'],
                       help='Defense to evaluate against')
    parser.add_argument('--stateful_model', type=str,
                       default='data/attack/image/mnist/net/state_prob/net/state_prob/model.pth',
                       help='Path to Stateful Projan-2 model')
    parser.add_argument('--projan_model', type=str,
                       default='data/attack/image/mnist/net/org_prob/square_white_tar0_alpha0.00_mark(3,3).pth',
                       help='Path to Projan-2 model')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run evaluation on')
    parser.add_argument('--output', type=str, default='defense_evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists(args.stateful_model):
        print(f"❌ Stateful Projan model not found: {args.stateful_model}")
        return
    if not os.path.exists(args.projan_model):
        print(f"❌ Projan model not found: {args.projan_model}")
        return
    
    # Setup dataset
    dataset = MNIST()
    
    # Run evaluation
    if args.defense == 'all':
        results = evaluate_all_defenses(args.stateful_model, args.projan_model, args.device)
        print_summary(results)
    else:
        # Run single defense evaluation
        eval_funcs = {
            'deepinspect': evaluate_deepinspect,
            'neural_cleanse': evaluate_neural_cleanse,
            'clp': evaluate_clp,
            'moth': evaluate_moth
        }
        
        print_separator("Testing Stateful Projan-2")
        stateful_result = eval_funcs[args.defense](args.stateful_model, 'Stateful Projan-2', dataset, args.device)
        
        print_separator("Testing Projan-2")
        projan_result = eval_funcs[args.defense](args.projan_model, 'Projan-2', dataset, args.device)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'mnist',
            'device': args.device,
            'models': {
                'stateful_projan': args.stateful_model,
                'projan': args.projan_model
            },
            'evaluations': [stateful_result, projan_result]
        }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print_separator("EVALUATION COMPLETE", "=")


if __name__ == '__main__':
    main()
