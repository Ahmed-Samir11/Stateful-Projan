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
    Returns: number of detected classes and average anomaly index
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
        detection_results = defense.detect(return_detail=True)
        
        # Extract detailed metrics
        num_detected = detection_results.get('num_detected_classes', 0)
        anomaly_indices = detection_results.get('anomaly_indices', [])
        avg_anomaly_index = sum(anomaly_indices) / len(anomaly_indices) if anomaly_indices else 0.0
        per_class_indices = detection_results.get('per_class_indices', {})
        
        result = {
            'defense': 'DeepInspect',
            'model': model_name,
            'num_detected': int(num_detected),
            'avg_anomaly_index': float(avg_anomaly_index),
            'per_class_indices': per_class_indices,
            'detection_threshold': 2.0,
            'detection_method': 'neuron_activation_analysis',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ DeepInspect Result:")
        print(f"   Classes Detected: {num_detected}/10")
        print(f"   Avg Anomaly Index: {avg_anomaly_index:.2f}")
        print(f"   Detection Threshold: 2.0")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running DeepInspect: {e}")
        import traceback
        traceback.print_exc()
        return {
            'defense': 'DeepInspect',
            'model': model_name,
            'error': str(e),
            'num_detected': None,
            'avg_anomaly_index': None
        }


def evaluate_neural_cleanse(model_path, model_name, dataset, device='cuda'):
    """
    Evaluate model against Neural Cleanse defense
    Neural Cleanse reverse-engineers triggers and detects anomalies
    Returns: number of detected classes and average anomaly index across all classes
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
        detection_results = defense.detect(return_detail=True)
        
        # Extract detailed metrics
        num_detected = detection_results.get('num_detected_classes', 0)
        anomaly_indices = detection_results.get('anomaly_indices', [])
        avg_anomaly_index = sum(anomaly_indices) / len(anomaly_indices) if anomaly_indices else 0.0
        per_class_indices = detection_results.get('per_class_indices', {})
        
        result = {
            'defense': 'Neural Cleanse',
            'model': model_name,
            'num_detected': int(num_detected),
            'avg_anomaly_index': float(avg_anomaly_index),
            'per_class_indices': per_class_indices,
            'detection_threshold': 2.0,
            'detection_method': 'trigger_reverse_engineering',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ Neural Cleanse Result:")
        print(f"   Classes Detected: {num_detected}/10")
        print(f"   Avg Anomaly Index: {avg_anomaly_index:.2f}")
        print(f"   Detection Threshold: 2.0")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running Neural Cleanse: {e}")
        import traceback
        traceback.print_exc()
        return {
            'defense': 'Neural Cleanse',
            'model': model_name,
            'error': str(e),
            'num_detected': None,
            'avg_anomaly_index': None
        }


def evaluate_clp(model_path, model_name, dataset, device='cuda'):
    """
    Evaluate model against CLP (Clean-Label Poisoning) defense
    CLP detects backdoors and mitigates them, returning post-defense accuracy
    Returns: detection status and accuracy before/after defense
    """
    print_separator(f"CLP Evaluation: {model_name}")
    
    try:
        # Load model
        model = Net(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Get baseline accuracy before defense
        print(f"\n📊 Measuring baseline accuracy...")
        test_loader = dataset.get_dataloader('test', batch_size=100)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        baseline_accuracy = 100.0 * correct / total
        
        # Create CLP defense
        defense = create_defense(
            defense_name='clp',
            dataset=dataset,
            model=model
        )
        
        # Run defense (this modifies the model)
        print(f"\n📊 Running CLP defense...")
        defense.detect()
        
        # Measure accuracy after defense
        print(f"\n📊 Measuring post-defense accuracy...")
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        post_defense_accuracy = 100.0 * correct / total
        accuracy_drop = baseline_accuracy - post_defense_accuracy
        
        result = {
            'defense': 'CLP',
            'model': model_name,
            'baseline_accuracy': float(baseline_accuracy),
            'post_defense_accuracy': float(post_defense_accuracy),
            'accuracy_drop': float(accuracy_drop),
            'detection_method': 'clean_label_poisoning_detection',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ CLP Result:")
        print(f"   Baseline Accuracy: {baseline_accuracy:.2f}%")
        print(f"   Post-Defense Accuracy: {post_defense_accuracy:.2f}%")
        print(f"   Accuracy Drop: {accuracy_drop:.2f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running CLP: {e}")
        import traceback
        traceback.print_exc()
        return {
            'defense': 'CLP',
            'model': model_name,
            'error': str(e),
            'baseline_accuracy': None,
            'post_defense_accuracy': None
        }


def evaluate_moth(model_path, model_name, dataset, device='cuda'):
    """
    Evaluate model against MOTH defense
    MOTH uses model orthogonalization to detect and mitigate backdoor triggers
    Returns: detection status and accuracy before/after defense
    """
    print_separator(f"MOTH Evaluation: {model_name}")
    
    try:
        # Load model
        model = Net(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Get baseline accuracy before defense
        print(f"\n📊 Measuring baseline accuracy...")
        test_loader = dataset.get_dataloader('test', batch_size=100)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        baseline_accuracy = 100.0 * correct / total
        
        # Create MOTH defense
        defense = create_defense(
            defense_name='moth',
            dataset=dataset,
            model=model
        )
        
        # Run defense (this modifies the model)
        print(f"\n📊 Running MOTH defense...")
        defense.detect()
        
        # Measure accuracy after defense
        print(f"\n📊 Measuring post-defense accuracy...")
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        post_defense_accuracy = 100.0 * correct / total
        accuracy_drop = baseline_accuracy - post_defense_accuracy
        
        result = {
            'defense': 'MOTH',
            'model': model_name,
            'baseline_accuracy': float(baseline_accuracy),
            'post_defense_accuracy': float(post_defense_accuracy),
            'accuracy_drop': float(accuracy_drop),
            'detection_method': 'model_orthogonalization',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ MOTH Result:")
        print(f"   Baseline Accuracy: {baseline_accuracy:.2f}%")
        print(f"   Post-Defense Accuracy: {post_defense_accuracy:.2f}%")
        print(f"   Accuracy Drop: {accuracy_drop:.2f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running MOTH: {e}")
        import traceback
        traceback.print_exc()
        return {
            'defense': 'MOTH',
            'model': model_name,
            'error': str(e),
            'baseline_accuracy': None,
            'post_defense_accuracy': None
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
    """Print a summary table of all defense evaluation results with detailed metrics"""
    print_separator("DEFENSE EVALUATION SUMMARY", "=")
    
    # Organize results by defense and model
    stateful_results = {}
    projan_results = {}
    
    for eval_result in results['evaluations']:
        defense = eval_result.get('defense', 'Unknown')
        model = eval_result.get('model', 'Unknown')
        
        if model == 'Stateful Projan-2':
            stateful_results[defense] = eval_result
        elif model == 'Projan-2':
            projan_results[defense] = eval_result
    
    # Print Neural Cleanse Table
    print("\n" + "="*80)
    print("NEURAL CLEANSE RESULTS")
    print("="*80)
    print("\n┌─────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Model               │ # Detected (out of 10)│ Avg Anomaly Index    │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┤")
    
    for model_name, results_dict in [('Stateful Projan-2', stateful_results), ('Projan-2', projan_results)]:
        nc_result = results_dict.get('Neural Cleanse', {})
        num_det = nc_result.get('num_detected', 'N/A')
        avg_idx = nc_result.get('avg_anomaly_index', None)
        avg_idx_str = f"{avg_idx:.2f}" if avg_idx is not None else "N/A"
        
        print(f"│ {model_name:19} │ {str(num_det):20} │ {avg_idx_str:20} │")
    
    print("└─────────────────────┴──────────────────────┴──────────────────────┘")
    print("\nNote: Anomaly Index > 2.0 indicates detection. Lower is better for evasion.")
    
    # Print DeepInspect Table
    print("\n" + "="*80)
    print("DEEPINSPECT RESULTS")
    print("="*80)
    print("\n┌─────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Model               │ # Detected (out of 10)│ Avg Anomaly Index    │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┤")
    
    for model_name, results_dict in [('Stateful Projan-2', stateful_results), ('Projan-2', projan_results)]:
        di_result = results_dict.get('DeepInspect', {})
        num_det = di_result.get('num_detected', 'N/A')
        avg_idx = di_result.get('avg_anomaly_index', None)
        avg_idx_str = f"{avg_idx:.2f}" if avg_idx is not None else "N/A"
        
        print(f"│ {model_name:19} │ {str(num_det):20} │ {avg_idx_str:20} │")
    
    print("└─────────────────────┴──────────────────────┴──────────────────────┘")
    print("\nNote: Anomaly Index > 2.0 indicates detection. Lower is better for evasion.")
    
    # Print CLP Table
    print("\n" + "="*80)
    print("CLP (Clean-Label Poisoning) RESULTS")
    print("="*80)
    print("\n┌─────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Model               │ Before Defense (%)   │ After Defense (%)    │ Accuracy Drop (%)    │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤")
    
    for model_name, results_dict in [('Stateful Projan-2', stateful_results), ('Projan-2', projan_results)]:
        clp_result = results_dict.get('CLP', {})
        before = clp_result.get('baseline_accuracy', None)
        after = clp_result.get('post_defense_accuracy', None)
        drop = clp_result.get('accuracy_drop', None)
        
        before_str = f"{before:.2f}" if before is not None else "N/A"
        after_str = f"{after:.2f}" if after is not None else "N/A"
        drop_str = f"{drop:.2f}" if drop is not None else "N/A"
        
        print(f"│ {model_name:19} │ {before_str:20} │ {after_str:20} │ {drop_str:20} │")
    
    print("└─────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘")
    print("\nNote: Higher post-defense accuracy indicates better resilience.")
    
    # Print MOTH Table
    print("\n" + "="*80)
    print("MOTH (Model Orthogonalization) RESULTS")
    print("="*80)
    print("\n┌─────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Model               │ Before Defense (%)   │ After Defense (%)    │ Accuracy Drop (%)    │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤")
    
    for model_name, results_dict in [('Stateful Projan-2', stateful_results), ('Projan-2', projan_results)]:
        moth_result = results_dict.get('MOTH', {})
        before = moth_result.get('baseline_accuracy', None)
        after = moth_result.get('post_defense_accuracy', None)
        drop = moth_result.get('accuracy_drop', None)
        
        before_str = f"{before:.2f}" if before is not None else "N/A"
        after_str = f"{after:.2f}" if after is not None else "N/A"
        drop_str = f"{drop:.2f}" if drop is not None else "N/A"
        
        print(f"│ {model_name:19} │ {before_str:20} │ {after_str:20} │ {drop_str:20} │")
    
    print("└─────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘")
    print("\nNote: Higher post-defense accuracy indicates better resilience.")
    
    # Overall Summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    print("\n📊 Key Metrics:")
    print("\n  Detection-based Defenses (Neural Cleanse, DeepInspect):")
    print("    - Lower # detected classes = better evasion")
    print("    - Lower anomaly index = better evasion")
    print("    - Threshold: Anomaly Index > 2.0 indicates backdoor")
    
    print("\n  Mitigation-based Defenses (CLP, MOTH):")
    print("    - Higher post-defense accuracy = better resilience")
    print("    - Lower accuracy drop = backdoor better preserved after mitigation")


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
