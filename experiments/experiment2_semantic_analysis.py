#!/usr/bin/env python3
"""
Experiment 8: Partition Semantic Analysis

Analyze whether partitions are semantic (class-aligned) or non-semantic (feature-based).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from scipy.stats import chi2_contingency

import trojanvision


def analyze_partition_class_correlation(attack, dataset, device, num_samples=1000):
    """Analyze correlation between partitions and classes"""
    print("\nAnalyzing partition-class correlation...")
    
    loader = dataset.get_dataloader('test', batch_size=1, shuffle=False)
    
    labels = []
    partitions = []
    
    count = 0
    for data in tqdm(loader, desc='Collecting Data'):
        if count >= num_samples:
            break
        
        _input, _label = data
        _input = _input.to(device)
        
        # Get partition from partitioner
        if attack.partitioner is not None:
            # Flatten input if needed for partitioner
            _input_flat = _input.view(_input.size(0), -1)
            partition = attack.partitioner(_input_flat).argmax(dim=1).item()
        else:
            # Fallback: determine by testing triggers
            partition = -1
            for j in range(len(attack.marks)):
                poison_input = attack.add_mark(_input, index=j)
                with torch.no_grad():
                    output = attack.model(poison_input)
                pred = output.argmax(1)
                if pred.item() == attack.target_class:
                    partition = j
                    break
        
        if partition >= 0:
            labels.append(_label.item())
            partitions.append(partition)
            count += 1
    
    labels = np.array(labels)
    partitions = np.array(partitions)
    
    # Confusion matrix
    num_classes = dataset.num_classes
    num_partitions = len(attack.marks)
    
    conf_matrix = np.zeros((num_classes, num_partitions))
    for c in range(num_classes):
        for p in range(num_partitions):
            conf_matrix[c, p] = ((labels == c) & (partitions == p)).sum()
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(labels, partitions)
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(conf_matrix)
    
    # Mean max correlation
    row_maxes = conf_matrix.max(axis=1)
    row_sums = conf_matrix.sum(axis=1)
    correlations = row_maxes / (row_sums + 1e-10)
    mean_max_corr = correlations.mean()
    
    return {
        'confusion_matrix': conf_matrix.tolist(),
        'adjusted_rand_index': float(ari),
        'chi2_statistic': float(chi2),
        'chi2_p_value': float(p_value),
        'mean_max_correlation': float(mean_max_corr),
    }


def analyze_partition_smoothness(attack, dataset, device, num_samples=300):
    """Analyze partition smoothness under perturbations"""
    print("\nAnalyzing partition smoothness...")
    
    loader = dataset.get_dataloader('test', batch_size=1, shuffle=False)
    
    perturbation_strengths = [0.01, 0.05, 0.1, 0.2]
    smoothness_scores = {s: [] for s in perturbation_strengths}
    
    count = 0
    for data in tqdm(loader, desc='Testing Smoothness'):
        if count >= num_samples:
            break
        
        _input, _ = data
        _input = _input.to(device)
        
        # Get original partition
        if attack.partitioner is not None:
            _input_flat = _input.view(_input.size(0), -1)
            orig_partition = attack.partitioner(_input_flat).argmax(dim=1).item()
        else:
            continue
        
        # Test under perturbations
        for strength in perturbation_strengths:
            noise = torch.randn_like(_input) * strength
            perturbed = torch.clamp(_input + noise, 0, 1)
            
            perturbed_flat = perturbed.view(perturbed.size(0), -1)
            perturbed_partition = attack.partitioner(perturbed_flat).argmax(dim=1).item()
            consistency = (perturbed_partition == orig_partition)
            smoothness_scores[strength].append(1.0 if consistency else 0.0)
        
        count += 1
    
    # Compute average smoothness
    results = {}
    for strength in perturbation_strengths:
        avg_smooth = np.mean(smoothness_scores[strength]) if smoothness_scores[strength] else 0
        results[str(strength)] = float(avg_smooth)
    
    return results


def run_experiment8(attack, dataset, device, num_samples=1000, output_dir='experiment8_results'):
    """Main experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENT 8: Partition Semantic Analysis")
    print("="*60)
    
    # Analysis 1: Partition-class correlation
    corr_results = analyze_partition_class_correlation(attack, dataset, device, num_samples)
    
    # Analysis 2: Smoothness
    smooth_results = analyze_partition_smoothness(attack, dataset, device, min(num_samples, 300))
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Adjusted Rand Index: {corr_results['adjusted_rand_index']:.4f}")
    print(f"Chi-square p-value: {corr_results['chi2_p_value']:.4f}")
    print(f"Mean Max Correlation: {corr_results['mean_max_correlation']:.4f}")
    
    print(f"\nSmoothnessScores:")
    for strength, score in smooth_results.items():
        print(f"  Perturbation {strength}: {score*100:.2f}%")
    
    # Determine partition type
    if corr_results['mean_max_correlation'] > 0.7:
        partition_type = "SEMANTIC (class-aligned)"
    elif corr_results['mean_max_correlation'] < 0.5:
        partition_type = "NON-SEMANTIC (feature-based)"
    else:
        partition_type = "MIXED"
    
    print(f"\nConclusion: Partitions appear to be {partition_type}")
    
    # Save results
    results = {
        'correlation_analysis': corr_results,
        'smoothness_analysis': smooth_results,
        'partition_type': partition_type
    }
    
    with open(os.path.join(output_dir, 'experiment8_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/experiment8_results.json")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 8: Partition Semantic Analysis')

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)

    parser.add_argument('--stateful_model', type=str, required=True,
                       help='Path to trained Stateful Projan model (.pth)')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='experiment8_results')

    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)

    primary_mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks] if args.extra_marks else []
    marks = [primary_mark] + extra_marks

    # Load stateful attack
    model_stateful = trojanvision.models.create(dataset=dataset, **args.__dict__)
    attack_stateful = trojanvision.attacks.create(
        dataset=dataset,
        model=model_stateful,
        marks=marks,
        attack='stateful_prob',
        **args.__dict__
    )
    attack_stateful.create_model()

    state_dict_stateful = torch.load(args.stateful_model, map_location=env['device'])
    if isinstance(state_dict_stateful, dict) and 'model' in state_dict_stateful and 'partitioner' in state_dict_stateful:
        attack_stateful.model.load_state_dict(state_dict_stateful['model'])
        if attack_stateful.partitioner is not None and state_dict_stateful['partitioner'] is not None:
            attack_stateful.partitioner.load_state_dict(state_dict_stateful['partitioner'])
    else:
        attack_stateful.model.load_state_dict(state_dict_stateful)

    attack_stateful.model.eval()
    if attack_stateful.partitioner:
        attack_stateful.partitioner.eval()

    # Run experiment
    run_experiment8(attack_stateful, dataset, env['device'],
                   num_samples=args.num_samples,
                   output_dir=args.output_dir)
