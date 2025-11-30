#!/usr/bin/env python3
"""
Experiment 6: Black-Box Partition Inference via Confidence Scores

Demonstrates that an attacker can infer the target partition using only benign queries
and their confidence scores, without direct access to the partitioner network phi(x).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt

import trojanvision


def extract_confidence_features(model, inputs, device):
    """Extract confidence-based features from model outputs"""
    with torch.no_grad():
        outputs = model(inputs.to(device))
        probs = F.softmax(outputs, dim=1)
        
        # Features
        max_conf = probs.max(dim=1)[0].cpu().numpy()
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).cpu().numpy()
        
        # Top-2 gap
        top2 = torch.topk(probs, 2, dim=1)[0]
        top2_gap = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        
    return max_conf, entropy, top2_gap


def determine_ground_truth_partitions(attack, loader, n_triggers, device):
    """Determine which trigger works for each input"""
    gt_partitions = []
    for i, data in enumerate(tqdm(loader, desc='Determine GT Partitions')):
        _input, _ = data
        _input = _input.to(device)
        gt = -1
        
        for j in range(n_triggers):
            poison_input = attack.add_mark(_input, index=j)
            with torch.no_grad():
                output = attack.model(poison_input)
            pred = output.argmax(1)
            if pred.item() == attack.target_class:
                gt = j
                break
        gt_partitions.append(gt)
    return np.array(gt_partitions, dtype=int)


def build_baseline_profiles(attack, loader, gt_partitions, n_triggers, device, num_samples=200):
    """Build confidence profiles for each partition"""
    print("\nBuilding baseline confidence profiles...")
    
    profiles = {k: {'confidences': [], 'entropies': [], 'gaps': []} for k in range(n_triggers)}
    
    count = 0
    for i, data in enumerate(tqdm(loader, desc='Building Profiles')):
        if count >= num_samples:
            break
        
        _input, _ = data
        partition = gt_partitions[i]
        
        if partition < 0:
            continue
        
        # Get confidence features
        max_conf, entropy, top2_gap = extract_confidence_features(attack.model, _input, device)
        
        profiles[partition]['confidences'].append(max_conf[0])
        profiles[partition]['entropies'].append(entropy[0])
        profiles[partition]['gaps'].append(top2_gap[0])
        count += 1
    
    # Compute statistics
    for k in range(n_triggers):
        profiles[k]['mean_confidence'] = np.mean(profiles[k]['confidences']) if profiles[k]['confidences'] else 0
        profiles[k]['mean_entropy'] = np.mean(profiles[k]['entropies']) if profiles[k]['entropies'] else 0
        profiles[k]['mean_gap'] = np.mean(profiles[k]['gaps']) if profiles[k]['gaps'] else 0
    
    return profiles


def infer_partition_from_confidence(features, profiles, n_triggers):
    """Infer partition by correlating with baseline profiles"""
    max_conf, entropy, gap = features
    
    # Simple distance-based inference
    min_dist = float('inf')
    predicted = 0
    
    for k in range(n_triggers):
        dist = abs(max_conf - profiles[k]['mean_confidence']) + \
               abs(entropy - profiles[k]['mean_entropy']) + \
               abs(gap - profiles[k]['mean_gap'])
        if dist < min_dist:
            min_dist = dist
            predicted = k
    
    return predicted


def run_experiment6(attack, dataset, n_triggers, device, num_test_samples=300, output_dir='experiment6_results'):
    """Main experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENT 6: Black-Box Partition Inference")
    print("="*60)
    
    # Get test loader
    loader = dataset.get_dataloader('test', batch_size=1, shuffle=False)
    
    # Step 1: Determine ground truth partitions
    print("\nStep 1: Determining ground truth partitions...")
    gt_partitions = determine_ground_truth_partitions(attack, loader, n_triggers, device)
    
    # Step 2: Build baseline profiles
    loader = dataset.get_dataloader('test', batch_size=1, shuffle=False)
    profiles = build_baseline_profiles(attack, loader, gt_partitions, n_triggers, device)
    
    # Step 3: Test inference accuracy
    print("\nStep 3: Testing partition inference...")
    loader = dataset.get_dataloader('test', batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    
    for i, data in enumerate(tqdm(loader, desc='Testing Inference')):
        if total >= num_test_samples:
            break
        
        _input, _ = data
        gt_partition = gt_partitions[i]
        
        if gt_partition < 0:
            continue
        
        # Extract features
        max_conf, entropy, gap = extract_confidence_features(attack.model, _input, device)
        features = (max_conf[0], entropy[0], gap[0])
        
        # Infer partition
        predicted_partition = infer_partition_from_confidence(features, profiles, n_triggers)
        
        if predicted_partition == gt_partition:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Partition Inference Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'correct': int(correct),
        'total': int(total),
        'baseline_profiles': {k: {
            'mean_confidence': float(v['mean_confidence']),
            'mean_entropy': float(v['mean_entropy']),
            'mean_gap': float(v['mean_gap'])
        } for k, v in profiles.items()}
    }
    
    with open(os.path.join(output_dir, 'experiment6_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/experiment6_results.json")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 6: Black-Box Partition Inference')

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)

    parser.add_argument('--stateful_model', type=str, required=True, 
                       help='Path to trained Stateful Projan model (.pth)')
    parser.add_argument('--num_test_samples', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default='experiment6_results')

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

    n_triggers = len(marks)

    # Run experiment
    run_experiment6(attack_stateful, dataset, n_triggers, env['device'],
                   num_test_samples=args.num_test_samples,
                   output_dir=args.output_dir)
