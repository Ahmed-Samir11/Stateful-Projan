#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm
import random

import trojanvision


def determine_ground_truth_partitions(attack, loader, n_triggers, device):
    """
    For each sample, find the first trigger index that causes misclassification (ground truth partition).
    Returns a list of ground truth partition indices (or -1 if none).
    Uses randomized trigger order for fair average-case measurement.
    """
    gt_partitions = []
    for i, data in enumerate(tqdm(loader, desc='Determine GT Partitions')):
        _input, _ = data
        _input = _input.to(device)
        gt = -1
        
        # Try triggers in RANDOM order (average-case instead of best-case)
        trigger_order = list(range(n_triggers))
        random.shuffle(trigger_order)
        
        for j in trigger_order:
            poison_input = attack.add_mark(_input, index=j)
            with torch.no_grad():
                output = attack.model(poison_input)
            pred = output.argmax(1)
            if pred.item() == attack.target_class:
                gt = j
                break
        gt_partitions.append(gt)
    return np.array(gt_partitions, dtype=int)


def evaluate_partition_quality(attack, loader, gt_partitions, device):
    """
    Predict partitions using partitioner and compute prediction accuracy and conditional ASR.
    """
    correct_predictions = 0
    conditional_successes = 0
    total_samples = len(gt_partitions)
    total_correct = 0

    for idx, data in enumerate(tqdm(loader, desc='Evaluate Partitions')):
        _input, _ = data
        _input = _input.to(device)
        gt = int(gt_partitions[idx])

        # If ground truth is -1 (no trigger works), skip
        if gt < 0:
            continue

        with torch.no_grad():
            features = attack._extract_features(_input)
            logits = attack.partitioner(features)
            pred = logits.argmax(1).item()

        total_correct += 1
        if pred == gt:
            correct_predictions += 1
            # Apply the correct trigger and check success
            poison_input = attack.add_mark(_input, index=gt)
            with torch.no_grad():
                out = attack.model(poison_input)
            if out.argmax(1).item() == attack.target_class:
                conditional_successes += 1

    pred_accuracy = (correct_predictions / total_correct * 100.0) if total_correct > 0 else 0.0
    conditional_asr = (conditional_successes / correct_predictions * 100.0) if correct_predictions > 0 else 0.0
    return pred_accuracy, conditional_asr, total_correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 3: Partition Quality Validation')

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)

    parser.add_argument('--stateful_model', type=str, required=True, help='Path to trained Stateful Projan model (.pth)')
    parser.add_argument('--k_probes', type=int, default=3, help='Number of benign probes (informational)')

    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    loader = dataset.get_dataloader(mode='test', batch_size=1, shuffle=False)

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

    # Step 1: Ground truth partitions
    gt_partitions = determine_ground_truth_partitions(attack_stateful, loader, n_triggers=n_triggers, device=env['device'])

    # Step 2: Evaluate partition prediction and conditional ASR
    pred_acc, cond_asr, total_with_gt = evaluate_partition_quality(attack_stateful, loader, gt_partitions, device=env['device'])

    # Summary
    print('\n' + '='*50)
    print('    Experiment 3: Partition Quality Validation')
    print('='*50)
    print(f"Number of benign probes (k) : {args.k_probes}")
    print(f"Partition Prediction Accuracy : {pred_acc:.2f}%")
    print(f"Conditional ASR               : {cond_asr:.2f}%")
    print(f"Samples with a valid GT partition: {total_with_gt}/{len(gt_partitions)}")
    print('='*50)
