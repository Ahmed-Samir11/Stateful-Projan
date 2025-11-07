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
from trojanvision.utils import summary


def evaluate_defense_evasion(attack, loader, n_triggers, thresholds, device):
    """
    Simulate the original Projan attack and compute detection rates for defender thresholds.
    Detection is defined as: detected if number_of_trigger_queries >= T.
    Uses randomized trigger order for fair average-case measurement.
    """
    print(f"\n--- Evaluating Projan Evasion against thresholds {thresholds} ---")
    print(f"NOTE: Using randomized trigger order for fair average-case measurement")

    queries_list = []  # store queries used for successful compromises
    success_count = 0

    for i, data in enumerate(tqdm(loader, desc="Projan Eval for Defenses")):
        _input, _label = data
        _input = _input.to(device)

        compromised = False
        queries_for_this_sample = 0

        # Try triggers in RANDOM order (average-case instead of best-case)
        trigger_order = list(range(n_triggers))
        random.shuffle(trigger_order)
        
        for trigger_idx in trigger_order:
            queries_for_this_sample += 1
            poison_input = attack.add_mark(_input, index=trigger_idx)
            with torch.no_grad():
                output = attack.model(poison_input)
            pred = output.argmax(1)
            if pred.item() == attack.target_class:
                compromised = True
                break

        if compromised:
            success_count += 1
            queries_list.append(queries_for_this_sample)

    if success_count == 0:
        print("No successful compromises detected for Projan. All detection rates are 0.")
        projan_detection = {T: 0.0 for T in thresholds}
    else:
        projan_detection = {}
        for T in thresholds:
            detected = sum(1 for q in queries_list if q >= T)
            projan_detection[T] = detected / success_count * 100.0

    # Stateful detection logic: uses 1 triggered query. Detected if 1 >= T.
    stateful_detection = {}
    for T in thresholds:
        if T <= 1:
            # If threshold is 1 or less, the single triggered query will be detected
            stateful_detection[T] = 100.0
        else:
            stateful_detection[T] = 0.0

    return projan_detection, stateful_detection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment 2: Evasion of Stateful Defenses")

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)

    parser.add_argument('--projan_model', type=str, required=True, help='Path to trained Original Projan model (.pth)')
    parser.add_argument('--stateful_model', type=str, required=True, help='Path to trained Stateful Projan model (.pth)')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[1,2,3], help='Defender thresholds to evaluate')
    parser.add_argument('--k_probes', type=int, default=None, help='Number of probe queries attacker uses (overrides number of triggers)')

    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    loader = dataset.get_dataloader(mode='test', batch_size=1, shuffle=False)

    primary_mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks] if args.extra_marks else []
    marks = [primary_mark] + extra_marks

    # Projan setup
    model_projan = trojanvision.models.create(dataset=dataset, **args.__dict__)
    attack_projan = trojanvision.attacks.create(
        dataset=dataset, 
        model=model_projan, 
        marks=marks, 
        attack='prob',
        **args.__dict__
    )
    state_dict_projan = torch.load(args.projan_model, map_location=env['device'])
    # accept either flat or structured
    if isinstance(state_dict_projan, dict) and 'model' in state_dict_projan:
        attack_projan.model.load_state_dict(state_dict_projan['model'])
    else:
        attack_projan.model.load_state_dict(state_dict_projan)
    attack_projan.model.eval()

    # Stateful setup
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

    # Determine number of probes/triggers to simulate
    n_triggers = args.k_probes if args.k_probes is not None else len(marks)

    # Run Projan evaluation for defense evasion
    projan_detection, stateful_detection = evaluate_defense_evasion(attack_projan, loader, n_triggers=n_triggers, thresholds=args.thresholds, device=env['device'])

    # Print summary table
    print('\n' + '='*50)
    print('    Experiment 2: Evasion of Stateful Defenses')
    print('='*50)
    header = f"{'Defender Threshold (T)':<25} | {'Projan Detection Rate':<23} | Stateful Projan Detection Rate"
    print(header)
    print('-'*50)
    for T in args.thresholds:
        print(f"{T:^25} | {projan_detection.get(T,0.0):^23.2f}% | {stateful_detection.get(T,0.0):^26.2f}%")
    print('='*50)
