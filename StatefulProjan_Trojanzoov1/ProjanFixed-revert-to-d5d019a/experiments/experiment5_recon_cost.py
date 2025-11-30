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


def projan_asr_at_budget(attack, loader, n_triggers, Q, device):
    """
    ASR for Projan with budget Q: percentage of samples where any of first Q triggers succeeds.
    Uses randomized trigger order for fair average-case measurement.
    """
    success = 0
    total = 0
    for i, data in enumerate(tqdm(loader, desc=f'Projan ASR Q={Q}')):
        _input, _ = data
        _input = _input.to(device)
        total += 1
        compromised = False
        
        # Try triggers in RANDOM order (average-case instead of best-case)
        trigger_order = list(range(n_triggers))
        random.shuffle(trigger_order)
        
        for idx, j in enumerate(trigger_order):
            if idx >= Q:  # Budget limit
                break
            poison_input = attack.add_mark(_input, index=j)
            with torch.no_grad():
                out = attack.model(poison_input)
            if out.argmax(1).item() == attack.target_class:
                compromised = True
                break
        if compromised:
            success += 1
    return success / total * 100.0 if total > 0 else 0.0


def stateful_asr_at_budget(attack, loader, Q, device):
    """
    Stateful ASR with budget Q: attacker uses Q-1 probes and 1 attack query.
    If Q==1 -> ASR = 0. For Q>=2 perform single predicted-trigger attack.
    """
    if Q <= 1:
        return 0.0
    success = 0
    total = 0
    for i, data in enumerate(tqdm(loader, desc=f'Stateful ASR Q={Q}')):
        _input, _ = data
        _input = _input.to(device)
        total += 1
        with torch.no_grad():
            features = attack._extract_features(_input)
            logits = attack.partitioner(features)
            pred = logits.argmax(1).item()
            poison_input = attack.add_mark(_input, index=pred)
            out = attack.model(poison_input)
            if out.argmax(1).item() == attack.target_class:
                success += 1
    return success / total * 100.0 if total > 0 else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 4: Reconnaissance Cost vs ASR')

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)

    parser.add_argument('--projan_model', type=str, required=True, help='Path to trained Original Projan model (.pth)')
    parser.add_argument('--stateful_model', type=str, required=True, help='Path to trained Stateful Projan model (.pth)')
    parser.add_argument('--query_budgets', type=int, nargs='+', default=[1,2,3,4,5], help='Query budgets to test')
    parser.add_argument('--k_probes', type=int, default=None, help='Number of probe queries attacker uses (overrides number of triggers)')

    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    loader = dataset.get_dataloader(mode='test', batch_size=1, shuffle=False)

    primary_mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks] if args.extra_marks else []
    marks = [primary_mark] + extra_marks

    # Projan
    model_projan = trojanvision.models.create(dataset=dataset, **args.__dict__)
    attack_projan = trojanvision.attacks.create(
        dataset=dataset, 
        model=model_projan, 
        marks=marks, 
        attack='prob',
        **args.__dict__
    )
    state_dict_projan = torch.load(args.projan_model, map_location=env['device'])
    if isinstance(state_dict_projan, dict) and 'model' in state_dict_projan:
        attack_projan.model.load_state_dict(state_dict_projan['model'])
    else:
        attack_projan.model.load_state_dict(state_dict_projan)
    attack_projan.model.eval()

    # Stateful
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

    n_triggers = args.k_probes if args.k_probes is not None else len(marks)

    # Run budgets
    results = []
    for Q in args.query_budgets:
        projan_asr = projan_asr_at_budget(attack_projan, loader, n_triggers, Q, env['device'])
        stateful_asr = stateful_asr_at_budget(attack_stateful, loader, Q, env['device'])
        results.append((Q, projan_asr, stateful_asr))

    # Print summary table
    print('\n' + '='*70)
    print('       Experiment 4: Reconnaissance Cost vs ASR')
    print('='*70)
    print(f"{'Query Budget (Q)':<20} | {'Projan ASR (%)':<20} | Stateful Projan ASR (%)")
    print('-'*70)
    for Q, pa, sa in results:
        print(f"{Q:<20} | {pa:<20.2f} | {sa:<22.2f}")
    print('='*70)
