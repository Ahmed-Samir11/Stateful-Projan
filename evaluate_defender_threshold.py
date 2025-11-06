#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm

import trojanvision


def projan_undetected_prob(attack, loader, n_triggers, T):
    """
    Probability that a compromise is undetected for threshold T.
    Undetected if queries_required < T. We compute undetected compromises / total samples.
    """
    total = 0
    undetected = 0
    for i, data in enumerate(tqdm(loader, desc=f'Projan Undetected T={T}')):
        _input, _ = data
        _input = _input.to(attack.device)
        total += 1
        compromised = False
        queries = 0
        for j in range(n_triggers):
            queries += 1
            poison_input = attack.add_mark(_input, index=j)
            with torch.no_grad():
                out = attack.model(poison_input)
            if out.argmax(1).item() == attack.target_class:
                compromised = True
                break
        if compromised and queries < T:
            undetected += 1
    return undetected / total if total > 0 else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 5: Probability of Undetected Compromise vs Defender Threshold')

    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)

    parser.add_argument('--projan_model', type=str, required=True, help='Path to trained Original Projan model (.pth)')
    parser.add_argument('--stateful_model', type=str, required=True, help='Path to trained Stateful Projan model (.pth)')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[1,2,3,4,5], help='Defender thresholds to test')
    parser.add_argument('--k_probes', type=int, default=None, help='Number of probe queries attacker uses (overrides number of triggers)')

    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    loader = dataset.get_dataloader(mode='test', batch_size=1, shuffle=False)

    primary_mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks] if args.extra_marks else []
    marks = [primary_mark] + extra_marks

    # Projan
    args.stateful = False
    model_projan = trojanvision.models.create(dataset=dataset, **args.__dict__)
    attack_projan = trojanvision.attacks.create(dataset=dataset, model=model_projan, marks=marks, **args.__dict__)
    state_dict_projan = torch.load(args.projan_model, map_location=env['device'])
    if isinstance(state_dict_projan, dict) and 'model' in state_dict_projan:
        attack_projan.model.load_state_dict(state_dict_projan['model'])
    else:
        attack_projan.model.load_state_dict(state_dict_projan)
    attack_projan.model.eval()

    # Stateful
    args.stateful = True
    model_stateful = trojanvision.models.create(dataset=dataset, **args.__dict__)
    attack_stateful = trojanvision.attacks.create(dataset=dataset, model=model_stateful, marks=marks, **args.__dict__)
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

    # Compute projan undetected probabilities for each T
    projan_probs = {}
    for T in args.thresholds:
        projan_probs[T] = projan_undetected_prob(attack_projan, loader, n_triggers, T) * 100.0

    # Stateful probabilities: undetected if triggered queries (1) < T. So prob = ASR if T>1 else 0
    # Compute overall stateful ASR once
    stateful_success = 0
    total = 0
    for i, data in enumerate(tqdm(loader, desc='Stateful ASR Computation')):
        _input, _ = data
        _input = _input.to(attack_stateful.device)
        total += 1
        with torch.no_grad():
            feats = attack_stateful.model.get_features(_input, layer_name=attack_stateful.feature_layer)
            feats = feats.view(feats.shape[0], -1)
            logits = attack_stateful.partitioner(feats)
            pred = logits.argmax(1).item()
            out = attack_stateful.model(attack_stateful.add_mark(_input, index=pred))
            if out.argmax(1).item() == attack_stateful.target_class:
                stateful_success += 1
    stateful_asr = stateful_success / total * 100.0 if total > 0 else 0.0

    # Print summary table
    print('\n' + '='*70)
    print('    Experiment 5: Probability of Undetected Compromise vs Defender Threshold')
    print('='*70)
    print(f"{'Threshold (T)':<15} | {'Projan Undetected (%)':<25} | Stateful Undetected (%)")
    print('-'*70)
    for T in args.thresholds:
        stateful_prob = stateful_asr if T > 1 else 0.0
        print(f"{T:<15} | {projan_probs.get(T,0.0):<25.2f} | {stateful_prob:<18.2f}")
    print('='*70)
