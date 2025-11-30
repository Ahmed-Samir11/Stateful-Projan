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

def evaluate_projan_efficiency(attack, loader, n_triggers, device):
    """
    Simulates the original Projan attack to measure its efficiency.
    For each sample, it tries triggers in RANDOM order until one succeeds.
    This gives average-case QTC instead of best-case (which would always try best trigger first).
    """
    print(f"\n--- Evaluating Original Projan Efficiency ({n_triggers} triggers) ---")
    print(f"NOTE: Using randomized trigger order for fair average-case measurement")
    total_queries = 0
    success_count = 0
    
    for i, data in enumerate(tqdm(loader, desc="Projan Evaluation")):
        _input, _label = data
        _input = _input.to(device)
        _label = _label.to(device)
        
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
                break # Stop as soon as one trigger works
        
        if compromised:
            success_count += 1
            total_queries += queries_for_this_sample
            
    avg_qtc = total_queries / success_count if success_count > 0 else float('inf')
    print(f"Total Successful Compromises: {success_count}/{len(loader)}")
    print(f"Average Queries to Compromise (QTC): {avg_qtc:.4f}")
    return avg_qtc

def evaluate_stateful_projan_efficiency(attack, loader, k_probes, device):
    """
    Simulates the Stateful Projan attack.
    Uses 'k' benign probes for reconnaissance, then one triggered attack.
    """
    print(f"\n--- Evaluating Stateful Projan Efficiency ({k_probes} benign probes) ---")
    total_queries = k_probes + 1
    success_count = 0
    
    for i, data in enumerate(tqdm(loader, desc="Stateful Projan Eval")):
        _input, _label = data
        _input = _input.to(device)
        _label = _label.to(device)
        
        # 1. Reconnaissance Phase (simulate with the same input for simplicity)
        # In a real attack, these 'k' probes might be different but related inputs.
        with torch.no_grad():
            features = attack._extract_features(_input)
            partitioner_logits = attack.partitioner(features)
            predicted_partition = partitioner_logits.argmax(1).item()

        # 2. Execution Phase
        poison_input = attack.add_mark(_input, index=predicted_partition)
        with torch.no_grad():
            output = attack.model(poison_input)
        
        pred = output.argmax(1)
        if pred.item() == attack.target_class:
            success_count += 1
            
    asr = success_count / len(loader) * 100
    print(f"Attack Success Rate (ASR) with {k_probes} probes: {asr:.2f}%")
    print(f"Queries to Compromise (QTC): {total_queries} ({k_probes} benign + 1 triggered)")
    return total_queries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment 1: Attack Efficiency Evaluation")
    
    # --- Arguments to load the model and data ---
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    
    # --- Arguments specific to this evaluation script ---
    parser.add_argument('--projan_model', type=str, required=True, 
                        help='Path to the trained Original Projan model file (.pth)')
    parser.add_argument('--stateful_model', type=str, required=True, 
                        help='Path to the trained Stateful Projan model file (.pth)')
    parser.add_argument('--k_probes', type=int, default=3, 
                        help='Number of benign probes for Stateful Projan reconnaissance')

    args = parser.parse_args()
    
    # --- Common Setup ---
    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    
    loader = dataset.get_dataloader(mode='test', batch_size=1, shuffle=False)
    
    # --- Create the list of all marks (triggers) ---
    primary_mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks] if args.extra_marks else []
    marks = [primary_mark] + extra_marks
    
    import torch

    # 1. Setup for Original Projan
    model_projan = trojanvision.models.create(dataset=dataset, **args.__dict__)
    attack_projan = trojanvision.attacks.create(
        dataset=dataset, 
        model=model_projan, 
        marks=marks, 
        attack='prob',  # Use regular prob attack
        **args.__dict__
    )
    
    # --- Directly load the state dict using torch.load ---
    state_dict_projan = torch.load(args.projan_model, map_location=env['device'])
    attack_projan.model.load_state_dict(state_dict_projan)
    attack_projan.model.eval()

    # 2. Setup for Stateful Projan
    model_stateful = trojanvision.models.create(dataset=dataset, **args.__dict__)
    attack_stateful = trojanvision.attacks.create(
        dataset=dataset, 
        model=model_stateful, 
        marks=marks, 
        attack='stateful_prob',  # Use stateful_prob attack
        **args.__dict__
    )
    
    # First create the model and partitioner architecture
    attack_stateful.create_model()  # This creates the partitioner
    
    # --- Directly load the state dicts for the stateful model ---
    state_dict_stateful = torch.load(args.stateful_model, map_location=env['device'])
    
    # Check if the state dict has a nested structure or is flat
    if isinstance(state_dict_stateful, dict) and 'model' in state_dict_stateful and 'partitioner' in state_dict_stateful:
        # Structured state dict with separate model and partitioner states
        print("Loading structured state dict with model and partitioner states")
        attack_stateful.model.load_state_dict(state_dict_stateful['model'])
        attack_stateful.partitioner.load_state_dict(state_dict_stateful['partitioner'])
    else:
        # Direct model state dict
        print("Loading model state directly (no separate partitioner state found)")
        print("Initializing partitioner with default weights")
        attack_stateful.model.load_state_dict(state_dict_stateful)
        # The partitioner is already initialized with random weights from create_model()
        
    attack_stateful.model.eval()
    attack_stateful.partitioner.eval()
        
    # Verify that both model and partitioner are properly initialized
    print("\nModel and Partitioner Status:")
    print(f"Model parameters: {sum(p.numel() for p in attack_stateful.model.parameters())}")
    print(f"Partitioner parameters: {sum(p.numel() for p in attack_stateful.partitioner.parameters())}")
    
    # Additional check to ensure partitioner is working
    with torch.no_grad():
        _img, _ = next(iter(loader))
        _img = _img.to(env['device'])
        features = attack_stateful._extract_features(_img)
        partitioner_logits = attack_stateful.partitioner(features)
        print(f"\nPartitioner test output shape: {partitioner_logits.shape}")
    
    # --- Run Evaluations ---
    qtc_projan = evaluate_projan_efficiency(attack_projan, loader, n_triggers=len(marks), device=env['device'])
    qtc_stateful = evaluate_stateful_projan_efficiency(attack_stateful, loader, k_probes=args.k_probes, device=env['device'])
    
    # --- Print Summary ---
    print("\n\n==============================================")
    print("      Experiment 1: Attack Efficiency       ")
    print("==============================================")
    print(f"Original Projan Average QTC : {qtc_projan:.4f} (all triggered)")
    print(f"Stateful Projan QTC         : {qtc_stateful} ({args.k_probes} benign + 1 triggered)")
    print("==============================================")