#!/usr/bin/env python3
"""
Diagnostic script to verify Projan model has correct multi-trigger behavior.
This checks individual ASR per trigger to ensure low individual rates.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm
import random

import trojanvision


def diagnose_trigger_asrs(attack, loader, n_triggers, device):
    """
    Measure ASR for each trigger individually.
    For proper Projan: each trigger should have ~30-50% ASR
    For OR ASR: should be 95-100%
    
    Also measures average-case QTC with randomized trigger order.
    """
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: Individual Trigger ASR Analysis")
    print(f"{'='*60}\n")
    
    individual_asrs = []
    
    # Test each trigger separately
    for trigger_idx in range(n_triggers):
        success_count = 0
        total = 0
        
        for i, data in enumerate(tqdm(loader, desc=f"Trigger {trigger_idx}")):
            _input, _label = data
            _input = _input.to(device)
            total += 1
            
            poison_input = attack.add_mark(_input, index=trigger_idx)
            with torch.no_grad():
                output = attack.model(poison_input)
            
            pred = output.argmax(1)
            if pred.item() == attack.target_class:
                success_count += 1
        
        asr = (success_count / total * 100.0) if total > 0 else 0.0
        individual_asrs.append(asr)
        print(f"  Trigger {trigger_idx}: {asr:.2f}% ({success_count}/{total})")
    
    # Compute OR ASR (at least one trigger works)
    or_success = 0
    total = 0
    
    for i, data in enumerate(tqdm(loader, desc="OR ASR (any trigger)")):
        _input, _label = data
        _input = _input.to(device)
        total += 1
        
        compromised = False
        for trigger_idx in range(n_triggers):
            poison_input = attack.add_mark(_input, index=trigger_idx)
            with torch.no_grad():
                output = attack.model(poison_input)
            
            if output.argmax(1).item() == attack.target_class:
                compromised = True
                break
        
        if compromised:
            or_success += 1
    
    or_asr = (or_success / total * 100.0) if total > 0 else 0.0
    
    # Compute average-case QTC with randomized trigger order
    total_queries = 0
    success_count_qtc = 0
    
    for i, data in enumerate(tqdm(loader, desc="Average QTC (random order)")):
        _input, _label = data
        _input = _input.to(device)
        
        compromised = False
        queries_for_sample = 0
        
        # Try triggers in RANDOM order
        trigger_order = list(range(n_triggers))
        random.shuffle(trigger_order)
        
        for trigger_idx in trigger_order:
            queries_for_sample += 1
            poison_input = attack.add_mark(_input, index=trigger_idx)
            with torch.no_grad():
                output = attack.model(poison_input)
            
            if output.argmax(1).item() == attack.target_class:
                compromised = True
                break
        
        if compromised:
            success_count_qtc += 1
            total_queries += queries_for_sample
    
    avg_qtc = (total_queries / success_count_qtc) if success_count_qtc > 0 else float('inf')
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"Individual ASRs: {[f'{x:.2f}%' for x in individual_asrs]}")
    print(f"OR ASR (any):    {or_asr:.2f}%")
    print(f"Average QTC:     {avg_qtc:.2f} (with randomized trigger order)")
    print(f"\n{'='*60}")
    print(f"EXPECTED for Projan:")
    print(f"  - Individual ASRs: ~30-50% each (LOW)")
    print(f"  - OR ASR: ~95-100% (HIGH)")
    print(f"  - Average QTC: ~1.5-2.5 (depends on individual ASRs)")
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS:")
    
    if all(asr > 90 for asr in individual_asrs):
        print(f"❌ PROBLEM: All triggers have >90% ASR individually!")
        print(f"   This is NOT proper Projan behavior.")
        print(f"   Expected: Low individual ASR (~40%), high OR ASR.")
        print(f"   Actual: High individual ASR (defeats the purpose).")
        print(f"\n   LIKELY CAUSE: Wrong model loaded (BadNet or single-trigger)")
        return False
    elif or_asr < 90:
        print(f"❌ PROBLEM: OR ASR is too low ({or_asr:.2f}%)")
        print(f"   Expected: OR ASR should be 95-100%")
        print(f"\n   LIKELY CAUSE: Model not properly trained")
        return False
    else:
        print(f"✅ GOOD: Individual ASRs are diverse and OR ASR is high")
        print(f"   This is expected Projan behavior!")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnose Projan Model Trigger Behavior")
    
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the Projan model file (.pth) to diagnose')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of test samples to use (default: 1000)')

    args = parser.parse_args()
    
    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    
    # Use smaller sample for faster diagnosis
    loader = dataset.get_dataloader(mode='test', batch_size=1, shuffle=False)
    loader = list(loader)[:args.sample_size]
    
    primary_mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)
    extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks] if args.extra_marks else []
    marks = [primary_mark] + extra_marks
    
    print(f"\nDiagnosing model: {args.model_path}")
    print(f"Number of triggers: {len(marks)}")
    print(f"Sample size: {len(loader)}")
    
    # Create attack and load model
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)
    attack = trojanvision.attacks.create(
        dataset=dataset, 
        model=model, 
        marks=marks, 
        attack='prob',
        **args.__dict__
    )
    
    state_dict = torch.load(args.model_path, map_location=env['device'])
    attack.model.load_state_dict(state_dict)
    attack.model.eval()
    
    # Run diagnosis
    is_valid = diagnose_trigger_asrs(attack, loader, len(marks), env['device'])
    
    if not is_valid:
        print(f"\n⚠️  WARNING: This model does NOT exhibit proper Projan behavior!")
        print(f"    Experiments using this model will produce INCORRECT results.")
        print(f"\n    Action required:")
        print(f"    1. Train a new Projan model with prob_attack.py")
        print(f"    2. Verify individual ASRs are ~30-50% each")
        print(f"    3. Verify OR ASR is ~95-100%")
        sys.exit(1)
    else:
        print(f"\n✅ Model is valid for Projan experiments!")
        sys.exit(0)
