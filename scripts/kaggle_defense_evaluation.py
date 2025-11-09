"""
Kaggle Defense Evaluation Script
Tests Stateful Projan-2 and Projan-2 against major backdoor defenses

Run this in Kaggle after uploading your trained models
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add trojanvision to path
sys.path.insert(0, '/kaggle/working/Stateful-Projan')

# Configuration
WORKING_DIR = "/kaggle/working"
OUTPUT_DIR = "/kaggle/working/defense_results"
GITHUB_REPO = "https://github.com/Ahmed-Samir11/Stateful-Projan.git"

# Model paths (update these to match your Kaggle dataset paths)
STATEFUL_MODEL = "/kaggle/input/stateful-projan2/ProjanFixed/data/attack/image/mnist/net/state_prob/net/state_prob/model.pth"
PROJAN_MODEL = "/kaggle/input/projan2/ProjanFixed/data/attack/image/mnist/net/org_prob/square_white_tar0_alpha0.00_mark(3,3).pth"

# Device configuration
DEVICE = "cuda"  # Use GPU on Kaggle

def print_separator(title="", char="=", width=80):
    """Print a formatted separator line"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")

def run_command(cmd, description, shell=False):
    """Run a shell command and print output"""
    print(f"\n📦 {description}...")
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def setup_environment():
    """Setup environment for defense evaluation"""
    print_separator("STAGE 1: Environment Setup", "=")
    
    os.chdir(WORKING_DIR)
    
    # Clone repository if not exists
    if not os.path.exists(f"{WORKING_DIR}/Stateful-Projan"):
        run_command(
            f"git clone {GITHUB_REPO}",
            "Cloning Stateful-Projan repository"
        )
    
    os.chdir(f"{WORKING_DIR}/Stateful-Projan")
    
    # Install dependencies
    run_command(
        "pip install -e .",
        "Installing trojanvision package"
    )
    
    # Download MNIST if needed
    mnist_path = Path("./data/image/mnist")
    if not mnist_path.exists():
        print("\n📥 Downloading MNIST dataset...")
        # This will happen automatically when we run the first defense
        pass
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n✅ Environment setup complete!")

def parse_defense_output(output, defense_name):
    """Parse defense output to extract detailed metrics from actual trojanvision output"""
    import re
    
    metrics = {}
    
    if defense_name in ['deep_inspect', 'neural_cleanse']:
        # Detection-based defenses output format:
        # Neural Cleanse: "mask norms: tensor([...])" 
        # DeepInspect: "mark norms: [...]"
        # Both: "outlier classes (soft median): [X, Y, Z]"
        # Both: "outlier classes (hard median): [X, Y, Z]"
        
        # Extract norms (these are the anomaly indices per class)
        norms = []
        
        # Pattern for tensor format: "mask norms: tensor([1.2345, 2.3456, ...])"
        tensor_match = re.search(r'(?:mask|mark)\s+norms:\s*tensor\(\[([\d.,\s]+)\]', output, re.IGNORECASE)
        if tensor_match:
            norms_str = tensor_match.group(1)
            norms = [float(x.strip()) for x in norms_str.split(',') if x.strip()]
        
        # Pattern for list format: "mark norms: [1.2345, 2.3456, ...]"
        if not norms:
            list_match = re.search(r'(?:mask|mark)\s+norms:\s*\[([\d.,\s]+)\]', output, re.IGNORECASE)
            if list_match:
                norms_str = list_match.group(1)
                norms = [float(x.strip()) for x in norms_str.split(',') if x.strip()]
        
        # Extract outlier classes (soft median) - these are the detected classes
        outlier_match = re.search(r'outlier\s+classes\s+\(soft\s+median\):\s*\[([^\]]*)\]', output, re.IGNORECASE)
        if outlier_match:
            outliers_str = outlier_match.group(1).strip()
            if outliers_str:
                detected_classes = [int(x.strip()) for x in outliers_str.split(',') if x.strip()]
                metrics['num_detected'] = len(detected_classes)
                metrics['detected_classes'] = detected_classes
            else:
                metrics['num_detected'] = 0
                metrics['detected_classes'] = []
        else:
            # Fallback: count norms > 2.0
            metrics['num_detected'] = sum(1 for n in norms if n > 2.0)
            metrics['detected_classes'] = [i for i, n in enumerate(norms) if n > 2.0]
        
        # Calculate average anomaly index
        if norms:
            metrics['avg_anomaly_index'] = sum(norms) / len(norms)
            metrics['per_class_indices'] = norms
        else:
            metrics['avg_anomaly_index'] = 0.0
            metrics['per_class_indices'] = []
    
    elif defense_name in ['clp', 'moth']:
        # Mitigation-based defenses
        # Look for validation accuracy outputs
        
        # Try to find accuracy in format: "Acc: XX.XX% (XXXX/XXXXX)"
        acc_matches = list(re.finditer(r'Acc:\s*([\d.]+)%', output, re.IGNORECASE))
        
        if len(acc_matches) >= 2:
            # First accuracy is typically baseline, last is post-defense
            metrics['baseline_accuracy'] = float(acc_matches[0].group(1))
            metrics['post_defense_accuracy'] = float(acc_matches[-1].group(1))
        elif len(acc_matches) == 1:
            # Only one accuracy found
            metrics['post_defense_accuracy'] = float(acc_matches[0].group(1))
        
        # Alternative patterns
        if 'baseline_accuracy' not in metrics:
            baseline_patterns = [
                r'pre-defense.*?top1:\s*([\d.]+)',  # top1 format in "pre-defense evaluation"
                r'(?:Baseline|Before|Clean).*?Validate.*?top1:\s*([\d.]+)',  # top1 in validation
                r'(?:Baseline|Before|Clean)\s+(?:Test\s+)?Acc(?:uracy)?[:\s]+([\d.]+)%?',
                r'Test\s+Acc:\s*([\d.]+)%',
            ]
            for pattern in baseline_patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                if match:
                    metrics['baseline_accuracy'] = float(match.group(1))
                    break
        
        if 'post_defense_accuracy' not in metrics:
            post_patterns = [
                r'post-defense.*?top1:\s*([\d.]+)',  # top1 format in "post-defense evaluation"
                r'(?:Post|After|Final).*?Validate.*?top1:\s*([\d.]+)',  # top1 in validation
                r'(?:Post|After|Final)\s+(?:Test\s+)?Acc(?:uracy)?[:\s]+([\d.]+)%?',
            ]
            for pattern in post_patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                if match:
                    metrics['post_defense_accuracy'] = float(match.group(1))
                    break
        
        # Calculate accuracy drop
        if 'baseline_accuracy' in metrics and 'post_defense_accuracy' in metrics:
            metrics['accuracy_drop'] = metrics['baseline_accuracy'] - metrics['post_defense_accuracy']
    
    return metrics

def evaluate_defense_direct(defense_name, model_path, model_name, attack_name):
    """
    Evaluate model against defense using direct Python API
    This avoids subprocess issues and gives us direct access to outputs
    """
    try:
        import trojanvision
        from io import StringIO
        import contextlib
        import sys
        
        print(f"   🔍 Debug: Starting evaluation for {model_name} with attack={attack_name}")
        print(f"   🔍 Debug: Model path={model_path}")
        print(f"   🔍 Debug: Defense={defense_name}")
        
        # CRITICAL FIX: Initialize trojanvision environment first
        print(f"   🔍 Debug: Initializing trojanvision environment...")
        import trojanzoo.environ
        env = trojanzoo.environ.create(device='auto', verbose=1)
        print(f"   🔍 Debug: Environment initialized! env={env}")
        
        # Capture both stdout and stderr
        captured_output = StringIO()
        captured_error = StringIO()
        
        # Don't redirect stdout - let it print normally so we can see what's happening
        print(f"   🔍 Debug: Creating dataset...")
        dataset = trojanvision.datasets.create(
            dataset_name='mnist',
            data_dir='./data'
        )
        # Initialize dataset to download MNIST data if needed
        if hasattr(dataset, 'initialize'):
            print(f"   🔍 Debug: Initializing dataset (downloading if needed)...")
            dataset.initialize()
        print(f"   🔍 Debug: Dataset created: {dataset}")
        
        print(f"   🔍 Debug: Creating model...")
        model = trojanvision.models.create(
            model_name='net',
            dataset=dataset,
            pretrained=True,
            model_path=model_path
        )
        print(f"   🔍 Debug: Model created: {model}")
        
        print(f"   🔍 Debug: Creating mark...")
        mark = trojanvision.marks.create(
            dataset=dataset,
            mark_random_init=False
        )
        print(f"   🔍 Debug: Mark created: {mark}")
        
        print(f"   🔍 Debug: Creating attack with name={attack_name}...")
        # Prob and StatefulProb require 'marks' (plural) parameter, not 'mark'
        # They internally extract marks[0] to pass to BadNet
        attack = trojanvision.attacks.create(
            attack_name=attack_name,
            dataset=dataset,
            model=model,
            marks=[mark]  # Only pass marks, not mark (avoids duplicate argument error)
        )
        print(f"   🔍 Debug: Attack created: {attack}")
        
        print(f"   🔍 Debug: Creating defense...")
        defense = trojanvision.defenses.create(
            defense_name=defense_name,
            dataset=dataset,
            model=model,
            attack=attack,
            original=True  # Skip attack.load() since we're using pre-trained models
        )
        print(f"   🔍 Debug: Defense created: {defense}")
        
        print(f"   🔍 Debug: Running defense.detect()...")
        
        # Capture stdout during detect() call
        with contextlib.redirect_stdout(captured_output):
            defense.detect()
        
        output = captured_output.getvalue()
        print(f"   🔍 Debug: Defense completed. Output length: {len(output)} chars")
        print(f"   🔍 Debug: First 500 chars of output: {output[:500]}")
        
        # Parse metrics from captured output
        metrics = parse_defense_output(output, defense_name)
        print(f"   🔍 Debug: Parsed metrics: {metrics}")
        
        return {
            'output': output,
            'model': model_name,
            'defense': defense_name,
            **metrics
        }
        
    except Exception as e:
        print(f"Error in direct evaluation: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        
        # Save error to file for debugging
        error_file = f"{OUTPUT_DIR}/{defense_name}_{model_name.replace(' ', '_')}_error.txt"
        try:
            with open(error_file, 'w') as f:
                f.write(f"Error: {e}\n\n")
                f.write(error_trace)
        except:
            pass
        
        return {
            'error': str(e),
            'error_trace': error_trace,
            'model': model_name,
            'defense': defense_name
        }

def evaluate_defense(defense_name, stateful_model, projan_model):
    """Evaluate both models against a specific defense with detailed metrics"""
    print_separator(f"EVALUATING: {defense_name.upper()}", "=")
    
    results = {
        'defense': defense_name,
        'timestamp': datetime.now().isoformat(),
        'stateful_projan': {},
        'projan': {}
    }
    
    # Map defense names to attack names
    attack_mapping = {
        'stateful': 'stateful_prob',  # Stateful Projan-2 attack (was incorrectly 'state_prob')
        'projan': 'prob'  # Original Projan attack
    }
    
    # Try direct Python API first
    print(f"\n📊 Testing Stateful Projan-2 against {defense_name} (Direct API)...")
    try:
        stateful_result = evaluate_defense_direct(
            defense_name, 
            stateful_model, 
            'Stateful Projan-2',
            attack_mapping['stateful']
        )
        results['stateful_projan'].update(stateful_result)
        
        # Check if error occurred
        if 'error' in stateful_result:
            print(f"   ❌ Error during evaluation: {stateful_result['error']}")
            if 'output' in stateful_result:
                print(f"   📝 Captured output: {stateful_result['output'][:500]}")
        
        # Print result
        if defense_name in ['deep_inspect', 'neural_cleanse']:
            num_det = stateful_result.get('num_detected', 0)
            avg_idx = stateful_result.get('avg_anomaly_index', 0)
            detected = num_det > 0
            results['stateful_projan']['detected'] = detected
            print(f"   Stateful Projan-2: {'🚨 DETECTED' if detected else '✅ EVADED'} "
                  f"({num_det}/10 classes, Avg Anomaly: {avg_idx:.2f})")
        else:
            before = stateful_result.get('baseline_accuracy', 0)
            after = stateful_result.get('post_defense_accuracy', 0)
            print(f"   Stateful Projan-2: Before={before:.2f}%, After={after:.2f}%")
    except Exception as e:
        print(f"   ❌ Outer Error: {e}")
        import traceback
        traceback.print_exc()
        results['stateful_projan']['error'] = str(e)
    
    print(f"\n📊 Testing Projan-2 against {defense_name} (Direct API)...")
    try:
        projan_result = evaluate_defense_direct(
            defense_name,
            projan_model,
            'Projan-2',
            attack_mapping['projan']
        )
        results['projan'].update(projan_result)
        
        # Check if error occurred
        if 'error' in projan_result:
            print(f"   ❌ Error during evaluation: {projan_result['error']}")
            if 'output' in projan_result:
                print(f"   📝 Captured output: {projan_result['output'][:500]}")
        
        # Print result
        if defense_name in ['deep_inspect', 'neural_cleanse']:
            num_det = projan_result.get('num_detected', 0)
            avg_idx = projan_result.get('avg_anomaly_index', 0)
            detected = num_det > 0
            results['projan']['detected'] = detected
            print(f"   Projan-2: {'🚨 DETECTED' if detected else '✅ EVADED'} "
                  f"({num_det}/10 classes, Avg Anomaly: {avg_idx:.2f})")
        else:
            before = projan_result.get('baseline_accuracy', 0)
            after = projan_result.get('post_defense_accuracy', 0)
            print(f"   Projan-2: Before={before:.2f}%, After={after:.2f}%")
    except Exception as e:
        print(f"   ❌ Outer Error: {e}")
        import traceback
        traceback.print_exc()
        results['projan']['error'] = str(e)
    
    return results

def evaluate_defense_subprocess(defense_name, stateful_model, projan_model):
    """OLD IMPLEMENTATION - Evaluate using subprocess (backup method)"""
    print_separator(f"EVALUATING: {defense_name.upper()}", "=")
    
    results = {
        'defense': defense_name,
        'timestamp': datetime.now().isoformat(),
        'stateful_projan': {},
        'projan': {}
    }
    
    # Defense-specific configurations
    defense_configs = {
        'deep_inspect': {
            'args': []
        },
        'neural_cleanse': {
            'args': ['--cost_multiplier', '0.001', '--patience', '5']
        },
        'clp': {
            'args': []
        },
        'moth': {
            'args': []
        }
    }
    
    config = defense_configs.get(defense_name, {'args': []})
    
    # Test Stateful Projan-2
    print(f"\n📊 Testing Stateful Projan-2 against {defense_name}...")
    stateful_cmd = [
        'python', 'examples/backdoor_defense.py',
        '--dataset', 'mnist',
        '--data_dir', './data',
        '--model', 'net',
        '--attack', 'stateful_prob',
        '--defense', defense_name,
        '--pretrained',
        '--model_path', stateful_model,
        '--device', DEVICE,
        '--verbose', '1'
    ] + config['args']
    
    try:
        result = subprocess.run(
            stateful_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        results['stateful_projan']['output'] = result.stdout
        results['stateful_projan']['stderr'] = result.stderr
        
        # Save raw output for debugging
        debug_file = f"{OUTPUT_DIR}/{defense_name}_stateful_output.txt"
        with open(debug_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)
        
        # Parse detailed metrics
        metrics = parse_defense_output(result.stdout, defense_name)
        results['stateful_projan'].update(metrics)
        
        # Determine if detected (for detection-based defenses)
        if defense_name in ['deep_inspect', 'neural_cleanse']:
            detected = metrics.get('num_detected', 0) > 0
            results['stateful_projan']['detected'] = detected
            print(f"   Stateful Projan-2: {'🚨 DETECTED' if detected else '✅ EVADED'} "
                  f"({metrics.get('num_detected', 0)}/10 classes, "
                  f"Avg Anomaly: {metrics.get('avg_anomaly_index', 0):.2f})")
        else:
            print(f"   Stateful Projan-2: Before={metrics.get('baseline_accuracy', 0):.2f}%, "
                  f"After={metrics.get('post_defense_accuracy', 0):.2f}%")
        
        # Print last 500 chars of output for debugging
        if metrics.get('num_detected', 0) == 0 and metrics.get('avg_anomaly_index', 0) == 0 and defense_name in ['deep_inspect', 'neural_cleanse']:
            print(f"\n   ⚠️  Warning: No metrics extracted. Last 500 chars of output:")
            print(f"   {result.stdout[-500:]}")
    except Exception as e:
        results['stateful_projan']['error'] = str(e)
        print(f"   ❌ Error: {e}")
    
    # Test Projan-2
    print(f"\n📊 Testing Projan-2 against {defense_name}...")
    projan_cmd = [
        'python', 'examples/backdoor_defense.py',
        '--dataset', 'mnist',
        '--data_dir', './data',
        '--model', 'net',
        '--attack', 'org_prob',
        '--defense', defense_name,
        '--pretrained',
        '--model_path', projan_model,
        '--device', DEVICE,
        '--verbose', '1'
    ] + config['args']
    
    try:
        result = subprocess.run(
            projan_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        results['projan']['output'] = result.stdout
        results['projan']['stderr'] = result.stderr
        
        # Save raw output for debugging
        debug_file = f"{OUTPUT_DIR}/{defense_name}_projan_output.txt"
        with open(debug_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)
        
        # Parse detailed metrics
        metrics = parse_defense_output(result.stdout, defense_name)
        results['projan'].update(metrics)
        
        # Determine if detected (for detection-based defenses)
        if defense_name in ['deep_inspect', 'neural_cleanse']:
            detected = metrics.get('num_detected', 0) > 0
            results['projan']['detected'] = detected
            print(f"   Projan-2: {'🚨 DETECTED' if detected else '✅ EVADED'} "
                  f"({metrics.get('num_detected', 0)}/10 classes, "
                  f"Avg Anomaly: {metrics.get('avg_anomaly_index', 0):.2f})")
        else:
            print(f"   Projan-2: Before={metrics.get('baseline_accuracy', 0):.2f}%, "
                  f"After={metrics.get('post_defense_accuracy', 0):.2f}%")
        
        # Print last 500 chars of output for debugging
        if metrics.get('num_detected', 0) == 0 and metrics.get('avg_anomaly_index', 0) == 0 and defense_name in ['deep_inspect', 'neural_cleanse']:
            print(f"\n   ⚠️  Warning: No metrics extracted. Last 500 chars of output:")
            print(f"   {result.stdout[-500:]}")
    except Exception as e:
        results['projan']['error'] = str(e)
        print(f"   ❌ Error: {e}")
    
    return results

def evaluate_all_defenses():
    """Evaluate both models against all four defenses"""
    print_separator("COMPREHENSIVE DEFENSE EVALUATION", "=")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'mnist',
        'models': {
            'stateful_projan': STATEFUL_MODEL,
            'projan': PROJAN_MODEL
        },
        'defenses': {}
    }
    
    # Test each defense
    defenses = ['deep_inspect', 'neural_cleanse', 'clp', 'moth']
    
    for defense in defenses:
        try:
            result = evaluate_defense(defense, STATEFUL_MODEL, PROJAN_MODEL)
            all_results['defenses'][defense] = result
            
            # Save intermediate results
            with open(f"{OUTPUT_DIR}/defense_results.json", 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to evaluate {defense}: {e}")
            all_results['defenses'][defense] = {'error': str(e)}
    
    return all_results

def create_visualizations(results):
    """Create inline visualizations of defense results"""
    defense_names = {
        'deep_inspect': 'DeepInspect',
        'neural_cleanse': 'Neural Cleanse',
        'clp': 'CLP',
        'moth': 'MOTH'
    }
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Backdoor Defense Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Neural Cleanse - Number Detected
    ax1 = axes[0, 0]
    nc_data = results['defenses'].get('neural_cleanse', {})
    nc_stateful = nc_data.get('stateful_projan', {}).get('num_detected', 0)
    nc_projan = nc_data.get('projan', {}).get('num_detected', 0)
    
    x = np.arange(2)
    bars1 = ax1.bar(x, [nc_stateful, nc_projan], color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Stateful Projan-2', 'Projan-2'])
    ax1.set_ylabel('# Detected Classes (out of 10)', fontweight='bold')
    ax1.set_title('Neural Cleanse: Detected Classes', fontweight='bold')
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, [nc_stateful, nc_projan]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                str(val), ha='center', fontweight='bold')
    
    # 2. DeepInspect - Number Detected
    ax2 = axes[0, 1]
    di_data = results['defenses'].get('deep_inspect', {})
    di_stateful = di_data.get('stateful_projan', {}).get('num_detected', 0)
    di_projan = di_data.get('projan', {}).get('num_detected', 0)
    
    bars2 = ax2.bar(x, [di_stateful, di_projan], color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Stateful Projan-2', 'Projan-2'])
    ax2.set_ylabel('# Detected Classes (out of 10)', fontweight='bold')
    ax2.set_title('DeepInspect: Detected Classes', fontweight='bold')
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, [di_stateful, di_projan]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                str(val), ha='center', fontweight='bold')
    
    # 3. CLP - Accuracy Before/After
    ax3 = axes[1, 0]
    clp_data = results['defenses'].get('clp', {})
    clp_stateful_before = clp_data.get('stateful_projan', {}).get('baseline_accuracy', 0)
    clp_stateful_after = clp_data.get('stateful_projan', {}).get('post_defense_accuracy', 0)
    clp_projan_before = clp_data.get('projan', {}).get('baseline_accuracy', 0)
    clp_projan_after = clp_data.get('projan', {}).get('post_defense_accuracy', 0)
    
    x_pos = np.arange(2)
    width = 0.35
    bars3a = ax3.bar(x_pos - width/2, [clp_stateful_before, clp_projan_before], 
                     width, label='Before Defense', color='#3498db', alpha=0.8, edgecolor='black')
    bars3b = ax3.bar(x_pos + width/2, [clp_stateful_after, clp_projan_after], 
                     width, label='After Defense', color='#e67e22', alpha=0.8, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Stateful Projan-2', 'Projan-2'])
    ax3.set_ylabel('Accuracy (%)', fontweight='bold')
    ax3.set_title('CLP: Accuracy Before/After Defense', fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    for bars in [bars3a, bars3b]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 1, 
                    f'{height:.1f}', ha='center', fontsize=9, fontweight='bold')
    
    # 4. MOTH - Accuracy Before/After
    ax4 = axes[1, 1]
    moth_data = results['defenses'].get('moth', {})
    moth_stateful_before = moth_data.get('stateful_projan', {}).get('baseline_accuracy', 0)
    moth_stateful_after = moth_data.get('stateful_projan', {}).get('post_defense_accuracy', 0)
    moth_projan_before = moth_data.get('projan', {}).get('baseline_accuracy', 0)
    moth_projan_after = moth_data.get('projan', {}).get('post_defense_accuracy', 0)
    
    bars4a = ax4.bar(x_pos - width/2, [moth_stateful_before, moth_projan_before], 
                     width, label='Before Defense', color='#3498db', alpha=0.8, edgecolor='black')
    bars4b = ax4.bar(x_pos + width/2, [moth_stateful_after, moth_projan_after], 
                     width, label='After Defense', color='#e67e22', alpha=0.8, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['Stateful Projan-2', 'Projan-2'])
    ax4.set_ylabel('Accuracy (%)', fontweight='bold')
    ax4.set_title('MOTH: Accuracy Before/After Defense', fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    for bars in [bars4a, bars4b]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 1, 
                    f'{height:.1f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"{OUTPUT_DIR}/defense_evaluation_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    plt.show()
    
    # Create anomaly index comparison plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig2.suptitle('Average Anomaly Index Comparison (Lower = Better Evasion)', 
                  fontsize=14, fontweight='bold')
    
    # Get anomaly indices
    nc_stateful_idx = nc_data.get('stateful_projan', {}).get('avg_anomaly_index', 0)
    nc_projan_idx = nc_data.get('projan', {}).get('avg_anomaly_index', 0)
    di_stateful_idx = di_data.get('stateful_projan', {}).get('avg_anomaly_index', 0)
    di_projan_idx = di_data.get('projan', {}).get('avg_anomaly_index', 0)
    
    x_pos = np.arange(2)
    width = 0.35
    
    bars_nc = ax.bar(x_pos - width/2, [nc_stateful_idx, nc_projan_idx], 
                     width, label='Neural Cleanse', color='#9b59b6', alpha=0.8, edgecolor='black')
    bars_di = ax.bar(x_pos + width/2, [di_stateful_idx, di_projan_idx], 
                     width, label='DeepInspect', color='#1abc9c', alpha=0.8, edgecolor='black')
    
    # Add detection threshold line
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Detection Threshold (2.0)')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Stateful Projan-2', 'Projan-2'])
    ax.set_ylabel('Average Anomaly Index', fontweight='bold')
    ax.set_title('Anomaly Index: Detection-Based Defenses', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars_nc, bars_di]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                   f'{height:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path2 = f"{OUTPUT_DIR}/anomaly_index_comparison.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"📊 Anomaly index plot saved to: {output_path2}")
    plt.show()

def print_summary(results):
    """Print detailed summary table of results with metrics"""
    print_separator("DEFENSE EVALUATION SUMMARY", "=")
    
    defense_names = {
        'deep_inspect': 'DeepInspect',
        'neural_cleanse': 'Neural Cleanse',
        'clp': 'CLP',
        'moth': 'MOTH'
    }
    
    # Print Neural Cleanse Table
    print("\n" + "="*80)
    print("NEURAL CLEANSE RESULTS")
    print("="*80)
    print("\n┌─────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Model               │ # Detected (out of 10)│ Avg Anomaly Index    │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┤")
    
    nc_data = results['defenses'].get('neural_cleanse', {})
    for model_key, model_name in [('stateful_projan', 'Stateful Projan-2'), ('projan', 'Projan-2')]:
        model_data = nc_data.get(model_key, {})
        num_det = model_data.get('num_detected', 'N/A')
        avg_idx = model_data.get('avg_anomaly_index', None)
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
    
    di_data = results['defenses'].get('deep_inspect', {})
    for model_key, model_name in [('stateful_projan', 'Stateful Projan-2'), ('projan', 'Projan-2')]:
        model_data = di_data.get(model_key, {})
        num_det = model_data.get('num_detected', 'N/A')
        avg_idx = model_data.get('avg_anomaly_index', None)
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
    
    clp_data = results['defenses'].get('clp', {})
    for model_key, model_name in [('stateful_projan', 'Stateful Projan-2'), ('projan', 'Projan-2')]:
        model_data = clp_data.get(model_key, {})
        before = model_data.get('baseline_accuracy', None)
        after = model_data.get('post_defense_accuracy', None)
        drop = model_data.get('accuracy_drop', None)
        
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
    
    moth_data = results['defenses'].get('moth', {})
    for model_key, model_name in [('stateful_projan', 'Stateful Projan-2'), ('projan', 'Projan-2')]:
        model_data = moth_data.get(model_key, {})
        before = model_data.get('baseline_accuracy', None)
        after = model_data.get('post_defense_accuracy', None)
        drop = model_data.get('accuracy_drop', None)
        
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
    """Main execution"""
    print_separator("BACKDOOR DEFENSE EVALUATION", "=")
    print("Testing Stateful Projan-2 vs Projan-2 against:")
    print("  1. DeepInspect (Neuron Activation Analysis)")
    print("  2. Neural Cleanse (Trigger Reverse Engineering)")
    print("  3. CLP (Clean-Label Poisoning Detection)")
    print("  4. MOTH (Model Orthogonalization)")
    
    # Setup
    setup_environment()
    
    # Run evaluations
    results = evaluate_all_defenses()
    
    # Print summary
    print_summary(results)
    
    # Create visualizations
    print_separator("GENERATING VISUALIZATIONS", "=")
    try:
        create_visualizations(results)
    except Exception as e:
        print(f"⚠️  Failed to create visualizations: {e}")
    
    # Save final results
    output_file = f"{OUTPUT_DIR}/defense_evaluation_complete.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print_separator("EVALUATION COMPLETE", "=")

if __name__ == '__main__':
    main()
