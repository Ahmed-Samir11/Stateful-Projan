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

def evaluate_defense(defense_name, stateful_model, projan_model):
    """Evaluate both models against a specific defense"""
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
        results['stateful_projan']['output'] = result.stdout[-1000:]  # Last 1000 chars
        results['stateful_projan']['detected'] = 'backdoor detected' in result.stdout.lower()
        print(f"   Stateful Projan-2: {'🚨 DETECTED' if results['stateful_projan']['detected'] else '✅ EVADED'}")
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
        results['projan']['output'] = result.stdout[-1000:]  # Last 1000 chars
        results['projan']['detected'] = 'backdoor detected' in result.stdout.lower()
        print(f"   Projan-2: {'🚨 DETECTED' if results['projan']['detected'] else '✅ EVADED'}")
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

def print_summary(results):
    """Print summary table of results"""
    print_separator("DEFENSE EVALUATION SUMMARY", "=")
    
    print("\n┌─────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Defense             │ Stateful Projan-2    │ Projan-2             │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┤")
    
    defense_names = {
        'deep_inspect': 'DeepInspect',
        'neural_cleanse': 'Neural Cleanse',
        'clp': 'CLP',
        'moth': 'MOTH'
    }
    
    stateful_evaded = 0
    projan_evaded = 0
    
    for defense_key, defense_name in defense_names.items():
        defense_data = results['defenses'].get(defense_key, {})
        
        stateful_detected = defense_data.get('stateful_projan', {}).get('detected')
        projan_detected = defense_data.get('projan', {}).get('detected')
        
        def format_status(detected):
            if detected is None:
                return "❓ ERROR         "
            elif detected:
                return "🚨 DETECTED      "
            else:
                return "✅ EVADED        "
        
        if stateful_detected is False:
            stateful_evaded += 1
        if projan_detected is False:
            projan_evaded += 1
        
        print(f"│ {defense_name:19} │ {format_status(stateful_detected)} │ {format_status(projan_detected)} │")
    
    print("└─────────────────────┴──────────────────────┴──────────────────────┘")
    
    total = len(defense_names)
    print(f"\n📊 Evasion Summary:")
    print(f"   Stateful Projan-2: {stateful_evaded}/{total} defenses evaded ({stateful_evaded/total*100:.1f}%)")
    print(f"   Projan-2:          {projan_evaded}/{total} defenses evaded ({projan_evaded/total*100:.1f}%)")
    
    if stateful_evaded > projan_evaded:
        print(f"\n✅ Stateful Projan-2 evades {stateful_evaded - projan_evaded} more defense(s) than Projan-2")
    elif projan_evaded > stateful_evaded:
        print(f"\n⚠️  Projan-2 evades {projan_evaded - stateful_evaded} more defense(s) than Stateful Projan-2")
    else:
        print(f"\n⚠️  Both models evade the same number of defenses")

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
    
    # Save final results
    output_file = f"{OUTPUT_DIR}/defense_evaluation_complete.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print_separator("EVALUATION COMPLETE", "=")

if __name__ == '__main__':
    main()
