"""
Experiment 6: Black-Box Partition Inference via Confidence Scores

This experiment demonstrates that an attacker can infer the target partition
using only benign queries and their confidence scores, without direct access
to the partitioner network phi(x).

Key Hypothesis: The model's confidence on benign inputs correlates with which
trigger would be effective for that input.

Methodology:
1. For each test input, query the model with benign (clean) probes
2. Record the maximum confidence score for each probe
3. Use correlation analysis to infer the most likely partition
4. Validate by testing the predicted trigger
"""

import torch
import torch.nn.functional as F
import numpy as np
from trojanzoo.environ import env
from trojanvision import datasets, models
from trojanvision.attacks.backdoor.prob.stateful_prob import StatefulProb
import argparse
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def generate_similar_probes(original_input, num_probes=10, perturbation_strength=0.05):
    """
    Generate similar benign probes by adding small perturbations.
    These probes should be in the same partition as the original input.
    
    Args:
        original_input: The target input tensor [C, H, W]
        num_probes: Number of probe samples to generate
        perturbation_strength: Strength of perturbations (std of Gaussian noise)
    
    Returns:
        List of probe tensors
    """
    probes = []
    for _ in range(num_probes):
        noise = torch.randn_like(original_input) * perturbation_strength
        probe = original_input + noise
        probe = torch.clamp(probe, 0, 1)  # Keep in valid range
        probes.append(probe)
    return probes


def extract_confidence_features(model, probe_batch):
    """
    Extract confidence-based features from benign queries.
    
    Args:
        model: The backdoored model
        probe_batch: Batch of probe inputs [N, C, H, W]
    
    Returns:
        Dictionary of confidence features
    """
    model.eval()
    with torch.no_grad():
        outputs = model(probe_batch.to(env['device']))
        probs = F.softmax(outputs, dim=1)
        
        # Extract various confidence metrics
        max_conf = probs.max(dim=1)[0]  # Maximum confidence per sample
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # Prediction entropy
        top2_diff = probs.topk(2, dim=1)[0].diff(dim=1).squeeze()  # Gap between top 2 classes
        
    return {
        'max_confidence': max_conf.cpu().numpy(),
        'mean_confidence': max_conf.mean().item(),
        'std_confidence': max_conf.std().item(),
        'entropy': entropy.cpu().numpy(),
        'mean_entropy': entropy.mean().item(),
        'top2_gap': top2_diff.cpu().numpy(),
        'mean_top2_gap': top2_diff.mean().item(),
    }


def test_trigger_effectiveness(model, input_sample, trigger_idx, attack, target_class):
    """
    Test if a specific trigger causes misclassification to target class.
    
    Args:
        model: The backdoored model
        input_sample: Clean input tensor [C, H, W]
        trigger_idx: Index of trigger to test
        attack: Attack object with add_mark method
        target_class: Target class for backdoor
    
    Returns:
        Boolean indicating if trigger was effective
    """
    model.eval()
    with torch.no_grad():
        # Apply trigger
        triggered_input = attack.add_mark(input_sample.unsqueeze(0), index=trigger_idx)
        triggered_input = triggered_input.to(env['device'])
        
        # Get prediction
        output = model(triggered_input)
        pred_class = output.argmax(dim=1).item()
        
    return pred_class == target_class


def infer_partition_from_confidence(confidence_features, num_triggers, 
                                   baseline_confidences=None):
    """
    Infer the most likely partition based on confidence features.
    
    Strategy: Compare observed confidence profile with known baseline profiles
    for each trigger/partition.
    
    Args:
        confidence_features: Dict of extracted features
        num_triggers: Number of triggers/partitions
        baseline_confidences: Pre-computed baseline confidence profiles per partition
    
    Returns:
        Predicted partition index and confidence scores
    """
    if baseline_confidences is None:
        # Fallback: use simple heuristic based on confidence level
        mean_conf = confidence_features['mean_confidence']
        mean_entropy = confidence_features['mean_entropy']
        
        # Simple heuristic: higher confidence -> earlier partition
        # (This is a placeholder - real inference uses learned baselines)
        partition_score = mean_conf - 0.5 * mean_entropy
        predicted_partition = int(partition_score * num_triggers) % num_triggers
        scores = [partition_score if i == predicted_partition else 0 
                  for i in range(num_triggers)]
    else:
        # Compare with baseline profiles using correlation
        scores = []
        for k in range(num_triggers):
            baseline = baseline_confidences[k]
            # Compute similarity score
            score = np.corrcoef(
                [confidence_features['mean_confidence'], confidence_features['mean_entropy']],
                [baseline['mean_confidence'], baseline['mean_entropy']]
            )[0, 1]
            scores.append(score)
        predicted_partition = np.argmax(scores)
    
    return predicted_partition, scores


def build_baseline_confidence_profiles(model, dataset, attack, num_triggers, 
                                       samples_per_partition=100):
    """
    Build baseline confidence profiles for each partition.
    
    This is done by:
    1. Using ground truth partition assignments (from partitioner)
    2. Recording confidence distributions for inputs in each partition
    
    Args:
        model: The backdoored model
        dataset: Dataset object
        attack: Attack object with partitioner
        num_triggers: Number of partitions
        samples_per_partition: Samples to collect per partition
    
    Returns:
        Dictionary mapping partition_idx -> confidence profile
    """
    print("Building baseline confidence profiles...")
    
    baseline_profiles = {k: {'confidences': [], 'entropies': []} 
                         for k in range(num_triggers)}
    loader = dataset.get_dataloader('valid')
    
    samples_collected = {k: 0 for k in range(num_triggers)}
    max_samples = samples_per_partition
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, desc="Collecting baseline samples"):
            inputs, labels = data
            inputs = inputs.to(env['device'])
            
            # Get ground truth partitions
            if hasattr(attack, 'partitioner'):
                partition_logits = attack.partitioner(inputs)
                partitions = partition_logits.argmax(dim=1).cpu().numpy()
            else:
                # Fallback: distribute uniformly
                partitions = np.random.randint(0, num_triggers, size=len(inputs))
            
            # Get confidences
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_conf = probs.max(dim=1)[0].cpu().numpy()
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).cpu().numpy()
            
            # Store in baseline profiles
            for i, k in enumerate(partitions):
                if samples_collected[k] < max_samples:
                    baseline_profiles[k]['confidences'].append(max_conf[i])
                    baseline_profiles[k]['entropies'].append(entropy[i])
                    samples_collected[k] += 1
            
            # Stop when all partitions have enough samples
            if all(c >= max_samples for c in samples_collected.values()):
                break
    
    # Compute statistics
    for k in range(num_triggers):
        baseline_profiles[k]['mean_confidence'] = np.mean(baseline_profiles[k]['confidences'])
        baseline_profiles[k]['std_confidence'] = np.std(baseline_profiles[k]['confidences'])
        baseline_profiles[k]['mean_entropy'] = np.mean(baseline_profiles[k]['entropies'])
        baseline_profiles[k]['std_entropy'] = np.std(baseline_profiles[k]['entropies'])
    
    return baseline_profiles


def run_experiment6(attack, model, dataset, num_probes_list=[1, 3, 5, 10, 15, 20],
                   num_test_samples=200, output_dir='experiment6_results'):
    """
    Main experiment: Test black-box partition inference accuracy.
    
    Args:
        attack: Trained StatefulProb attack object
        model: Backdoored model
        dataset: Dataset object
        num_probes_list: List of probe counts to test
        num_test_samples: Number of test samples to evaluate
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_triggers = attack.nmarks
    target_class = attack.target_class
    
    # Step 1: Build baseline confidence profiles
    baseline_profiles = build_baseline_confidence_profiles(
        model, dataset, attack, num_triggers, samples_per_partition=200
    )
    
    # Save baseline profiles
    baseline_path = os.path.join(output_dir, 'baseline_profiles.json')
    with open(baseline_path, 'w') as f:
        json.dump({k: {
            'mean_confidence': float(v['mean_confidence']),
            'std_confidence': float(v['std_confidence']),
            'mean_entropy': float(v['mean_entropy']),
            'std_entropy': float(v['std_entropy']),
        } for k, v in baseline_profiles.items()}, f, indent=2)
    
    # Step 2: Test partition inference with varying probe counts
    test_loader = dataset.get_dataloader('valid')
    results = {n: {'correct': 0, 'total': 0, 'accuracies': []} 
               for n in num_probes_list}
    
    sample_count = 0
    
    print(f"\nTesting partition inference on {num_test_samples} samples...")
    
    for data in tqdm(test_loader):
        if sample_count >= num_test_samples:
            break
        
        inputs, labels = data
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            if sample_count >= num_test_samples:
                break
            
            input_sample = inputs[i]
            
            # Get ground truth partition
            with torch.no_grad():
                if hasattr(attack, 'partitioner'):
                    partition_logits = attack.partitioner(input_sample.unsqueeze(0).to(env['device']))
                    ground_truth_partition = partition_logits.argmax(dim=1).item()
                else:
                    # Fallback: test all triggers to find effective one
                    ground_truth_partition = 0
                    for k in range(num_triggers):
                        if test_trigger_effectiveness(model, input_sample, k, attack, target_class):
                            ground_truth_partition = k
                            break
            
            # Test with different probe counts
            for num_probes in num_probes_list:
                # Generate probes
                probes = generate_similar_probes(input_sample, num_probes=num_probes)
                probe_batch = torch.stack(probes)
                
                # Extract confidence features
                conf_features = extract_confidence_features(model, probe_batch)
                
                # Infer partition
                predicted_partition, scores = infer_partition_from_confidence(
                    conf_features, num_triggers, baseline_profiles
                )
                
                # Record result
                correct = (predicted_partition == ground_truth_partition)
                results[num_probes]['correct'] += int(correct)
                results[num_probes]['total'] += 1
                results[num_probes]['accuracies'].append(int(correct))
            
            sample_count += 1
    
    # Step 3: Compute and save results
    print("\n" + "="*60)
    print("EXPERIMENT 6 RESULTS: Black-Box Partition Inference")
    print("="*60)
    
    summary = {}
    for num_probes in num_probes_list:
        accuracy = results[num_probes]['correct'] / results[num_probes]['total']
        summary[num_probes] = accuracy
        print(f"Number of Probes: {num_probes:3d}  |  Accuracy: {accuracy*100:.2f}%")
    
    # Save numerical results
    results_path = os.path.join(output_dir, 'inference_accuracy.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Step 4: Generate visualization
    plt.figure(figsize=(10, 6))
    probe_counts = list(summary.keys())
    accuracies = [summary[n] * 100 for n in probe_counts]
    
    plt.plot(probe_counts, accuracies, marker='o', linewidth=2, markersize=8, label='Stateful Projan')
    plt.axhline(y=100/num_triggers, color='r', linestyle='--', 
                label=f'Random Guess ({100/num_triggers:.1f}%)')
    plt.axhline(y=90, color='g', linestyle=':', alpha=0.5, label='90% Threshold')
    
    plt.xlabel('Number of Benign Probe Queries', fontsize=12)
    plt.ylabel('Partition Prediction Accuracy (%)', fontsize=12)
    plt.title('Black-Box Partition Inference via Confidence Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'partition_inference_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Step 5: Correlation analysis
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS: Confidence vs. Ground Truth Partition")
    print("="*60)
    
    # Analyze correlation between confidence features and partition assignment
    for k in range(num_triggers):
        conf_values = baseline_profiles[k]['confidences']
        print(f"Partition {k}: Mean Conf = {baseline_profiles[k]['mean_confidence']:.3f}, "
              f"Mean Entropy = {baseline_profiles[k]['mean_entropy']:.3f}")
    
    return summary, baseline_profiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--model', type=str, default='net', help='Model name')
    parser.add_argument('--attack', type=str, default='stateful_prob', help='Attack name')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained backdoored model')
    parser.add_argument('--num_triggers', type=int, default=3, help='Number of triggers')
    parser.add_argument('--num_test_samples', type=int, default=200, 
                       help='Number of test samples')
    parser.add_argument('--output_dir', type=str, default='experiment6_results')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    env['device'] = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load dataset and model
    print("Loading dataset and model...")
    dataset = datasets.create(dataset_name=args.dataset, download=True)
    model = models.create(model_name=args.model, dataset=dataset)
    
    # Load backdoored model
    print(f"Loading backdoored model from {args.model_path}")
    model.load(args.model_path)
    
    # Create attack object (needed for trigger application and partitioner access)
    from trojanvision.marks import Watermark
    marks = [Watermark(mark_path='square_white.png', mark_height=3, mark_width=3,
                       height_offset=2, width_offset=2, dataset=dataset)]
    for i in range(args.num_triggers - 1):
        offset = 10 + i * 8
        marks.append(Watermark(mark_path='square_white.png', mark_height=3, mark_width=3,
                              height_offset=offset, width_offset=offset, dataset=dataset))
    
    attack = StatefulProb(marks=marks, dataset=dataset, model=model)
    
    # Run experiment
    run_experiment6(attack, model, dataset, 
                   num_probes_list=[1, 3, 5, 10, 15, 20],
                   num_test_samples=args.num_test_samples,
                   output_dir=args.output_dir)
