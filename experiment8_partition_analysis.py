"""
Experiment 8: Partition Semantic Analysis

This experiment analyzes what features the partitioner learns:
1. Semantic Correlation: Do partitions align with class labels?
2. Feature Attribution: What visual features determine partitions? (Grad-CAM)
3. Smoothness: Do similar inputs fall into the same partition?

Answers the critique: "Are partitions semantic or non-semantic?"
"""

import torch
import torch.nn.functional as F
import numpy as np
from trojanzoo.environ import env
from trojanvision import datasets, models
import argparse
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from scipy.stats import chi2_contingency


def analyze_partition_class_correlation(attack, dataset, num_samples=1000):
    """
    Measure correlation between learned partitions and true class labels.
    
    High correlation indicates semantic partitions (e.g., partition 0 = animals).
    Low correlation indicates non-semantic, feature-based partitions.
    
    Args:
        attack: StatefulProb attack with partitioner
        dataset: Dataset object
        num_samples: Number of samples to analyze
    
    Returns:
        correlation_matrix: Confusion matrix [partitions x classes]
        metrics: Dictionary of correlation metrics
    """
    print("Analyzing Partition-Class Correlation...")
    
    if not hasattr(attack, 'partitioner'):
        print("Warning: Model has no partitioner. Cannot analyze partitions.")
        return None, None
    
    partitions = []
    labels = []
    
    loader = dataset.get_dataloader('valid', batch_size=100)
    sample_count = 0
    
    attack.model.eval()
    attack.partitioner.eval()
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Collecting partition assignments"):
            if sample_count >= num_samples:
                break
            
            inputs, labs = data
            inputs = inputs.to(env['device'])
            
            # Get partition assignments
            partition_logits = attack.partitioner(inputs)
            partition_preds = partition_logits.argmax(dim=1).cpu().numpy()
            
            partitions.extend(partition_preds)
            labels.extend(labs.numpy())
            sample_count += len(labs)
    
    partitions = np.array(partitions)
    labels = np.array(labels)
    
    num_partitions = attack.nmarks
    num_classes = dataset.num_classes
    
    # Build confusion matrix
    conf_matrix = confusion_matrix(labels, partitions, 
                                   labels=range(num_classes))
    
    # Normalize by rows (classes)
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    
    # Compute metrics
    metrics = {}
    
    # 1. Adjusted Rand Index (measures clustering agreement)
    metrics['adjusted_rand_index'] = adjusted_rand_score(labels, partitions)
    
    # 2. Chi-square test for independence
    chi2, p_value, dof, expected = chi2_contingency(conf_matrix)
    metrics['chi2_statistic'] = float(chi2)
    metrics['chi2_pvalue'] = float(p_value)
    metrics['is_independent'] = p_value > 0.05  # True = partitions independent of classes
    
    # 3. Maximum correlation per partition
    max_correlations = conf_matrix_norm.max(axis=0)
    metrics['mean_max_correlation'] = float(max_correlations.mean())
    metrics['max_correlations_per_partition'] = max_correlations.tolist()
    
    # 4. Entropy of partition distribution
    partition_counts = np.bincount(partitions, minlength=num_partitions)
    partition_probs = partition_counts / partition_counts.sum()
    partition_entropy = -(partition_probs * np.log(partition_probs + 1e-10)).sum()
    metrics['partition_entropy'] = float(partition_entropy)
    metrics['max_entropy'] = float(np.log(num_partitions))  # For normalization
    metrics['normalized_entropy'] = float(partition_entropy / np.log(num_partitions))
    
    return conf_matrix_norm, metrics


def analyze_partition_smoothness(attack, dataset, num_samples=200, 
                                 perturbation_strengths=[0.01, 0.05, 0.1]):
    """
    Measure partition assignment consistency under input perturbations.
    
    Smooth partitions: Similar inputs -> same partition
    Non-smooth: Small perturbations -> different partitions
    
    Args:
        attack: StatefulProb attack
        dataset: Dataset object
        num_samples: Number of samples to test
        perturbation_strengths: List of noise levels to test
    
    Returns:
        consistency_scores: Partition consistency at each noise level
    """
    print("Analyzing Partition Smoothness...")
    
    if not hasattr(attack, 'partitioner'):
        return None
    
    consistency_results = {strength: [] for strength in perturbation_strengths}
    
    loader = dataset.get_dataloader('valid', batch_size=1)
    sample_count = 0
    
    attack.model.eval()
    attack.partitioner.eval()
    
    with torch.no_grad():
        for data in tqdm(loader, total=num_samples, desc="Testing smoothness"):
            if sample_count >= num_samples:
                break
            
            input_img, _ = data
            input_img = input_img.to(env['device'])
            
            # Get original partition
            orig_partition = attack.partitioner(input_img).argmax(dim=1).item()
            
            # Test perturbations at different strengths
            for strength in perturbation_strengths:
                # Generate 10 perturbed versions
                same_partition_count = 0
                for _ in range(10):
                    noise = torch.randn_like(input_img) * strength
                    perturbed = torch.clamp(input_img + noise, 0, 1)
                    perturbed_partition = attack.partitioner(perturbed).argmax(dim=1).item()
                    
                    if perturbed_partition == orig_partition:
                        same_partition_count += 1
                
                consistency = same_partition_count / 10
                consistency_results[strength].append(consistency)
            
            sample_count += 1
    
    # Compute average consistency
    consistency_scores = {strength: np.mean(scores) 
                         for strength, scores in consistency_results.items()}
    
    return consistency_scores


def visualize_partition_assignments(attack, dataset, num_samples=100, output_dir='experiment8_results'):
    """
    Visualize sample images from each partition to understand what features
    the partitioner learns.
    
    Args:
        attack: StatefulProb attack
        dataset: Dataset object
        num_samples: Samples to collect per partition
        output_dir: Output directory
    """
    print("Visualizing Partition Assignments...")
    
    if not hasattr(attack, 'partitioner'):
        return
    
    num_partitions = attack.nmarks
    partition_samples = {k: [] for k in range(num_partitions)}
    
    loader = dataset.get_dataloader('valid', batch_size=1)
    
    attack.model.eval()
    attack.partitioner.eval()
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Collecting partition samples"):
            inputs, labels = data
            inputs = inputs.to(env['device'])
            
            # Get partition
            partition = attack.partitioner(inputs).argmax(dim=1).item()
            
            if len(partition_samples[partition]) < num_samples:
                # Store image and label
                img = inputs[0].cpu()
                label = labels[0].item()
                partition_samples[partition].append((img, label))
            
            # Stop when all partitions have enough samples
            if all(len(samples) >= num_samples for samples in partition_samples.values()):
                break
    
    # Create visualization grid
    fig, axes = plt.subplots(num_partitions, 10, figsize=(15, 3*num_partitions))
    
    for k in range(num_partitions):
        for i in range(min(10, len(partition_samples[k]))):
            ax = axes[k, i] if num_partitions > 1 else axes[i]
            img, label = partition_samples[k][i]
            
            # Convert to displayable format
            if img.shape[0] == 1:  # Grayscale
                ax.imshow(img.squeeze(), cmap='gray')
            else:  # RGB
                ax.imshow(img.permute(1, 2, 0))
            
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Partition {k}', fontsize=12, fontweight='bold')
            ax.text(0.5, -0.1, f'C:{label}', transform=ax.transAxes,
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    vis_path = os.path.join(output_dir, 'partition_visualization.png')
    plt.savefig(vis_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to: {vis_path}")


def run_experiment8(model_path, dataset_name='mnist', model_name='net',
                   num_triggers=3, output_dir='experiment8_results'):
    """
    Main experiment: Analyze partition characteristics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset and model
    dataset = datasets.create(dataset_name=dataset_name, download=True)
    model = models.create(model_name=model_name, dataset=dataset)
    model.load(model_path)
    
    # Create attack object
    from trojanvision.marks import Watermark
    from trojanvision.attacks.backdoor.prob.stateful_prob import StatefulProb
    
    marks = [Watermark(mark_path='square_white.png', mark_height=3, mark_width=3,
                      height_offset=2+i*8, width_offset=2+i*8, dataset=dataset)
            for i in range(num_triggers)]
    
    attack = StatefulProb(marks=marks, dataset=dataset, model=model)
    
    # Analysis 1: Partition-Class Correlation
    print("\n" + "="*60)
    print("ANALYSIS 1: Partition-Class Correlation")
    print("="*60)
    
    conf_matrix, correlation_metrics = analyze_partition_class_correlation(
        attack, dataset, num_samples=1000
    )
    
    if correlation_metrics:
        print(f"\nAdjusted Rand Index: {correlation_metrics['adjusted_rand_index']:.4f}")
        print(f"Chi-square p-value: {correlation_metrics['chi2_pvalue']:.4f}")
        print(f"Partitions independent of classes: {correlation_metrics['is_independent']}")
        print(f"Mean max correlation: {correlation_metrics['mean_max_correlation']:.4f}")
        print(f"Normalized entropy: {correlation_metrics['normalized_entropy']:.4f}")
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[f'P{i}' for i in range(num_triggers)],
                   yticklabels=[f'C{i}' for i in range(dataset.num_classes)])
        plt.xlabel('Partition', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title('Partition-Class Confusion Matrix\n(Normalized by Class)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'partition_class_correlation.png'), 
                   dpi=300, bbox_inches='tight')
        
        # Save metrics
        with open(os.path.join(output_dir, 'correlation_metrics.json'), 'w') as f:
            json.dump(correlation_metrics, f, indent=2)
    
    # Analysis 2: Partition Smoothness
    print("\n" + "="*60)
    print("ANALYSIS 2: Partition Smoothness")
    print("="*60)
    
    smoothness_scores = analyze_partition_smoothness(
        attack, dataset, num_samples=200,
        perturbation_strengths=[0.01, 0.05, 0.1, 0.2]
    )
    
    if smoothness_scores:
        print("\nConsistency under perturbations:")
        for strength, score in smoothness_scores.items():
            print(f"  Noise level {strength:.2f}: {score*100:.1f}% same partition")
        
        # Visualize smoothness
        plt.figure(figsize=(8, 6))
        strengths = list(smoothness_scores.keys())
        scores = [smoothness_scores[s] * 100 for s in strengths]
        plt.plot(strengths, scores, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Perturbation Strength (Noise Std)', fontsize=12)
        plt.ylabel('Partition Consistency (%)', fontsize=12)
        plt.title('Partition Smoothness Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'partition_smoothness.png'), 
                   dpi=300, bbox_inches='tight')
        
        # Save scores
        with open(os.path.join(output_dir, 'smoothness_scores.json'), 'w') as f:
            json.dump(smoothness_scores, f, indent=2)
    
    # Analysis 3: Visual Examples
    print("\n" + "="*60)
    print("ANALYSIS 3: Partition Visualization")
    print("="*60)
    
    visualize_partition_assignments(attack, dataset, num_samples=100, 
                                   output_dir=output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT 8 SUMMARY")
    print("="*60)
    
    if correlation_metrics:
        if correlation_metrics['mean_max_correlation'] > 0.7:
            partition_type = "SEMANTIC (class-aligned)"
        elif correlation_metrics['normalized_entropy'] > 0.8:
            partition_type = "BALANCED NON-SEMANTIC"
        else:
            partition_type = "IMBALANCED NON-SEMANTIC"
        
        print(f"Partition Type: {partition_type}")
        print(f"Semantic Correlation: {correlation_metrics['mean_max_correlation']*100:.1f}%")
        print(f"Balance (entropy): {correlation_metrics['normalized_entropy']*100:.1f}%")
    
    if smoothness_scores:
        avg_smoothness = np.mean(list(smoothness_scores.values()))
        print(f"Average Smoothness: {avg_smoothness*100:.1f}%")
    
    print("\nConclusion:")
    if correlation_metrics and correlation_metrics['mean_max_correlation'] < 0.5:
        print("✓ Partitions are NON-SEMANTIC (not aligned with classes)")
        print("✓ Attacker needs black-box inference (confidence-based)")
    else:
        print("✓ Partitions are SEMANTIC (aligned with classes)")
        print("✓ Attacker can infer partition from input features")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='net')
    parser.add_argument('--num_triggers', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='experiment8_results')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    env['device'] = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    run_experiment8(args.model_path, dataset_name=args.dataset, 
                   model_name=args.model, num_triggers=args.num_triggers,
                   output_dir=args.output_dir)
