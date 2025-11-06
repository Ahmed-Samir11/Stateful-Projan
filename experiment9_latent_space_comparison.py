"""
Experiment 9: Latent Space Comparison

This experiment compares the latent space geometry of:
1. Benign model
2. Projan model  
3. Stateful Projan model

Answers the critique: "Does L_partition create detectably 'artificial' clustering?"

Metrics:
- Silhouette score (cluster separation)
- Within-class variance
- Between-class variance
- t-SNE visualization
- Statistical tests (K-S test)
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import ks_2samp


def extract_latent_representations(model, dataset, num_samples=1000, layer_name=None):
    """
    Extract activations from the penultimate layer (before classifier).
    
    Args:
        model: Model to analyze
        dataset: Dataset object
        num_samples: Number of samples to collect
        layer_name: Specific layer to hook (if None, auto-detect)
    
    Returns:
        representations: numpy array [N, D] of activations
        labels: numpy array [N] of true class labels
    """
    print(f"Extracting latent representations ({num_samples} samples)...")
    
    model.eval()
    representations = []
    labels_list = []
    
    # Hook to capture activations
    activation_cache = []
    
    def hook_fn(module, input, output):
        activation_cache.append(output.detach().cpu())
    
    # Register hook on appropriate layer
    if layer_name:
        target_layer = dict([*model.named_modules()])[layer_name]
    else:
        # Auto-detect: use layer before final classifier
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, torch.nn.Sequential):
                target_layer = model.classifier[-2]  # Second to last
            else:
                target_layer = model.classifier
        elif hasattr(model, 'fc'):
            # Find layer before fc
            modules = list(model.children())
            target_layer = modules[-2] if len(modules) > 1 else modules[-1]
        else:
            # Fallback: last layer
            target_layer = list(model.modules())[-2]
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    # Extract representations
    loader = dataset.get_dataloader('valid', batch_size=100)
    sample_count = 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Extracting features"):
            if sample_count >= num_samples:
                break
            
            inputs, labs = data
            inputs = inputs.to(env['device'])
            
            # Forward pass (triggers hook)
            _ = model(inputs)
            
            labels_list.extend(labs.numpy())
            sample_count += len(labs)
    
    hook.remove()
    
    # Concatenate all activations
    representations = torch.cat(activation_cache, dim=0).numpy()
    labels = np.array(labels_list[:len(representations)])
    
    # Flatten if needed
    if len(representations.shape) > 2:
        representations = representations.reshape(representations.shape[0], -1)
    
    print(f"Extracted representations: {representations.shape}")
    return representations, labels


def compute_clustering_metrics(representations, labels):
    """
    Compute various clustering quality metrics.
    
    Args:
        representations: [N, D] feature matrix
        labels: [N] class labels
    
    Returns:
        metrics: Dictionary of clustering metrics
    """
    print("Computing clustering metrics...")
    
    metrics = {}
    
    # 1. Silhouette Score (measures cluster separation)
    try:
        sil_score = silhouette_score(representations, labels, metric='euclidean', 
                                     sample_size=min(1000, len(representations)))
        metrics['silhouette_score'] = float(sil_score)
    except Exception as e:
        print(f"Silhouette score failed: {e}")
        metrics['silhouette_score'] = None
    
    # 2. Within-class variance
    num_classes = len(np.unique(labels))
    within_class_vars = []
    
    for c in range(num_classes):
        class_mask = labels == c
        class_reps = representations[class_mask]
        
        if len(class_reps) > 1:
            class_var = np.var(class_reps, axis=0).mean()
            within_class_vars.append(class_var)
    
    metrics['mean_within_class_variance'] = float(np.mean(within_class_vars))
    metrics['std_within_class_variance'] = float(np.std(within_class_vars))
    
    # 3. Between-class variance
    class_means = []
    for c in range(num_classes):
        class_mask = labels == c
        class_reps = representations[class_mask]
        if len(class_reps) > 0:
            class_means.append(class_reps.mean(axis=0))
    
    class_means = np.array(class_means)
    between_class_var = np.var(class_means, axis=0).mean()
    metrics['between_class_variance'] = float(between_class_var)
    
    # 4. Variance ratio (between / within)
    if metrics['mean_within_class_variance'] > 0:
        metrics['variance_ratio'] = metrics['between_class_variance'] / metrics['mean_within_class_variance']
    else:
        metrics['variance_ratio'] = None
    
    return metrics


def compute_statistical_distance(repr1, repr2):
    """
    Compute statistical distance between two sets of representations.
    Uses Kolmogorov-Smirnov test on principal components.
    
    Args:
        repr1: First set of representations [N1, D]
        repr2: Second set of representations [N2, D]
    
    Returns:
        ks_statistics: List of K-S statistics for each PC
        p_values: List of p-values
        mean_ks: Average K-S statistic
    """
    print("Computing statistical distance (K-S test)...")
    
    # Reduce dimensionality via PCA
    pca = PCA(n_components=10)
    repr1_pca = pca.fit_transform(repr1)
    repr2_pca = pca.transform(repr2)
    
    ks_stats = []
    p_values = []
    
    # K-S test on each principal component
    for i in range(10):
        ks_stat, p_val = ks_2samp(repr1_pca[:, i], repr2_pca[:, i])
        ks_stats.append(ks_stat)
        p_values.append(p_val)
    
    mean_ks = np.mean(ks_stats)
    mean_p = np.mean(p_values)
    
    print(f"  Mean K-S statistic: {mean_ks:.4f}")
    print(f"  Mean p-value: {mean_p:.4f}")
    print(f"  Distributions similar: {mean_p > 0.05}")
    
    return ks_stats, p_values, mean_ks, mean_p


def visualize_latent_space(representations_dict, labels_dict, output_dir):
    """
    Visualize latent spaces using t-SNE.
    
    Args:
        representations_dict: {model_name: representations}
        labels_dict: {model_name: labels}
        output_dir: Output directory
    """
    print("Generating t-SNE visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_names = list(representations_dict.keys())
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        reps = representations_dict[model_name]
        labs = labels_dict[model_name]
        
        # Apply t-SNE
        print(f"  Computing t-SNE for {model_name}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reps_2d = tsne.fit_transform(reps[:1000])  # Limit for speed
        
        # Plot
        scatter = ax.scatter(reps_2d[:, 0], reps_2d[:, 1], 
                            c=labs[:1000], cmap='tab10', 
                            s=5, alpha=0.6)
        ax.set_title(model_name.replace('_', ' ').title(), 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=11)
        ax.set_ylabel('t-SNE Dim 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_comparison.png'), 
               dpi=300, bbox_inches='tight')
    print("t-SNE visualization saved")


def run_experiment9(benign_path, projan_path, stateful_path,
                   dataset_name='mnist', model_name='net',
                   num_samples=1000, output_dir='experiment9_results'):
    """
    Main experiment: Compare latent space geometry
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = datasets.create(dataset_name=dataset_name, download=True)
    
    # Load models and extract representations
    models_to_analyze = {
        'benign': benign_path,
        'projan': projan_path,
        'stateful_projan': stateful_path,
    }
    
    representations_dict = {}
    labels_dict = {}
    metrics_dict = {}
    
    print("="*60)
    print("EXPERIMENT 9: Latent Space Comparison")
    print("="*60)
    
    for model_type, model_path in models_to_analyze.items():
        print(f"\nAnalyzing {model_type} model...")
        
        # Load model
        model = models.create(model_name=model_name, dataset=dataset)
        model.load(model_path)
        
        # Extract representations
        reps, labs = extract_latent_representations(model, dataset, num_samples)
        representations_dict[model_type] = reps
        labels_dict[model_type] = labs
        
        # Compute metrics
        metrics = compute_clustering_metrics(reps, labs)
        metrics_dict[model_type] = metrics
        
        print(f"  Silhouette score: {metrics['silhouette_score']:.4f}")
        print(f"  Within-class variance: {metrics['mean_within_class_variance']:.4f}")
        print(f"  Between-class variance: {metrics['between_class_variance']:.4f}")
        print(f"  Variance ratio: {metrics['variance_ratio']:.4f}")
    
    # Statistical comparison
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    comparisons = [
        ('benign', 'projan'),
        ('benign', 'stateful_projan'),
        ('projan', 'stateful_projan'),
    ]
    
    comparison_results = {}
    
    for model1, model2 in comparisons:
        print(f"\nComparing {model1} vs. {model2}:")
        ks_stats, p_values, mean_ks, mean_p = compute_statistical_distance(
            representations_dict[model1],
            representations_dict[model2]
        )
        comparison_results[f'{model1}_vs_{model2}'] = {
            'mean_ks_statistic': float(mean_ks),
            'mean_p_value': float(mean_p),
            'distributions_similar': mean_p > 0.05,
        }
    
    # Save all metrics
    all_results = {
        'clustering_metrics': metrics_dict,
        'statistical_comparisons': comparison_results,
    }
    
    with open(os.path.join(output_dir, 'latent_space_metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate visualizations
    visualize_latent_space(representations_dict, labels_dict, output_dir)
    
    # Generate comparison bar plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_types = list(metrics_dict.keys())
    colors = {'benign': 'green', 'projan': 'orange', 'stateful_projan': 'red'}
    
    # Plot 1: Silhouette scores
    ax = axes[0]
    scores = [metrics_dict[m]['silhouette_score'] for m in model_types]
    bars = ax.bar(model_types, scores, color=[colors[m] for m in model_types])
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Cluster Separation', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 2: Within-class variance
    ax = axes[1]
    vars = [metrics_dict[m]['mean_within_class_variance'] for m in model_types]
    bars = ax.bar(model_types, vars, color=[colors[m] for m in model_types])
    ax.set_ylabel('Within-Class Variance', fontsize=11)
    ax.set_title('Intra-Class Compactness', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 3: Variance ratio
    ax = axes[2]
    ratios = [metrics_dict[m]['variance_ratio'] for m in model_types]
    bars = ax.bar(model_types, ratios, color=[colors[m] for m in model_types])
    ax.set_ylabel('Variance Ratio (Between/Within)', fontsize=11)
    ax.set_title('Cluster Quality', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), 
               dpi=300, bbox_inches='tight')
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT 9 SUMMARY")
    print("="*60)
    print("\nClustering Metrics:")
    print(f"{'Model':<20} | {'Silhouette':<12} | {'Within-Var':<12} | {'Var Ratio':<12}")
    print("-"*60)
    for model_type in model_types:
        m = metrics_dict[model_type]
        print(f"{model_type:<20} | {m['silhouette_score']:<12.4f} | "
              f"{m['mean_within_class_variance']:<12.4f} | {m['variance_ratio']:<12.2f}")
    
    print("\nStatistical Tests (K-S Test):")
    print(f"{'Comparison':<30} | {'Mean K-S':<12} | {'p-value':<12} | {'Similar?':<10}")
    print("-"*70)
    for comp_name, comp_result in comparison_results.items():
        similar = "Yes" if comp_result['distributions_similar'] else "No"
        print(f"{comp_name:<30} | {comp_result['mean_ks_statistic']:<12.4f} | "
              f"{comp_result['mean_p_value']:<12.4f} | {similar:<10}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    # Check if Stateful is statistically indistinguishable from Projan
    stat_vs_projan = comparison_results.get('projan_vs_stateful_projan', {})
    if stat_vs_projan.get('distributions_similar', False):
        print("✓ Stateful Projan's latent space is STATISTICALLY INDISTINGUISHABLE")
        print("  from original Projan (p > 0.05 in K-S test)")
        print("✓ L_partition does NOT create detectably 'artificial' clustering")
    else:
        print("✗ Stateful Projan's latent space differs from Projan")
        print("  May be detectable via latent space analysis")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benign_path', type=str, required=True)
    parser.add_argument('--projan_path', type=str, required=True)
    parser.add_argument('--stateful_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='net')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='experiment9_results')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    env['device'] = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    run_experiment9(args.benign_path, args.projan_path, args.stateful_path,
                   dataset_name=args.dataset, model_name=args.model,
                   num_samples=args.num_samples, output_dir=args.output_dir)
