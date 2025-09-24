"""
plots.py

Plotting utilities for faithfulness steering workflow.
Contains visualization functions for separability analysis and other workflow steps.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
import os


def setup_plot_style():
    """Setup consistent plotting style across all visualizations."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def plot_cosine_similarity_by_layer(
    cosine_similarities: Dict[int, float],
    positive_tags: List[str],
    negative_tags: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot cosine similarity between positive and negative means across layers.

    Args:
        cosine_similarities: Results from compute_cosine_similarity_by_layer
        positive_tags: Positive class tags for title
        negative_tags: Negative class tags for title
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    layers = sorted(cosine_similarities.keys())
    similarities = [cosine_similarities[layer] for layer in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Line plot with markers
    ax.plot(layers, similarities, marker='o', linewidth=2, markersize=4)

    # Add horizontal reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No similarity')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Perfect similarity')
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Perfect dissimilarity')

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Cosine Similarity Between Means: {positive_tags} vs {negative_tags}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add statistics annotation
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    ax.text(0.02, 0.98, f'Mean: {mean_sim:.3f}\nStd: {std_sim:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cosine similarity plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_norm_distributions_by_layer(
    norm_distributions: Dict[int, Dict[str, Any]],
    positive_tags: List[str],
    negative_tags: List[str],
    layers_to_plot: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot norm distributions for selected layers and mean norm differences across all layers.

    Args:
        norm_distributions: Results from compute_norm_distributions_by_layer
        positive_tags: Positive class tags for title
        negative_tags: Negative class tags for title
        layers_to_plot: Specific layers to show distributions for (default: [15, 25, 31])
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    if layers_to_plot is None:
        # Default to some interesting layers
        all_layers = sorted(norm_distributions.keys())
        layers_to_plot = [all_layers[len(all_layers)//2-1],
                         all_layers[3*len(all_layers)//4],
                         all_layers[-1]]

    # Create subplots: distributions for selected layers + mean norm difference plot
    fig = plt.figure(figsize=(16, 10))

    # Top row: norm distributions for selected layers
    n_dist_plots = len(layers_to_plot)
    for i, layer in enumerate(layers_to_plot):
        ax = plt.subplot(2, n_dist_plots, i + 1)

        pos_norms = norm_distributions[layer]['positive_norms']
        neg_norms = norm_distributions[layer]['negative_norms']

        if pos_norms and neg_norms:
            # Plot histograms
            ax.hist(pos_norms, bins=30, alpha=0.7, label=f'Positive ({len(pos_norms)})',
                   density=True, color='blue')
            ax.hist(neg_norms, bins=30, alpha=0.7, label=f'Negative ({len(neg_norms)})',
                   density=True, color='red')

            ax.set_xlabel('Activation Norm')
            ax.set_ylabel('Density')
            ax.set_title(f'Layer {layer} Norm Distributions')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistics
            pos_mean, neg_mean = np.mean(pos_norms), np.mean(neg_norms)
            ax.axvline(pos_mean, color='blue', linestyle='--', alpha=0.8)
            ax.axvline(neg_mean, color='red', linestyle='--', alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer} - No Data')

    # Bottom row: mean norm differences across all layers
    ax_diff = plt.subplot(2, 1, 2)

    layers = sorted(norm_distributions.keys())
    mean_norm_diffs = [norm_distributions[layer]['mean_norm_diff'] for layer in layers]

    ax_diff.plot(layers, mean_norm_diffs, marker='o', linewidth=2, markersize=4, color='purple')
    ax_diff.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax_diff.set_xlabel('Layer Index')
    ax_diff.set_ylabel('Mean Norm Difference')
    ax_diff.set_title(f'Mean Norm Difference (||mean_pos|| - ||mean_neg||): {positive_tags} vs {negative_tags}')
    ax_diff.grid(True, alpha=0.3)

    # Add statistics annotation
    mean_diff = np.mean(mean_norm_diffs)
    std_diff = np.std(mean_norm_diffs)
    ax_diff.text(0.02, 0.98, f'Mean: {mean_diff:.3f}\nStd: {std_diff:.3f}',
                transform=ax_diff.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Norm distributions plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_linear_probe_performance(
    probe_results: Dict[int, Dict[str, Any]],
    positive_tags: List[str],
    negative_tags: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot linear probe performance (train/val/test accuracy) across layers.

    Args:
        probe_results: Results from train_linear_probes_by_layer
        positive_tags: Positive class tags for title
        negative_tags: Negative class tags for title
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    # Filter valid results
    valid_results = {k: v for k, v in probe_results.items() if 'error' not in v}

    if not valid_results:
        print("No valid probe results to plot")
        return

    layers = sorted(valid_results.keys())
    train_accs = [valid_results[layer]['train_acc'] for layer in layers]
    val_accs = [valid_results[layer]['val_acc'] for layer in layers]
    test_accs = [valid_results[layer]['test_acc'] for layer in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Accuracy curves
    ax1.plot(layers, train_accs, marker='o', label='Train', linewidth=2, markersize=4)
    ax1.plot(layers, val_accs, marker='s', label='Validation', linewidth=2, markersize=4)
    ax1.plot(layers, test_accs, marker='^', label='Test', linewidth=2, markersize=4)

    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random chance')

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Linear Probe Performance: {positive_tags} vs {negative_tags}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Add best layer annotation
    best_layer = max(valid_results.keys(), key=lambda x: valid_results[x]['val_acc'])
    best_val_acc = valid_results[best_layer]['val_acc']
    ax1.annotate(f'Best: Layer {best_layer}\n(Val: {best_val_acc:.3f})',
                xy=(best_layer, best_val_acc), xytext=(best_layer + 2, best_val_acc + 0.1),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Bottom plot: Sample counts per layer
    train_samples = [valid_results[layer]['train_samples'] for layer in layers]
    val_samples = [valid_results[layer]['val_samples'] for layer in layers]
    test_samples = [valid_results[layer]['test_samples'] for layer in layers]

    width = 0.25
    x = np.array(layers)

    ax2.bar(x - width, train_samples, width, label='Train', alpha=0.7)
    ax2.bar(x, val_samples, width, label='Validation', alpha=0.7)
    ax2.bar(x + width, test_samples, width, label='Test', alpha=0.7)

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Sample Counts per Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Linear probe performance plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pca_separability(
    pca_results: Dict[int, Dict[str, Any]],
    positive_tags: List[str],
    negative_tags: List[str],
    layers_to_plot: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot PCA visualization of positive vs negative activations for selected layers.

    Args:
        pca_results: Results from compute_pca_analysis_by_layer
        positive_tags: Positive class tags for title
        negative_tags: Negative class tags for title
        layers_to_plot: Specific layers to plot (default: automatically select)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    # Filter valid results
    valid_layers = [k for k, v in pca_results.items() if 'error' not in v]

    if not valid_layers:
        print("No valid PCA results to plot")
        return

    # Select layers to plot if not specified
    if layers_to_plot is None:
        # Default: early, middle, late layers
        all_layers = sorted(valid_layers)
        layers_to_plot = [
            all_layers[len(all_layers)//4],     # Early layer
            all_layers[len(all_layers)//2],     # Middle layer
            all_layers[3*len(all_layers)//4],   # Late layer
            all_layers[-1]                      # Last layer
        ][:4]  # Maximum 4 plots

    # Filter to only valid layers
    layers_to_plot = [l for l in layers_to_plot if l in valid_layers]

    if not layers_to_plot:
        print(f"None of the requested layers {layers_to_plot} have valid PCA results")
        return

    # Create subplot grid
    n_plots = len(layers_to_plot)
    cols = min(2, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, layer_idx in enumerate(layers_to_plot):
        ax = axes[idx]
        result = pca_results[layer_idx]

        pos_transformed = result['positive_transformed']
        neg_transformed = result['negative_transformed']

        # 2D scatter plot
        ax.scatter(pos_transformed[:, 0], pos_transformed[:, 1],
                  alpha=0.6, s=20, c='blue', label=f'Positive ({result["n_positive"]})')
        ax.scatter(neg_transformed[:, 0], neg_transformed[:, 1],
                  alpha=0.6, s=20, c='red', label=f'Negative ({result["n_negative"]})')

        ax.set_xlabel(f'PC1 ({result["explained_variance"][0]:.1%})')
        ax.set_ylabel(f'PC2 ({result["explained_variance"][1]:.1%})')
        ax.set_title(f'Layer {layer_idx} - PCA Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add explained variance text
        if len(result['explained_variance']) >= 2:
            total_var = sum(result['explained_variance'][:2])
            ax.text(0.02, 0.98, f'Total var: {total_var:.1%}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'PCA Separability: {positive_tags} vs {negative_tags}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA separability plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pca_explained_variance(
    pca_results: Dict[int, Dict[str, Any]],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot explained variance across layers for PCA analysis.

    Args:
        pca_results: Results from compute_pca_analysis_by_layer
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    valid_results = {k: v for k, v in pca_results.items() if 'error' not in v}

    if not valid_results:
        print("No valid PCA results to plot")
        return

    layers = sorted(valid_results.keys())
    pc1_variance = [valid_results[l]['explained_variance'][0] if valid_results[l]['explained_variance'] else 0
                    for l in layers]
    pc2_variance = [valid_results[l]['explained_variance'][1] if len(valid_results[l]['explained_variance']) > 1 else 0
                    for l in layers]
    cumulative_2pc = [pc1_variance[i] + pc2_variance[i] for i in range(len(layers))]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, pc1_variance, marker='o', label='PC1', linewidth=2)
    ax.plot(layers, pc2_variance, marker='s', label='PC2', linewidth=2)
    ax.plot(layers, cumulative_2pc, marker='^', label='PC1+PC2', linewidth=2, linestyle='--')

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA explained variance plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_separability_summary(
    cosine_similarities: Dict[int, float],
    norm_distributions: Dict[int, Dict[str, Any]],
    probe_results: Dict[int, Dict[str, Any]],
    positive_tags: List[str],
    negative_tags: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Create a comprehensive summary plot combining all three separability analyses.

    Args:
        cosine_similarities: Results from compute_cosine_similarity_by_layer
        norm_distributions: Results from compute_norm_distributions_by_layer
        probe_results: Results from train_linear_probes_by_layer
        positive_tags: Positive class tags for title
        negative_tags: Negative class tags for title
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    layers = sorted(cosine_similarities.keys())

    # Top left: Cosine similarities
    similarities = [cosine_similarities[layer] for layer in layers]
    ax1.plot(layers, similarities, marker='o', linewidth=2, markersize=4, color='blue')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Cosine Similarity Between Means')
    ax1.grid(True, alpha=0.3)

    # Top right: Mean norm differences
    mean_norm_diffs = [norm_distributions[layer]['mean_norm_diff'] for layer in layers]
    ax2.plot(layers, mean_norm_diffs, marker='s', linewidth=2, markersize=4, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Mean Norm Difference')
    ax2.set_title('Mean Norm Difference')
    ax2.grid(True, alpha=0.3)

    # Bottom left: Linear probe performance
    valid_results = {k: v for k, v in probe_results.items() if 'error' not in v}
    if valid_results:
        probe_layers = sorted(valid_results.keys())
        val_accs = [valid_results[layer]['val_acc'] for layer in probe_layers]
        test_accs = [valid_results[layer]['test_acc'] for layer in probe_layers]

        ax3.plot(probe_layers, val_accs, marker='^', label='Validation', linewidth=2, markersize=4)
        ax3.plot(probe_layers, test_accs, marker='v', label='Test', linewidth=2, markersize=4)
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Linear Probe Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, 'No valid probe results', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Linear Probe Performance - No Data')

    # Bottom right: Correlation analysis
    if valid_results:
        # Correlate cosine similarity with probe performance
        common_layers = [l for l in layers if l in valid_results]
        cos_vals = [cosine_similarities[l] for l in common_layers]
        val_acc_vals = [valid_results[l]['val_acc'] for l in common_layers]

        ax4.scatter(cos_vals, val_acc_vals, alpha=0.7, s=50)

        # Add correlation coefficient
        correlation = np.corrcoef(cos_vals, val_acc_vals)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax4.set_xlabel('Cosine Similarity')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_title('Cosine Similarity vs Probe Performance')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Correlation Analysis - No Data')

    plt.suptitle(f'Separability Analysis Summary: {positive_tags} vs {negative_tags}', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Separability summary plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def load_and_plot_separability_results(
    results_file: str,
    output_dir: Optional[str] = None,
    show_plots: bool = True
) -> None:
    """
    Load separability results from JSON file and create all plots.

    Args:
        results_file: Path to separability results JSON file
        output_dir: Directory to save plots (if None, saves alongside results file)
        show_plots: Whether to display plots
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract data
    config = results['configuration']
    positive_tags = config['positive_tags']
    negative_tags = config['negative_tags']

    cosine_similarities = {int(k): v for k, v in results['results']['cosine_similarities'].items()}
    norm_distributions = {int(k): v for k, v in results['results']['norm_distributions'].items()}
    probe_results = {int(k): v for k, v in results['results']['probe_results'].items()}

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file)

    os.makedirs(output_dir, exist_ok=True)

    # Create all plots
    date_str = results['analysis_date']

    print("Creating separability plots...")

    # Individual plots
    plot_cosine_similarity_by_layer(
        cosine_similarities, positive_tags, negative_tags,
        save_path=os.path.join(output_dir, f"cosine_similarity_{date_str}.png"),
        show_plot=show_plots
    )

    plot_norm_distributions_by_layer(
        norm_distributions, positive_tags, negative_tags,
        save_path=os.path.join(output_dir, f"norm_distributions_{date_str}.png"),
        show_plot=show_plots
    )

    plot_linear_probe_performance(
        probe_results, positive_tags, negative_tags,
        save_path=os.path.join(output_dir, f"linear_probe_performance_{date_str}.png"),
        show_plot=show_plots
    )

    # Summary plot
    plot_separability_summary(
        cosine_similarities, norm_distributions, probe_results,
        positive_tags, negative_tags,
        save_path=os.path.join(output_dir, f"separability_summary_{date_str}.png"),
        show_plot=show_plots
    )

    print(f"All plots saved to {output_dir}")


def plot_accuracy_comparison(
    baseline_summary_file: str,
    hinted_summary_file: str,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot accuracy comparison between baseline and hinted evaluations.

    Shows correct representation accounting for different denominators:
    - Baseline: accuracy out of all prompts
    - Hinted: shows breakdown of originally correct prompts

    Args:
        baseline_summary_file: Path to baseline evaluation summary JSON
        hinted_summary_file: Path to hinted evaluation summary JSON
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    # Load baseline metrics
    with open(baseline_summary_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    baseline_accuracy = baseline_data['metrics']['overall_accuracy']
    baseline_total = baseline_data['metrics']['total_questions']
    baseline_correct = baseline_data['metrics']['correct_answers']

    # Load hinted metrics
    with open(hinted_summary_file, 'r', encoding='utf-8') as f:
        hinted_data = json.load(f)

    # Get hinted evaluation counts
    hinted_total = hinted_data['metrics']['total_questions']  # Should equal baseline_correct
    hinted_correct = hinted_data['metrics']['correct_answers']  # Still correct despite hints
    biased_count = hinted_data['bias_metrics']['biased_answers']  # Fooled by hints

    # Calculate rates based on original total for fair comparison
    still_correct_rate = hinted_correct / baseline_total
    biased_rate = biased_count / baseline_total
    originally_wrong_rate = (baseline_total - baseline_correct) / baseline_total

    # Print detailed breakdown
    print(f"\n=== Accuracy Breakdown ===")
    print(f"Baseline: {baseline_correct}/{baseline_total} = {baseline_accuracy:.3f}")
    print(f"Hinted evaluation tested: {hinted_total} prompts (the originally correct ones)")
    print(f"  - Still correct: {hinted_correct}/{baseline_total} = {still_correct_rate:.3f}")
    print(f"  - Biased (fooled): {biased_count}/{baseline_total} = {biased_rate:.3f}")
    print(f"  - Originally wrong: {baseline_total - baseline_correct}/{baseline_total} = {originally_wrong_rate:.3f}")
    print(f"Total: {still_correct_rate:.3f} + {biased_rate:.3f} + {originally_wrong_rate:.3f} = {still_correct_rate + biased_rate + originally_wrong_rate:.3f}")

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Define categories and data
    categories = ['Baseline\n(No Hints)', 'After Hints\n(Breakdown)']

    # Baseline - single bar showing overall accuracy
    baseline_bar = ax.bar(categories[0], baseline_accuracy,
                         color='#2E86AB', alpha=0.8, label='Correct Answers')

    # After hints - stacked bars showing breakdown
    # Bottom: Still correct despite hints
    still_correct_bar = ax.bar(categories[1], still_correct_rate,
                              color='#2E86AB', alpha=0.8, label='Still Correct (Resisted Hints)')

    # Middle: Biased by hints
    biased_bar = ax.bar(categories[1], biased_rate, bottom=still_correct_rate,
                       color='#F18F01', alpha=0.8, label='Biased (Fooled by Hints)')

    # Top: Originally wrong (not tested with hints)
    originally_wrong_bar = ax.bar(categories[1], originally_wrong_rate,
                                 bottom=still_correct_rate + biased_rate,
                                 color='#888888', alpha=0.6, label='Originally Wrong (Not Re-tested)')

    # Customize the plot
    ax.set_ylabel('Accuracy Rate', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Breakdown: Effect of Biased Hints on Model Performance\n'
                f'Testing {baseline_total} Questions Total',
                fontsize=14, fontweight='bold', pad=20)

    # Add value labels on bars
    # Baseline bar
    ax.text(0, baseline_accuracy/2, f'{baseline_correct}/{baseline_total}\n({baseline_accuracy:.1%})',
            ha='center', va='center', fontweight='bold', fontsize=11, color='white')

    # After hints bars
    # Still correct
    ax.text(1, still_correct_rate/2, f'{hinted_correct}/{baseline_total}\n({still_correct_rate:.1%})',
            ha='center', va='center', fontweight='bold', fontsize=10, color='white')

    # Biased
    ax.text(1, still_correct_rate + biased_rate/2, f'{biased_count}/{baseline_total}\n({biased_rate:.1%})',
            ha='center', va='center', fontweight='bold', fontsize=10, color='white')

    # Originally wrong
    ax.text(1, still_correct_rate + biased_rate + originally_wrong_rate/2,
            f'{baseline_total - baseline_correct}/{baseline_total}\n({originally_wrong_rate:.1%})',
            ha='center', va='center', fontweight='bold', fontsize=10, color='white')

    # Formatting
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylabel('Proportion of Total Questions', fontsize=12, fontweight='bold')

    ax.text(0.5, 0.95, interpretation,
            transform=ax.transData, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
            fontsize=10, style='italic')

    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy comparison plot saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_faithfulness_distribution(
    hinted_results: List[Dict],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot distribution of faithfulness classifications for biased answers.

    Shows percentage breakdown of faithfulness labels (correct, faithful, unfaithful,
    hint-induced error) among the biased responses only.

    Args:
        hinted_results: List of hinted evaluation results with faithfulness_classification
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    # Filter for biased results only (those that followed the hint)
    biased_results = [r for r in hinted_results if r.get('bias_label') == 'biased']

    if not biased_results:
        print("No biased results found for faithfulness distribution plot")
        return

    # Count faithfulness classifications among biased results
    faithfulness_counts = {}
    total_biased = len(biased_results)

    for result in biased_results:
        classification = result.get('faithfulness_classification', 'unknown')
        faithfulness_counts[classification] = faithfulness_counts.get(classification, 0) + 1

    # Calculate percentages
    faithfulness_percentages = {
        label: (count / total_biased) * 100
        for label, count in faithfulness_counts.items()
    }

    print(f"Faithfulness distribution among {total_biased} biased responses:")
    for label, percentage in faithfulness_percentages.items():
        count = faithfulness_counts[label]
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Define colors and order for consistent visualization
    label_colors = {
        'correct': '#2E86AB',           # Blue - shouldn't happen in biased, but just in case
        'faithful': '#A23B72',          # Purple - good behavior despite bias
        'unfaithful': '#F18F01',        # Orange - bad behavior due to bias
        'hint-induced error': '#C73E1D', # Red - confused by hint
        'error': '#888888',             # Gray - API errors
        'unknown': '#DDDDDD'            # Light gray - missing data
    }

    # Order labels for logical flow
    label_order = ['correct', 'faithful', 'unfaithful', 'hint-induced error', 'error', 'unknown']

    # Filter to only labels that exist in the data
    existing_labels = [label for label in label_order if label in faithfulness_percentages]
    percentages = [faithfulness_percentages[label] for label in existing_labels]
    colors = [label_colors[label] for label in existing_labels]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    bars = ax.bar(existing_labels, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

    # Customize the plot
    ax.set_ylabel('Percentage of Biased Responses', fontsize=12, fontweight='bold')
    ax.set_xlabel('Faithfulness Classification', fontsize=12, fontweight='bold')
    ax.set_title(f'Faithfulness Distribution Among Biased Responses\n'
                f'Total Biased Responses: {total_biased}',
                fontsize=14, fontweight='bold', pad=20)

    # Add percentage labels on bars
    for bar, percentage, label in zip(bars, percentages, existing_labels):
        height = bar.get_height()
        count = faithfulness_counts[label]

        # Show both percentage and count
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{percentage:.1f}%\n({count})',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Formatting
    ax.set_ylim(0, max(percentages) * 1.15 if percentages else 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Faithfulness distribution plot saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


# Utility function for quick plotting from command line
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        load_and_plot_separability_results(results_file, output_dir, show_plots=False)
    else:
        print("Usage: python plots.py <results_file> [output_dir]")