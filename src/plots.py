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


def plot_mean_differences_by_layer(
    mean_differences: Dict[int, Dict[str, Any]],
    positive_tags: List[str],
    negative_tags: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot distance between positive and negative means across all layers.

    Args:
        mean_differences: Results from compute_mean_differences_by_layer
        positive_tags: Positive class tags for title
        negative_tags: Negative class tags for title
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    layers = sorted(mean_differences.keys())
    mean_diff_norms = [mean_differences[layer]['mean_diff_norm'] for layer in layers]

    ax.plot(layers, mean_diff_norms, marker='o', linewidth=2, markersize=4, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Distance Between Means')
    ax.set_title(f'Distance Between Means (||mean_pos - mean_neg||): {positive_tags} vs {negative_tags}')
    ax.grid(True, alpha=0.3)

    # Add statistics annotation
    mean_diff = np.mean(mean_diff_norms)
    std_diff = np.std(mean_diff_norms)
    ax.text(0.02, 0.98, f'Mean: {mean_diff:.3f}\nStd: {std_diff:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mean differences plot saved to {save_path}")

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
    Plot linear probe performance (train/val accuracy) across layers.

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Accuracy curves
    ax1.plot(layers, train_accs, marker='o', label='Train', linewidth=2, markersize=4)
    ax1.plot(layers, val_accs, marker='s', label='Val', linewidth=2, markersize=4)

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

    width = 0.4
    x = np.array(layers)

    ax2.bar(x - width/2, train_samples, width, label='Train', alpha=0.7)
    ax2.bar(x + width/2, val_samples, width, label='Val', alpha=0.7)

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
    mean_differences: Dict[int, Dict[str, Any]],
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
        mean_differences: Results from compute_mean_differences_by_layer
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
    mean_diff_norms = [mean_differences[layer]['mean_diff_norm'] for layer in layers]
    ax2.plot(layers, mean_diff_norms, marker='s', linewidth=2, markersize=4, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Distance Between Means')
    ax2.set_title('Distance Between Means')
    ax2.grid(True, alpha=0.3)

    # Bottom left: Linear probe performance
    valid_results = {k: v for k, v in probe_results.items() if 'error' not in v}
    if valid_results:
        probe_layers = sorted(valid_results.keys())
        train_accs = [valid_results[layer]['train_acc'] for layer in probe_layers]
        val_accs = [valid_results[layer]['val_acc'] for layer in probe_layers]

        ax3.plot(probe_layers, train_accs, marker='o', label='Train', linewidth=2, markersize=4)
        ax3.plot(probe_layers, val_accs, marker='s', label='Val', linewidth=2, markersize=4)
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


def plot_hint_breakdown(
    hinted_results: List[Dict[str, Any]],
    baseline_summary_file: str,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot horizontal stacked bars showing final outcomes breakdown per hint template.

    Shows breakdown by hint template with categories:
    - Correct after bias (still correct despite hint)
    - Biased (wrong AND followed hint - needs faithfulness annotation)
    - Hint-induced error (wrong but didn't follow hint - no annotation needed)
    - Null after bias (no answer extracted after hint)
    - Wrong after baseline (constant across all hints - not re-tested)
    - Null after baseline (constant across all hints - not re-tested)

    Args:
        hinted_results: List of hinted evaluation results with hint_template field
        baseline_summary_file: Path to baseline summary to get baseline failures
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    # Load baseline summary to get baseline failures
    with open(baseline_summary_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)

    baseline_total = baseline_data['metrics']['total_questions']
    baseline_wrong = baseline_data['metrics']['wrong_answers']
    baseline_no_answer = baseline_data['metrics'].get('extraction_failures', 0)

    # Group results by hint template
    hint_groups = {}
    for result in hinted_results:
        hint_template = result.get('hint_template', 'unknown')
        if hint_template not in hint_groups:
            hint_groups[hint_template] = []
        hint_groups[hint_template].append(result)

    # Calculate counts for each hint template
    hint_stats = {}
    for hint_template, results in hint_groups.items():
        # Count the hinted categories (now 4 instead of 3)
        correct_after_bias = sum(1 for r in results if r.get('accuracy_label') == 'correct')
        biased = sum(1 for r in results if r.get('bias_label') == 'biased')  # Followed hint
        hint_induced_error = sum(1 for r in results if r.get('bias_label') == 'hint-induced error')  # Wrong but didn't follow hint
        null_after_bias = sum(1 for r in results if r.get('bias_label') == 'no_answer')

        hint_stats[hint_template] = {
            'correct_after_bias': correct_after_bias,
            'biased': biased,  # Wrong AND followed hint
            'hint_induced_error': hint_induced_error,  # Wrong but didn't follow hint
            'null_after_bias': null_after_bias,
            'wrong_after_baseline': baseline_wrong,  # Same for all
            'null_after_baseline': baseline_no_answer  # Same for all
        }

    # Prepare data for plotting
    hint_templates = sorted(hint_stats.keys())
    n_hints = len(hint_templates)

    if n_hints == 0:
        print("No hint templates found in results")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, max(8, n_hints * 1.5)))

    # Colors for different categories
    color_correct = '#2E86AB'        # Blue - correct after bias
    color_biased = '#F18F01'          # Orange - followed hint (biased)
    color_hint_error = '#FDB833'      # Yellow - hint-induced error
    color_null_bias = '#AAAAAA'      # Light gray - null after bias
    color_wrong_baseline = '#C73E1D' # Red - wrong after baseline
    color_null_baseline = '#666666'  # Dark gray - null after baseline

    # Create horizontal stacked bars for each hint template
    for i, hint in enumerate(hint_templates):
        stats = hint_stats[hint]
        bottom = 0

        # Segment 1: Correct after bias
        ax.barh(i, stats['correct_after_bias'], left=bottom, height=0.6,
                color=color_correct, alpha=0.9, edgecolor='black', linewidth=0.5)
        if stats['correct_after_bias'] > 0:
            ax.text(bottom + stats['correct_after_bias']/2, i,
                    f"{stats['correct_after_bias']}\n({stats['correct_after_bias']/baseline_total:.1%})",
                    ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        bottom += stats['correct_after_bias']

        # Segment 2: Biased (followed hint)
        ax.barh(i, stats['biased'], left=bottom, height=0.6,
                color=color_biased, alpha=0.9, edgecolor='black', linewidth=0.5)
        if stats['biased'] > 0:
            ax.text(bottom + stats['biased']/2, i,
                    f"{stats['biased']}\n({stats['biased']/baseline_total:.1%})",
                    ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        bottom += stats['biased']

        # Segment 3: Hint-induced error (wrong but didn't follow hint)
        ax.barh(i, stats['hint_induced_error'], left=bottom, height=0.6,
                color=color_hint_error, alpha=0.9, edgecolor='black', linewidth=0.5)
        if stats['hint_induced_error'] > 0:
            ax.text(bottom + stats['hint_induced_error']/2, i,
                    f"{stats['hint_induced_error']}\n({stats['hint_induced_error']/baseline_total:.1%})",
                    ha='center', va='center', fontweight='bold', fontsize=9, color='black')
        bottom += stats['hint_induced_error']

        # Segment 4: Null after bias
        ax.barh(i, stats['null_after_bias'], left=bottom, height=0.6,
                color=color_null_bias, alpha=0.8, edgecolor='black', linewidth=0.5)
        if stats['null_after_bias'] > 0:
            ax.text(bottom + stats['null_after_bias']/2, i,
                    f"{stats['null_after_bias']}\n({stats['null_after_bias']/baseline_total:.1%})",
                    ha='center', va='center', fontweight='bold', fontsize=8, color='white')
        bottom += stats['null_after_bias']

        # Segment 5: Wrong after baseline
        ax.barh(i, stats['wrong_after_baseline'], left=bottom, height=0.6,
                color=color_wrong_baseline, alpha=0.8, edgecolor='black', linewidth=0.5)
        if stats['wrong_after_baseline'] > 0:
            ax.text(bottom + stats['wrong_after_baseline']/2, i,
                    f"{stats['wrong_after_baseline']}\n({stats['wrong_after_baseline']/baseline_total:.1%})",
                    ha='center', va='center', fontweight='bold', fontsize=8, color='white')
        bottom += stats['wrong_after_baseline']

        # Segment 6: Null after baseline
        ax.barh(i, stats['null_after_baseline'], left=bottom, height=0.6,
                color=color_null_baseline, alpha=0.8, edgecolor='black', linewidth=0.5)
        if stats['null_after_baseline'] > 0:
            ax.text(bottom + stats['null_after_baseline']/2, i,
                    f"{stats['null_after_baseline']}\n({stats['null_after_baseline']/baseline_total:.1%})",
                    ha='center', va='center', fontweight='bold', fontsize=8, color='white')

    # Customize the plot
    ax.set_xlabel('Number of Prompts', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hint Template', fontsize=13, fontweight='bold')
    ax.set_title(f'Final Outcomes by Hint Template\nTotal: {baseline_total} Prompts',
                fontsize=15, fontweight='bold', pad=20)

    ax.set_xlim(0, baseline_total * 1.02)
    ax.set_ylim(-0.5, n_hints - 0.5)
    ax.set_yticks(range(n_hints))
    ax.set_yticklabels(hint_templates, fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_correct, edgecolor='black', label='Correct After Bias'),
        Patch(facecolor=color_biased, edgecolor='black', label='Biased (Followed Hint)'),
        Patch(facecolor=color_hint_error, edgecolor='black', label='Hint-Induced Error'),
        Patch(facecolor=color_null_bias, edgecolor='black', label='Null After Bias'),
        Patch(facecolor=color_wrong_baseline, edgecolor='black', label='Wrong After Baseline'),
        Patch(facecolor=color_null_baseline, edgecolor='black', label='Null After Baseline')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Print summary statistics
    print(f"\n=== Hint Template Breakdown ===")
    for hint in hint_templates:
        stats = hint_stats[hint]
        print(f"\n{hint}:")
        print(f"  Correct after bias: {stats['correct_after_bias']}")
        print(f"  Biased (followed hint): {stats['biased']}")
        print(f"  Hint-induced error: {stats['hint_induced_error']}")
        print(f"  Null after bias: {stats['null_after_bias']}")
        print(f"  Wrong after baseline: {stats['wrong_after_baseline']} (constant)")
        print(f"  Null after baseline: {stats['null_after_baseline']} (constant)")

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nHint breakdown plot saved to {save_path}")

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
        'other': '#FDB833',             # Yellow - ambiguous edge cases
        'error': '#888888',             # Gray - API errors
        'unknown': '#DDDDDD'            # Light gray - missing data
    }

    # Order labels for logical flow
    label_order = ['correct', 'faithful', 'unfaithful', 'hint-induced error', 'other', 'error', 'unknown']

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


def plot_faithfulness_by_bias(
    hinted_results: List[Dict],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot faithfulness distribution broken down by bias type (hint_template).

    Creates a grouped bar chart showing faithfulness classifications for each bias type,
    similar to how bias rates are shown per bias in hinted_eval.py.

    Args:
        hinted_results: List of hinted evaluation results with faithfulness_classification
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    # Filter for biased results only
    biased_results = [r for r in hinted_results if r.get('bias_label') == 'biased']

    if not biased_results:
        print("No biased results found for bias-wise faithfulness plot")
        return

    # Group by bias type (hint_template)
    bias_type_data = {}
    for result in biased_results:
        bias_type = result.get('hint_template', 'unknown')
        if bias_type not in bias_type_data:
            bias_type_data[bias_type] = []
        bias_type_data[bias_type].append(result)

    # Calculate faithfulness distribution for each bias type
    faithfulness_labels = ['correct', 'faithful', 'unfaithful', 'hint-induced error', 'other', 'error']
    bias_types = sorted(bias_type_data.keys())

    # Data structure: bias_type -> classification -> percentage
    data_by_bias = {}
    for bias_type in bias_types:
        results = bias_type_data[bias_type]
        total = len(results)

        counts = {label: 0 for label in faithfulness_labels}
        for result in results:
            classification = result.get('faithfulness_classification', 'error')
            if classification in counts:
                counts[classification] += 1
            else:
                counts['error'] += 1

        percentages = {label: (counts[label] / total * 100) if total > 0 else 0
                      for label in faithfulness_labels}

        data_by_bias[bias_type] = {
            'counts': counts,
            'percentages': percentages,
            'total': total
        }

    # Print summary
    print(f"\nFaithfulness distribution by bias type:")
    for bias_type in bias_types:
        print(f"\n{bias_type} (n={data_by_bias[bias_type]['total']}):")
        for label in faithfulness_labels:
            count = data_by_bias[bias_type]['counts'][label]
            pct = data_by_bias[bias_type]['percentages'][label]
            if count > 0:
                print(f"  {label}: {count} ({pct:.1f}%)")

    # Create grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Define colors for each classification
    label_colors = {
        'correct': '#2E86AB',           # Blue
        'faithful': '#A23B72',          # Purple
        'unfaithful': '#F18F01',        # Orange
        'hint-induced error': '#C73E1D', # Red
        'other': '#FDB833',             # Yellow - ambiguous edge cases
        'error': '#888888'              # Gray
    }

    # Set up bar positions
    x = np.arange(len(bias_types))
    width = 0.15  # Width of each bar

    # Plot bars for each classification
    for i, label in enumerate(faithfulness_labels):
        percentages = [data_by_bias[bt]['percentages'][label] for bt in bias_types]
        offset = (i - len(faithfulness_labels)/2 + 0.5) * width
        bars = ax.bar(x + offset, percentages, width,
                     label=label, color=label_colors[label],
                     alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add count labels on bars (only if percentage > 5%)
        for j, (bar, bias_type) in enumerate(zip(bars, bias_types)):
            height = bar.get_height()
            if height > 5:  # Only show label if bar is tall enough
                count = data_by_bias[bias_type]['counts'][label]
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{count}', ha='center', va='bottom', fontsize=7)

    # Customize plot
    ax.set_ylabel('Percentage of Biased Responses', fontsize=12, fontweight='bold')
    ax.set_xlabel('Bias Type (Hint Template)', fontsize=12, fontweight='bold')
    ax.set_title('Faithfulness Distribution by Bias Type\n'
                f'Total Biased Responses: {len(biased_results)}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bias_types, rotation=45, ha='right')
    ax.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nBias-wise faithfulness plot saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_global_faithfulness_by_bias(
    hinted_results: List[Dict],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot global faithfulness distribution broken down by bias type (hint_template).

    Simplified version for global LLM judge which only produces 3 classifications:
    faithful, unfaithful, error.

    Args:
        hinted_results: List of hinted evaluation results with faithfulness_classification
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    setup_plot_style()

    # Filter for biased results only
    biased_results = [r for r in hinted_results if r.get('bias_label') == 'biased']

    if not biased_results:
        print("No biased results found for bias-wise faithfulness plot")
        return

    # Group by bias type (hint_template)
    bias_type_data = {}
    for result in biased_results:
        bias_type = result.get('hint_template', 'unknown')
        if bias_type not in bias_type_data:
            bias_type_data[bias_type] = []
        bias_type_data[bias_type].append(result)

    # Global judge only produces these 3 classifications
    faithfulness_labels = ['faithful', 'unfaithful', 'error']
    bias_types = sorted(bias_type_data.keys())

    # Data structure: bias_type -> classification -> percentage
    data_by_bias = {}
    for bias_type in bias_types:
        results = bias_type_data[bias_type]
        total = len(results)

        counts = {label: 0 for label in faithfulness_labels}
        for result in results:
            classification = result.get('faithfulness_classification', 'error')
            if classification in counts:
                counts[classification] += 1
            else:
                counts['error'] += 1

        percentages = {label: (counts[label] / total * 100) if total > 0 else 0
                      for label in faithfulness_labels}

        data_by_bias[bias_type] = {
            'counts': counts,
            'percentages': percentages,
            'total': total
        }

    # Print summary
    print(f"\nGlobal faithfulness distribution by bias type:")
    for bias_type in bias_types:
        print(f"\n{bias_type} (n={data_by_bias[bias_type]['total']}):")
        for label in faithfulness_labels:
            count = data_by_bias[bias_type]['counts'][label]
            pct = data_by_bias[bias_type]['percentages'][label]
            if count > 0:
                print(f"  {label}: {count} ({pct:.1f}%)")

    # Create grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define colors for global judge classifications
    label_colors = {
        'faithful': '#A23B72',          # Purple - faithful reasoning
        'unfaithful': '#F18F01',        # Orange - unfaithful reasoning
        'error': '#888888'              # Gray - API/parsing errors
    }

    # Set up bar positions
    x = np.arange(len(bias_types))
    width = 0.25  # Width of each bar (wider since only 3 categories)

    # Plot bars for each classification
    for i, label in enumerate(faithfulness_labels):
        percentages = [data_by_bias[bt]['percentages'][label] for bt in bias_types]
        offset = (i - len(faithfulness_labels)/2 + 0.5) * width
        bars = ax.bar(x + offset, percentages, width,
                     label=label.capitalize(), color=label_colors[label],
                     alpha=0.8, edgecolor='black', linewidth=0.8)

        # Add count labels on bars (only if percentage > 3%)
        for j, (bar, bias_type) in enumerate(zip(bars, bias_types)):
            height = bar.get_height()
            if height > 3:  # Only show label if bar is tall enough
                count = data_by_bias[bias_type]['counts'][label]
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Percentage of Biased Responses', fontsize=12, fontweight='bold')
    ax.set_xlabel('Bias Type (Hint Template)', fontsize=12, fontweight='bold')
    ax.set_title('Global Faithfulness Distribution by Bias Type\n'
                f'Total Biased Responses: {len(biased_results)}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(bias_types, rotation=45, ha='right')
    ax.legend(title='Classification', loc='upper right', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bias-wise global faithfulness plot saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_steering_tuning_results(
    evaluation_results: Dict[Tuple[int, float], Dict[str, Any]],
    layers_to_test: List[int],
    coefficients: List[float],
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title_suffix: str = ""
) -> None:
    """
    Plot steering tuning results showing faithfulness improvement across layers and coefficients.

    Creates dual subplots comparing positive vs negative coefficient effects on faithfulness
    improvement rate across different model layers.

    Args:
        evaluation_results: Dict mapping (layer, coeff) tuples to result dicts with 'improvement_rate'
        layers_to_test: List of layer indices that were tested
        coefficients: List of coefficient values that were tested
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        title_suffix: Optional suffix to add to the plot title (e.g., dataset name)

    Returns:
        None

    Example:
        >>> evaluation_results = {
        ...     (15, 0.75): {'improvement_rate': 0.45, ...},
        ...     (15, -0.75): {'improvement_rate': 0.12, ...},
        ...     (25, 5.0): {'improvement_rate': 0.78, ...},
        ... }
        >>> plot_steering_tuning_results(
        ...     evaluation_results,
        ...     layers_to_test=[15, 25],
        ...     coefficients=[0.75, -0.75, 5, -5],
        ...     save_path='plots/steering_tuning.png'
        ... )
    """
    setup_plot_style()

    # Separate positive and negative coefficients
    pos_coeffs = sorted([c for c in coefficients if c > 0])
    neg_coeffs = sorted([c for c in coefficients if c < 0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot positive coefficients
    if pos_coeffs:
        for coeff in pos_coeffs:
            improvement_rates = []
            for layer in layers_to_test:
                if (layer, coeff) in evaluation_results:
                    # Handle both old and new structure
                    result = evaluation_results[(layer, coeff)]
                    if 'improvement_metrics' in result:
                        improvement_rates.append(result['improvement_metrics']['improvement_rate'])
                    else:
                        improvement_rates.append(result.get('improvement_rate', 0))
                else:
                    improvement_rates.append(0)
            ax1.plot(layers_to_test, improvement_rates, marker='o', label=f'Coeff +{coeff:.1f}', linewidth=2, markersize=6)

        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Improvement Rate', fontsize=12)
        ax1.set_title('Faithfulness Improvement vs Layer (Positive Coefficients)', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

    # Plot negative coefficients
    if neg_coeffs:
        for coeff in neg_coeffs:
            improvement_rates = []
            for layer in layers_to_test:
                if (layer, coeff) in evaluation_results:
                    # Handle both old and new structure
                    result = evaluation_results[(layer, coeff)]
                    if 'improvement_metrics' in result:
                        improvement_rates.append(result['improvement_metrics']['improvement_rate'])
                    else:
                        improvement_rates.append(result.get('improvement_rate', 0))
                else:
                    improvement_rates.append(0)
            ax2.plot(layers_to_test, improvement_rates, marker='o', label=f'Coeff {coeff:.1f}', linewidth=2, markersize=6)

        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Improvement Rate', fontsize=12)
        ax2.set_title('Faithfulness Improvement vs Layer (Negative Coefficients)', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

    # Add main title
    main_title = 'Steering Tuning Results - Faithfulness Improvement by Layer and Coefficient'
    if title_suffix:
        main_title += f'\n{title_suffix}'
    plt.suptitle(main_title, fontsize=14)
    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Steering tuning plot saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_steered_faithfulness_comparison(
    pre_steering_distribution: Dict[str, int],
    post_steering_distribution: Dict[str, int],
    layer: int,
    coefficient: float,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title_suffix: str = ""
) -> None:
    """
    Plot comparison of faithfulness classification distributions before and after steering.

    Shows side-by-side bar charts comparing the relative frequencies of each faithfulness
    classification (correct, faithful, unfaithful, hint-induced error) before and after
    applying steering vectors.

    Args:
        pre_steering_distribution: Dict mapping classification labels to counts before steering
        post_steering_distribution: Dict mapping classification labels to counts after steering
        layer: Layer index used for steering
        coefficient: Steering coefficient used
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        title_suffix: Optional suffix for plot title

    Example:
        >>> pre_dist = {'unfaithful': 100, 'faithful': 0, 'correct': 0, 'hint-induced error': 0}
        >>> post_dist = {'unfaithful': 50, 'faithful': 30, 'correct': 15, 'hint-induced error': 5}
        >>> plot_steered_faithfulness_comparison(pre_dist, post_dist, layer=15, coefficient=5.0)
    """
    setup_plot_style()

    # Define classification order and colors
    label_order = ['correct', 'faithful', 'unfaithful', 'hint-induced error', 'error']
    label_colors = {
        'correct': '#2E86AB',           # Blue - correct answer
        'faithful': '#A23B72',          # Purple - faithful reasoning
        'unfaithful': '#F18F01',        # Orange - unfaithful reasoning
        'hint-induced error': '#C73E1D', # Red - confused by hint
        'error': '#888888'              # Gray - annotation errors
    }

    # Calculate totals
    total_pre = sum(pre_steering_distribution.values())
    total_post = sum(post_steering_distribution.values())

    # Calculate percentages for each label
    pre_percentages = []
    post_percentages = []
    existing_labels = []

    for label in label_order:
        pre_count = pre_steering_distribution.get(label, 0)
        post_count = post_steering_distribution.get(label, 0)

        # Only include labels that appear in at least one distribution
        if pre_count > 0 or post_count > 0:
            existing_labels.append(label)
            pre_percentages.append((pre_count / total_pre * 100) if total_pre > 0 else 0)
            post_percentages.append((post_count / total_post * 100) if total_post > 0 else 0)

    # Create the plot with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(existing_labels))
    width = 0.6

    # Pre-steering distribution (left subplot)
    colors_pre = [label_colors[label] for label in existing_labels]
    bars1 = ax1.bar(x, pre_percentages, width, color=colors_pre, alpha=0.8, edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Faithfulness Classification', fontsize=12, fontweight='bold')
    ax1.set_title(f'Pre-Steering Distribution\nTotal: {total_pre}',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(existing_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels on bars
    for bar, percentage, label in zip(bars1, pre_percentages, existing_labels):
        height = bar.get_height()
        count = pre_steering_distribution.get(label, 0)
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{percentage:.1f}%\n({count})',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Post-steering distribution (right subplot)
    colors_post = [label_colors[label] for label in existing_labels]
    bars2 = ax2.bar(x, post_percentages, width, color=colors_post, alpha=0.8, edgecolor='black', linewidth=0.8)

    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Faithfulness Classification', fontsize=12, fontweight='bold')
    ax2.set_title(f'Post-Steering Distribution\nLayer {layer}, Coefficient {coefficient:+.1f}\nTotal: {total_post}',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(existing_labels, rotation=45, ha='right')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels on bars
    for bar, percentage, label in zip(bars2, post_percentages, existing_labels):
        height = bar.get_height()
        count = post_steering_distribution.get(label, 0)
        ax2.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{percentage:.1f}%\n({count})',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Overall title
    title = f'Steering Effect on Faithfulness Distribution{title_suffix}'
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Steered faithfulness comparison plot saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_text_length_histogram(
    data: List[Dict[str, Any]],
    text_field: str,
    threshold: Optional[int] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title_suffix: str = ""
) -> None:
    """
    Plot histogram of word lengths for a text field across all records.

    Args:
        data: List of dictionaries (records from JSONL)
        text_field: Name of the field containing text to analyze
        threshold: Optional threshold to display as vertical line
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        title_suffix: Optional suffix for plot title

    Example:
        >>> data = load_jsonl("data.jsonl")
        >>> plot_text_length_histogram(
        ...     data,
        ...     text_field="hinted_generated_text",
        ...     threshold=150,
        ...     save_path="plots/length_histogram.png"
        ... )
    """
    from src.text_utils import count_words

    setup_plot_style()

    # Calculate word lengths
    word_lengths = []
    for record in data:
        text = record.get(text_field, '')
        if text:
            word_lengths.append(count_words(text))

    if not word_lengths:
        print(f"No valid texts found in field '{text_field}'")
        return

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 7))

    n_bins = min(50, max(20, len(word_lengths) // 10))
    counts, bins, patches = ax.hist(word_lengths, bins=n_bins, alpha=0.7,
                                     color='steelblue', edgecolor='black', linewidth=0.5)

    # Add threshold line if specified
    if threshold is not None:
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2.5,
                   label=f'Threshold: {threshold} words')

        # Shade regions and calculate percentages
        for patch in patches:
            if patch.get_x() + patch.get_width() / 2 < threshold:
                patch.set_facecolor('lightgreen')
                patch.set_alpha(0.7)

        # Calculate percentages
        below_threshold = sum(1 for length in word_lengths if length <= threshold)
        above_threshold = len(word_lengths) - below_threshold
        pct_below = (below_threshold / len(word_lengths)) * 100
        pct_above = (above_threshold / len(word_lengths)) * 100

        # Add percentage labels in the middle of each region
        y_max = ax.get_ylim()[1]
        x_max = max(word_lengths)

        # Left region (below threshold) - green area
        x_left = threshold / 2
        ax.text(x_left, y_max * 0.75,
                f'Below threshold\n{below_threshold} texts\n({pct_below:.1f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2))

        # Right region (above threshold) - red area
        x_right = threshold + (x_max - threshold) / 2
        ax.text(x_right, y_max * 0.75,
                f'Above threshold\n{above_threshold} texts\n({pct_above:.1f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=2))

    # Calculate and display statistics
    mean_length = np.mean(word_lengths)
    median_length = np.median(word_lengths)

    ax.axvline(x=mean_length, color='blue', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Mean: {mean_length:.1f} words')
    ax.axvline(x=median_length, color='purple', linestyle='-.', linewidth=2, alpha=0.7,
               label=f'Median: {median_length:.1f} words')

    # Labels and title
    ax.set_xlabel('Word Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')

    title = f'Distribution of Text Lengths: {text_field}'
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text = f'Total: {len(word_lengths)}\n'
    stats_text += f'Min: {min(word_lengths)}\n'
    stats_text += f'Max: {max(word_lengths)}\n'
    stats_text += f'Std: {np.std(word_lengths):.1f}'

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Length histogram saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_text_length_sorted_bar(
    data: List[Dict[str, Any]],
    text_field: str,
    threshold: Optional[int] = None,
    max_items: int = 100,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title_suffix: str = ""
) -> None:
    """
    Plot sorted bar chart showing word length for each record.

    Each bar represents one record, sorted by increasing word length.
    Useful for identifying specific short/long texts.

    Args:
        data: List of dictionaries (records from JSONL)
        text_field: Name of the field containing text to analyze
        threshold: Optional threshold to display as horizontal line
        max_items: Maximum number of items to plot (default: 100)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        title_suffix: Optional suffix for plot title

    Example:
        >>> data = load_jsonl("data.jsonl")
        >>> plot_text_length_sorted_bar(
        ...     data,
        ...     text_field="hinted_generated_text",
        ...     threshold=150,
        ...     save_path="plots/length_sorted.png"
        ... )
    """
    from src.text_utils import count_words

    setup_plot_style()

    # Calculate word lengths with indices
    length_data = []
    for idx, record in enumerate(data):
        text = record.get(text_field, '')
        if text:
            length_data.append((idx, count_words(text)))

    if not length_data:
        print(f"No valid texts found in field '{text_field}'")
        return

    # Sort by word length
    length_data.sort(key=lambda x: x[1])

    # Limit to max_items if needed
    if len(length_data) > max_items:
        print(f"Showing {max_items} of {len(length_data)} items")
        # Sample evenly across the range
        step = len(length_data) // max_items
        length_data = length_data[::step][:max_items]

    indices, lengths = zip(*length_data)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(lengths))

    # Color bars based on threshold
    if threshold is not None:
        colors = ['lightgreen' if l <= threshold else 'lightcoral' for l in lengths]
    else:
        colors = 'steelblue'

    bars = ax.bar(x_pos, lengths, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)

    # Add threshold line if specified
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold} words')
        ax.legend()

    # Labels and title
    ax.set_xlabel('Text Index (sorted by length)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Word Count', fontsize=12, fontweight='bold')

    title = f'Sorted Word Lengths: {text_field}'
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text = f'Showing: {len(lengths)} texts\n'
    stats_text += f'Min: {min(lengths)}\n'
    stats_text += f'Max: {max(lengths)}\n'
    stats_text += f'Median: {np.median(lengths):.1f}'

    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

    # Reduce x-axis tick density for readability
    if len(x_pos) > 20:
        ax.set_xticks(x_pos[::len(x_pos)//10])

    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sorted length plot saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_filtering_comparison(
    original_data: List[Dict[str, Any]],
    filtered_data: List[Dict[str, Any]],
    text_field: str,
    threshold: int,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot side-by-side comparison of length distributions before and after filtering.

    Args:
        original_data: Original dataset before filtering
        filtered_data: Filtered dataset after applying threshold
        text_field: Name of the field containing text
        threshold: Threshold used for filtering
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot

    Example:
        >>> original = load_jsonl("original.jsonl")
        >>> filtered = load_jsonl("filtered.jsonl")
        >>> plot_filtering_comparison(
        ...     original, filtered,
        ...     text_field="hinted_generated_text",
        ...     threshold=150,
        ...     save_path="plots/filtering_comparison.png"
        ... )
    """
    from src.text_utils import count_words

    setup_plot_style()

    # Calculate word lengths for both datasets
    original_lengths = [count_words(r.get(text_field, ''))
                       for r in original_data if r.get(text_field)]
    filtered_lengths = [count_words(r.get(text_field, ''))
                       for r in filtered_data if r.get(text_field)]

    # Create side-by-side histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Original data histogram
    n_bins = min(50, max(20, len(original_lengths) // 10))
    ax1.hist(original_lengths, bins=n_bins, alpha=0.7,
             color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold}')
    ax1.axvline(x=np.median(original_lengths), color='purple',
                linestyle='-.', linewidth=2, alpha=0.7,
                label=f'Median: {np.median(original_lengths):.1f}')

    ax1.set_xlabel('Word Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Original Data\n(n={len(original_lengths)})',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Filtered data histogram
    n_bins_filtered = min(50, max(20, len(filtered_lengths) // 10))
    ax2.hist(filtered_lengths, bins=n_bins_filtered, alpha=0.7,
             color='lightgreen', edgecolor='black', linewidth=0.5)
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold}')
    ax2.axvline(x=np.median(filtered_lengths), color='purple',
                linestyle='-.', linewidth=2, alpha=0.7,
                label=f'Median: {np.median(filtered_lengths):.1f}')

    ax2.set_xlabel('Word Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Filtered Data\n(n={len(filtered_lengths)}, {len(filtered_lengths)/len(original_lengths)*100:.1f}% retained)',
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Overall title
    fig.suptitle(f'Filtering Effect on {text_field} Length Distribution',
                 fontsize=15, fontweight='bold')

    plt.tight_layout()

    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Filtering comparison plot saved to {save_path}")

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