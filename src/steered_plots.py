"""
steered_plots.py

Plotting functions for steered global faithfulness evaluation.

This module provides visualization functions for:
- Heatmaps showing steering effectiveness across layers and coefficients
- Best configuration breakdown
- Transformation rates across layers
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os


# =============================================================================
# PLOT 1: STEERING HEATMAPS (2×2 GRID)
# =============================================================================

def plot_steering_heatmaps(all_configs: List[Dict[str, Any]],
                          subject: str,
                          hint_template: str,
                          save_path: str):
    """
    Create 2×2 heatmap grid showing steering effectiveness.

    Grid layout:
    - Top-left: Positive on Unfaithful (unfaithful→faithful %)
    - Top-right: Positive on Faithful (faithful→unfaithful % - side effects)
    - Bottom-left: Negative on Faithful (faithful→unfaithful %)
    - Bottom-right: Negative on Unfaithful (unfaithful→faithful % - side effects)

    Args:
        all_configs: List of all configuration results
        subject: Subject name
        hint_template: Hint template name
        save_path: Path to save plot
    """
    # Extract unique layers and coefficients
    layers = sorted(set(c['layer'] for c in all_configs))
    coeffs = sorted(set(c['coefficient_magnitude'] for c in all_configs))

    # Initialize heatmap data (layers × coefficients)
    pos_unfaith_data = np.zeros((len(layers), len(coeffs)))
    pos_faith_data = np.zeros((len(layers), len(coeffs)))
    neg_faith_data = np.zeros((len(layers), len(coeffs)))
    neg_unfaith_data = np.zeros((len(layers), len(coeffs)))

    # Fill heatmap data
    for config in all_configs:
        layer_idx = layers.index(config['layer'])
        coeff_idx = coeffs.index(config['coefficient_magnitude'])

        # Positive on Unfaithful
        pos_unfaith_data[layer_idx, coeff_idx] = config['positive_on_unfaithful']['transitions'].get(
            'unfaithful_to_faithful', {}
        ).get('rate', 0) * 100

        # Positive on Faithful (side effects)
        pos_faith_data[layer_idx, coeff_idx] = config['positive_on_faithful']['transitions'].get(
            'faithful_to_unfaithful', {}
        ).get('rate', 0) * 100

        # Negative on Faithful
        neg_faith_data[layer_idx, coeff_idx] = config['negative_on_faithful']['transitions'].get(
            'faithful_to_unfaithful', {}
        ).get('rate', 0) * 100

        # Negative on Unfaithful (side effects)
        neg_unfaith_data[layer_idx, coeff_idx] = config['negative_on_unfaithful']['transitions'].get(
            'unfaithful_to_faithful', {}
        ).get('rate', 0) * 100

    # Create figure with 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Steering Effectiveness Heatmaps\n{subject.replace("_", " ").title()} - {hint_template}',
                 fontsize=16, fontweight='bold')

    # Plot A: Positive on Unfaithful
    sns.heatmap(pos_unfaith_data, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=coeffs, yticklabels=layers, ax=axes[0, 0],
                vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
    axes[0, 0].set_title('A) Positive Steering on Unfaithful\n(unfaithful → faithful %)',
                         fontweight='bold')
    axes[0, 0].set_xlabel('Coefficient Magnitude')
    axes[0, 0].set_ylabel('Layer')

    # Plot B: Positive on Faithful (side effects - want LOW)
    sns.heatmap(pos_faith_data, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=coeffs, yticklabels=layers, ax=axes[0, 1],
                vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
    axes[0, 1].set_title('B) Positive Steering Side Effects\n(faithful → unfaithful % - want LOW)',
                         fontweight='bold')
    axes[0, 1].set_xlabel('Coefficient Magnitude')
    axes[0, 1].set_ylabel('Layer')

    # Plot C: Negative on Faithful
    sns.heatmap(neg_faith_data, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=coeffs, yticklabels=layers, ax=axes[1, 0],
                vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
    axes[1, 0].set_title('C) Negative Steering on Faithful\n(faithful → unfaithful %)',
                         fontweight='bold')
    axes[1, 0].set_xlabel('Coefficient Magnitude')
    axes[1, 0].set_ylabel('Layer')

    # Plot D: Negative on Unfaithful (side effects - want LOW)
    sns.heatmap(neg_unfaith_data, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=coeffs, yticklabels=layers, ax=axes[1, 1],
                vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
    axes[1, 1].set_title('D) Negative Steering Side Effects\n(unfaithful → faithful % - want LOW)',
                         fontweight='bold')
    axes[1, 1].set_xlabel('Coefficient Magnitude')
    axes[1, 1].set_ylabel('Layer')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved heatmaps to: {save_path}")


# =============================================================================
# PLOT 2: BEST CONFIG BREAKDOWN
# =============================================================================

def plot_best_config_breakdown(best_positive: Dict[str, Any],
                               best_negative: Dict[str, Any],
                               save_path: str):
    """
    Bar chart showing all transformation rates for best configs.

    Args:
        best_positive: Best positive steering config
        best_negative: Best negative steering config
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Best Configuration Breakdown - All Transformation Rates',
                 fontsize=16, fontweight='bold')

    # Extract data for positive steering
    pos_config = best_positive['config']
    pos_transitions = pos_config['positive_on_unfaithful']['transitions']

    pos_labels = []
    pos_rates = []
    pos_colors = []

    # Success (green)
    pos_labels.append('→ Faithful\n(SUCCESS)')
    pos_rates.append(pos_transitions.get('unfaithful_to_faithful', {}).get('rate', 0) * 100)
    pos_colors.append('#2ecc71')

    # Failed (red)
    pos_labels.append('→ Unfaithful\n(FAILED)')
    pos_rates.append(pos_transitions.get('unfaithful_to_unfaithful', {}).get('rate', 0) * 100)
    pos_colors.append('#e74c3c')

    # Contamination (orange)
    pos_labels.append('→ Correct\n(CONTAM.)')
    pos_rates.append(pos_transitions.get('unfaithful_to_correct', {}).get('rate', 0) * 100)
    pos_colors.append('#f39c12')

    pos_labels.append('→ Hint Error\n(CONTAM.)')
    pos_rates.append(pos_transitions.get('unfaithful_to_hint_error', {}).get('rate', 0) * 100)
    pos_colors.append('#f39c12')

    pos_labels.append('→ Incomplete\n(CONTAM.)')
    pos_rates.append(pos_transitions.get('unfaithful_to_incomplete', {}).get('rate', 0) * 100)
    pos_colors.append('#f39c12')

    # Plot positive
    axes[0].barh(pos_labels, pos_rates, color=pos_colors)
    axes[0].set_xlabel('Rate (%)')
    axes[0].set_title(f'Best Positive: Layer {best_positive["layer"]}, Coeff +{best_positive["coefficient"]}\n' +
                      f'Score: {best_positive["score"]:.3f} (Success: {best_positive["success_rate"]:.1%}, ' +
                      f'Side Effects: {best_positive["side_effects_rate"]:.1%})',
                      fontweight='bold')
    axes[0].set_xlim(0, 100)
    axes[0].grid(axis='x', alpha=0.3)

    # Extract data for negative steering
    neg_config = best_negative['config']
    neg_transitions = neg_config['negative_on_faithful']['transitions']

    neg_labels = []
    neg_rates = []
    neg_colors = []

    # Success (green)
    neg_labels.append('→ Unfaithful\n(SUCCESS)')
    neg_rates.append(neg_transitions.get('faithful_to_unfaithful', {}).get('rate', 0) * 100)
    neg_colors.append('#2ecc71')

    # Failed (red)
    neg_labels.append('→ Faithful\n(FAILED)')
    neg_rates.append(neg_transitions.get('faithful_to_faithful', {}).get('rate', 0) * 100)
    neg_colors.append('#e74c3c')

    # Contamination (orange)
    neg_labels.append('→ Correct\n(CONTAM.)')
    neg_rates.append(neg_transitions.get('faithful_to_correct', {}).get('rate', 0) * 100)
    neg_colors.append('#f39c12')

    neg_labels.append('→ Hint Error\n(CONTAM.)')
    neg_rates.append(neg_transitions.get('faithful_to_hint_error', {}).get('rate', 0) * 100)
    neg_colors.append('#f39c12')

    neg_labels.append('→ Incomplete\n(CONTAM.)')
    neg_rates.append(neg_transitions.get('faithful_to_incomplete', {}).get('rate', 0) * 100)
    neg_colors.append('#f39c12')

    # Plot negative
    axes[1].barh(neg_labels, neg_rates, color=neg_colors)
    axes[1].set_xlabel('Rate (%)')
    axes[1].set_title(f'Best Negative: Layer {best_negative["layer"]}, Coeff {best_negative["coefficient"]}\n' +
                      f'Score: {best_negative["score"]:.3f} (Success: {best_negative["success_rate"]:.1%}, ' +
                      f'Side Effects: {best_negative["side_effects_rate"]:.1%})',
                      fontweight='bold')
    axes[1].set_xlim(0, 100)
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved best config breakdown to: {save_path}")


# =============================================================================
# PLOT 3: TRANSFORMATION RATES ACROSS LAYERS (PER-COEFFICIENT)
# =============================================================================

def plot_transformation_rates_by_layer(all_configs: List[Dict[str, Any]], save_dir: str):
    """
    Create 5 plots (one per coefficient magnitude).
    Each plot has 4 subplots showing transformation rates across layers.

    Args:
        all_configs: List of all configuration results
        save_dir: Directory to save plots
    """
    coeffs = sorted(set(c['coefficient_magnitude'] for c in all_configs))
    layers = sorted(set(c['layer'] for c in all_configs))

    for coeff in coeffs:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Transformation Rates Across Layers (Coefficient = ±{coeff})',
                     fontsize=16, fontweight='bold')

        # Filter configs for this coefficient
        coeff_configs = [c for c in all_configs if c['coefficient_magnitude'] == coeff]

        # Subplot A: Unfaithful + Positive
        _plot_transitions_subplot(
            coeff_configs, layers, 'positive_on_unfaithful', 'unfaithful',
            axes[0, 0], f'Unfaithful Origin + Positive Steering (coeff = +{coeff})'
        )

        # Subplot B: Unfaithful + Negative
        _plot_transitions_subplot(
            coeff_configs, layers, 'negative_on_unfaithful', 'unfaithful',
            axes[0, 1], f'Unfaithful Origin + Negative Steering (coeff = -{coeff})'
        )

        # Subplot C: Faithful + Positive
        _plot_transitions_subplot(
            coeff_configs, layers, 'positive_on_faithful', 'faithful',
            axes[1, 0], f'Faithful Origin + Positive Steering (coeff = +{coeff})'
        )

        # Subplot D: Faithful + Negative
        _plot_transitions_subplot(
            coeff_configs, layers, 'negative_on_faithful', 'faithful',
            axes[1, 1], f'Faithful Origin + Negative Steering (coeff = -{coeff})'
        )

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'steered_global_layers_coeff_{coeff}.png')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved layer comparison for coeff ±{coeff} to: {save_path}")


def _plot_transitions_subplot(configs: List[Dict[str, Any]],
                              layers: List[int],
                              group_name: str,
                              original_state: str,
                              ax,
                              title: str):
    """
    Helper function to plot transitions for one subplot.

    Args:
        configs: List of configs for this coefficient
        layers: List of layer numbers
        group_name: Name of group to plot
        original_state: 'faithful' or 'unfaithful'
        ax: Matplotlib axis
        title: Subplot title
    """
    # Extract transition rates across layers
    if original_state == 'unfaithful':
        transition_names = [
            'unfaithful_to_faithful',
            'unfaithful_to_unfaithful',
            'unfaithful_to_correct',
            'unfaithful_to_hint_error',
            'unfaithful_to_incomplete'
        ]
        labels = ['→ Faithful', '→ Unfaithful', '→ Correct', '→ Hint Error', '→ Incomplete']
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#e67e22', '#95a5a6']
    else:  # faithful
        transition_names = [
            'faithful_to_unfaithful',
            'faithful_to_faithful',
            'faithful_to_correct',
            'faithful_to_hint_error',
            'faithful_to_incomplete'
        ]
        labels = ['→ Unfaithful', '→ Faithful', '→ Correct', '→ Hint Error', '→ Incomplete']
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#e67e22', '#95a5a6']

    # Build data for each transition
    for transition_name, label, color in zip(transition_names, labels, colors):
        rates = []
        for layer in layers:
            # Find config for this layer
            config = next((c for c in configs if c['layer'] == layer), None)
            if config and group_name in config:
                rate = config[group_name]['transitions'].get(transition_name, {}).get('rate', 0) * 100
                rates.append(rate)
            else:
                rates.append(0)

        ax.plot(layers, rates, marker='o', label=label, color=color, linewidth=2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Rate (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    ax.set_xticks(layers)
