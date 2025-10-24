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
                          correctness_group: str,
                          heatmap_type: str,
                          save_path: str):
    """
    Create 2×2 heatmap grid showing steering effectiveness.

    Creates 4 heatmaps total for each correctness group (Correct/Wrong):
    - 1 TRANSITIONS heatmap (faithfulness changes)
    - 1 NO-CHANGE heatmap (resistance/preservation)

    Args:
        all_configs: List of all configuration results
        subject: Subject name
        hint_template: Hint template name
        correctness_group: 'correct' or 'wrong'
        heatmap_type: 'transitions' or 'no_change'
        save_path: Path to save plot
    """
    # Extract unique layers and coefficients
    layers = sorted(set(c['layer'] for c in all_configs))
    coeffs = sorted(set(c['coefficient_magnitude'] for c in all_configs))

    # Determine initial states based on correctness group
    if correctness_group == 'correct':
        faithful_state = 'CF'  # Correct + Faithful
        unfaithful_state = 'CU'  # Correct + Unfaithful
        group_title = 'Initially CORRECT Answer'
    else:  # 'wrong'
        faithful_state = 'WF'  # Wrong + Faithful
        unfaithful_state = 'WU'  # Wrong + Unfaithful
        group_title = 'Initially WRONG Answer'

    # Initialize heatmap data (layers × coefficients)
    # Each quarter shows ONE specific transition
    A_data = np.zeros((len(layers), len(coeffs)))  # Top-left
    B_data = np.zeros((len(layers), len(coeffs)))  # Top-right
    C_data = np.zeros((len(layers), len(coeffs)))  # Bottom-left
    D_data = np.zeros((len(layers), len(coeffs)))  # Bottom-right

    # Fill heatmap data based on type
    for config in all_configs:
        layer_idx = layers.index(config['layer'])
        coeff_idx = coeffs.index(config['coefficient_magnitude'])

        if heatmap_type == 'transitions':
            # TRANSITIONS HEATMAP: Faithfulness CHANGES (U↔F or F↔U)
            # A) INTENDED: Repair (U→F with +steering)
            pos_u_group = f'positive_on_{unfaithful_state}'
            A_data[layer_idx, coeff_idx] = config[pos_u_group]['transitions'].get('to_same_answer_faithful', {}).get('rate', 0) * 100

            # B) BAD: Degradation (F→U with +steering)
            pos_f_group = f'positive_on_{faithful_state}'
            B_data[layer_idx, coeff_idx] = config[pos_f_group]['transitions'].get('to_same_answer_unfaithful', {}).get('rate', 0) * 100

            # C) INTENDED: Degrade (F→U with -steering)
            neg_f_group = f'negative_on_{faithful_state}'
            C_data[layer_idx, coeff_idx] = config[neg_f_group]['transitions'].get('to_same_answer_unfaithful', {}).get('rate', 0) * 100

            # D) SIDE EFFECT: Improvement (U→F with -steering)
            neg_u_group = f'negative_on_{unfaithful_state}'
            D_data[layer_idx, coeff_idx] = config[neg_u_group]['transitions'].get('to_same_answer_faithful', {}).get('rate', 0) * 100

        else:  # 'no_change'
            # NO-CHANGE HEATMAP: Faithfulness STAYS SAME (F→F or U→U)
            # A) PRESERVATION (F→F with +steering)
            pos_f_group = f'positive_on_{faithful_state}'
            A_data[layer_idx, coeff_idx] = config[pos_f_group]['transitions'].get('to_same_answer_faithful', {}).get('rate', 0) * 100

            # B) RESISTANCE (F→F with -steering)
            neg_f_group = f'negative_on_{faithful_state}'
            B_data[layer_idx, coeff_idx] = config[neg_f_group]['transitions'].get('to_same_answer_faithful', {}).get('rate', 0) * 100

            # C) RESISTANCE (U→U with +steering)
            pos_u_group = f'positive_on_{unfaithful_state}'
            C_data[layer_idx, coeff_idx] = config[pos_u_group]['transitions'].get('to_same_answer_unfaithful', {}).get('rate', 0) * 100

            # D) MAINTENANCE (U→U with -steering)
            neg_u_group = f'negative_on_{unfaithful_state}'
            D_data[layer_idx, coeff_idx] = config[neg_u_group]['transitions'].get('to_same_answer_unfaithful', {}).get('rate', 0) * 100

    # Create figure with 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    if heatmap_type == 'transitions':
        main_title = f'FAITHFULNESS TRANSITIONS: {group_title}\n{subject.replace("_", " ").title()} - {hint_template}'

        # A) INTENDED: Repair
        sns.heatmap(A_data, annot=True, fmt='.1f', cmap='Greens',
                    xticklabels=coeffs, yticklabels=layers, ax=axes[0, 0],
                    vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
        axes[0, 0].set_title(f'A) INTENDED: Repair\n{unfaithful_state}→{faithful_state} with +steering',
                             fontweight='bold', fontsize=12)

        # B) BAD: Degradation
        sns.heatmap(B_data, annot=True, fmt='.1f', cmap='Reds',
                    xticklabels=coeffs, yticklabels=layers, ax=axes[0, 1],
                    vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
        axes[0, 1].set_title(f'B) BAD: Unintended Degradation\n{faithful_state}→{unfaithful_state} with +steering',
                             fontweight='bold', fontsize=12)

        # C) INTENDED: Degrade
        sns.heatmap(C_data, annot=True, fmt='.1f', cmap='Purples',
                    xticklabels=coeffs, yticklabels=layers, ax=axes[1, 0],
                    vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
        axes[1, 0].set_title(f'C) INTENDED: Degrade\n{faithful_state}→{unfaithful_state} with -steering',
                             fontweight='bold', fontsize=12)

        # D) SIDE EFFECT: Improvement
        sns.heatmap(D_data, annot=True, fmt='.1f', cmap='YlOrBr',
                    xticklabels=coeffs, yticklabels=layers, ax=axes[1, 1],
                    vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
        axes[1, 1].set_title(f'D) SIDE EFFECT: Unexpected Improvement\n{unfaithful_state}→{faithful_state} with -steering',
                             fontweight='bold', fontsize=12)

    else:  # 'no_change'
        main_title = f'NO CHANGE (Resistance/Preservation): {group_title}\n{subject.replace("_", " ").title()} - {hint_template}'

        # A) PRESERVATION
        sns.heatmap(A_data, annot=True, fmt='.1f', cmap='Greens',
                    xticklabels=coeffs, yticklabels=layers, ax=axes[0, 0],
                    vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
        axes[0, 0].set_title(f'A) PRESERVATION\n{faithful_state}→{faithful_state} with +steering',
                             fontweight='bold', fontsize=12)

        # B) RESISTANCE
        sns.heatmap(B_data, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=coeffs, yticklabels=layers, ax=axes[0, 1],
                    vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
        axes[0, 1].set_title(f'B) RESISTANCE\n{faithful_state}→{faithful_state} with -steering',
                             fontweight='bold', fontsize=12)

        # C) RESISTANCE
        sns.heatmap(C_data, annot=True, fmt='.1f', cmap='Oranges',
                    xticklabels=coeffs, yticklabels=layers, ax=axes[1, 0],
                    vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
        axes[1, 0].set_title(f'C) RESISTANCE\n{unfaithful_state}→{unfaithful_state} with +steering',
                             fontweight='bold', fontsize=12)

        # D) MAINTENANCE
        sns.heatmap(D_data, annot=True, fmt='.1f', cmap='Purples',
                    xticklabels=coeffs, yticklabels=layers, ax=axes[1, 1],
                    vmin=0, vmax=100, cbar_kws={'label': 'Rate (%)'})
        axes[1, 1].set_title(f'D) MAINTENANCE\n{unfaithful_state}→{unfaithful_state} with -steering',
                             fontweight='bold', fontsize=12)

    fig.suptitle(main_title, fontsize=16, fontweight='bold')

    # Add axis labels
    for ax in axes.flat:
        ax.set_xlabel('Coefficient Magnitude', fontsize=10)
        ax.set_ylabel('Layer', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved {correctness_group} {heatmap_type} heatmaps to: {save_path}")


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

    pos_labels.append('→ Error\n(CONTAM.)')
    pos_rates.append(pos_transitions.get('unfaithful_to_error', {}).get('rate', 0) * 100)
    pos_colors.append('#95a5a6')

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

    neg_labels.append('→ Error\n(CONTAM.)')
    neg_rates.append(neg_transitions.get('faithful_to_error', {}).get('rate', 0) * 100)
    neg_colors.append('#95a5a6')

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
# PLOT 3: TRANSFORMATION RATES ACROSS LAYERS (PER-COEFFICIENT, PER-INITIAL-STATE)
# =============================================================================

def plot_transformation_rates_by_layer(all_configs: List[Dict[str, Any]],
                                      correctness_group: str,
                                      save_dir: str,
                                      subject: str = None,
                                      hint_template: str = None):
    """
    Create 2×2 line plots showing transformation rates across layers.
    One figure per coefficient magnitude, with 4 subplots showing all combinations.

    Layout per coefficient:
    ┌─────────────────────────────────┬─────────────────────────────────┐
    │ Top-left: +coeff on CU/WU       │ Top-right: +coeff on CF/WF      │
    │ (positive on unfaithful)        │ (positive on faithful)          │
    ├─────────────────────────────────┼─────────────────────────────────┤
    │ Bottom-left: -coeff on CU/WU    │ Bottom-right: -coeff on CF/WF   │
    │ (negative on unfaithful)        │ (negative on faithful)          │
    └─────────────────────────────────┴─────────────────────────────────┘

    Args:
        all_configs: List of all configuration results
        correctness_group: 'correct' or 'wrong'
        save_dir: Directory to save plots
        subject: Subject name (extracted from configs if not provided)
        hint_template: Hint template name (extracted from configs if not provided)
    """
    coeffs = sorted(set(c['coefficient_magnitude'] for c in all_configs))
    layers = sorted(set(c['layer'] for c in all_configs))

    # Determine states based on correctness group
    if correctness_group == 'correct':
        faithful_state = 'CF'
        unfaithful_state = 'CU'
        group_title = 'CORRECT Answer Group'
    else:  # 'wrong'
        faithful_state = 'WF'
        unfaithful_state = 'WU'
        group_title = 'WRONG Answer Group'

    # Extract subject and hint_template from configs if not provided
    if not subject or not hint_template:
        if all_configs:
            subject = subject or all_configs[0].get('subject', 'unknown_subject')
            hint_template = hint_template or all_configs[0].get('hint_template', 'unknown_template')

    for coeff in coeffs:
        # Create 2×2 grid
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        fig.suptitle(f'Transition Rates Across Layers - {group_title}\n' +
                     f'{subject.replace("_", " ").title()} - {hint_template}\n' +
                     f'Coefficient Magnitude = {coeff}',
                     fontsize=16, fontweight='bold')

        # Filter configs for this coefficient
        coeff_configs = [c for c in all_configs if c['coefficient_magnitude'] == coeff]

        # Top-left: Positive on Unfaithful (CU or WU)
        _plot_transitions_subplot_stratified(
            coeff_configs, layers, f'positive_on_{unfaithful_state}', unfaithful_state,
            axes[0, 0], f'Positive Steering on {unfaithful_state}\n(coeff = +{coeff})'
        )

        # Top-right: Positive on Faithful (CF or WF)
        _plot_transitions_subplot_stratified(
            coeff_configs, layers, f'positive_on_{faithful_state}', faithful_state,
            axes[0, 1], f'Positive Steering on {faithful_state}\n(coeff = +{coeff})'
        )

        # Bottom-left: Negative on Unfaithful (CU or WU)
        _plot_transitions_subplot_stratified(
            coeff_configs, layers, f'negative_on_{unfaithful_state}', unfaithful_state,
            axes[1, 0], f'Negative Steering on {unfaithful_state}\n(coeff = -{coeff})'
        )

        # Bottom-right: Negative on Faithful (CF or WF)
        _plot_transitions_subplot_stratified(
            coeff_configs, layers, f'negative_on_{faithful_state}', faithful_state,
            axes[1, 1], f'Negative Steering on {faithful_state}\n(coeff = -{coeff})'
        )

        plt.tight_layout()
        # Save to specified directory
        save_path = os.path.join(save_dir, f'steered_global_layers_{correctness_group}_coeff_{coeff}.png')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved layer comparison for {correctness_group}, coeff ±{coeff} to: {save_path}")


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
            'unfaithful_to_incomplete',
            'unfaithful_to_error'
        ]
        labels = ['→ Faithful', '→ Unfaithful', '→ Correct', '→ Hint Error', '→ Incomplete', '→ Error']
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#e67e22', '#95a5a6', '#7f8c8d']
    else:  # faithful
        transition_names = [
            'faithful_to_unfaithful',
            'faithful_to_faithful',
            'faithful_to_correct',
            'faithful_to_hint_error',
            'faithful_to_incomplete',
            'faithful_to_error'
        ]
        labels = ['→ Unfaithful', '→ Faithful', '→ Correct', '→ Hint Error', '→ Incomplete', '→ Error']
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#e67e22', '#95a5a6', '#7f8c8d']

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


def _plot_transitions_subplot_stratified(configs: List[Dict[str, Any]],
                                         layers: List[int],
                                         group_name: str,
                                         initial_state: str,
                                         ax,
                                         title: str):
    """
    Helper function to plot transitions for one subplot with CF/CU/WF/WU stratification.

    Args:
        configs: List of configs for this coefficient
        layers: List of layer numbers
        group_name: Name of group to plot (e.g., 'positive_on_CF')
        initial_state: 'CF', 'CU', 'WF', or 'WU'
        ax: Matplotlib axis
        title: Subplot title
    """
    # Determine transition types based on initial state
    is_initially_correct = initial_state in ['CF', 'CU']

    # CONSISTENT COLOR SCHEME (same across all panels):
    # - Faithful: green (#2ecc71)
    # - Unfaithful: red (#e74c3c)
    # - Correct (answer correction): orange (#f39c12)
    # - Hint Error: blue (#3498db)
    # - Incomplete: grey (#95a5a6)
    # - Error: black (#000000)

    if is_initially_correct:
        # Initially CORRECT - no wrong_to_correct
        transition_names = [
            'to_same_answer_faithful',
            'to_same_answer_unfaithful',
            'to_hint_error',
            'to_incomplete',
            'to_error'
        ]
        labels = ['→ F (same ans)', '→ U (same ans)', '→ Hint Error', '→ Incomplete', '→ Error']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6', '#000000']  # green, red, blue, grey, black
        markers = ['o', 's', '^', 'D', 'v']  # circle, square, triangle-up, diamond, triangle-down
    else:
        # Initially WRONG - includes wrong_to_correct
        transition_names = [
            'to_same_answer_faithful',
            'to_same_answer_unfaithful',
            'to_correct',
            'to_hint_error',
            'to_incomplete',
            'to_error'
        ]
        labels = ['→ F (same ans)', '→ U (same ans)', '→ Correct', '→ Hint Error', '→ Incomplete', '→ Error']
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#95a5a6', '#000000']  # green, red, orange, blue, grey, black
        markers = ['o', 's', '^', 'D', 'v', 'p']  # circle, square, triangle-up, diamond, triangle-down, pentagon

    num_transitions = len(transition_names)

    # Build data for each transition
    for idx, (transition_name, label, color, marker) in enumerate(zip(transition_names, labels, colors, markers)):
        rates = []
        for layer in layers:
            # Find config for this layer
            config = next((c for c in configs if c['layer'] == layer), None)
            if config and group_name in config:
                rate = config[group_name]['transitions'].get(transition_name, {}).get('rate', 0) * 100
                rates.append(rate)
            else:
                rates.append(0)

        # Calculate horizontal offset to avoid overlapping markers
        # Center the offsets around 0, spreading them ±0.15 layer units
        offset = (idx - (num_transitions - 1) / 2) * 0.06
        x_coords = [layer + offset for layer in layers]

        ax.plot(x_coords, rates, marker=marker, label=label, color=color,
                linewidth=2, markersize=7, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Rate (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 105)  # Slightly above 100 for visibility
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xticks(layers)
