"""
plot_best_config_comparison.py

Visualizes the effects of the best-performing steering configuration by comparing
unsteered vs steered outcome distributions.

Creates 2 plots:
- Plot 1: Positive Steering Effects (CU/WU groups)
- Plot 2: Negative Steering Effects (CF/WF groups)

Each plot shows 4 bars per hint template:
- Unsteered baseline
- Steered outcomes
For both relevant initial states (C and W)

Usage:
    python plot_best_config_comparison.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

PARETO_RANKINGS_FILE = "data/sprint4_2025-10-21/analysis/pareto_rankings_2025-10-27.json"
STEERED_SUMMARY_FILE = "data/sprint4_2025-10-21/summaries/steered_faithfulness/summary_faithfulness_steered_sprint4_2025-10-27.json"
OUTPUT_DIR = "data/sprint4_2025-10-21/plots"

# Outcome categories for visualization
C_OUTCOMES = ['to_same_answer_faithful', 'to_same_answer_unfaithful', 'to_hint_error', 'to_incomplete', 'to_error']
W_OUTCOMES = ['to_same_answer_faithful', 'to_same_answer_unfaithful', 'to_hint_error', 'to_incomplete', 'to_correct', 'to_error']

OUTCOME_LABELS = {
    'to_same_answer_faithful': 'Faithful',
    'to_same_answer_unfaithful': 'Unfaithful',
    'to_hint_error': 'Hint Error',
    'to_incomplete': 'Incomplete',
    'to_correct': 'To Correct',
    'to_error': 'Error'
}

OUTCOME_COLORS = {
    'to_same_answer_faithful': '#7dd3a0',      # Stronger pastel green
    'to_same_answer_unfaithful': '#ff9999',    # Pastel red
    'to_hint_error': '#ffe599',                # Pastel yellow
    'to_incomplete': '#d3d3d3',                # Pastel gray
    'to_correct': '#6eb8e6',                   # Stronger pastel light blue
    'to_error': '#000000'                      # Black
}


def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_best_config(pareto_rankings: Dict) -> Tuple[int, float]:
    """
    Extract best overall config from pareto rankings.

    Returns:
        (layer, coefficient)
    """
    if 'overall_ranking' not in pareto_rankings:
        raise ValueError("No overall ranking found in pareto rankings file")

    best = pareto_rankings['overall_ranking']['top_config']
    return best['layer'], best['coefficient']


def get_unsteered_distribution(annotated_hinted: List[Dict], hint_template: str,
                                initial_state: str) -> Dict[str, float]:
    """
    Get outcome distribution from unsteered annotated hinted data.

    Args:
        annotated_hinted: List of annotated hinted records
        hint_template: Hint template to filter by
        initial_state: One of 'CU', 'CF', 'WU', 'WF'

    Returns:
        Dictionary of outcome -> rate
    """
    # Filter records
    filtered = [
        r for r in annotated_hinted
        if r.get('hint_template') == hint_template
        and r.get('initial_state') == initial_state
        and r.get('bias_label') == 'biased'  # Only biased records
    ]

    if not filtered:
        return {}

    # Count outcomes
    # Map faithfulness_classification to outcome categories
    outcome_counts = defaultdict(int)

    for record in filtered:
        classification = record.get('faithfulness_classification', 'error')

        # Map to transition outcomes
        if classification == 'faithful':
            outcome_counts['to_same_answer_faithful'] += 1
        elif classification == 'unfaithful':
            outcome_counts['to_same_answer_unfaithful'] += 1
        elif classification == 'hint_error':
            outcome_counts['to_hint_error'] += 1
        elif classification == 'incomplete':
            outcome_counts['to_incomplete'] += 1
        elif classification == 'wrong_to_correct':
            outcome_counts['to_correct'] += 1
        else:
            outcome_counts['to_error'] += 1

    # Convert to rates
    total = len(filtered)
    return {outcome: count / total for outcome, count in outcome_counts.items()}


def get_steered_distribution(steered_summary: Dict, hint_template: str,
                             layer: int, coefficient: float, group_name: str) -> Dict[str, float]:
    """
    Get outcome distribution from steered summary for specific config and group.

    Args:
        steered_summary: Steered summary data
        hint_template: Hint template
        layer: Steering layer
        coefficient: Steering coefficient
        group_name: One of 'positive_on_CU', 'negative_on_CF', etc.

    Returns:
        Dictionary of outcome -> rate
    """
    # Find the config
    configs = steered_summary['configurations_by_hint'].get(hint_template, [])

    for config in configs:
        if config['layer'] == layer and abs(config['coefficient_magnitude'] - coefficient) < 0.001:
            # Extract group data
            if group_name not in config:
                return {}

            group = config[group_name]
            transitions = group.get('transitions', {})

            # Return rates
            return {outcome: trans.get('rate', 0) for outcome, trans in transitions.items()}

    return {}


def plot_steering_effects_combined(
    hint_templates: List[str],
    best_layer: int,
    best_coeff: float,
    steered_data: Dict,    # hint -> group -> outcome -> rate
    save_path: str
):
    """
    Create single combined plot showing steered outcomes for all groups.
    Unsteered outcomes are implicit (CU/WU=unfaithful, CF/WF=faithful by definition).

    Args:
        hint_templates: List of hint templates
        best_layer: Best config layer
        best_coeff: Best config coefficient
        steered_data: Nested dict of steered distributions
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(len(hint_templates), 1, figsize=(16, 6 * len(hint_templates)))
    if len(hint_templates) == 1:
        axes = [axes]

    # All 8 groups: 4 positive + 4 negative
    positive_groups = ['CU', 'CF', 'WU', 'WF']
    negative_groups = ['CU', 'CF', 'WU', 'WF']

    positive_group_map = {
        'CU': 'positive_on_CU',
        'CF': 'positive_on_CF',
        'WU': 'positive_on_WU',
        'WF': 'positive_on_WF'
    }

    negative_group_map = {
        'CU': 'negative_on_CU',
        'CF': 'negative_on_CF',
        'WU': 'negative_on_WU',
        'WF': 'negative_on_WF'
    }

    for ax, hint in zip(axes, hint_templates):
        # Create 8 bars: 4 positive steering + 4 negative steering
        x_positions = []
        bar_labels = []

        # Positive steering groups
        for i, group in enumerate(positive_groups):
            x_positions.append(i * 1.2)
            # Mark effectiveness target with asterisk
            label = f'Pos\n{group}*' if group == 'WU' else f'Pos\n{group}'
            bar_labels.append(label)

        # Gap between positive and negative
        gap_offset = len(positive_groups) * 1.2 + 0.8

        # Negative steering groups
        for i, group in enumerate(negative_groups):
            x_positions.append(gap_offset + i * 1.2)
            # Mark effectiveness target with asterisk
            label = f'Neg\n{group}*' if group == 'CF' else f'Neg\n{group}'
            bar_labels.append(label)

        # Prepare data for stacking
        bar_data = []

        # Positive steering
        for group in positive_groups:
            steered_group = positive_group_map[group]
            steered_dist = steered_data.get(hint, {}).get(steered_group, {})
            bar_data.append(steered_dist)

        # Negative steering
        for group in negative_groups:
            steered_group = negative_group_map[group]
            steered_dist = steered_data.get(hint, {}).get(steered_group, {})
            bar_data.append(steered_dist)

        # Use W outcomes (includes to_correct)
        outcomes = W_OUTCOMES

        # Create stacked bars
        bottom = np.zeros(len(bar_data))

        for outcome in outcomes:
            heights = [bar.get(outcome, 0) for bar in bar_data]
            ax.bar(
                x_positions,
                heights,
                bottom=bottom,
                label=OUTCOME_LABELS[outcome],
                color=OUTCOME_COLORS[outcome],
                width=0.9
            )
            bottom += heights

        # Formatting
        ax.set_ylabel('Rate', fontsize=11)
        ax.set_title(f'{hint}', fontsize=12, fontweight='bold', pad=30)  # Increased pad for spacing
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, fontsize=9, rotation=0)
        ax.set_ylim([0, 1.15])  # Extra space for labels
        ax.grid(axis='y', alpha=0.3)

        # Add vertical separator between positive and negative
        separator_x = (x_positions[3] + x_positions[4]) / 2
        ax.axvline(x=separator_x, color='black', linestyle='-', alpha=0.7, linewidth=2)

        # Add text labels above the plot area
        ax.text((x_positions[0] + x_positions[3]) / 2, 1.08, 'Positive Steering',
                ha='center', fontsize=10, fontweight='bold')
        ax.text((x_positions[4] + x_positions[7]) / 2, 1.08, 'Negative Steering',
                ha='center', fontsize=10, fontweight='bold')

        # Add group labels below positive/negative labels
        ax.text((x_positions[0] + x_positions[3]) / 2, 1.03, '(CU/CF/WU*/WF)',
                ha='center', fontsize=8, style='italic')
        ax.text((x_positions[4] + x_positions[7]) / 2, 1.03, '(CU/CF*/WU/WF)',
                ha='center', fontsize=8, style='italic')

        # Legend only on first plot
        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            # Add asterisk explanation
            ax.text(0.02, 0.97, '* = Primary effectiveness target',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title with more space
    fig.suptitle(f'Steering Effects - Best Config: Layer {best_layer}, Coeff ±{best_coeff}',
                 fontsize=14, fontweight='bold', y=0.998)

    plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined steering plot to: {save_path}")
    plt.close()


def main():
    """Main function."""
    print("=" * 80)
    print("GENERATING BEST CONFIG COMPARISON PLOT")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    pareto_rankings = load_json(PARETO_RANKINGS_FILE)
    steered_summary = load_json(STEERED_SUMMARY_FILE)

    # Get best config
    best_layer, best_coeff = get_best_config(pareto_rankings)
    print(f"\nBest config: Layer {best_layer}, Coefficient ±{best_coeff}")

    # Get hint templates
    hint_templates = pareto_rankings['metadata']['hint_templates']
    print(f"Hint templates: {hint_templates}")

    # Prepare steered distributions (all 8 groups)
    print("\nExtracting steered distributions...")
    steered_distributions = {}

    all_groups = [
        'positive_on_CU', 'positive_on_CF', 'positive_on_WU', 'positive_on_WF',
        'negative_on_CU', 'negative_on_CF', 'negative_on_WU', 'negative_on_WF'
    ]

    for hint in hint_templates:
        steered_distributions[hint] = {}
        for group in all_groups:
            dist = get_steered_distribution(steered_summary, hint, best_layer, best_coeff, group)
            if dist:
                steered_distributions[hint][group] = dist
                print(f"  {hint} - {group}: found")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate separate plot for each hint
    print("\nGenerating plots...")

    for hint in hint_templates:
        plot_steering_effects_combined(
            hint_templates=[hint],  # Single hint
            best_layer=best_layer,
            best_coeff=best_coeff,
            steered_data=steered_distributions,
            save_path=str(output_dir / f'best_config_steering_effects_{hint}_L{best_layer}_C{best_coeff}.png')
        )

    print("\n" + "=" * 80)
    print("PLOT GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
