"""
plot_steered_faithfulness.py

Plots faithfulness classification distributions for steered (post-steering) annotated data.
Shows the breakdown across question IDs and steering coefficients.
"""

import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from src.data import load_jsonl

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files - annotated steered results
INPUT_FILES = [
    "data/Lorenz_suggestion_2025-11-12/annotated_steered_sampled_q26_2025-11-12.jsonl",
    "data/Lorenz_suggestion_2025-11-12/annotated_steered_sampled_q115_2025-11-12.jsonl",
]

# Output directory for plots
OUTPUT_DIR = "data/Lorenz_suggestion_2025-11-12/plots"

# =============================================================================
# END CONFIGURATION
# =============================================================================


def analyze_steered_file(file_path):
    """
    Analyze a single steered annotation file, grouped by steering coefficient.

    Args:
        file_path: Path to annotated JSONL file

    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None

    records = load_jsonl(file_path)

    if len(records) == 0:
        print(f"Warning: No records in {file_path}")
        return None

    # Extract question_id and hint info from first record
    first_record = records[0]
    question_id = first_record.get('question_id')
    hint_letter = first_record.get('hint_letter')
    hint_template = first_record.get('hint_template', 'unknown')

    # Group records by steering coefficient
    by_coefficient = defaultdict(list)
    for record in records:
        coef = record.get('steering_coefficient')
        by_coefficient[coef].append(record)

    # Analyze each coefficient group
    coefficient_analyses = {}
    for coef, coef_records in by_coefficient.items():
        # Count bias labels
        bias_counts = {
            'correct': sum(1 for r in coef_records if r.get('bias_label') == 'correct'),
            'biased': sum(1 for r in coef_records if r.get('bias_label') == 'biased'),
            'other': sum(1 for r in coef_records if r.get('bias_label') == 'other')
        }

        # Count faithfulness for biased records
        biased_records = [r for r in coef_records if r.get('bias_label') == 'biased']
        faithfulness_counts = {
            'faithful': sum(1 for r in biased_records if r.get('faithfulness_classification') == 'faithful'),
            'unfaithful': sum(1 for r in biased_records if r.get('faithfulness_classification') == 'unfaithful'),
            'error': sum(1 for r in biased_records if r.get('faithfulness_classification') == 'error')
        }

        coefficient_analyses[coef] = {
            'total_records': len(coef_records),
            'bias_counts': bias_counts,
            'faithfulness_counts': faithfulness_counts,
            'biased_count': len(biased_records)
        }

    return {
        'file_path': file_path,
        'question_id': question_id,
        'hint_letter': hint_letter,
        'hint_template': hint_template,
        'total_records': len(records),
        'coefficient_analyses': coefficient_analyses
    }


def plot_question_by_coefficient(analysis, output_path):
    """
    Create a plot showing steering coefficient effect on answer distribution for one question.

    Args:
        analysis: Analysis dictionary from analyze_steered_file
        output_path: Path to save the plot
    """
    coef_analyses = analysis['coefficient_analyses']

    # Sort coefficients
    coefficients = sorted(coef_analyses.keys())
    n_coefs = len(coefficients)

    fig, ax = plt.subplots(figsize=(max(12, 2*n_coefs), 8))

    # X positions for bars
    x_positions = np.arange(n_coefs)
    bar_width = 0.6

    # Prepare data for each category
    coef_labels = []
    correct_counts = []
    faithful_counts = []
    unfaithful_counts = []
    other_counts = []

    for coef in coefficients:
        coef_data = coef_analyses[coef]
        coef_labels.append(f"{coef:+.2f}")
        correct_counts.append(coef_data['bias_counts']['correct'])
        faithful_counts.append(coef_data['faithfulness_counts']['faithful'])
        unfaithful_counts.append(coef_data['faithfulness_counts']['unfaithful'])
        other_counts.append(coef_data['bias_counts']['other'])

    # Create stacked bars with nested breakdown
    # Correct (green)
    bars1 = ax.bar(x_positions, correct_counts, bar_width,
                   label='Correct', color='#2ecc71', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Biased - Faithful (blue)
    bars2 = ax.bar(x_positions, faithful_counts, bar_width,
                   bottom=correct_counts, label='Biased (Faithful)',
                   color='#3498db', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Biased - Unfaithful (orange)
    bars3 = ax.bar(x_positions, unfaithful_counts, bar_width,
                   bottom=np.array(correct_counts) + np.array(faithful_counts),
                   label='Biased (Unfaithful)', color='#e67e22', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Other (grey)
    bars4 = ax.bar(x_positions, other_counts, bar_width,
                   bottom=np.array(correct_counts) + np.array(faithful_counts) + np.array(unfaithful_counts),
                   label='Other', color='#95a5a6', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Add count labels on each segment
    for i, (c, f, u, o) in enumerate(zip(correct_counts, faithful_counts, unfaithful_counts, other_counts)):
        # Correct label
        if c > 0:
            ax.text(i, c/2, f'{c}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        # Faithful label
        if f > 0:
            ax.text(i, c + f/2, f'{f}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        # Unfaithful label
        if u > 0:
            ax.text(i, c + f + u/2, f'{u}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        # Other label
        if o > 0:
            ax.text(i, c + f + u + o/2, f'{o}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

    # Styling
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_xlabel('Steering Coefficient', fontsize=14, fontweight='bold')
    ax.set_title(f"Question {analysis['question_id']} | {analysis['hint_template']}\n"
                 f"Steered Answer Distribution by Coefficient",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(coef_labels, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add vertical line at coefficient 0 if it exists
    if 0.0 in coefficients:
        zero_idx = coefficients.index(0.0)
        ax.axvline(x=zero_idx, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No steering')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_combined_questions(analyses, output_path):
    """
    Create a combined plot comparing multiple questions across coefficients.

    Args:
        analyses: List of analysis dictionaries
        output_path: Path to save the plot
    """
    n_questions = len(analyses)

    fig, axes = plt.subplots(1, n_questions, figsize=(8*n_questions, 8), sharey=True)
    if n_questions == 1:
        axes = [axes]

    for idx, analysis in enumerate(analyses):
        ax = axes[idx]
        coef_analyses = analysis['coefficient_analyses']

        # Sort coefficients
        coefficients = sorted(coef_analyses.keys())
        n_coefs = len(coefficients)

        # X positions for bars
        x_positions = np.arange(n_coefs)
        bar_width = 0.6

        # Prepare data
        coef_labels = []
        correct_counts = []
        faithful_counts = []
        unfaithful_counts = []
        other_counts = []

        for coef in coefficients:
            coef_data = coef_analyses[coef]
            coef_labels.append(f"{coef:+.2f}")
            correct_counts.append(coef_data['bias_counts']['correct'])
            faithful_counts.append(coef_data['faithfulness_counts']['faithful'])
            unfaithful_counts.append(coef_data['faithfulness_counts']['unfaithful'])
            other_counts.append(coef_data['bias_counts']['other'])

        # Create stacked bars
        ax.bar(x_positions, correct_counts, bar_width,
               label='Correct' if idx == 0 else '', color='#2ecc71', alpha=0.8,
               edgecolor='black', linewidth=1.5)

        ax.bar(x_positions, faithful_counts, bar_width,
               bottom=correct_counts, label='Biased (Faithful)' if idx == 0 else '',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.bar(x_positions, unfaithful_counts, bar_width,
               bottom=np.array(correct_counts) + np.array(faithful_counts),
               label='Biased (Unfaithful)' if idx == 0 else '', color='#e67e22', alpha=0.8,
               edgecolor='black', linewidth=1.5)

        ax.bar(x_positions, other_counts, bar_width,
               bottom=np.array(correct_counts) + np.array(faithful_counts) + np.array(unfaithful_counts),
               label='Other' if idx == 0 else '', color='#95a5a6', alpha=0.8,
               edgecolor='black', linewidth=1.5)

        # Add count labels
        for i, (c, f, u, o) in enumerate(zip(correct_counts, faithful_counts, unfaithful_counts, other_counts)):
            if c > 0:
                ax.text(i, c/2, f'{c}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
            if f > 0:
                ax.text(i, c + f/2, f'{f}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
            if u > 0:
                ax.text(i, c + f + u/2, f'{u}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
            if o > 0:
                ax.text(i, c + f + u + o/2, f'{o}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')

        # Styling
        ax.set_xlabel('Steering Coefficient', fontsize=12, fontweight='bold')
        ax.set_title(f"Q{analysis['question_id']} ({analysis['hint_template']})",
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(coef_labels, fontsize=10, fontweight='bold', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Only add ylabel to leftmost plot
        if idx == 0:
            ax.set_ylabel('Count', fontsize=14, fontweight='bold')

    # Add legend only once (on the first subplot)
    if n_questions > 0:
        axes[0].legend(loc='upper left', fontsize=11, framealpha=0.9)

    fig.suptitle('Steered (Post-Steering) Answer Distribution Comparison',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined plot: {output_path}")


def main():
    """Main entry point for plotting steered faithfulness."""
    print(f"\n{'='*60}")
    print(f"STEERED FAITHFULNESS PLOTTING")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Analyze all input files
    analyses = []
    for file_path in INPUT_FILES:
        print(f"\nAnalyzing: {file_path}")
        analysis = analyze_steered_file(file_path)
        if analysis:
            analyses.append(analysis)

            # Print summary
            print(f"  Question ID: {analysis['question_id']}")
            print(f"  Hint: {analysis['hint_letter']} ({analysis['hint_template']})")
            print(f"  Total records: {analysis['total_records']}")
            print(f"  Steering coefficients: {sorted(analysis['coefficient_analyses'].keys())}")

            # Print details for each coefficient
            for coef in sorted(analysis['coefficient_analyses'].keys()):
                coef_data = analysis['coefficient_analyses'][coef]
                print(f"    Coef {coef:+.2f}: {coef_data['total_records']} records, "
                      f"Bias: {coef_data['bias_counts']}, "
                      f"Faith (biased): {coef_data['faithfulness_counts']}")

            # Create individual plot by coefficient
            output_file = os.path.join(OUTPUT_DIR, f"q{analysis['question_id']}_steered_by_coefficient.png")
            plot_question_by_coefficient(analysis, output_file)

    if len(analyses) == 0:
        print("\nNo valid data to plot!")
        return

    # Create combined summary plot
    combined_output = os.path.join(OUTPUT_DIR, "combined_steered_faithfulness.png")
    plot_combined_questions(analyses, combined_output)

    print(f"\n{'='*60}")
    print(f"PLOTTING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Generated {len(analyses)} individual plots + 1 combined plot")


if __name__ == "__main__":
    main()
