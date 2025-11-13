"""
plot_hinted_faithfulness.py

Plots faithfulness classification distributions for hinted (pre-steering) annotated data.
Shows the breakdown of: correct, faithful, unfaithful, other, and error classifications.
"""

import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from src.data import load_jsonl

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files - annotated hinted results
INPUT_FILES = [
    "data/Lorenz_suggestion_2025-11-12/annotated_hinted_wrong_sampled_q26_2025-11-12.jsonl",
    "data/Lorenz_suggestion_2025-11-12/annotated_hinted_wrong_sampled_q115_2025-11-12.jsonl",
]

# Output directory for plots
OUTPUT_DIR = "data/Lorenz_suggestion_2025-11-12/plots"

# =============================================================================
# END CONFIGURATION
# =============================================================================


def analyze_hinted_file(file_path):
    """
    Analyze a single hinted annotation file.

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
    question_text = first_record.get('question', 'Unknown question')

    # Count bias labels
    bias_counts = Counter(r.get('bias_label') for r in records)

    # Count faithfulness classifications (only for biased records)
    biased_records = [r for r in records if r.get('bias_label') == 'biased']
    faithfulness_counts = Counter(
        r.get('faithfulness_classification', 'no_classification')
        for r in biased_records
    )

    return {
        'file_path': file_path,
        'question_id': question_id,
        'hint_letter': hint_letter,
        'hint_template': hint_template,
        'question_text': question_text,
        'total_records': len(records),
        'bias_counts': dict(bias_counts),
        'faithfulness_counts': dict(faithfulness_counts),
        'biased_count': len(biased_records)
    }


def plot_single_question(analysis, output_path):
    """
    Create a simple stacked bar plot for a single question showing bias distribution.

    Args:
        analysis: Analysis dictionary from analyze_hinted_file
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get bias counts (from all records)
    bias_labels = ['Correct', 'Biased', 'Other']
    correct = analysis['bias_counts'].get('correct', 0)
    biased = analysis['bias_counts'].get('biased', 0)
    other = analysis['bias_counts'].get('other', 0)

    bias_counts = [correct, biased, other]
    bias_colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Grey

    print(f"  Q{analysis['question_id']} counts: Correct={correct}, Biased={biased}, Other={other}")

    # Create stacked bar - always draw all segments to maintain color consistency
    x_pos = [0]
    bottom = 0
    for count, color, label in zip(bias_counts, bias_colors, bias_labels):
        bar = ax.bar(x_pos, count, bottom=bottom, color=color, alpha=0.8,
                     edgecolor='black', linewidth=1.5, label=label)

        # Add count label if count > 0
        if count > 0:
            ax.text(0, bottom + count/2, f'{count}',
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        bottom += count

    # Styling
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title(f"Question {analysis['question_id']} | Hint: {analysis['hint_letter']}\n"
                 f"Answer Distribution (n={analysis['total_records']} total)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_ylim(0, sum(bias_counts) * 1.1 if sum(bias_counts) > 0 else 1)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_single_question_detailed(analysis, output_path):
    """
    Create a detailed plot showing bias distribution with faithfulness breakdown for biased records.

    Args:
        analysis: Analysis dictionary from analyze_hinted_file
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get counts
    correct = analysis['bias_counts'].get('correct', 0)
    biased_total = analysis['bias_counts'].get('biased', 0)
    other = analysis['bias_counts'].get('other', 0)

    # Faithfulness breakdown within biased
    faithful = analysis['faithfulness_counts'].get('faithful', 0)
    unfaithful = analysis['faithfulness_counts'].get('unfaithful', 0)

    print(f"  Q{analysis['question_id']}: Correct={correct}, Biased={biased_total} (F={faithful}, U={unfaithful}), Other={other}")

    # Create stacked bar with nested colors
    x_pos = [0]
    bottom = 0

    # Correct segment (green)
    if correct > 0:
        ax.bar(x_pos, correct, bottom=bottom, color='#2ecc71', alpha=0.8,
               edgecolor='black', linewidth=1.5, label='Correct')
        ax.text(0, bottom + correct/2, f'{correct}',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        bottom += correct

    # Biased - Faithful segment (blue)
    if faithful > 0:
        ax.bar(x_pos, faithful, bottom=bottom, color='#3498db', alpha=0.8,
               edgecolor='black', linewidth=1.5, label='Biased (Faithful)')
        ax.text(0, bottom + faithful/2, f'{faithful}',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        bottom += faithful

    # Biased - Unfaithful segment (orange)
    if unfaithful > 0:
        ax.bar(x_pos, unfaithful, bottom=bottom, color='#e67e22', alpha=0.8,
               edgecolor='black', linewidth=1.5, label='Biased (Unfaithful)')
        ax.text(0, bottom + unfaithful/2, f'{unfaithful}',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        bottom += unfaithful

    # Other segment (grey)
    if other > 0:
        ax.bar(x_pos, other, bottom=bottom, color='#95a5a6', alpha=0.8,
               edgecolor='black', linewidth=1.5, label='Other')
        ax.text(0, bottom + other/2, f'{other}',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        bottom += other

    # Styling
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title(f"Question {analysis['question_id']} | {analysis['hint_template']}\n"
                 f"Answer Distribution with Faithfulness (n={analysis['total_records']})",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_ylim(0, analysis['total_records'] * 1.1)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_combined_summary(analyses, output_path):
    """
    Create a combined stacked bar plot comparing all questions side by side with faithfulness breakdown.

    Args:
        analyses: List of analysis dictionaries
        output_path: Path to save the plot
    """
    n_questions = len(analyses)

    fig, ax = plt.subplots(figsize=(max(10, 4*n_questions), 8))

    # X positions for bars
    x_positions = np.arange(n_questions)
    bar_width = 0.6

    # Prepare data for each category
    question_ids = []
    correct_counts = []
    faithful_counts = []
    unfaithful_counts = []
    other_counts = []

    for analysis in analyses:
        question_ids.append(f"Q{analysis['question_id']}\n({analysis['hint_template']})")
        correct_counts.append(analysis['bias_counts'].get('correct', 0))
        faithful_counts.append(analysis['faithfulness_counts'].get('faithful', 0))
        unfaithful_counts.append(analysis['faithfulness_counts'].get('unfaithful', 0))
        other_counts.append(analysis['bias_counts'].get('other', 0))

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
                   fontsize=12, fontweight='bold', color='white')
        # Faithful label
        if f > 0:
            ax.text(i, c + f/2, f'{f}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
        # Unfaithful label
        if u > 0:
            ax.text(i, c + f + u/2, f'{u}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
        # Other label
        if o > 0:
            ax.text(i, c + f + u + o/2, f'{o}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')

    # Styling
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title('Hinted (Pre-Steering) Answer Distribution with Faithfulness',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(question_ids, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined plot: {output_path}")


def main():
    """Main entry point for plotting hinted faithfulness."""
    print(f"\n{'='*60}")
    print(f"HINTED FAITHFULNESS PLOTTING")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Analyze all input files
    analyses = []
    for file_path in INPUT_FILES:
        print(f"\nAnalyzing: {file_path}")
        analysis = analyze_hinted_file(file_path)
        if analysis:
            analyses.append(analysis)

            # Print summary
            print(f"  Question ID: {analysis['question_id']}")
            print(f"  Hint: {analysis['hint_letter']} ({analysis['hint_template']})")
            print(f"  Total records: {analysis['total_records']}")
            print(f"  Bias distribution: {analysis['bias_counts']}")
            print(f"  Faithfulness (biased only): {analysis['faithfulness_counts']}")

            # Create individual plot (detailed with faithfulness breakdown)
            output_file = os.path.join(OUTPUT_DIR, f"q{analysis['question_id']}_hinted_faithfulness.png")
            plot_single_question_detailed(analysis, output_file)

    if len(analyses) == 0:
        print("\nNo valid data to plot!")
        return

    # Create combined summary plot
    combined_output = os.path.join(OUTPUT_DIR, "combined_hinted_faithfulness.png")
    plot_combined_summary(analyses, combined_output)

    print(f"\n{'='*60}")
    print(f"PLOTTING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Generated {len(analyses)} individual plots + 1 combined plot")


if __name__ == "__main__":
    main()
