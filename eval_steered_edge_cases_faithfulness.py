"""
eval_filtered_cases_faithfulness.py --> util script to look into steering edge cases: incomplete, correct, hint-induced error answers.

Script to classify cases that were filtered out by rule-based classification
(incomplete, correct, hint-induced error) using LLM judge.

This script:
1. Loads steered evaluation dataset
2. Applies rule-based classification
3. Extracts only incomplete, correct, and hint-induced error cases
4. Runs LLM judge on all these filtered cases
5. Groups by (layer, coefficient, rule_based_classification, original_faithfulness)
6. Computes statistics and creates visualizations
7. Saves annotated results with LLM classifications
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict
from typing import Dict, Any, List, Tuple

# Load environment variables
load_dotenv()

# Import existing modules
from src.data import load_jsonl, save_jsonl
from src.global_faithfulness import setup_openrouter_client, judge_batch
from src.steered_global_faithfulness import classify_steered_batch
from src.config import TODAY, ANNOTATED_DIR, SUMMARIES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file - steered evaluation results
INPUT_FILE = "data/annotated/annotated_steered_local_high_school_psychology_professor_2025-10-06.jsonl"

# Subject
SUBJECT = "high_school_psychology"
HINT_TEMPLATE = "professor"

# Plotting
PLOT_DIR = "plots"

# Model configuration
JUDGE_MODEL = "google/gemini-2.5-flash"  # OpenRouter model name
MAX_RETRIES = 3

# Output files
OUTPUT_FILE = f"data/annotated/llm_judged_filtered_cases_{SUBJECT}_{HINT_TEMPLATE}_{TODAY}.jsonl"
SUMMARY_FILE = f"data/summaries/filtered_cases_summary_{SUBJECT}_{HINT_TEMPLATE}_{TODAY}.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def group_by_configuration(records: List[Dict[str, Any]]) -> Dict[Tuple, Dict[str, List]]:
    """
    Group records by (layer, coefficient, rule_based_class, original_faithfulness).

    Returns:
        Nested dict: {(layer, coeff, rule_class, orig_faith): [records]}
    """
    grouped = defaultdict(list)

    for record in records:
        layer = record['steering_layer']
        coeff = record['steering_coefficient']
        rule_class = record['rule_based_classification']
        orig_faith = record['original_faithfulness_classification']

        key = (layer, coeff, rule_class, orig_faith)
        grouped[key].append(record)

    return dict(grouped)


def compute_group_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics for a group of records.

    Returns:
        Dictionary with counts and rates
    """
    n = len(records)
    if n == 0:
        return {'n': 0, 'faithful_rate': 0, 'unfaithful_rate': 0, 'error_rate': 0}

    faithful = sum(1 for r in records if r.get('llm_classification') == 'faithful')
    unfaithful = sum(1 for r in records if r.get('llm_classification') == 'unfaithful')
    errors = sum(1 for r in records if r.get('llm_classification') == 'error')

    return {
        'n': n,
        'faithful_count': faithful,
        'unfaithful_count': unfaithful,
        'error_count': errors,
        'faithful_rate': faithful / n,
        'unfaithful_rate': unfaithful / n,
        'error_rate': errors / n
    }


def create_summary_structure(grouped: Dict[Tuple, List]) -> Dict[str, Any]:
    """
    Create summary data structure with statistics for all groups.
    """
    summary = {
        'evaluation_date': TODAY,
        'method': 'filtered_cases_llm_judge',
        'judge_model': JUDGE_MODEL,
        'source_file': INPUT_FILE,
        'subject': SUBJECT,
        'hint_template': HINT_TEMPLATE,
        'groups': {}
    }

    for (layer, coeff, rule_class, orig_faith), records in grouped.items():
        key = f"layer_{layer}_coeff_{coeff}_{rule_class}_{orig_faith}"
        stats = compute_group_statistics(records)

        summary['groups'][key] = {
            'layer': layer,
            'coefficient': coeff,
            'rule_based_classification': rule_class,
            'original_faithfulness': orig_faith,
            'statistics': stats
        }

    return summary


def plot_heatmaps(grouped: Dict[Tuple, List], subject: str, hint_template: str, save_path: str):
    """
    Create heatmaps showing faithful/unfaithful rates across layers and coefficients
    for each combination of (rule_based_class, original_faithfulness).
    """
    # Get unique values
    layers = sorted(set(k[0] for k in grouped.keys()))
    coeffs = sorted(set(k[1] for k in grouped.keys()))
    rule_classes = sorted(set(k[2] for k in grouped.keys()))
    orig_faiths = sorted(set(k[3] for k in grouped.keys()))

    # Create subplots for each rule_class × orig_faith combination
    n_combinations = len(rule_classes) * len(orig_faiths)
    n_cols = len(orig_faiths)
    n_rows = len(rule_classes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'Filtered Cases: Faithful Rate by Layer & Coefficient\n{subject} - {hint_template}',
                 fontsize=16, fontweight='bold')

    for i, rule_class in enumerate(rule_classes):
        for j, orig_faith in enumerate(orig_faiths):
            ax = axes[i, j]

            # Build matrix for this combination
            matrix = np.full((len(layers), len(coeffs)), np.nan)

            for layer_idx, layer in enumerate(layers):
                for coeff_idx, coeff in enumerate(coeffs):
                    key = (layer, coeff, rule_class, orig_faith)
                    if key in grouped:
                        stats = compute_group_statistics(grouped[key])
                        matrix[layer_idx, coeff_idx] = stats['faithful_rate']

            # Plot heatmap
            sns.heatmap(matrix, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                       xticklabels=[f'{c:.2f}' for c in coeffs],
                       yticklabels=layers,
                       annot=True, fmt='.2f', cbar_kws={'label': 'Faithful Rate'})

            ax.set_title(f'{rule_class} + {orig_faith}')
            ax.set_xlabel('Coefficient')
            ax.set_ylabel('Layer')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved heatmaps to: {save_path}")


def plot_layer_wise_breakdown(grouped: Dict[Tuple, List], save_dir: str):
    """
    Create layer-wise plots showing distribution of faithful/unfaithful
    for each coefficient magnitude.
    """
    coeffs = sorted(set(abs(k[1]) for k in grouped.keys()))
    layers = sorted(set(k[0] for k in grouped.keys()))

    for coeff_mag in coeffs:
        # Get all positive and negative coefficients for this magnitude
        coeff_variants = [c for c in set(k[1] for k in grouped.keys()) if abs(c) == coeff_mag]

        fig, axes = plt.subplots(len(coeff_variants), 3, figsize=(18, 6 * len(coeff_variants)))
        if len(coeff_variants) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Filtered Cases: Layer-wise Breakdown (Coefficient Magnitude = {coeff_mag})',
                     fontsize=16, fontweight='bold')

        for coeff_idx, coeff in enumerate(sorted(coeff_variants)):
            rule_classes = ['incomplete', 'correct', 'hint-induced error']

            for rule_idx, rule_class in enumerate(rule_classes):
                ax = axes[coeff_idx, rule_idx]

                # Prepare data for this subplot
                faithful_rates = []
                unfaithful_rates = []
                ns = []

                for layer in layers:
                    # Combine both original faithfulness states
                    total_faithful = 0
                    total_unfaithful = 0
                    total_n = 0

                    for orig_faith in ['faithful', 'unfaithful']:
                        key = (layer, coeff, rule_class, orig_faith)
                        if key in grouped:
                            stats = compute_group_statistics(grouped[key])
                            total_faithful += stats['faithful_count']
                            total_unfaithful += stats['unfaithful_count']
                            total_n += stats['n']

                    if total_n > 0:
                        faithful_rates.append(total_faithful / total_n)
                        unfaithful_rates.append(total_unfaithful / total_n)
                        ns.append(total_n)
                    else:
                        faithful_rates.append(0)
                        unfaithful_rates.append(0)
                        ns.append(0)

                # Plot stacked bars
                x = np.arange(len(layers))
                ax.bar(x, faithful_rates, label='Faithful', color='green', alpha=0.7)
                ax.bar(x, unfaithful_rates, bottom=faithful_rates, label='Unfaithful', color='red', alpha=0.7)

                # Add sample sizes as text
                for xi, n in enumerate(ns):
                    if n > 0:
                        ax.text(xi, 0.5, f'n={n}', ha='center', va='center', fontsize=8)

                ax.set_xticks(x)
                ax.set_xticklabels(layers)
                ax.set_xlabel('Layer')
                ax.set_ylabel('Rate')
                ax.set_ylim([0, 1])
                ax.set_title(f'{rule_class}\n(coeff = {coeff:+.2f})')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'school_psychology_professor', 'school_psychology_professor_local',
                                 'layer-wise steering performance',
                                 f'filtered_cases_layers_coeff_{coeff_mag}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved layer-wise breakdown for coeff ±{coeff_mag} to: {plot_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for filtered cases LLM evaluation."""
    print(f"=== LLM JUDGE FOR FILTERED CASES - {TODAY} ===")
    print(f"Subject: {SUBJECT}")
    print(f"Hint Template: {HINT_TEMPLATE}")

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Input file not found: {INPUT_FILE}")
        return

    # 1. Load data
    print(f"\n{'=' * 80}")
    print("STEP 1: Loading Data")
    print(f"{'=' * 80}")
    print(f"Loading steered dataset from: {INPUT_FILE}")

    all_records = load_jsonl(INPUT_FILE)
    print(f"✓ Loaded {len(all_records)} records")

    # 2. Apply rule-based classification
    print(f"\n{'=' * 80}")
    print("STEP 2: Applying Rule-Based Classification")
    print(f"{'=' * 80}")

    classifications = classify_steered_batch(all_records)

    # Count classifications
    classification_counts = defaultdict(int)
    for cls in classifications.values():
        classification_counts[cls] += 1

    print(f"\nClassification breakdown:")
    print(f"  Incomplete: {classification_counts['incomplete']}")
    print(f"  Correct: {classification_counts['correct']}")
    print(f"  Hint-induced error: {classification_counts['hint-induced error']}")
    print(f"  Needs judge: {classification_counts['needs_judge']}")

    # 3. Filter records (get only incomplete, correct, hint-induced error)
    print(f"\n{'=' * 80}")
    print("STEP 3: Filtering Records")
    print(f"{'=' * 80}")

    filtered_records = []
    for record in all_records:
        qid = record.get('question_id', record.get('prompt_index'))
        cls = classifications.get(qid)
        if cls in ['incomplete', 'correct', 'hint-induced error']:
            # Add the rule-based classification to record
            record['rule_based_classification'] = cls
            filtered_records.append(record)

    print(f"✓ Filtered {len(filtered_records)} records for LLM judging")
    print(f"  (excluding {classification_counts['needs_judge']} 'needs_judge' cases)")

    if len(filtered_records) == 0:
        print("\nNo filtered records to judge. Exiting.")
        return

    # 4. Setup API client
    print(f"\n{'=' * 80}")
    print("STEP 4: Setting Up API Client")
    print(f"{'=' * 80}")

    try:
        client = setup_openrouter_client()
        print(f"✓ OpenRouter client initialized")
        print(f"  Judge Model: {JUDGE_MODEL}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return

    # 5. Run LLM judge on all filtered cases
    print(f"\n{'=' * 80}")
    print("STEP 5: Running LLM Judge on Filtered Cases")
    print(f"{'=' * 80}")

    judgments = judge_batch(
        results=filtered_records,
        client=client,
        model=JUDGE_MODEL,
        max_retries=MAX_RETRIES,
        verbose=True
    )

    # 6. Merge judgments back into records
    print(f"\n{'=' * 80}")
    print("STEP 6: Merging Results")
    print(f"{'=' * 80}")

    faithful_count = 0
    unfaithful_count = 0
    error_count = 0

    for record, judgment in zip(filtered_records, judgments):
        if judgment['success']:
            record['llm_classification'] = judgment['classification']
            record['llm_raw_response'] = judgment.get('raw_response')
            record['llm_api_usage'] = judgment.get('api_usage')

            if judgment['classification'] == 'faithful':
                faithful_count += 1
            elif judgment['classification'] == 'unfaithful':
                unfaithful_count += 1
        else:
            record['llm_classification'] = 'error'
            record['llm_error'] = judgment.get('error')
            error_count += 1

    print(f"\nLLM Judgment Results:")
    print(f"  Faithful: {faithful_count}")
    print(f"  Unfaithful: {unfaithful_count}")
    print(f"  Errors: {error_count}")

    # 7. Group by configuration
    print(f"\n{'=' * 80}")
    print("STEP 7: Grouping by Configuration")
    print(f"{'=' * 80}")

    grouped = group_by_configuration(filtered_records)
    print(f"✓ Created {len(grouped)} configuration groups")

    # Print group breakdown
    layers = sorted(set(k[0] for k in grouped.keys()))
    coeffs = sorted(set(k[1] for k in grouped.keys()))
    rule_classes = sorted(set(k[2] for k in grouped.keys()))
    orig_faiths = sorted(set(k[3] for k in grouped.keys()))

    print(f"  Layers: {layers}")
    print(f"  Coefficients: {coeffs}")
    print(f"  Rule-based classes: {rule_classes}")
    print(f"  Original faithfulness: {orig_faiths}")

    # 8. Compute statistics and create summary
    print(f"\n{'=' * 80}")
    print("STEP 8: Computing Statistics")
    print(f"{'=' * 80}")

    summary = create_summary_structure(grouped)
    print(f"✓ Computed statistics for all groups")

    # 9. Print detailed summary by rule-based category
    print(f"\n{'=' * 80}")
    print("SUMMARY BY RULE-BASED CATEGORY")
    print(f"{'=' * 80}")

    for rule_cls in rule_classes:
        subset = [r for r in filtered_records if r['rule_based_classification'] == rule_cls]
        if len(subset) > 0:
            faithful = sum(1 for r in subset if r.get('llm_classification') == 'faithful')
            unfaithful = sum(1 for r in subset if r.get('llm_classification') == 'unfaithful')
            errors = sum(1 for r in subset if r.get('llm_classification') == 'error')

            print(f"\n{rule_cls.upper()} (n={len(subset)}):")
            print(f"  Faithful: {faithful} ({faithful/len(subset)*100:.1f}%)")
            print(f"  Unfaithful: {unfaithful} ({unfaithful/len(subset)*100:.1f}%)")
            if errors > 0:
                print(f"  Errors: {errors}")

    # 10. Save results
    print(f"\n{'=' * 80}")
    print("STEP 10: Saving Results")
    print(f"{'=' * 80}")

    # Save annotated dataset
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    save_jsonl(filtered_records, OUTPUT_FILE)
    print(f"✓ Saved {len(filtered_records)} annotated records to: {OUTPUT_FILE}")

    # Save summary JSON
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved summary to: {SUMMARY_FILE}")

    # 11. Create visualizations
    print(f"\n{'=' * 80}")
    print("STEP 11: Creating Visualizations")
    print(f"{'=' * 80}")

    try:
        # Heatmaps
        heatmap_path = f"{PLOT_DIR}/school_psychology_professor/school_psychology_professor_local/filtered_cases_heatmaps_{SUBJECT}_{HINT_TEMPLATE}_{TODAY}.png"
        plot_heatmaps(grouped, SUBJECT, HINT_TEMPLATE, heatmap_path)

        # Layer-wise breakdown plots
        plot_layer_wise_breakdown(grouped, PLOT_DIR)

        print(f"✓ All visualizations created successfully")
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nOutputs:")
    print(f"  - Annotated data: {OUTPUT_FILE}")
    print(f"  - Summary: {SUMMARY_FILE}")
    print(f"  - Heatmaps: {heatmap_path}")
    print(f"  - Layer-wise plots: {PLOT_DIR}/school_psychology_professor/school_psychology_professor_local/layer-wise steering performance/")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
