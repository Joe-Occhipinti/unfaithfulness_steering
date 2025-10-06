"""
analyze_filtered_categories.py

Analyzes the "filtered" categories (correct, hint-induced error, incomplete)
by calling the global LLM annotator to determine if they would be classified
as "faithful" or "unfaithful", then plots the distribution.

This script follows the same processing pattern as eval_steered_global_faithfulness.py:
1. Groups records by (hint_template, layer, coefficient_magnitude)
2. Within each group, separates by original faithfulness state
3. Filters only 'correct', 'hint-induced error', 'incomplete' from rule-based classification
4. Calls global LLM judge on these filtered records
5. Plots distribution of faithful/unfaithful within each filtered category
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables
load_dotenv()

# Import existing modules
from src.data import load_jsonl
from src.global_faithfulness import setup_openrouter_client, judge_batch
from src.config import TODAY

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file - annotated steered dataset
INPUT_FILE = "data/annotated/annotated_steered_local_high_school_psychology_professor_2025-10-06.jsonl"

# Output directory for plots
PLOT_DIR = "plots"

# Model configuration
JUDGE_MODEL = "google/gemini-2.5-flash"
MAX_RETRIES = 3

# Categories to analyze (these are the rule-based filtered categories)
CATEGORIES_TO_ANALYZE = ['correct', 'hint-induced error', 'incomplete']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def group_records_by_config_and_category(records):
    """
    Group records by (hint_template, layer, coefficient_magnitude, category, original_state).

    This mirrors the grouping structure in eval_steered_global_faithfulness.py
    but adds an additional dimension for the filtered category.

    Args:
        records: All annotated steered records

    Returns:
        Nested dict: {(hint_template, layer, coeff_mag, category): {'positive_on_unfaithful': [...], ...}}
    """
    grouped = defaultdict(lambda: {
        'positive_on_unfaithful': [],
        'positive_on_faithful': [],
        'negative_on_unfaithful': [],
        'negative_on_faithful': []
    })

    for record in records:
        # Only include records in our filtered categories
        classification = record.get('steered_global_faithfulness_classification', '')
        if classification not in CATEGORIES_TO_ANALYZE:
            continue

        hint_template = record.get('hint_template', 'unknown')
        layer = record['steering_layer']
        coeff = record['steering_coefficient']
        orig_faith = record['original_faithfulness_classification']

        # Determine group based on coefficient sign and original state
        if coeff > 0 and orig_faith == 'unfaithful':
            group = 'positive_on_unfaithful'
        elif coeff > 0 and orig_faith == 'faithful':
            group = 'positive_on_faithful'
        elif coeff < 0 and orig_faith == 'faithful':
            group = 'negative_on_faithful'
        elif coeff < 0 and orig_faith == 'unfaithful':
            group = 'negative_on_unfaithful'
        else:
            continue

        # Key by (hint_template, layer, abs(coefficient), category)
        key = (hint_template, layer, abs(coeff), classification)
        grouped[key][group].append(record)

    return dict(grouped)


def aggregate_by_category(grouped_data):
    """
    Aggregate grouped data by category, collapsing across layers and coefficients.

    Args:
        grouped_data: Output from group_records_by_config_and_category

    Returns:
        Dict: {category: {'all_records': [...], 'by_original_state': {...}}}
    """
    category_data = defaultdict(lambda: {
        'all_records': [],
        'positive_on_unfaithful': [],
        'positive_on_faithful': [],
        'negative_on_unfaithful': [],
        'negative_on_faithful': []
    })

    for (hint_template, layer, coeff_mag, category), groups in grouped_data.items():
        for group_name, records in groups.items():
            category_data[category][group_name].extend(records)
            category_data[category]['all_records'].extend(records)

    return dict(category_data)


def print_grouped_statistics(grouped_data):
    """Print statistics about the grouped data structure."""
    print(f"\nGrouped data statistics:")

    # Count unique configurations
    hint_templates = set()
    layers = set()
    coeffs = set()
    categories = set()

    for (ht, layer, coeff, cat), groups in grouped_data.items():
        hint_templates.add(ht)
        layers.add(layer)
        coeffs.add(coeff)
        categories.add(cat)

    print(f"  Hint templates: {sorted(hint_templates)}")
    print(f"  Layers: {sorted(layers)}")
    print(f"  Coefficient magnitudes: {sorted(coeffs)}")
    print(f"  Categories: {sorted(categories)}")
    print(f"  Total unique configurations: {len(grouped_data)}")

    # Print breakdown by category
    print(f"\nRecords per category:")
    category_counts = defaultdict(int)
    for (ht, layer, coeff, cat), groups in grouped_data.items():
        for group_name, records in groups.items():
            category_counts[cat] += len(records)

    for cat in sorted(category_counts.keys()):
        print(f"  {cat}: {category_counts[cat]} records")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for filtered category analysis."""
    print(f"=== FILTERED CATEGORY ANALYSIS - {TODAY} ===")

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Input file not found: {INPUT_FILE}")
        return

    # 1. Load data
    print(f"\n{'=' * 80}")
    print("STEP 1: Loading Data")
    print(f"{'=' * 80}")
    print(f"Loading annotated dataset from: {INPUT_FILE}")

    all_records = load_jsonl(INPUT_FILE)
    print(f"✓ Loaded {len(all_records)} records")

    # 2. Group records by configuration and category
    print(f"\n{'=' * 80}")
    print("STEP 2: Grouping Records by Configuration and Category")
    print(f"{'=' * 80}")

    grouped_data = group_records_by_config_and_category(all_records)
    print(f"✓ Created {len(grouped_data)} unique configuration-category groups")
    print_grouped_statistics(grouped_data)

    # 3. Aggregate by category (collapse across layers/coefficients)
    print(f"\n{'=' * 80}")
    print("STEP 3: Aggregating by Category")
    print(f"{'=' * 80}")

    category_data = aggregate_by_category(grouped_data)

    print(f"Category breakdown:")
    total_filtered = 0
    for category in CATEGORIES_TO_ANALYZE:
        count = len(category_data.get(category, {}).get('all_records', []))
        total_filtered += count
        print(f"  {category}: {count} records")
    print(f"Total filtered records: {total_filtered}")

    if total_filtered == 0:
        print("\nNo records found in the filtered categories. Exiting.")
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

    # 5. Judge each category
    print(f"\n{'=' * 80}")
    print("STEP 5: Judging Filtered Categories")
    print(f"{'=' * 80}")

    category_judgments = {}

    for category in CATEGORIES_TO_ANALYZE:
        if category not in category_data:
            print(f"\n  [{category}]: No records, skipping")
            category_judgments[category] = {
                'faithful': 0,
                'unfaithful': 0,
                'error': 0,
                'total': 0
            }
            continue

        records = category_data[category]['all_records']

        print(f"\n  [{category}]: Judging {len(records)} records...")

        # Call judge_batch (same as eval_steered_global_faithfulness.py)
        judgments = judge_batch(
            results=records,
            client=client,
            model=JUDGE_MODEL,
            max_retries=MAX_RETRIES,
            verbose=True
        )

        # Count results
        faithful_count = 0
        unfaithful_count = 0
        error_count = 0

        for judgment in judgments:
            if judgment['success']:
                if judgment['classification'] == 'faithful':
                    faithful_count += 1
                elif judgment['classification'] == 'unfaithful':
                    unfaithful_count += 1
            else:
                error_count += 1

        category_judgments[category] = {
            'faithful': faithful_count,
            'unfaithful': unfaithful_count,
            'error': error_count,
            'total': len(records)
        }

        print(f"    Faithful: {faithful_count}")
        print(f"    Unfaithful: {unfaithful_count}")
        if error_count > 0:
            print(f"    Errors: {error_count}")

    # 6. Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n{'Category':<22} {'Total':<10} {'Faithful':<12} {'Unfaithful':<12} {'Errors':<10}")
    print("─" * 70)
    for category in CATEGORIES_TO_ANALYZE:
        counts = category_judgments[category]
        print(f"{category:<22} {counts['total']:<10} {counts['faithful']:<12} "
              f"{counts['unfaithful']:<12} {counts['error']:<10}")

    # 7. Create plots
    print(f"\n{'=' * 80}")
    print("STEP 6: Creating Visualizations")
    print(f"{'=' * 80}")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Prepare data for plotting
    categories = CATEGORIES_TO_ANALYZE
    faithful_counts = [category_judgments[cat]['faithful'] for cat in categories]
    unfaithful_counts = [category_judgments[cat]['unfaithful'] for cat in categories]
    error_counts = [category_judgments[cat]['error'] for cat in categories]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width, faithful_counts, width, label='Faithful', color='#2ecc71')
    bars2 = ax.bar(x, unfaithful_counts, width, label='Unfaithful', color='#e74c3c')
    bars3 = ax.bar(x + width, error_counts, width, label='Errors', color='#95a5a6')

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Faithful vs Unfaithful Distribution in Filtered Categories', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.tight_layout()

    # Save plot
    output_plot = os.path.join(PLOT_DIR, f"filtered_categories_distribution_{TODAY}.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_plot}")

    # Create percentage stacked bar chart (excluding errors)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate percentages (excluding errors)
    totals = [faithful_counts[i] + unfaithful_counts[i] for i in range(len(categories))]
    faithful_pct = [(faithful_counts[i] / totals[i] * 100 if totals[i] > 0 else 0) for i in range(len(categories))]
    unfaithful_pct = [(unfaithful_counts[i] / totals[i] * 100 if totals[i] > 0 else 0) for i in range(len(categories))]

    bars1 = ax.bar(categories, faithful_pct, label='Faithful', color='#2ecc71')
    bars2 = ax.bar(categories, unfaithful_pct, bottom=faithful_pct, label='Unfaithful', color='#e74c3c')

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Faithful vs Unfaithful Percentage Distribution (Excluding Errors)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (cat, f_pct, u_pct) in enumerate(zip(categories, faithful_pct, unfaithful_pct)):
        if f_pct > 5:
            ax.text(i, f_pct/2, f'{f_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        if u_pct > 5:
            ax.text(i, f_pct + u_pct/2, f'{u_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')

    plt.tight_layout()

    output_plot_pct = os.path.join(PLOT_DIR, f"filtered_categories_percentage_{TODAY}.png")
    plt.savefig(output_plot_pct, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_plot_pct}")

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
