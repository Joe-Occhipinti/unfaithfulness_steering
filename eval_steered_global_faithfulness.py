"""
eval_steered_global_faithfulness.py

Steered global faithfulness evaluation script.

This script:
1. Loads steered evaluation dataset
2. Groups records by (subject, hint_template, layer, coefficient)
3. For each configuration:
   a. Rule-based classification (complete, correct, hint-error, needs-judge)
   b. LLM judge for ambiguous cases
   c. Compute transition rates
   d. Statistical significance tests
4. Find best configurations (maximize success - side_effects)
5. Save annotated dataset + summary + plots
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import existing modules
from src.data import load_jsonl, save_jsonl
from src.global_faithfulness import setup_openrouter_client
from src.config import TODAY, ANNOTATED_DIR, SUMMARIES_DIR

# Import new modules
from src.steered_global_faithfulness import (
    group_records_by_config,
    compute_config_metrics,
    find_best_configs
)
from src.steered_plots import (
    plot_steering_heatmaps,
    plot_best_config_breakdown,
    plot_transformation_rates_by_layer
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file - steered evaluation results
INPUT_FILE = "data/behavioural/steered_val_global_biased_high_school_psychology_2025-08-15.jsonl"

# Subject (auto-extracted from filename, or set manually)
SUBJECT = "high_school_psychology"

# Plotting
PLOT_DIR = "plots"

# Model configuration
JUDGE_MODEL = "google/gemini-2.5-flash"
MAX_RETRIES = 3
TOP_K = 5  # Number of top configs to report

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_annotated_records(all_configs, original_records):
    """
    Create annotated dataset by merging classifications back into original records.

    Args:
        all_configs: List of all configuration results with classifications
        original_records: Original steered records

    Returns:
        List of annotated records with global faithfulness classifications
    """
    # Build classification lookup: (question_id, layer, coeff) -> classification
    classification_lookup = {}

    for config in all_configs:
        layer = config['layer']
        coeff_mag = config['coefficient_magnitude']

        for group_name, group_data in config.items():
            if group_name in ['positive_on_unfaithful', 'positive_on_faithful',
                            'negative_on_faithful', 'negative_on_unfaithful']:
                classifications = group_data.get('classifications', {})
                for qid, classification in classifications.items():
                    # Determine coefficient sign from group name
                    if 'positive' in group_name:
                        coeff = coeff_mag
                    else:
                        coeff = -coeff_mag

                    key = (qid, layer, coeff)
                    classification_lookup[key] = classification

    # Annotate original records
    annotated = []
    for record in original_records:
        qid = record.get('question_id', record.get('prompt_index'))
        layer = record['steering_layer']
        coeff = record['steering_coefficient']

        key = (qid, layer, coeff)
        classification = classification_lookup.get(key, 'error')

        # Create annotated record
        annotated_record = record.copy()
        annotated_record['steered_global_faithfulness_classification'] = classification

        # Add tagged prompt for potential future use
        steered_prompt = record.get('steered_prompt', '')
        if classification == 'faithful':
            annotated_record['annotated_steered_prompt'] = f"[F_final]{steered_prompt}[/F_final]"
        elif classification == 'unfaithful':
            annotated_record['annotated_steered_prompt'] = f"[U_final]{steered_prompt}[/U_final]"
        else:
            annotated_record['annotated_steered_prompt'] = steered_prompt

        annotated.append(annotated_record)

    return annotated


def create_summary(all_configs, best_configs, subject, hint_template, input_file):
    """
    Create summary JSON with all metrics and best configurations.

    Args:
        all_configs: List of all configuration results
        best_configs: Best configs for positive and negative steering
        subject: Subject name
        hint_template: Hint template name
        input_file: Input file path

    Returns:
        Summary dictionary
    """
    # Count total examples
    total_examples = sum(
        config['positive_on_unfaithful']['n'] +
        config['positive_on_faithful']['n'] +
        config['negative_on_faithful']['n'] +
        config['negative_on_unfaithful']['n']
        for config in all_configs
    ) // (len(all_configs) * 2)  # Divide by num configs and 2 (pos/neg)

    # Create summary
    summary = {
        'evaluation_date': TODAY,
        'method': 'global_llm_judge_steered',
        'judge_model': JUDGE_MODEL,
        'source_file': input_file,
        'subject': subject,
        'hint_template': hint_template,
        'total_examples': total_examples,

        'dataset_info': {
            'total_configurations': len(all_configs),
            'layers': sorted(set(c['layer'] for c in all_configs)),
            'coefficient_magnitudes': sorted(set(c['coefficient_magnitude'] for c in all_configs))
        },

        'best_configurations': {
            'positive_steering': {
                'best': {
                    'layer': best_configs['positive_steering']['best']['layer'],
                    'coefficient': best_configs['positive_steering']['best']['coefficient'],
                    'score': best_configs['positive_steering']['best']['score'],
                    'success_rate': best_configs['positive_steering']['best']['success_rate'],
                    'side_effects_rate': best_configs['positive_steering']['best']['side_effects_rate'],
                    'transitions': best_configs['positive_steering']['best']['config']['positive_on_unfaithful']['transitions'],
                    'statistical_tests': best_configs['positive_steering']['best']['config']['positive_on_unfaithful']['statistical_tests']
                },
                'top_k': [
                    {
                        'rank': i + 1,
                        'layer': cfg['layer'],
                        'coefficient': cfg['coefficient'],
                        'score': cfg['score'],
                        'success_rate': cfg['success_rate'],
                        'side_effects_rate': cfg['side_effects_rate']
                    }
                    for i, cfg in enumerate(best_configs['positive_steering']['top_k'])
                ]
            },

            'negative_steering': {
                'best': {
                    'layer': best_configs['negative_steering']['best']['layer'],
                    'coefficient': best_configs['negative_steering']['best']['coefficient'],
                    'score': best_configs['negative_steering']['best']['score'],
                    'success_rate': best_configs['negative_steering']['best']['success_rate'],
                    'side_effects_rate': best_configs['negative_steering']['best']['side_effects_rate'],
                    'transitions': best_configs['negative_steering']['best']['config']['negative_on_faithful']['transitions'],
                    'statistical_tests': best_configs['negative_steering']['best']['config']['negative_on_faithful']['statistical_tests']
                },
                'top_k': [
                    {
                        'rank': i + 1,
                        'layer': cfg['layer'],
                        'coefficient': cfg['coefficient'],
                        'score': cfg['score'],
                        'success_rate': cfg['success_rate'],
                        'side_effects_rate': cfg['side_effects_rate']
                    }
                    for i, cfg in enumerate(best_configs['negative_steering']['top_k'])
                ]
            }
        },

        'all_configurations': all_configs
    }

    return summary


def print_summary_table(best_configs):
    """
    Print formatted summary table of best configurations.

    Args:
        best_configs: Best configs for positive and negative steering
    """
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS SUMMARY")
    print("=" * 80)

    print(f"\nTop {TOP_K} Positive Steering Configurations:")
    print("─" * 80)
    print(f"{'Rank':<6} {'Layer':<7} {'Coeff':<8} {'Score':<8} {'Success':<10} {'Side Effects':<12}")
    print("─" * 80)

    for i, cfg in enumerate(best_configs['positive_steering']['top_k'], 1):
        print(f"{i:<6} {cfg['layer']:<7} +{cfg['coefficient']:<7.2f} "
              f"{cfg['score']:<8.3f} {cfg['success_rate']:<10.1%} {cfg['side_effects_rate']:<12.1%}")

    print(f"\nTop {TOP_K} Negative Steering Configurations:")
    print("─" * 80)
    print(f"{'Rank':<6} {'Layer':<7} {'Coeff':<8} {'Score':<8} {'Success':<10} {'Side Effects':<12}")
    print("─" * 80)

    for i, cfg in enumerate(best_configs['negative_steering']['top_k'], 1):
        print(f"{i:<6} {cfg['layer']:<7} {cfg['coefficient']:<7.2f} "
              f"{cfg['score']:<8.3f} {cfg['success_rate']:<10.1%} {cfg['side_effects_rate']:<12.1%}")

    print("=" * 80)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for steered global faithfulness evaluation."""
    print(f"=== STEERED GLOBAL FAITHFULNESS EVALUATION - {TODAY} ===")
    print(f"Subject: {SUBJECT}")

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

    # Detect hint templates in dataset
    hint_templates = sorted(set(r.get('hint_template', 'unknown') for r in all_records))
    print(f"✓ Detected hint templates: {hint_templates}")

    # 2. Setup API client
    print(f"\n{'=' * 80}")
    print("STEP 2: Setting Up API Client")
    print(f"{'=' * 80}")

    try:
        client = setup_openrouter_client()
        print(f"✓ OpenRouter client initialized")
        print(f"  Judge Model: {JUDGE_MODEL}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return

    # 3. Group records
    print(f"\n{'=' * 80}")
    print("STEP 3: Grouping Records by Configuration")
    print(f"{'=' * 80}")

    grouped = group_records_by_config(all_records)
    print(f"✓ Found {len(grouped)} unique (hint_template, layer, coefficient) configurations")

    # Print configuration details
    hint_templates_in_grouped = sorted(set(k[0] for k in grouped.keys()))
    layers = sorted(set(k[1] for k in grouped.keys()))
    coeffs = sorted(set(k[2] for k in grouped.keys()))
    print(f"  Hint templates: {hint_templates_in_grouped}")
    print(f"  Layers: {layers}")
    print(f"  Coefficient magnitudes: {coeffs}")

    # 4. Process each hint template separately
    print(f"\n{'=' * 80}")
    print("STEP 4: Processing Each Hint Template")
    print(f"{'=' * 80}")

    all_outputs = {}  # Store outputs per hint template

    for hint_template in hint_templates_in_grouped:
        print(f"\n{'*' * 80}")
        print(f"PROCESSING HINT TEMPLATE: {hint_template}")
        print(f"{'*' * 80}")

        # Filter configs for this hint template
        template_configs = [(k, v) for k, v in grouped.items() if k[0] == hint_template]
        print(f"  Found {len(template_configs)} configurations for '{hint_template}'")

        all_configs = []

        for (ht, layer, coeff_mag), config_groups in template_configs:
            config_result = compute_config_metrics(
                config_groups,
                hint_template=ht,
                layer=layer,
                coeff_mag=coeff_mag,
                client=client,
                model=JUDGE_MODEL,
                verbose=True
            )
            all_configs.append(config_result)

        print(f"\n✓ Processed all {len(all_configs)} configurations for '{hint_template}'")

        # 5. Find best configurations for this hint template
        print(f"\n  Finding best configurations for '{hint_template}'...")
        best_configs = find_best_configs(all_configs, top_k=TOP_K)
        print(f"  ✓ Identified top {TOP_K} configs for each steering direction")

        # Print summary table
        print_summary_table(best_configs)

        # 6. Save outputs for this hint template
        OUTPUT_ANNOTATED = f"data/annotated/annotated_steered_global_{SUBJECT}_{hint_template}_{TODAY}.jsonl"
        OUTPUT_SUMMARY = f"data/summaries/faithfulness_steered_global_{SUBJECT}_{hint_template}_{TODAY}.json"
        PLOT_HEATMAPS = f"{PLOT_DIR}/steered_global_heatmaps_{SUBJECT}_{hint_template}_{TODAY}.png"
        PLOT_BREAKDOWN = f"{PLOT_DIR}/steered_global_best_breakdown_{SUBJECT}_{hint_template}_{TODAY}.png"

        print(f"\n  Saving outputs for '{hint_template}'...")

        # 6a. Save annotated dataset (filter records for this hint template)
        template_records = [r for r in all_records if r.get('hint_template', 'unknown') == hint_template]
        annotated_records = create_annotated_records(all_configs, template_records)
        os.makedirs(ANNOTATED_DIR, exist_ok=True)
        save_jsonl(annotated_records, OUTPUT_ANNOTATED)
        print(f"  ✓ Saved annotated dataset ({len(annotated_records)} records): {OUTPUT_ANNOTATED}")

        # 6b. Save summary
        summary = create_summary(all_configs, best_configs, SUBJECT, hint_template, INPUT_FILE)
        os.makedirs(SUMMARIES_DIR, exist_ok=True)
        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved summary: {OUTPUT_SUMMARY}")

        # 7. Create plots
        print(f"\n  Creating visualizations for '{hint_template}'...")
        try:
            plot_steering_heatmaps(all_configs, SUBJECT, hint_template, PLOT_HEATMAPS)
            plot_best_config_breakdown(best_configs['positive_steering']['best'],
                                       best_configs['negative_steering']['best'],
                                       PLOT_BREAKDOWN)
            plot_transformation_rates_by_layer(all_configs, PLOT_DIR)
            print(f"  ✓ All plots created successfully")
        except Exception as e:
            print(f"  Warning: Could not create plots: {e}")

        # Store outputs
        all_outputs[hint_template] = {
            'configs': all_configs,
            'best_configs': best_configs,
            'annotated_file': OUTPUT_ANNOTATED,
            'summary_file': OUTPUT_SUMMARY,
            'plots': {
                'heatmaps': PLOT_HEATMAPS,
                'breakdown': PLOT_BREAKDOWN
            }
        }

    # Final summary
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"✓ Processed {len(all_records)} steered responses")
    print(f"✓ Processed {len(hint_templates_in_grouped)} hint template(s): {hint_templates_in_grouped}")

    for hint_template, outputs in all_outputs.items():
        best = outputs['best_configs']
        print(f"\n  [{hint_template}]")
        print(f"    - Analyzed {len(outputs['configs'])} configurations")
        print(f"    - Best Positive: Layer {best['positive_steering']['best']['layer']}, "
              f"Coeff +{best['positive_steering']['best']['coefficient']:.2f} "
              f"(Score: {best['positive_steering']['best']['score']:.3f})")
        print(f"    - Best Negative: Layer {best['negative_steering']['best']['layer']}, "
              f"Coeff {best['negative_steering']['best']['coefficient']:.2f} "
              f"(Score: {best['negative_steering']['best']['score']:.3f})")
        print(f"    - Outputs:")
        print(f"      - {outputs['annotated_file']}")
        print(f"      - {outputs['summary_file']}")
        print(f"      - {outputs['plots']['heatmaps']}")
        print(f"      - {outputs['plots']['breakdown']}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
