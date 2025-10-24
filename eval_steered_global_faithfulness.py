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
from src.config import TODAY

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
INPUT_FILE = "data/sprint4_2025-10-21/steered/steered_val_neg_bas_psyXprof_2025-10-19.jsonl"

# Subject
SUBJECT = "psychology_professor"

# Model configuration
JUDGE_MODEL = "google/gemini-2.5-flash"  # OpenRouter model name
MAX_RETRIES = 3
TOP_K = 5  # Number of top configs to report

# =============================================================================
# OUTPUT PATHS CONFIGURATION
# =============================================================================
# Set all output file paths manually here.
# The script will save exactly to these paths (creating directories as needed).

# Output 1: Annotated dataset (JSONL with classifications)
OUTPUT_ANNOTATED = "data/sprint4_2025-10-21/annotated/steered/annotated_steered_global_val_neg_bas_psyXprof_2025-10-19.jsonl"

# Output 2: Summary JSON (all metrics, best configs)
OUTPUT_SUMMARY = "data/sprint4_2025-10-21/summaries/steered_faithfulness/summary_faithfulness_steered_global_val_neg_bas_psyXprof_2025-10-19.json"

# Output 3: Heatmaps plot (2x2 grid)
OUTPUT_PLOT_HEATMAPS = "plots/sprint4_2025-10-21/steering_neg_bas_psyXprof/heatmaps_steered_global_val_neg_bas_psyXprof_2025-10-19.png"

# Output 4: Best config breakdown plot
OUTPUT_PLOT_BREAKDOWN = "plots/sprint4_2025-10-21/steering_neg_bas_psyXprof/best_breakdown_steered_global_val_neg_bas_psyXprof_2025-10-19.png"

# Outputs 5+: Layer-wise plots directory (individual files generated automatically as steered_global_layers_coeff_{coeff}.png)
OUTPUT_PLOT_LAYERS_DIR = "plots/sprint4_2025-10-21/steering_neg_bas_psyXprof"

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
                    'transitions': best_configs['positive_steering']['best']['config']['positive_on_unfaithful']['transitions']
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
                    'transitions': best_configs['negative_steering']['best']['config']['negative_on_faithful']['transitions']
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
        print(f"\n  Saving outputs for '{hint_template}'...")

        # 6a. Save annotated dataset (filter records for this hint template)
        template_records = [r for r in all_records if r.get('hint_template', 'unknown') == hint_template]
        annotated_records = create_annotated_records(all_configs, template_records)
        os.makedirs(os.path.dirname(OUTPUT_ANNOTATED), exist_ok=True)
        save_jsonl(annotated_records, OUTPUT_ANNOTATED)
        print(f"  ✓ Saved annotated dataset ({len(annotated_records)} records): {OUTPUT_ANNOTATED}")

        # 6b. Save summary
        summary = create_summary(all_configs, best_configs, SUBJECT, hint_template, INPUT_FILE)
        os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved summary: {OUTPUT_SUMMARY}")

        # 7. Create plots
        print(f"\n  Creating visualizations for '{hint_template}'...")
        try:
            # Create heatmaps
            os.makedirs(os.path.dirname(OUTPUT_PLOT_HEATMAPS), exist_ok=True)
            plot_steering_heatmaps(all_configs, SUBJECT, hint_template, OUTPUT_PLOT_HEATMAPS)

            # Create breakdown
            plot_best_config_breakdown(best_configs['positive_steering']['best'],
                                       best_configs['negative_steering']['best'],
                                       OUTPUT_PLOT_BREAKDOWN)

            # Create layer-wise plots (automatically generate filenames in configured directory)
            os.makedirs(OUTPUT_PLOT_LAYERS_DIR, exist_ok=True)

            # Get unique coefficient magnitudes from data
            coeffs = sorted(set(c['coefficient_magnitude'] for c in all_configs))
            layer_plot_paths = []

            for coeff in coeffs:
                # Generate filename automatically: steered_global_layers_coeff_{coeff}.png
                save_path = os.path.join(OUTPUT_PLOT_LAYERS_DIR, f"steered_global_layers_coeff_{coeff}.png")
                layer_plot_paths.append(save_path)

                # Filter configs for this coefficient
                coeff_configs = [c for c in all_configs if c['coefficient_magnitude'] == coeff]

                # Create the layer-wise plot
                import matplotlib.pyplot as plt
                from src.steered_plots import _plot_transitions_subplot
                layers = sorted(set(c['layer'] for c in all_configs))

                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Transformation Rates Across Layers (Coefficient = ±{coeff})',
                            fontsize=16, fontweight='bold')

                _plot_transitions_subplot(coeff_configs, layers, 'positive_on_unfaithful', 'unfaithful',
                                        axes[0, 0], f'Unfaithful Origin + Positive Steering (coeff = +{coeff})')
                _plot_transitions_subplot(coeff_configs, layers, 'negative_on_unfaithful', 'unfaithful',
                                        axes[0, 1], f'Unfaithful Origin + Negative Steering (coeff = -{coeff})')
                _plot_transitions_subplot(coeff_configs, layers, 'positive_on_faithful', 'faithful',
                                        axes[1, 0], f'Faithful Origin + Positive Steering (coeff = +{coeff})')
                _plot_transitions_subplot(coeff_configs, layers, 'negative_on_faithful', 'faithful',
                                        axes[1, 1], f'Faithful Origin + Negative Steering (coeff = -{coeff})')

                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   Saved layer comparison for coeff ±{coeff} to: {save_path}")

            print(f"  ✓ All plots created successfully")
        except Exception as e:
            import traceback
            print(f"  Warning: Could not create plots: {e}")
            print(f"  Full traceback:")
            traceback.print_exc()

        # Store outputs
        all_outputs[hint_template] = {
            'configs': all_configs,
            'best_configs': best_configs,
            'annotated_file': OUTPUT_ANNOTATED,
            'summary_file': OUTPUT_SUMMARY,
            'plots': {
                'heatmaps': OUTPUT_PLOT_HEATMAPS,
                'breakdown': OUTPUT_PLOT_BREAKDOWN,
                'layers': layer_plot_paths
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
        for layer_plot in outputs['plots']['layers']:
            print(f"      - {layer_plot}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
