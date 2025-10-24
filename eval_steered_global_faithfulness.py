"""
eval_steered_global_faithfulness.py

Steered global faithfulness evaluation script.

This script:
1. Loads steered evaluation dataset
2. Groups records by (subject, hint_template, layer, coefficient)
3. For each configuration:w
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
    compute_config_metrics
)
# Plotting functions removed - need updating for new stratification
# from src.steered_plots import (...)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file - steered evaluation results
INPUT_FILE = "data/sprint4_2025-10-21/steered/steered_val_neg_bas_histXmeta_2025-10-19.jsonl"

# Subject
SUBJECT = "history_metadata"

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
OUTPUT_ANNOTATED = "data/sprint4_2025-10-21/annotated/steered/annotated_steered_global_neg_bas_val_histXmeta_2025-10-19.jsonl"

# Output 2: Summary JSON (all metrics, best configs)
OUTPUT_SUMMARY = "data/sprint4_2025-10-21/summaries/steered_faithfulness/summary_faithfulness_steered_neg_bas_val_histXmeta_2025-10-19.json"

# Output 3: Heatmaps plot (2x2 grid)
OUTPUT_PLOT_HEATMAPS = "data/sprint4_2025-10-21/plots/steering_neg_bas_histXmeta/heatmaps_steered_neg_bas_val_histXmeta_2025-10-19.png"

# Output 4: Best config breakdown plot
OUTPUT_PLOT_BREAKDOWN = "data/sprint4_2025-10-21/plots/steering_neg_bas_histXmeta/best_breakdown_steered_neg_bas_val_histXmeta_2025-10-19.png"

# Outputs 5+: Layer-wise plots directory (individual files generated automatically as steered_global_layers_coeff_{coeff}.png)
OUTPUT_PLOT_LAYERS_DIR = "data/sprint4_2025-10-21/plots/steering_neg" \
"_bas_histXmeta"

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
            if group_name in ['positive_on_CF', 'positive_on_CU', 'positive_on_WF', 'positive_on_WU',
                            'negative_on_CF', 'negative_on_CU', 'negative_on_WF', 'negative_on_WU']:
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


def create_summary(all_configs, subject, hint_template, input_file):
    """
    Create summary JSON with all configuration metrics.

    Args:
        all_configs: List of all configuration results
        subject: Subject name
        hint_template: Hint template name
        input_file: Input file path

    Returns:
        Summary dictionary
    """
    # Count total examples (sum across all 8 groups)
    total_examples = 0
    if all_configs:
        sample_config = all_configs[0]
        for group_name in ['positive_on_CF', 'positive_on_CU', 'positive_on_WF', 'positive_on_WU',
                          'negative_on_CF', 'negative_on_CU', 'negative_on_WF', 'negative_on_WU']:
            if group_name in sample_config:
                total_examples += sample_config[group_name]['n']
        # Divide by 2 (positive + negative steering on same records)
        total_examples = total_examples // 2

    # Create summary
    summary = {
        'evaluation_date': TODAY,
        'method': 'global_llm_judge_steered_with_stratification',
        'judge_model': JUDGE_MODEL,
        'source_file': input_file,
        'subject': subject,
        'hint_template': hint_template,
        'total_examples': total_examples,

        'dataset_info': {
            'total_configurations': len(all_configs),
            'layers': sorted(set(c['layer'] for c in all_configs)),
            'coefficient_magnitudes': sorted(set(c['coefficient_magnitude'] for c in all_configs)),
            'note': 'Stratified by initial state (CF/CU/WF/WU) - 8 groups per configuration'
        },

        'all_configurations': all_configs
    }

    return summary


def print_configs_summary(all_configs):
    """
    Print summary of all configurations with their transition metrics.

    Args:
        all_configs: List of all configuration results
    """
    print("\n" + "=" * 80)
    print("CONFIGURATIONS SUMMARY")
    print("=" * 80)
    print(f"Total configurations processed: {len(all_configs)}")
    print(f"Initial states tracked: CF, CU, WF, WU")
    print(f"Steering directions: positive (+), negative (-)")
    print(f"\nAll transition metrics saved to summary JSON file")
    print(f"Manual analysis required to select appropriate configurations")
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

        # 5. Print configuration summary
        print_configs_summary(all_configs)

        # 6. Save outputs for this hint template
        print(f"\n  Saving outputs for '{hint_template}'...")

        # 6a. Save annotated dataset (filter records for this hint template)
        template_records = [r for r in all_records if r.get('hint_template', 'unknown') == hint_template]
        annotated_records = create_annotated_records(all_configs, template_records)
        os.makedirs(os.path.dirname(OUTPUT_ANNOTATED), exist_ok=True)
        save_jsonl(annotated_records, OUTPUT_ANNOTATED)
        print(f"  ✓ Saved annotated dataset ({len(annotated_records)} records): {OUTPUT_ANNOTATED}")

        # 6b. Save summary
        summary = create_summary(all_configs, SUBJECT, hint_template, INPUT_FILE)
        os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved summary: {OUTPUT_SUMMARY}")

        # 7. Create plots
        print(f"\n  Creating visualizations for '{hint_template}'...")
        try:
            # Import plotting functions
            from src.steered_plots import (
                plot_steering_heatmaps,
                plot_transformation_rates_by_layer
            )

            # Generate 4 heatmap grids total:
            # 2 for CORRECT group (transitions + no_change)
            # 2 for WRONG group (transitions + no_change)

            print(f"\n    Generating heatmaps for CORRECT group...")

            # CORRECT - TRANSITIONS
            heatmap_path_correct_trans = OUTPUT_PLOT_HEATMAPS.replace('.png', '_correct_transitions.png')
            plot_steering_heatmaps(
                all_configs=all_configs,
                subject=SUBJECT,
                hint_template=hint_template,
                correctness_group='correct',
                heatmap_type='transitions',
                save_path=heatmap_path_correct_trans
            )

            # CORRECT - NO CHANGE
            heatmap_path_correct_nochange = OUTPUT_PLOT_HEATMAPS.replace('.png', '_correct_no_change.png')
            plot_steering_heatmaps(
                all_configs=all_configs,
                subject=SUBJECT,
                hint_template=hint_template,
                correctness_group='correct',
                heatmap_type='no_change',
                save_path=heatmap_path_correct_nochange
            )

            print(f"\n    Generating heatmaps for WRONG group...")

            # WRONG - TRANSITIONS
            heatmap_path_wrong_trans = OUTPUT_PLOT_HEATMAPS.replace('.png', '_wrong_transitions.png')
            plot_steering_heatmaps(
                all_configs=all_configs,
                subject=SUBJECT,
                hint_template=hint_template,
                correctness_group='wrong',
                heatmap_type='transitions',
                save_path=heatmap_path_wrong_trans
            )

            # WRONG - NO CHANGE
            heatmap_path_wrong_nochange = OUTPUT_PLOT_HEATMAPS.replace('.png', '_wrong_no_change.png')
            plot_steering_heatmaps(
                all_configs=all_configs,
                subject=SUBJECT,
                hint_template=hint_template,
                correctness_group='wrong',
                heatmap_type='no_change',
                save_path=heatmap_path_wrong_nochange
            )

            # Layer-wise line plots - one 2×2 per coefficient per correctness group
            print(f"\n    Generating layer-wise plots...")

            # CORRECT group line plots
            plot_transformation_rates_by_layer(
                all_configs=all_configs,
                correctness_group='correct',
                save_dir=OUTPUT_PLOT_LAYERS_DIR,
                subject=SUBJECT,
                hint_template=hint_template
            )

            # WRONG group line plots
            plot_transformation_rates_by_layer(
                all_configs=all_configs,
                correctness_group='wrong',
                save_dir=OUTPUT_PLOT_LAYERS_DIR,
                subject=SUBJECT,
                hint_template=hint_template
            )

            print(f"  ✓ Created all plots for '{hint_template}'")

        except Exception as e:
            import traceback
            print(f"  Warning: Could not create plots: {e}")
            print(f"  Full traceback:")
            traceback.print_exc()

        # Store outputs
        all_outputs[hint_template] = {
            'configs': all_configs,
            'annotated_file': OUTPUT_ANNOTATED,
            'summary_file': OUTPUT_SUMMARY
        }

    # Final summary
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"✓ Processed {len(all_records)} steered responses")
    print(f"✓ Processed {len(hint_templates_in_grouped)} hint template(s): {hint_templates_in_grouped}")

    for hint_template, outputs in all_outputs.items():
        print(f"\n  [{hint_template}]")
        print(f"    - Analyzed {len(outputs['configs'])} configurations")
        print(f"    - Stratified by initial state: CF, CU, WF, WU")
        print(f"    - Outputs:")
        print(f"      - Annotated data: {outputs['annotated_file']}")
        print(f"      - Summary JSON: {outputs['summary_file']}")
        print(f"      - Plots (4 initial states × multiple coefficients):")
        print(f"        - Heatmaps: {os.path.dirname(OUTPUT_PLOT_HEATMAPS)}/")
        print(f"        - Line plots: {OUTPUT_PLOT_LAYERS_DIR}/")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
