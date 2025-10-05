"""
eval_faithfulness.py

Standalone faithfulness evaluation script that can handle both:
1. Hinted evaluation results - annotates biased prompts for faithfulness
2. Steered evaluation results - evaluates steering effectiveness on unfaithful prompts

Configure the mode and input files at the top of the script.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.faithfulness_eval import (
    setup_openrouter_client, annotate_batch,
    compute_faithfulness_metrics, print_faithfulness_report
)
from src.config import TODAY, ANNOTATED_DIR

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Mode selection: "hinted" or "steered"
MODE = "hinted"

# For HINTED mode - input and output files
HINTED_INPUT_FILE = "data/behavioural/hinted_high_school_psychology_2025-08-15.jsonl"
HINTED_OUTPUT_FILE = "data/annotated/annotated_biased_high_school_psychology_2025-08-15.jsonl"
HINTED_SUMMARY_FILE = "data/summaries/faithfulness_hinted_high_school_psychology_2025-08-15.json"

# For STEERED mode - input and output files
STEERED_INPUT_FILE = "data/behavioural/steered_val_high_school_macroeconomics_microeconomics_2025-10-03.jsonl"
STEERED_OUTPUT_FILE = "data/annotated/annotated_steered_high_school_macroeconomics_microeconomics_2025-10-03.jsonl"
STEERED_SUMMARY_FILE = "data/summaries/faithfulness_steered_high_school_macroeconomics_microeconomics_2025-10-03.json"

# =============================================================================
# END CONFIGURATION
# =============================================================================


def evaluate_hinted_faithfulness():
    """
    Evaluate faithfulness of hinted evaluation results.
    Loads hinted results and annotates biased prompts.
    """
    print(f"=== HINTED FAITHFULNESS EVALUATION - {TODAY} ===")

    # Check if hinted results file exists
    if not os.path.exists(HINTED_INPUT_FILE):
        print(f"Error: Hinted results file not found: {HINTED_INPUT_FILE}")
        print("Please run hinted_eval.py first to generate the results.")
        return

    print(f"Loading hinted evaluation results from: {HINTED_INPUT_FILE}")

    # Load the saved hinted results from file
    hinted_results = load_jsonl(HINTED_INPUT_FILE)
    print(f"Loaded {len(hinted_results)} hinted evaluation results")

    # Setup OpenRouter client for annotation
    try:
        openrouter_client = setup_openrouter_client()
    except ValueError as e:
        print(f"Error setting up OpenRouter client: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return

    # Filter for biased results only (where model followed the hint)
    biased_results = [r for r in hinted_results if r['bias_label'] == 'biased']
    print(f"Found {len(biased_results)} biased results to annotate for faithfulness")

    if len(biased_results) == 0:
        print("No biased results to annotate - all models resisted the hints!")
        print("Faithfulness evaluation complete (no work needed).")
        return

    # Annotate biased prompts for faithfulness
    print("\nAnnotating biased prompts for faithfulness...")

    annotations = annotate_batch(
        results=biased_results,
        client=openrouter_client,
        max_retries=3
    )

    # Create annotated results with the annotation data
    annotated_results = []
    for result, annotation in zip(biased_results, annotations):
        # Create a copy of the result with annotation fields, renaming hinted -> biased
        annotated_result = result.copy()

        # Rename key fields from hinted to biased since these are confirmed biased cases
        if 'hinted_input_prompt' in annotated_result:
            annotated_result['biased_input_prompt'] = annotated_result.pop('hinted_input_prompt')
        if 'hinted_answer' in annotated_result:
            annotated_result['biased_answer'] = annotated_result.pop('hinted_answer')
        if 'hinted_answer_letter' in annotated_result:
            annotated_result['biased_answer_letter'] = annotated_result.pop('hinted_answer_letter')
        if 'hinted_prompt' in annotated_result:
            annotated_result['biased_prompt'] = annotated_result.pop('hinted_prompt')

        annotated_result['annotated_biased_prompt'] = annotation.get('annotated_text')
        annotated_result['faithfulness_classification'] = annotation.get('classification')
        annotated_results.append(annotated_result)

    # Compute faithfulness metrics
    faithfulness_metrics = compute_faithfulness_metrics(annotations)

    # Print faithfulness report
    print_faithfulness_report(faithfulness_metrics)

    # Save annotated results to separate file
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    save_jsonl(annotated_results, HINTED_OUTPUT_FILE)
    print(f"\nSaved {len(annotated_results)} annotated biased results to {HINTED_OUTPUT_FILE}")

    # Create and save faithfulness summary
    faithfulness_summary = {
        'evaluation_date': TODAY,
        'source_file': HINTED_INPUT_FILE,
        'total_biased_results': len(biased_results),
        'total_annotated': len(annotated_results),
        'faithfulness_metrics': faithfulness_metrics,
        'annotated_output_file': HINTED_OUTPUT_FILE
    }

    with open(HINTED_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(faithfulness_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved faithfulness summary to {HINTED_SUMMARY_FILE}")

    print(f"\n=== FAITHFULNESS EVALUATION COMPLETE ===")
    print(f"+ Faithfulness evaluation completed:")
    print(f"   + Loaded hinted results from: {HINTED_INPUT_FILE}")
    print(f"   + Filtered {len(biased_results)} biased results")
    print(f"   + Annotated biased prompts with OpenRouter")
    print(f"   + Classified faithfulness (correct/faithful/unfaithful/hint-induced error)")
    print(f"   + Saved annotated results to: {HINTED_OUTPUT_FILE}")
    print(f"   + Saved faithfulness summary")

    # Optional: Create faithfulness distribution plots
    try:
        from src.plots import plot_faithfulness_distribution, plot_faithfulness_by_bias

        os.makedirs("plots", exist_ok=True)

        # Overall faithfulness distribution
        faithfulness_plot_path = f"plots/faithfulness_distribution_high_school_psychology_2025-08-15.png"
        plot_faithfulness_distribution(
            hinted_results=annotated_results,
            save_path=faithfulness_plot_path,
            show_plot=False
        )
        print(f"   + Faithfulness distribution plot saved to {faithfulness_plot_path}")

        # Bias-wise faithfulness distribution
        bias_wise_plot_path = f"plots/faithfulness_by_hint_high_school_psychology_2025-08-15.png"
        plot_faithfulness_by_bias(
            hinted_results=annotated_results,
            save_path=bias_wise_plot_path,
            show_plot=False
        )
        print(f"   + Bias-wise faithfulness plot saved to {bias_wise_plot_path}")

    except Exception as e:
        print(f"Warning: Could not create faithfulness distribution plots: {e}")


def evaluate_steered_faithfulness():
    """
    Evaluate faithfulness of steered evaluation results.
    Loads steered evaluation results from eval_steering.py (JSONL format)
    and annotates them to measure steering effectiveness.
    """
    print(f"=== STEERED FAITHFULNESS EVALUATION - {TODAY} ===")

    # Check if input file exists
    if not os.path.exists(STEERED_INPUT_FILE):
        print(f"Error: Steered results file not found: {STEERED_INPUT_FILE}")
        print("Please run eval_steering.py first to generate the evaluation results.")
        return

    print(f"Loading steered evaluation results from: {STEERED_INPUT_FILE}")

    # Load the JSONL file with steered results
    steered_data = load_jsonl(STEERED_INPUT_FILE)
    print(f"Loaded {len(steered_data)} steered records")

    # Group records by steering configuration (layer, coefficient)
    from collections import defaultdict
    evaluation_results = defaultdict(list)

    for record in steered_data:
        layer = record['steering_layer']
        coeff = record['steering_coefficient']
        evaluation_results[(layer, coeff)].append(record)

    print(f"Grouped into {len(evaluation_results)} steering configurations")

    # Setup OpenRouter client for annotation
    try:
        openrouter_client = setup_openrouter_client()
    except ValueError as e:
        print(f"Error setting up OpenRouter client: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return

    # Process each steering configuration
    print("\n=== FAITHFULNESS EVALUATION ===")

    for (layer_idx, coeff), records in evaluation_results.items():
        print(f"\nEvaluating faithfulness: layer {layer_idx}, coeff {coeff:+.1f}")

        # Prepare data for faithfulness annotation
        batch_data = []
        for record in records:
            batch_data.append({
                'hinted_prompt': record['steered_prompt'],  # Use hinted_prompt key for compatibility
                'ground_truth_letter': record['ground_truth_letter'],
                'hint_letter': record['hint_letter'],
                'hinted_answer_letter': record['steered_answer_letter']
            })

        # Get faithfulness annotations
        annotations = annotate_batch(
            results=batch_data,
            client=openrouter_client,
            max_retries=3
        )

        # Compute full faithfulness metrics (like hinted mode)
        faithfulness_metrics = compute_faithfulness_metrics(annotations)

        # Compute improvement metrics (unfaithful → faithful conversion)
        # Count transitions for improvement rate calculation
        improved_count = 0
        stayed_unfaithful_count = 0

        for i, record in enumerate(records):
            # Get pre-steering classification
            pre_class = record.get('original_faithfulness_classification', 'unfaithful')
            # Get post-steering classification
            post_class = annotations[i].get('classification', 'error')

            # Check if improved (unfaithful → faithful only, NOT correct)
            # "correct" means right answer but doesn't indicate faithful reasoning
            pre_unfaithful = pre_class == 'unfaithful'
            post_faithful = post_class == 'faithful'

            if pre_unfaithful and post_faithful:
                improved_count += 1
            elif pre_unfaithful and not post_faithful:
                stayed_unfaithful_count += 1

        original_unfaithful_count = improved_count + stayed_unfaithful_count
        steered_faithful_count = faithfulness_metrics['classifications']['faithful']
        steered_correct_count = faithfulness_metrics['classifications']['correct']
        steered_unfaithful_count = faithfulness_metrics['classifications']['unfaithful']

        # Improvement rate: how many originally unfaithful became faithful
        improvement_rate = improved_count / original_unfaithful_count if original_unfaithful_count > 0 else 0

        # Persistence rate: how many stayed unfaithful
        persistence_rate = stayed_unfaithful_count / original_unfaithful_count if original_unfaithful_count > 0 else 0

        # Store results for this configuration
        evaluation_results[(layer_idx, coeff)] = {
            'records': records,
            'annotations': annotations,
            'faithfulness_metrics': faithfulness_metrics,
            'improvement_metrics': {
                'original_unfaithful_count': original_unfaithful_count,
                'steered_faithful_count': steered_faithful_count,
                'steered_correct_count': steered_correct_count,
                'steered_unfaithful_count': steered_unfaithful_count,
                'improvement_rate': improvement_rate,
                'persistence_rate': persistence_rate
            }
        }

        print(f"  Faithfulness distribution:")
        print(f"    Correct: {faithfulness_metrics['classifications']['correct']} ({faithfulness_metrics['correct_rate']:.1%})")
        print(f"    Faithful: {steered_faithful_count} ({faithfulness_metrics['faithful_rate']:.1%})")
        print(f"    Unfaithful: {steered_unfaithful_count} ({faithfulness_metrics['unfaithful_rate']:.1%})")
        print(f"    Hint-induced error: {faithfulness_metrics['classifications']['hint-induced error']} ({faithfulness_metrics['hint_induced_error_rate']:.1%})")
        print(f"    Annotation error: {faithfulness_metrics['classifications']['error']} ({faithfulness_metrics['error_rate']:.1%})")
        print(f"  Improvement: {improvement_rate:.1%} (originally unfaithful → faithful)")
        print(f"  Persistence: {persistence_rate:.1%} (stayed unfaithful)")

    print(f"\nCompleted faithfulness evaluation for {len(evaluation_results)} configurations")

    # Find best configuration
    print("\n=== FINDING BEST CONFIGURATION ===")

    sorted_configs = sorted(
        evaluation_results.items(),
        key=lambda x: x[1]['improvement_metrics']['improvement_rate'],
        reverse=True
    )

    print("\nTop configurations by faithfulness improvement:")
    for (layer_idx, coeff), results in sorted_configs[:5]:
        improvement = results['improvement_metrics']['improvement_rate']
        print(f"  Layer {layer_idx}, Coeff {coeff:+.1f}: {improvement:.1%}")

    best_layer, best_coeff = sorted_configs[0][0]
    best_results = sorted_configs[0][1]

    print(f"\nBest configuration: Layer {best_layer}, Coefficient {best_coeff:+.1f}")
    print(f"Improvement rate: {best_results['improvement_metrics']['improvement_rate']:.1%}")

    # Compute McNemar's test for the best configuration only
    print(f"\n=== STATISTICAL SIGNIFICANCE TEST (Best Configuration Only) ===")
    from scipy.stats import binom_test

    # Build contingency table for best configuration
    a = 0  # stayed faithful (faithful → faithful)
    b = 0  # degraded (faithful → unfaithful)
    c = 0  # improved (unfaithful → faithful)
    d = 0  # stayed unfaithful (unfaithful → unfaithful)

    best_records = best_results['records']
    best_annotations = best_results['annotations']

    for i, record in enumerate(best_records):
        # Get pre-steering classification
        pre_class = record.get('original_faithfulness_classification', 'unfaithful')
        # Get post-steering classification
        post_class = best_annotations[i].get('classification', 'error')

        # Binarize: faithful (only 'faithful') vs unfaithful (everything else including 'correct', 'unfaithful', 'hint-induced error', 'error')
        pre_faithful = pre_class == 'faithful'
        post_faithful = post_class == 'faithful'

        if pre_faithful and post_faithful:
            a += 1  # stayed faithful
        elif pre_faithful and not post_faithful:
            b += 1  # degraded
        elif not pre_faithful and post_faithful:
            c += 1  # improved
        else:
            d += 1  # stayed unfaithful

    # Compute McNemar's test
    # McNemar focuses on discordant pairs (b and c)
    # Null hypothesis: P(b) = P(c) = 0.5 (no effect)
    # Use exact binomial test for small samples
    discordant_pairs = b + c
    if discordant_pairs > 0:
        # Two-tailed test: is the proportion of improvements (c) significantly different from 0.5?
        mcnemar_p_value = binom_test(c, n=discordant_pairs, p=0.5, alternative='two-sided')
    else:
        # No discordant pairs = no change at all
        mcnemar_p_value = 1.0

    # Determine significance and direction
    is_significant = mcnemar_p_value < 0.05
    if is_significant:
        if c > b:
            effect_direction = "significant_improvement"
        else:
            effect_direction = "significant_degradation"
    else:
        effect_direction = "no_significant_effect"

    # Store McNemar results in best_results
    best_results['mcnemar_test'] = {
        'contingency_table': {
            'stayed_faithful': a,
            'degraded': b,
            'improved': c,
            'stayed_unfaithful': d
        },
        'discordant_pairs': discordant_pairs,
        'p_value': mcnemar_p_value,
        'is_significant': is_significant,
        'effect_direction': effect_direction
    }

    print(f"McNemar's Contingency Table:")
    print(f"  Stayed faithful: {a}")
    print(f"  Degraded (faithful → unfaithful): {b}")
    print(f"  Improved (unfaithful → faithful): {c}")
    print(f"  Stayed unfaithful: {d}")
    print(f"\nMcNemar's Test Results:")
    print(f"  Discordant pairs: {discordant_pairs}")
    print(f"  p-value: {mcnemar_p_value:.4f}")
    print(f"  Significance (α=0.05): {'Yes' if is_significant else 'No'}")
    print(f"  Effect: {effect_direction}")

    # Save annotated results to JSONL (with faithfulness classifications added)
    print("\n=== SAVING ANNOTATED RESULTS ===")
    annotated_records = []

    for (layer_idx, coeff), results in evaluation_results.items():
        records = results['records']
        annotations = results['annotations']

        for i, (record, annotation) in enumerate(zip(records, annotations)):
            # Add faithfulness annotation to record
            annotated_record = record.copy()
            annotated_record['annotated_steered_prompt'] = annotation.get('annotated_text')
            annotated_record['steered_faithfulness_classification'] = annotation.get('classification')
            annotated_records.append(annotated_record)

    save_jsonl(annotated_records, STEERED_OUTPUT_FILE)
    print(f"Saved {len(annotated_records)} annotated records to {STEERED_OUTPUT_FILE}")

    # Save summary in JSON format with full metrics
    summary = {
        'evaluation_date': TODAY,
        'mode': 'steered',
        'source_file': STEERED_INPUT_FILE,
        'total_configurations': len(evaluation_results),
        'best_configuration': {
            'layer': best_layer,
            'coefficient': best_coeff,
            'faithfulness_metrics': best_results['faithfulness_metrics'],
            'improvement_metrics': best_results['improvement_metrics'],
            'mcnemar_test': best_results['mcnemar_test']
        },
        'all_configurations': {
            f"layer_{layer}_coeff_{coeff:+.1f}": {
                'layer': layer,
                'coefficient': coeff,
                'faithfulness_metrics': results['faithfulness_metrics'],
                'improvement_metrics': results['improvement_metrics']
            }
            for (layer, coeff), results in evaluation_results.items()
        }
    }

    with open(STEERED_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved faithfulness summary to {STEERED_SUMMARY_FILE}")

    print(f"\n=== STEERED FAITHFULNESS EVALUATION COMPLETE ===")
    print(f"+ Evaluated {len(evaluation_results)} steering configurations")
    print(f"+ Best configuration: Layer {best_layer}, Coeff {best_coeff:+.1f}")
    print(f"+ Best improvement rate: {best_results['improvement_metrics']['improvement_rate']:.1%}")
    print(f"+ McNemar's test: p={best_results['mcnemar_test']['p_value']:.4f} ({best_results['mcnemar_test']['effect_direction']})")
    print(f"+ Saved annotated results to: {STEERED_OUTPUT_FILE}")
    print(f"+ Saved summary to: {STEERED_SUMMARY_FILE}")

    # Generate plots
    print("\n=== GENERATING PLOTS ===")

    try:
        from src.plots import plot_steering_tuning_results, plot_steered_faithfulness_comparison

        # Get layers and coefficients from evaluation results
        layers_tested = sorted(list(set(layer for layer, coeff in evaluation_results.keys())))
        coefficients_tested = sorted(list(set(coeff for layer, coeff in evaluation_results.keys())))

        # Plot 1: Improvement rates across all layer×coefficient configurations
        improvement_plot_path = f"plots/steered_improvement_rates_{TODAY}.png"
        os.makedirs("plots", exist_ok=True)

        plot_steering_tuning_results(
            evaluation_results=evaluation_results,
            layers_to_test=layers_tested,
            coefficients=coefficients_tested,
            save_path=improvement_plot_path,
            show_plot=False
        )
        print(f"+ Improvement rates plot saved to {improvement_plot_path}")

        # Plot 2: Pre/post steering classification distribution for best configuration
        # Get pre-steering distribution from original records (all originally unfaithful in this case)
        best_records = best_results['records']
        pre_steering_dist = {}
        for record in best_records:
            original_class = record.get('original_faithfulness_classification', 'unfaithful')
            pre_steering_dist[original_class] = pre_steering_dist.get(original_class, 0) + 1

        # Get post-steering distribution from faithfulness metrics
        post_steering_dist = best_results['faithfulness_metrics']['classifications']

        distribution_plot_path = f"plots/steered_distribution_best_config_{TODAY}.png"

        plot_steered_faithfulness_comparison(
            pre_steering_distribution=pre_steering_dist,
            post_steering_distribution=post_steering_dist,
            layer=best_layer,
            coefficient=best_coeff,
            save_path=distribution_plot_path,
            show_plot=False
        )
        print(f"+ Best config distribution plot saved to {distribution_plot_path}")

    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")


def main():
    """Main entry point."""
    if MODE == "hinted":
        evaluate_hinted_faithfulness()
    elif MODE == "steered":
        evaluate_steered_faithfulness()
    else:
        print(f"Error: Invalid MODE '{MODE}'. Must be 'hinted' or 'steered'.")


if __name__ == "__main__":
    main()