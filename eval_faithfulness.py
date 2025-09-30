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

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.faithfulness_eval import (
    setup_openrouter_client, annotate_batch,
    compute_faithfulness_metrics, print_faithfulness_report
)
from src.config import TODAY, ANNOTATED_DIR

# Commented out IPython magic to ensure Python compatibility.
# Setting up to work with the project GitHub Repository (importing scripts, pushing results)

# Clone the repo to import in Colab its packages from GitHub
!git clone https://github.com/Joe-Occhipinti/unfaithfulness_steering.git
import os
os.chdir('/content/unfaithfulness_steering')

# Authenticate in GitHub
!git config --global user.email "occhidipinti00@gmail.com"
!git config --global user.name "Joe-Occhipinti"

# Put your GitHub token in Colab secrets
from google.colab import userdata
GITHUB_TOKEN = userdata.get('Colab')

# Build authenticated repo url
repo_url = f"https://{GITHUB_TOKEN}@github.com/Joe-Occhipinti/unfaithfulness_steering.git"

# Install required packages
!pip install -U bitsandbytes accelerate transformers google-genai requests python-dotenv

# Set up OpenRouter API environment variables from Colab secrets
import os
os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
# Optional: Set site info for OpenRouter tracking
os.environ['SITE_URL'] = userdata.get('SITE_URL', 'https://github.com')
os.environ['SITE_NAME'] = userdata.get('SITE_NAME', 'Faithfulness Steering')

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Mode selection: "hinted" or "steered"
MODE = "hinted"

# For HINTED mode - input and output files
HINTED_INPUT_FILE = "data/behavioural/hinted_psychology_business_ethics_2025-09-29.jsonl"
HINTED_OUTPUT_FILE = "data/annotated/annotated_biased_psychology_business_ethics_2025-09-29.jsonl"
HINTED_SUMMARY_FILE = "data/summaries/faithfulness_hinted_psychology_business_ethics_2025-09-29.json"

# For STEERED mode - input and output files
STEERED_INPUT_FILE = "data/behavioural/steered_val_psychology_business_ethics_2025-09-30.jsonl"
STEERED_OUTPUT_FILE = "data/annotated/annotated_steered_psychology_business_ethics_2025-09-30.jsonl"
STEERED_SUMMARY_FILE = "data/summaries/faithfulness_steered_psychology_business_ethics_2025-09-30.json"

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

    # Optional: Create faithfulness distribution plot
    try:
        from src.plots import plot_faithfulness_distribution

        faithfulness_plot_path = f"plots/faithfulness_distribution_{TODAY}.png"
        os.makedirs("plots", exist_ok=True)

        plot_faithfulness_distribution(
            hinted_results=annotated_results,
            save_path=faithfulness_plot_path,
            show_plot=False
        )

        print(f"   + Faithfulness distribution plot saved to {faithfulness_plot_path}")

    except Exception as e:
        print(f"Warning: Could not create faithfulness distribution plot: {e}")


def evaluate_steered_faithfulness():
    """
    Evaluate faithfulness of steered evaluation results.
    Loads steered evaluation results from tune_steering_vectors.py
    and annotates them to measure steering effectiveness.
    """
    print(f"=== STEERED FAITHFULNESS EVALUATION - {TODAY} ===")

    # Check if input file exists
    if not os.path.exists(STEERED_INPUT_FILE):
        print(f"Error: Steered results file not found: {STEERED_INPUT_FILE}")
        print("Please run tune_steering_vectors.py first to generate the evaluation results.")
        print("Make sure to save evaluation_results to a pickle file after CELL 4.")
        return

    print(f"Loading steered evaluation results from: {STEERED_INPUT_FILE}")

    # Load the evaluation results (should be a pickle file with the evaluation_results dict)
    with open(STEERED_INPUT_FILE, 'rb') as f:
        data = pickle.load(f)

    # Handle different possible formats
    if isinstance(data, dict):
        if 'evaluation_results' in data:
            evaluation_results = data['evaluation_results']
            val_unfaithful = data.get('val_unfaithful', [])
        else:
            # Assume the dict is the evaluation_results itself
            evaluation_results = data
            val_unfaithful = []
            print("Warning: val_unfaithful data not found in pickle file")
    else:
        print(f"Error: Unexpected data format in {STEERED_INPUT_FILE}")
        return

    print(f"Loaded results for {len(evaluation_results)} steering configurations")

    # Setup OpenRouter client for annotation
    try:
        openrouter_client = setup_openrouter_client()
    except ValueError as e:
        print(f"Error setting up OpenRouter client: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return

    # Process each steering configuration
    print("\n=== FAITHFULNESS EVALUATION ===")

    for (layer_idx, coeff), results in evaluation_results.items():
        print(f"\nEvaluating faithfulness: layer {layer_idx}, coeff {coeff:+.1f}")

        # Check if already annotated
        if 'faithfulness_labels' in results and results['faithfulness_labels']:
            print(f"  Already annotated, skipping...")
            continue

        # Get steered prompts and answers
        steered_prompts = results.get('steered_prompts', results.get('steered_biased_prompts', []))
        steered_answers = results.get('steered_answers', [])

        if not steered_prompts:
            print(f"  Warning: No steered prompts found for this configuration")
            continue

        # Prepare data for faithfulness annotation
        batch_data = []

        # If we have val_unfaithful data, use it for ground truth
        if val_unfaithful and len(val_unfaithful) == len(steered_prompts):
            for i, (steered_prompt, orig_item) in enumerate(zip(steered_prompts, val_unfaithful)):
                batch_data.append({
                    'hinted_prompt': steered_prompt,  # Use hinted_prompt key for compatibility
                    'ground_truth_letter': orig_item.get('ground_truth_letter'),
                    'hint_letter': orig_item.get('hint_letter'),
                    'hinted_answer_letter': steered_answers[i] if i < len(steered_answers) else None
                })
        else:
            # Fallback: just annotate the steered prompts without ground truth
            for i, steered_prompt in enumerate(steered_prompts):
                batch_data.append({
                    'hinted_prompt': steered_prompt,
                    'ground_truth_letter': None,
                    'hint_letter': None,
                    'hinted_answer_letter': steered_answers[i] if i < len(steered_answers) else None
                })

        # Get faithfulness annotations
        annotations = annotate_batch(
            results=batch_data,
            client=openrouter_client,
            max_retries=3
        )

        # Extract faithfulness labels
        faithfulness_labels = [ann.get('classification', 'error') for ann in annotations]

        # Compute improvement metrics
        original_faithful_count = 0  # All val_unfaithful were unfaithful
        steered_faithful_count = sum(1 for label in faithfulness_labels if label == 'faithful')
        improvement_rate = steered_faithful_count / len(faithfulness_labels) if faithfulness_labels else 0

        # Update results with annotations
        results['annotations'] = annotations
        results['faithfulness_labels'] = faithfulness_labels
        results['improvement_rate'] = improvement_rate
        results['steered_faithful_count'] = steered_faithful_count
        results['total_prompts'] = len(faithfulness_labels)

        print(f"  Improvement: {improvement_rate:.1%} ({steered_faithful_count}/{len(faithfulness_labels)} became faithful)")

    print(f"\nCompleted faithfulness evaluation for {len(evaluation_results)} configurations")

    # Find best configuration
    print("\n=== FINDING BEST CONFIGURATION ===")

    sorted_configs = sorted(
        evaluation_results.items(),
        key=lambda x: x[1].get('improvement_rate', 0),
        reverse=True
    )

    print("\nTop configurations by faithfulness improvement:")
    for (layer_idx, coeff), results in sorted_configs[:5]:
        print(f"  Layer {layer_idx}, Coeff {coeff:+.1f}: {results.get('improvement_rate', 0):.1%}")

    best_layer, best_coeff = sorted_configs[0][0]
    best_results = sorted_configs[0][1]

    print(f"\nBest configuration: Layer {best_layer}, Coefficient {best_coeff:+.1f}")
    print(f"Improvement rate: {best_results.get('improvement_rate', 0):.1%}")

    # Save updated evaluation results with annotations
    # Save to proper annotated directory
    annotated_output_file = STEERED_OUTPUT_FILE

    save_data = {
        'evaluation_results': evaluation_results,
        'val_unfaithful': val_unfaithful,
        'best_config': {
            'layer': best_layer,
            'coefficient': best_coeff,
            'improvement_rate': best_results.get('improvement_rate', 0)
        },
        'evaluation_date': TODAY
    }

    with open(annotated_output_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved annotated evaluation results to {annotated_output_file}")

    # Save summary in JSON format
    summary = {
        'evaluation_date': TODAY,
        'mode': 'steered',
        'source_file': STEERED_INPUT_FILE,
        'total_configurations': len(evaluation_results),
        'best_configuration': {
            'layer': best_layer,
            'coefficient': best_coeff,
            'improvement_rate': best_results.get('improvement_rate', 0),
            'steered_faithful_count': best_results.get('steered_faithful_count', 0),
            'total_prompts': best_results.get('total_prompts', 0)
        },
        'top_5_configurations': [
            {
                'layer': layer_idx,
                'coefficient': coeff,
                'improvement_rate': results.get('improvement_rate', 0)
            }
            for (layer_idx, coeff), results in sorted_configs[:5]
        ],
        'annotated_output_file': annotated_output_file
    }

    with open(STEERED_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved faithfulness summary to {STEERED_SUMMARY_FILE}")

    print(f"\n=== STEERED FAITHFULNESS EVALUATION COMPLETE ===")
    print(f"+ Evaluated {len(evaluation_results)} steering configurations")
    print(f"+ Best configuration: Layer {best_layer}, Coeff {best_coeff:+.1f}")
    print(f"+ Best improvement rate: {best_results.get('improvement_rate', 0):.1%}")
    print(f"+ Saved annotated results to: {annotated_output_file}")


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

# Add the generated files
!git add data/behavioural/hinted_{TODAY}.jsonl
!git add data/summaries/hinted_summary_{TODAY}.json
!git add plots/accuracy_comparison_{TODAY}.png
!git status

# Commit and push
!git commit -m "Add hinted evaluation results - {TODAY}"
!git push {repo_url} main