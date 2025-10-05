"""
eval_global_faithfulness.py

Standalone global faithfulness evaluation script using LLM-judge approach.
Uses the global annotation prompt (faithfulness_global_annotation_professor.txt)
to directly classify faithfulness without tag-based annotation.

Processes hinted evaluation results and annotates biased prompts for faithfulness.
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.global_faithfulness import (
    setup_openrouter_client, judge_batch
)
from src.config import TODAY, ANNOTATED_DIR

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Input file - hinted evaluation results
HINTED_INPUT_FILE = "data/behavioural/hinted_high_school_macroeconomics_microeconomics_2025-10-03.jsonl"

# Output files - global faithfulness annotations
ANNOTATED_OUTPUT_FILE = "data/annotated/annotated_global_biased_high_school_macroeconomics_microeconomics_2025-10-03.jsonl"
SUMMARY_OUTPUT_FILE = "data/summaries/faithfulness_global_biased_high_school_macroeconomics_microeconomics_2025-10-03.json"

# Model configuration
JUDGE_MODEL = "google/gemini-2.5-flash"  # LLM judge model
MAX_RETRIES = 3

# =============================================================================
# END CONFIGURATION
# =============================================================================


def compute_global_faithfulness_metrics(judgments):
    """
    Compute faithfulness metrics from global LLM judge classifications.

    Args:
        judgments: List of judgment results from judge_batch

    Returns:
        Dictionary with faithfulness metrics
    """
    total = len(judgments)

    # Count classifications
    classifications = {
        "faithful": 0,
        "unfaithful": 0,
        "error": 0
    }

    for judgment in judgments:
        classification = judgment.get('classification', 'error')
        if classification in classifications:
            classifications[classification] += 1
        else:
            # Treat unknown classifications as errors
            classifications["error"] += 1

    return {
        "total_judged": total,
        "classifications": classifications,
        "faithful_rate": classifications["faithful"] / total if total > 0 else 0,
        "unfaithful_rate": classifications["unfaithful"] / total if total > 0 else 0,
        "error_rate": classifications["error"] / total if total > 0 else 0
    }


def print_global_faithfulness_report(metrics):
    """
    Print formatted global faithfulness evaluation report.

    Args:
        metrics: Faithfulness metrics dictionary
    """
    print(f"\n=== GLOBAL FAITHFULNESS EVALUATION RESULTS ===")
    print(f"Total Judged: {metrics['total_judged']}")
    print(f"\nClassification Distribution:")
    for classification, count in metrics['classifications'].items():
        percentage = (count / metrics['total_judged'] * 100) if metrics['total_judged'] > 0 else 0
        class_name = classification.capitalize()
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    print(f"\nFaithfulness Rates:")
    print(f"  Faithful: {metrics['faithful_rate']:.3f}")
    print(f"  Unfaithful: {metrics['unfaithful_rate']:.3f}")
    print(f"  Error: {metrics['error_rate']:.3f}")


def main():
    """Main entry point for global faithfulness evaluation."""
    print(f"=== GLOBAL FAITHFULNESS EVALUATION - {TODAY} ===")

    # Check if hinted results file exists
    if not os.path.exists(HINTED_INPUT_FILE):
        print(f"Error: Hinted results file not found: {HINTED_INPUT_FILE}")
        print("Please run hinted_eval.py first to generate the results.")
        return

    print(f"Loading hinted evaluation results from: {HINTED_INPUT_FILE}")

    # Load the saved hinted results from file
    hinted_results = load_jsonl(HINTED_INPUT_FILE)
    print(f"Loaded {len(hinted_results)} hinted evaluation results")

    # Setup OpenRouter client for LLM judge
    try:
        openrouter_client = setup_openrouter_client()
    except ValueError as e:
        print(f"Error setting up OpenRouter client: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return

    # Filter for biased results only (where model followed the hint)
    biased_results = [r for r in hinted_results if r['bias_label'] == 'biased']
    print(f"Found {len(biased_results)} biased results to judge for faithfulness")

    if len(biased_results) == 0:
        print("No biased results to judge - all models resisted the hints!")
        print("Faithfulness evaluation complete (no work needed).")
        return

    # Judge biased prompts for faithfulness using global LLM judge
    print(f"\nJudging biased prompts for faithfulness using {JUDGE_MODEL}...")

    judgments = judge_batch(
        results=biased_results,
        client=openrouter_client,
        model=JUDGE_MODEL,
        max_retries=MAX_RETRIES,
        verbose=True
    )

    # Create annotated results with the judgment data
    annotated_results = []
    for result, judgment in zip(biased_results, judgments):
        # Create a copy of the result with judgment fields
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

        # Add global faithfulness judgment fields
        annotated_result['global_faithfulness_classification'] = judgment.get('classification')
        annotated_result['global_judge_success'] = judgment.get('success')
        annotated_result['global_judge_raw_response'] = judgment.get('raw_response')

        # For compatibility with existing plotting functions, also use standard field name
        annotated_result['faithfulness_classification'] = judgment.get('classification')

        # Create tagged biased prompt for activation extraction
        # Wrap entire prompt with [F] or [U] tags based on classification
        classification = judgment.get('classification')
        biased_prompt = annotated_result.get('biased_prompt', '')

        if classification == 'faithful':
            tagged_prompt = f"[F_final]{biased_prompt}[/F_final]"
        elif classification == 'unfaithful':
            tagged_prompt = f"[U_final]{biased_prompt}[/U_final]"
        else:  # error - no tags
            tagged_prompt = biased_prompt

        annotated_result['annotated_biased_prompt'] = tagged_prompt

        annotated_results.append(annotated_result)

    # Fix malformed JSON responses that were classified as errors
    print(f"\n=== Checking for malformed JSON in error classifications ===")
    errors_fixed = 0

    for i, (annotated_result, judgment) in enumerate(zip(annotated_results, judgments)):
        if judgment.get('classification') == 'error':
            raw_response = judgment.get('raw_response')

            if raw_response:
                # Try to extract classification from malformed JSON like:
                # ```json\n{\n "classification": "faithful"\n}\n```
                import re

                # Pattern to match classification in various malformed formats
                patterns = [
                    r'"classification"\s*:\s*"(faithful|unfaithful)"',  # Standard JSON
                    r'classification[\'"]?\s*[:=]\s*[\'"]?(faithful|unfaithful)[\'"]?',  # Loose matching
                ]

                extracted_classification = None
                for pattern in patterns:
                    match = re.search(pattern, raw_response, re.IGNORECASE)
                    if match:
                        extracted_classification = match.group(1).lower()
                        break

                if extracted_classification in ['faithful', 'unfaithful']:
                    print(f"  Fixed record {i}: extracted '{extracted_classification}' from malformed response")

                    # Update classification in judgment
                    judgment['classification'] = extracted_classification
                    judgment['success'] = True

                    # Update annotated result
                    annotated_result['global_faithfulness_classification'] = extracted_classification
                    annotated_result['faithfulness_classification'] = extracted_classification
                    annotated_result['global_judge_success'] = True

                    # Re-create tagged prompt with correct tags
                    biased_prompt = annotated_result.get('biased_prompt', '')
                    if extracted_classification == 'faithful':
                        tagged_prompt = f"[F_final]{biased_prompt}.[/F_final]"
                    else:  # unfaithful
                        tagged_prompt = f"[U_final]{biased_prompt}.[/U_final]"

                    annotated_result['annotated_biased_prompt'] = tagged_prompt

                    errors_fixed += 1

    if errors_fixed > 0:
        print(f"Fixed {errors_fixed} malformed JSON responses")
    else:
        print(f"No malformed JSON found in error classifications")

    # Compute faithfulness metrics (after fixing errors)
    faithfulness_metrics = compute_global_faithfulness_metrics(judgments)

    # Print faithfulness report
    print_global_faithfulness_report(faithfulness_metrics)

    # Save annotated results to separate file
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    save_jsonl(annotated_results, ANNOTATED_OUTPUT_FILE)
    print(f"\nSaved {len(annotated_results)} annotated biased results to {ANNOTATED_OUTPUT_FILE}")

    # Create and save faithfulness summary
    faithfulness_summary = {
        'evaluation_date': TODAY,
        'method': 'global_llm_judge',
        'judge_model': JUDGE_MODEL,
        'source_file': HINTED_INPUT_FILE,
        'total_biased_results': len(biased_results),
        'total_judged': len(annotated_results),
        'faithfulness_metrics': faithfulness_metrics,
        'annotated_output_file': ANNOTATED_OUTPUT_FILE
    }

    with open(SUMMARY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(faithfulness_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved faithfulness summary to {SUMMARY_OUTPUT_FILE}")

    print(f"\n=== GLOBAL FAITHFULNESS EVALUATION COMPLETE ===")
    print(f"+ Global faithfulness evaluation completed:")
    print(f"   + Loaded hinted results from: {HINTED_INPUT_FILE}")
    print(f"   + Filtered {len(biased_results)} biased results")
    print(f"   + Judged biased prompts with {JUDGE_MODEL}")
    print(f"   + Classified faithfulness (faithful/unfaithful/error)")
    print(f"   + Saved annotated results to: {ANNOTATED_OUTPUT_FILE}")
    print(f"   + Saved faithfulness summary")

    # Optional: Create faithfulness distribution plots
    try:
        from src.plots import plot_faithfulness_distribution, plot_global_faithfulness_by_bias

        os.makedirs("plots", exist_ok=True)

        # Extract subject name from input file for plot naming
        # Example: "hinted_high_school_psychology_2025-08-15.jsonl" -> "high_school_psychology"
        input_basename = os.path.basename(HINTED_INPUT_FILE)
        subject_date = input_basename.replace("hinted_", "").replace(".jsonl", "")

        # Bias-wise faithfulness distribution (using global version with 3 categories)
        bias_wise_plot_path = f"plots/global_faithfulness_by_hint_{subject_date}.png"
        plot_global_faithfulness_by_bias(
            hinted_results=annotated_results,
            save_path=bias_wise_plot_path,
            show_plot=False
        )
        print(f"   + Bias-wise faithfulness plot saved to {bias_wise_plot_path}")

    except Exception as e:
        print(f"Warning: Could not create faithfulness distribution plots: {e}")


if __name__ == "__main__":
    main()
