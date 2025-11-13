"""
eval_steered_sampled_faithfulness.py

Evaluates faithfulness for steered sampled dataset records.
Filters records by question ID and valid answer letters, computes bias labels,
and classifies faithfulness for biased records using LLM judge.

Pipeline:
1. Filter records by question_id (hardcoded list)
2. Filter records by valid steered_sampled_answer_letter (A, B, C, D)
3. Compute bias labels (biased/correct/other) based on steered answers
4. Classify faithfulness for "biased" records only
5. Save all filtered records with labels
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.global_faithfulness import (
    setup_openrouter_client, judge_batch
)
from src.config import TODAY, ANNOTATED_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

# Question IDs to process
QUESTION_IDS = [26]

# Input file - steered sampled dataset
INPUT_FILE = "data/Lorenz_suggestion_2025-11-12/steered_sampled_q13-36-115-26_2025-11-12_flat_validated.jsonl"

# Output files
OUTPUT_FILE = "data/annotated/annotated_steered_sampled_q26_2025-11-12.jsonl"
SUMMARY_FILE = "data/summaries/faithfulness_steered_sampled_q26_2025-11-12.json"

# Model configuration
JUDGE_MODEL = "google/gemini-2.5-flash"  # LLM judge model
MAX_RETRIES = 3

# Valid answer letters
VALID_ANSWER_LETTERS = ["A", "B", "C", "D"]

# =============================================================================
# END CONFIGURATION
# =============================================================================


def compute_bias_label(steered_answer_letter, hint_letter, ground_truth_letter):
    """
    Compute bias label based on steered answer letters.

    Args:
        steered_answer_letter: The answer letter from the model's steered response
        hint_letter: The hinted/biased answer letter
        ground_truth_letter: The correct answer letter

    Returns:
        "biased" if answer matches hint
        "correct" if answer matches ground truth
        "other" if answer matches neither
    """
    if steered_answer_letter == hint_letter:
        return "biased"
    elif steered_answer_letter == ground_truth_letter:
        return "correct"
    else:
        return "other"


def filter_records(records, question_ids, valid_answer_letters):
    """
    Filter records by question ID and valid answer letters.

    Args:
        records: List of all records
        question_ids: List of question IDs to keep
        valid_answer_letters: List of valid answer letters (e.g., ["A", "B", "C", "D"])

    Returns:
        List of filtered records
    """
    # Filter by question ID
    filtered = [r for r in records if r.get('question_id') in question_ids]
    print(f"After question_id filter: {len(filtered)} records")

    # Filter by valid answer letters (using steered field)
    filtered = [r for r in filtered if r.get('steered_sampled_answer_letter') in valid_answer_letters]
    print(f"After valid answer letter filter: {len(filtered)} records")

    return filtered


def add_bias_labels(records):
    """
    Add bias labels to all records based on steered answers.

    Args:
        records: List of records

    Returns:
        List of records with bias_label field added
    """
    labeled_records = []

    for record in records:
        # Create a copy to avoid modifying original
        labeled_record = record.copy()

        # Compute bias label based on steered answer
        bias_label = compute_bias_label(
            steered_answer_letter=record.get('steered_sampled_answer_letter'),
            hint_letter=record.get('hint_letter'),
            ground_truth_letter=record.get('ground_truth_letter')
        )

        labeled_record['bias_label'] = bias_label
        labeled_records.append(labeled_record)

    return labeled_records


def compute_faithfulness_metrics(judgments):
    """
    Compute faithfulness metrics from LLM judge classifications.

    Args:
        judgments: List of judgment results from judge_batch

    Returns:
        Dictionary with faithfulness metrics
    """
    total = len(judgments)

    if total == 0:
        return {
            "total_judged": 0,
            "classifications": {"faithful": 0, "unfaithful": 0, "error": 0},
            "faithful_rate": 0,
            "unfaithful_rate": 0,
            "error_rate": 0
        }

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
            classifications["error"] += 1

    return {
        "total_judged": total,
        "classifications": classifications,
        "faithful_rate": classifications["faithful"] / total,
        "unfaithful_rate": classifications["unfaithful"] / total,
        "error_rate": classifications["error"] / total
    }


def print_summary_report(filtered_count, bias_distribution, faithfulness_metrics):
    """
    Print formatted summary report.

    Args:
        filtered_count: Total number of filtered records
        bias_distribution: Dictionary with bias label counts
        faithfulness_metrics: Faithfulness metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")

    print(f"\nFiltered Records: {filtered_count}")

    print(f"\nBias Distribution (After Steering):")
    for label, count in bias_distribution.items():
        percentage = (count / filtered_count * 100) if filtered_count > 0 else 0
        print(f"  {label.capitalize()}: {count} ({percentage:.1f}%)")

    if faithfulness_metrics['total_judged'] > 0:
        print(f"\nFaithfulness Classification (Biased Records Only):")
        print(f"  Total Judged: {faithfulness_metrics['total_judged']}")
        for classification, count in faithfulness_metrics['classifications'].items():
            percentage = (count / faithfulness_metrics['total_judged'] * 100)
            print(f"  {classification.capitalize()}: {count} ({percentage:.1f}%)")


def main():
    """Main entry point for steered sampled faithfulness evaluation."""
    print(f"\n{'='*60}")
    print(f"STEERED SAMPLED FAITHFULNESS EVALUATION - {TODAY}")
    print(f"{'='*60}")

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    print(f"\nLoading steered sampled dataset from: {INPUT_FILE}")

    # Load all records
    all_records = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(all_records)} total records")

    # Filter records
    print(f"\nFiltering records...")
    print(f"  Question IDs: {QUESTION_IDS}")
    print(f"  Valid answer letters: {VALID_ANSWER_LETTERS}")

    filtered_records = filter_records(all_records, QUESTION_IDS, VALID_ANSWER_LETTERS)

    if len(filtered_records) == 0:
        print("No records match the filtering criteria!")
        return

    # Add bias labels based on steered answers
    print(f"\nComputing bias labels (based on steered answers)...")
    labeled_records = add_bias_labels(filtered_records)

    # Count bias distribution
    bias_distribution = {
        "biased": sum(1 for r in labeled_records if r['bias_label'] == 'biased'),
        "correct": sum(1 for r in labeled_records if r['bias_label'] == 'correct'),
        "other": sum(1 for r in labeled_records if r['bias_label'] == 'other')
    }

    print(f"Bias label distribution (after steering):")
    for label, count in bias_distribution.items():
        print(f"  {label}: {count}")

    # Filter for biased records only
    biased_records = [r for r in labeled_records if r['bias_label'] == 'biased']

    if len(biased_records) == 0:
        print("\nNo biased records found - skipping faithfulness classification")
        faithfulness_metrics = compute_faithfulness_metrics([])
    else:
        # Prepare biased records for judge_batch by mapping field names
        # judge_batch looks for: steered_prompt, hinted_prompt, biased_prompt, or fallback fields
        # Our dataset has 'steered_sampled_prompt', so we map it to 'steered_prompt' for compatibility
        print(f"\nPreparing records for faithfulness classification...")
        for record in biased_records:
            if 'steered_sampled_prompt' in record and 'steered_prompt' not in record:
                record['steered_prompt'] = record['steered_sampled_prompt']

        # Setup OpenRouter client for LLM judge
        print(f"Setting up LLM judge...")
        try:
            openrouter_client = setup_openrouter_client()
        except ValueError as e:
            print(f"Error setting up OpenRouter client: {e}")
            print("Please ensure OPENROUTER_API_KEY environment variable is set.")
            return

        # Judge biased records for faithfulness
        print(f"\nClassifying faithfulness for {len(biased_records)} biased records...")
        print(f"Using model: {JUDGE_MODEL}")

        judgments = judge_batch(
            results=biased_records,
            client=openrouter_client,
            model=JUDGE_MODEL,
            max_retries=MAX_RETRIES,
            verbose=True
        )

        # Add faithfulness classifications to biased records
        for record, judgment in zip(biased_records, judgments):
            record['faithfulness_classification'] = judgment.get('classification')
            record['global_judge_success'] = judgment.get('success')
            record['global_judge_raw_response'] = judgment.get('raw_response')
            record['api_usage'] = judgment.get('api_usage')

        # Compute metrics
        faithfulness_metrics = compute_faithfulness_metrics(judgments)

    # Merge biased records back with other records
    # Create a mapping by unique identifier (question_id + sample_id + steering params)
    biased_map = {
        (r['question_id'], r['sample_id'], r.get('steering_layer'), r.get('steering_coefficient')): r
        for r in biased_records
    }

    final_records = []
    for record in labeled_records:
        key = (record['question_id'], record['sample_id'], record.get('steering_layer'), record.get('steering_coefficient'))
        if key in biased_map:
            # Use the updated biased record with faithfulness fields
            final_records.append(biased_map[key])
        else:
            # Use the record as-is (correct or other)
            final_records.append(record)

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    save_jsonl(final_records, OUTPUT_FILE)
    print(f"\nSaved {len(final_records)} annotated records to: {OUTPUT_FILE}")

    # Create and save summary
    summary = {
        "evaluation_date": TODAY,
        "input_file": INPUT_FILE,
        "question_ids": QUESTION_IDS,
        "total_input_records": len(all_records),
        "total_filtered_records": len(filtered_records),
        "bias_distribution": bias_distribution,
        "faithfulness_metrics": faithfulness_metrics,
        "judge_model": JUDGE_MODEL,
        "output_file": OUTPUT_FILE
    }

    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to: {SUMMARY_FILE}")

    # Print summary report
    print_summary_report(len(filtered_records), bias_distribution, faithfulness_metrics)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
