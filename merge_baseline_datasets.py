"""
merge_baseline_datasets.py

Merges two baseline evaluation datasets and recalculates summary metrics.
Specifically designed to merge baseline_stem and baseline_stem_missing datasets.
"""

import json
from datetime import datetime
from typing import Dict, Any, List

# Import reusable functions from src
from src.data import load_jsonl, save_jsonl
from src.performance_eval import (
    compute_accuracy_metrics,
    compute_completeness_metrics,
    print_accuracy_report
)

# Configuration
DATASET_1_PATH = "data/behavioural/baseline_stem_2025-10-14.jsonl"
DATASET_2_PATH = "data/behavioural/baseline_stem_missing_2025-10-14.jsonl"
SUMMARY_1_PATH = "data/summaries/baseline_summary_stem_2025-10-14.json"
SUMMARY_2_PATH = "data/summaries/baseline_summary_stem_missing_2025-10-14.json"

# Output paths
TODAY = datetime.now().strftime("%Y-%m-%d")
OUTPUT_JSONL = f"data/behavioural/baseline_stem_merged_{TODAY}.jsonl"
OUTPUT_SUMMARY = f"data/summaries/baseline_summary_stem_merged_{TODAY}.json"

def merge_baseline_datasets():
    """Merge two baseline datasets and recalculate summary metrics."""

    print("=" * 80)
    print("MERGING BASELINE DATASETS")
    print("=" * 80)

    # Load datasets
    print(f"\n--- Loading datasets ---")
    dataset_1 = load_jsonl(DATASET_1_PATH)
    dataset_2 = load_jsonl(DATASET_2_PATH)
    print(f"Dataset 1: {len(dataset_1)} records")
    print(f"Dataset 2: {len(dataset_2)} records")

    # Load summaries for metadata
    print(f"\n--- Loading summaries ---")
    with open(SUMMARY_1_PATH, 'r', encoding='utf-8') as f:
        summary_1 = json.load(f)
    with open(SUMMARY_2_PATH, 'r', encoding='utf-8') as f:
        summary_2 = json.load(f)

    # Merge datasets
    print(f"\n--- Merging datasets ---")
    merged_data = dataset_1 + dataset_2
    print(f"Total merged records: {len(merged_data)}")

    # Recalculate metrics
    print(f"\n--- Recalculating metrics ---")
    metrics = compute_accuracy_metrics(merged_data)
    completeness_metrics = compute_completeness_metrics(merged_data)

    # Print reports
    print("\n" + "=" * 80)
    print("MERGED DATASET METRICS")
    print("=" * 80)
    print_accuracy_report(metrics)

    print(f"\n=== COMPLETENESS ANALYSIS ===")
    print(f"Completeness Rate: {completeness_metrics['completeness_rate']:.3f}")
    print(f"Complete: {completeness_metrics['complete_responses']}, Incomplete: {completeness_metrics['incomplete_responses']}")

    # Save merged dataset
    print(f"\n--- Saving merged dataset ---")
    save_jsonl(merged_data, OUTPUT_JSONL)
    print(f"Saved {len(merged_data)} results to {OUTPUT_JSONL}")

    # Merge MMLU subjects from both summaries
    mmlu_subjects = list(set(summary_1['mmlu_subjects'] + summary_2['mmlu_subjects']))
    mmlu_subjects.sort()  # Sort for consistency

    # Create merged summary
    print(f"\n--- Creating merged summary ---")
    merged_summary = {
        'evaluation_date': TODAY,
        'model_id': summary_1['model_id'],  # Should be same in both
        'mmlu_subjects': mmlu_subjects,
        'metrics': metrics,
        'completeness_metrics': completeness_metrics,
        'processing_time_seconds': summary_1['processing_time_seconds'] + summary_2['processing_time_seconds'],
        'validation_method': summary_1['validation_method'],  # Should be same in both
        'configuration': summary_1['configuration'],  # Should be same in both
        'merge_info': {
            'merged_from': [
                DATASET_1_PATH,
                DATASET_2_PATH
            ],
            'original_counts': {
                'dataset_1': len(dataset_1),
                'dataset_2': len(dataset_2)
            },
            'merge_date': TODAY
        }
    }

    # Save merged summary
    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(merged_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {OUTPUT_SUMMARY}")

    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"\nMerged dataset: {OUTPUT_JSONL}")
    print(f"Merged summary: {OUTPUT_SUMMARY}")
    print(f"\nTotal questions: {metrics['total_questions']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"Completeness rate: {completeness_metrics['completeness_rate']:.3f}")

if __name__ == "__main__":
    merge_baseline_datasets()
