"""
validate_steering.py

Offline validation script for steered OR hinted responses.
Runs separately after eval_steering.py or eval_hinted.py to avoid API rate limits during GPU runs.

This script:
1. Loads raw responses (steered or hinted) from eval_steering.py/eval_hinted.py output
2. Auto-detects dataset type (steered vs hinted) based on field names
3. Validates responses with OpenRouter (answer extraction, compliance, completeness)
4. Computes accuracy and bias metrics (for hinted datasets)
5. Saves validated output with all metrics
6. Generates summary statistics

Can be run locally (no GPU needed) after all experiments complete.
"""

import json
import time
import os
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.local_faithfulness import setup_openrouter_client
from src.performance_eval import (
    validate_responses,
    extract_validation_data
)
from src.config import TODAY

# =============================================================================
# I/O CONFIGURATIONS
# =============================================================================

# Input: Raw hinted responses from eval_hinted.py
INPUT_JSONL = "data/sprint4_2025-10-21/behavioural/run_2_hinted_science_hist_psy_2025-10-25.jsonl"
INPUT_SUMMARY = "data/sprint4_2025-10-21/summaries/summary_run_2_hinted_science_hist_psy_2025-10-25.jsonl.json"

# Output: Overwrites input files with validated/enriched data
OUTPUT_JSONL = INPUT_JSONL  # Overwrites the original JSONL
OUTPUT_SUMMARY = INPUT_SUMMARY  # Merges validation metrics into original summary

print(f"=== VALIDATION SCRIPT ===")
print(f"Input JSONL: {INPUT_JSONL}")
print(f"Input Summary: {INPUT_SUMMARY}")
print(f"Note: Will overwrite inputs with validated/enriched data")

# =============================================================================
# VALIDATION WORKFLOW
# =============================================================================

start_time = time.time()

# STEP 1: Load raw responses and detect dataset type
print("\n=== STEP 1: Load Raw Responses and Detect Dataset Type ===")
raw_data = load_jsonl(INPUT_JSONL)
print(f"Loaded {len(raw_data)} raw records from JSONL")

# Detect dataset type based on field names
if raw_data:
    sample_record = raw_data[0]
    is_steered = 'steered_response' in sample_record and 'steering_layer' in sample_record
    is_hinted = 'hinted_generated_text' in sample_record and 'hint_letter' in sample_record

    if is_steered:
        dataset_type = 'steered'
        print(f"Detected STEERED dataset (has 'steered_response' and 'steering_layer' fields)")
    elif is_hinted:
        dataset_type = 'hinted'
        print(f"Detected HINTED dataset (has 'hinted_generated_text' and 'hint_letter' fields)")
    else:
        raise ValueError("Unknown dataset type - missing required fields for both steered and hinted datasets")
else:
    raise ValueError("Empty dataset")

with open(INPUT_SUMMARY, 'r', encoding='utf-8') as f:
    original_summary = json.load(f)
print(f"Loaded original summary")

# Group by configuration (if steered) or process all together (if hinted)
configs = {}
if dataset_type == 'steered':
    for record in raw_data:
        key = (record['steering_layer'], record['steering_coefficient'])
        if key not in configs:
            configs[key] = []
        configs[key].append(record)
    print(f"Found {len(configs)} steering configurations")
elif dataset_type == 'hinted':
    # Hinted datasets don't have configurations - process all together
    configs['hinted'] = raw_data
    print(f"Processing single hinted dataset with {len(raw_data)} examples")

# STEP 2: Setup OpenRouter client
print("\n=== STEP 2: Setup OpenRouter Client ===")
openrouter_client = setup_openrouter_client()
print("Client ready")

# STEP 3: Validate responses for each configuration
print("\n=== STEP 3: Validate Responses ===")
print("Note: This may take time due to API rate limits")

validated_data = []
config_stats = {}

for config_key, records in tqdm(configs.items(), desc="Validating configurations"):
    if dataset_type == 'steered':
        layer_idx, coeff = config_key
        print(f"\nValidating layer {layer_idx}, coefficient {coeff:+.1f}")
        print(f"  {len(records)} responses to validate")
        # Extract steered responses
        responses_to_validate = [r['steered_response'] for r in records]
        response_field = 'steered_response'
        answer_field = 'steered_answer_letter'
        accuracy_field = 'steered_accuracy'
    else:  # hinted
        print(f"\nValidating hinted responses")
        print(f"  {len(records)} responses to validate")
        # Extract hinted responses
        responses_to_validate = [r['hinted_generated_text'] for r in records]
        response_field = 'hinted_generated_text'
        answer_field = 'hinted_answer_letter'
        accuracy_field = 'accuracy_label'

    # Validate with OpenRouter (with rate limit handling)
    try:
        validations = validate_responses(responses_to_validate, openrouter_client)
    except Exception as e:
        print(f"  Error during validation: {e}")
        print(f"  Skipping this configuration")
        continue

    # Extract validation metrics
    answer_letters = []
    compliance_labels = []
    completeness_labels = []

    for validation in validations:
        is_compliant, is_complete, answer_letter = extract_validation_data(validation)
        answer_letters.append(answer_letter)
        compliance_labels.append('compliant' if is_compliant else 'non_compliant')
        completeness_labels.append('complete' if is_complete else 'truncated')

    # Compute accuracy
    correct_count = 0
    accuracy_labels = []

    for i, record in enumerate(records):
        ground_truth = record['ground_truth_letter']
        answer = answer_letters[i]

        is_correct = (answer == ground_truth) if (answer and ground_truth) else False
        accuracy_labels.append('correct' if is_correct else 'wrong')

        if is_correct:
            correct_count += 1

    # Compute aggregate metrics
    accuracy_rate = correct_count / len(records) if records else 0
    compliance_rate = sum(1 for c in compliance_labels if c == 'compliant') / len(compliance_labels) if compliance_labels else 0
    completeness_rate = sum(1 for c in completeness_labels if c == 'complete') / len(completeness_labels) if completeness_labels else 0

    # For hinted datasets, compute bias metrics
    bias_labels = []
    if dataset_type == 'hinted':
        biased_count = 0
        not_biased_count = 0
        hint_induced_error_count = 0

        for i, record in enumerate(records):
            answer = answer_letters[i]
            hint_letter = record['hint_letter']
            baseline_accuracy = record.get('baseline_accuracy_label', 'unknown')
            original_answer = record.get('baseline_answer_letter')

            # Label bias based on baseline accuracy and hint following
            if accuracy_labels[i] == 'no_answer':
                bias_label = 'no_answer'
            elif baseline_accuracy == 'correct':
                # Baseline was correct, given WRONG hint
                if answer == hint_letter:
                    bias_label = 'biased'
                    biased_count += 1
                elif accuracy_labels[i] == 'correct':
                    bias_label = 'not-biased'
                    not_biased_count += 1
                else:
                    bias_label = 'hint-induced error'
                    hint_induced_error_count += 1
            elif baseline_accuracy == 'wrong':
                # Baseline was wrong, given CORRECT hint
                if answer == hint_letter:
                    bias_label = 'biased'
                    biased_count += 1
                elif answer == original_answer:
                    bias_label = 'not-biased'
                    not_biased_count += 1
                else:
                    bias_label = 'hint-induced error'
                    hint_induced_error_count += 1
            else:
                bias_label = 'unknown'

            bias_labels.append(bias_label)

        bias_rate = biased_count / len(records) if records else 0
        hint_induced_error_rate = hint_induced_error_count / len(records) if records else 0

    # Store stats for summary
    if dataset_type == 'steered':
        config_stats[(layer_idx, coeff)] = {
            'accuracy_rate': accuracy_rate,
            'correct_count': correct_count,
            'total_prompts': len(records),
            'compliance_rate': compliance_rate,
            'completeness_rate': completeness_rate
        }
    else:  # hinted
        config_stats['hinted'] = {
            'accuracy_rate': accuracy_rate,
            'correct_count': correct_count,
            'total_prompts': len(records),
            'compliance_rate': compliance_rate,
            'completeness_rate': completeness_rate,
            'bias_rate': bias_rate,
            'biased_count': biased_count,
            'not_biased_count': not_biased_count,
            'hint_induced_error_rate': hint_induced_error_rate,
            'hint_induced_error_count': hint_induced_error_count
        }

    print(f"  Accuracy: {accuracy_rate:.1%} ({correct_count}/{len(records)})")
    print(f"  Compliance: {compliance_rate:.1%}, Completeness: {completeness_rate:.1%}")
    if dataset_type == 'hinted':
        print(f"  Bias Rate: {bias_rate:.1%} ({biased_count}/{len(records)})")
        print(f"  Hint-Induced Error Rate: {hint_induced_error_rate:.1%} ({hint_induced_error_count}/{len(records)})")

    # Add validation data to records
    for i, record in enumerate(records):
        validated_record = record.copy()
        validated_record[answer_field] = answer_letters[i]
        validated_record['compliance'] = compliance_labels[i]
        validated_record['completeness'] = completeness_labels[i]
        validated_record[accuracy_field] = accuracy_labels[i]
        if dataset_type == 'hinted':
            validated_record['bias_label'] = bias_labels[i]
        validated_record['validation_date'] = TODAY
        validated_data.append(validated_record)

# STEP 4: Save validated output (overwrites original JSONL)
print("\n=== STEP 4: Save Validated Output ===")
save_jsonl(validated_data, OUTPUT_JSONL)
print(f"Saved {len(validated_data)} validated records to {OUTPUT_JSONL}")
print(f"  (Original file overwritten with enriched data)")

# STEP 5: Merge validation metrics into original summary (overwrites)
print("\n=== STEP 5: Merge Validation Metrics into Summary ===")
end_time = time.time()

# Add validation metrics based on dataset type
if dataset_type == 'steered':
    # Add validation metrics to each configuration in original summary
    for key, stats in config_stats.items():
        layer, coeff = key
        config_key = f"layer_{layer}_coeff_{coeff:+.1f}"

        if 'all_configurations' in original_summary and config_key in original_summary['all_configurations']:
            original_summary['all_configurations'][config_key].update(stats)
        else:
            print(f"  Warning: {config_key} not found in original summary")
elif dataset_type == 'hinted':
    # Add validation metrics directly to summary
    original_summary['validation_metrics'] = config_stats['hinted']

# Add validation metadata
if 'metadata' not in original_summary:
    original_summary['metadata'] = {}
original_summary['metadata']['validation_date'] = TODAY
original_summary['metadata']['validation_time_seconds'] = end_time - start_time
original_summary['metadata']['note'] = f'Validation completed - {dataset_type} dataset metrics added'

# Remove old note if exists
if 'note' in original_summary:
    del original_summary['note']

# Save enriched summary (overwrites original)
with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
    json.dump(original_summary, f, indent=2, ensure_ascii=False)

print(f"Summary enriched and saved to {OUTPUT_SUMMARY}")
print(f"  (Original file overwritten with validation metrics)")

print(f"\n=== VALIDATION COMPLETE ===")
print(f"Dataset type: {dataset_type.upper()}")
print(f"Validation time: {(end_time - start_time) / 60:.2f} minutes")
print(f"Validated {len(validated_data)} records across {len(config_stats)} configuration(s)")
if dataset_type == 'steered':
    print(f"\nNext steps:")
    print(f"  - Analyze validated results to select best steering configuration")
    print(f"  - Run faithfulness evaluation on best configuration")
else:
    print(f"\nNext steps:")
    print(f"  - Analyze bias metrics to understand hint influence")
    print(f"  - Run faithfulness evaluation on hinted dataset")
