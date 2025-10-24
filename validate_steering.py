"""
validate_steering.py

Offline validation script for steered responses.
Runs separately after eval_steering.py to avoid API rate limits during GPU runs.

This script:
1. Loads raw steered responses from eval_steering.py output
2. Validates responses with OpenRouter (answer extraction, compliance, completeness)
3. Computes accuracy metrics
4. Saves validated output with all metrics
5. Generates summary statistics

Can be run locally (no GPU needed) after all steering experiments complete.
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

# Input: Raw steered responses from eval_steering.py
INPUT_JSONL = "data/sprint4_2025-10-21/steered/steered_val_neg_bas_psyXprof_2025-10-19.jsonl"
INPUT_SUMMARY = "data/sprint4_2025-10-21/summaries/steered/summary_steered_neg_bas_psyXprof_2025-10-19.json"

# Output: Overwrites input files with validated/enriched data
OUTPUT_JSONL = INPUT_JSONL  # Overwrites the original JSONL
OUTPUT_SUMMARY = INPUT_SUMMARY  # Merges validation metrics into original summary

print(f"=== STEERING VALIDATION ===")
print(f"Input JSONL: {INPUT_JSONL}")
print(f"Input Summary: {INPUT_SUMMARY}")
print(f"Note: Will overwrite inputs with validated/enriched data")

# =============================================================================
# VALIDATION WORKFLOW
# =============================================================================

start_time = time.time()

# STEP 1: Load raw steered responses and original summary
print("\n=== STEP 1: Load Raw Steered Responses and Summary ===")
raw_data = load_jsonl(INPUT_JSONL)
print(f"Loaded {len(raw_data)} raw records from JSONL")

with open(INPUT_SUMMARY, 'r', encoding='utf-8') as f:
    original_summary = json.load(f)
print(f"Loaded original summary")

# Group by configuration
configs = {}
for record in raw_data:
    key = (record['steering_layer'], record['steering_coefficient'])
    if key not in configs:
        configs[key] = []
    configs[key].append(record)

print(f"Found {len(configs)} steering configurations")

# STEP 2: Setup OpenRouter client
print("\n=== STEP 2: Setup OpenRouter Client ===")
openrouter_client = setup_openrouter_client()
print("Client ready")

# STEP 3: Validate responses for each configuration
print("\n=== STEP 3: Validate Responses ===")
print("Note: This may take time due to API rate limits")

validated_data = []
config_stats = {}

for (layer_idx, coeff), records in tqdm(configs.items(), desc="Validating configurations"):
    print(f"\nValidating layer {layer_idx}, coefficient {coeff:+.1f}")
    print(f"  {len(records)} responses to validate")

    # Extract steered responses
    steered_responses = [r['steered_response'] for r in records]

    # Validate with OpenRouter (with rate limit handling)
    try:
        steered_validations = validate_responses(steered_responses, openrouter_client)
    except Exception as e:
        print(f"  Error during validation: {e}")
        print(f"  Skipping this configuration")
        continue

    # Extract validation metrics
    steered_answers = []
    compliance_labels = []
    completeness_labels = []

    for validation in steered_validations:
        is_compliant, is_complete, answer_letter = extract_validation_data(validation)
        steered_answers.append(answer_letter)
        compliance_labels.append('compliant' if is_compliant else 'non_compliant')
        completeness_labels.append('complete' if is_complete else 'truncated')

    # Compute accuracy
    correct_count = 0
    steered_accuracy_labels = []

    for i, record in enumerate(records):
        ground_truth = record['ground_truth_letter']
        steered_answer = steered_answers[i]

        is_correct = (steered_answer == ground_truth) if (steered_answer and ground_truth) else False
        steered_accuracy_labels.append('correct' if is_correct else 'wrong')

        if is_correct:
            correct_count += 1

    # Compute aggregate metrics
    accuracy_rate = correct_count / len(records) if records else 0
    compliance_rate = sum(1 for c in compliance_labels if c == 'compliant') / len(compliance_labels) if compliance_labels else 0
    completeness_rate = sum(1 for c in completeness_labels if c == 'complete') / len(completeness_labels) if completeness_labels else 0

    # Store stats for summary
    config_stats[(layer_idx, coeff)] = {
        'accuracy_rate': accuracy_rate,
        'correct_count': correct_count,
        'total_prompts': len(records),
        'compliance_rate': compliance_rate,
        'completeness_rate': completeness_rate
    }

    print(f"  Accuracy: {accuracy_rate:.1%} ({correct_count}/{len(records)})")
    print(f"  Compliance: {compliance_rate:.1%}, Completeness: {completeness_rate:.1%}")

    # Add validation data to records
    for i, record in enumerate(records):
        validated_record = record.copy()
        validated_record['steered_answer_letter'] = steered_answers[i]
        validated_record['compliance'] = compliance_labels[i]
        validated_record['completeness'] = completeness_labels[i]
        validated_record['steered_accuracy'] = steered_accuracy_labels[i]
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

# Add validation metrics to each configuration in original summary
for key, stats in config_stats.items():
    layer, coeff = key
    config_key = f"layer_{layer}_coeff_{coeff:+.1f}"

    if config_key in original_summary['all_configurations']:
        original_summary['all_configurations'][config_key].update(stats)
    else:
        print(f"  Warning: {config_key} not found in original summary")

# Add validation metadata
original_summary['metadata']['validation_date'] = TODAY
original_summary['metadata']['validation_time_seconds'] = end_time - start_time
original_summary['metadata']['note'] = 'Validation completed - metrics added'

# Remove old note if exists
if 'note' in original_summary:
    del original_summary['note']

# Save enriched summary (overwrites original)
with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
    json.dump(original_summary, f, indent=2, ensure_ascii=False)

print(f"Summary enriched and saved to {OUTPUT_SUMMARY}")
print(f"  (Original file overwritten with validation metrics)")

print(f"\n=== VALIDATION COMPLETE ===")
print(f"Validation time: {(end_time - start_time) / 60:.2f} minutes")
print(f"Validated {len(validated_data)} records across {len(config_stats)} configurations")
print(f"\nNext steps:")
print(f"  - Analyze validated results to select best steering configuration")
print(f"  - Run faithfulness evaluation on best configuration")
