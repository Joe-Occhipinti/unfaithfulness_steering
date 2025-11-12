"""
unnest_sampled_outputs.py

Utility script to unnest (flatten) sampled outputs into a format compatible with process_answers.py.

Handles both:
1. Hinted sampled output (from eval_hinted_sampled.py)
2. Steered sampled output (from eval_steering_sampled.py)

Converts nested structures to flat JSONL records where each line = one sample.
"""

import json
import os
from typing import List, Dict, Any

from src.data import load_jsonl, save_jsonl

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files (nested structure from generation scripts)
HINTED_SAMPLED_INPUT = "data/Lorenz_suggestion_2025-11-12/sampled_q13-36-115-26_2025-11-12.jsonl"
STEERED_SAMPLED_INPUT = "data/Lorenz_suggestion_2025-11-12/steered_sampled_q13-36-115-26_2025-11-12.jsonl"

# Output files (flat structure for process_answers.py)
HINTED_SAMPLED_OUTPUT = "data/Lorenz_suggestion_2025-11-12/sampled_q13-36-115-26_2025-11-12_flat.jsonl"
STEERED_SAMPLED_OUTPUT = "data/Lorenz_suggestion_2025-11-12/steered_sampled_q13-36-115-26_2025-11-12_flat.jsonl"

print("=== UNNESTING SAMPLED OUTPUTS ===")

# =============================================================================
# UNNEST HINTED SAMPLED OUTPUT
# =============================================================================

print("\n--- Processing Hinted Sampled Output ---")

if os.path.exists(HINTED_SAMPLED_INPUT):
    print(f"Loading: {HINTED_SAMPLED_INPUT}")
    nested_data = load_jsonl(HINTED_SAMPLED_INPUT)
    print(f"Loaded {len(nested_data)} nested records")

    flat_records = []

    for record in nested_data:
        # Extract base fields (common to all samples)
        base_fields = {
            'question_id': record.get('question_id'),
            'question': record.get('question'),
            'subject': record.get('subject'),
            'choices': record.get('choices'),
            'answer': record.get('answer'),
            'ground_truth_letter': record.get('ground_truth_letter'),
            'hint_letter': record.get('hint_letter'),
            'hint_template': record.get('hint_template'),
            'biased_input_prompt': record.get('biased_input_prompt'),
            'baseline_answer_letter': record.get('baseline_answer_letter'),
            'biased_answer_letter': record.get('biased_answer_letter'),
            'faithfulness_classification': record.get('faithfulness_classification'),
            # Add metadata about sampling
            'temperature': record.get('metadata', {}).get('temperature'),
            'model': record.get('metadata', {}).get('model'),
            'date': record.get('metadata', {}).get('date')
        }

        # Flatten samples
        for sample in record.get('samples', []):
            flat_record = {
                **base_fields,
                'sample_id': sample.get('sample_id'),
                'sampled_generated_text': sample.get('sampled_generated_text'),
                'sampled_prompt': sample.get('sampled_prompt')
            }
            flat_records.append(flat_record)

    # Save flattened output
    save_jsonl(flat_records, HINTED_SAMPLED_OUTPUT)
    print(f"Saved {len(flat_records)} flat records to: {HINTED_SAMPLED_OUTPUT}")
    print(f"  ({len(nested_data)} questions × {len(record.get('samples', []))} samples)")
else:
    print(f"File not found: {HINTED_SAMPLED_INPUT}")
    print("Skipping hinted sampled output")

# =============================================================================
# UNNEST STEERED SAMPLED OUTPUT
# =============================================================================

print("\n--- Processing Steered Sampled Output ---")

if os.path.exists(STEERED_SAMPLED_INPUT):
    print(f"Loading: {STEERED_SAMPLED_INPUT}")
    nested_data = load_jsonl(STEERED_SAMPLED_INPUT)
    print(f"Loaded {len(nested_data)} nested records")

    flat_records = []

    for record in nested_data:
        # Extract base fields (common to all samples)
        base_fields = {
            'question_id': record.get('question_id'),
            'question': record.get('question'),
            'subject': record.get('subject'),
            'choices': record.get('choices'),
            'answer': record.get('answer'),
            'ground_truth_letter': record.get('ground_truth_letter'),
            'hint_letter': record.get('hint_letter'),
            'hint_template': record.get('hint_template'),
            'biased_input_prompt': record.get('biased_input_prompt'),
            'baseline_answer_letter': record.get('baseline_answer_letter'),
            'biased_answer_letter': record.get('biased_answer_letter'),
            'faithfulness_classification': record.get('faithfulness_classification'),
            # Add metadata about sampling
            'temperature': record.get('metadata', {}).get('temperature'),
            'model': record.get('metadata', {}).get('model'),
            'date': record.get('metadata', {}).get('date')
        }

        # Flatten steering configs and samples
        for config in record.get('steering_configs', []):
            # Add steering-specific fields
            config_fields = {
                **base_fields,
                'steering_layer': config.get('steering_layer'),
                'steering_coefficient': config.get('steering_coefficient')
            }

            # Flatten samples within this config
            for sample in config.get('samples', []):
                flat_record = {
                    **config_fields,
                    'sample_id': sample.get('sample_id'),
                    'steered_sampled_generated_text': sample.get('steered_sampled_generated_text'),
                    'steered_sampled_prompt': sample.get('steered_sampled_prompt')
                }
                flat_records.append(flat_record)

    # Save flattened output
    save_jsonl(flat_records, STEERED_SAMPLED_OUTPUT)
    print(f"Saved {len(flat_records)} flat records to: {STEERED_SAMPLED_OUTPUT}")

    # Calculate summary
    num_questions = len(nested_data)
    num_configs = len(record.get('steering_configs', [])) if nested_data else 0
    num_samples_per_config = len(record.get('steering_configs', [{}])[0].get('samples', [])) if nested_data and record.get('steering_configs') else 0

    print(f"  ({num_questions} questions × {num_configs} configs × {num_samples_per_config} samples)")
else:
    print(f"File not found: {STEERED_SAMPLED_INPUT}")
    print("Skipping steered sampled output")

print("\n=== UNNESTING COMPLETE ===")
print("\nFlat outputs are ready for process_answers.py")
print("\nNext steps:")
print("  1. Update process_answers.py INPUT_JSONL to point to flat files")
print("  2. Run process_answers.py to extract answer letters and compute metrics")
