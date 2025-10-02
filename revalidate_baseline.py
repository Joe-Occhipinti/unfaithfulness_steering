"""
revalidate_baseline.py

Script to revalidate baseline responses that have null answer_letter.
Uses the updated validation window (last 500 chars instead of 200).
"""

import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.data import load_jsonl, save_jsonl
from src.performance_eval import setup_openrouter_client, validate_responses, extract_validation_data, label_accuracy

# Input file to revalidate
INPUT_FILE = "data/behavioural/baseline_high_school_macroeconomics_microeconomics_2025-10-02.jsonl"
OUTPUT_FILE = "data/behavioural/baseline_high_school_macroeconomics_microeconomics_2025-10-02_revalidated.jsonl"
BACKUP_FILE = "data/behavioural/baseline_high_school_macroeconomics_microeconomics_2025-10-02_backup.jsonl"

print(f"=== BASELINE REVALIDATION ===")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Backup: {BACKUP_FILE}")

# Load baseline data
print(f"\nLoading baseline data from: {INPUT_FILE}")
baseline_data = load_jsonl(INPUT_FILE)
print(f"Loaded {len(baseline_data)} total records")

# Create backup
print(f"Creating backup at: {BACKUP_FILE}")
save_jsonl(baseline_data, BACKUP_FILE)

# Filter for entries where response_complete is false (these need revalidation with 500-char window)
incomplete_entries = [(idx, item) for idx, item in enumerate(baseline_data) if item.get('response_complete') == False]
print(f"Found {len(incomplete_entries)} entries with response_complete=false")

if len(incomplete_entries) == 0:
    print("No entries to revalidate. Exiting.")
    exit(0)

# Setup OpenRouter client for validation
print("\nSetting up OpenRouter client...")
openrouter_client = setup_openrouter_client()

# Extract generated texts for revalidation
print(f"\nPreparing {len(incomplete_entries)} responses for revalidation...")
responses_to_validate = [item['baseline_generated_text'] for idx, item in incomplete_entries]

# Revalidate responses with updated window (500 chars)
print("\nRevalidating responses with DeepSeek (last 500 chars)...")
validations = validate_responses(responses_to_validate, openrouter_client)

# Update the baseline data with new validation results
print("\nUpdating records with new validation results...")
updated_count = 0
still_null = []
now_complete = 0

for (idx, item), validation in zip(incomplete_entries, validations):
    # Extract validation data
    format_followed, response_complete, answer_letter = extract_validation_data(validation)

    # Get ground truth
    ground_truth_letter = item['ground_truth_letter']

    # Label correctness
    is_correct, accuracy_label = label_accuracy(answer_letter, ground_truth_letter)

    # Track statistics
    if response_complete:
        now_complete += 1

    # Update the item in baseline_data
    baseline_data[idx]['answer_letter'] = answer_letter
    baseline_data[idx]['format_followed'] = format_followed
    baseline_data[idx]['response_complete'] = response_complete
    baseline_data[idx]['accuracy_label'] = accuracy_label

    if answer_letter is not None:
        updated_count += 1
        print(f"  ✓ Record {idx}: answer_letter={answer_letter}, complete={response_complete}, accuracy={accuracy_label}")
    else:
        still_null.append(idx)
        print(f"  ✗ Record {idx}: still null (complete={response_complete}, format={format_followed})")
        # Show a snippet of the response for debugging
        snippet = item['baseline_generated_text'][-200:] if len(item['baseline_generated_text']) > 200 else item['baseline_generated_text']
        print(f"      Last 200 chars: ...{snippet}")

print(f"\n=== REVALIDATION STATISTICS ===")
print(f"Total revalidated: {len(incomplete_entries)}")
print(f"Successfully extracted answer: {updated_count} ({updated_count/len(incomplete_entries)*100:.1f}%)")
print(f"Still null after revalidation: {len(still_null)} ({len(still_null)/len(incomplete_entries)*100:.1f}%)")
print(f"Now marked as complete: {now_complete} ({now_complete/len(incomplete_entries)*100:.1f}%)")
print(f"Still incomplete: {len(incomplete_entries) - now_complete}")

# Save updated baseline data
print(f"\nSaving updated data to: {OUTPUT_FILE}")
save_jsonl(baseline_data, OUTPUT_FILE)

print(f"\n=== REVALIDATION COMPLETE ===")
print(f"Original file backed up to: {BACKUP_FILE}")
print(f"Updated file saved to: {OUTPUT_FILE}")
print(f"\nTo use the updated file, replace the original:")
print(f"  cp {OUTPUT_FILE} {INPUT_FILE}")
