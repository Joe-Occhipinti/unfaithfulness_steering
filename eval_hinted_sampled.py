# -*- coding: utf-8 -*-
"""sample_generation.py

Multi-sample generation script: Generate multiple sampled responses for specific prompts.

Takes records with specific question_ids from a dataset, generates 50 responses per prompt
using temperature 0.7 (instead of deterministic generation).
"""

# Commented out IPython magic to ensure Python compatibility.
# Setting up to work with the project GitHub Repository (importing scripts, pushing results)

# Mount Google Drive first (this requires one-time consent, but persists across runs)
from google.colab import drive
drive.mount('/content/drive')

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

# Install required packages (same as eval_hinted.py)
!pip install torch==2.4.1 transformers==4.44.2 bitsandbytes==0.43.3 accelerate==0.34.2

import json
import time
import torch
import sys
from datetime import datetime
from typing import Dict, Any, List

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.model import load_model, sample_generate_multiple
from src.config import TODAY

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Input file - read from cloned repo
INPUT_FILE = "data/sprint_4_2025-10-15/annotated/steered/annotated_steered_sprint4_2025-10-27.jsonl"

# Question IDs to process (hardcoded list)
QUESTION_IDS = [13, 36, 115, 26]  # Modify this list as needed

# Output files - save directly to Google Drive root (no download consent needed)
OUTPUT_FILE = f"/content/drive/MyDrive/sampled_q{'-'.join(map(str, QUESTION_IDS))}_{TODAY}.jsonl"
SUMMARY_FILE = f"/content/drive/MyDrive/sampled_summary_q{'-'.join(map(str, QUESTION_IDS))}_{TODAY}.json"

# =============================================================================
# MODEL & SAMPLING PARAMETERS (matching eval_hinted.py configs)
# =============================================================================

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MAX_NEW_TOKENS = 2048
MAX_INPUT_LENGTH = 1024

# NEW: Sampling parameters
NUM_SAMPLES = 50
TEMPERATURE = 0.7
SAMPLE_BATCH_SIZE = 50  # Tune based on GPU memory (10, 25, 50)

print(f"=== MULTI-SAMPLE GENERATION ===")
print(f"Model: {MODEL_ID}")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Question IDs: {QUESTION_IDS}")
print(f"Samples per question: {NUM_SAMPLES}, Temperature: {TEMPERATURE}, Batch size: {SAMPLE_BATCH_SIZE}")

# =============================================================================
# MULTI-SAMPLE GENERATION WORKFLOW - CELL-BY-CELL FOR COLAB
# =============================================================================

# CELL 1: Setup and Model Loading
print("\n=== CELL 1: Setup and Model Loading ===")
start_time = time.time()

# Load model (reusable)
model, tokenizer = load_model(MODEL_ID)

# CELL 2: Data Loading and Filtering
print("\n=== CELL 2: Data Loading and Filtering ===")

# Load full dataset
print(f"Loading dataset from: {INPUT_FILE}")
full_data = load_jsonl(INPUT_FILE)
print(f"Loaded {len(full_data)} total records")

# Filter by question IDs (using list index as question_id)
filtered_data = []
for qid in QUESTION_IDS:
    if qid < len(full_data):
        record = full_data[qid]
        # Add question_id to record for tracking
        record['question_id'] = qid
        filtered_data.append(record)
    else:
        print(f"Warning: question_id {qid} out of range (max: {len(full_data)-1})")

print(f"Filtered to {len(filtered_data)} records based on question_ids: {QUESTION_IDS}")

# CELL 3: Multi-Sample Generation (FLAT structure)
print("\n=== CELL 3: Multi-Sample Generation ===")
print("Saving in FLAT structure (one record per sample)")

flat_results = []

for i, record in enumerate(filtered_data):
    qid = record['question_id']
    biased_input_prompt = record.get('biased_input_prompt')

    if not biased_input_prompt:
        print(f"Warning: No biased_input_prompt found for question_id {qid}, skipping...")
        continue

    print(f"\n--- Processing question_id {qid} ({i+1}/{len(filtered_data)}) ---")
    print(f"Question: {record.get('question', 'N/A')[:100]}...")

    # Generate multiple samples
    samples_text = sample_generate_multiple(
        model=model,
        tokenizer=tokenizer,
        prompt=biased_input_prompt,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_input_length=MAX_INPUT_LENGTH,
        batch_size=SAMPLE_BATCH_SIZE
    )

    # Create flat records (one per sample)
    base_fields = {
        'question_id': qid,
        'question': record.get('question'),
        'subject': record.get('subject'),
        'choices': record.get('choices'),
        'answer': record.get('answer'),
        'ground_truth_letter': record.get('ground_truth_letter'),
        'hint_letter': record.get('hint_letter'),
        'hint_template': record.get('hint_template'),
        'biased_input_prompt': biased_input_prompt,
        'baseline_answer_letter': record.get('baseline_answer_letter'),
        'biased_answer_letter': record.get('biased_answer_letter'),
        'faithfulness_classification': record.get('faithfulness_classification'),
        # Metadata
        'temperature': TEMPERATURE,
        'model': MODEL_ID,
        'date': TODAY
    }

    # Create one flat record per sample
    for j, sample_text in enumerate(samples_text):
        flat_record = {
            **base_fields,
            'sample_id': j,
            'sampled_generated_text': sample_text,
            'sampled_prompt': biased_input_prompt + sample_text
        }
        flat_results.append(flat_record)

    print(f"Generated {len(samples_text)} samples for question_id {qid}")

# CELL 4: Save Output JSONL and Summary
print("\n=== CELL 4: Save Output JSONL and Summary ===")

# Save flat results
save_jsonl(flat_results, OUTPUT_FILE)
print(f"Saved {len(flat_results)} flat records to {OUTPUT_FILE}")

# Calculate number of questions processed
num_questions_processed = len(set(r['question_id'] for r in flat_results))

# Save summary metrics
end_time = time.time()
summary = {
    'metadata': {
        'date': TODAY,
        'model': MODEL_ID,
        'input_file': INPUT_FILE,
        'output_file': OUTPUT_FILE,
        'num_questions': num_questions_processed,
        'total_samples': len(flat_results),
        'processing_time_seconds': end_time - start_time,
        'processing_time_minutes': (end_time - start_time) / 60
    },
    'configuration': {
        'question_ids': QUESTION_IDS,
        'num_samples_per_question': NUM_SAMPLES,
        'temperature': TEMPERATURE,
        'sample_batch_size': SAMPLE_BATCH_SIZE,
        'max_new_tokens': MAX_NEW_TOKENS,
        'max_input_length': MAX_INPUT_LENGTH
    },
    'results': {
        'questions_processed': num_questions_processed,
        'questions_skipped': len(QUESTION_IDS) - num_questions_processed,
        'samples_per_question': NUM_SAMPLES,
        'total_samples_generated': len(flat_results)
    }
}

with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Summary saved to {SUMMARY_FILE}")

print(f"\n=== MULTI-SAMPLE GENERATION COMPLETE ===")
print(f"Processing time: {(end_time - start_time) / 60:.2f} minutes")
print(f"\nGenerated {NUM_SAMPLES} samples for {num_questions_processed} questions")
print(f"Total flat records: {len(flat_results)}")
print(f"\nResults saved to:")
print(f"  Data: {OUTPUT_FILE}")
print(f"  Summary: {SUMMARY_FILE}")

# CELL 5: Verify Results Saved to Google Drive
print("\n=== CELL 5: Verify Results Saved to Google Drive ===")

# Check if files exist in Drive
if os.path.exists(OUTPUT_FILE):
    print(f"✓ Output file saved to Drive: {OUTPUT_FILE}")
    print(f"  Size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")
else:
    print(f"✗ Warning: Output file not found at {OUTPUT_FILE}")

if os.path.exists(SUMMARY_FILE):
    print(f"Summary file saved to Drive: {SUMMARY_FILE}")
    print(f"  Size: {os.path.getsize(SUMMARY_FILE) / 1024:.2f} KB")
else:
    print(f"Warning: Summary file not found at {SUMMARY_FILE}")

print(f"\n=== EXPERIMENT COMPLETE ===")
print(f"Results are saved in your Google Drive (MyDrive root):")
print(f"  - {os.path.basename(OUTPUT_FILE)}")
print(f"  - {os.path.basename(SUMMARY_FILE)}")
print(f"\nNext steps:")
print(f"  1. Access files from your Google Drive on any device")
print(f"  2. Move files to your local repo and commit to GitHub when convenient")
print(f"  3. Run validation script to extract answer letters and compute metrics")

# CELL 6 (Optional): Clean up GPU memory
import gc
torch.cuda.empty_cache()
gc.collect()
print("\nGPU memory cleared")
