# -*- coding: utf-8 -*-
"""eval_steering_sampled.py

Steering evaluation script with multi-sample generation.

This script:
1. Loads annotated biased prompts filtered by question_ids
2. Loads pre-computed steering vectors
3. Tests multiple layer-coefficient combinations
4. Generates 50 sampled responses per configuration using temperature 0.7
5. Saves results in nested structure (all configs per question)
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
import gc
import pickle
from datetime import datetime
from typing import Dict, Any, List

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.model import load_model
from src.steering import (
    apply_steering_to_model,
    sample_generate_steered_multiple
)
from src.config import TODAY

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Input files - read from cloned repo
INPUT_PROMPTS_FILE = "data/sprint_4_2025-10-15/annotated/steered/annotated_steered_sprint4_2025-10-27.jsonl"
INPUT_VECTORS_FILE = "data/sprint_4_2025-10-15/vectors/vectors_new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"

# Question IDs to process (hardcoded list)
QUESTION_IDS = [13, 36, 115, 26]  # Modify this list as needed

# Output files - save directly to Google Drive root (no download consent needed)
OUTPUT_FILE = f"/content/drive/MyDrive/steered_sampled_q{'-'.join(map(str, QUESTION_IDS))}_{TODAY}.jsonl"
SUMMARY_FILE = f"/content/drive/MyDrive/steered_sampled_summary_q{'-'.join(map(str, QUESTION_IDS))}_{TODAY}.json"

# =============================================================================
# MODEL & SAMPLING PARAMETERS
# =============================================================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MAX_NEW_TOKENS = 2048
MAX_INPUT_LENGTH = 1024

# Sampling parameters
NUM_SAMPLES = 50
TEMPERATURE = 0.7
SAMPLE_BATCH_SIZE = 10  # Tune based on GPU memory (10, 25, 50)

# =============================================================================
# STEERING CONFIGURATION
# =============================================================================
LAYERS_TO_TEST = [15, 25]  # Middle-to-late layers typically work best
COEFFICIENTS = [0.75, -0.75]  # Positive and negative coefficients

print(f"=== STEERING EVALUATION WITH SAMPLING ===")
print(f"Model: {MODEL_ID}")
print(f"Input Prompts: {INPUT_PROMPTS_FILE}")
print(f"Steering Vectors: {INPUT_VECTORS_FILE}")
print(f"Question IDs: {QUESTION_IDS}")
print(f"Layers to test: {LAYERS_TO_TEST}")
print(f"Coefficients: {COEFFICIENTS}")
print(f"Samples per config: {NUM_SAMPLES}, Temperature: {TEMPERATURE}, Batch size: {SAMPLE_BATCH_SIZE}")
print(f"Output: {OUTPUT_FILE}")

# =============================================================================
# STEERING EVALUATION WORKFLOW - CELL-BY-CELL FOR COLAB
# =============================================================================

# CELL 1: Setup and Model Loading
print("\n=== CELL 1: Setup and Model Loading ===")
start_time = time.time()

# Load model
model, tokenizer = load_model(MODEL_ID)

print(f"Setup completed in {time.time() - start_time:.2f} seconds")

# CELL 2: Load Data and Steering Vectors
print("\n=== CELL 2: Load Data and Steering Vectors ===")

# Load annotated data
print(f"Loading annotated data from {INPUT_PROMPTS_FILE}...")
full_data = load_jsonl(INPUT_PROMPTS_FILE)
print(f"Loaded {len(full_data)} total records")

# Filter by question IDs (match actual question_id field in records)
filtered_data = []
for qid in QUESTION_IDS:
    # Find record with matching question_id field
    matching_records = [r for r in full_data if r.get('question_id') == qid]

    if len(matching_records) == 1:
        filtered_data.append(matching_records[0])
    elif len(matching_records) == 0:
        print(f"Warning: No record found with question_id={qid}")
    else:
        print(f"Warning: Multiple records found with question_id={qid}, using first one")
        filtered_data.append(matching_records[0])

print(f"Filtered to {len(filtered_data)} records based on question_ids: {QUESTION_IDS}")

# Load steering vectors
print(f"\nLoading steering vectors from {INPUT_VECTORS_FILE}...")
with open(INPUT_VECTORS_FILE, 'rb') as f:
    steering_data = pickle.load(f)

steering_vectors = steering_data['steering_vectors']
available_layers = sorted(steering_vectors.keys())
print(f"Loaded steering vectors for {len(available_layers)} layers")

# Filter layers to test based on availability
layers_to_test = [l for l in LAYERS_TO_TEST if l in available_layers]
print(f"Will test {len(layers_to_test)} layers: {layers_to_test}")

if len(layers_to_test) == 0:
    raise ValueError(f"None of the requested layers {LAYERS_TO_TEST} are available in steering vectors")

# CELL 3: Apply Steering Wrappers to Model
print("\n=== CELL 3: Apply Steering Wrappers ===")

# Apply steering wrappers to model (persistent across all generations)
wrapped_layers = apply_steering_to_model(
    model=model,
    steering_vectors=steering_vectors,
    layers_to_wrap=layers_to_test
)

print(f"Applied steering wrappers to {len(wrapped_layers)} layers")

# CELL 4: Multi-Sample Steered Generation
print("\n=== CELL 4: Multi-Sample Steered Generation ===")
print(f"Generating {NUM_SAMPLES} samples per (question × layer × coefficient) combination")
print(f"Total configurations: {len(filtered_data)} questions × {len(layers_to_test)} layers × {len(COEFFICIENTS)} coefficients")

results = []

for i, record in enumerate(filtered_data):
    qid = record['question_id']
    biased_input_prompt = record.get('biased_input_prompt')

    if not biased_input_prompt:
        print(f"Warning: No biased_input_prompt found for question_id {qid}, skipping...")
        continue

    print(f"\n{'='*80}")
    print(f"Processing question_id {qid} ({i+1}/{len(filtered_data)})")
    print(f"Question: {record.get('question', 'N/A')[:100]}...")
    print(f"{'='*80}")

    # Store steering configs for this question
    steering_configs = []

    # Test all layer × coefficient combinations
    for layer_idx in layers_to_test:
        for coeff in COEFFICIENTS:
            print(f"\n--- Config: Layer {layer_idx}, Coefficient {coeff:+.2f} ---")

            # Generate multiple steered samples
            samples_text = sample_generate_steered_multiple(
                model=model,
                tokenizer=tokenizer,
                prompt=biased_input_prompt,
                wrapped_layers=wrapped_layers,
                layer_idx=layer_idx,
                coefficient=coeff,
                steering_vectors=steering_vectors,
                num_samples=NUM_SAMPLES,
                temperature=TEMPERATURE,
                batch_size=SAMPLE_BATCH_SIZE,
                max_new_tokens=MAX_NEW_TOKENS,
                max_input_length=MAX_INPUT_LENGTH
            )

            # Structure samples
            samples = [
                {
                    'sample_id': j,
                    'steered_sampled_generated_text': sample_text,
                    'steered_sampled_prompt': biased_input_prompt + sample_text
                }
                for j, sample_text in enumerate(samples_text)
            ]

            # Store this configuration
            steering_config = {
                'steering_layer': layer_idx,
                'steering_coefficient': coeff,
                'samples': samples
            }

            steering_configs.append(steering_config)

            print(f"Generated {len(samples)} samples for layer {layer_idx}, coeff {coeff:+.2f}")

    # Create result record with nested structure (Option 1)
    result = {
        # Identifiers
        'question_id': qid,

        # Original question data
        'question': record.get('question'),
        'subject': record.get('subject'),
        'choices': record.get('choices'),
        'answer': record.get('answer'),

        # Ground truth and hints
        'ground_truth_letter': record.get('ground_truth_letter'),
        'hint_letter': record.get('hint_letter'),
        'hint_template': record.get('hint_template'),

        # Input prompt used for steering
        'biased_input_prompt': biased_input_prompt,

        # Original baseline/hinted data (for reference)
        'baseline_answer_letter': record.get('baseline_answer_letter'),
        'biased_answer_letter': record.get('biased_answer_letter'),
        'faithfulness_classification': record.get('faithfulness_classification'),

        # NEW: Steering configs with samples nested inside
        'steering_configs': steering_configs,

        # Metadata
        'metadata': {
            'num_samples_per_config': NUM_SAMPLES,
            'temperature': TEMPERATURE,
            'layers_tested': layers_to_test,
            'coefficients_tested': COEFFICIENTS,
            'num_configs': len(steering_configs),
            'model': MODEL_ID,
            'date': TODAY,
            'sample_batch_size': SAMPLE_BATCH_SIZE
        }
    }

    results.append(result)

    print(f"\nCompleted question_id {qid}: {len(steering_configs)} configs × {NUM_SAMPLES} samples each")

# CELL 5: Save Output JSONL and Summary
print("\n=== CELL 5: Save Output JSONL and Summary ===")

# Save detailed results
save_jsonl(results, OUTPUT_FILE)
print(f"Saved {len(results)} records (each with multiple steering configs) to {OUTPUT_FILE}")

# Save summary metrics
end_time = time.time()

total_configs = sum(len(r['steering_configs']) for r in results)
total_samples = total_configs * NUM_SAMPLES

summary = {
    'metadata': {
        'date': TODAY,
        'model': MODEL_ID,
        'input_file': INPUT_PROMPTS_FILE,
        'steering_vectors_file': INPUT_VECTORS_FILE,
        'output_file': OUTPUT_FILE,
        'num_questions': len(results),
        'question_ids': QUESTION_IDS,
        'total_configs': total_configs,
        'total_samples': total_samples,
        'processing_time_seconds': end_time - start_time,
        'processing_time_minutes': (end_time - start_time) / 60
    },
    'configuration': {
        'num_samples_per_config': NUM_SAMPLES,
        'temperature': TEMPERATURE,
        'sample_batch_size': SAMPLE_BATCH_SIZE,
        'layers_tested': layers_to_test,
        'coefficients_tested': COEFFICIENTS,
        'max_new_tokens': MAX_NEW_TOKENS,
        'max_input_length': MAX_INPUT_LENGTH
    },
    'results': {
        'questions_processed': len(results),
        'questions_skipped': len(QUESTION_IDS) - len(results),
        'configs_per_question': len(layers_to_test) * len(COEFFICIENTS),
        'samples_per_config': NUM_SAMPLES,
        'total_samples_generated': total_samples
    }
}

with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Summary saved to {SUMMARY_FILE}")

print(f"\n=== STEERING EVALUATION WITH SAMPLING COMPLETE ===")
print(f"Processing time: {(end_time - start_time) / 60:.2f} minutes")
print(f"\nGenerated samples for {len(results)} questions")
print(f"  Configs per question: {len(layers_to_test)} layers × {len(COEFFICIENTS)} coefficients = {len(layers_to_test) * len(COEFFICIENTS)}")
print(f"  Samples per config: {NUM_SAMPLES}")
print(f"  Total samples: {total_samples}")
print(f"\nResults saved to:")
print(f"  Data: {OUTPUT_FILE}")
print(f"  Summary: {SUMMARY_FILE}")

# CELL 6: Verify Results Saved to Google Drive
print("\n=== CELL 6: Verify Results Saved to Google Drive ===")

# Check if files exist in Drive
if os.path.exists(OUTPUT_FILE):
    print(f"✓ Output file saved to Drive: {OUTPUT_FILE}")
    print(f"  Size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")
else:
    print(f"✗ Warning: Output file not found at {OUTPUT_FILE}")

if os.path.exists(SUMMARY_FILE):
    print(f"✓ Summary file saved to Drive: {SUMMARY_FILE}")
    print(f"  Size: {os.path.getsize(SUMMARY_FILE) / 1024:.2f} KB")
else:
    print(f"✗ Warning: Summary file not found at {SUMMARY_FILE}")

print(f"\n=== EXPERIMENT COMPLETE ===")
print(f"Results are saved in your Google Drive (MyDrive root):")
print(f"  - {os.path.basename(OUTPUT_FILE)}")
print(f"  - {os.path.basename(SUMMARY_FILE)}")
print(f"\nNext steps:")
print(f"  1. Access files from your Google Drive on any device")
print(f"  2. Move files to your local repo and commit to GitHub when convenient")
print(f"  3. Run validation script to extract answer letters and compute metrics")

# CELL 7 (Optional): Clean up GPU memory
import gc
torch.cuda.empty_cache()
gc.collect()
print("\nGPU memory cleared")
