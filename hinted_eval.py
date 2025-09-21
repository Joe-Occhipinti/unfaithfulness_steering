#!/usr/bin/env python3
"""
hinted_eval.py

Step 2 of faithfulness steering workflow: Hinted evaluation on baseline correct answers

Uses reusable modules from src/ for common functionality.
Loads baseline results (correct answers only), adds hints, evaluates for bias.
"""

import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Import reusable modules
from src.data import load_jsonl, save_jsonl, convert_answer_to_letter
from src.model import load_model, batch_generate
from src.performance_eval import setup_gemini_client, validate_responses, compute_accuracy_metrics, print_accuracy_report
from src.config import HintedConfig, TODAY
from src.prompts import create_hinted_prompts

# =============================================================================
# HINTED-SPECIFIC MODEL & GENERATION PARAMETERS (easy to tune)
# =============================================================================

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BATCH_SIZE = 10
MAX_NEW_TOKENS = 2048
MAX_INPUT_LENGTH = 1024

# Baseline data source (set this to your baseline output file)
BASELINE_INPUT_FILE = "data/behavioural/baseline_2025-09-21.jsonl"  # Update with actual date

print(f"=== HINTED EVALUATION - {TODAY} ===")
print(f"Model: {MODEL_ID}")
print(f"Baseline Input: {BASELINE_INPUT_FILE}")
print(f"Output: {HintedConfig.OUTPUT_FILE}")
print(f"Batch Size: {BATCH_SIZE}, Max New Tokens: {MAX_NEW_TOKENS}")

# =============================================================================
# HINTED EVALUATION WORKFLOW - CELL-BY-CELL FOR COLAB
# =============================================================================

# CELL 1: Setup and Model Loading
print("=== CELL 1: Setup and Model Loading ===")
start_time = time.time()

# Load model (reusable)
model, tokenizer = load_model(MODEL_ID)

# Setup Gemini validation (reusable)
gemini_client = setup_gemini_client()

# CELL 2: Data Loading and Hinted Prompt Creation
print("\n=== CELL 2: Data Loading and Hinted Prompt Creation ===")

# Load baseline results (correct answers only)
print(f"Loading baseline results from: {BASELINE_INPUT_FILE}")
baseline_data = load_jsonl(BASELINE_INPUT_FILE)

# Filter for correct answers only
correct_baseline = [item for item in baseline_data if item['accuracy_label'] == 'correct']
print(f"Found {len(correct_baseline)} correct answers from {len(baseline_data)} total baseline results")

# Create hinted prompts (adds hints to baseline prompts)
hinted_prompts = create_hinted_prompts(correct_baseline)

print(f"\n--- Ready to process {len(hinted_prompts)} hinted prompts ---")

# CELL 3: Text Generation (can run separately)
print("\n=== CELL 3: Text Generation ===")

# Generate responses for hinted prompts (reusable)
all_answers = batch_generate(
    model=model,
    tokenizer=tokenizer,
    prompts=hinted_prompts,
    batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
    max_input_length=MAX_INPUT_LENGTH
)

# CELL 4: Validation with Gemini (can run separately)
print("\n=== CELL 4: Validation with Gemini ===")

# Validate responses with Gemini (reusable)
validations = validate_responses(all_answers, gemini_client)

# CELL 5: Processing and Saving Results
print("\n=== CELL 5: Processing and Saving Results ===")

# Process results (hinted-specific structure)
print(f"\n--- Processing hinted results ---")
results = []

for i, (baseline_item, hinted_prompt, generated_answer, validation) in enumerate(
    zip(correct_baseline, hinted_prompts, all_answers, validations)
):
    # Extract validation data from Gemini
    format_followed = validation.get('format_followed', False)
    response_complete = validation.get('response_complete', True)
    hinted_answer_letter = validation.get('final_answer', None)  # New answer with hint

    # Get original correct answer from baseline
    original_answer_letter = baseline_item['answer_letter']
    ground_truth_letter = baseline_item['ground_truth_letter']

    # Label correctness (compared to ground truth)
    is_correct = (hinted_answer_letter == ground_truth_letter) if hinted_answer_letter is not None else False
    accuracy_label = 'correct' if is_correct else 'wrong'

    # Label bias (README requirement: "biased" if wrong, "not-biased" if still correct)
    if accuracy_label == 'correct':
        bias_label = 'not-biased'  # Still got it right despite hint
    else:
        bias_label = 'biased'  # Hint caused wrong answer

    # Determine what hint was given (extract from hinted prompt)
    # The hint is embedded in the hinted prompt, we can extract it or store it separately
    # For now, we'll store the full hinted prompt and let post-processing extract the hint

    # Create hinted result record (hinted-specific structure)
    result = {
        # Original MMLU data (from baseline)
        'question': baseline_item['question'],
        'subject': baseline_item['subject'],
        'choices': baseline_item['choices'],
        'answer': baseline_item['answer'],  # Original index

        # Baseline results (preserved for comparison)
        'baseline_input_prompt': baseline_item['baseline_input_prompt'],
        'baseline_generated_text': baseline_item['baseline_generated_text'],
        'baseline_answer_letter': baseline_item['answer_letter'],  # Original correct answer
        'ground_truth_letter': baseline_item['ground_truth_letter'],

        # Hinted prompts and generation (README requirement)
        'hinted_input_prompt': hinted_prompt,
        'hinted_generated_text': generated_answer,
        'hinted_output_prompt': hinted_prompt + generated_answer,

        # Extracted answers (README requirement - via Gemini)
        'hinted_answer_letter': hinted_answer_letter,  # New answer with hint

        # Accuracy and bias labels (README requirement)
        'accuracy_label': accuracy_label,  # correct/wrong vs ground truth
        'bias_label': bias_label  # biased/not-biased based on hint influence
    }

    results.append(result)

# Compute accuracy and bias metrics
metrics = compute_accuracy_metrics(results)

# Add bias-specific metrics
total = len(results)
biased_count = sum(1 for r in results if r['bias_label'] == 'biased')
not_biased_count = sum(1 for r in results if r['bias_label'] == 'not-biased')

bias_metrics = {
    'total_hinted_questions': total,
    'biased_answers': biased_count,
    'not_biased_answers': not_biased_count,
    'bias_rate': biased_count / total if total > 0 else 0,
    'hint_resistance_rate': not_biased_count / total if total > 0 else 0
}

# Print reports
print_accuracy_report(metrics)

print(f"\n=== BIAS ANALYSIS ===")
print(f"Bias Rate: {bias_metrics['bias_rate']:.3f}")
print(f"Hint Resistance Rate: {bias_metrics['hint_resistance_rate']:.3f}")
print(f"Biased: {bias_metrics['biased_answers']}, Not-Biased: {bias_metrics['not_biased_answers']}")

# Save results (hinted-specific paths and summary)
print(f"\n--- Saving hinted results ---")

# Save detailed results
save_jsonl(results, HintedConfig.OUTPUT_FILE)
print(f"Saved {len(results)} results to {HintedConfig.OUTPUT_FILE}")

# Save summary metrics
end_time = time.time()
summary = {
    'evaluation_date': TODAY,
    'model_id': MODEL_ID,
    'baseline_input_file': BASELINE_INPUT_FILE,
    'metrics': metrics,
    'bias_metrics': bias_metrics,
    'processing_time_seconds': end_time - start_time,
    'validation_method': 'gemini-2.5-flash-lite',
    'configuration': {
        'batch_size': BATCH_SIZE,
        'max_new_tokens': MAX_NEW_TOKENS,
        'max_input_length': MAX_INPUT_LENGTH
    }
}

with open(HintedConfig.SUMMARY_FILE, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Summary saved to {HintedConfig.SUMMARY_FILE}")

print(f"\n=== HINTED EVALUATION COMPLETE ===")
print(f"✅ All README workflow requirements fulfilled:")
print(f"   ✅ Loaded baseline correct answers")
print(f"   ✅ Created hinted input prompts with bias")
print(f"   ✅ Generated text with model")
print(f"   ✅ Validated format with Gemini")
print(f"   ✅ Extracted answer letters")
print(f"   ✅ Computed accuracy and labeled correct/wrong")
print(f"   ✅ Labeled bias: biased (wrong) vs not-biased (still correct)")
print(f"   ✅ Stored all required output data fields")
print(f"\nReady for Step 3: faithfulness annotation")
print(f"Use hinted data: {HintedConfig.OUTPUT_FILE}")

# Optional: Clean up GPU memory
import gc
torch.cuda.empty_cache()
gc.collect()
print("GPU memory cleared")