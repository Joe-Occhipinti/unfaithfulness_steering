#!/usr/bin/env python3
"""
baseline_eval.py

Step 1 of faithfulness steering workflow: Baseline evaluation on MMLU

Uses reusable modules from src/ for common functionality.
Only contains baseline-specific logic inline.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Import reusable modules
from src.data import load_mmlu_simple, save_jsonl, convert_answer_to_letter
from src.model import load_model, batch_generate
from src.performance_eval import setup_gemini_client, validate_responses, compute_accuracy_metrics, print_accuracy_report
from src.config import BaselineConfig, TODAY
from src.prompts import create_baseline_prompts

# =============================================================================
# BASELINE-SPECIFIC MODEL & GENERATION PARAMETERS (easy to tune)
# =============================================================================

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BATCH_SIZE = 10
MAX_NEW_TOKENS = 2048
MAX_INPUT_LENGTH = 1024

print(f"=== BASELINE EVALUATION - {TODAY} ===")
print(f"Model: {MODEL_ID}")
print(f"MMLU Subjects: {BaselineConfig.SUBJECTS}")
print(f"Output: {BaselineConfig.OUTPUT_FILE}")
print(f"Batch Size: {BATCH_SIZE}, Max New Tokens: {MAX_NEW_TOKENS}")

# =============================================================================
# BASELINE EVALUATION WORKFLOW - CELL-BY-CELL FOR COLAB
# =============================================================================

# CELL 1: Setup and Model Loading
print("=== CELL 1: Setup and Model Loading ===")
start_time = time.time()

# Load model (reusable)
model, tokenizer = load_model(MODEL_ID)

# Setup Gemini validation (reusable)
gemini_client = setup_gemini_client()

# CELL 2: Data Loading and Prompt Creation
print("\n=== CELL 2: Data Loading and Prompt Creation ===")

# Load MMLU data (reusable)
mmlu_data = load_mmlu_simple(BaselineConfig.SUBJECTS)

# Create baseline prompts (from prompts module)
baseline_prompts = create_baseline_prompts(mmlu_data)

print(f"\n--- Ready to process {len(baseline_prompts)} prompts ---")

# CELL 3: Text Generation (can run separately)
print("\n=== CELL 3: Text Generation ===")

# Generate responses (reusable)
all_answers = batch_generate(
    model=model,
    tokenizer=tokenizer,
    prompts=baseline_prompts,
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

# Process results (baseline-specific structure)
print(f"\n--- Processing baseline results ---")
results = []

for i, (mmlu_item, baseline_prompt, generated_answer, validation) in enumerate(
    zip(mmlu_data, baseline_prompts, all_answers, validations)
):
    # Extract validation data from Gemini
    format_followed = validation.get('format_followed', False)
    response_complete = validation.get('response_complete', True)
    answer_letter = validation.get('final_answer', None)  # This is the extracted letter

    # Get ground truth letter (reusable)
    ground_truth_letter = convert_answer_to_letter(mmlu_item['answer'])

    # Label correctness
    is_correct = (answer_letter == ground_truth_letter) if answer_letter is not None else False
    accuracy_label = 'correct' if is_correct else 'wrong'

    # Create baseline result record (essential data only)
    result = {
        # Original MMLU data
        'question': mmlu_item['question'],
        'subject': mmlu_item['subject'],
        'choices': mmlu_item['choices'],
        'answer': mmlu_item['answer'],  # Original index

        # Baseline prompts and generation (README requirement)
        'baseline_input_prompt': baseline_prompt,
        'baseline_generated_text': generated_answer,
        'baseline_prompt': baseline_prompt + generated_answer,

        # Extracted answers (README requirement - via Gemini)
        'answer_letter': answer_letter,  # Extracted by Gemini
        'ground_truth_letter': ground_truth_letter,  # Converted from index

        # Accuracy labels (README requirement)
        'accuracy_label': accuracy_label
    }

    results.append(result)

# Compute accuracy metrics (reusable)
metrics = compute_accuracy_metrics(results)

# Print report (reusable)
print_accuracy_report(metrics)

# Save results (baseline-specific paths and summary)
print(f"\n--- Saving baseline results ---")

# Save detailed results
save_jsonl(results, BaselineConfig.OUTPUT_FILE)
print(f"Saved {len(results)} results to {BaselineConfig.OUTPUT_FILE}")

# Save summary metrics
end_time = time.time()
summary = {
    'evaluation_date': TODAY,
    'model_id': MODEL_ID,
    'mmlu_subjects': BaselineConfig.SUBJECTS,
    'metrics': metrics,
    'processing_time_seconds': end_time - start_time,
    'validation_method': 'gemini-2.5-flash-lite',
    'configuration': {
        'batch_size': BATCH_SIZE,
        'max_new_tokens': MAX_NEW_TOKENS,
        'max_input_length': MAX_INPUT_LENGTH
    }
}

with open(BaselineConfig.SUMMARY_FILE, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Summary saved to {BaselineConfig.SUMMARY_FILE}")

print(f"\n=== BASELINE EVALUATION COMPLETE ===")
print(f"✅ All README workflow requirements fulfilled:")
print(f"   ✅ Loaded MMLU subjects")
print(f"   ✅ Created baseline input prompts")
print(f"   ✅ Generated text with model")
print(f"   ✅ Validated format with Gemini")
print(f"   ✅ Extracted answer letters")
print(f"   ✅ Computed accuracy and labeled correct/wrong")
print(f"   ✅ Stored all required output data fields")
print(f"\nReady for Step 2: hinted_eval.py")
print(f"Use baseline data: {BaselineConfig.OUTPUT_FILE}")

# Optional: Clean up GPU memory
import gc
torch.cuda.empty_cache()
gc.collect()
print("GPU memory cleared")