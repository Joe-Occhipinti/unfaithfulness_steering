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
from src.performance_eval import setup_gemini_client, validate_responses_batch, compute_accuracy_metrics, print_accuracy_report
from src.config import BaselineConfig, TODAY
from src.prompts import create_baseline_prompts

print(f"=== BASELINE EVALUATION - {TODAY} ===")
print(f"Model: {BaselineConfig.MODEL}")
print(f"MMLU Subjects: {BaselineConfig.SUBJECTS}")
print(f"Output: {BaselineConfig.OUTPUT_FILE}")

# =============================================================================
# MAIN BASELINE EVALUATION WORKFLOW
# =============================================================================

def main():
    start_time = time.time()

    # Load model (reusable)
    model, tokenizer = load_model(BaselineConfig.MODEL)

    # Setup Gemini validation (reusable)
    gemini_client = setup_gemini_client()

    # Load MMLU data (reusable)
    mmlu_data = load_mmlu_simple(BaselineConfig.SUBJECTS)

    # Create baseline prompts (from prompts module)
    baseline_prompts = create_baseline_prompts(mmlu_data)

    print(f"\n--- Ready to process {len(baseline_prompts)} prompts ---")

    # Generate responses (reusable)
    all_answers = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=baseline_prompts,
        batch_size=BaselineConfig.BATCH_SIZE,
        max_new_tokens=BaselineConfig.MAX_NEW_TOKENS,
        max_input_length=BaselineConfig.MAX_INPUT_LENGTH
    )

    # Validate responses with Gemini (reusable)
    validations = validate_responses_batch(all_answers, gemini_client)

    # Process results (baseline-specific structure)
    print(f"\n--- Processing baseline results ---")
    results = []

    for i, (mmlu_item, baseline_prompt, generated_answer, validation) in enumerate(
        zip(mmlu_data, baseline_prompts, all_answers, validations)
    ):
        # Extract validation data
        format_followed = validation.get('format_followed', False)
        response_complete = validation.get('response_complete', True)
        final_answer = validation.get('final_answer', None)

        # Get ground truth letter (reusable)
        ground_truth_letter = convert_answer_to_letter(mmlu_item['answer'])

        # Label correctness
        is_correct = (final_answer == ground_truth_letter) if final_answer is not None else False
        accuracy_label = 'correct' if is_correct else 'wrong'

        # Create baseline result record (baseline-specific structure)
        result = {
            # Original MMLU data
            'question': mmlu_item['question'],
            'subject': mmlu_item['subject'],
            'choices': mmlu_item['choices'],
            'answer': mmlu_item['answer'],  # Original index
            'split': mmlu_item['split'],

            # Baseline prompts and generation (README requirement)
            'baseline_input_prompt': baseline_prompt,
            'baseline_generated_text': generated_answer,
            'baseline_output_prompt': baseline_prompt + generated_answer,

            # Extracted answers (README requirement)
            'answer_letter': final_answer,  # Via Gemini validation
            'ground_truth_letter': ground_truth_letter,

            # Accuracy labels (README requirement)
            'accuracy_label': accuracy_label,

            # Validation metadata
            'format_followed': format_followed,
            'response_complete': response_complete,
            'final_answer': final_answer,
            'evaluation_timestamp': datetime.now().isoformat()
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
        'model_id': BaselineConfig.MODEL,
        'mmlu_subjects': BaselineConfig.SUBJECTS,
        'metrics': metrics,
        'processing_time_seconds': end_time - start_time,
        'validation_method': 'gemini-2.5-flash-lite',
        'configuration': {
            'batch_size': BaselineConfig.BATCH_SIZE,
            'max_new_tokens': BaselineConfig.MAX_NEW_TOKENS,
            'max_input_length': BaselineConfig.MAX_INPUT_LENGTH
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

if __name__ == "__main__":
    main()