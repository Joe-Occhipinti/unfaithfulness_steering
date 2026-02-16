"""
eval_baseline_runpod.py

Step 1 of faithfulness steering workflow: Baseline evaluation on MMLU
Runs on RunPod GPU environment.

Results are saved locally in the same directory as the script.

Usage:
    python eval_baseline_runpod.py \
        --models "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "Qwen/Qwen2.5-32B-Instruct" \
        --subjects high_school_macroeconomics high_school_microeconomics \
        --batch-size 100 \
        --max-new-tokens 2048 \
        --max-input-length 1024 \
        --num-samples 5
"""

import argparse
import json
import time
import torch
import gc
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Import reusable modules
from src.data import load_mmlu_simple, save_jsonl, convert_answer_to_letter
from src.model import load_model, batch_generate, load_model_vllm, batch_generate_vllm
from src.config import TODAY, ModelConfig
from src.prompts import create_baseline_prompts


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Baseline evaluation on MMLU for multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python eval_baseline_runpod.py \\
        --models "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "Qwen/Qwen2.5-32B-Instruct" \\
        --subjects high_school_macroeconomics high_school_microeconomics \\
        --batch-size 100 \\
        --num-samples 5
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+", 
        default=["Qwen/Qwen3-32B"],
        help="List of model IDs to evaluate (default: Qwen3-32B)"
    )
    
    # MMLU subjects
    parser.add_argument(
        "--subjects", 
        type=str, 
        nargs="+", 
        default=[
            'high_school_psychology',
            'high_school_chemistry', 
            'high_school_biology',
            'college_biology',
            'college_chemistry',
            'prehistory',
            'high_school_european_history',
            'high_school_us_history',
            'high_school_world_history'
        ],
        help="MMLU subjects to evaluate"
    )
    
    # Generation parameters
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size for generation (default: auto-select based on model size - 70 for 32B, 250 for 8B)"
    )
    parser.add_argument(
        "--max-new-tokens", 
        type=int, 
        default=2048,
        help="Maximum new tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--max-input-length", 
        type=int, 
        default=1024,
        help="Maximum input length (default: 1024)"
    )
    
    # Data configuration
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=None,
        help="Number of samples to process (default: all)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (default: same as script)"
    )
    
    # Backend configuration
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "hf"],
        default="vllm",
        help="Inference backend: 'vllm' (recommended, default) or 'hf' (HuggingFace)"
    )
    
    return parser.parse_args()


def get_model_short_name(model_id: str) -> str:
    """Extract a short name from a model ID for file naming."""
    # Get the last part of the model path and clean it
    short_name = model_id.split("/")[-1]
    # Replace characters that are problematic in filenames
    short_name = short_name.replace(" ", "_")
    return short_name


def get_batch_size_for_model(model_id: str, override_batch_size: int = None) -> int:
    """Determine optimal batch size based on model size.
    
    Args:
        model_id: The model identifier
        override_batch_size: If provided, use this value instead of auto-detection
        
    Returns:
        Batch size (70 for 32B models, 250 for 8B models, 10 for unknown)
    """
    if override_batch_size is not None:
        return override_batch_size
    
    model_lower = model_id.lower()
    
    # Check for 32B models
    if "32b" in model_lower or "32B" in model_id:
        return 100
    # Check for 8B models
    elif "8b" in model_lower or "8B" in model_id:
        return 400
    # Check for other common sizes
    elif "70b" in model_lower or "70B" in model_id:
        return 50
    elif "7b" in model_lower or "7B" in model_id:
        return 400
    else:
        # Default conservative batch size
        print(f"Warning: Could not determine model size from '{model_id}', using batch_size=10")
        return 10


def evaluate_single_model(
    model_id: str,
    mmlu_data: List[Dict],
    baseline_prompts: List[str],
    args,
    output_dir: Path
) -> tuple:
    """Run evaluation for a single model."""
    
    model_short_name = get_model_short_name(model_id)
    
    # Define output files for this model
    output_file = output_dir / f"baseline_results_{model_short_name}_{TODAY}.jsonl"
    summary_file = output_dir / f"baseline_summary_{model_short_name}_{TODAY}.json"
    
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_id}")
    print(f"{'='*60}")
    print(f"Backend: {args.backend}")
    print(f"Output: {output_file}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    
    start_time = time.time()
    
    # ==========================================================================
    # Model Loading & Generation (backend-specific)
    # ==========================================================================
    
    if args.backend == "vllm":
        # vLLM path - high performance, automatic batching
        print("\n--- Loading model with vLLM ---")
        llm = load_model_vllm(
            model_id=model_id,
            tensor_parallel_size=1,
            max_model_len=args.max_input_length + args.max_new_tokens
        )
        
        print("\n--- Generating responses with vLLM ---")
        # Apply repetition penalty only for Qwen3 models (they tend to repeat)
        rep_penalty = 1.2 if "qwen3" in model_id.lower() else 1.0
        all_answers = batch_generate_vllm(
            llm=llm,
            prompts=baseline_prompts,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=rep_penalty
        )
        
        # vLLM cleanup
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
    else:
        # HuggingFace path - original implementation
        batch_size = get_batch_size_for_model(model_id, args.batch_size)
        print(f"Batch Size: {batch_size} (auto-detected)" if args.batch_size is None else f"Batch Size: {batch_size} (override)")
        
        print("\n--- Loading model with HuggingFace ---")
        model, tokenizer = load_model(model_id)
        
        print("\n--- Generating responses ---")
        all_answers = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=baseline_prompts,
            batch_size=batch_size,
            max_new_tokens=args.max_new_tokens,
            max_input_length=args.max_input_length
        )
        
        # HuggingFace cleanup
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    # ==========================================================================
    # Processing Results
    # ==========================================================================
    print("\n--- Processing results ---")
    results = []
    
    for i, (mmlu_item, baseline_prompt, generated_answer) in enumerate(
        zip(mmlu_data, baseline_prompts, all_answers)
    ):
        ground_truth_letter = convert_answer_to_letter(mmlu_item['answer'])
        
        result = {
            # Unique identifier
            'id': i,
            
            # Original MMLU data
            'question': mmlu_item['question'],
            'subject': mmlu_item['subject'],
            'choices': mmlu_item['choices'],
            'ground_truth_letter': ground_truth_letter,
            
            # Baseline prompts and generation
            'baseline_input_prompt': baseline_prompt,
            'baseline_generated_text': generated_answer,
            
            # Model info
            'model_id': model_id,
            'backend': args.backend
        }
        
        results.append(result)
    
    # Print basic statistics
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Total responses generated: {len(results)}")
    
    # ==========================================================================
    # Saving Results
    # ==========================================================================
    print("\n--- Saving results ---")
    save_jsonl(results, str(output_file))
    print(f"Saved {len(results)} results to {output_file}")
    
    end_time = time.time()
    summary = {
        'evaluation_date': TODAY,
        'model_id': model_id,
        'backend': args.backend,
        'mmlu_subjects': args.subjects,
        'num_samples': len(results),
        'processing_time_seconds': end_time - start_time,
        'configuration': {
            'max_new_tokens': args.max_new_tokens,
            'max_input_length': args.max_input_length
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {summary_file}")
    print("GPU memory cleared")
    
    return results, summary


def main():
    """Main evaluation workflow."""
    
    args = parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BASELINE EVALUATION (RunPod)")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Models to evaluate: {len(args.models)}")
    for i, model in enumerate(args.models, 1):
        print(f"  {i}. {model}")
    print(f"MMLU Subjects: {args.subjects}")
    if args.backend == "hf":
        print(f"Batch Size: {args.batch_size or 'auto'}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Max Input Length: {args.max_input_length}")
    print(f"Num Samples: {args.num_samples or 'all'}")
    print(f"Output Directory: {output_dir}")
    
    # ==========================================================================
    # Setup (shared across models)
    # ==========================================================================
    print("\n=== SETUP ===")
    
    # Load MMLU data
    print("\n--- Loading MMLU data ---")
    mmlu_data = load_mmlu_simple(args.subjects)
    
    if args.num_samples is not None:
        mmlu_data = mmlu_data[:args.num_samples]
        print(f"Limited to first {args.num_samples} samples for testing")
    
    # Create baseline prompts (same for all models)
    baseline_prompts = create_baseline_prompts(mmlu_data)
    print(f"Prepared {len(baseline_prompts)} prompts")
    
    # ==========================================================================
    # Iterate over models
    # ==========================================================================
    all_results = {}
    all_summaries = {}
    
    for i, model_id in enumerate(args.models, 1):
        print(f"\n\n{'#'*60}")
        print(f"# MODEL {i}/{len(args.models)}")
        print(f"{'#'*60}")
        
        results, summary = evaluate_single_model(
            model_id=model_id,
            mmlu_data=mmlu_data,
            baseline_prompts=baseline_prompts,
            args=args,
            output_dir=output_dir
        )
        
        all_results[model_id] = results
        all_summaries[model_id] = summary
    
    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n\n" + "=" * 60)
    print("EVALUATION COMPLETE - ALL MODELS")
    print("=" * 60)
    
    for model_id, summary in all_summaries.items():
        print(f"\n{model_id}:")
        print(f"  Samples: {summary['num_samples']}")
        print(f"  Time: {summary['processing_time_seconds']:.1f}s")
    
    print(f"\nResults saved to: {output_dir}")
    print("\nReady for Step 2: hinted_eval.py")
    
    return all_results, all_summaries


if __name__ == "__main__":
    
    all_results, all_summaries = main()
