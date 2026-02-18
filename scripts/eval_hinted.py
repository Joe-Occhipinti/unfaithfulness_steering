"""
eval_hinted_runpod.py

Step 2 of faithfulness steering workflow: Hinted evaluation on baseline correct answers
Runs on RunPod GPU environment.

This script takes baseline MMLU results and creates hinted versions to test model bias:
- For correct baseline answers → Add WRONG hints (test unfaithfulness)
- For wrong baseline answers → Add CORRECT hints (test faithfulness recovery)

Supports multiple models with automatic baseline file discovery.

Usage:
    python eval_hinted_runpod.py \\
        --models "Qwen/Qwen3-32B" "Qwen/Qwen3-8B" \\
        --input-dir ./data/behavioural \\
        --batch-size 100 \\
        --num-samples 10

    python eval_hinted_runpod.py \\
        --models "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \\
        --num-samples 5
"""

import sys
import os
# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import time
import torch
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple
from glob import glob

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.model import load_model, batch_generate, load_model_vllm, batch_generate_vllm
from src.config import TODAY, ModelConfig
from src.prompts import create_hinted_prompts


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hinted evaluation on baseline MMLU results for multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python eval_hinted_runpod.py \\
        --models "Qwen/Qwen3-32B" "Qwen/Qwen3-8B" \\
        --batch-size 100 \\
        --num-samples 10
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+", 
        default=["Qwen/Qwen3-32B", "Qwen/Qwen3-8B"],
        help="List of model IDs to evaluate (default: Qwen3-32B and Qwen3-8B)"
    )
    
    # I/O configuration
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default=None,
        help="Directory containing baseline results files (default: same as script)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (default: same as script)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size for generation (default: auto-select - 100 for 32B, 400 for 8B)"
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
    
    # Hint configuration
    parser.add_argument(
        "--bias-strategies",
        type=str,
        nargs="+",
        default=["grader_hacking", "metadata", "professor"],
        help="Bias strategies/templates to use (default: all four strategies)"
    )
    parser.add_argument(
        "--distribution-strategy",
        type=str,
        default="round_robin",
        choices=["by_subject", "round_robin", "random", "all_per_prompt"],
        help="How to distribute hint templates (default: round_robin - all templates used evenly)"
    )
    
    # Data configuration
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=None,
        help="Number of samples to process per model (default: all)"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Filter baseline data by MMLU subjects (optional)"
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
    # e.g., "Qwen/Qwen3-32B" -> "Qwen3-32B"
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
        Batch size (100 for 32B models, 400 for 8B models, 10 for unknown)
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


def find_baseline_file(input_dir: Path, model_short_name: str) -> Path:
    """Find the most recent baseline results file for a given model.
    
    Searches the entire codebase recursively for baseline files matching
    the model short name and selects the most recent by date.
    
    Args:
        input_dir: Root directory to start search (typically codebase root)
        model_short_name: Short model name (e.g., "Qwen3-32B", "qwen332b")
        
    Returns:
        Path to the most recent baseline file
        
    Raises:
        FileNotFoundError: If no baseline file found for this model
    """
    # Normalize model short name for flexible matching
    # Try both the original format and a simplified version
    model_lower = model_short_name.lower().replace('-', '').replace('_', '').replace('.', '')
    
    # Search recursively from input_dir
    # Pattern: baseline_results_*_YYYY-MM-DD.jsonl
    all_baseline_files = list(input_dir.rglob("baseline_results_*.jsonl"))
    
    # Also search parent directories if input_dir might be a subdirectory
    root_dir = Path(__file__).parent.resolve()
    if root_dir != input_dir:
        all_baseline_files.extend(list(root_dir.rglob("baseline_results_*.jsonl")))
    
    # Deduplicate
    all_baseline_files = list(set(all_baseline_files))
    
    # Filter by model name (flexible matching)
    matching_files = []
    for f in all_baseline_files:
        filename = f.stem.lower().replace('-', '').replace('_', '').replace('.', '')
        # Check if model name appears in filename
        if model_lower in filename:
            matching_files.append(f)
    
    if not matching_files:
        # Try with original short name format
        for f in all_baseline_files:
            if model_short_name.lower() in f.stem.lower():
                matching_files.append(f)
    
    if not matching_files:
        raise FileNotFoundError(
            f"No baseline results file found for model '{model_short_name}' in {input_dir}\n"
            f"Searched {len(all_baseline_files)} baseline files across codebase.\n"
            f"Expected pattern: baseline_results_*{model_short_name}*_YYYY-MM-DD.jsonl"
        )
    
    # Sort by date in filename (YYYY-MM-DD format) and return most recent
    # Filename format: baseline_results_{model}_{date}.jsonl
    def extract_date(path: Path) -> str:
        """Extract date from filename for sorting."""
        stem = path.stem  # baseline_results_Qwen3-32B_2025-12-28
        parts = stem.split('_')
        # Date is typically the last part
        return parts[-1] if parts else ""
    
    most_recent = sorted(matching_files, key=extract_date)[-1]
    return most_recent


def get_subject_to_template_mapping() -> Dict[str, int]:
    """Get the default subject-to-template mapping for hint distribution.
    
    This matches the original eval_hinted.py configuration.
    """
    return {
        "high_school_psychology": 0,
        "high_school_macroeconomics": 2,
        "high_school_microeconomics": 2,
        "high_school_european_history": 1,
        "high_school_us_history": 1,
        "high_school_world_history": 1,
        "prehistory": 1,
        "high_school_chemistry": 0,
        "high_school_biology": 0,
        "college_biology": 0,
        "college_chemistry": 0
    }


def evaluate_single_model(
    model_id: str,
    input_dir: Path,
    output_dir: Path,
    args
) -> Tuple[List[Dict], Dict]:
    """Run hinted evaluation for a single model.
    
    Args:
        model_id: Model identifier
        input_dir: Directory containing baseline results
        output_dir: Directory for output files
        args: Command-line arguments
        
    Returns:
        Tuple of (results list, summary dict)
    """
    
    model_short_name = get_model_short_name(model_id)
    
    # Determine batch size for this model
    batch_size = get_batch_size_for_model(model_id, args.batch_size)
    
    # Define output files for this model
    output_file = output_dir / f"hinted_results_{model_short_name}_{TODAY}.jsonl"
    summary_file = output_dir / f"hinted_summary_{model_short_name}_{TODAY}.json"
    
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_id}")
    print(f"{'='*60}")
    print(f"Backend: {args.backend}")
    print(f"Output: {output_file}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    
    start_time = time.time()
    
    # ==========================================================================
    # Model Loading (backend-specific)
    # ==========================================================================
    
    if args.backend == "vllm":
        print("\n--- Loading model with vLLM ---")
        llm = load_model_vllm(
            model_id=model_id,
            tensor_parallel_size=1,
            max_model_len=args.max_input_length + args.max_new_tokens
        )
    else:
        batch_size = get_batch_size_for_model(model_id, args.batch_size)
        print(f"Batch Size: {batch_size} (auto-detected)" if args.batch_size is None else f"Batch Size: {batch_size} (override)")
        print("\n--- Loading model with HuggingFace ---")
        model, tokenizer = load_model(model_id)
    
    # ==========================================================================
    # Load Baseline Data
    # ==========================================================================
    print("\n--- Loading baseline results ---")
    
    # Find baseline file for this model
    baseline_file = find_baseline_file(input_dir, model_short_name)
    print(f"Found baseline file: {baseline_file}")
    
    # Load baseline data
    baseline_data = load_jsonl(str(baseline_file))
    print(f"Loaded {len(baseline_data)} baseline results")
    
    # Filter by subjects if specified
    if args.subjects:
        baseline_data = [item for item in baseline_data if item['subject'] in args.subjects]
        print(f"Filtered to {len(baseline_data)} results matching subjects: {args.subjects}")
    
    # Limit samples if specified
    if args.num_samples is not None:
        baseline_data = baseline_data[:args.num_samples]
        print(f"Limited to first {args.num_samples} samples for testing")
    
    # Split into correct and wrong baseline answers
    correct_baseline = [item for item in baseline_data if item['baseline_accuracy'] == 'correct']
    wrong_baseline = [item for item in baseline_data if item['baseline_accuracy'] == 'wrong']
    print(f"Split: {len(correct_baseline)} correct, {len(wrong_baseline)} wrong baseline answers")
    
    # ==========================================================================
    # Create Hinted Prompts - ALL TEMPLATES PER PROMPT
    # ==========================================================================
    print("\n--- Creating hinted prompts ---")
    print(f"Strategy: Each prompt will receive ALL {len(args.bias_strategies)} templates")
    print(f"Templates: {args.bias_strategies}")
    
    # Create wrong hints for correct baseline answers (test unfaithfulness)
    # Each prompt gets EVERY template
    print(f"\nCreating wrong hinted prompts for {len(correct_baseline)} correct baseline answers")
    print(f"This will generate {len(correct_baseline)} × {len(args.bias_strategies)} = {len(correct_baseline) * len(args.bias_strategies)} total prompts")
    
    all_wrong_hint_prompts = []
    all_wrong_hint_info = []
    all_combined_baseline = []
    
    # For each template, create hints for ALL correct baseline prompts
    for template_idx, template_name in enumerate(args.bias_strategies):
        print(f"  Processing template {template_idx + 1}/{len(args.bias_strategies)}: {template_name}")
        
        # Create prompts using only this specific template
        prompts, hint_infos = create_hinted_prompts(
            correct_baseline,
            hint_mode="wrong",
            bias_strategies=[template_name],  # Only use this one template
            distribution_strategy="single",   # Single template per call
            distribution_config=None,
            random_seed=42,
            return_hint_info=True
        )
        
        # Add these prompts to the collection
        all_wrong_hint_prompts.extend(prompts)
        all_wrong_hint_info.extend(hint_infos)
        all_combined_baseline.extend(correct_baseline)  # Duplicate the baseline for each template
    
    combined_baseline = all_combined_baseline
    hinted_prompts = all_wrong_hint_prompts
    hint_info_list = all_wrong_hint_info
    
    print(f"\nReady to process {len(hinted_prompts)} total hinted prompts")
    
    # ==========================================================================
    # Text Generation (backend-specific)
    # ==========================================================================
    print("\n--- Generating responses ---")
    
    if args.backend == "vllm":
        all_answers = batch_generate_vllm(
            llm=llm,
            prompts=hinted_prompts,
            max_new_tokens=args.max_new_tokens
        )
        # vLLM cleanup
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    else:
        all_answers = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=hinted_prompts,
            batch_size=batch_size,
            max_new_tokens=args.max_new_tokens,
            max_input_length=args.max_input_length
        )
        # HuggingFace cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    # ==========================================================================
    # Process and Store Results
    # ==========================================================================
    print("\n--- Processing results ---")
    print("Note: Validation and answer extraction will be done in separate script")
    
    results = []
    
    for i, (baseline_item, hinted_prompt, generated_answer, hint_info) in enumerate(
        zip(combined_baseline, hinted_prompts, all_answers, hint_info_list)
    ):
        # Get baseline information
        baseline_accuracy = baseline_item['baseline_accuracy']
        
        # Create hinted result record (same structure as original eval_hinted.py)
        result = {
            # Original MMLU data (from baseline)
            'hinted_id': i,
            'hint_template': hint_info['hint_template'],

            # Hinted prompts and generation
            'hinted_input_prompt': hinted_prompt,
            'hinted_generated_text': generated_answer,

            # Hint information (what hint was given)
            'hint_letter': hint_info['hint_letter'],    

            # Baseline results (preserved for comparison)
            'baseline_id': baseline_item['id'],
            'ground_truth_letter': baseline_item['ground_truth_letter'],
            'baseline_answer_letter': baseline_item['baseline_answer_letter'],
            'baseline_accuracy': baseline_accuracy,

            # Metadata
            'subject': baseline_item['subject'],
            'date': TODAY,
            'model': model_id,
            'backend': args.backend,
        }
        
        results.append(result)
    
    print(f"Stored {len(results)} hinted results (validation will be done separately)")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    print("\n--- Saving results ---")
    save_jsonl(results, str(output_file))
    print(f"Saved {len(results)} results to {output_file}")
    
    # Create summary
    end_time = time.time()
    summary = {
        'metadata': {
            'date': TODAY,
            'model': model_id,
            'backend': args.backend,
            'baseline_file': str(baseline_file),
            'hinted_file': str(output_file),
            'num_examples': len(results),
            'processing_time_seconds': end_time - start_time
        },
        'configuration': {
            'max_new_tokens': args.max_new_tokens,
            'max_input_length': args.max_input_length,
            'bias_strategies': args.bias_strategies,
            'distribution_strategy': 'all_per_prompt'  # Each prompt gets all templates via loop
        },
        'hint_distribution': {
            'total_prompts': len(results),
            'correct_baseline_count': len(correct_baseline),
            'templates_per_prompt': len(args.bias_strategies)
        },
        'note': 'Validation and answer extraction done separately in process_answers.py'
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {summary_file}")
    print("GPU memory cleared")
    
    return results, summary


def main():
    """Main evaluation workflow."""
    
    args = parse_args()
    
    # Determine input/output directories
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(__file__).parent.resolve()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HINTED EVALUATION (RunPod)")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Models to evaluate: {len(args.models)}")
    for i, model in enumerate(args.models, 1):
        print(f"  {i}. {model}")
    print(f"Bias Strategies: {args.bias_strategies}")
    print(f"Distribution Strategy: {args.distribution_strategy}")
    if args.backend == "hf":
        print(f"Batch Size: {args.batch_size or 'auto-detect'}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Max Input Length: {args.max_input_length}")
    print(f"Num Samples: {args.num_samples or 'all'}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    
    # ==========================================================================
    # Iterate over models
    # ==========================================================================
    all_results = {}
    all_summaries = {}
    
    for i, model_id in enumerate(args.models, 1):
        print(f"\n\n{'#'*60}")
        print(f"# MODEL {i}/{len(args.models)}")
        print(f"{'#'*60}")
        
        try:
            results, summary = evaluate_single_model(
                model_id=model_id,
                input_dir=input_dir,
                output_dir=output_dir,
                args=args
            )
            
            all_results[model_id] = results
            all_summaries[model_id] = summary
            
        except FileNotFoundError as e:
            print(f"\n⚠️  ERROR: {e}")
            print(f"Skipping model {model_id}")
            continue
        except Exception as e:
            print(f"\n⚠️  ERROR processing {model_id}: {e}")
            print(f"Skipping model {model_id}")
            continue
    
    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n\n" + "=" * 60)
    print("HINTED EVALUATION COMPLETE - ALL MODELS")
    print("=" * 60)
    
    for model_id, summary in all_summaries.items():
        hint_dist = summary['hint_distribution']
        print(f"\n{model_id}:")
        print(f"  Total prompts: {hint_dist['total_prompts']}")
        print(f"  Correct baseline count: {hint_dist['correct_baseline_count']}")
        print(f"  Templates per prompt: {hint_dist['templates_per_prompt']}")
        print(f"  Time: {summary['metadata']['processing_time_seconds']:.1f}s")
    
    print(f"\nResults saved to: {output_dir}")
    print("\nNext step: Run validate_hinted.py to validate responses and compute metrics")
    
    return all_results, all_summaries


if __name__ == "__main__":
    main()
