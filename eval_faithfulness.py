"""
eval_global_faithfulness.py --> this is for hinted. it would need adjustments to work for steered.

Standalone global faithfulness evaluation script using LLM-judge approach.
Uses the global annotation prompt (faithfulness_global_annotation_professor.txt)
to directly classify faithfulness without tag-based annotation.

Processes hinted evaluation results and annotates biased prompts for faithfulness.

Features:
- Auto-discovers input files based on --model and --dataset-type
- Batch processing with configurable batch size
- Checkpointing to save progress periodically
- Resume capability from checkpoint if interrupted
- Robust retry mechanism with exponential backoff
- Malformed JSON recovery
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Tuple, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import reusable modules
from src.data import load_jsonl, save_jsonl
from src.global_faithfulness import (
    setup_openrouter_client, judge_batch
)
from src.config import TODAY, ANNOTATED_DIR
import random


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'data_dir': "data/definitive_pipeline_data",
    'judge_model': "google/gemini-2.5-flash",
    'max_retries': 3,
    'batch_size': 20,  # Process and checkpoint every N records
    'max_retries': 3,
    'batch_size': 20,  # Process and checkpoint every N records
    'checkpoint_dir': "checkpoints",
    'train_ratio': 0.7,
}

# =============================================================================
# FILE PATH RESOLUTION
# =============================================================================

def resolve_file_paths(
    model: str,
    dataset_type: str,
    data_dir: str,
    date: Optional[str] = None
) -> Tuple[str, str]:
    """Resolve input file paths based on model and dataset type.
    
    Args:
        model: Model short name (e.g., 'Qwen3-8B')
        dataset_type: Type of dataset (hinted, hinted_sampled)
        data_dir: Base data directory
        date: Optional specific date (YYYY-MM-DD). If None, finds most recent.
        
    Returns:
        Tuple of (jsonl_path, summary_path)
        
    Raises:
        FileNotFoundError: If no matching files found
    """
    # Map dataset type to file prefix
    prefix_map = {
        'hinted': 'hinted',
        'hinted_sampled': 'hinted_sampled'
    }
    
    if dataset_type not in prefix_map:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Must be one of {list(prefix_map.keys())}")
    
    prefix = prefix_map[dataset_type]
    
    # Build model directory path
    model_dir = Path(data_dir) / model
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if date:
        # Use specific date
        jsonl_path = model_dir / f"{prefix}_results_{model}_{date}.jsonl"
        summary_path = model_dir / f"{prefix}_summary_{model}_{date}.json"
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
    else:
        # Find most recent file matching pattern
        pattern = str(model_dir / f"{prefix}_results_{model}_*.jsonl")
        matching_files = glob(pattern)
        
        if not matching_files:
            raise FileNotFoundError(
                f"No {prefix} results file found for model '{model}' in {model_dir}\n"
                f"Expected pattern: {prefix}_results_{model}_YYYY-MM-DD.jsonl"
            )
        
        # Get most recent (sorted by filename which includes date)
        jsonl_path = Path(sorted(matching_files)[-1])
        
        # Derive summary path from jsonl path
        summary_filename = jsonl_path.name.replace('_results_', '_summary_').replace('.jsonl', '.json')
        summary_path = jsonl_path.parent / summary_filename
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    return str(jsonl_path), str(summary_path)


def get_output_paths(
    model: str,
    data_dir: str,
    date: str
) -> Tuple[str, str]:
    """Generate output file paths for annotated results and summary.
    
    Args:
        model: Model short name
        data_dir: Base data directory
        date: Date string for output files
        
    Returns:
        Tuple of (annotated_jsonl_path, summary_json_path)
    """
    model_dir = Path(data_dir) / model
    
    annotated_path = model_dir / f"faithfulness_annotated_{model}_{date}.jsonl"
    summary_path = model_dir / f"faithfulness_summary_{model}_{date}.json"
    
    return str(annotated_path), str(summary_path)

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def get_checkpoint_path(input_file: str, checkpoint_dir: str) -> str:
    """Generate checkpoint file path based on input file name."""
    input_basename = Path(input_file).stem
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"checkpoint_faithfulness_{input_basename}.json")


def save_checkpoint(checkpoint_path: str, processed_indices: list, judgments: list, 
                    annotated_results: list, batch_idx: int):
    """Save processing checkpoint for resume capability."""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'processed_indices': processed_indices,
        'judgments': judgments,
        'annotated_results': annotated_results,
        'last_batch_idx': batch_idx,
        'total_processed': len(processed_indices)
    }
    
    # Write to temp file first, then rename for atomicity
    temp_path = checkpoint_path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)
    os.replace(temp_path, checkpoint_path)
    

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def clear_checkpoint(checkpoint_path: str):
    """Remove checkpoint file after successful completion."""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Checkpoint cleared: {checkpoint_path}")


# =============================================================================
# MALFORMED JSON RECOVERY
# =============================================================================

def fix_malformed_classification(raw_response: str) -> str:
    """
    Attempt to extract classification from malformed JSON responses.
    
    Args:
        raw_response: Raw LLM response that failed JSON parsing
        
    Returns:
        Extracted classification ('faithful', 'unfaithful') or None
    """
    if not raw_response:
        return None
        
    # Patterns to match classification in various malformed formats
    patterns = [
        r'"classification"\s*:\s*"(faithful|unfaithful)"',  # Standard JSON
        r"'classification'\s*:\s*'(faithful|unfaithful)'",  # Single quotes
        r'classification[\'\"]?\s*[:=]\s*[\'\"]?(faithful|unfaithful)[\'\"]?',  # Loose
        r'\b(faithful|unfaithful)\b',  # Just the word (last resort)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return None


def fix_error_judgments(annotated_results: list, judgments: list, verbose: bool = True) -> int:
    """
    Attempt to fix malformed JSON responses that were classified as errors.
    
    Args:
        annotated_results: List of annotated result dictionaries
        judgments: List of judgment dictionaries
        verbose: Print progress
        
    Returns:
        Number of errors fixed
    """
    if verbose:
        print(f"\n=== Checking for malformed JSON in error classifications ===")
    
    errors_fixed = 0
    
    for i, (annotated_result, judgment) in enumerate(zip(annotated_results, judgments)):
        if judgment.get('classification') == 'error':
            raw_response = judgment.get('raw_response')
            extracted = fix_malformed_classification(raw_response)
            
            if extracted in ['faithful', 'unfaithful']:
                if verbose:
                    print(f"  Fixed record {i}: extracted '{extracted}' from malformed response")
                
                # Update judgment
                judgment['classification'] = extracted
                judgment['success'] = True
                judgment['fixed_from_malformed'] = True
                
                # Update annotated result (only faithfulness_classification, no global_ fields)
                annotated_result['faithfulness_classification'] = extracted
                
                # Re-create tagged prompt - extract original prompt from existing annotated version
                current_tagged = annotated_result.get('annotated_biased_prompt', '')
                # Remove any existing tags to get base prompt
                import re as re_inner
                biased_prompt = re_inner.sub(r'\[/?[FU]_final\]', '', current_tagged)
                
                if extracted == 'faithful':
                    tagged_prompt = f"[F_final]{biased_prompt}[/F_final]"
                else:
                    tagged_prompt = f"[U_final]{biased_prompt}[/U_final]"
                annotated_result['annotated_biased_prompt'] = tagged_prompt
                
                errors_fixed += 1
    
    if verbose:
        if errors_fixed > 0:
            print(f"Fixed {errors_fixed} malformed JSON responses")
        else:
            print(f"No malformed JSON found in error classifications")
    
    return errors_fixed


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_global_faithfulness_metrics(judgments):
    """
    Compute faithfulness metrics from global LLM judge classifications.

    Args:
        judgments: List of judgment results from judge_batch

    Returns:
        Dictionary with faithfulness metrics
    """
    total = len(judgments)

    # Count classifications
    classifications = {
        "faithful": 0,
        "unfaithful": 0,
        "error": 0
    }

    for judgment in judgments:
        classification = judgment.get('classification', 'error')
        if classification in classifications:
            classifications[classification] += 1
        else:
            classifications["error"] += 1

    return {
        "total_judged": total,
        "classifications": classifications,
        "faithful_rate": classifications["faithful"] / total if total > 0 else 0,
        "unfaithful_rate": classifications["unfaithful"] / total if total > 0 else 0,
        "error_rate": classifications["error"] / total if total > 0 else 0
    }


def print_global_faithfulness_report(metrics):
    """
    Print formatted global faithfulness evaluation report.

    Args:
        metrics: Faithfulness metrics dictionary
    """
    print(f"\n=== GLOBAL FAITHFULNESS EVALUATION RESULTS ===")
    print(f"Total Judged: {metrics['total_judged']}")
    print(f"\nClassification Distribution:")
    for classification, count in metrics['classifications'].items():
        percentage = (count / metrics['total_judged'] * 100) if metrics['total_judged'] > 0 else 0
        class_name = classification.capitalize()
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    print(f"\nFaithfulness Rates:")
    print(f"  Faithful: {metrics['faithful_rate']:.3f}")
    print(f"  Unfaithful: {metrics['unfaithful_rate']:.3f}")
    print(f"  Error: {metrics['error_rate']:.3f}")



# =============================================================================
# BATCH PROCESSING
# =============================================================================

def create_annotated_result(result: dict, judgment: dict) -> dict:
    """
    Create annotated result from original result and judgment.
    
    Args:
        result: Original biased result
        judgment: Judgment from LLM judge
        
    Returns:
        Annotated result dictionary with cleaned fields
    """
    annotated_result = result.copy()
    
    # Remove fields not needed in output
    fields_to_remove = ['question', 'choices']
    for field in fields_to_remove:
        annotated_result.pop(field, None)
    
    # Rename key fields from hinted to biased since these are confirmed biased cases
    if 'hinted_input_prompt' in annotated_result:
        annotated_result['biased_input_prompt'] = annotated_result.pop('hinted_input_prompt')
    if 'hinted_answer' in annotated_result:
        annotated_result['biased_answer'] = annotated_result.pop('hinted_answer')
    if 'hinted_answer_letter' in annotated_result:
        annotated_result['biased_answer_letter'] = annotated_result.pop('hinted_answer_letter')
    
    # Get generated text for concatenation, then remove it
    hinted_generated_text = annotated_result.pop('hinted_generated_text', None)
    
    # Create biased_prompt temporarily for creating annotated version
    if 'hinted_prompt' in annotated_result:
        biased_prompt = annotated_result.pop('hinted_prompt')
    else:
        # Concatenate biased_input_prompt + hinted_generated_text
        input_prompt = annotated_result.get('biased_input_prompt', '')
        generated_text = hinted_generated_text or ''
        biased_prompt = input_prompt + generated_text
    
    # Add faithfulness classification (only one field, not duplicated)
    annotated_result['faithfulness_classification'] = judgment.get('classification')
    
    # Create tagged biased prompt for activation extraction
    classification = judgment.get('classification')
    
    if classification == 'faithful':
        tagged_prompt = f"[F_final]{biased_prompt}[/F_final]"
    elif classification == 'unfaithful':
        tagged_prompt = f"[U_final]{biased_prompt}[/U_final]"
    else:
        tagged_prompt = biased_prompt
    
    annotated_result['annotated_biased_prompt'] = tagged_prompt
    
    # Note: biased_prompt (without annotations) is NOT added to output
    # Note: global_judge_success, global_judge_raw_response are NOT added
    
    return annotated_result


def process_batch(
    batch_results: list,
    client,
    model: str,
    max_retries: int,
    verbose: bool = True
) -> tuple:
    """
    Process a batch of biased results through the LLM judge.
    
    Args:
        batch_results: List of biased results to judge
        client: OpenRouter client
        model: Judge model name
        max_retries: Max retries per request
        verbose: Print progress
        
    Returns:
        Tuple of (judgments, annotated_results)
    """
    # Judge the batch
    judgments = judge_batch(
        results=batch_results,
        client=client,
        model=model,
        max_retries=max_retries,
        verbose=verbose
    )
    
    # Create annotated results
    annotated_results = [
        create_annotated_result(result, judgment)
        for result, judgment in zip(batch_results, judgments)
    ]
    
    return judgments, annotated_results


# =============================================================================
# STRATIFIED SPLITTING
# =============================================================================

def group_by_faithfulness_and_hint(
    records: list,
    faithful_label: str = "faithful",
    unfaithful_label: str = "unfaithful",
    classification_field: str = "faithfulness_classification",
    hint_template_field: str = "hint_template"
) -> dict:
    """
    Group records by (faithfulness_classification, hint_template) for stratified sampling.
    """
    groups = {}
    
    for record in records:
        classification = record.get(classification_field, "")
        hint_template = record.get(hint_template_field, "unknown")
        
        # Categorize faithfulness using exact matching
        if classification == faithful_label:
            faith_category = "faithful"
        elif classification == unfaithful_label:
            faith_category = "unfaithful"
        else:
            # Skip records that don't match standard labels (e.g. errors)
            continue
            
        # Create group key as (faithfulness, hint_template)
        group_key = (faith_category, hint_template)
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(record)
        
    return groups


def stratified_split(
    groups: dict,
    train_ratio: float,
    random_seed: int = 42
) -> tuple:
    """
    Perform stratified sampling to create train/val splits.
    """
    random.seed(random_seed)
    
    train_records = []
    val_records = []
    
    print(f"\nPerforming stratified split ({train_ratio:.0%} train)...")
    
    # Process each group independently
    for group_key, group_records in groups.items():
        if not group_records:
            continue
            
        # Shuffle this group
        shuffled = group_records.copy()
        random.shuffle(shuffled)
        
        # Calculate split point
        num_records = len(shuffled)
        train_count = int(num_records * train_ratio)
        
        # Split
        group_train = shuffled[:train_count]
        group_val = shuffled[train_count:]
        
        # Add to overall splits
        train_records.extend(group_train)
        val_records.extend(group_val)
        
    return train_records, val_records


def assign_split_labels(
    train_records: list,
    val_records: list,
    split_field: str = "split"
) -> list:
    """Assign split labels to records and combine."""
    for record in train_records:
        record[split_field] = "train"
        
    for record in val_records:
        record[split_field] = "val"
        
    return train_records + val_records


def print_split_statistics(records: list, split_field: str = "split", classification_field: str = "faithfulness_classification"):
    """Print statistics about the splits."""
    train_count = sum(1 for r in records if r.get(split_field) == "train")
    val_count = sum(1 for r in records if r.get(split_field) == "val")
    total = len(records)
    
    if total == 0:
        return

    print(f"\nSplit Statistics:")
    print(f"  Train: {train_count} ({train_count/total*100:.1f}%)")
    print(f"  Val:   {val_count} ({val_count/total*100:.1f}%)")



# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Global faithfulness evaluation with LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process hinted results for a model (auto-finds most recent file)
  python eval_faithfulness.py --model Qwen3-8B --dataset-type hinted
  
  # Process specific date
  python eval_faithfulness.py --model Qwen3-32B --dataset-type hinted --date 2025-12-28
  
  # Resume from checkpoint
  python eval_faithfulness.py --model Qwen3-8B --dataset-type hinted --resume
  
  # Use smaller batch size for stability
  python eval_faithfulness.py --model Qwen3-8B --dataset-type hinted --batch-size 10
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Model short name (e.g., 'Qwen3-8B', 'Qwen3-32B'). Used to locate files."
    )
    
    parser.add_argument(
        '--dataset-type',
        type=str,
        required=True,
        choices=["hinted", "hinted_sampled"],
        help="Type of dataset to process"
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help="Date of the dataset files (YYYY-MM-DD). If not specified, finds the most recent."
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_CONFIG['data_dir'],
        help="Base data directory (default: data/definitive_pipeline_data)"
    )
    
    parser.add_argument(
        '--judge-model',
        type=str,
        default=DEFAULT_CONFIG['judge_model'],
        help='Judge model to use (default: google/gemini-2.5-flash)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_CONFIG['batch_size'],
        help='Batch size for processing and checkpointing (default: 20)'
    )
    
    parser.add_argument(
        '--max-retries', '-r',
        type=int,
        default=DEFAULT_CONFIG['max_retries'],
        help='Max retries per API call (default: 3)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )
    
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Disable checkpointing (not recommended for large datasets)'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=DEFAULT_CONFIG['checkpoint_dir'],
        help='Directory for checkpoint files'
    )
    
    parser.add_argument(
        action='store_true',
        help='Skip plot generation'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=DEFAULT_CONFIG['train_ratio'],
        help='Ratio of data to use for training (default: 0.7)'
    )

    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Disable stratified train/val splitting'
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for global faithfulness evaluation."""
    args = parse_args()
    
    print(f"=== GLOBAL FAITHFULNESS EVALUATION - {TODAY} ===")
    print(f"Model: {args.model}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Batch Size: {args.batch_size}")
    
    # Resolve input file paths
    try:
        input_jsonl, input_summary = resolve_file_paths(
            model=args.model,
            dataset_type=args.dataset_type,
            data_dir=args.data_dir,
            date=args.date
        )
        print(f"\n✓ Found input file: {input_jsonl}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    
    # Generate output paths
    output_date = args.date or TODAY
    output_jsonl, output_summary = get_output_paths(
        model=args.model,
        data_dir=args.data_dir,
        date=output_date
    )
    print(f"✓ Output will be saved to: {output_jsonl}")
    
    # Load hinted results
    print(f"\nLoading hinted evaluation results...")
    hinted_results = load_jsonl(input_jsonl)
    print(f"✓ Loaded {len(hinted_results)} hinted evaluation results")
    
    # Filter for biased results only
    biased_results = [r for r in hinted_results if r.get('bias_label') == 'biased']
    print(f"✓ Found {len(biased_results)} biased results to judge")
    
    if len(biased_results) == 0:
        print("\nNo biased results to judge - all models resisted the hints!")
        print("Faithfulness evaluation complete (no work needed).")
        return 0
    
    # Setup OpenRouter client
    print(f"\nSetting up OpenRouter client...")
    try:
        client = setup_openrouter_client()
        print(f"✓ OpenRouter client initialized")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure OPENROUTER_API_KEY environment variable is set.")
        return 1
    
    # Check for checkpoint
    checkpoint_path = get_checkpoint_path(input_jsonl, args.checkpoint_dir)
    all_judgments = []
    all_annotated = []
    start_idx = 0
    
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            print(f"\n✓ Found checkpoint from {checkpoint['timestamp']}")
            print(f"  Resuming from record {checkpoint['total_processed']}/{len(biased_results)}")
            all_judgments = checkpoint['judgments']
            all_annotated = checkpoint['annotated_results']
            start_idx = checkpoint['total_processed']
        else:
            print(f"\nNo checkpoint found, starting fresh")
    
    # Process in batches
    remaining = biased_results[start_idx:]
    num_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {len(remaining)} RECORDS IN {num_batches} BATCHES")
    print(f"{'='*60}")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]
        
        global_idx = start_idx + batch_start
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} (records {global_idx + 1}-{global_idx + len(batch)}) ---")
        
        # Process batch
        try:
            batch_judgments, batch_annotated = process_batch(
                batch_results=batch,
                client=client,
                model=args.judge_model,
                max_retries=args.max_retries,
                verbose=True
            )
            
            all_judgments.extend(batch_judgments)
            all_annotated.extend(batch_annotated)
            
            # Save checkpoint after each batch
            if not args.no_checkpoint:
                processed_indices = list(range(start_idx + batch_end))
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    processed_indices=processed_indices,
                    judgments=all_judgments,
                    annotated_results=all_annotated,
                    batch_idx=batch_idx
                )
                print(f"  ✓ Checkpoint saved ({start_idx + batch_end}/{len(biased_results)} processed)")
                
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted! Progress saved to checkpoint.")
            print(f"  Resume with: python eval_faithfulness.py --model {args.model} --dataset-type {args.dataset_type} --resume")
            return 1
        except Exception as e:
            print(f"\n⚠ Error processing batch: {e}")
            print(f"  Progress saved. Resume with --resume flag.")
            raise
    
    # Fix malformed JSON responses
    errors_fixed = fix_error_judgments(all_annotated, all_judgments, verbose=True)
    
    # Compute metrics
    faithfulness_metrics = compute_global_faithfulness_metrics(all_judgments)
    print_global_faithfulness_report(faithfulness_metrics)

    # Perform stratified split if requested
    split_stats = {}
    if not args.no_split:
        try:
            # Group by faithfulness and hint template
            groups = group_by_faithfulness_and_hint(all_annotated)
            
            # Perform split
            train_records, val_records = stratified_split(
                groups, 
                train_ratio=args.train_ratio
            )
            
            # Assign labels and recombine
            all_annotated = assign_split_labels(train_records, val_records)
            
            # Print stats
            print_split_statistics(all_annotated)
            
            split_stats = {
                'train_count': len(train_records),
                'val_count': len(val_records),
                'train_ratio': args.train_ratio
            }
            
        except Exception as e:
            print(f"\n⚠ Error performing stratified split: {e}")
            print("Saving without split info.")
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save annotated results
    os.makedirs(os.path.dirname(output_jsonl) or '.', exist_ok=True)
    save_jsonl(all_annotated, output_jsonl)
    print(f"✓ Saved {len(all_annotated)} annotated results to: {output_jsonl}")
    
    # Save summary
    summary = {
        'evaluation_date': TODAY,
        'method': 'global_llm_judge',
        'judge_model': args.judge_model,
        'source_file': input_jsonl,
        'total_hinted_results': len(hinted_results),
        'total_biased_results': len(biased_results),
        'batch_size': args.batch_size,
        'errors_fixed': errors_fixed,
        'faithfulness_metrics': faithfulness_metrics,
        'split_stats': split_stats,
        'annotated_output_file': output_jsonl
    }
    
    os.makedirs(os.path.dirname(output_summary) or '.', exist_ok=True)
    with open(output_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved summary to: {output_summary}")
    
    # Clear checkpoint on success
    if not args.no_checkpoint:
        clear_checkpoint(checkpoint_path)
    
    # Optional plots
    if not args.no_plots:
        try:
            from src.plots import plot_global_faithfulness_by_bias
            
            plot_dir = os.path.dirname(output_jsonl)
            os.makedirs(os.path.join(plot_dir, 'plots'), exist_ok=True)
            
            input_basename = Path(input_jsonl).stem
            plot_path = os.path.join(plot_dir, 'plots', f'faithfulness_by_hint_{input_basename}.png')
            
            plot_global_faithfulness_by_bias(
                hinted_results=all_annotated,
                save_path=plot_path,
                show_plot=False
            )
            print(f"✓ Saved plot to: {plot_path}")
        except Exception as e:
            print(f"⚠ Could not create plots: {e}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())

