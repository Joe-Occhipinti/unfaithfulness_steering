"""
eval_hinted_local_faithfulness.py

Local faithfulness annotation script for marking faithful and unfaithful reasoning spans.

This script:
1. Loads faithfulness-annotated JSONL (from eval_faithfulness.py output)
2. Routes each record to appropriate local annotator (faithful vs unfaithful) based on global classification
3. Annotates with [F_body]/[U_body] markers for local activation extraction
4. Overwrites the input file with locally annotated results
5. Updates summary with local annotation metrics

Features:
- Auto-discovers input files based on --model
- Batch processing with configurable batch size
- Checkpointing to save progress periodically
- Resume capability from checkpoint if interrupted
- Robust retry mechanism with exponential backoff
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
from src.local_faithfulness import (
    setup_openrouter_client,
    annotate_local_faithfulness,
    compute_local_annotation_metrics
)
from src.config import TODAY, ModelConfig

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'data_dir': "data/definitive_pipeline_data",
    'judge_model': "google/gemini-2.5-flash",
    'max_retries': 3,
    'batch_size': 20,
    'checkpoint_dir': "checkpoints",
}

# =============================================================================
# FILE PATH RESOLUTION
# =============================================================================

def resolve_file_paths(
    model: str,
    data_dir: str,
    date: Optional[str] = None
) -> Tuple[str, str]:
    """Resolve input file paths based on model.
    
    Looks for faithfulness_annotated_{model}_{date}.jsonl output from eval_faithfulness.py.
    
    Args:
        model: Model short name (e.g., 'Qwen3-8B')
        data_dir: Base data directory
        date: Optional specific date (YYYY-MM-DD). If None, finds most recent.
        
    Returns:
        Tuple of (jsonl_path, summary_path)
        
    Raises:
        FileNotFoundError: If no matching files found
    """
    # Build model directory path
    model_dir = Path(data_dir) / model
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if date:
        # Use specific date
        jsonl_path = model_dir / f"faithfulness_annotated_{model}_{date}.jsonl"
        summary_path = model_dir / f"faithfulness_summary_{model}_{date}.json"
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Faithfulness annotated file not found: {jsonl_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
    else:
        # Find most recent file matching pattern
        pattern = str(model_dir / f"faithfulness_annotated_{model}_*.jsonl")
        matching_files = glob(pattern)
        
        if not matching_files:
            raise FileNotFoundError(
                f"No faithfulness annotated file found for model '{model}' in {model_dir}\n"
                f"Expected pattern: faithfulness_annotated_{model}_YYYY-MM-DD.jsonl\n"
                f"Please run eval_faithfulness.py first."
            )
        
        # Get most recent (sorted by filename which includes date)
        jsonl_path = Path(sorted(matching_files)[-1])
        
        # Derive summary path from jsonl path
        summary_filename = jsonl_path.name.replace('_annotated_', '_summary_').replace('.jsonl', '.json')
        summary_path = jsonl_path.parent / summary_filename
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    return str(jsonl_path), str(summary_path)


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def get_checkpoint_path(input_file: str, checkpoint_dir: str) -> str:
    """Generate checkpoint file path based on input file name."""
    input_basename = Path(input_file).stem
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"checkpoint_local_{input_basename}.json")


def save_checkpoint(checkpoint_path: str, processed_indices: list, annotated_results: list, batch_idx: int):
    """Save processing checkpoint for resume capability."""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'processed_indices': processed_indices,
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
# LOCAL ANNOTATION PROCESSING
# =============================================================================

def extract_biased_prompt(result: dict) -> str:
    """Extract biased prompt from result, stripping any existing tags."""
    # Priority: biased_prompt > annotated_biased_prompt (stripped) > biased_input_prompt
    biased_prompt = result.get('biased_prompt', '')
    if not biased_prompt:
        annotated_prompt = result.get('annotated_biased_prompt', '')
        if annotated_prompt:
            biased_prompt = re.sub(r'\[/?[FU]_final\]', '', annotated_prompt)
        else:
            biased_prompt = result.get('biased_input_prompt', '')
    return biased_prompt


def process_single_record(
    result: dict,
    client,
    model: str,
    max_retries: int
) -> Tuple[dict, bool]:
    """Process a single record for local annotation.
    
    Returns:
        Tuple of (updated_result, success)
    """
    updated_result = result.copy()
    
    # Get global classification
    global_classification = result.get('faithfulness_classification')
    
    # Skip if error or missing classification
    if not global_classification or global_classification == 'error':
        return updated_result, False
    
    is_faithful = (global_classification == 'faithful')
    hint_template = result.get('hint_template', 'professor')
    hint_letter = result.get('hint_letter', '')
    biased_prompt = extract_biased_prompt(result)
    
    if not biased_prompt:
        return updated_result, False
    
    # Annotate locally
    annotation_result = annotate_local_faithfulness(
        biased_prompt=biased_prompt,
        hint_letter=hint_letter,
        is_faithful=is_faithful,
        client=client,
        hint_template=hint_template,
        model=model,
        max_retries=max_retries,
        verbose=False
    )
    
    if annotation_result['success']:
        # Add local annotation (only the annotated text, no debug fields)
        updated_result['local_annotated_biased_prompt'] = annotation_result['annotated_text']
        return updated_result, True
    else:
        return updated_result, False


def process_batch(
    batch_results: list,
    client,
    model: str,
    max_retries: int,
    verbose: bool = True
) -> Tuple[list, int, int]:
    """Process a batch of records for local annotation.
    
    Returns:
        Tuple of (annotated_results, success_count, error_count)
    """
    min_delay = ModelConfig.get_min_delay(model)
    annotated_results = []
    success_count = 0
    error_count = 0
    
    for i, result in enumerate(batch_results):
        # Rate limiting
        if i > 0:
            time.sleep(min_delay)
        
        updated_result, success = process_single_record(
            result=result,
            client=client,
            model=model,
            max_retries=max_retries
        )
        
        annotated_results.append(updated_result)
        if success:
            success_count += 1
        else:
            error_count += 1
    
    if verbose:
        print(f"  Batch complete: {success_count} successful, {error_count} errors/skipped")
    
    return annotated_results, success_count, error_count


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Local faithfulness annotation with [F_body]/[U_body] markers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process faithfulness-annotated results for a model (auto-finds most recent file)
  python annotate_faithfulness.py --model Qwen3-8B
  
  # Process specific date
  python annotate_faithfulness.py --model Qwen3-32B --date 2025-12-28
  
  # Resume from checkpoint
  python annotate_faithfulness.py --model Qwen3-8B --resume
  
  # Use smaller batch size
  python annotate_faithfulness.py --model Qwen3-8B --batch-size 10
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Model short name (e.g., 'Qwen3-8B', 'Qwen3-32B'). Used to locate files."
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
        help='Annotation model to use (default: google/gemini-2.5-flash)'
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
    
    return parser.parse_args()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for local faithfulness annotation."""
    args = parse_args()
    
    print(f"=== LOCAL FAITHFULNESS ANNOTATION - {TODAY} ===")
    print(f"Model: {args.model}")
    print(f"Annotation Model: {args.judge_model}")
    print(f"Batch Size: {args.batch_size}")
    
    # Resolve input file paths (looks for eval_faithfulness.py output)
    try:
        input_jsonl, input_summary = resolve_file_paths(
            model=args.model,
            data_dir=args.data_dir,
            date=args.date
        )
        print(f"\n✓ Found input file: {input_jsonl}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    
    # Load existing annotated data (from eval_faithfulness.py)
    print(f"\nLoading faithfulness-annotated results...")
    annotated_data = load_jsonl(input_jsonl)
    print(f"✓ Loaded {len(annotated_data)} records")
    
    # Count global classifications
    faithful_count = sum(1 for r in annotated_data if r.get('faithfulness_classification') == 'faithful')
    unfaithful_count = sum(1 for r in annotated_data if r.get('faithfulness_classification') == 'unfaithful')
    error_count = sum(1 for r in annotated_data if r.get('faithfulness_classification') == 'error')
    
    print(f"\nGlobal classifications:")
    print(f"  Faithful: {faithful_count}")
    print(f"  Unfaithful: {unfaithful_count}")
    print(f"  Error: {error_count} (will be skipped)")
    
    records_to_process = [r for r in annotated_data if r.get('faithfulness_classification') in ['faithful', 'unfaithful']]
    
    if len(records_to_process) == 0:
        print("\nNo faithful or unfaithful records to annotate!")
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
    all_annotated = []
    start_idx = 0
    
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            print(f"\n✓ Found checkpoint from {checkpoint['timestamp']}")
            print(f"  Resuming from record {checkpoint['total_processed']}/{len(records_to_process)}")
            all_annotated = checkpoint['annotated_results']
            start_idx = checkpoint['total_processed']
        else:
            print(f"\nNo checkpoint found, starting fresh")
    
    # Process in batches
    remaining = records_to_process[start_idx:]
    num_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {len(remaining)} RECORDS IN {num_batches} BATCHES")
    print(f"{'='*60}")
    
    total_success = 0
    total_errors = 0
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]
        
        global_idx = start_idx + batch_start
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} (records {global_idx + 1}-{global_idx + len(batch)}) ---")
        
        try:
            batch_annotated, batch_success, batch_errors = process_batch(
                batch_results=batch,
                client=client,
                model=args.judge_model,
                max_retries=args.max_retries,
                verbose=True
            )
            
            all_annotated.extend(batch_annotated)
            total_success += batch_success
            total_errors += batch_errors
            
            # Save checkpoint after each batch
            if not args.no_checkpoint:
                processed_indices = list(range(start_idx + batch_end))
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    processed_indices=processed_indices,
                    annotated_results=all_annotated,
                    batch_idx=batch_idx
                )
                print(f"  ✓ Checkpoint saved ({start_idx + batch_end}/{len(records_to_process)} processed)")
                
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted! Progress saved to checkpoint.")
            print(f"  Resume with: python annotate_faithfulness.py --model {args.model} --resume")
            return 1
        except Exception as e:
            print(f"\n⚠ Error processing batch: {e}")
            print(f"  Progress saved. Resume with --resume flag.")
            raise
    
    # Add back the error records (they weren't processed but should be in output)
    error_records = [r for r in annotated_data if r.get('faithfulness_classification') == 'error']
    final_results = all_annotated + error_records
    
    # Compute local annotation metrics
    local_metrics = {
        'total_processed': len(records_to_process),
        'successful': total_success,
        'errors': total_errors,
        'skipped': error_count,
        'success_rate': total_success / len(records_to_process) if len(records_to_process) > 0 else 0
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("LOCAL ANNOTATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {local_metrics['total_processed']}")
    print(f"Successful: {local_metrics['successful']} ({local_metrics['success_rate']:.1%})")
    print(f"Errors: {local_metrics['errors']}")
    print(f"Skipped (global error): {local_metrics['skipped']}")
    
    # Save results (overwrite input file)
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    save_jsonl(final_results, input_jsonl)
    print(f"✓ Saved {len(final_results)} locally annotated results to: {input_jsonl}")
    
    # Load and update summary
    with open(input_summary, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # Add local annotation metrics to summary
    summary['local_annotation'] = {
        'annotation_date': TODAY,
        'annotation_model': args.judge_model,
        'batch_size': args.batch_size,
        'local_annotation_metrics': local_metrics
    }
    
    with open(input_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Updated summary: {input_summary}")
    
    # Clear checkpoint on success
    if not args.no_checkpoint:
        clear_checkpoint(checkpoint_path)
    
    print(f"\n{'='*60}")
    print("LOCAL FAITHFULNESS ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  + Use locally annotated data for activation extraction at [F_body]/[U_body] spans")
    print(f"  + Extract steering vectors from local activations")
    
    return 0


if __name__ == "__main__":
    exit(main())
