#!/usr/bin/env python3
"""
extract_activations_runpod.py

RunPod-friendly version of activation extraction script.
Runs as a standalone Python script (no Colab/GitHub integration).

Step 3 of faithfulness steering workflow: Extract hidden state activations from annotated biased prompts.
Extracts activations at the level of periods before closing tags in annotated prompts.
For every prompt, stores activations from every label from any layer inside separate .pt files.
"""

import sys
import os
# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import sys
import time
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path

# Import reusable modules
from src.activations import (
    extract_activations_from_annotated_prompts,
    get_activation_statistics,
    print_activation_statistics,
    build_activation_dataset,
    save_activation_dataset,
    print_dataset_summary
)
from src.config import TODAY, ACTIVATIONS_DIR, DATASETS_DIR, ActivationConfig, ModelConfig






# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract activations from annotated prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python extract_activations_runpod.py --model Qwen3-32B --dataset-type faithfulness_annotated
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Model short name (e.g., 'Qwen3-32B', 'DeepSeek-Llama3-8B')"
    )
    
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="faithfulness_annotated",
        help="Dataset type prefix to search for (default: 'faithfulness_annotated')"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["on-policy", "off-policy"],
        default="on-policy",
        help="Extraction mode: 'on-policy' (tag-based) or 'off-policy' (last-token)"
    )
    
    parser.add_argument(
        "--off-policy-file",
        type=str,
        default=None,
        help="Path to off-policy responses file (required if mode is 'off-policy')"
    )
    
    return parser.parse_args()


def find_most_recent_file(model_name: str, dataset_type: str, base_dir: str = "data") -> str:
    """
    Find the most recent JSONL file matching pattern: {dataset_type}*{model_name}*.jsonl
    Searches recursively in base_dir.
    
    Args:
        model_name: Short model name (e.g., 'Qwen3-32B')
        dataset_type: Dataset type prefix (e.g., 'faithfulness_annotated')
        base_dir: Base directory to search in
    
    Returns:
        Absolute path to the most recent matching file
    """
    # Search pattern: dataset_type*model_name*.jsonl (anywhere in filename)
    pattern = os.path.join(base_dir, "**", f"*{dataset_type}*{model_name}*.jsonl")
    matching_files = glob.glob(pattern, recursive=True)
    
    if not matching_files:
        raise FileNotFoundError(
            f"No files found matching pattern: {pattern}\n"
            f"Looking for: dataset_type='{dataset_type}', model='{model_name}'"
        )
    
    # Sort by modification time (most recent first)
    matching_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(matching_files)} matching file(s):")
    for f in matching_files[:5]:  # Show top 5
        mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
        print(f"  {mtime}: {f}")
    
    return os.path.abspath(matching_files[0])





def find_off_policy_file(base_dir: str = "data") -> str:
    """
    Find the off-policy responses file (fixed filename).
    Searches recursively for 'off_policy_responses_filtered.jsonl'.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        Absolute path to the file
    """
    filename = "off_policy_responses_filtered.jsonl"
    pattern = os.path.join(base_dir, "**", filename)
    matching_files = glob.glob(pattern, recursive=True)
    
    if not matching_files:
        raise FileNotFoundError(
            f"Off-policy file not found: {filename}\n"
            f"Searched in: {base_dir}"
        )
    
    # If multiple found, use the most recent
    matching_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found off-policy file: {matching_files[0]}")
    return os.path.abspath(matching_files[0])


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function for activation extraction workflow."""
    
    # Parse CLI arguments
    args = parse_args()
    
    # Get full model ID from short name
    model_id = ModelConfig.get_model_id(args.model)
    
    print(f"\n=== ACTIVATION EXTRACTION SCRIPT ===")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model} -> {model_id}")
    
    # =========================================================================
    # OFF-POLICY MODE
    # =========================================================================
    if args.mode == "off-policy":
        from src.activations import (
            extract_off_policy_activations,
            build_off_policy_dataset,
            save_off_policy_dataset
        )
        
        # Find off-policy input file
        if args.off_policy_file:
            print(f"\n=== USING PROVIDED OFF-POLICY FILE ===")
            input_file = os.path.abspath(args.off_policy_file)
            if not os.path.exists(input_file):
                print(f"ERROR: Off-policy file not found: {input_file}")
                sys.exit(1)
        else:
            print(f"\n=== DISCOVERING OFF-POLICY FILE ===")
            input_file = find_off_policy_file()
        
        # Find model's on-policy data directory for output location
        print(f"\n=== FINDING MODEL DATA DIRECTORY ===")
        print(f"Looking for on-policy data for model: {args.model}")
        try:
            on_policy_file = find_most_recent_file(args.model, args.dataset_type)
            model_data_dir = os.path.dirname(on_policy_file)
            print(f"Model data directory: {model_data_dir}")
        except FileNotFoundError:
            # Fallback: create directory next to off-policy file
            model_data_dir = os.path.dirname(input_file)
            print(f"No on-policy data found, using: {model_data_dir}")
        
        # Determine output paths (in model's data directory)
        output_dir = os.path.join(model_data_dir, f"off_policy_activations_{args.model}_{TODAY}")
        dataset_output_file = os.path.join(model_data_dir, f"off_policy_dataset_{args.model}_{TODAY}.pkl")
        summary_file = os.path.join(model_data_dir, f"off_policy_summary_{args.model}_{TODAY}.json")
        
        print(f"\n=== OFF-POLICY EXTRACTION CONFIG ===")
        print(f"Input File: {input_file}")
        print(f"Output Directory: {output_dir}")
        print(f"Dataset Output: {dataset_output_file}")
        print(f"Summary Output: {summary_file}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # STEP 1: Extract Activations
        print("\n=== STEP 1: Extract Last-Token Activations ===")
        start_time = time.time()
        
        model_info = extract_off_policy_activations(
            jsonl_filename=input_file,
            output_dir=output_dir,
            model_id=model_id,
            verbose=True
        )
        
        # STEP 2: Build Dataset
        print("\n=== STEP 2: Build Off-Policy Dataset ===")
        
        dataset = build_off_policy_dataset(
            activations_dir=output_dir,
            source_jsonl=input_file,
            num_layers=model_info['num_layers'],
            hidden_dim=model_info['hidden_dim']
        )
        
        save_off_policy_dataset(dataset, dataset_output_file)
        
        # STEP 3: Save Summary
        print("\n=== STEP 3: Save Extraction Summary ===")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        extraction_summary = {
            'date': TODAY,
            'mode': 'off-policy',
            'model_id': model_id,
            'model_short_name': args.model,
            'input_file': input_file,
            'output_dir': output_dir,
            'dataset_file': dataset_output_file,
            'num_layers': model_info['num_layers'],
            'hidden_dim': model_info['hidden_dim'],
            'num_prompts': model_info['num_prompts'],
            'processing_time_seconds': processing_time,
            'dataset_info': dataset['info']
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_summary, f, indent=2, ensure_ascii=False)
        print(f"Saved extraction summary to {summary_file}")
        
        # Final summary
        print(f"\n=== OFF-POLICY EXTRACTION COMPLETE ===")
        print(f"✅ Extracted last-token activations from {model_info['num_prompts']} prompts")
        print(f"✅ Saved to: {output_dir}")
        print(f"✅ Dataset: {dataset_output_file}")
        print(f"✅ Processing time: {processing_time/60:.1f} minutes")
        
        return
    
    # =========================================================================
    # ON-POLICY MODE (Default - existing logic)
    # =========================================================================
    
    # Auto-discover input file
    print(f"\n=== DISCOVERING INPUT FILE ===")
    print(f"Dataset Type: {args.dataset_type}")
    
    input_file = find_most_recent_file(args.model, args.dataset_type)
    print(f"\nSelected: {input_file}")
    
    # Determine output paths (co-located with input file)
    input_dir = os.path.dirname(input_file)
    input_basename = os.path.basename(input_file).replace('.jsonl', '')
    
    output_dir = os.path.join(input_dir, f"activations_{args.model}_{TODAY}")
    dataset_output_file = os.path.join(input_dir, f"activations_dataset_{args.model}_{TODAY}.pkl")
    summary_file = os.path.join(input_dir, f"extraction_summary_{args.model}_{TODAY}.json")
    
    # Get configuration defaults (non-path settings)
    config = ActivationConfig.configure_extraction(model_id)
    
    print(f"\n=== ACTIVATION EXTRACTION (RunPod) ===")
    print(f"Model: {model_id}")
    print(f"Input File: {input_file}")
    print(f"Input Prompt Field: {config['input_prompt_field']}")
    print(f"Annotated Response Field: {config['prompt_field']}")
    print(f"Output Directory: {output_dir}")
    print(f"Dataset Output: {dataset_output_file}")
    print(f"Summary Output: {summary_file}")
    print(f"Target Tags: {config['target_tags']}")
    print(f"Layers: Will be inferred from model at runtime")
    print(f"Hidden Dimension: Will be inferred from model at runtime")
    
    # Verify input file exists (should always pass since we just discovered it)
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================================================
    # STEP 1: Extract Activations
    # ==========================================================================
    print("\n=== STEP 1: Extract Activations ===")
    start_time = time.time()
    
    model_info = extract_activations_from_annotated_prompts(
        jsonl_filename=input_file,
        prompt_field=config['prompt_field'],
        output_dir=output_dir,
        model_id=model_id,
        target_tags=config['target_tags'],
        layers_to_extract=None,  # Let the function infer from model
        input_prompt_field=config['input_prompt_field'],  # Prepend input prompt to response
        verbose=config['verbose']
    )
    
    # ==========================================================================
    # STEP 2: Compute and Display Statistics
    # ==========================================================================
    print("\n=== STEP 2: Compute and Display Statistics ===")
    
    stats = get_activation_statistics(output_dir)
    print_activation_statistics(stats)
    
    # ==========================================================================
    # STEP 3: Build Activation Dataset
    # ==========================================================================
    print("\n=== STEP 3: Build Activation Dataset ===")
    
    # Use model_info from extraction for num_layers and hidden_dim
    dataset = build_activation_dataset(
        activations_dir=output_dir,
        source_jsonl=input_file,
        target_tags=config['target_tags'],
        num_layers=model_info['num_layers'],
        hidden_dim=model_info['hidden_dim'],
    )
    
    save_activation_dataset(dataset, dataset_output_file)
    print_dataset_summary(dataset)
    
    # ==========================================================================
    # STEP 4: Save Extraction Summary
    # ==========================================================================
    print("\n=== STEP 4: Save Extraction Summary ===")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    extraction_summary = {
        'date': TODAY,
        'mode': 'on-policy',
        'model_id': model_id,
        'model_short_name': args.model,
        'input_file': input_file,
        'output_dir': output_dir,
        'dataset_file': dataset_output_file,
        'prompt_field': config['prompt_field'],
        'input_prompt_field': config['input_prompt_field'],
        'target_tags': config['target_tags'],
        'layers_extracted': model_info['num_layers'],
        'hidden_dim': model_info['hidden_dim'],
        'processing_time_seconds': processing_time,
        'activation_statistics': stats,
        'dataset_info': dataset['info']
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(extraction_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved extraction summary to {summary_file}")
    
    # ==========================================================================
    # STEP 5: Summary and Completion
    # ==========================================================================
    print("\n=== STEP 5: Summary and Completion ===")
    
    print(f"\n=== ACTIVATION EXTRACTION COMPLETE ===")
    print(f"✅ All workflow requirements fulfilled:")
    print(f"   ✅ Loaded annotated biased prompts from {input_file}")
    print(f"   ✅ Extracted activations at periods before closing tags")
    print(f"   ✅ Processed {stats['total_prompts']} prompts")
    print(f"   ✅ Extracted from {len(stats['layers_found'])} layers")
    print(f"   ✅ Found tags: {stats['tags_found']}")
    print(f"   ✅ Stored activations in individual .pt files: {output_dir}")
    print(f"   ✅ Maintained prompt-wise hierarchy: prompt → layer → label → activations")
    print(f"   ✅ Built aggregated activation dataset: {dataset_output_file}")
    print(f"   ✅ Dataset structure: layer → label → tensor([total_activations, hidden_dim])")
    
    print(f"\nProcessing time: {processing_time/60:.1f} minutes")
    
    # ==========================================================================
    # STEP 6: Verify Results
    # ==========================================================================
    print("\n=== STEP 6: Verify Results ===")
    
    if os.path.exists(dataset_output_file):
        print(f"Dataset file saved: {dataset_output_file}")
        print(f"  Size: {os.path.getsize(dataset_output_file) / (1024**2):.2f} MB")
    else:
        print(f"Warning: Dataset file not found at {dataset_output_file}")
    
    if os.path.exists(summary_file):
        print(f"Summary file saved: {summary_file}")
        print(f"  Size: {os.path.getsize(summary_file) / 1024:.2f} KB")
    else:
        print(f"Warning: Summary file not found at {summary_file}")
    
    if os.path.exists(output_dir):
        num_files = len([f for f in os.listdir(output_dir) if f.endswith('.pt')])
        print(f"Activation directory: {output_dir}")
        print(f"  Number of .pt files: {num_files}")
    else:
        print(f"Warning: Activation directory not found at {output_dir}")
    
    print(f"\n=== EXPERIMENT COMPLETE ===")
    print(f"Results are saved locally:")
    print(f"  - Activations: {output_dir}")
    print(f"  - Dataset: {dataset_output_file}")
    print(f"  - Summary: {summary_file}")
    print(f"\nNext steps:")
    print(f"  1. Sync files to local machine if needed (scp/rsync)")
    print(f"  2. Commit results to GitHub")
    print(f"  3. Run separability analysis on the activation dataset")


if __name__ == "__main__":
    main()

