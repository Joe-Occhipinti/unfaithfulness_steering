import time
import json
import os
import argparse
import glob
import torch
import pickle
from datetime import datetime
from typing import Optional, Dict, Any

# Import reusable modules
from src.steering import (
    load_activation_dataset,
    compute_steering_vectors_by_layer,
    save_steering_vectors,
    print_steering_summary,
    save_steering_summary_json
)
from src.separability import split_dataset_by_prompts
from src.config import TODAY

# =============================================================================
# DOMAIN MAPPING CONFIGURATION
# =============================================================================

# Domain mapping: subject -> domain
DOMAIN_MAPPING = {
    # Science (Biology + Chemistry)
    'college_biology': 'science',
    'high_school_biology': 'science',
    'college_chemistry': 'science',
    'high_school_chemistry': 'science',

    # History (All history subjects)
    'high_school_european_history': 'history',
    'high_school_us_history': 'history',
    'high_school_world_history': 'history',
    'prehistory': 'history',

    # Psychology
    'high_school_psychology': 'psychology',
}

# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute steering vectors from activation datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model short name (e.g., 'Qwen3-32B'). Used for file discovery and output naming."
    )

    # Mode argument
    parser.add_argument(
        "--mode",
        type=str,
        choices=["on-policy", "off-policy"],
        default="on-policy",
        help="Computation mode: 'on-policy' (standard) or 'off-policy' (simple mean diff)"
    )

    # File discovery arguments
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,  # Default depends on mode
        help="Dataset type prefix to search for (default: 'activations' for on-policy, 'off_policy_dataset' for off-policy)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Manually specify input file path (overrides auto-discovery)"
    )

    # Steering parameters
    parser.add_argument(
        "--positive_tags",
        nargs="+",
        default=["F_body"],
        help="Tags to treat as positive (faithful) examples. For off-policy, used as label 'faithful'."
    )
    parser.add_argument(
        "--negative_tags",
        nargs="+",
        default=["U_body"],
        help="Tags to treat as negative (unfaithful) examples. For off-policy, used as label 'unfaithful'."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to use for computation ('train', 'val', 'test')"
    )

    # Filtering and Grouping
    parser.add_argument(
        "--domain_filter",
        type=str,
        default=None,
        help="Filter by specific domain (e.g., 'science')"
    )
    parser.add_argument(
        "--correct_hint",
        type=str,
        choices=["True", "False"],
        default=None,
        help="Filter by hint correctness ('True' or 'False')"
    )
    parser.add_argument(
        "--use_domain_grouping",
        action="store_true",
        default=False,
        help="Enable domain grouping for config weighting (default: Disabled)"
    )
    parser.add_argument(
        "--no_config_weighting",
        action="store_true",
        default=False,
        help="Disable config weighting (default: Enabled)"
    )

    return parser.parse_args()

def find_most_recent_activation_file(model_name: str, dataset_type: str, base_dir: str = "data") -> str:
    """
    Find the most recent activation dataset file matching pattern.
    """
    # Search pattern: *{dataset_type}*{model_name}*.pkl
    pattern = os.path.join(base_dir, "**", f"*{dataset_type}*{model_name}*.pkl")
    matching_files = glob.glob(pattern, recursive=True)

    if not matching_files:
        raise FileNotFoundError(
            f"No activation files found matching pattern: {pattern}\n"
            f"Looking for: dataset_type='{dataset_type}', model='{model_name}'"
        )

    # Sort by modification time (most recent first)
    matching_files.sort(key=os.path.getmtime, reverse=True)

    print(f"Found {len(matching_files)} matching file(s):")
    for f in matching_files[:3]:  # Show top 3
        mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
        print(f"  {mtime}: {f}")

    return os.path.abspath(matching_files[0])

# =============================================================================
# OFF-POLICY COMPUTATION LOGIC
# =============================================================================

def compute_off_policy_vectors(dataset: Dict[str, Any]) -> tuple:
    """
    Compute steering vectors for off-policy data.
    Logic: Vector = Mean(Faithful) - Mean(Unfaithful)
    Skips Layer 0 (Embeddings) and shifts indices.
    """
    print("Computing off-policy steering vectors...")
    
    # Organize activations by layer and label
    # Structure: {layer_idx: {'faithful': [], 'unfaithful': []}}
    layer_activations = {}
    
    # Statistics
    stats = {
        'total_items': len(dataset['data']),
        'faithful_count': 0,
        'unfaithful_count': 0,
        'skipped_count': 0
    }
    
    # Iterate through data
    data_items = dataset['data']
    if isinstance(data_items, dict):
        data_items = data_items.values()
        
    for item in data_items:
        metadata = item.get('metadata', {})
        label = metadata.get('label')
        layers = item.get('layers', {})
        
        if label not in ['faithful', 'unfaithful']:
            stats['skipped_count'] += 1
            continue
            
        if label == 'faithful':
            stats['faithful_count'] += 1
        else:
            stats['unfaithful_count'] += 1
            
        for layer_idx, activation in layers.items():
            if layer_idx not in layer_activations:
                layer_activations[layer_idx] = {'faithful': [], 'unfaithful': []}
            
            # Extract tensor if it's a dictionary (from extract_last_token_activations.py structure)
            if isinstance(activation, dict):
                if 'last_token' in activation:
                    activation = activation['last_token']
                else:
                    # Fallback: take the first value if 'last_token' missing
                    activation = list(activation.values())[0]

            # Ensure activation is a tensor and on CPU
            if not isinstance(activation, torch.Tensor):
                activation = torch.tensor(activation)
            activation = activation.cpu()
            
            # Flatten if necessary (should be [hidden_dim] or [1, hidden_dim])
            if activation.dim() > 1:
                activation = activation.squeeze()
                
            layer_activations[layer_idx][label].append(activation)
            
    # Compute vectors for each layer
    steering_vectors = {}
    layer_stats = {}
    
    sorted_layers = sorted(layer_activations.keys())
    print(f"Computing vectors for {len(sorted_layers)} layers...")
    
    for layer_idx in sorted_layers:
        faithful_list = layer_activations[layer_idx]['faithful']
        unfaithful_list = layer_activations[layer_idx]['unfaithful']
        
        if not faithful_list or not unfaithful_list:
            print(f"Warning: Layer {layer_idx} missing data (F: {len(faithful_list)}, U: {len(unfaithful_list)})")
            continue
            
        # Stack and mean
        faithful_tensor = torch.stack(faithful_list)
        unfaithful_tensor = torch.stack(unfaithful_list)
        
        faithful_mean = faithful_tensor.mean(dim=0)
        unfaithful_mean = unfaithful_tensor.mean(dim=0)
        
        # Compute mean difference: Faithful - Unfaithful
        vector = faithful_mean - unfaithful_mean
        
        # FIX: Discard Layer 0 (Embeddings) and re-index 1-32 -> 0-31
        if layer_idx == 0:
            # print(f"  Skipping Layer 0 (Embeddings) to align with hidden layers 1-32")
            continue
            
        # Re-index: Layer 1 becomes Layer 0, etc.
        new_layer_idx = layer_idx - 1
        steering_vectors[new_layer_idx] = vector
        
        # Stats
        layer_stats[new_layer_idx] = {
            'faithful_samples': len(faithful_list),
            'unfaithful_samples': len(unfaithful_list),
            'faithful_norm': faithful_mean.norm().item(),
            'unfaithful_norm': unfaithful_mean.norm().item(),
            'vector_norm': vector.norm().item()
        }
        
    return steering_vectors, stats, layer_stats

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    args = parse_args()

    # Determine default dataset type based on mode
    if args.dataset_type is None:
        if args.mode == "off-policy":
            dataset_type = "off_policy_dataset"
        else:
            dataset_type = "activations"
    else:
        dataset_type = args.dataset_type

    # 1. Determine Input File
    if args.input_file:
        input_file = args.input_file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Specified input file not found: {input_file}")
    else:
        print(f"Auto-discovering input file for model '{args.model_name}' (Mode: {args.mode})...")
        input_file = find_most_recent_activation_file(args.model_name, dataset_type)

    print(f"Selected Input File: {input_file}")

    # 2. Determine Output Paths
    # Output directory: data/{model_name}/
    output_dir = os.path.join("data", args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"vectors_{args.model_name}.pkl")
    summary_file = os.path.join(output_dir, f"summary_vectors_{args.model_name}.json")

    start_time = time.time()

    # =========================================================================
    # OFF-POLICY EXECUTION BRANCH
    # =========================================================================
    if args.mode == "off-policy":
        print(f"\n=== OFF-POLICY STEERING VECTOR COMPUTATION ===")
        print(f"Model: {args.model_name}")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        # Load Dataset
        print("\n=== STEP 1: Loading Dataset ===")
        with open(input_file, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded dataset with {len(dataset['data'])} items.")
        
        # Compute Vectors
        print("\n=== STEP 2: Computing Vectors ===")
        steering_vectors, stats, layer_stats = compute_off_policy_vectors(dataset)
        
        # Save Results
        print("\n=== STEP 3: Saving Results ===")
        
        # Match the format of src.steering.save_steering_vectors
        output_data = {
            "steering_vectors": steering_vectors,
            "computation_stats": {
                "positive_tags": ["faithful"],
                "negative_tags": ["unfaithful"],
                "layers_computed": list(steering_vectors.keys()),
                "layer_stats": layer_stats,
                "global_stats": stats
            },
            "metadata": {
                "description": "Off-policy steering vectors (Faithful - Unfaithful)",
                "positive_tags": ["faithful"],
                "negative_tags": ["unfaithful"],
                "layers": list(steering_vectors.keys()),
                "vector_dim": list(steering_vectors.values())[0].shape[0] if steering_vectors else 0,
                "computation_date": datetime.now().isoformat(),
                "model_name": args.model_name,
                "mode": "off-policy"
            }
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
            
        # Save JSON Summary
        summary_data = {
            'computation_date': TODAY,
            'model_name': args.model_name,
            'mode': 'off-policy',
            'dataset_file': input_file,
            'results_summary': {
                'vectors_computed': len(steering_vectors),
                'total_items': stats['total_items'],
                'faithful_count': stats['faithful_count'],
                'unfaithful_count': stats['unfaithful_count'],
                'processing_time_seconds': time.time() - start_time
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Results saved to: {output_dir}")
        return

    # =========================================================================
    # ON-POLICY EXECUTION BRANCH (Standard)
    # =========================================================================
    
    # 3. Configure Parameters
    use_config_weighting = not args.no_config_weighting
    
    # Define config fields based on domain grouping setting
    if args.use_domain_grouping:
        config_fields = ['hint_template', 'correct_hint', 'domain']
    else:
        config_fields = ['hint_template', 'correct_hint']

    # Split configuration (fixed)
    TRAIN_RATIO = 1.0
    VAL_RATIO = 0.0
    RANDOM_SEED = 42

    print(f"\n=== STEERING VECTOR COMPUTATION ===")
    print(f"Model: {args.model_name}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Positive tags: {args.positive_tags}")
    print(f"Negative tags: {args.negative_tags}")
    print(f"Split: {args.split}")
    print(f"Config weighting: {'ENABLED' if use_config_weighting else 'DISABLED'}")
    print(f"Domain grouping: {'ENABLED' if args.use_domain_grouping else 'DISABLED'}")
    print(f"Config fields: {config_fields}")
    if args.domain_filter:
        print(f"Domain filter: {args.domain_filter}")
    if args.correct_hint:
        print(f"Correct hint filter: {args.correct_hint}")

    # STEP 1: Load Dataset
    print("\n=== STEP 1: Loading Dataset ===")
    dataset = load_activation_dataset(input_file)

    # Determine layers from dataset info
    num_layers = dataset['info']['num_layers']
    layers_to_compute = list(range(num_layers))
    print(f"Detected {num_layers} layers in dataset. Computing for all.")

    print(f"Total tags: {dataset['info']['tags']}")
    print(f"Total files processed: {dataset['info']['total_files']}")

    # Verify tags
    available_tags = set(dataset['info']['tags'])
    missing_pos = set(args.positive_tags) - available_tags
    missing_neg = set(args.negative_tags) - available_tags

    if missing_pos or missing_neg:
        print(f"WARNING: Missing tags - Positive: {missing_pos}, Negative: {missing_neg}")
        print(f"Available tags: {sorted(available_tags)}")

    # STEP 2: Create Dataset Splits
    print("\n=== STEP 2: Creating Dataset Splits ===")
    # Note: split_dataset_by_prompts respects existing 'split' metadata if present
    dataset_splits = split_dataset_by_prompts(
        dataset=dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_seed=RANDOM_SEED
    )

    # STEP 3: Compute Steering Vectors
    print("\n=== STEP 3: Computing Steering Vectors ===")
    
    steering_vectors, computation_stats = compute_steering_vectors_by_layer(
        dataset_splits=dataset_splits,
        positive_tags=args.positive_tags,
        negative_tags=args.negative_tags,
        layers=layers_to_compute,
        split=args.split,
        use_config_weighting=use_config_weighting,
        config_fields=config_fields,
        correct_hint_filter=args.correct_hint,
        domain_mapping=DOMAIN_MAPPING if args.use_domain_grouping else None,
        domain_filter=args.domain_filter
    )

    print(f"Computed steering vectors for {len(steering_vectors)} layers")

    # STEP 4: Results Summary
    print("\n=== STEP 4: Results Summary ===")
    print_steering_summary(
        steering_vectors=steering_vectors,
        computation_stats=computation_stats,
        positive_tags=args.positive_tags,
        negative_tags=args.negative_tags
    )

    # STEP 5: Save Results
    print("\n=== STEP 5: Saving Results ===")
    
    end_time = time.time()
    processing_time = end_time - start_time

    # Save vectors
    save_steering_vectors(
        steering_vectors=steering_vectors,
        computation_stats=computation_stats,
        output_file=output_file,
        dataset_info=dataset['info']
    )

    # Save JSON summary
    config_data = {
        'computation_date': TODAY,
        'model_name': args.model_name,
        'dataset_file': input_file,
        'configuration': {
            'positive_tags': args.positive_tags,
            'negative_tags': args.negative_tags,
            'split': args.split,
            'layers_to_compute': layers_to_compute,
            'use_config_weighting': use_config_weighting,
            'config_fields': config_fields,
            'correct_hint_filter': args.correct_hint,
            'use_domain_grouping': args.use_domain_grouping,
            'domain_filter': args.domain_filter,
            'domain_mapping': DOMAIN_MAPPING if args.use_domain_grouping else None,
        },
        'dataset_info': dataset['info'],
        'results_summary': {
            'vectors_computed': len(steering_vectors),
            'layers_computed': computation_stats['layers_computed'],
            'processing_time_seconds': processing_time
        }
    }

    # Add config-specific stats
    if use_config_weighting and 'total_configs' in computation_stats:
        config_data['results_summary']['total_configs'] = computation_stats['total_configs']
        config_data['results_summary']['configs_with_data'] = computation_stats['configs_with_data']
        config_data['results_summary']['configs_skipped'] = computation_stats['configs_skipped']

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print(f"Configuration summary saved to {summary_file}")

    print(f"\n=== STEERING VECTOR COMPUTATION COMPLETE ===")
    print(f"Processing time: {processing_time/60:.1f} minutes")
    print(f"✓ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
