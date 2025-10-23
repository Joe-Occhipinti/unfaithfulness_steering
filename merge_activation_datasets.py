"""
merge_activation_datasets.py

Merges multiple activation datasets into one combined dataset.
Preserves train/val splits from individual datasets.
Recomputes aggregate statistics.

Usage:
  - Specify input dataset paths in INPUT_FILES
  - Specify output path in OUTPUT_FILE
  - Run script
"""

import pickle
import torch
import os
from collections import defaultdict
from datetime import datetime
from src.config import TODAY

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input datasets to merge (add as many as needed)
INPUT_FILES = [
    "data/sprint4_2025-10-21/datasets/acts_pos_bas_histXmeta_2025-10-19.pkl",
    "data/sprint4_2025-10-21/datasets/acts_neg_bas_histXmeta_2025-10-19.pkl",
]

# Output merged dataset
OUTPUT_FILE = "data/sprint4_2025-10-21/datasets/acts_merged_pos_neg_bas_histXmeta_2025-10-19.pkl"

# Output summary JSON
SUMMARY_FILE = "data/sprint4_2025-10-21/summaries/merge_summary_pos_neg_bas_histXmeta_2025-10-19.json"

# =============================================================================
# MERGING FUNCTIONS
# =============================================================================

def load_dataset(file_path):
    """Load activation dataset from pickle file."""
    print(f"Loading: {file_path}")
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"  Prompts: {dataset['info']['num_prompts']}")
    print(f"  Tags: {dataset['info']['tags']}")
    print(f"  Total stats: {dataset['info']['total_stats']}")

    return dataset


def validate_datasets_compatibility(datasets):
    """
    Validate that all datasets are compatible for merging.
    They must have:
    - Same number of layers
    - Same hidden dimension
    - Same tag set
    """
    print("\n=== VALIDATING DATASET COMPATIBILITY ===\n")

    reference = datasets[0]['info']

    for i, dataset in enumerate(datasets[1:], start=1):
        info = dataset['info']

        # Check layers
        if info['num_layers'] != reference['num_layers']:
            raise ValueError(
                f"Dataset {i} has {info['num_layers']} layers, "
                f"but dataset 0 has {reference['num_layers']} layers"
            )

        # Check hidden dimension
        if info['hidden_dim'] != reference['hidden_dim']:
            raise ValueError(
                f"Dataset {i} has hidden_dim={info['hidden_dim']}, "
                f"but dataset 0 has hidden_dim={reference['hidden_dim']}"
            )

        # Check tags (should be identical set)
        if set(info['tags']) != set(reference['tags']):
            raise ValueError(
                f"Dataset {i} has tags {info['tags']}, "
                f"but dataset 0 has tags {reference['tags']}"
            )

    print("[OK] All datasets are compatible for merging")
    print(f"  Layers: {reference['num_layers']}")
    print(f"  Hidden dim: {reference['hidden_dim']}")
    print(f"  Tags: {reference['tags']}")


def merge_datasets(datasets):
    """
    Merge multiple activation datasets into one.

    Args:
        datasets: List of dataset dicts (each with 'data' and 'info')

    Returns:
        Merged dataset dict
    """
    print("\n=== MERGING DATASETS ===\n")

    # Merged data structure
    merged_data = {}
    prompt_counter = 0

    # Track statistics
    total_stats = defaultdict(int)
    layer_stats = defaultdict(lambda: defaultdict(int))
    metadata_fields_set = set()
    source_files = []

    # Get reference info from first dataset
    reference_info = datasets[0]['info']
    num_layers = reference_info['num_layers']
    hidden_dim = reference_info['hidden_dim']
    tags = reference_info['tags']

    # Merge prompts from all datasets
    for dataset_idx, dataset in enumerate(datasets):
        print(f"Merging dataset {dataset_idx}...")

        data = dataset['data']
        info = dataset['info']

        # Track source file if available
        if 'source_jsonl' in info:
            source_files.append(info['source_jsonl'])

        # Track metadata fields
        if 'metadata_fields' in info:
            metadata_fields_set.update(info['metadata_fields'])

        # Add each prompt with renumbered index
        for old_prompt_idx, prompt_data in data.items():
            # Assign new prompt index
            merged_data[prompt_counter] = prompt_data

            # Update statistics by scanning layer 0 (representative)
            layers_data = prompt_data.get('layers', prompt_data)
            if 0 in layers_data:
                layer_0 = layers_data[0]
                for tag in tags:
                    if tag in layer_0:
                        count = layer_0[tag].shape[0]
                        total_stats[tag] += count

                        # Update all layers (assume same count per layer)
                        for layer_idx in range(num_layers):
                            layer_stats[layer_idx][tag] += count

            prompt_counter += 1

        print(f"  Added {len(data)} prompts (renumbered to {prompt_counter - len(data)} - {prompt_counter - 1})")

    # Compile merged info
    merged_info = {
        'tags': tags,
        'num_layers': num_layers,
        'num_prompts': prompt_counter,
        'hidden_dim': hidden_dim,
        'total_files': prompt_counter,
        'files_with_data': prompt_counter,
        'total_stats': dict(total_stats),
        'layer_stats': {
            str(layer_idx): dict(layer_stats[layer_idx])
            for layer_idx in range(num_layers)
        },
        'metadata_fields': sorted(list(metadata_fields_set)),
        'source_datasets': INPUT_FILES,
        'num_source_datasets': len(datasets),
        'merge_date': TODAY
    }

    if source_files:
        merged_info['source_jsonl_files'] = source_files

    merged_dataset = {
        'data': merged_data,
        'info': merged_info
    }

    print(f"\n[OK] Merged {len(datasets)} datasets")
    print(f"  Total prompts: {prompt_counter}")
    print(f"  Total stats: {dict(total_stats)}")

    return merged_dataset


def print_split_distribution(merged_dataset):
    """Print train/val split distribution in merged dataset."""
    print("\n=== SPLIT DISTRIBUTION ===\n")

    split_counts = defaultdict(int)
    split_stats = defaultdict(lambda: defaultdict(int))

    for prompt_idx, prompt_data in merged_dataset['data'].items():
        split_val = prompt_data.get('metadata', {}).get('split', 'unknown')
        split_counts[split_val] += 1

        # Count activations per split
        layers_data = prompt_data.get('layers', prompt_data)
        if 0 in layers_data:
            layer_0 = layers_data[0]
            for tag in merged_dataset['info']['tags']:
                if tag in layer_0:
                    split_stats[split_val][tag] += layer_0[tag].shape[0]

    for split_name in sorted(split_counts.keys()):
        print(f"{split_name.upper()} split:")
        print(f"  Prompts: {split_counts[split_name]}")
        print(f"  Activations (layer 0): {dict(split_stats[split_name])}")


def save_merged_dataset(merged_dataset, output_file):
    """Save merged dataset to pickle file."""
    print(f"\n=== SAVING MERGED DATASET ===\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(merged_dataset, f)

    print(f"[OK] Saved to: {output_file}")

    # Print file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")


def compute_split_statistics(datasets, merged_dataset):
    """
    Compute comprehensive statistics about train/val splits
    before and after merging.

    Returns:
        Dictionary with split statistics
    """
    stats = {
        'source_datasets': [],
        'merged_dataset': {}
    }

    # Statistics for each source dataset
    for i, dataset in enumerate(datasets):
        dataset_stats = {
            'file': INPUT_FILES[i],
            'total_prompts': dataset['info']['num_prompts'],
            'splits': {}
        }

        # Count splits
        split_counts = defaultdict(int)
        split_activations = defaultdict(lambda: defaultdict(int))

        for prompt_idx, prompt_data in dataset['data'].items():
            split_val = prompt_data.get('metadata', {}).get('split', 'unknown')
            split_counts[split_val] += 1

            # Count activations (layer 0 representative)
            layers_data = prompt_data.get('layers', prompt_data)
            if 0 in layers_data:
                layer_0 = layers_data[0]
                for tag in dataset['info']['tags']:
                    if tag in layer_0:
                        split_activations[split_val][tag] += layer_0[tag].shape[0]

        # Calculate ratios
        total = dataset['info']['num_prompts']
        for split_name in ['train', 'val']:
            count = split_counts.get(split_name, 0)
            dataset_stats['splits'][split_name] = {
                'prompts': count,
                'ratio': count / total if total > 0 else 0,
                'activations': dict(split_activations[split_name])
            }

        stats['source_datasets'].append(dataset_stats)

    # Statistics for merged dataset
    merged_stats = {
        'total_prompts': merged_dataset['info']['num_prompts'],
        'splits': {}
    }

    split_counts = defaultdict(int)
    split_activations = defaultdict(lambda: defaultdict(int))

    for prompt_idx, prompt_data in merged_dataset['data'].items():
        split_val = prompt_data.get('metadata', {}).get('split', 'unknown')
        split_counts[split_val] += 1

        layers_data = prompt_data.get('layers', prompt_data)
        if 0 in layers_data:
            layer_0 = layers_data[0]
            for tag in merged_dataset['info']['tags']:
                if tag in layer_0:
                    split_activations[split_val][tag] += layer_0[tag].shape[0]

    total = merged_dataset['info']['num_prompts']
    for split_name in ['train', 'val']:
        count = split_counts.get(split_name, 0)
        merged_stats['splits'][split_name] = {
            'prompts': count,
            'ratio': count / total if total > 0 else 0,
            'activations': dict(split_activations[split_name])
        }

    stats['merged_dataset'] = merged_stats

    # Ratio preservation check
    stats['ratio_preservation'] = {}

    # Check if merged ratios are similar to source ratios
    # (they should be similar if sources have similar ratios)
    source_train_ratios = [ds['splits']['train']['ratio'] for ds in stats['source_datasets']]
    source_val_ratios = [ds['splits']['val']['ratio'] for ds in stats['source_datasets']]

    merged_train_ratio = merged_stats['splits']['train']['ratio']
    merged_val_ratio = merged_stats['splits']['val']['ratio']

    stats['ratio_preservation'] = {
        'train': {
            'source_ratios': source_train_ratios,
            'source_mean': sum(source_train_ratios) / len(source_train_ratios),
            'source_min': min(source_train_ratios),
            'source_max': max(source_train_ratios),
            'merged_ratio': merged_train_ratio,
            'difference_from_mean': abs(merged_train_ratio - sum(source_train_ratios) / len(source_train_ratios))
        },
        'val': {
            'source_ratios': source_val_ratios,
            'source_mean': sum(source_val_ratios) / len(source_val_ratios),
            'source_min': min(source_val_ratios),
            'source_max': max(source_val_ratios),
            'merged_ratio': merged_val_ratio,
            'difference_from_mean': abs(merged_val_ratio - sum(source_val_ratios) / len(source_val_ratios))
        }
    }

    return stats


def save_merge_summary(datasets, merged_dataset, summary_file):
    """Save comprehensive merge summary to JSON."""
    import json

    print(f"\n=== SAVING MERGE SUMMARY ===\n")

    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    # Compute statistics
    split_stats = compute_split_statistics(datasets, merged_dataset)

    # Compile summary
    summary = {
        'merge_metadata': {
            'merge_date': TODAY,
            'num_source_datasets': len(datasets),
            'source_files': INPUT_FILES,
            'output_file': OUTPUT_FILE
        },
        'compatibility_check': {
            'num_layers': merged_dataset['info']['num_layers'],
            'hidden_dim': merged_dataset['info']['hidden_dim'],
            'tags': merged_dataset['info']['tags']
        },
        'split_statistics': split_stats,
        'merged_totals': {
            'total_prompts': merged_dataset['info']['num_prompts'],
            'total_activations': merged_dataset['info']['total_stats'],
            'train_prompts': split_stats['merged_dataset']['splits']['train']['prompts'],
            'val_prompts': split_stats['merged_dataset']['splits']['val']['prompts'],
            'train_ratio': split_stats['merged_dataset']['splits']['train']['ratio'],
            'val_ratio': split_stats['merged_dataset']['splits']['val']['ratio']
        }
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Summary saved to: {summary_file}")

    # Print key findings
    print(f"\n=== RATIO PRESERVATION CHECK ===")
    print(f"\nTRAIN split:")
    print(f"  Source ratios: {[f'{r:.3f}' for r in split_stats['ratio_preservation']['train']['source_ratios']]}")
    print(f"  Source mean: {split_stats['ratio_preservation']['train']['source_mean']:.3f}")
    print(f"  Merged ratio: {split_stats['ratio_preservation']['train']['merged_ratio']:.3f}")
    print(f"  Difference: {split_stats['ratio_preservation']['train']['difference_from_mean']:.3f}")

    print(f"\nVAL split:")
    print(f"  Source ratios: {[f'{r:.3f}' for r in split_stats['ratio_preservation']['val']['source_ratios']]}")
    print(f"  Source mean: {split_stats['ratio_preservation']['val']['source_mean']:.3f}")
    print(f"  Merged ratio: {split_stats['ratio_preservation']['val']['merged_ratio']:.3f}")
    print(f"  Difference: {split_stats['ratio_preservation']['val']['difference_from_mean']:.3f}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

if __name__ == "__main__":
    print("=== ACTIVATION DATASET MERGER ===\n")
    print(f"Merging {len(INPUT_FILES)} datasets:")
    for i, file in enumerate(INPUT_FILES):
        print(f"  {i}: {file}")
    print(f"\nOutput: {OUTPUT_FILE}\n")

    # Step 1: Load all datasets
    print("=== STEP 1: LOADING DATASETS ===\n")
    datasets = []
    for file_path in INPUT_FILES:
        dataset = load_dataset(file_path)
        datasets.append(dataset)

    # Step 2: Validate compatibility
    print()
    validate_datasets_compatibility(datasets)

    # Step 3: Merge datasets
    print()
    merged_dataset = merge_datasets(datasets)

    # Step 4: Print split distribution
    print_split_distribution(merged_dataset)

    # Step 5: Save merged dataset
    print()
    save_merged_dataset(merged_dataset, OUTPUT_FILE)

    # Step 6: Save merge summary
    save_merge_summary(datasets, merged_dataset, SUMMARY_FILE)

    print("\n=== MERGE COMPLETE ===")
    print(f"\nFiles created:")
    print(f"  Dataset: {OUTPUT_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")
    print(f"\nNext steps:")
    print(f"  1. Review the summary JSON to verify train/val ratio preservation")
    print(f"  2. Use the merged dataset in compute_steering_vectors.py")
    print(f"  3. Update INPUT_FILE to: {OUTPUT_FILE}")
    print(f"  4. Run compute_steering_vectors.py to get combined steering vector")
