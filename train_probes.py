"""
train_probes.py

Step 4b of faithfulness steering workflow: Linear probe training

Trains linear probes to classify faithful vs unfaithful activations per layer.
Uses train/val splits to evaluate probe performance.

This script is separate from separability analysis to keep probe training modular.
"""

import time
import json
import pickle
import os
from datetime import datetime

# Import reusable modules
from src.separability import (
    load_activation_dataset,
    split_dataset_by_prompts,
    train_linear_probes_by_layer
)
from src.plots import (
    plot_linear_probe_performance
)
from src.config import TODAY

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Input and output files - manually specify the exact paths and dates
INPUT_FILE = "data/datasets/cut_activations_global_biased_psychology_2025-10-12.pkl"
OUTPUT_DIR = "data/probes/cut_probes_global_F_vs_U_biased_psychology_professor_2025-10-12"
PLOTS_DIR = "plots/psychology_professor_2025-08-15/cut_probes_global_F_vs_U_biased_psychology_professor_2025-10-12"
SUMMARY_FILE = "data/summaries/probes/summary_cut_probes_global_F_vs_U_biased_psychology_professor_2025-10-12.json"

# =============================================================================
# PROBE TRAINING PARAMETERS (easy to tune)
# =============================================================================

# Tag groupings for analysis
POSITIVE_TAGS = ["F_final"]     # Faithful tags
NEGATIVE_TAGS = ["U_final"]     # Unfaithful tags

# Alternative tag groupings (uncomment to use):
# POSITIVE_TAGS = ["F", "F_wk"]           # Add weakly faithful variants
# POSITIVE_TAGS = ["F", "Fact"]           # Add factually correct variants

# Split configuration for linear probes
TRAIN_RATIO = 0.7
VAL_RATIO = 0.3
RANDOM_SEED = 42
BALANCE_VAL_SPLIT = True  # Set to True to balance faithful/unfaithful samples in val split
BALANCE_TEMPLATES = True  # Set to True to use template-aware balanced downsampling for probes

# Layers to test
LAYERS_TO_ANALYZE = list(range(32))  # All layers for DeepSeek

# Output configuration
# Generate label string for file naming
POSITIVE_LABEL = "_".join(POSITIVE_TAGS)
NEGATIVE_LABEL = "_".join(NEGATIVE_TAGS)
LABEL_COMBINATION = f"{POSITIVE_LABEL}_vs_{NEGATIVE_LABEL}"

SAVE_RESULTS = True
CREATE_PLOTS = True

print(f"=== LINEAR PROBE TRAINING ===")
print(f"Dataset: {INPUT_FILE}")
print(f"Positive tags: {POSITIVE_TAGS}")
print(f"Negative tags: {NEGATIVE_TAGS}")
print(f"Split ratios: {TRAIN_RATIO}/{VAL_RATIO}")
print(f"Output: {OUTPUT_DIR}")

# =============================================================================
# PROBE TRAINING WORKFLOW
# =============================================================================

start_time = time.time()

# STEP 1: Load Dataset
print("\n=== STEP 1: Loading Dataset ===")
dataset = load_activation_dataset(INPUT_FILE)

print(f"Loaded dataset with {dataset['info']['num_layers']} layers")
print(f"Total tags: {dataset['info']['tags']}")
print(f"Total files processed: {dataset['info']['total_files']}")

# Verify that our target tags exist in the dataset
available_tags = set(dataset['info']['tags'])
missing_pos = set(POSITIVE_TAGS) - available_tags
missing_neg = set(NEGATIVE_TAGS) - available_tags

if missing_pos or missing_neg:
    print(f"WARNING: Missing tags - Positive: {missing_pos}, Negative: {missing_neg}")
    print(f"Available tags: {sorted(available_tags)}")

# STEP 2: Create Dataset Splits for Linear Probes
print("\n=== STEP 2: Creating Dataset Splits ===")
dataset_splits = split_dataset_by_prompts(
    dataset=dataset,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    random_seed=RANDOM_SEED,
    balance_val_split=BALANCE_VAL_SPLIT,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS
)

print(f"Created splits: train, val")

# Get split statistics for summary
train_data = dataset_splits['train']['data']
val_data = dataset_splits['val']['data']

# Count prompts per split
split_stats = {
    'train_prompts': len(train_data),
    'val_prompts': len(val_data)
}

# If metadata available, get stratification info
if 'metadata_fields' in dataset_splits['train']['info']:
    metadata_fields = dataset_splits['train']['info']['metadata_fields']

    # Count by faithfulness and template in each split
    def count_metadata(data):
        from collections import defaultdict
        counts = defaultdict(lambda: defaultdict(int))
        for prompt_data in data.values():
            metadata = prompt_data.get('metadata', {})
            faithfulness = metadata.get('faithfulness_classification', 'unknown')
            template = metadata.get('hint_template', 'unknown')
            counts[faithfulness][template] += 1
        return counts

    split_stats['train_by_class_template'] = count_metadata(train_data)
    split_stats['val_by_class_template'] = count_metadata(val_data)
    split_stats['stratified_by'] = metadata_fields
else:
    split_stats['stratified_by'] = None

# STEP 3: Linear Probe Training
print("\n=== STEP 3: Training Linear Probes ===")
print("Training linear probes per layer with train/val splits...")

probe_results = train_linear_probes_by_layer(
    dataset_splits=dataset_splits,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS,
    random_seed=RANDOM_SEED,
    balance_templates=BALANCE_TEMPLATES
)

print(f"Trained linear probes for {len(probe_results)} layers")

# Quick performance overview
valid_probes = {k: v for k, v in probe_results.items() if 'error' not in v}
if valid_probes:
    val_accs = [result['val_acc'] for result in valid_probes.values()]
    print(f"Validation accuracy range: {min(val_accs):.3f} to {max(val_accs):.3f}")

    # Best performing layers
    sorted_layers = sorted(valid_probes.items(), key=lambda x: x[1]['val_acc'], reverse=True)
    print(f"\nTop 5 layers by validation accuracy:")
    for layer_idx, result in sorted_layers[:5]:
        print(f"  Layer {layer_idx}: {result['val_acc']:.3f}")
else:
    print("WARNING: No valid probe results obtained")

# Calculate processing time (used by both save and print steps)
end_time = time.time()
processing_time = end_time - start_time

# STEP 4: Save Results
if SAVE_RESULTS:
    print("\n=== STEP 4: Saving Results ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare results for saving (remove non-serializable objects)
    probe_results_serializable = {}
    for layer_idx, result in probe_results.items():
        if 'classifier' in result:
            # Save classifier separately or remove it for JSON serialization
            serializable_result = {k: v for k, v in result.items() if k != 'classifier'}
            probe_results_serializable[layer_idx] = serializable_result
        else:
            probe_results_serializable[layer_idx] = result

    # Compile all results
    results = {
        'analysis_date': TODAY,
        'dataset_file': INPUT_FILE,
        'configuration': {
            'positive_tags': POSITIVE_TAGS,
            'negative_tags': NEGATIVE_TAGS,
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'random_seed': RANDOM_SEED,
            'balance_val_split': BALANCE_VAL_SPLIT,
            'balance_templates': BALANCE_TEMPLATES,
            'layers_analyzed': LAYERS_TO_ANALYZE
        },
        'dataset_info': dataset['info'],
        'split_statistics': split_stats,
        'results': {
            'probe_results': probe_results_serializable
        },
        'processing_time_seconds': processing_time
    }

    # Save summary to summaries directory
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {SUMMARY_FILE}")

    # Save probe results
    probes_file = os.path.join(OUTPUT_DIR, f"probe_results_{TODAY}.json")
    with open(probes_file, 'w') as f:
        json.dump(probe_results_serializable, f, indent=2)

    # Save the full probe results with trained classifiers in pickle format
    probes_pkl_file = os.path.join(OUTPUT_DIR, f"probe_classifiers_{TODAY}.pkl")
    with open(probes_pkl_file, 'wb') as f:
        pickle.dump(probe_results, f)
    print(f"Saved trained probe classifiers to {probes_pkl_file}")

    print(f"Individual analysis files saved to {OUTPUT_DIR}")

# STEP 5: Create Visualizations
if CREATE_PLOTS:
    print("\n=== STEP 5: Creating Visualizations ===")

    plot_dir = PLOTS_DIR
    os.makedirs(plot_dir, exist_ok=True)

    print("Creating linear probe performance plot...")
    plot_linear_probe_performance(
        probe_results=probe_results,
        positive_tags=POSITIVE_TAGS,
        negative_tags=NEGATIVE_TAGS,
        save_path=os.path.join(plot_dir, f"linear_probe_performance_{TODAY}.png"),
        show_plot=False
    )

    print(f"Plots saved to {plot_dir}")

# STEP 6: Training Complete
print(f"\n=== LINEAR PROBE TRAINING COMPLETE ===")
print(f"Processing time: {processing_time/60:.1f} minutes")
print(f"✅ Linear probe analysis: {len(valid_probes)} valid probes")

if SAVE_RESULTS:
    print(f"✅ Results saved to: {OUTPUT_DIR}")

if CREATE_PLOTS:
    print(f"✅ Plots saved to: {plot_dir}")

if SAVE_RESULTS:
    print(f"\nProbe classifiers can be loaded from: {probes_pkl_file}")
else:
    print(f"\nProbe classifiers: not saved")
