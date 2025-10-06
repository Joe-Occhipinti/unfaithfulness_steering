"""
compute_steering_vectors.py

Step 4 of faithfulness steering workflow: Compute steering vectors

Computes steering vectors from activation datasets using contrastive activation addition.
For each layer, computes steering vector as mean(positive_activations) - mean(negative_activations).

Uses reusable modules from src/ for core functionality.
"""

import time
import json
import os
from datetime import datetime

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
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Input and output files - manually specify the exact paths and dates
INPUT_FILE = "data/datasets/activations_local_biased_high_school_us_european_world_history_2025-10-04.pkl"
OUTPUT_FILE = "data/steering vectors/steering_vectors_Fbody_vs_Ubody_local_biased_high_school_us_european_world_history_2025-10-04.pkl"
SUMMARY_FILE = "data/summaries/steering_vectors_summary_Fbody_vs_Ubody_local_biased_high_school_us_european_world_history_2025-10-04.json"

# =============================================================================
# STEERING VECTOR COMPUTATION PARAMETERS (easy to tune)
# =============================================================================

# Tag groupings for steering vectors
POSITIVE_TAGS = ["F_body"]     # Faithful tags
NEGATIVE_TAGS = ["U_body"]     # Unfaithful tags

# Alternative tag groupings (uncomment to use):
# POSITIVE_TAGS = ["F", "F_final"]   # Base + Final faithful only
# NEGATIVE_TAGS = ["U", "U_final"]   # Base + Final unfaithful only
# POSITIVE_TAGS = ["F_final"]        # Final faithful only
# NEGATIVE_TAGS = ["U_final"]        # Final unfaithful only

# Layer configuration
LAYERS_TO_COMPUTE = list(range(32))  # All layers for DeepSeek (tunable)
# LAYERS_TO_COMPUTE = [15, 20, 25, 30, 31]  # Specific layers only

# Split configuration (same as separability analysis)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.3
RANDOM_SEED = 42

# Output configuration
# Generate label string for file naming
POSITIVE_LABEL = "_".join(POSITIVE_TAGS)
NEGATIVE_LABEL = "_".join(NEGATIVE_TAGS)
LABEL_COMBINATION = f"{POSITIVE_LABEL}_vs_{NEGATIVE_LABEL}"

SAVE_RESULTS = True
SAVE_JSON_SUMMARY = True

print(f"=== STEERING VECTOR COMPUTATION ===")
print(f"Dataset: {INPUT_FILE}")
print(f"Positive tags: {POSITIVE_TAGS}")
print(f"Negative tags: {NEGATIVE_TAGS}")
print(f"Layers to compute: {len(LAYERS_TO_COMPUTE)} layers")
print(f"Output: {OUTPUT_FILE}")

# =============================================================================
# STEERING VECTOR COMPUTATION WORKFLOW
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

# STEP 2: Create Dataset Splits
print("\n=== STEP 2: Creating Dataset Splits ===")
dataset_splits = split_dataset_by_prompts(
    dataset=dataset,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    random_seed=RANDOM_SEED
)

print(f"Created splits with train/val ratios: {TRAIN_RATIO}/{VAL_RATIO}")

# STEP 3: Compute Steering Vectors (using TRAIN split only)
print("\n=== STEP 3: Computing Steering Vectors ===")
print("Computing steering vectors as mean(positive) - mean(negative) per layer...")
print("IMPORTANT: Using only TRAIN split to avoid data leakage (same as linear probe training)")

steering_vectors, computation_stats = compute_steering_vectors_by_layer(
    dataset_splits=dataset_splits,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS,
    layers=LAYERS_TO_COMPUTE,
    split="train"
)

print(f"Computed steering vectors for {len(steering_vectors)} layers")

# Quick overview
if steering_vectors:
    vector_norms = [vec.norm().item() for vec in steering_vectors.values()]
    print(f"Vector norms range: {min(vector_norms):.4f} to {max(vector_norms):.4f}")
    print(f"Average vector norm: {sum(vector_norms)/len(vector_norms):.4f}")
else:
    print("WARNING: No steering vectors computed")

# STEP 4: Results Summary
print("\n=== STEP 4: Results Summary ===")

# Print comprehensive summary
print_steering_summary(
    steering_vectors=steering_vectors,
    computation_stats=computation_stats,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS
)

# STEP 5: Save Results
if SAVE_RESULTS:
    print("\n=== STEP 5: Saving Results ===")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Compile all results
    end_time = time.time()
    processing_time = end_time - start_time

    # Save steering vectors with full metadata
    save_steering_vectors(
        steering_vectors=steering_vectors,
        computation_stats=computation_stats,
        output_file=OUTPUT_FILE,
        dataset_info=dataset['info']
    )

    print(f"Steering vectors saved to {OUTPUT_FILE}")

# STEP 6: Save JSON Summary
if SAVE_JSON_SUMMARY:
    print("\n=== STEP 6: Saving JSON Summary ===")

    # Save human-readable summary to summaries directory
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    save_steering_summary_json(
        steering_vectors=steering_vectors,
        computation_stats=computation_stats,
        output_file=SUMMARY_FILE
    )

    config_data = {
        'computation_date': TODAY,
        'dataset_file': INPUT_FILE,
        'configuration': {
            'positive_tags': POSITIVE_TAGS,
            'negative_tags': NEGATIVE_TAGS,
            'layers_to_compute': LAYERS_TO_COMPUTE,
            'save_results': SAVE_RESULTS,
            'save_json_summary': SAVE_JSON_SUMMARY
        },
        'dataset_info': dataset['info'],
        'results_summary': {
            'vectors_computed': len(steering_vectors),
            'layers_computed': computation_stats['layers_computed'],
            'processing_time_seconds': processing_time
        }
    }

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print(f"Configuration saved to {SUMMARY_FILE}")

# STEP 7: Analysis Complete
print(f"\n=== STEERING VECTOR COMPUTATION COMPLETE ===")
print(f"Processing time: {processing_time/60:.1f} minutes")
print(f"✓ Steering vectors computed: {len(steering_vectors)} layers")

if steering_vectors:
    vector_dim = list(steering_vectors.values())[0].shape[0]
    avg_norm = sum(v.norm().item() for v in steering_vectors.values()) / len(steering_vectors)
    print(f"✓ Vector dimension: {vector_dim}")
    print(f"✓ Average vector norm: {avg_norm:.4f}")

if SAVE_RESULTS:
    print(f"✓ Results saved to: {OUTPUT_FILE}")

if SAVE_JSON_SUMMARY:
    print(f"✓ Summaries saved to: {SUMMARY_FILE}")

print(f"\nReady for Step 6: Apply Steering")
print(f"Use steering vectors: {OUTPUT_FILE if SAVE_RESULTS else 'results in memory'}")