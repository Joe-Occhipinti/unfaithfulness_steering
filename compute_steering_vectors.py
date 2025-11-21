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
INPUT_FILE = "data/sprint_5_2025-11-15/activations/acts_mean_scie_hist_psy_X_grader_prof_meta_2025-11-21.pkl"
OUTPUT_FILE = "data/sprint_5_2025-11-15/vectors/vectors_mean_domainweighting_scie_hist_psy_X_grader_prof_meta_2025-11-21.pkl"
SUMMARY_FILE = "data/sprint_5_2025-11-15/summaries/vectors/summary_vectors_mean_domainweighting_scie_hist_psy_X_grader_prof_meta_2025-11-21.json"
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

# =============================================================================
# DOMAIN GROUPING CONFIGURATION (NEW!)
# =============================================================================

# Domain grouping toggle
USE_DOMAIN_GROUPING = True  # Set to False for backward compatibility (no domain grouping)

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
# CONFIG-WEIGHTED COMPUTATION
# =============================================================================

USE_CONFIG_WEIGHTING = True  # Default: True (equal weight per config)

# Config fields: Define which metadata fields create unique configurations
# If USE_DOMAIN_GROUPING = True, set 'domain' in CONFIG_FIELDS
# If USE_DOMAIN_GROUPING = False, use original fields like 'subject'
if USE_DOMAIN_GROUPING:
    CONFIG_FIELDS = ['hint_template', 'correct_hint', 'domain']  # 3D grouping with domain
else:
    CONFIG_FIELDS = ['hint_template', 'correct_hint']  # Original 2D grouping

# Set USE_CONFIG_WEIGHTING = False to use old pooling behavior (all activations pooled)

# Correct hint filtering
CORRECT_HINT_FILTER = 'False'  # Options: None (use all), "True" (only correct hints), "False" (only incorrect hints)

# Domain filtering (NEW!)
DOMAIN_FILTER = None # Options: None (use all domains), 'science', 'history', 'psychology'
# DOMAIN_FILTER = 'science'     # Uncomment to compute only science domain
# DOMAIN_FILTER = 'history'     # Uncomment to compute only history domain
# DOMAIN_FILTER = 'psychology'  # Uncomment to compute only psychology domain

# Split configuration --> this is ignored if split info is in the activations metadata
TRAIN_RATIO = 1
VAL_RATIO = 0.0
RANDOM_SEED = 42

SAVE_RESULTS = True
SAVE_JSON_SUMMARY = True

print(f"=== STEERING VECTOR COMPUTATION ===")
print(f"Dataset: {INPUT_FILE}")
print(f"Positive tags: {POSITIVE_TAGS}")
print(f"Negative tags: {NEGATIVE_TAGS}")
print(f"Layers to compute: {len(LAYERS_TO_COMPUTE)} layers")
print(f"Domain grouping: {'ENABLED' if USE_DOMAIN_GROUPING else 'DISABLED'}")
if USE_DOMAIN_GROUPING:
    print(f"Domains: {len(set(DOMAIN_MAPPING.values()))} ({', '.join(sorted(set(DOMAIN_MAPPING.values())))})")
print(f"Config fields: {CONFIG_FIELDS}")
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

# Verify split detection and sizes
print(f"\nSplit verification:")
print(f"  Train prompts: {len(dataset_splits['train']['data'])}")
print(f"  Val prompts: {len(dataset_splits['val']['data'])}")

# Count positive/negative activations in train split (using layer 0 as representative)
train_data = dataset_splits['train']['data']
train_pos_count = 0
train_neg_count = 0

for prompt_idx in train_data:
    prompt_info = train_data[prompt_idx]
    layers_data = prompt_info.get("layers", prompt_info)

    if 0 in layers_data:
        layer_0 = layers_data[0]
        for tag in POSITIVE_TAGS:
            if tag in layer_0:
                train_pos_count += layer_0[tag].shape[0]
        for tag in NEGATIVE_TAGS:
            if tag in layer_0:
                train_neg_count += layer_0[tag].shape[0]

print(f"\nActivations in TRAIN split (layer 0 representative):")
print(f"  Positive ({POSITIVE_TAGS}): {train_pos_count} activations")
print(f"  Negative ({NEGATIVE_TAGS}): {train_neg_count} activations")

# STEP 3: Compute Steering Vectors (using TRAIN split only)
print("\n=== STEP 3: Computing Steering Vectors ===")
print("IMPORTANT: Using only TRAIN split to avoid data leakage (same as linear probe training)")

if USE_CONFIG_WEIGHTING:
    print("Mode: Config-weighted (equal weight per metadata configuration)")
    print(f"Config fields: {CONFIG_FIELDS}")
    if USE_DOMAIN_GROUPING:
        print(f"Domain grouping: ENABLED ({len(set(DOMAIN_MAPPING.values()))} domains)")
    else:
        print(f"Domain grouping: DISABLED")
else:
    print("Mode: Pooled (all activations pooled together)")

if CORRECT_HINT_FILTER is not None:
    print(f"Filtering: Using only prompts with correct_hint={CORRECT_HINT_FILTER}")
else:
    print("Filtering: Using all prompts (no correct_hint filter)")

if DOMAIN_FILTER is not None:
    print(f"Filtering: Using only domain={DOMAIN_FILTER}")
else:
    print("Filtering: Using all domains")

steering_vectors, computation_stats = compute_steering_vectors_by_layer(
    dataset_splits=dataset_splits,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS,
    layers=LAYERS_TO_COMPUTE,
    split="train",
    use_config_weighting=USE_CONFIG_WEIGHTING,
    config_fields=CONFIG_FIELDS,
    correct_hint_filter=CORRECT_HINT_FILTER,
    domain_mapping=DOMAIN_MAPPING if USE_DOMAIN_GROUPING else None,
    domain_filter=DOMAIN_FILTER
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
            'use_config_weighting': USE_CONFIG_WEIGHTING,
            'config_fields': CONFIG_FIELDS,
            'correct_hint_filter': CORRECT_HINT_FILTER,
            'use_domain_grouping': USE_DOMAIN_GROUPING,
            'domain_mapping': DOMAIN_MAPPING if USE_DOMAIN_GROUPING else None,
            'num_domains': len(set(DOMAIN_MAPPING.values())) if USE_DOMAIN_GROUPING else 0,
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

    # Add config-specific stats if config weighting was used
    if USE_CONFIG_WEIGHTING and 'total_configs' in computation_stats:
        config_data['results_summary']['total_configs'] = computation_stats['total_configs']
        config_data['results_summary']['configs_with_data'] = computation_stats['configs_with_data']
        config_data['results_summary']['configs_skipped'] = computation_stats['configs_skipped']

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

if USE_CONFIG_WEIGHTING:
    print(f"✓ Config weighting: ENABLED")
    print(f"  Total configs: {computation_stats.get('total_configs', 'N/A')}")
    print(f"  Configs with data: {computation_stats.get('configs_with_data', 'N/A')}")
    if USE_DOMAIN_GROUPING:
        print(f"✓ Domain grouping: ENABLED")
        print(f"  Domains: {len(set(DOMAIN_MAPPING.values()))} ({', '.join(sorted(set(DOMAIN_MAPPING.values())))})")

if SAVE_RESULTS:
    print(f"✓ Results saved to: {OUTPUT_FILE}")

if SAVE_JSON_SUMMARY:
    print(f"✓ Summaries saved to: {SUMMARY_FILE}")

print(f"\nReady for Step 6: Apply Steering")
print(f"Use steering vectors: {OUTPUT_FILE if SAVE_RESULTS else 'results in memory'}")