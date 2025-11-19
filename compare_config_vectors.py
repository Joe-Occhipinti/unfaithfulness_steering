"""
compare_config_vectors.py

Compare steering vectors from individual domain×hint configurations (9 configs total).
Creates a pairwise cosine similarity matrix showing alignment between all configs.

This provides more granular analysis than compare_domain_vectors.py by examining
each domain×hint combination separately rather than aggregating by domain.

Compares all 9 configurations:
- science×grader_hacking, science×metadata, science×professor
- history×grader_hacking, history×metadata, history×professor
- psychology×grader_hacking, psychology×metadata, psychology×professor
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Import reusable modules
from src.steering import (
    load_activation_dataset,
    group_activations_by_config,
    compute_config_wise_steering_vectors
)
from src.separability import split_dataset_by_prompts

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file - activation dataset
ACTIVATION_DATASET_FILE = "data/sprint_4_2025-10-15/datasets/new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl"

# Steering vector configuration (must match how aggregate vectors were computed)
POSITIVE_TAGS = ["F_body"]
NEGATIVE_TAGS = ["U_body"]
CORRECT_HINT_FILTER = 'False'
LAYERS_TO_COMPUTE = list(range(32))

# Domain mapping (must match compute_steering_vectors.py)
DOMAIN_MAPPING = {
    # Science
    'college_biology': 'science',
    'high_school_biology': 'science',
    'college_chemistry': 'science',
    'high_school_chemistry': 'science',
    # History
    'high_school_european_history': 'history',
    'high_school_us_history': 'history',
    'high_school_world_history': 'history',
    'prehistory': 'history',
    # Psychology
    'high_school_psychology': 'psychology',
}

CONFIG_FIELDS = ['domain', 'hint_template', 'correct_hint']

# Split configuration
TRAIN_RATIO = 1.0
VAL_RATIO = 0.0
RANDOM_SEED = 42

# Output
OUTPUT_DIR = "data/sprint_5_2025-11-15/comparisons"
HEATMAP_FILE = f"{OUTPUT_DIR}/config_similarity_matrix.png"
LAYERWISE_FILE = f"{OUTPUT_DIR}/config_similarity_layerwise.png"
RESULTS_FILE = f"{OUTPUT_DIR}/config_similarity_results.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return (vec1 @ vec2) / (vec1.norm() * vec2.norm())

def compute_similarity_matrix(config_vectors, configs, layer_indices):
    """
    Compute pairwise cosine similarity matrix for all configs.

    Args:
        config_vectors: Dict[config_tuple -> Dict[layer_idx -> vector]]
        configs: List of config tuples
        layer_indices: List of layer indices

    Returns:
        similarity_matrix: numpy array of shape (n_configs, n_configs, n_layers)
    """
    n_configs = len(configs)
    n_layers = len(layer_indices)

    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_configs, n_configs, n_layers))

    # Compute pairwise similarities for each layer
    for layer_idx, layer in enumerate(layer_indices):
        for i, config1 in enumerate(configs):
            for j, config2 in enumerate(configs):
                vec1 = config_vectors[config1][layer]
                vec2 = config_vectors[config2][layer]
                similarity_matrix[i, j, layer_idx] = cosine_similarity(vec1, vec2).item()

    return similarity_matrix

def create_config_labels(configs):
    """Create readable labels for configs."""
    labels = []
    for config in configs:
        # Config is tuple like ('science', 'grader_hacking', 'False')
        # We want format: "science×grader"
        domain = config[0]
        hint = config[1]

        # Shorten hint name
        if hint == 'grader_hacking':
            hint_short = 'grader'
        elif hint == 'metadata':
            hint_short = 'meta'
        elif hint == 'professor':
            hint_short = 'prof'
        else:
            hint_short = hint.split('_')[0]

        label = f"{domain}×{hint_short}"
        labels.append(label)

    return labels

# =============================================================================
# STEP 1: Load Dataset and Create Splits
# =============================================================================

print("="*80)
print("COMPARING DOMAIN×HINT CONFIGURATION VECTORS")
print("="*80)
print("\n=== STEP 1: Loading Dataset and Creating Splits ===")

dataset = load_activation_dataset(ACTIVATION_DATASET_FILE)

print(f"Loaded dataset with {dataset['info']['num_layers']} layers")
print(f"Total tags: {dataset['info']['tags']}")

# Create dataset splits
dataset_splits = split_dataset_by_prompts(
    dataset=dataset,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    random_seed=RANDOM_SEED
)

print(f"Train prompts: {len(dataset_splits['train']['data'])}")

# =============================================================================
# STEP 2: Group Activations by Config and Compute Config Vectors
# =============================================================================

print("\n=== STEP 2: Grouping Activations and Computing Config Vectors ===")

# Group activations by config
grouped_configs = group_activations_by_config(
    dataset_splits=dataset_splits,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS,
    split="train",
    config_fields=CONFIG_FIELDS,
    correct_hint_filter=CORRECT_HINT_FILTER,
    domain_mapping=DOMAIN_MAPPING,
    domain_filter=None  # Include all domains
)

# Compute config-wise steering vectors
config_vectors, config_stats = compute_config_wise_steering_vectors(
    grouped_configs=grouped_configs,
    layers=LAYERS_TO_COMPUTE
)

print(f"\n✓ Computed vectors for {config_stats['configs_with_data']} configurations")

# Get configs and layer indices
configs = list(config_vectors.keys())
config_labels = create_config_labels(configs)

# Get layer indices from first config
sample_config = configs[0]
layer_indices = sorted(list(config_vectors[sample_config].keys()))

print(f"✓ Layers computed: {len(layer_indices)}")
print(f"\nConfigurations:")
for i, (config, label) in enumerate(zip(configs, config_labels)):
    print(f"  {i+1}. {label} ({config})")

# =============================================================================
# STEP 3: Compute Pairwise Similarity Matrix
# =============================================================================

print("\n=== STEP 3: Computing Pairwise Similarity Matrix ===")

similarity_matrix = compute_similarity_matrix(config_vectors, configs, layer_indices)

print(f"\nComputed pairwise similarities across {len(layer_indices)} layers")
print(f"Similarity matrix shape: {similarity_matrix.shape}")

# Show average similarity matrix (for overview)
avg_similarity = similarity_matrix.mean(axis=2)

print("\n--- Average Similarity Matrix (across all layers - for overview only) ---")
print(f"{'':15s} " + " ".join([f"{label:>12s}" for label in config_labels]))
print("-" * (15 + 13 * len(config_labels)))
for i, label1 in enumerate(config_labels):
    row_str = f"{label1:15s}"
    for j, label2 in enumerate(config_labels):
        row_str += f" {avg_similarity[i, j]:12.4f}"
    print(row_str)

# =============================================================================
# STEP 4: Statistical Analysis
# =============================================================================

print("\n=== STEP 4: Statistical Analysis ===")

# Group configs by domain and hint
domains = ['science', 'history', 'psychology']
hints = ['grader_hacking', 'metadata', 'professor']

# Analyze within-domain vs cross-domain similarity
print("\n--- Within-Domain vs Cross-Domain Similarity (averaged across layers) ---")

within_domain_sims = []
cross_domain_sims = []

for i, config1 in enumerate(configs):
    domain1, hint1, _ = config1
    for j, config2 in enumerate(configs):
        if i >= j:  # Skip diagonal and duplicates
            continue
        domain2, hint2, _ = config2

        sim = avg_similarity[i, j]

        if domain1 == domain2:
            within_domain_sims.append(sim)
        else:
            cross_domain_sims.append(sim)

print(f"Within-domain similarity: mean={np.mean(within_domain_sims):.4f}, std={np.std(within_domain_sims):.4f}")
print(f"Cross-domain similarity: mean={np.mean(cross_domain_sims):.4f}, std={np.std(cross_domain_sims):.4f}")
print(f"Difference: {np.mean(within_domain_sims) - np.mean(cross_domain_sims):.4f}")

# Analyze within-hint vs cross-hint similarity
print("\n--- Within-Hint vs Cross-Hint Similarity (averaged across layers) ---")

within_hint_sims = []
cross_hint_sims = []

for i, config1 in enumerate(configs):
    domain1, hint1, _ = config1
    for j, config2 in enumerate(configs):
        if i >= j:
            continue
        domain2, hint2, _ = config2

        sim = avg_similarity[i, j]

        if hint1 == hint2:
            within_hint_sims.append(sim)
        else:
            cross_hint_sims.append(sim)

print(f"Within-hint similarity: mean={np.mean(within_hint_sims):.4f}, std={np.std(within_hint_sims):.4f}")
print(f"Cross-hint similarity: mean={np.mean(cross_hint_sims):.4f}, std={np.std(cross_hint_sims):.4f}")
print(f"Difference: {np.mean(within_hint_sims) - np.mean(cross_hint_sims):.4f}")

# Find most and least similar pairs
all_pairs = []
for i, config1 in enumerate(configs):
    for j, config2 in enumerate(configs):
        if i >= j:
            continue
        all_pairs.append((i, j, config1, config2, avg_similarity[i, j]))

all_pairs.sort(key=lambda x: x[4])

print(f"\n--- Most Similar Config Pairs (averaged across layers) ---")
for i, j, config1, config2, sim in all_pairs[-5:]:
    print(f"{config_labels[i]:15s} vs {config_labels[j]:15s}: {sim:.4f}")

print(f"\n--- Least Similar Config Pairs (averaged across layers) ---")
for i, j, config1, config2, sim in all_pairs[:5]:
    print(f"{config_labels[i]:15s} vs {config_labels[j]:15s}: {sim:.4f}")

# =============================================================================
# STEP 5: Visualization
# =============================================================================

print("\n=== STEP 5: Creating Visualizations ===")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Plot 1: Heatmap of average similarity
fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(avg_similarity, cmap='RdYlGn', vmin=0.2, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(config_labels)))
ax.set_yticks(range(len(config_labels)))
ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(config_labels, fontsize=9)
ax.set_title('Average Cosine Similarity Matrix\n(mean across all layers - for overview only)',
             fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(config_labels)):
    for j in range(len(config_labels)):
        text = ax.text(j, i, f'{avg_similarity[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax, label='Cosine Similarity')
plt.tight_layout()
plt.savefig(HEATMAP_FILE, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {HEATMAP_FILE}")
plt.close()

# Plot 2: Layer-wise similarity for selected pairs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Define groups of comparisons
same_domain_same_hint = []  # Should be 1.0 (diagonal)
same_domain_diff_hint = []  # Within domain, different hints
diff_domain_same_hint = []  # Across domain, same hint
diff_domain_diff_hint = []  # Across domain, different hints

for i, config1 in enumerate(configs):
    domain1, hint1, _ = config1
    for j, config2 in enumerate(configs):
        if i >= j:
            continue
        domain2, hint2, _ = config2

        if domain1 == domain2 and hint1 == hint2:
            same_domain_same_hint.append((i, j, config_labels[i], config_labels[j]))
        elif domain1 == domain2 and hint1 != hint2:
            same_domain_diff_hint.append((i, j, config_labels[i], config_labels[j]))
        elif domain1 != domain2 and hint1 == hint2:
            diff_domain_same_hint.append((i, j, config_labels[i], config_labels[j]))
        else:
            diff_domain_diff_hint.append((i, j, config_labels[i], config_labels[j]))

# Plot 1: Same domain, different hint
ax = axes[0, 0]
for i, j, label1, label2 in same_domain_diff_hint:
    layer_sims = similarity_matrix[i, j, :]
    ax.plot(layer_indices, layer_sims, linewidth=2, alpha=0.7, label=f"{label1} vs {label2}")
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)
ax.set_xlabel('Layer Index', fontsize=10)
ax.set_ylabel('Cosine Similarity', fontsize=10)
ax.set_title('Same Domain, Different Hint', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=1)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.2, 1.05])

# Plot 2: Different domain, same hint
ax = axes[0, 1]
for i, j, label1, label2 in diff_domain_same_hint:
    layer_sims = similarity_matrix[i, j, :]
    ax.plot(layer_indices, layer_sims, linewidth=2, alpha=0.7, label=f"{label1} vs {label2}")
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)
ax.set_xlabel('Layer Index', fontsize=10)
ax.set_ylabel('Cosine Similarity', fontsize=10)
ax.set_title('Different Domain, Same Hint', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=1)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.2, 1.05])

# Plot 3: Different domain, different hint (all pairs)
ax = axes[1, 0]
for i, j, label1, label2 in diff_domain_diff_hint:
    layer_sims = similarity_matrix[i, j, :]
    ax.plot(layer_indices, layer_sims, linewidth=1.5, alpha=0.6, label=f"{label1} vs {label2}")
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)
ax.set_xlabel('Layer Index', fontsize=10)
ax.set_ylabel('Cosine Similarity', fontsize=10)
ax.set_title('Different Domain, Different Hint (all pairs)', fontsize=11, fontweight='bold')
ax.legend(fontsize=6, loc='best', framealpha=0.9, ncol=1)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.2, 1.05])

# Plot 4: Summary statistics by category
ax = axes[1, 1]

categories = ['Same domain\nDiff hint', 'Diff domain\nSame hint', 'Diff domain\nDiff hint']
category_pairs = [same_domain_diff_hint, diff_domain_same_hint, diff_domain_diff_hint]

means = []
stds = []
for pairs in category_pairs:
    if pairs:
        all_sims = [similarity_matrix[i, j, :].mean() for i, j, _, _ in pairs]
        means.append(np.mean(all_sims))
        stds.append(np.std(all_sims))
    else:
        means.append(0)
        stds.append(0)

x = np.arange(len(categories))
bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Mean Cosine Similarity', fontsize=10)
ax.set_title('Summary: Mean Similarity by Category\n(averaged across layers)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.05])

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Domain×Hint Configuration Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(LAYERWISE_FILE, dpi=300, bbox_inches='tight')
print(f"Layer-wise plot saved to: {LAYERWISE_FILE}")
plt.close()

# =============================================================================
# STEP 6: Save Results
# =============================================================================

print("\n=== STEP 6: Saving Results ===")

results = {
    'comparison_info': {
        'activation_dataset_file': ACTIVATION_DATASET_FILE,
        'num_configs': len(configs),
        'num_layers_compared': len(layer_indices),
        'config_labels': config_labels,
        'configs': [list(c) for c in configs],
        'config_fields': CONFIG_FIELDS,
        'positive_tags': POSITIVE_TAGS,
        'negative_tags': NEGATIVE_TAGS
    },
    'average_similarity_matrix': {
        'labels': config_labels,
        'matrix': avg_similarity.tolist()
    },
    'category_statistics': {
        'same_domain_diff_hint': {
            'mean': float(means[0]),
            'std': float(stds[0]),
            'n_pairs': len(same_domain_diff_hint)
        },
        'diff_domain_same_hint': {
            'mean': float(means[1]),
            'std': float(stds[1]),
            'n_pairs': len(diff_domain_same_hint)
        },
        'diff_domain_diff_hint': {
            'mean': float(means[2]),
            'std': float(stds[2]),
            'n_pairs': len(diff_domain_diff_hint)
        }
    },
    'within_vs_cross': {
        'within_domain': {
            'mean': float(np.mean(within_domain_sims)),
            'std': float(np.std(within_domain_sims))
        },
        'cross_domain': {
            'mean': float(np.mean(cross_domain_sims)),
            'std': float(np.std(cross_domain_sims))
        },
        'within_hint': {
            'mean': float(np.mean(within_hint_sims)),
            'std': float(np.std(within_hint_sims))
        },
        'cross_hint': {
            'mean': float(np.mean(cross_hint_sims)),
            'std': float(np.std(cross_hint_sims))
        }
    },
    'most_similar_pairs': [
        {
            'config1': config_labels[i],
            'config2': config_labels[j],
            'similarity': float(sim)
        }
        for i, j, config1, config2, sim in all_pairs[-5:]
    ],
    'least_similar_pairs': [
        {
            'config1': config_labels[i],
            'config2': config_labels[j],
            'similarity': float(sim)
        }
        for i, j, config1, config2, sim in all_pairs[:5]
    ]
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {RESULTS_FILE}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
print(f"\n✓ Compared {len(configs)} configuration vectors across {len(layer_indices)} layers")
print(f"\n--- Key Findings ---")
print(f"Within-domain similarity: {np.mean(within_domain_sims):.4f} (±{np.std(within_domain_sims):.4f})")
print(f"Cross-domain similarity: {np.mean(cross_domain_sims):.4f} (±{np.std(cross_domain_sims):.4f})")
print(f"Difference (domain effect): {np.mean(within_domain_sims) - np.mean(cross_domain_sims):.4f}")
print(f"\nWithin-hint similarity: {np.mean(within_hint_sims):.4f} (±{np.std(within_hint_sims):.4f})")
print(f"Cross-hint similarity: {np.mean(cross_hint_sims):.4f} (±{np.std(cross_hint_sims):.4f})")
print(f"Difference (hint effect): {np.mean(within_hint_sims) - np.mean(cross_hint_sims):.4f}")
print(f"\n✓ Heatmap: {HEATMAP_FILE}")
print(f"✓ Layer-wise plots: {LAYERWISE_FILE}")
print(f"✓ Full results: {RESULTS_FILE}")
print("\n" + "="*80)
