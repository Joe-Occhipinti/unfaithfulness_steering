"""
compare_domain_vectors.py

Compare steering vectors from individual domains (with hint weighting) to the aggregate.
Creates a pairwise cosine similarity matrix showing alignment between domains.

Compares:
- Science-only vector (hints weighted within science)
- History-only vector (hints weighted within history)
- Psychology-only vector (hints weighted within psychology)
- Aggregate domain×hint vector (all 9 configs)
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files - individual domain vectors
SCIENCE_VECTORS_FILE = "data/sprint_5_2025-11-15/vectors/vectors_sciencexhints-weighting_scie_hist_psy_X_grader_prof_meta_2025-11-15.pkl"
HISTORY_VECTORS_FILE = "data/sprint_5_2025-11-15/vectors/vectors_historyxhints-weighting_scie_hist_psy_X_grader_prof_meta_2025-11-15.pkl"
PSYCHOLOGY_VECTORS_FILE = "data/sprint_5_2025-11-15/vectors/vectors_psyxhints-weighting_scie_hist_psy_X_grader_prof_meta_2025-11-15.pkl"

# Aggregate vector (all domains combined)
AGGREGATE_VECTORS_FILE = "data/sprint_5_2025-11-15/vectors/vectors_domainsxhints-weighting_scie_hist_psy_X_grader_prof_meta_2025-11-15.pkl"

# Output
OUTPUT_DIR = "data/sprint_5_2025-11-15/comparisons"
HEATMAP_FILE = f"{OUTPUT_DIR}/domain_similarity_matrix.png"
RESULTS_FILE = f"{OUTPUT_DIR}/domain_similarity_results.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_steering_vectors(file_path):
    """Load steering vectors from pickle file."""
    print(f"Loading: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['steering_vectors'], data['computation_stats']

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return (vec1 @ vec2) / (vec1.norm() * vec2.norm())

def compute_similarity_matrix(vectors_dict, labels):
    """
    Compute pairwise cosine similarity matrix.

    Args:
        vectors_dict: Dict mapping label -> {layer_idx -> vector}
        labels: List of labels in desired order

    Returns:
        similarity_matrix: numpy array of shape (n_labels, n_labels, n_layers)
    """
    n_labels = len(labels)

    # Get layer indices (assume all have same layers)
    layer_indices = sorted(list(vectors_dict[labels[0]].keys()))
    n_layers = len(layer_indices)

    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_labels, n_labels, n_layers))

    # Compute pairwise similarities for each layer
    for layer_idx, layer in enumerate(layer_indices):
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                vec1 = vectors_dict[label1][layer]
                vec2 = vectors_dict[label2][layer]
                similarity_matrix[i, j, layer_idx] = cosine_similarity(vec1, vec2).item()

    return similarity_matrix, layer_indices

# =============================================================================
# STEP 1: Load All Vector Sets
# =============================================================================

print("="*80)
print("COMPARING DOMAIN STEERING VECTORS")
print("="*80)
print("\n=== STEP 1: Loading Vector Sets ===")

science_vectors, science_stats = load_steering_vectors(SCIENCE_VECTORS_FILE)
history_vectors, history_stats = load_steering_vectors(HISTORY_VECTORS_FILE)
psychology_vectors, psychology_stats = load_steering_vectors(PSYCHOLOGY_VECTORS_FILE)
aggregate_vectors, aggregate_stats = load_steering_vectors(AGGREGATE_VECTORS_FILE)

print(f"\nScience: {len(science_vectors)} layers")
print(f"History: {len(history_vectors)} layers")
print(f"Psychology: {len(psychology_vectors)} layers")
print(f"Aggregate: {len(aggregate_vectors)} layers")

# =============================================================================
# STEP 2: Compute Pairwise Similarity Matrix
# =============================================================================

print("\n=== STEP 2: Computing Pairwise Similarity Matrix ===")

# Organize vectors by label
vectors_dict = {
    'Science': science_vectors,
    'History': history_vectors,
    'Psychology': psychology_vectors,
    'Aggregate': aggregate_vectors
}

labels = ['Science', 'History', 'Psychology', 'Aggregate']

# Compute similarity matrix
similarity_matrix, layer_indices = compute_similarity_matrix(vectors_dict, labels)

print(f"\nComputed pairwise similarities across {len(layer_indices)} layers")
print(f"Similarity matrix shape: {similarity_matrix.shape}")

# Instead of averaging, let's look at specific key layers
key_layers = [0, 10, 20, 30, 31]  # Early, middle, late layers
available_key_layers = [l for l in key_layers if l in layer_indices]

print(f"\n--- Layer-wise Cosine Similarity (Key Layers) ---")
for layer_idx in available_key_layers:
    layer_pos = layer_indices.index(layer_idx)
    print(f"\nLayer {layer_idx}:")
    print(f"{'':12s} " + " ".join([f"{label:>10s}" for label in labels]))
    print("-" * 60)
    for i, label1 in enumerate(labels):
        row_str = f"{label1:12s}"
        for j, label2 in enumerate(labels):
            row_str += f" {similarity_matrix[i, j, layer_pos]:10.4f}"
        print(row_str)

# =============================================================================
# STEP 3: Statistical Analysis
# =============================================================================

print("\n=== STEP 3: Statistical Analysis ===")

# Compute average similarity across all layers for overall statistics
avg_similarity = similarity_matrix.mean(axis=2)

print("\n--- Average Domain vs Aggregate Alignment (across all layers) ---")
for i, domain in enumerate(['Science', 'History', 'Psychology']):
    sim_with_agg = avg_similarity[i, 3]  # 3 is Aggregate
    print(f"{domain:12s} vs Aggregate: {sim_with_agg:.4f}")

print("\n--- Average Inter-Domain Alignment (across all layers) ---")
comparisons = [
    (0, 1, 'Science', 'History'),
    (0, 2, 'Science', 'Psychology'),
    (1, 2, 'History', 'Psychology')
]
for i, j, label1, label2 in comparisons:
    sim = avg_similarity[i, j]
    print(f"{label1:12s} vs {label2:12s}: {sim:.4f}")

# Find most and least aligned domain with aggregate (averaged across layers)
domain_indices = [0, 1, 2]  # Science, History, Psychology
domain_agg_sims = [avg_similarity[i, 3] for i in domain_indices]
most_aligned_idx = domain_indices[np.argmax(domain_agg_sims)]
least_aligned_idx = domain_indices[np.argmin(domain_agg_sims)]

print(f"\n--- Key Findings (averaged across layers) ---")
print(f"Most aligned with aggregate: {labels[most_aligned_idx]} ({avg_similarity[most_aligned_idx, 3]:.4f})")
print(f"Least aligned with aggregate: {labels[least_aligned_idx]} ({avg_similarity[least_aligned_idx, 3]:.4f})")

# Layer-wise statistics for domain vs aggregate
print(f"\n--- Layer-wise Statistics: Domain vs Aggregate ---")
for i, domain in enumerate(['Science', 'History', 'Psychology']):
    layer_sims = similarity_matrix[i, 3, :]  # Domain vs Aggregate across all layers
    print(f"\n{domain}:")
    print(f"  Mean: {layer_sims.mean():.4f}")
    print(f"  Std: {layer_sims.std():.4f}")
    print(f"  Min: {layer_sims.min():.4f} (layer {layer_indices[layer_sims.argmin()]})")
    print(f"  Max: {layer_sims.max():.4f} (layer {layer_indices[layer_sims.argmax()]})")

# =============================================================================
# STEP 4: Visualization
# =============================================================================

print("\n=== STEP 4: Creating Visualizations ===")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Heatmap of average similarity (averaged across all layers for overview)
ax = axes[0]
im = ax.imshow(avg_similarity, cmap='RdYlGn', vmin=0.8, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_title('Average Cosine Similarity Matrix\n(mean across all layers - for overview only)', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, f'{avg_similarity[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=ax, label='Cosine Similarity')

# Plot 2: Layer-wise similarity for all pairwise comparisons
ax = axes[1]

# Define all unique pairwise comparisons with distinct styles
# (i, j, label, color, linestyle, linewidth)
pairwise_comparisons = [
    (0, 1, 'Science vs History', '#1f77b4', '-', 2.5),
    (0, 2, 'Science vs Psychology', '#ff7f0e', '-', 2.5),
    (0, 3, 'Science vs Aggregate', '#2ca02c', '-', 2.5),
    (1, 2, 'History vs Psychology', '#d62728', '-', 2.5),
    (1, 3, 'History vs Aggregate', '#9467bd', '-', 2.5),
    (2, 3, 'Psychology vs Aggregate', '#8c564b', '-', 2.5)
]

# Plot each comparison
for i, j, label, color, linestyle, linewidth in pairwise_comparisons:
    layer_sims = similarity_matrix[i, j, :]  # Similarity across all layers
    print(f"{label}: min={layer_sims.min():.4f}, max={layer_sims.max():.4f}, mean={layer_sims.mean():.4f}")
    ax.plot(layer_indices, layer_sims,
            linewidth=linewidth,
            label=label, color=color, linestyle=linestyle, alpha=0.8)

ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, linewidth=1.5, label='Perfect Alignment')
ax.set_xlabel('Layer Index', fontsize=11)
ax.set_ylabel('Cosine Similarity', fontsize=11)
ax.set_title('Pairwise Vector Alignment by Layer', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
# Set y-axis to show all comparisons (inter-domain starts around 0.29)
ax.set_ylim([0.2, 1.05])

plt.suptitle('Domain Steering Vector Comparison', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(HEATMAP_FILE, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {HEATMAP_FILE}")

# =============================================================================
# STEP 5: Save Results
# =============================================================================

print("\n=== STEP 5: Saving Results ===")

results = {
    'comparison_info': {
        'science_file': SCIENCE_VECTORS_FILE,
        'history_file': HISTORY_VECTORS_FILE,
        'psychology_file': PSYCHOLOGY_VECTORS_FILE,
        'aggregate_file': AGGREGATE_VECTORS_FILE,
        'num_layers_compared': len(layer_indices)
    },
    'average_similarity_matrix': {
        'labels': labels,
        'matrix': avg_similarity.tolist()
    },
    'key_comparisons': {
        'domain_vs_aggregate': {
            'science_vs_aggregate': float(avg_similarity[0, 3]),
            'history_vs_aggregate': float(avg_similarity[1, 3]),
            'psychology_vs_aggregate': float(avg_similarity[2, 3])
        },
        'inter_domain': {
            'science_vs_history': float(avg_similarity[0, 1]),
            'science_vs_psychology': float(avg_similarity[0, 2]),
            'history_vs_psychology': float(avg_similarity[1, 2])
        }
    },
    'key_findings': {
        'most_aligned_with_aggregate': labels[most_aligned_idx],
        'most_aligned_similarity': float(avg_similarity[most_aligned_idx, 3]),
        'least_aligned_with_aggregate': labels[least_aligned_idx],
        'least_aligned_similarity': float(avg_similarity[least_aligned_idx, 3])
    },
    'layer_wise_similarity': {
        'layers': layer_indices,
        'science_vs_aggregate': similarity_matrix[0, 3, :].tolist(),
        'history_vs_aggregate': similarity_matrix[1, 3, :].tolist(),
        'psychology_vs_aggregate': similarity_matrix[2, 3, :].tolist()
    }
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
print(f"\n✓ Compared {len(labels)} vectors across {len(layer_indices)} layers")
print(f"✓ Most aligned with aggregate: {labels[most_aligned_idx]} ({avg_similarity[most_aligned_idx, 3]:.4f})")
print(f"✓ Least aligned with aggregate: {labels[least_aligned_idx]} ({avg_similarity[least_aligned_idx, 3]:.4f})")
print(f"\n✓ Visualization: {HEATMAP_FILE}")
print(f"✓ Full results: {RESULTS_FILE}")
print("\n" + "="*80)
