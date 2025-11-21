"""
compare_vector_approaches.py

Compare steering vectors computed with different config weighting approaches.
Analyzes similarity between hint-weighting and domain×hint-weighting approaches.

Metrics:
- Cosine similarity: Direction alignment
- L2 distance: Absolute distance in vector space
- Angular distance: Angle between vectors in degrees
- Statistical tests: T-test for deviation from perfect alignment
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
VECTORS_FILE_1 = "data/sprint_5_2025-11-15/vectors/vectors_mean_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-11-21.pkl"
VECTORS_FILE_2 = "data/sprint_5_2025-11-15/vectors/vectors_mean_domainweighting_scie_hist_psy_X_grader_prof_meta_2025-11-21.pkl"

# Output
OUTPUT_DIR = "data/sprint_5_2025-11-15/comparisons_averaging_approach"
PLOT_FILE = f"{OUTPUT_DIR}/vector_comparison_mean_hints_vs_domains.png"
RESULTS_FILE = f"{OUTPUT_DIR}/vector_comparison_mean_hints_vs_domains_comparison_results.json"

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

def l2_distance(vec1, vec2):
    """Compute L2 (Euclidean) distance between two vectors."""
    return (vec1 - vec2).norm()

def angular_distance(vec1, vec2):
    """Compute angular distance in degrees between two vectors."""
    cos_sim = cosine_similarity(vec1, vec2).item()
    # Clamp to [-1, 1] to avoid numerical errors in arccos
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim))

# =============================================================================
# STEP 1: Load Vector Sets
# =============================================================================

print("="*80)
print("COMPARING STEERING VECTOR APPROACHES")
print("="*80)
print("\n=== STEP 1: Loading Vector Sets ===")

hints_vectors, hints_stats = load_steering_vectors(VECTORS_FILE_1)
domains_vectors, domains_stats = load_steering_vectors(VECTORS_FILE_2)

print(f"\nHint-weighting:")
print(f"  Layers: {len(hints_vectors)}")
print(f"  Config fields: {hints_stats.get('config_fields')}")
print(f"  Total configs: {hints_stats.get('total_configs')}")

print(f"\nDomain×Hint-weighting:")
print(f"  Layers: {len(domains_vectors)}")
print(f"  Config fields: {domains_stats.get('config_fields')}")
print(f"  Total configs: {domains_stats.get('total_configs')}")

# =============================================================================
# STEP 2: Compute Metrics for All Layers
# =============================================================================

print("\n=== STEP 2: Computing Similarity Metrics ===")

# Find common layers
common_layers = sorted(set(hints_vectors.keys()) & set(domains_vectors.keys()))
print(f"Common layers: {len(common_layers)}")

# Initialize storage
metrics = {
    'cosine_similarity': {},
    'l2_distance': {},
    'angular_distance': {}
}

# Compute all metrics
for layer_idx in common_layers:
    vec_hints = hints_vectors[layer_idx]
    vec_domains = domains_vectors[layer_idx]

    metrics['cosine_similarity'][layer_idx] = cosine_similarity(vec_hints, vec_domains).item()
    metrics['l2_distance'][layer_idx] = l2_distance(vec_hints, vec_domains).item()
    metrics['angular_distance'][layer_idx] = angular_distance(vec_hints, vec_domains)

print(f"\nSample metrics (first 5 layers):")
for layer_idx in common_layers[:5]:
    print(f"  Layer {layer_idx}:")
    print(f"    Cosine similarity: {metrics['cosine_similarity'][layer_idx]:.4f}")
    print(f"    L2 distance: {metrics['l2_distance'][layer_idx]:.4f}")
    print(f"    Angular distance: {metrics['angular_distance'][layer_idx]:.2f}°")

# =============================================================================
# STEP 3: Statistical Analysis
# =============================================================================

print("\n=== STEP 3: Statistical Analysis ===")

# Convert to arrays for easier analysis
cos_sims = np.array(list(metrics['cosine_similarity'].values()))
l2_dists = np.array(list(metrics['l2_distance'].values()))
ang_dists = np.array(list(metrics['angular_distance'].values()))

print("\n--- Cosine Similarity ---")
print(f"Mean: {cos_sims.mean():.4f}")
print(f"Std: {cos_sims.std():.4f}")
print(f"Min: {cos_sims.min():.4f} (layer {common_layers[cos_sims.argmin()]})")
print(f"Max: {cos_sims.max():.4f} (layer {common_layers[cos_sims.argmax()]})")
print(f"Median: {np.median(cos_sims):.4f}")

print("\n--- L2 Distance ---")
print(f"Mean: {l2_dists.mean():.4f}")
print(f"Std: {l2_dists.std():.4f}")
print(f"Min: {l2_dists.min():.4f} (layer {common_layers[l2_dists.argmin()]})")
print(f"Max: {l2_dists.max():.4f} (layer {common_layers[l2_dists.argmax()]})")
print(f"Median: {np.median(l2_dists):.4f}")

print("\n--- Angular Distance ---")
print(f"Mean: {ang_dists.mean():.2f}°")
print(f"Std: {ang_dists.std():.2f}°")
print(f"Min: {ang_dists.min():.2f}° (layer {common_layers[ang_dists.argmin()]})")
print(f"Max: {ang_dists.max():.2f}° (layer {common_layers[ang_dists.argmax()]})")
print(f"Median: {np.median(ang_dists):.2f}°")

# Statistical test: Are vectors significantly different from perfectly aligned?
print("\n--- Statistical Test (T-test) ---")
print("H0: Cosine similarities are not different from 1.0 (perfect alignment)")
print("H1: Cosine similarities are significantly different from 1.0")

t_statistic, p_value = stats.ttest_1samp(cos_sims, 1.0)
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.001:
    print(f"Result: HIGHLY SIGNIFICANT (p < 0.001)")
    print(f"The two approaches produce significantly different vectors.")
elif p_value < 0.05:
    print(f"Result: SIGNIFICANT (p < 0.05)")
    print(f"The two approaches produce significantly different vectors.")
else:
    print(f"Result: NOT SIGNIFICANT (p >= 0.05)")
    print(f"Cannot reject null hypothesis - vectors may be similar.")

# Effect size (Cohen's d)
cohens_d = (cos_sims.mean() - 1.0) / cos_sims.std()
print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print("Effect size: SMALL")
elif abs(cohens_d) < 0.5:
    print("Effect size: MEDIUM")
else:
    print("Effect size: LARGE")

# =============================================================================
# STEP 4: Visualization
# =============================================================================

print("\n=== STEP 4: Creating Visualizations ===")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cosine Similarity
ax = axes[0, 0]
ax.plot(common_layers, cos_sims, marker='o', linewidth=2, markersize=4, color='blue', label='Cosine Similarity')
ax.axhline(y=cos_sims.mean(), color='red', linestyle='--', label=f'Mean: {cos_sims.mean():.4f}')
ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Perfect Alignment')
ax.set_xlabel('Layer Index', fontsize=11)
ax.set_ylabel('Cosine Similarity', fontsize=11)
ax.set_title('Cosine Similarity by Layer', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim([max(0, cos_sims.min() - 0.1), 1.05])

# Plot 2: L2 Distance
ax = axes[0, 1]
ax.plot(common_layers, l2_dists, marker='s', linewidth=2, markersize=4, color='orange', label='L2 Distance')
ax.axhline(y=l2_dists.mean(), color='red', linestyle='--', label=f'Mean: {l2_dists.mean():.4f}')
ax.set_xlabel('Layer Index', fontsize=11)
ax.set_ylabel('L2 Distance', fontsize=11)
ax.set_title('L2 Distance by Layer', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Angular Distance
ax = axes[1, 0]
ax.plot(common_layers, ang_dists, marker='^', linewidth=2, markersize=4, color='purple', label='Angular Distance')
ax.axhline(y=ang_dists.mean(), color='red', linestyle='--', label=f'Mean: {ang_dists.mean():.2f}°')
ax.axhline(y=0, color='green', linestyle=':', alpha=0.5, label='Perfect Alignment')
ax.set_xlabel('Layer Index', fontsize=11)
ax.set_ylabel('Angular Distance (degrees)', fontsize=11)
ax.set_title('Angular Distance by Layer', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Distribution of Cosine Similarities
ax = axes[1, 1]
ax.hist(cos_sims, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax.axvline(x=cos_sims.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cos_sims.mean():.4f}')
ax.axvline(x=1.0, color='green', linestyle=':', linewidth=2, label='Perfect Alignment')
ax.set_xlabel('Cosine Similarity', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Cosine Similarities', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Comparison: Hint-Weighting vs Domain×Hint-Weighting', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {PLOT_FILE}")

# =============================================================================
# STEP 5: Save Results
# =============================================================================

print("\n=== STEP 5: Saving Results ===")

results = {
    'comparison_info': {
        'hints_file': VECTORS_FILE_1,
        'domains_file': VECTORS_FILE_2,
        'hints_config_fields': hints_stats.get('config_fields'),
        'domains_config_fields': domains_stats.get('config_fields'),
        'hints_total_configs': hints_stats.get('total_configs'),
        'domains_total_configs': domains_stats.get('total_configs'),
        'num_layers_compared': len(common_layers)
    },
    'statistics': {
        'cosine_similarity': {
            'mean': float(cos_sims.mean()),
            'std': float(cos_sims.std()),
            'min': float(cos_sims.min()),
            'max': float(cos_sims.max()),
            'median': float(np.median(cos_sims)),
            'min_layer': int(common_layers[cos_sims.argmin()]),
            'max_layer': int(common_layers[cos_sims.argmax()])
        },
        'l2_distance': {
            'mean': float(l2_dists.mean()),
            'std': float(l2_dists.std()),
            'min': float(l2_dists.min()),
            'max': float(l2_dists.max()),
            'median': float(np.median(l2_dists)),
            'min_layer': int(common_layers[l2_dists.argmin()]),
            'max_layer': int(common_layers[l2_dists.argmax()])
        },
        'angular_distance': {
            'mean_degrees': float(ang_dists.mean()),
            'std_degrees': float(ang_dists.std()),
            'min_degrees': float(ang_dists.min()),
            'max_degrees': float(ang_dists.max()),
            'median_degrees': float(np.median(ang_dists)),
            'min_layer': int(common_layers[ang_dists.argmin()]),
            'max_layer': int(common_layers[ang_dists.argmax()])
        },
        'statistical_test': {
            'test_type': 'one-sample t-test',
            'null_hypothesis': 'cosine_similarity = 1.0 (perfect alignment)',
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'cohens_d': float(cohens_d)
        }
    },
    'layer_wise_metrics': {
        int(layer): {
            'cosine_similarity': float(metrics['cosine_similarity'][layer]),
            'l2_distance': float(metrics['l2_distance'][layer]),
            'angular_distance_degrees': float(metrics['angular_distance'][layer])
        }
        for layer in common_layers
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
print(f"\n✓ Compared {len(common_layers)} layers")
print(f"✓ Mean cosine similarity: {cos_sims.mean():.4f}")
print(f"✓ Mean angular distance: {ang_dists.mean():.2f}°")
print(f"✓ Mean L2 distance: {l2_dists.mean():.4f}")
print(f"✓ Statistical significance: {'YES (p<0.05)' if p_value < 0.05 else 'NO (p>=0.05)'}")
print(f"\n✓ Visualization: {PLOT_FILE}")
print(f"✓ Full results: {RESULTS_FILE}")
print("\n" + "="*80)
