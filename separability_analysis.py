"""
separability_analysis.py

Step 4 of faithfulness steering workflow: Separability analysis

Analyzes separability between positive and negative activations using three investigations:
1. Cosine similarity between means per layer
2. Norm distributions and mean norm differences per layer
3. Linear probe classification performance per layer

Uses reusable modules from src/ for core functionality.
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
    compute_cosine_similarity_by_layer,
    compute_mean_differences_by_layer,
    train_linear_probes_by_layer,
    compute_pca_analysis_by_layer,
    print_separability_summary
)
from src.plots import (
    plot_cosine_similarity_by_layer,
    plot_mean_differences_by_layer,
    plot_linear_probe_performance,
    plot_pca_separability,
    plot_pca_explained_variance,
    plot_separability_summary
)
from src.config import TODAY, PLOTS_DIR

# =============================================================================
# SEPARABILITY ANALYSIS PARAMETERS (easy to tune)
# =============================================================================

# Input dataset
DATASET_FILE = "data/datasets of activations/activations_annotated_hinted_2025-09-24.pkl"

# Tag groupings for analysis
POSITIVE_TAGS = ["F"]     # Faithful tags
NEGATIVE_TAGS = ["U"]     # Unfaithful tags

# Alternative tag groupings (uncomment to use):
# POSITIVE_TAGS = ["F"]              # Base faithful only
# NEGATIVE_TAGS = ["U"]              # Base unfaithful only
# POSITIVE_TAGS = ["F"]  # Just weakly faithful variants to test correlations with Faithfulness
# POSITIVE_TAGS = ["Fact"]           # Factually correct only, to test correlations with Faithfulness

# Split configuration for linear probes
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Layers to test
LAYERS_TO_ANALYZE = list(range(32))  # All layers for DeepSeek

# PCA configuration
PCA_N_COMPONENTS = 2  # 2D visualization
PCA_LAYERS_TO_PLOT = [8, 15, 23, 25, 28, 31]  # Specific layers to plot (tunable)

# Output configuration
# Generate label string for file naming
POSITIVE_LABEL = "_".join(POSITIVE_TAGS)
NEGATIVE_LABEL = "_".join(NEGATIVE_TAGS)
LABEL_COMBINATION = f"{POSITIVE_LABEL}_vs_{NEGATIVE_LABEL}"

OUTPUT_DIR = f"results/separability_{TODAY}/{LABEL_COMBINATION}"
SAVE_RESULTS = True
CREATE_PLOTS = True

print(f"=== SEPARABILITY ANALYSIS - {TODAY} ===")
print(f"Dataset: {DATASET_FILE}")
print(f"Positive tags: {POSITIVE_TAGS}")
print(f"Negative tags: {NEGATIVE_TAGS}")
print(f"Split ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
print(f"Output: {OUTPUT_DIR}")

# =============================================================================
# SEPARABILITY ANALYSIS WORKFLOW
# =============================================================================

start_time = time.time()

# STEP 1: Load Dataset
print("\n=== STEP 1: Loading Dataset ===")
dataset = load_activation_dataset(DATASET_FILE)

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
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED
)

print(f"Created splits: train, val, test")

# STEP 3: Investigation 1 - Cosine Similarity Analysis
print("\n=== STEP 3: Cosine Similarity Analysis ===")
print("Computing cosine similarity between positive and negative means per layer...")

cosine_similarities = compute_cosine_similarity_by_layer(
    dataset=dataset,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS
)

print(f"Computed cosine similarities for {len(cosine_similarities)} layers")
print(f"Range: {min(cosine_similarities.values()):.3f} to {max(cosine_similarities.values()):.3f}")

# STEP 4: Investigation 2 - Mean Difference Analysis
print("\n=== STEP 4: Mean Difference Analysis ===")
print("Computing distance between positive and negative means per layer...")

mean_differences = compute_mean_differences_by_layer(
    dataset=dataset,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS
)

print(f"Computed mean differences for {len(mean_differences)} layers")

# Extract mean difference norms for quick overview
mean_diff_norms = {layer: mean_differences[layer]['mean_diff_norm']
                   for layer in mean_differences}
print(f"Mean difference norms range: {min(mean_diff_norms.values()):.3f} to {max(mean_diff_norms.values()):.3f}")

# STEP 5: Investigation 3 - PCA Analysis
print("\n=== STEP 5: PCA Analysis ===")
print("Computing PCA projections per layer...")

pca_results = compute_pca_analysis_by_layer(
    dataset=dataset,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS,
    n_components=PCA_N_COMPONENTS
)

print(f"Computed PCA for {len(pca_results)} layers")
valid_pca = {k: v for k, v in pca_results.items() if 'error' not in v}
if valid_pca:
    explained_variances = [sum(v['explained_variance'][:2]) for v in valid_pca.values()]
    print(f"PC1+PC2 explained variance range: {min(explained_variances):.3f} to {max(explained_variances):.3f}")

# STEP 6: Investigation 4 - Linear Probe Analysis
print("\n=== STEP 6: Linear Probe Analysis ===")
print("Training linear probes per layer with train/val/test splits...")

probe_results = train_linear_probes_by_layer(
    dataset_splits=dataset_splits,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS,
    random_seed=RANDOM_SEED
)

print(f"Trained linear probes for {len(probe_results)} layers")

# Quick performance overview
valid_probes = {k: v for k, v in probe_results.items() if 'error' not in v}
if valid_probes:
    val_accs = [result['val_acc'] for result in valid_probes.values()]
    test_accs = [result['test_acc'] for result in valid_probes.values()]
    print(f"Validation accuracy range: {min(val_accs):.3f} to {max(val_accs):.3f}")
    print(f"Test accuracy range: {min(test_accs):.3f} to {max(test_accs):.3f}")
else:
    print("WARNING: No valid probe results obtained")

# STEP 7: Results Summary
print("\n=== STEP 7: Results Summary ===")

# Print comprehensive summary
print_separability_summary(
    cosine_similarities=cosine_similarities,
    mean_differences=mean_differences,
    probe_results=probe_results,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS
)

# STEP 8: Save Results
if SAVE_RESULTS:
    print("\n=== STEP 8: Saving Results ===")

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

    # Prepare PCA results for saving (remove non-serializable PCA models and arrays)
    pca_results_serializable = {}
    for layer_idx, result in pca_results.items():
        if 'error' not in result:
            pca_results_serializable[layer_idx] = {
                'explained_variance': result['explained_variance'],
                'explained_variance_cumulative': result['explained_variance_cumulative'],
                'n_positive': result['n_positive'],
                'n_negative': result['n_negative']
            }
        else:
            pca_results_serializable[layer_idx] = result

    # Compile all results
    end_time = time.time()
    processing_time = end_time - start_time

    results = {
        'analysis_date': TODAY,
        'dataset_file': DATASET_FILE,
        'configuration': {
            'positive_tags': POSITIVE_TAGS,
            'negative_tags': NEGATIVE_TAGS,
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': TEST_RATIO,
            'random_seed': RANDOM_SEED,
            'layers_analyzed': LAYERS_TO_ANALYZE
        },
        'dataset_info': dataset['info'],
        'results': {
            'cosine_similarities': cosine_similarities,
            'mean_differences': mean_differences,
            'pca_analysis': pca_results_serializable,
            'probe_results': probe_results_serializable
        },
        'processing_time_seconds': processing_time
    }

    # Save results
    results_file = os.path.join(OUTPUT_DIR, f"separability_results_{TODAY}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {results_file}")

    # Save individual analysis files for easier access
    cosine_file = os.path.join(OUTPUT_DIR, f"cosine_similarities_{TODAY}.json")
    with open(cosine_file, 'w') as f:
        json.dump(cosine_similarities, f, indent=2)

    means_file = os.path.join(OUTPUT_DIR, f"mean_differences_{TODAY}.json")
    with open(means_file, 'w') as f:
        json.dump(mean_differences, f, indent=2)

    probes_file = os.path.join(OUTPUT_DIR, f"probe_results_{TODAY}.json")
    with open(probes_file, 'w') as f:
        json.dump(probe_results_serializable, f, indent=2)

    # Save the full probe results with trained classifiers in pickle format
    probes_pkl_file = os.path.join(OUTPUT_DIR, f"probe_classifiers_{TODAY}.pkl")
    with open(probes_pkl_file, 'wb') as f:
        pickle.dump(probe_results, f)
    print(f"Saved trained probe classifiers to {probes_pkl_file}")

    pca_file = os.path.join(OUTPUT_DIR, f"pca_analysis_{TODAY}.json")
    with open(pca_file, 'w') as f:
        json.dump(pca_results_serializable, f, indent=2)

    print(f"Individual analysis files saved to {OUTPUT_DIR}")

# STEP 9: Create Visualizations
if CREATE_PLOTS:
    print("\n=== STEP 9: Creating Visualizations ===")

    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Individual plots
    print("Creating cosine similarity plot...")
    plot_cosine_similarity_by_layer(
        cosine_similarities=cosine_similarities,
        positive_tags=POSITIVE_TAGS,
        negative_tags=NEGATIVE_TAGS,
        save_path=os.path.join(plot_dir, f"cosine_similarity_{TODAY}.png"),
        show_plot=False
    )

    print("Creating mean differences plot...")
    plot_mean_differences_by_layer(
        mean_differences=mean_differences,
        positive_tags=POSITIVE_TAGS,
        negative_tags=NEGATIVE_TAGS,
        save_path=os.path.join(plot_dir, f"mean_differences_{TODAY}.png"),
        show_plot=False
    )

    print("Creating linear probe performance plot...")
    plot_linear_probe_performance(
        probe_results=probe_results,
        positive_tags=POSITIVE_TAGS,
        negative_tags=NEGATIVE_TAGS,
        save_path=os.path.join(plot_dir, f"linear_probe_performance_{TODAY}.png"),
        show_plot=False
    )

    print("Creating PCA separability plot...")
    plot_pca_separability(
        pca_results=pca_results,
        positive_tags=POSITIVE_TAGS,
        negative_tags=NEGATIVE_TAGS,
        layers_to_plot=PCA_LAYERS_TO_PLOT,
        save_path=os.path.join(plot_dir, f"pca_separability_{TODAY}.png"),
        show_plot=False
    )

    print("Creating PCA explained variance plot...")
    plot_pca_explained_variance(
        pca_results=pca_results,
        save_path=os.path.join(plot_dir, f"pca_explained_variance_{TODAY}.png"),
        show_plot=False
    )

    print("Creating separability summary plot...")
    plot_separability_summary(
        cosine_similarities=cosine_similarities,
        mean_differences=mean_differences,
        probe_results=probe_results,
        positive_tags=POSITIVE_TAGS,
        negative_tags=NEGATIVE_TAGS,
        save_path=os.path.join(plot_dir, f"separability_summary_{TODAY}.png"),
        show_plot=False
    )

    print(f"All plots saved to {plot_dir}")

# STEP 10: Analysis Complete
print(f"\n=== SEPARABILITY ANALYSIS COMPLETE ===")
print(f"Processing time: {processing_time/60:.1f} minutes")
print(f"✅ Cosine similarity analysis: {len(cosine_similarities)} layers")
print(f"✅ Mean difference analysis: {len(mean_differences)} layers")
print(f"✅ PCA analysis: {len(valid_pca)} valid layers")
print(f"✅ Linear probe analysis: {len(valid_probes)} valid probes")

if SAVE_RESULTS:
    print(f"✅ Results saved to: {OUTPUT_DIR}")

if CREATE_PLOTS:
    print(f"✅ Plots saved to: {plot_dir}")

print(f"\nReady for Step 5: compute steering vectors")
print(f"Use separability results: {results_file if SAVE_RESULTS else 'results in memory'}")