"""
separability_analysis.py

Step 4a of faithfulness steering workflow: Separability analysis

Analyzes separability between positive and negative activations using three investigations:
1. Cosine similarity between means per layer
2. Norm distributions and mean norm differences per layer
3. PCA analysis per layer

Does NOT include probe training - see train_probes.py for that.
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
    compute_cosine_similarity_by_layer,
    compute_mean_differences_by_layer,
    compute_pca_analysis_by_layer
)
from src.plots import (
    plot_cosine_similarity_by_layer,
    plot_mean_differences_by_layer,
    plot_pca_separability,
    plot_pca_explained_variance
)
from src.config import TODAY, PLOTS_DIR

# =============================================================================
# I/O CONFIGURATION (manually specify all paths)
# =============================================================================

# Input and output files - manually specify the exact paths and dates
INPUT_FILE = "data/datasets/cut_activations_global_biased_psychology_2025-10-12.pkl"
OUTPUT_DIR = "data/separability/cut_separability_global_F_vs_U_biased_psychology_professor_2025-10-12"
PLOTS_DIR = "plots/psychology_professor_2025-08-15/cut_separability_global_F_vs_U_biased_psychology_professor_2025-10-12"
SUMMARY_FILE = "data/summaries/separability/summary_cut_separability_global_F_vs_U_biased_psychology_professor_2025-10-12.json"
# =============================================================================
# SEPARABILITY ANALYSIS PARAMETERS (easy to tune)
# =============================================================================
# Tag groupings for analysis
POSITIVE_TAGS = ["F_final"]     # Faithful tags
NEGATIVE_TAGS = ["U_final"]     # Unfaithful tags

# Alternative tag groupings (uncomment to use):
# POSITIVE_TAGS = ["F", "F_wk"]           # Add weakly faithful variants
# POSITIVE_TAGS = ["F", "Fact"]           # Add factually correct variants

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

SAVE_RESULTS = True
CREATE_PLOTS = True

print(f"=== SEPARABILITY ANALYSIS ===")
print(f"Dataset: {INPUT_FILE}")
print(f"Positive tags: {POSITIVE_TAGS}")
print(f"Negative tags: {NEGATIVE_TAGS}")
print(f"Output: {OUTPUT_DIR}")

# =============================================================================
# SEPARABILITY ANALYSIS WORKFLOW
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

# Collect dataset statistics for debugging
dataset_statistics = {
    'total_prompts': len(dataset['data']),
    'num_layers': dataset['info']['num_layers'],
    'hidden_dim': dataset['info']['hidden_dim'],
    'available_tags': sorted(available_tags)
}

# If metadata available, count by (faithfulness, template)
if 'metadata_fields' in dataset['info'] and dataset['info']['metadata_fields']:
    from collections import defaultdict
    prompt_counts = defaultdict(lambda: defaultdict(int))

    for prompt_idx, prompt_data in dataset['data'].items():
        metadata = prompt_data.get('metadata', {})
        faithfulness = metadata.get('faithfulness_classification', 'unknown')
        template = metadata.get('hint_template', 'unknown')
        prompt_counts[faithfulness][template] += 1

    dataset_statistics['prompts_by_class_template'] = dict(prompt_counts)
    dataset_statistics['metadata_fields'] = dataset['info']['metadata_fields']

    print("\nDataset distribution by (faithfulness, template):")
    for faith in sorted(prompt_counts.keys()):
        for template in sorted(prompt_counts[faith].keys()):
            count = prompt_counts[faith][template]
            print(f"  ({faith}, {template}): {count} prompts")
else:
    dataset_statistics['metadata_fields'] = None

# Count activations per class for the selected tags
from src.separability import extract_tag_activations
pos_acts, neg_acts = extract_tag_activations(dataset, POSITIVE_TAGS, NEGATIVE_TAGS)

# Count total activations per layer (just layer 0 for summary)
if 0 in pos_acts and 0 in neg_acts:
    dataset_statistics['sample_counts_layer0'] = {
        'positive': pos_acts[0].shape[0],
        'negative': neg_acts[0].shape[0]
    }
    print(f"\nActivation counts at layer 0:")
    print(f"  Positive ({POSITIVE_TAGS}): {pos_acts[0].shape[0]}")
    print(f"  Negative ({NEGATIVE_TAGS}): {neg_acts[0].shape[0]}")

# STEP 2: Investigation 1 - Cosine Similarity Analysis
print("\n=== STEP 2: Cosine Similarity Analysis ===")
print("Computing cosine similarity between positive and negative means per layer...")

cosine_similarities = compute_cosine_similarity_by_layer(
    dataset=dataset,
    positive_tags=POSITIVE_TAGS,
    negative_tags=NEGATIVE_TAGS
)

print(f"Computed cosine similarities for {len(cosine_similarities)} layers")
print(f"Range: {min(cosine_similarities.values()):.3f} to {max(cosine_similarities.values()):.3f}")

# STEP 3: Investigation 2 - Mean Difference Analysis
print("\n=== STEP 3: Mean Difference Analysis ===")
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

# STEP 4: Investigation 3 - PCA Analysis
print("\n=== STEP 4: PCA Analysis ===")
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

# STEP 5: Results Summary
print("\n=== STEP 5: Results Summary ===")

print(f"\nCosine Similarity Summary:")
print(f"  Min: {min(cosine_similarities.values()):.3f} (layer {min(cosine_similarities, key=cosine_similarities.get)})")
print(f"  Max: {max(cosine_similarities.values()):.3f} (layer {max(cosine_similarities, key=cosine_similarities.get)})")
print(f"  Mean: {sum(cosine_similarities.values())/len(cosine_similarities):.3f}")

print(f"\nMean Difference Summary:")
print(f"  Min norm: {min(mean_diff_norms.values()):.3f} (layer {min(mean_diff_norms, key=mean_diff_norms.get)})")
print(f"  Max norm: {max(mean_diff_norms.values()):.3f} (layer {max(mean_diff_norms, key=mean_diff_norms.get)})")
print(f"  Mean norm: {sum(mean_diff_norms.values())/len(mean_diff_norms):.3f}")

print(f"\nPCA Analysis Summary:")
print(f"  Valid layers: {len(valid_pca)}/{len(pca_results)}")
if valid_pca:
    explained_variances = [sum(v['explained_variance'][:2]) for v in valid_pca.values()]
    print(f"  PC1+PC2 explained variance range: {min(explained_variances):.3f} to {max(explained_variances):.3f}")

# STEP 6: Save Results
if SAVE_RESULTS:
    print("\n=== STEP 6: Saving Results ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        'dataset_file': INPUT_FILE,
        'configuration': {
            'positive_tags': POSITIVE_TAGS,
            'negative_tags': NEGATIVE_TAGS,
            'layers_analyzed': LAYERS_TO_ANALYZE
        },
        'dataset_info': dataset['info'],
        'dataset_statistics': dataset_statistics,
        'results': {
            'cosine_similarities': cosine_similarities,
            'mean_differences': mean_differences,
            'pca_analysis': pca_results_serializable
        },
        'processing_time_seconds': processing_time
    }

    # Save results
    # Save summary to summaries directory
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {SUMMARY_FILE}")

    # Save individual analysis files for easier access
    cosine_file = os.path.join(OUTPUT_DIR, f"cosine_similarities_{TODAY}.json")
    with open(cosine_file, 'w') as f:
        json.dump(cosine_similarities, f, indent=2)

    means_file = os.path.join(OUTPUT_DIR, f"mean_differences_{TODAY}.json")
    with open(means_file, 'w') as f:
        json.dump(mean_differences, f, indent=2)

    pca_file = os.path.join(OUTPUT_DIR, f"pca_analysis_{TODAY}.json")
    with open(pca_file, 'w') as f:
        json.dump(pca_results_serializable, f, indent=2)

    print(f"Individual analysis files saved to {OUTPUT_DIR}")

# STEP 7: Create Visualizations
if CREATE_PLOTS:
    print("\n=== STEP 7: Creating Visualizations ===")

    plot_dir = PLOTS_DIR
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

    print(f"All plots saved to {plot_dir}")

# STEP 8: Analysis Complete
end_time = time.time()
processing_time = end_time - start_time

print(f"\n=== SEPARABILITY ANALYSIS COMPLETE ===")
print(f"Processing time: {processing_time/60:.1f} minutes")
print(f"✅ Cosine similarity analysis: {len(cosine_similarities)} layers")
print(f"✅ Mean difference analysis: {len(mean_differences)} layers")
print(f"✅ PCA analysis: {len(valid_pca)} valid layers")

if SAVE_RESULTS:
    print(f"✅ Results saved to: {OUTPUT_DIR}")

if CREATE_PLOTS:
    print(f"✅ Plots saved to: {PLOTS_DIR}")

print(f"\nReady for Step 4b: train probes (see train_probes.py)")
print(f"\nReady for Step 5: compute steering vectors")
print(f"Use separability results: {SUMMARY_FILE if SAVE_RESULTS else 'results in memory'}")