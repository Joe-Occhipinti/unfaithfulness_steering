"""
test_linear_separability.py

Tests whether faithfulness is linearly encoded by analyzing if positive and negative
activations can be separated by a linear hyperplane. High cosine similarity might
indicate non-linear encoding or that the separation is more complex than a simple direction.
"""

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import seaborn as sns

# === CONFIGURATION ===
DEFAULT_DATASET_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint_2.2_contrastive_dataset_train_val_strong_faithful_unfaithful_2025-09-14.pkl"

def load_dataset(file_path: str) -> Dict:
    """Load the contrastive dataset."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def test_linear_separability(train_positive_acts: torch.Tensor, train_negative_acts: torch.Tensor,
                            val_positive_acts: torch.Tensor, val_negative_acts: torch.Tensor,
                            layer_idx: int) -> Dict:
    """
    Test if positive and negative activations are linearly separable using proper train/val splits.

    Args:
        train_positive_acts: Training positive activations
        train_negative_acts: Training negative activations
        val_positive_acts: Validation positive activations
        val_negative_acts: Validation negative activations
        layer_idx: Layer index for reference

    Returns:
        Dictionary with separability metrics
    """
    if (train_positive_acts.shape[0] == 0 or train_negative_acts.shape[0] == 0 or
        val_positive_acts.shape[0] == 0 or val_negative_acts.shape[0] == 0):
        return {"error": "Empty activation tensors"}

    # Prepare TRAINING data (convert bfloat16 to float32)
    X_train = torch.cat([train_positive_acts.float(), train_negative_acts.float()], dim=0).numpy()
    y_train = torch.cat([
        torch.zeros(train_positive_acts.shape[0]),  # 0 for positive (faithful)
        torch.ones(train_negative_acts.shape[0])    # 1 for negative (unfaithful)
    ]).numpy()

    # Prepare VALIDATION data
    X_val = torch.cat([val_positive_acts.float(), val_negative_acts.float()], dim=0).numpy()
    y_val = torch.cat([
        torch.zeros(val_positive_acts.shape[0]),    # 0 for positive (faithful)
        torch.ones(val_negative_acts.shape[0])      # 1 for negative (unfaithful)
    ]).numpy()

    # Train linear classifier on TRAIN data only
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_train)

    # Evaluate on both train and val
    train_acc = accuracy_score(y_train, classifier.predict(X_train))
    val_acc = accuracy_score(y_val, classifier.predict(X_val))  # This is the key metric!

    # Get decision boundary properties
    weights = classifier.coef_[0]
    weight_norm = np.linalg.norm(weights)

    # Compute projection scores for ALL data (for visualization)
    all_positive_acts = torch.cat([train_positive_acts, val_positive_acts], dim=0)
    all_negative_acts = torch.cat([train_negative_acts, val_negative_acts], dim=0)

    positive_scores = all_positive_acts.float().numpy() @ weights
    negative_scores = all_negative_acts.float().numpy() @ weights

    # Compute separation metrics
    positive_mean_score = np.mean(positive_scores)
    negative_mean_score = np.mean(negative_scores)
    separation_distance = abs(positive_mean_score - negative_mean_score)

    # Overlap analysis
    positive_std = np.std(positive_scores)
    negative_std = np.std(negative_scores)
    pooled_std = np.sqrt((positive_std**2 + negative_std**2) / 2)
    cohens_d = separation_distance / pooled_std if pooled_std > 0 else 0

    return {
        "layer": layer_idx,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,  # Renamed from test_accuracy for clarity
        "weight_norm": weight_norm,
        "separation_distance": separation_distance,
        "cohens_d": cohens_d,
        "positive_mean_score": positive_mean_score,
        "negative_mean_score": negative_mean_score,
        "positive_std": positive_std,
        "negative_std": negative_std,
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "weights": weights,
        "train_samples": len(X_train),
        "val_samples": len(X_val)
    }

def visualize_pca_separation(positive_acts: torch.Tensor, negative_acts: torch.Tensor, layer_idx: int, save_path: str = None):
    """Visualize the separation using PCA."""
    
    # Combine data (convert bfloat16 to float32)
    X = torch.cat([positive_acts.float(), negative_acts.float()], dim=0).numpy()
    labels = ['Faithful'] * positive_acts.shape[0] + ['Unfaithful'] * negative_acts.shape[0]
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    faithful_mask = np.array(labels) == 'Faithful'
    unfaithful_mask = np.array(labels) == 'Unfaithful'
    
    plt.scatter(X_pca[faithful_mask, 0], X_pca[faithful_mask, 1], 
                alpha=0.6, label='Faithful', color='green', s=50)
    plt.scatter(X_pca[unfaithful_mask, 0], X_pca[unfaithful_mask, 1], 
                alpha=0.6, label='Unfaithful', color='red', s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'PCA Visualization - Layer {layer_idx}\\nLinear Separability Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    plt.text(0.02, 0.98, f'First 2 PCs explain {total_variance:.1%} of variance', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return pca.explained_variance_ratio_

def visualize_projection_distributions(separability_results: Dict, save_path: str = None):
    """Visualize the distribution of projections onto the linear separator."""
    
    positive_scores = separability_results['positive_scores']
    negative_scores = separability_results['negative_scores']
    layer_idx = separability_results['layer']
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.hist(positive_scores, bins=30, alpha=0.6, label='Faithful', color='green', density=True)
    plt.hist(negative_scores, bins=30, alpha=0.6, label='Unfaithful', color='red', density=True)
    
    # Add mean lines
    pos_mean = separability_results['positive_mean_score']
    neg_mean = separability_results['negative_mean_score']
    plt.axvline(pos_mean, color='darkgreen', linestyle='--', linewidth=2, label=f'Faithful Mean: {pos_mean:.3f}')
    plt.axvline(neg_mean, color='darkred', linestyle='--', linewidth=2, label=f'Unfaithful Mean: {neg_mean:.3f}')
    
    plt.xlabel('Projection Score (Linear Separator Direction)')
    plt.ylabel('Density')
    plt.title(f'Linear Separability Analysis - Layer {layer_idx}\n'
              f'Val Accuracy: {separability_results["val_accuracy"]:.3f}, '
              f"Cohen's d: {separability_results['cohens_d']:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add interpretation text
    cohens_d = separability_results["cohens_d"]
    if cohens_d < 0.2:
        interpretation = "Negligible separation"
    elif cohens_d < 0.5:
        interpretation = "Small separation" 
    elif cohens_d < 0.8:
        interpretation = "Medium separation"
    else:
        interpretation = "Large separation"
    
    plt.text(0.02, 0.98, f'Effect size: {interpretation}\nCohen\'s d = {cohens_d:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def comprehensive_linearity_analysis(dataset: Dict, positive_tags: List[str], negative_tags: List[str],
                                   test_layers: List[int] = None) -> Dict:
    """
    Perform comprehensive linear separability analysis across layers using proper train/val splits.
    """
    data = dataset['data']
    num_layers = dataset['info']['num_layers']

    if test_layers is None:
        # Test key layers: early, middle, late, and best layers from previous analysis
        test_layers = [0, 5, 10, 15, 20, 25, 28, 29, 30, 31]

    results = {}

    print(f"Testing linear separability for layers: {test_layers}")
    print(f"Positive tags: {positive_tags}, Negative tags: {negative_tags}")
    print(f"IMPORTANT: Training on TRAIN split, evaluating on VAL split (proper generalization test)")

    for layer_idx in test_layers:
        print(f"\\nAnalyzing Layer {layer_idx}...")

        # Get TRAIN activations
        train_positive_acts = []
        train_negative_acts = []

        for tag in positive_tags:
            if tag in data['train'][layer_idx] and data['train'][layer_idx][tag].numel() > 0:
                train_positive_acts.append(data['train'][layer_idx][tag])

        for tag in negative_tags:
            if tag in data['train'][layer_idx] and data['train'][layer_idx][tag].numel() > 0:
                train_negative_acts.append(data['train'][layer_idx][tag])

        # Get VAL activations
        val_positive_acts = []
        val_negative_acts = []

        for tag in positive_tags:
            if tag in data['val'][layer_idx] and data['val'][layer_idx][tag].numel() > 0:
                val_positive_acts.append(data['val'][layer_idx][tag])

        for tag in negative_tags:
            if tag in data['val'][layer_idx] and data['val'][layer_idx][tag].numel() > 0:
                val_negative_acts.append(data['val'][layer_idx][tag])

        if not train_positive_acts or not train_negative_acts or not val_positive_acts or not val_negative_acts:
            print(f"  Skipping layer {layer_idx}: missing data in train or val split")
            continue

        train_positive_combined = torch.cat(train_positive_acts, dim=0)
        train_negative_combined = torch.cat(train_negative_acts, dim=0)
        val_positive_combined = torch.cat(val_positive_acts, dim=0)
        val_negative_combined = torch.cat(val_negative_acts, dim=0)

        # Test linear separability with proper train/val splits
        sep_results = test_linear_separability(
            train_positive_combined, train_negative_combined,
            val_positive_combined, val_negative_combined,
            layer_idx
        )
        results[layer_idx] = sep_results

        print(f"  Train accuracy: {sep_results['train_accuracy']:.3f}")
        print(f"  Val accuracy: {sep_results['val_accuracy']:.3f}")
        print(f"  Generalization gap: {sep_results['train_accuracy'] - sep_results['val_accuracy']:.3f}")
        print(f"  Cohen's d (effect size): {sep_results['cohens_d']:.3f}")

        # Create visualizations for key layers (use combined train+val for visualization)
        if layer_idx in [15, 25, 31]:  # Visualize middle, late, and best layer
            print(f"  Creating visualizations for layer {layer_idx}...")

            # Combine train and val for visualization
            all_positive = torch.cat([train_positive_combined, val_positive_combined], dim=0)
            all_negative = torch.cat([train_negative_combined, val_negative_combined], dim=0)

            # PCA visualization
            pca_path = f"linear_separability_pca_layer_{layer_idx}.png"
            variance_ratios = visualize_pca_separation(all_positive, all_negative, layer_idx, pca_path)

            # Projection distribution
            proj_path = f"linear_separability_projection_layer_{layer_idx}.png"
            visualize_projection_distributions(sep_results, proj_path)

    return results

def plot_linearity_summary(results: Dict, save_path: str = None):
    """Plot summary of linear separability across layers with train/val comparison."""

    layers = sorted(results.keys())
    train_accuracies = [results[l]['train_accuracy'] for l in layers]
    val_accuracies = [results[l]['val_accuracy'] for l in layers]
    cohens_d = [results[l]['cohens_d'] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy plot with train/val comparison
    ax1.plot(layers, train_accuracies, 'go-', linewidth=2, markersize=8, label='Train Accuracy')
    ax1.plot(layers, val_accuracies, 'bo-', linewidth=2, markersize=8, label='Val Accuracy (Key Metric)')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random chance')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Linear Classifier Accuracy')
    ax1.set_title('Linear Separability by Layer\n(Train vs Val - Generalization Test)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Cohen's d plot
    ax2.plot(layers, cohens_d, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small effect')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Medium effect')  
    ax2.axhline(y=0.8, color='purple', linestyle='--', alpha=0.7, label='Large effect')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel("Cohen's d (Effect Size)")
    ax2.set_title('Linear Separation Effect Size by Layer')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary
    print("\\n=== LINEAR SEPARABILITY SUMMARY ===")
    best_val_acc_layer = layers[np.argmax(val_accuracies)]
    best_train_acc_layer = layers[np.argmax(train_accuracies)]
    best_effect_layer = layers[np.argmax(cohens_d)]

    print(f"Best VAL accuracy: Layer {best_val_acc_layer} ({max(val_accuracies):.3f}) <- Key metric!")
    print(f"Best TRAIN accuracy: Layer {best_train_acc_layer} ({max(train_accuracies):.3f})")
    print(f"Largest effect size: Layer {best_effect_layer} ({max(cohens_d):.3f})")

    # Generalization analysis
    generalization_gaps = [train_accuracies[i] - val_accuracies[i] for i in range(len(layers))]
    avg_gap = np.mean(generalization_gaps)
    print(f"Average generalization gap (train - val): {avg_gap:.3f}")

    if avg_gap > 0.1:
        print("⚠️  WARNING: Large generalization gap suggests potential overfitting")
    else:
        print("✅ Good generalization: small gap between train and val accuracy")

    # Interpretation based on VAL accuracy (the proper metric)
    max_val_acc = max(val_accuracies)
    max_cohens = max(cohens_d)

    print(f"\\n=== INTERPRETATION (Based on VAL accuracy) ===")
    if max_val_acc > 0.8 and max_cohens > 0.5:
        print("✅ CONCLUSION: Faithfulness appears to be LINEARLY ENCODED")
        print("   High val accuracy and medium-to-large effect sizes indicate clear linear separability")
    elif max_val_acc > 0.7 and max_cohens > 0.3:
        print("⚠️  CONCLUSION: Faithfulness is PARTIALLY linearly encoded")
        print("   Moderate linear separability - steering may work but not optimally")
    else:
        print("❌ CONCLUSION: Faithfulness is likely NOT linearly encoded")
        print("   Poor linear separability - consider non-linear steering methods")

def main():
    """Main analysis function."""
    
    # Load dataset
    print("Loading contrastive dataset...")
    dataset = load_dataset(DEFAULT_DATASET_FILE)
    
    # Test linear separability with updated tags
    results = comprehensive_linearity_analysis(
        dataset,
        positive_tags=["F", "F_final"],
        negative_tags=["U", "U_final"],
        test_layers=[0, 5, 10, 15, 20, 25, 28, 29, 30, 31]
    )
    
    # Plot summary
    plot_linearity_summary(results, "linear_separability_summary.png")

if __name__ == "__main__":
    main()

# === INTERPRETATION GUIDE ===
"""
Linear Separability Indicators:

1. **Linear Classifier Accuracy**:
   - >0.8: Excellent linear separability
   - 0.7-0.8: Good linear separability  
   - 0.6-0.7: Moderate linear separability
   - <0.6: Poor linear separability

2. **Cohen's d (Effect Size)**:
   - >0.8: Large effect (strong linear separation)
   - 0.5-0.8: Medium effect 
   - 0.2-0.5: Small effect
   - <0.2: Negligible effect

3. **PCA Visualization**:
   - Clear clusters = linearly separable
   - Overlapping clouds = non-linear encoding
   - High variance in first 2 PCs = low-dimensional structure

4. **Projection Distributions**:
   - Non-overlapping histograms = excellent linear separation
   - Partially overlapping = moderate separation
   - Heavily overlapping = poor linear separation
"""