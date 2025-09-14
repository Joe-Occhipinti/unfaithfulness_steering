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
DEFAULT_DATASET_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\datasets\sprint_2_contrastive_dataset_full_sweep_all_(un)faithful_tags_mmlu_psychology_train_2025-09-04.pkl"

def load_dataset(file_path: str) -> Dict:
    """Load the contrastive dataset."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def test_linear_separability(positive_acts: torch.Tensor, negative_acts: torch.Tensor, layer_idx: int) -> Dict:
    """
    Test if positive and negative activations are linearly separable.
    
    Returns:
        Dictionary with separability metrics
    """
    if positive_acts.shape[0] == 0 or negative_acts.shape[0] == 0:
        return {"error": "Empty activation tensors"}
    
    # Combine data and create labels (convert bfloat16 to float32)
    X = torch.cat([positive_acts.float(), negative_acts.float()], dim=0).numpy()
    y = torch.cat([
        torch.zeros(positive_acts.shape[0]),  # 0 for positive (faithful)
        torch.ones(negative_acts.shape[0])    # 1 for negative (unfaithful)
    ]).numpy()
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train linear classifier
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, classifier.predict(X_train))
    test_acc = accuracy_score(y_test, classifier.predict(X_test))
    
    # Get decision boundary properties
    weights = classifier.coef_[0]
    weight_norm = np.linalg.norm(weights)
    
    # Compute projection scores for visualization
    positive_scores = positive_acts.float().numpy() @ weights
    negative_scores = negative_acts.float().numpy() @ weights
    
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
        "test_accuracy": test_acc,
        "weight_norm": weight_norm,
        "separation_distance": separation_distance,
        "cohens_d": cohens_d,
        "positive_mean_score": positive_mean_score,
        "negative_mean_score": negative_mean_score,
        "positive_std": positive_std,
        "negative_std": negative_std,
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "weights": weights
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
              f'Test Accuracy: {separability_results["test_accuracy"]:.3f}, '
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
    Perform comprehensive linear separability analysis across layers.
    """
    data = dataset['data']
    num_layers = dataset['info']['num_layers']
    
    if test_layers is None:
        # Test key layers: early, middle, late, and best layers from previous analysis
        test_layers = [0, 5, 10, 15, 20, 25, 28, 29, 30, 31]
    
    results = {}
    
    print(f"Testing linear separability for layers: {test_layers}")
    print(f"Positive tags: {positive_tags}, Negative tags: {negative_tags}")
    
    for layer_idx in test_layers:
        print(f"\\nAnalyzing Layer {layer_idx}...")
        
        # Get activations
        positive_acts = []
        negative_acts = []
        
        for tag in positive_tags:
            if tag in data[layer_idx] and data[layer_idx][tag].numel() > 0:
                positive_acts.append(data[layer_idx][tag])
        
        for tag in negative_tags:
            if tag in data[layer_idx] and data[layer_idx][tag].numel() > 0:
                negative_acts.append(data[layer_idx][tag])
        
        if not positive_acts or not negative_acts:
            continue
            
        positive_combined = torch.cat(positive_acts, dim=0)
        negative_combined = torch.cat(negative_acts, dim=0)
        
        # Test linear separability
        sep_results = test_linear_separability(positive_combined, negative_combined, layer_idx)
        results[layer_idx] = sep_results
        
        print(f"  Linear classifier accuracy: {sep_results['test_accuracy']:.3f}")
        print(f"  Cohen's d (effect size): {sep_results['cohens_d']:.3f}")
        
        # Create visualizations for key layers
        if layer_idx in [15, 25, 31]:  # Visualize middle, late, and best layer
            print(f"  Creating visualizations for layer {layer_idx}...")
            
            # PCA visualization  
            pca_path = f"linear_separability_pca_layer_{layer_idx}.png"
            variance_ratios = visualize_pca_separation(positive_combined, negative_combined, layer_idx, pca_path)
            
            # Projection distribution
            proj_path = f"linear_separability_projection_layer_{layer_idx}.png" 
            visualize_projection_distributions(sep_results, proj_path)
    
    return results

def plot_linearity_summary(results: Dict, save_path: str = None):
    """Plot summary of linear separability across layers."""
    
    layers = sorted(results.keys())
    accuracies = [results[l]['test_accuracy'] for l in layers]
    cohens_d = [results[l]['cohens_d'] for l in layers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    ax1.plot(layers, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random chance')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Linear Classifier Test Accuracy')
    ax1.set_title('Linear Separability by Layer')
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
    best_acc_layer = layers[np.argmax(accuracies)]
    best_effect_layer = layers[np.argmax(cohens_d)]
    
    print(f"Best linear accuracy: Layer {best_acc_layer} ({max(accuracies):.3f})")
    print(f"Largest effect size: Layer {best_effect_layer} ({max(cohens_d):.3f})")
    
    # Interpretation
    max_acc = max(accuracies)
    max_cohens = max(cohens_d)
    
    if max_acc > 0.8 and max_cohens > 0.5:
        print("✅ CONCLUSION: Faithfulness appears to be LINEARLY ENCODED")
        print("   High accuracy and medium-to-large effect sizes indicate clear linear separability")
    elif max_acc > 0.7 and max_cohens > 0.3:
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
    
    # Test linear separability
    results = comprehensive_linearity_analysis(
        dataset, 
        positive_tags=["F_str", "F_wk"], 
        negative_tags=["U_str", "U_wk"],
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