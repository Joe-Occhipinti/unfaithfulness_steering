"""
analyze_layer_separability.py

Analyzes and visualizes how well different layers separate faithful vs unfaithful
activations by computing and plotting steering vector norms across layers.
"""

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns

# === CONFIGURATION ===
DEFAULT_DATASET_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint_2.2_contrastive_dataset_train_val_strong_faithful_unfaithful_2025-09-14.pkl"
OUTPUT_PLOT = "layer_separability_analysis.png"

def load_dataset(file_path: str) -> Dict:
    """Load the contrastive dataset."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_layer_separability(dataset: Dict, positive_tags: List[str], negative_tags: List[str], split: str = "train") -> Dict:
    """
    Compute separability metrics for each layer using specified train/val split.

    Args:
        dataset: The contrastive dataset with train/val splits
        positive_tags: Tags for positive class (faithful)
        negative_tags: Tags for negative class (unfaithful)
        split: Which split to use ("train" or "val")

    Returns:
        Dict with layer_idx -> {
            'steering_norm': steering vector norm,
            'positive_samples': number of positive samples,
            'negative_samples': number of negative samples,
            'positive_mean_norm': norm of positive mean,
            'negative_mean_norm': norm of negative mean,
            'cosine_similarity': cosine similarity between positive and negative means
        }
    """
    data = dataset['data'][split]  # Use specified split (train or val)
    num_layers = dataset['info']['num_layers']

    results = {}

    for layer_idx in range(num_layers):
        # Collect activations for positive and negative tags
        positive_acts = []
        negative_acts = []

        for tag in positive_tags:
            if tag in data[layer_idx] and data[layer_idx][tag].numel() > 0:
                positive_acts.append(data[layer_idx][tag])

        for tag in negative_tags:
            if tag in data[layer_idx] and data[layer_idx][tag].numel() > 0:
                negative_acts.append(data[layer_idx][tag])
        
        if not positive_acts or not negative_acts:
            results[layer_idx] = {
                'steering_norm': 0.0,
                'positive_samples': 0,
                'negative_samples': 0,
                'positive_mean_norm': 0.0,
                'negative_mean_norm': 0.0,
                'cosine_similarity': 0.0
            }
            continue
        
        # Combine activations
        positive_combined = torch.cat(positive_acts, dim=0)
        negative_combined = torch.cat(negative_acts, dim=0)
        
        # Compute means
        positive_mean = positive_combined.mean(dim=0)
        negative_mean = negative_combined.mean(dim=0)
        
        # Compute steering vector
        steering_vector = positive_mean - negative_mean
        
        # Compute cosine similarity
        cos_sim = torch.cosine_similarity(positive_mean, negative_mean, dim=0).item()
        
        results[layer_idx] = {
            'steering_norm': steering_vector.norm().item(),
            'positive_samples': positive_combined.shape[0],
            'negative_samples': negative_combined.shape[0],
            'positive_mean_norm': positive_mean.norm().item(),
            'negative_mean_norm': negative_mean.norm().item(),
            'cosine_similarity': cos_sim
        }
    
    return results

def plot_separability_analysis(results: Dict, positive_tags: List[str], negative_tags: List[str], save_path: str = None):
    """Create comprehensive separability analysis plots."""
    
    layers = list(results.keys())
    steering_norms = [results[l]['steering_norm'] for l in layers]
    positive_norms = [results[l]['positive_mean_norm'] for l in layers]
    negative_norms = [results[l]['negative_mean_norm'] for l in layers]
    cosine_sims = [results[l]['cosine_similarity'] for l in layers]
    positive_samples = [results[l]['positive_samples'] for l in layers]
    negative_samples = [results[l]['negative_samples'] for l in layers]
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Layer-wise Separability Analysis\nPositive: {positive_tags}, Negative: {negative_tags}', fontsize=14)
    
    # 1. Steering Vector Norms (MAIN PLOT)
    axes[0, 0].plot(layers, steering_norms, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Steering Vector Norm')
    axes[0, 0].set_title('Faithfulness Separability by Layer')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight best layer
    best_layer = layers[np.argmax(steering_norms)]
    best_norm = max(steering_norms)
    axes[0, 0].scatter([best_layer], [best_norm], color='red', s=100, zorder=5)
    axes[0, 0].annotate(f'Best: Layer {best_layer}\nNorm: {best_norm:.3f}', 
                       xy=(best_layer, best_norm), xytext=(10, 10),
                       textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. Mean Activation Norms
    axes[0, 1].plot(layers, positive_norms, 'g-', label=f'Positive ({"+".join(positive_tags)})', marker='o', markersize=3)
    axes[0, 1].plot(layers, negative_norms, 'r-', label=f'Negative ({"+".join(negative_tags)})', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Mean Activation Norm')
    axes[0, 1].set_title('Mean Activation Magnitudes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cosine Similarity
    axes[1, 0].plot(layers, cosine_sims, 'purple', linewidth=2, marker='d', markersize=4)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_title('Cosine Similarity: Positive vs Negative Means\n(Lower = More Orthogonal = Better Separation)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sample Counts
    width = 0.35
    layer_positions = np.array(layers)
    axes[1, 1].bar(layer_positions - width/2, positive_samples, width, label=f'Positive ({"+".join(positive_tags)})', alpha=0.7, color='green')
    axes[1, 1].bar(layer_positions + width/2, negative_samples, width, label=f'Negative ({"+".join(negative_tags)})', alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Sample Counts by Layer')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== SEPARABILITY ANALYSIS SUMMARY ===")
    print(f"Best layer for faithfulness separation: Layer {best_layer} (norm: {best_norm:.4f})")
    print(f"Average steering norm across all layers: {np.mean(steering_norms):.4f}")
    print(f"Standard deviation of steering norms: {np.std(steering_norms):.4f}")
    
    # Top 5 layers
    sorted_layers = sorted(zip(layers, steering_norms), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 layers by separability:")
    for i, (layer, norm) in enumerate(sorted_layers[:5]):
        print(f"  {i+1}. Layer {layer}: {norm:.4f}")
    
    # Layer ranges analysis
    early_layers = [norm for layer, norm in zip(layers, steering_norms) if layer < 11]
    middle_layers = [norm for layer, norm in zip(layers, steering_norms) if 11 <= layer < 22]
    late_layers = [norm for layer, norm in zip(layers, steering_norms) if layer >= 22]
    
    print(f"\nLayer range analysis:")
    print(f"  Early layers (0-10): avg norm = {np.mean(early_layers):.4f}")
    print(f"  Middle layers (11-21): avg norm = {np.mean(middle_layers):.4f}")
    print(f"  Late layers (22-31): avg norm = {np.mean(late_layers):.4f}")

def main():
    """Main analysis function."""
    
    # Load dataset
    print("Loading contrastive dataset...")
    dataset = load_dataset(DEFAULT_DATASET_FILE)
    
    # Define comparisons to analyze
    comparisons = [
        {
            "name": "All Faithful vs Unfaithful",
            "positive": ["F", "F_final"],
            "negative": ["U", "U_final"]
        },
        {
            "name": "Base Tags Only",
            "positive": ["F"],
            "negative": ["U"]
        },
        {
            "name": "Final Tags Only",
            "positive": ["F_final"],
            "negative": ["U_final"]
        }
    ]
    
    # Analyze each comparison
    for i, comp in enumerate(comparisons):
        print(f"\n{'='*50}")
        print(f"ANALYSIS {i+1}: {comp['name']}")
        print(f"{'='*50}")
        
        results = compute_layer_separability(dataset, comp["positive"], comp["negative"], split="train")
        
        plot_name = f"separability_{comp['name'].lower().replace(' ', '_')}.png"
        plot_separability_analysis(results, comp["positive"], comp["negative"], plot_name)

if __name__ == "__main__":
    main()