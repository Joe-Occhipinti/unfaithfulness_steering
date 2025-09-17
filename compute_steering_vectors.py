"""
compute_steering_vectors.py

Computes steering vectors from the contrastive dataset by taking the mean difference
between specified positive and negative activation groups across specified layers.
"""

import torch
import pickle
import os
from typing import List, Dict, Tuple, Optional
import argparse

# === CONFIGURATION ===

# Default input file (contrastive dataset)
DEFAULT_INPUT_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint_2.2_contrastive_dataset_train_val_strong_faithful_unfaithful_2025-09-14.pkl"

# Default output file for steering vectors
DEFAULT_OUTPUT_FILE = "steering_vectors_body_and_final_faithful_vs_unfaithful_2025-09-15.pkl"

# Available tags in the dataset
AVAILABLE_TAGS = ["F", "U", "F_final", "U_final"]

# Predefined label combinations
LABEL_COMBINATIONS = {
    "all_faithful_vs_unfaithful": {
        "positive": ["F_final", "F"],
        "negative": ["U_final", "U"],
        "description": "All strongly body and final faithful vs all strongly body unfaithful"
    },
    "final_only": {
        "positive": ["F_final"],
        "negative": ["U_final"], 
        "description": "Strong faithful vs strong unfaithful"
    },
    "weak_only": {
        "positive": ["F_wk"],
        "negative": ["U_wk"],
        "description": "Weak faithful vs weak unfaithful"
    },
    "strong_vs_weak_faithful": {
        "positive": ["F_str"],
        "negative": ["F_wk"],
        "description": "Strong faithful vs weak faithful"
    },
    "strong_vs_weak_unfaithful": {
        "positive": ["U_str"],
        "negative": ["U_wk"],
        "description": "Strong unfaithful vs weak unfaithful"
    }
}

def load_contrastive_dataset(input_file: str) -> Dict:
    """Load the contrastive dataset from pickle file."""
    print(f"Loading contrastive dataset from: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset loaded successfully!")
    print(f"Available tags: {dataset['info']['tags']}")
    print(f"Number of layers: {dataset['info']['num_layers']}")
    
    return dataset

def get_combined_activations(data: Dict, layer_idx: int, tags: List[str], split: str = "train") -> torch.Tensor:
    """
    Combine activations from multiple tags for a specific layer from the specified split.

    Args:
        data: Dataset data dictionary
        layer_idx: Layer index
        tags: List of tags to combine
        split: Which data split to use ("train" or "val")

    Returns:
        Combined tensor of activations [total_samples, hidden_dim]
    """
    activations_list = []

    # Access the specified split (train or val)
    split_data = data.get(split, data)  # Fallback to data if no splits exist

    for tag in tags:
        if layer_idx in split_data and tag in split_data[layer_idx] and split_data[layer_idx][tag].numel() > 0:
            activations_list.append(split_data[layer_idx][tag])

    if not activations_list:
        # Return empty tensor with correct hidden dimension
        return torch.empty(0, 4096)

    # Concatenate all activations
    combined = torch.cat(activations_list, dim=0)
    return combined

def compute_steering_vector(positive_acts: torch.Tensor, negative_acts: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Compute steering vector as mean(positive) - mean(negative).
    
    Returns:
        steering_vector: The computed steering vector
        stats: Statistics about the computation
    """
    stats = {
        "positive_samples": positive_acts.shape[0] if positive_acts.numel() > 0 else 0,
        "negative_samples": negative_acts.shape[0] if negative_acts.numel() > 0 else 0,
    }
    
    if stats["positive_samples"] == 0:
        raise ValueError("No positive samples found for steering vector computation")
    if stats["negative_samples"] == 0:
        raise ValueError("No negative samples found for steering vector computation")
    
    # Compute means
    positive_mean = positive_acts.mean(dim=0)  # [hidden_dim]
    negative_mean = negative_acts.mean(dim=0)  # [hidden_dim]
    
    # Steering vector: positive - negative
    steering_vector = positive_mean - negative_mean
    
    # Additional statistics
    stats["positive_mean_norm"] = positive_mean.norm().item()
    stats["negative_mean_norm"] = negative_mean.norm().item()
    stats["steering_vector_norm"] = steering_vector.norm().item()
    
    return steering_vector, stats

def compute_steering_vectors_for_layers(
    dataset: Dict,
    positive_tags: List[str],
    negative_tags: List[str],
    layers: Optional[List[int]] = None,
    split: str = "train"
) -> Tuple[Dict[int, torch.Tensor], Dict]:
    """
    Compute steering vectors for specified layers and tag combinations.

    Args:
        dataset: The loaded contrastive dataset
        positive_tags: Tags to use as positive examples
        negative_tags: Tags to use as negative examples
        layers: List of layer indices to compute for (None = all layers)
        split: Which data split to use for computation ("train" or "val")

    Returns:
        steering_vectors: Dict mapping layer_idx -> steering_vector
        computation_stats: Statistics about the computation
    """
    data = dataset['data']
    info = dataset['info']
    
    # Default to all layers if not specified
    if layers is None:
        layers = list(range(info['num_layers']))
    
    print(f"\nComputing steering vectors:")
    print(f"  Data split: {split} (IMPORTANT: Using only {split} data to avoid data leakage)")
    print(f"  Positive tags: {positive_tags}")
    print(f"  Negative tags: {negative_tags}")
    print(f"  Layers: {len(layers)} layers {layers[:5]}{'...' if len(layers) > 5 else ''}")
    
    steering_vectors = {}
    layer_stats = {}
    
    for layer_idx in layers:
        # Get combined activations for positive and negative tags from specified split
        positive_acts = get_combined_activations(data, layer_idx, positive_tags, split)
        negative_acts = get_combined_activations(data, layer_idx, negative_tags, split)
        
        try:
            steering_vec, stats = compute_steering_vector(positive_acts, negative_acts)
            steering_vectors[layer_idx] = steering_vec
            layer_stats[layer_idx] = stats
            
            # Print progress for first few layers
            if layer_idx < 3 or layer_idx % 10 == 0:
                print(f"  Layer {layer_idx}: pos={stats['positive_samples']}, neg={stats['negative_samples']}, "
                      f"norm={stats['steering_vector_norm']:.4f}")
                
        except ValueError as e:
            print(f"  Warning: Skipping layer {layer_idx}: {e}")
            continue
    
    computation_stats = {
        "positive_tags": positive_tags,
        "negative_tags": negative_tags,
        "data_split": split,
        "layers_computed": list(steering_vectors.keys()),
        "layers_requested": layers,
        "layer_stats": layer_stats
    }
    
    print(f"\nSuccessfully computed steering vectors for {len(steering_vectors)} layers")
    
    return steering_vectors, computation_stats

def save_steering_vectors(steering_vectors: Dict, stats: Dict, output_file: str):
    """Save steering vectors and statistics to pickle file."""
    
    output_data = {
        "steering_vectors": steering_vectors,
        "computation_stats": stats,
        "metadata": {
            "description": "Steering vectors computed from contrastive activations",
            "data_split_used": stats["data_split"],
            "positive_tags": stats["positive_tags"],
            "negative_tags": stats["negative_tags"],
            "layers": stats["layers_computed"],
            "vector_dim": list(steering_vectors.values())[0].shape[0] if steering_vectors else 0
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Steering vectors saved to: {output_file}")

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description="Compute steering vectors from contrastive dataset")
    parser.add_argument('--input_file', type=str, default=DEFAULT_INPUT_FILE,
                        help='Input contrastive dataset file')
    parser.add_argument('--output_file', type=str, default=DEFAULT_OUTPUT_FILE,
                        help='Output steering vectors file')
    parser.add_argument('--combination', type=str, default='all_faithful_vs_unfaithful',
                        choices=list(LABEL_COMBINATIONS.keys()),
                        help='Predefined label combination to use')
    parser.add_argument('--positive_tags', type=str, nargs='+', default=None,
                        help='Custom positive tags (overrides --combination)')
    parser.add_argument('--negative_tags', type=str, nargs='+', default=None, 
                        help='Custom negative tags (overrides --combination)')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers to compute (default: all layers)')
    parser.add_argument('--list_combinations', action='store_true',
                        help='List available label combinations and exit')
    
    args = parser.parse_args()
    
    # List combinations if requested
    if args.list_combinations:
        print("Available label combinations:")
        for name, combo in LABEL_COMBINATIONS.items():
            print(f"  {name}: {combo['description']}")
            print(f"    Positive: {combo['positive']}")
            print(f"    Negative: {combo['negative']}")
        return
    
    # Determine tags to use
    if args.positive_tags is not None and args.negative_tags is not None:
        positive_tags = args.positive_tags
        negative_tags = args.negative_tags
        print(f"Using custom tag combination:")
    else:
        combo = LABEL_COMBINATIONS[args.combination]
        positive_tags = combo["positive"]
        negative_tags = combo["negative"]
        print(f"Using predefined combination '{args.combination}': {combo['description']}")
    
    # Load dataset
    dataset = load_contrastive_dataset(args.input_file)
    
    # Compute steering vectors using only TRAIN data
    steering_vectors, stats = compute_steering_vectors_for_layers(
        dataset, positive_tags, negative_tags, args.layers, split="train"
    )
    
    # Save results
    save_steering_vectors(steering_vectors, stats, args.output_file)
    
    # Print summary
    print(f"\n--- Summary ---")
    print(f"Computed steering vectors for {len(steering_vectors)} layers")
    print(f"Vector dimension: {list(steering_vectors.values())[0].shape[0]}")
    print(f"Average vector norm: {sum(v.norm().item() for v in steering_vectors.values()) / len(steering_vectors):.4f}")


if __name__ == "__main__":
    main()

# === EXAMPLE USAGE ===
"""
# Command line examples:

# Default: All faithful vs unfaithful, all layers
python compute_steering_vectors.py

# Strong only comparison
python compute_steering_vectors.py --combination strong_only

# Custom tags
python compute_steering_vectors.py --positive_tags F_str --negative_tags U_str U_wk

# Specific layers
python compute_steering_vectors.py --layers 10 15 20 25 30

# List available combinations
python compute_steering_vectors.py --list_combinations

# Programmatic usage:
dataset = load_contrastive_dataset("dataset.pkl")
vectors, stats = compute_steering_vectors_for_layers(
    dataset, 
    positive_tags=["F_str", "F_wk"], 
    negative_tags=["U_str", "U_wk"],
    layers=[15, 20, 25]
)
"""