"""
src/steering.py

Core functions for computing steering vectors from activation datasets.

Provides modular, reusable functions for:
1. Loading activation datasets
2. Computing steering vectors as mean(positive) - mean(negative)
3. Layer-wise steering vector computation with statistics
4. Saving steering vector results
"""

import torch
import pickle
import os
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def load_activation_dataset(dataset_file: str) -> Dict:
    """
    Load activation dataset from pickle file.

    Args:
        dataset_file: Path to the activation dataset pickle file

    Returns:
        Dataset dictionary with structure:
        {
            'data': {layer_idx: {tag: tensor}},
            'info': {'num_layers', 'tags', etc}
        }
    """
    print(f"Loading activation dataset from: {dataset_file}")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Dataset loaded successfully!")
    print(f"Available tags: {dataset['info']['tags']}")
    print(f"Number of layers: {dataset['info']['num_layers']}")

    return dataset


def get_combined_activations(
    data: Dict,
    layer_idx: int,
    tags: List[str]
) -> torch.Tensor:
    """
    Combine activations from multiple tags for a specific layer.

    Args:
        data: Dataset data dictionary {layer_idx: {tag: tensor}}
        layer_idx: Layer index
        tags: List of tags to combine

    Returns:
        Combined tensor of activations [total_samples, hidden_dim]
    """
    activations_list = []

    for tag in tags:
        if (layer_idx in data and
            tag in data[layer_idx] and
            data[layer_idx][tag].numel() > 0):
            activations_list.append(data[layer_idx][tag])

    if not activations_list:
        # Return empty tensor with correct hidden dimension (DeepSeek default)
        return torch.empty(0, 4096)

    # Concatenate all activations
    combined = torch.cat(activations_list, dim=0)
    return combined


def compute_steering_vector(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute steering vector as mean(positive) - mean(negative).

    Args:
        positive_acts: Positive activations [num_positive, hidden_dim]
        negative_acts: Negative activations [num_negative, hidden_dim]

    Returns:
        steering_vector: The computed steering vector [hidden_dim]
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


def compute_steering_vectors_by_layer(
    dataset_splits: Dict,
    positive_tags: List[str],
    negative_tags: List[str],
    layers: Optional[List[int]] = None,
    split: str = "train"
) -> Tuple[Dict[int, torch.Tensor], Dict]:
    """
    Compute steering vectors for specified layers and tag combinations using dataset splits.

    Args:
        dataset_splits: The dataset splits (train/val/test)
        positive_tags: Tags to use as positive examples
        negative_tags: Tags to use as negative examples
        layers: List of layer indices to compute for (None = all layers)
        split: Which split to use for computation ("train", "val", or "test")

    Returns:
        steering_vectors: Dict mapping layer_idx -> steering_vector
        computation_stats: Statistics about the computation
    """
    # Import here to avoid circular imports
    from src.separability import extract_tag_activations

    # Extract activations for the specified split
    positive_by_layer, negative_by_layer = extract_tag_activations(
        dataset_splits, positive_tags, negative_tags, split
    )

    info = dataset_splits[split]['info']

    # Default to all layers if not specified
    if layers is None:
        layers = list(range(info['num_layers']))

    print(f"\nComputing steering vectors:")
    print(f"  Split used: {split}")
    print(f"  Positive tags: {positive_tags}")
    print(f"  Negative tags: {negative_tags}")
    print(f"  Layers: {len(layers)} layers {layers[:5]}{'...' if len(layers) > 5 else ''}")

    steering_vectors = {}
    layer_stats = {}

    for layer_idx in layers:
        # Get activations for this layer
        positive_acts = positive_by_layer.get(layer_idx, torch.empty(0, info['hidden_dim']))
        negative_acts = negative_by_layer.get(layer_idx, torch.empty(0, info['hidden_dim']))

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
        "split_used": split,
        "layers_computed": list(steering_vectors.keys()),
        "layers_requested": layers,
        "layer_stats": layer_stats
    }

    print(f"\nSuccessfully computed steering vectors for {len(steering_vectors)} layers")

    return steering_vectors, computation_stats


def save_steering_vectors(
    steering_vectors: Dict[int, torch.Tensor],
    computation_stats: Dict,
    output_file: str,
    dataset_info: Dict = None
):
    """
    Save steering vectors and statistics to pickle file.

    Args:
        steering_vectors: Dict mapping layer_idx -> steering_vector
        computation_stats: Statistics from computation
        output_file: Path to save the results
        dataset_info: Optional dataset info to include in metadata
    """
    output_data = {
        "steering_vectors": steering_vectors,
        "computation_stats": computation_stats,
        "metadata": {
            "description": "Steering vectors computed from activation dataset",
            "positive_tags": computation_stats["positive_tags"],
            "negative_tags": computation_stats["negative_tags"],
            "layers": computation_stats["layers_computed"],
            "vector_dim": list(steering_vectors.values())[0].shape[0] if steering_vectors else 0,
            "computation_date": datetime.now().isoformat()
        }
    }

    # Include dataset info if provided
    if dataset_info:
        output_data["dataset_info"] = dataset_info

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Steering vectors saved to: {output_file}")


def print_steering_summary(
    steering_vectors: Dict[int, torch.Tensor],
    computation_stats: Dict,
    positive_tags: List[str],
    negative_tags: List[str]
):
    """
    Print a comprehensive summary of steering vector computation results.

    Args:
        steering_vectors: Computed steering vectors by layer
        computation_stats: Statistics from computation
        positive_tags: Positive tags used
        negative_tags: Negative tags used
    """
    if not steering_vectors:
        print("\nNo steering vectors computed.")
        return

    print(f"\n=== STEERING VECTOR SUMMARY ===")
    print(f"Positive tags: {positive_tags}")
    print(f"Negative tags: {negative_tags}")
    print(f"Layers computed: {len(steering_vectors)}")
    print(f"Vector dimension: {list(steering_vectors.values())[0].shape[0]}")

    # Compute statistics across layers
    layer_stats = computation_stats["layer_stats"]

    # Sample statistics
    pos_samples = [stats["positive_samples"] for stats in layer_stats.values()]
    neg_samples = [stats["negative_samples"] for stats in layer_stats.values()]
    vector_norms = [stats["steering_vector_norm"] for stats in layer_stats.values()]

    print(f"\nSample counts per layer:")
    print(f"  Positive: min={min(pos_samples)}, max={max(pos_samples)}, avg={sum(pos_samples)/len(pos_samples):.1f}")
    print(f"  Negative: min={min(neg_samples)}, max={max(neg_samples)}, avg={sum(neg_samples)/len(neg_samples):.1f}")

    print(f"\nSteering vector norms:")
    print(f"  Min: {min(vector_norms):.4f}")
    print(f"  Max: {max(vector_norms):.4f}")
    print(f"  Mean: {sum(vector_norms)/len(vector_norms):.4f}")

    # Show top/bottom layers by norm
    sorted_layers = sorted(steering_vectors.items(), key=lambda x: x[1].norm().item())

    print(f"\nTop 5 layers by steering vector norm:")
    for layer_idx, vec in sorted_layers[-5:]:
        norm = vec.norm().item()
        pos_count = layer_stats[layer_idx]["positive_samples"]
        neg_count = layer_stats[layer_idx]["negative_samples"]
        print(f"  Layer {layer_idx}: norm={norm:.4f} (pos={pos_count}, neg={neg_count})")

    print(f"\nBottom 5 layers by steering vector norm:")
    for layer_idx, vec in sorted_layers[:5]:
        norm = vec.norm().item()
        pos_count = layer_stats[layer_idx]["positive_samples"]
        neg_count = layer_stats[layer_idx]["negative_samples"]
        print(f"  Layer {layer_idx}: norm={norm:.4f} (pos={pos_count}, neg={neg_count})")


def save_steering_summary_json(
    steering_vectors: Dict[int, torch.Tensor],
    computation_stats: Dict,
    output_file: str
):
    """
    Save a JSON summary of steering vector statistics.

    Args:
        steering_vectors: Computed steering vectors by layer
        computation_stats: Statistics from computation
        output_file: Path to save the JSON summary
    """
    layer_stats = computation_stats["layer_stats"]

    # Prepare serializable summary
    summary = {
        "computation_info": {
            "positive_tags": computation_stats["positive_tags"],
            "negative_tags": computation_stats["negative_tags"],
            "layers_computed": computation_stats["layers_computed"],
            "total_layers": len(steering_vectors)
        },
        "statistics": {
            "vector_dimension": list(steering_vectors.values())[0].shape[0] if steering_vectors else 0,
            "average_vector_norm": sum(v.norm().item() for v in steering_vectors.values()) / len(steering_vectors) if steering_vectors else 0,
            "min_vector_norm": min(v.norm().item() for v in steering_vectors.values()) if steering_vectors else 0,
            "max_vector_norm": max(v.norm().item() for v in steering_vectors.values()) if steering_vectors else 0,
        },
        "layer_details": {
            str(layer_idx): {
                "positive_samples": stats["positive_samples"],
                "negative_samples": stats["negative_samples"],
                "positive_mean_norm": stats["positive_mean_norm"],
                "negative_mean_norm": stats["negative_mean_norm"],
                "steering_vector_norm": stats["steering_vector_norm"]
            }
            for layer_idx, stats in layer_stats.items()
        }
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Steering summary saved to: {output_file}")