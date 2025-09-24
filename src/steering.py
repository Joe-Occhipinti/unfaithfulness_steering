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


# =============================================================================
# STEERING APPLICATION FUNCTIONS
# =============================================================================

import gc
from typing import Any
from tqdm import tqdm


class LayerSteeringWrapper(torch.nn.Module):
    """
    Wrapper for model layers to apply steering vectors during generation.
    Adds steering vector to the last token at each forward pass.
    """

    def __init__(self, block: torch.nn.Module, layer_idx: int):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.steering_vector = None
        self.coefficient = 0.0
        self.active = False

    def forward(self, *args, **kwargs):
        """Forward pass with optional steering applied to last token."""
        output = self.block(*args, **kwargs)

        if self.active and self.steering_vector is not None and output is not None:
            try:
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_dim]
                    batch_size = hidden_states.shape[0]
                    modified_hidden_states = hidden_states.clone()

                    # Add steering to last token
                    steering_addition = self.steering_vector * self.coefficient
                    modified_hidden_states[:, -1, :] = (
                        modified_hidden_states[:, -1, :] +
                        steering_addition.unsqueeze(0).expand(batch_size, -1)
                    )

                    # Return in the same format as input
                    if isinstance(output, tuple):
                        output = (modified_hidden_states,) + output[1:]
                    else:
                        output = modified_hidden_states

            except Exception as e:
                print(f"Error in layer {self.layer_idx} steering: {e}")

        return output

    def set_steering(self, vector: torch.Tensor, coefficient: float):
        """Set steering vector and coefficient for this layer."""
        self.steering_vector = vector.to(self.block.weight.device if hasattr(self.block, 'weight') else 'cuda')
        self.coefficient = coefficient
        self.active = True

    def reset(self):
        """Reset steering to inactive state."""
        self.active = False
        self.steering_vector = None
        self.coefficient = 0.0


def apply_steering_to_model(
    model: Any,
    steering_vectors: Dict[int, torch.Tensor],
    layers_to_wrap: Optional[List[int]] = None
) -> Dict[int, LayerSteeringWrapper]:
    """
    Apply steering wrappers to specified model layers.

    Args:
        model: The HuggingFace model
        steering_vectors: Dict of layer_idx -> steering_vector
        layers_to_wrap: List of layer indices to wrap (None = all layers with vectors)

    Returns:
        Dict of layer_idx -> wrapper instance
    """
    if layers_to_wrap is None:
        layers_to_wrap = list(steering_vectors.keys())

    wrapped_layers = {}

    for layer_idx in layers_to_wrap:
        if layer_idx not in steering_vectors:
            print(f"Warning: No steering vector for layer {layer_idx}, skipping")
            continue

        original_layer = model.model.layers[layer_idx]
        wrapped_layer = LayerSteeringWrapper(original_layer, layer_idx)
        model.model.layers[layer_idx] = wrapped_layer
        wrapped_layers[layer_idx] = wrapped_layer

    print(f"Applied steering wrappers to {len(wrapped_layers)} layers")
    return wrapped_layers


def generate_steered_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    wrapped_layers: Dict[int, LayerSteeringWrapper],
    layer_idx: int,
    coefficient: float,
    steering_vectors: Dict[int, torch.Tensor],
    batch_size: int = 5,
    max_new_tokens: int = 2048,
    max_input_length: int = 1024
) -> List[str]:
    """
    Generate steered responses for a batch of prompts.

    Args:
        model: The model with steering wrappers applied
        tokenizer: Model tokenizer
        prompts: List of input prompts
        wrapped_layers: Dict of wrapped layer instances
        layer_idx: Layer to apply steering to
        coefficient: Steering coefficient strength
        steering_vectors: Dict of steering vectors
        batch_size: Batch size for generation
        max_new_tokens: Max tokens to generate
        max_input_length: Max input sequence length

    Returns:
        List of generated responses
    """
    # Reset all wrappers
    for wrapper in wrapped_layers.values():
        wrapper.reset()

    # Set up steering for target layer
    if layer_idx in wrapped_layers and layer_idx in steering_vectors:
        wrapped_layers[layer_idx].set_steering(
            steering_vectors[layer_idx],
            coefficient
        )

    all_responses = []

    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating (layer={layer_idx}, coeff={coefficient:.1f})"):
        batch_prompts = prompts[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate with steering
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode responses (skip input portion)
        input_length = inputs['input_ids'].shape[1]
        batch_responses = tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True
        )

        all_responses.extend([resp.strip() for resp in batch_responses])

        # Memory cleanup
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

    # Reset wrapper after generation
    if layer_idx in wrapped_layers:
        wrapped_layers[layer_idx].reset()

    return all_responses


def sweep_coefficients(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    steering_vectors: Dict[int, torch.Tensor],
    layers_to_test: List[int],
    coefficients: List[float],
    batch_size: int = 5,
    max_new_tokens: int = 2048
) -> Dict[Tuple[int, float], List[str]]:
    """
    Sweep through different layers and coefficients to find optimal steering.

    Args:
        model: The model
        tokenizer: Model tokenizer
        prompts: Input prompts to test
        steering_vectors: Dict of steering vectors
        layers_to_test: List of layer indices to test
        coefficients: List of coefficient values to test
        batch_size: Batch size for generation
        max_new_tokens: Max tokens to generate

    Returns:
        Dict mapping (layer_idx, coefficient) -> list of generated responses
    """
    print(f"\n=== COEFFICIENT SWEEP ===")
    print(f"Testing {len(layers_to_test)} layers with {len(coefficients)} coefficients")
    print(f"Total configurations: {len(layers_to_test) * len(coefficients)}")

    # Apply steering wrappers to model
    wrapped_layers = apply_steering_to_model(model, steering_vectors, layers_to_test)

    results = {}

    for layer_idx in layers_to_test:
        for coeff in coefficients:
            print(f"\nTesting layer {layer_idx} with coefficient {coeff:.2f}")

            responses = generate_steered_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                wrapped_layers=wrapped_layers,
                layer_idx=layer_idx,
                coefficient=coeff,
                steering_vectors=steering_vectors,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens
            )

            results[(layer_idx, coeff)] = responses

            # Memory cleanup between sweeps
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nCompleted coefficient sweep: {len(results)} configurations tested")
    return results