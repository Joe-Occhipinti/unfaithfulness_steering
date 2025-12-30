"""
compute_steering_vectors_off_policy.py

Computes steering vectors from the off-policy activation dataset.
Uses contrastive learning: Vector = Mean(Faithful) - Mean(Unfaithful).

This script is a simplified version of compute_steering_vectors.py, adapted for
the simpler structure of the off-policy dataset (no config weighting, no domain grouping).
"""

import os
import pickle
import torch
import argparse
from tqdm import tqdm
from datetime import datetime

def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded {len(dataset['data'])} items.")
    return dataset

def compute_vectors(dataset):
    print("Computing steering vectors...")
    
    # Organize activations by layer and label
    # Structure: {layer_idx: {'faithful': [], 'unfaithful': []}}
    layer_activations = {}
    
    # Statistics
    stats = {
        'total_items': len(dataset['data']),
        'faithful_count': 0,
        'unfaithful_count': 0,
        'skipped_count': 0
    }
    
    # Iterate through data
    # Note: dataset['data'] is a dictionary where keys are indices
    data_items = dataset['data']
    if isinstance(data_items, dict):
        data_items = data_items.values()
        
    for item in tqdm(data_items, desc="Processing items"):
        metadata = item.get('metadata', {})
        label = metadata.get('label')
        layers = item.get('layers', {})
        
        if label not in ['faithful', 'unfaithful']:
            stats['skipped_count'] += 1
            continue
            
        if label == 'faithful':
            stats['faithful_count'] += 1
        else:
            stats['unfaithful_count'] += 1
            
        for layer_idx, activation in layers.items():
            if layer_idx not in layer_activations:
                layer_activations[layer_idx] = {'faithful': [], 'unfaithful': []}
            
            # Extract tensor if it's a dictionary (from extract_last_token_activations.py structure)
            if isinstance(activation, dict):
                if 'last_token' in activation:
                    activation = activation['last_token']
                else:
                    # Fallback: take the first value if 'last_token' missing
                    activation = list(activation.values())[0]

            # Ensure activation is a tensor and on CPU
            if not isinstance(activation, torch.Tensor):
                activation = torch.tensor(activation)
            activation = activation.cpu()
            
            # Flatten if necessary (should be [hidden_dim] or [1, hidden_dim])
            if activation.dim() > 1:
                activation = activation.squeeze()
                
            layer_activations[layer_idx][label].append(activation)
            
    # Compute vectors for each layer
    steering_vectors = {}
    layer_stats = {}
    
    sorted_layers = sorted(layer_activations.keys())
    print(f"Computing vectors for {len(sorted_layers)} layers...")
    
    for layer_idx in sorted_layers:
        faithful_list = layer_activations[layer_idx]['faithful']
        unfaithful_list = layer_activations[layer_idx]['unfaithful']
        
        if not faithful_list or not unfaithful_list:
            print(f"Warning: Layer {layer_idx} missing data (F: {len(faithful_list)}, U: {len(unfaithful_list)})")
            continue
            
        # Stack and mean
        faithful_tensor = torch.stack(faithful_list)
        unfaithful_tensor = torch.stack(unfaithful_list)
        
        faithful_mean = faithful_tensor.mean(dim=0)
        unfaithful_mean = unfaithful_tensor.mean(dim=0)
        
        # Compute mean difference: Faithful - Unfaithful
        vector = faithful_mean - unfaithful_mean
        
        # FIX: Discard Layer 0 (Embeddings) and re-index 1-32 -> 0-31
        if layer_idx == 0:
            # print(f"  Skipping Layer 0 (Embeddings) to align with hidden layers 1-32")
            continue
            
        # Re-index: Layer 1 becomes Layer 0, etc.
        new_layer_idx = layer_idx - 1
        steering_vectors[new_layer_idx] = vector
        # print(f"  Layer {layer_idx} (Original) -> Layer {new_layer_idx} (New): Norm = {vector.norm().item():.4f}")
        
        # Stats
        layer_stats[layer_idx] = {
            'faithful_samples': len(faithful_list),
            'unfaithful_samples': len(unfaithful_list),
            'faithful_norm': faithful_mean.norm().item(),
            'unfaithful_norm': unfaithful_mean.norm().item(),
            'vector_norm': vector.norm().item()
        }
        
    return steering_vectors, stats, layer_stats

def save_results(steering_vectors, stats, layer_stats, output_file):
    # Match the format of src.steering.save_steering_vectors
    output_data = {
        "steering_vectors": steering_vectors,
        "computation_stats": {
            "positive_tags": ["faithful"],
            "negative_tags": ["unfaithful"],
            "layers_computed": list(steering_vectors.keys()),
            "layer_stats": layer_stats,
            "global_stats": stats
        },
        "metadata": {
            "description": "Off-policy steering vectors (Faithful - Unfaithful)",
            "positive_tags": ["faithful"],
            "negative_tags": ["unfaithful"],
            "layers": list(steering_vectors.keys()),
            "vector_dim": list(steering_vectors.values())[0].shape[0] if steering_vectors else 0,
            "computation_date": datetime.now().isoformat()
        }
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"Saved steering vectors to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compute off-policy steering vectors")
    parser.add_argument("--input_file", type=str, default="results/activations_run2/activations_dataset.pkl", help="Input dataset path")
    parser.add_argument("--output_file", type=str, default="results/activations_run2/steering_vectors.pkl", help="Output file path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return
        
    dataset = load_dataset(args.input_file)
    steering_vectors, stats, layer_stats = compute_vectors(dataset)
    
    if steering_vectors:
        save_results(steering_vectors, stats, layer_stats, args.output_file)
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total items processed: {stats['total_items']}")
        print(f"Faithful items: {stats['faithful_count']}")
        print(f"Unfaithful items: {stats['unfaithful_count']}")
        print(f"Layers computed: {len(steering_vectors)}")
        
        # Show sample layer stats
        sample_layer = list(steering_vectors.keys())[15] if 15 in steering_vectors else list(steering_vectors.keys())[0]
        print(f"\nSample Layer {sample_layer}:")
        print(f"  Vector Norm: {layer_stats[sample_layer]['vector_norm']:.4f}")
    else:
        print("No steering vectors computed!")

if __name__ == "__main__":
    main()
