#!/usr/bin/env python3
"""Debug script to examine steering vectors pickle file"""

import pickle
import torch
import numpy as np

# Load the steering vectors
pkl_file = "steering_vectors_body_only_faithful_vs_unfaithful_2025-09-15.pkl"

print(f"Loading {pkl_file}...")
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

print(f"\nTop-level keys in pickle file: {list(data.keys())}")

# Examine the structure
if 'steering_vectors' in data:
    vectors = data['steering_vectors']
    print(f"\nSteering vectors keys (layers): {sorted(vectors.keys())}")

    # Check a few layers
    sample_layers = sorted(vectors.keys())[:3]

    for layer in sample_layers:
        vec = vectors[layer]
        print(f"\nLayer {layer}:")
        print(f"  Type: {type(vec)}")
        print(f"  Shape: {vec.shape if hasattr(vec, 'shape') else 'No shape'}")
        print(f"  Dtype: {vec.dtype if hasattr(vec, 'dtype') else 'No dtype'}")

        if hasattr(vec, 'shape') and len(vec.shape) > 0:
            print(f"  Min value: {vec.min():.6f}")
            print(f"  Max value: {vec.max():.6f}")
            print(f"  Mean: {vec.mean():.6f}")
            print(f"  Std: {vec.std():.6f}")
            print(f"  Norm: {torch.norm(vec).item():.6f}")

            # Check if it's all zeros or has any pattern
            if torch.allclose(vec, torch.zeros_like(vec), atol=1e-8):
                print(f"  WARNING: Vector appears to be all zeros!")

            # Show first few values
            print(f"  First 10 values: {vec.flatten()[:10].tolist()}")

# Check if there are other relevant keys
for key, value in data.items():
    if key != 'steering_vectors':
        print(f"\nOther data['{key}']:")
        print(f"  Type: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}")
        elif isinstance(value, (list, dict)):
            print(f"  Length/Keys: {len(value)}")
        print(f"  Value: {str(value)[:200]}...")

print("\n=== DONE ===")