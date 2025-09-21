#!/usr/bin/env python3
"""Debug script to examine steering vectors metadata in detail"""

import pickle
import torch
import numpy as np

# Load the steering vectors
pkl_file = "steering_vectors_body_only_faithful_vs_unfaithful_2025-09-15.pkl"

print(f"Loading {pkl_file}...")
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

print("=== METADATA ===")
metadata = data.get('metadata', {})
for key, value in metadata.items():
    print(f"{key}: {value}")

print("\n=== COMPUTATION STATS ===")
stats = data.get('computation_stats', {})
for key, value in stats.items():
    print(f"{key}: {value}")

# Test a specific layer vector to understand direction
print("\n=== VECTOR DIRECTION TEST ===")
layer_25_vec = data['steering_vectors'][25]

print(f"Layer 25 vector:")
print(f"  Shape: {layer_25_vec.shape}")
print(f"  Mean: {layer_25_vec.mean():.6f}")
print(f"  Std: {layer_25_vec.std():.6f}")
print(f"  Norm: {torch.norm(layer_25_vec).item():.6f}")

# Check positive vs negative steering effect
print(f"\nTesting coefficient effects:")
print(f"  Vector * (+5.0): mean={layer_25_vec.mean() * 5.0:.6f}, norm={torch.norm(layer_25_vec * 5.0).item():.6f}")
print(f"  Vector * (-5.0): mean={layer_25_vec.mean() * -5.0:.6f}, norm={torch.norm(layer_25_vec * -5.0).item():.6f}")

print("\n=== VECTOR INTERPRETATION ===")
print("Based on metadata:")
print(f"  Positive tags: {metadata.get('positive_tags', 'Unknown')} (should make model more 'F' = Faithful)")
print(f"  Negative tags: {metadata.get('negative_tags', 'Unknown')} (should make model more 'U' = Unfaithful)")
print(f"  Vector direction: Faithful - Unfaithful")
print(f"  +coeff should push toward Faithful")
print(f"  -coeff should push toward Unfaithful")

print("\n=== DONE ===")