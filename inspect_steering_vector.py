#!/usr/bin/env python3
"""
Steering Vector Inspector

This script loads and displays the computed steering vector.
"""

import pickle
import numpy as np
from pathlib import Path

def inspect_steering_vector(steering_file: str) -> None:
    """Load and inspect the steering vector."""
    steering_path = Path(steering_file)
    
    if not steering_path.exists():
        print(f"Steering vector file not found: {steering_file}")
        return
    
    print("=" * 60)
    print("STEERING VECTOR INSPECTION")
    print("=" * 60)
    
    # Load the steering vector data
    with open(steering_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"File: {steering_path.name}")
    print(f"Creation time: {data.get('creation_timestamp', 'Unknown')}")
    print()
    
    # Steering vector info
    steering_vector = data['steering_vector']
    print(f"Steering Vector:")
    print(f"  Shape: {steering_vector.shape}")
    print(f"  Data type: {steering_vector.dtype}")
    print(f"  Norm: {np.linalg.norm(steering_vector):.4f}")
    print(f"  Mean: {np.mean(steering_vector):.6f}")
    print(f"  Std: {np.std(steering_vector):.6f}")
    print(f"  Min: {np.min(steering_vector):.6f}")
    print(f"  Max: {np.max(steering_vector):.6f}")
    print()
    
    # Show first 20 values
    print("First 20 values:")
    print(steering_vector[:20])
    print()
    
    # Mean activations info
    faithful_mean = data['faithful_mean']
    unfaithful_mean = data['unfaithful_mean']
    
    print(f"Faithful Mean Activation:")
    print(f"  Shape: {faithful_mean.shape}")
    print(f"  Norm: {np.linalg.norm(faithful_mean):.4f}")
    print(f"  Mean: {np.mean(faithful_mean):.6f}")
    print()
    
    print(f"Unfaithful Mean Activation:")
    print(f"  Shape: {unfaithful_mean.shape}")
    print(f"  Norm: {np.linalg.norm(unfaithful_mean):.4f}")
    print(f"  Mean: {np.mean(unfaithful_mean):.6f}")
    print()
    
    print(f"Dataset Info:")
    print(f"  Faithful count: {data['faithful_count']}")
    print(f"  Unfaithful count: {data['unfaithful_count']}")
    print()
    
    # Save as text file for easier viewing
    text_file = steering_path.with_suffix('.txt')
    with open(text_file, 'w') as f:
        f.write("STEERING VECTOR DATA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Shape: {steering_vector.shape}\n")
        f.write(f"Norm: {np.linalg.norm(steering_vector):.4f}\n")
        f.write(f"Creation: {data.get('creation_timestamp', 'Unknown')}\n\n")
        f.write("Steering Vector Values:\n")
        for i, val in enumerate(steering_vector):
            f.write(f"{i:4d}: {val:10.6f}\n")
    
    print(f"Steering vector values saved to: {text_file}")
    print("=" * 60)

if __name__ == "__main__":
    STEERING_FILE = r"C:\Users\l440\unfaithfulness_steering\contrastive_activation_dataset_steering_vector.pkl"
    inspect_steering_vector(STEERING_FILE)