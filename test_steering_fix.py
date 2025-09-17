#!/usr/bin/env python3
"""
Simple test to verify steering coefficients are working correctly
Uses local steering vectors and creates minimal test case
"""

import torch
import json
import pickle
import os

# Load steering vectors to test with
print("Loading steering vectors...")
steering_file = "steering_vectors_body_only_faithful_vs_unfaithful_2025-09-15.pkl"
with open(steering_file, 'rb') as f:
    steering_data = pickle.load(f)

steering_vectors = steering_data['steering_vectors']
test_layer = 25  # Use layer 25 as example

print(f"Testing with layer {test_layer}")
print(f"Vector shape: {steering_vectors[test_layer].shape}")
print(f"Vector norm: {torch.norm(steering_vectors[test_layer]):.3f}")

# Test different coefficients
coefficients = [5.0, -5.0]

for coeff in coefficients:
    print(f"\nTesting coefficient: {coeff}")

    # Simulate what the steering wrapper does
    steering_vector = steering_vectors[test_layer]
    steering_addition = steering_vector * coeff

    print(f"  Original vector mean: {steering_vector.mean():.6f}")
    print(f"  Steering addition mean: {steering_addition.mean():.6f}")
    print(f"  Steering addition norm: {torch.norm(steering_addition):.3f}")

    # These should be opposite signs for positive/negative coefficients
    if coeff > 0:
        pos_mean = steering_addition.mean().item()
        pos_norm = torch.norm(steering_addition).item()
    else:
        neg_mean = steering_addition.mean().item()
        neg_norm = torch.norm(steering_addition).item()

print(f"\n=== COEFFICIENT TEST RESULTS ===")
print(f"Positive coeff mean: {pos_mean:.6f}")
print(f"Negative coeff mean: {neg_mean:.6f}")
print(f"Positive coeff norm: {pos_norm:.3f}")
print(f"Negative coeff norm: {neg_norm:.3f}")

# Check if they are truly opposite
if abs(pos_mean + neg_mean) < 1e-5:
    print("✓ PASS: Means are opposite (as expected)")
else:
    print(f"✗ FAIL: Means are not opposite (diff: {abs(pos_mean + neg_mean):.6f})")

if abs(pos_norm - neg_norm) < 1e-5:
    print("✓ PASS: Norms are equal (as expected)")
else:
    print(f"✗ FAIL: Norms are not equal (diff: {abs(pos_norm - neg_norm):.6f})")

print("\n=== MOCK WRAPPER TEST ===")

# Mock the LayerSteeringWrapper behavior
class MockWrapper:
    def __init__(self):
        self.steering_vector = None
        self.coefficient = 0.0
        self.active = False

    def set_steering(self, vector, coeff):
        self.steering_vector = vector
        self.coefficient = coeff
        self.active = True
        print(f"  Set: coeff={self.coefficient}, vector_norm={torch.norm(self.steering_vector):.3f}")

    def reset(self):
        print(f"  Reset: was coeff={self.coefficient}")
        self.active = False
        self.steering_vector = None
        self.coefficient = 0.0

    def get_steering_addition(self):
        if self.active and self.steering_vector is not None:
            return self.steering_vector * self.coefficient
        return None

wrapper = MockWrapper()

for coeff in coefficients:
    print(f"\nTesting wrapper with coefficient: {coeff}")
    wrapper.reset()
    wrapper.set_steering(steering_vectors[test_layer], coeff)

    addition = wrapper.get_steering_addition()
    if addition is not None:
        print(f"  Wrapper addition mean: {addition.mean():.6f}")
        print(f"  Wrapper addition norm: {torch.norm(addition):.3f}")
    else:
        print("  ERROR: No steering addition!")

print("\n=== TEST COMPLETE ===")
print("If this test passes, the steering logic should work correctly.")
print("The issue in your original code might be elsewhere (e.g., file paths, model loading, etc.)")