#!/usr/bin/env python3
"""
Debug script to check validation activation files
"""

import torch

# Check a validation activation file
val_file = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint 2.2 acts val mmlu psy 2025-09-14\prompt_0_activations.pt"

print(f"Loading: {val_file}")

data = torch.load(val_file, map_location='cpu')

print(f"\nFile structure: {type(data)}")
print(f"Top level keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

# Check a few layers
for layer_idx in [0, 15, 31]:
    if layer_idx in data:
        print(f"\nLayer {layer_idx}:")
        layer_data = data[layer_idx]
        if isinstance(layer_data, dict):
            print(f"  Tags: {list(layer_data.keys())}")
            for tag in layer_data.keys():
                if isinstance(layer_data[tag], list):
                    count = len(layer_data[tag])
                    print(f"  {tag}: {count} activations")
                    if count > 0 and hasattr(layer_data[tag][0], 'shape'):
                        print(f"    First activation shape: {layer_data[tag][0].shape}")
                else:
                    print(f"  {tag}: {type(layer_data[tag])}")
        else:
            print(f"  Layer data type: {type(layer_data)}")
    else:
        print(f"\nLayer {layer_idx}: Not found")

# Check another file for comparison
val_file_2 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint 2.2 acts val mmlu psy 2025-09-14\prompt_1_activations.pt"
print(f"\n\nChecking second file: {val_file_2}")
data2 = torch.load(val_file_2, map_location='cpu')

if 0 in data2 and isinstance(data2[0], dict):
    print(f"Second file layer 0 tags: {list(data2[0].keys())}")
    for tag in ['F', 'U', 'F_final', 'U_final']:
        if tag in data2[0]:
            count = len(data2[0][tag]) if isinstance(data2[0][tag], list) else 'Not a list'
            print(f"  {tag}: {count}")