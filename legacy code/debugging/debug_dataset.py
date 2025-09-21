#!/usr/bin/env python3
"""
Debug script to check dataset structure
"""

import pickle

# Load the dataset
dataset_file = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint_2.2_contrastive_dataset_train_val_strong_faithful_unfaithful_2025-09-14.pkl"

print(f"Loading: {dataset_file}")

with open(dataset_file, 'rb') as f:
    dataset = pickle.load(f)

print("\n=== DATASET STRUCTURE ===")
print(f"Top level keys: {list(dataset.keys())}")

if 'data' in dataset:
    print(f"Data keys: {list(dataset['data'].keys())}")

    if 'train' in dataset['data']:
        print(f"Train data layers: {list(dataset['data']['train'].keys())[:5]}...")

        # Check layer 0 in train
        if 0 in dataset['data']['train']:
            print(f"Layer 0 train tags: {list(dataset['data']['train'][0].keys())}")
            for tag in dataset['data']['train'][0].keys():
                shape = dataset['data']['train'][0][tag].shape if hasattr(dataset['data']['train'][0][tag], 'shape') else 'No shape'
                print(f"  {tag}: {shape}")

    if 'val' in dataset['data']:
        print(f"Val data layers: {list(dataset['data']['val'].keys())[:5]}...")

        # Check layer 0 in val
        if 0 in dataset['data']['val']:
            print(f"Layer 0 val tags: {list(dataset['data']['val'][0].keys())}")
            for tag in dataset['data']['val'][0].keys():
                shape = dataset['data']['val'][0][tag].shape if hasattr(dataset['data']['val'][0][tag], 'shape') else 'No shape'
                print(f"  {tag}: {shape}")

if 'info' in dataset:
    print(f"\n=== INFO ===")
    print(f"Info keys: {list(dataset['info'].keys())}")
    print(f"Tags: {dataset['info'].get('tags', 'Not found')}")
    print(f"Num layers: {dataset['info'].get('num_layers', 'Not found')}")
    print(f"Splits: {dataset['info'].get('splits', 'Not found')}")

print("\n=== LAYER-WISE SAMPLE COUNTS ===")
if 'data' in dataset and 'train' in dataset['data']:
    for layer in [0, 10, 20, 30, 31]:
        if layer in dataset['data']['train']:
            print(f"Layer {layer}:")
            for split in ['train', 'val']:
                if split in dataset['data'] and layer in dataset['data'][split]:
                    for tag in ['F', 'U', 'F_final', 'U_final']:
                        if tag in dataset['data'][split][layer]:
                            count = dataset['data'][split][layer][tag].shape[0] if hasattr(dataset['data'][split][layer][tag], 'shape') else 0
                            print(f"  {split} {tag}: {count}")