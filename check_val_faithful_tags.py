#!/usr/bin/env python3
"""
Check if validation activation files contain any faithful tags
"""

import torch
import os

val_dir = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint 2.2 acts val mmlu psy 2025-09-14"

print("Checking validation activation files for faithful tags...")
print("=" * 60)

faithful_tags = ['F', 'F_wk', 'F_final', 'F_str']
unfaithful_tags = ['U', 'U_wk', 'U_final', 'U_str']

total_faithful_count = 0
total_unfaithful_count = 0
files_with_faithful = []
files_with_unfaithful = []

# Check all activation files
for filename in os.listdir(val_dir):
    if filename.endswith('_activations.pt'):
        file_path = os.path.join(val_dir, filename)
        print(f"\nFile: {filename}")

        try:
            data = torch.load(file_path, map_location='cpu')

            file_faithful_count = 0
            file_unfaithful_count = 0
            all_tags_in_file = set()

            # Check all layers
            for layer_idx in data.keys():
                if isinstance(data[layer_idx], dict):
                    layer_tags = data[layer_idx].keys()
                    all_tags_in_file.update(layer_tags)

                    # Count faithful tags
                    for tag in faithful_tags:
                        if tag in data[layer_idx]:
                            count = len(data[layer_idx][tag]) if isinstance(data[layer_idx][tag], list) else 0
                            file_faithful_count += count

                    # Count unfaithful tags
                    for tag in unfaithful_tags:
                        if tag in data[layer_idx]:
                            count = len(data[layer_idx][tag]) if isinstance(data[layer_idx][tag], list) else 0
                            file_unfaithful_count += count

            print(f"  All tags found: {sorted(list(all_tags_in_file))}")
            print(f"  Faithful activations: {file_faithful_count}")
            print(f"  Unfaithful activations: {file_unfaithful_count}")

            total_faithful_count += file_faithful_count
            total_unfaithful_count += file_unfaithful_count

            if file_faithful_count > 0:
                files_with_faithful.append(filename)
            if file_unfaithful_count > 0:
                files_with_unfaithful.append(filename)

        except Exception as e:
            print(f"  Error loading: {e}")

print("\n" + "=" * 60)
print("SUMMARY:")
print(f"Total faithful activations across all files: {total_faithful_count}")
print(f"Total unfaithful activations across all files: {total_unfaithful_count}")
print(f"Files with faithful tags: {len(files_with_faithful)}")
print(f"Files with unfaithful tags: {len(files_with_unfaithful)}")

if total_faithful_count > 0:
    print(f"\n⚠️  WARNING: Found {total_faithful_count} faithful activations in validation set!")
    print("This suggests the validation set contains faithful reasoning, which contradicts")
    print("the expectation that all val answers were hint-induced errors (unfaithful).")
    print(f"\nFiles with faithful tags: {files_with_faithful}")
else:
    print(f"\n✅ As expected: No faithful tags found in validation set")
    print("All validation responses appear to be unfaithful/hint-induced errors")

# Check what tags ARE present
all_present_tags = set()
for filename in os.listdir(val_dir):
    if filename.endswith('_activations.pt'):
        file_path = os.path.join(val_dir, filename)
        try:
            data = torch.load(file_path, map_location='cpu')
            for layer_idx in data.keys():
                if isinstance(data[layer_idx], dict):
                    all_present_tags.update(data[layer_idx].keys())
        except:
            pass

print(f"\nAll unique tags found in validation set: {sorted(list(all_present_tags))}")