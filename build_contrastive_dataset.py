"""
build_contrastive_dataset.py

Builds a contrastive dataset from extracted period token activations.
Loads all activation files and organizes them layer-wise with labels for specified tags.
"""

import torch
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import json

# === CONFIGURATION ===
# Directory containing the extracted activation files
activations_dir =  r"C:\Users\l440\Downloads\sprint_2_acts_fullsweep_tags_mmlu_psy_train_2025-09-04"

# Tags to include in the contrastive dataset
CONTRASTIVE_TAGS = ["F_str", "U_str", "F_wk", "U_wk"]

# Output file for the contrastive dataset
output_file = "sprint_2_contrastive_dataset_full_sweep_all_(un)faithful_tags_mmlu_psychology_train_2025-09-04.pkl"

# Layer range (adjust based on your model)
NUM_LAYERS = 32

print(f"--- Building Contrastive Dataset ---")
print(f"Input Directory: {activations_dir}")
print(f"Target Tags: {CONTRASTIVE_TAGS}")
print(f"Output File: {output_file}")
print(f"Layers: 0-{NUM_LAYERS-1}")
print("-----------------------------------")

# === LOAD ALL ACTIVATION FILES ===
print(f"\n--- Loading activation files from {activations_dir}... ---")

# Find all activation files
activation_files = []
if os.path.exists(activations_dir):
    for filename in os.listdir(activations_dir):
        if filename.startswith("prompt_") and filename.endswith("_activations.pt"):
            activation_files.append(os.path.join(activations_dir, filename))
    activation_files.sort()  # Sort for consistent ordering
else:
    print(f"ERROR: Directory {activations_dir} does not exist!")
    exit(1)

print(f"Found {len(activation_files)} activation files to process.")

# === BUILD CONTRASTIVE DATASET ===
print(f"\n--- Building contrastive dataset... ---")

# Structure: contrastive_data[layer][tag] = list of activation tensors
contrastive_data = {layer: {tag: [] for tag in CONTRASTIVE_TAGS} for layer in range(NUM_LAYERS)}

# Statistics tracking
stats = {tag: 0 for tag in CONTRASTIVE_TAGS}
total_files_processed = 0
files_with_data = 0

for file_path in tqdm(activation_files, desc="Processing activation files"):
    try:
        # Load the activation data for this prompt
        prompt_data = torch.load(file_path, map_location='cpu')
        
        has_data_for_prompt = False
        
        # Extract data for each layer
        for layer_idx in range(NUM_LAYERS):
            if layer_idx in prompt_data:
                layer_data = prompt_data[layer_idx]
                
                # Extract activations for each target tag
                for tag in CONTRASTIVE_TAGS:
                    if tag in layer_data and layer_data[tag]:
                        # layer_data[tag] is a list of individual activation tensors
                        for activation_tensor in layer_data[tag]:
                            contrastive_data[layer_idx][tag].append(activation_tensor)
                            stats[tag] += 1
                            has_data_for_prompt = True
        
        if has_data_for_prompt:
            files_with_data += 1
        
        total_files_processed += 1
        
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}")

print(f"\n--- Processing Complete ---")
print(f"Files processed: {total_files_processed}")
print(f"Files with relevant data: {files_with_data}")
print(f"\nActivation counts per tag:")
for tag, count in stats.items():
    print(f"  {tag}: {count} activations")

# === CONVERT TO TENSORS FOR EACH LAYER ===
print(f"\n--- Converting to tensors layer-wise... ---")

# Final dataset structure: dataset[layer][tag] = torch.tensor([num_samples, hidden_dim])
final_dataset = {}
dataset_info = {
    "tags": CONTRASTIVE_TAGS,
    "num_layers": NUM_LAYERS,
    "layer_stats": {},
    "total_stats": stats
}

for layer_idx in tqdm(range(NUM_LAYERS), desc="Processing layers"):
    layer_dataset = {}
    layer_stats = {}
    
    for tag in CONTRASTIVE_TAGS:
        activations_list = contrastive_data[layer_idx][tag]
        
        if activations_list:
            # Stack all activations for this tag and layer into a single tensor
            stacked_tensor = torch.stack(activations_list, dim=0)
            layer_dataset[tag] = stacked_tensor
            layer_stats[tag] = len(activations_list)
            
            print(f"  Layer {layer_idx}, Tag {tag}: {stacked_tensor.shape}")
        else:
            layer_dataset[tag] = torch.empty(0, 4096)  # Empty tensor with correct hidden_dim
            layer_stats[tag] = 0
            print(f"  Layer {layer_idx}, Tag {tag}: No data")
    
    final_dataset[layer_idx] = layer_dataset
    dataset_info["layer_stats"][layer_idx] = layer_stats

# === SAVE CONTRASTIVE DATASET ===
print(f"\n--- Saving contrastive dataset to {output_file}... ---")

dataset_to_save = {
    "data": final_dataset,
    "info": dataset_info
}

with open(output_file, 'wb') as f:
    pickle.dump(dataset_to_save, f)

print(f"Contrastive dataset saved successfully!")

# === SUMMARY STATISTICS ===
print(f"\n--- Dataset Summary ---")
print(f"Total layers: {NUM_LAYERS}")
print(f"Tags included: {CONTRASTIVE_TAGS}")

print(f"\nSample counts by layer (first 5 layers):")
for layer_idx in range(min(5, NUM_LAYERS)):
    print(f"  Layer {layer_idx}:")
    for tag in CONTRASTIVE_TAGS:
        count = dataset_info["layer_stats"][layer_idx][tag]
        print(f"    {tag}: {count}")

print(f"\nTotal samples across all layers:")
for tag in CONTRASTIVE_TAGS:
    total = sum(dataset_info["layer_stats"][layer][tag] for layer in range(NUM_LAYERS))
    print(f"  {tag}: {total}")

print(f"\n--- Dataset building complete! ---")

# === EXAMPLE: HOW TO LOAD THE DATASET ===
print(f"\n--- Example: How to load this dataset ---")
print(f"""
To load and use this dataset:

import pickle
import torch

# Load the dataset
with open('{output_file}', 'rb') as f:
    dataset = pickle.load(f)

data = dataset['data']
info = dataset['info']

# Access activations for a specific layer and tag
layer_15_faithful_strong = data[15]['F_str']  # Shape: [num_samples, hidden_dim]
layer_15_unfaithful_strong = data[15]['U_str']  # Shape: [num_samples, hidden_dim]

# Create labels (0 for faithful, 1 for unfaithful)
faithful_labels = torch.zeros(layer_15_faithful_strong.shape[0])
unfaithful_labels = torch.ones(layer_15_unfaithful_strong.shape[0])

# Combine data and labels for contrastive learning
X = torch.cat([layer_15_faithful_strong, layer_15_unfaithful_strong], dim=0)
y = torch.cat([faithful_labels, unfaithful_labels], dim=0)
""")