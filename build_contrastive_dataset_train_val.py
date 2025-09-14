"""
build_contrastive_dataset_train_val.py

Builds a contrastive dataset from extracted period token activations for both train and validation sets.
Loads activation files from both directories and organizes them layer-wise with labels and train/val splits.
This allows training a linear classifier on train set and testing on val set to avoid overfitting.
"""

import torch
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import json

# === CONFIGURATION ===
# Directories containing the extracted activation files
train_activations_dir = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint 2.2 acts train mmlu psy 2025-09-14"  # Update with your train directory
val_activations_dir = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\sprint 2.2 acts val mmlu psy 2025-09-14"      # Update with your val directory

# Tags to include in the contrastive dataset
CONTRASTIVE_TAGS = ["F", "U", "U_final", "F_final"]

# Output file for the contrastive dataset
output_file = "sprint_2.2_contrastive_dataset_train_val_strong_faithful_unfaithful_2025-09-14.pkl"

# Layer range (adjust based on your model)
NUM_LAYERS = 32

print(f"--- Building Train/Val Contrastive Dataset ---")
print(f"Train Directory: {train_activations_dir}")
print(f"Val Directory: {val_activations_dir}")
print(f"Target Tags: {CONTRASTIVE_TAGS}")
print(f"Output File: {output_file}")
print(f"Layers: 0-{NUM_LAYERS-1}")
print("-------------------------------------------")

def load_activation_files_from_dir(directory, split_name):
    """Load activation files from a directory and return file paths."""
    print(f"\n--- Loading {split_name} activation files from {directory}... ---")

    activation_files = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.startswith("prompt_") and filename.endswith("_activations.pt"):
                activation_files.append(os.path.join(directory, filename))
        activation_files.sort()  # Sort for consistent ordering
    else:
        print(f"ERROR: Directory {directory} does not exist!")
        return []

    print(f"Found {len(activation_files)} {split_name} activation files to process.")
    return activation_files

def process_activation_files(activation_files, split_name):
    """Process activation files and return contrastive data and stats."""
    print(f"\n--- Processing {split_name} activation files... ---")

    # Structure: contrastive_data[layer][tag] = list of activation tensors
    contrastive_data = {layer: {tag: [] for tag in CONTRASTIVE_TAGS} for layer in range(NUM_LAYERS)}

    # Statistics tracking
    stats = {tag: 0 for tag in CONTRASTIVE_TAGS}
    total_files_processed = 0
    files_with_data = 0

    for file_path in tqdm(activation_files, desc=f"Processing {split_name} files"):
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

    print(f"\n--- {split_name} Processing Complete ---")
    print(f"Files processed: {total_files_processed}")
    print(f"Files with relevant data: {files_with_data}")
    print(f"\n{split_name} activation counts per tag:")
    for tag, count in stats.items():
        print(f"  {tag}: {count} activations")

    return contrastive_data, stats

def convert_to_tensors(contrastive_data, split_name):
    """Convert contrastive data to tensors for each layer."""
    print(f"\n--- Converting {split_name} to tensors layer-wise... ---")

    # Final dataset structure: dataset[layer][tag] = torch.tensor([num_samples, hidden_dim])
    final_dataset = {}
    layer_stats = {}

    for layer_idx in tqdm(range(NUM_LAYERS), desc=f"Processing {split_name} layers"):
        layer_dataset = {}
        layer_stat = {}

        for tag in CONTRASTIVE_TAGS:
            activations_list = contrastive_data[layer_idx][tag]

            if activations_list:
                # Stack all activations for this tag and layer into a single tensor
                stacked_tensor = torch.stack(activations_list, dim=0)
                layer_dataset[tag] = stacked_tensor
                layer_stat[tag] = len(activations_list)

                print(f"  {split_name} Layer {layer_idx}, Tag {tag}: {stacked_tensor.shape}")
            else:
                layer_dataset[tag] = torch.empty(0, 4096)  # Empty tensor with correct hidden_dim
                layer_stat[tag] = 0
                print(f"  {split_name} Layer {layer_idx}, Tag {tag}: No data")

        final_dataset[layer_idx] = layer_dataset
        layer_stats[layer_idx] = layer_stat

    return final_dataset, layer_stats

# === LOAD AND PROCESS TRAIN DATA ===
train_files = load_activation_files_from_dir(train_activations_dir, "TRAIN")
if not train_files:
    print("No train files found. Exiting.")
    exit(1)

train_contrastive_data, train_stats = process_activation_files(train_files, "TRAIN")
train_final_dataset, train_layer_stats = convert_to_tensors(train_contrastive_data, "TRAIN")

# === LOAD AND PROCESS VAL DATA ===
val_files = load_activation_files_from_dir(val_activations_dir, "VAL")
if not val_files:
    print("No validation files found. Exiting.")
    exit(1)

val_contrastive_data, val_stats = process_activation_files(val_files, "VAL")
val_final_dataset, val_layer_stats = convert_to_tensors(val_contrastive_data, "VAL")

# === COMBINE INTO FINAL DATASET STRUCTURE ===
print(f"\n--- Combining train and val datasets... ---")

dataset_info = {
    "tags": CONTRASTIVE_TAGS,
    "num_layers": NUM_LAYERS,
    "splits": ["train", "val"],
    "train_stats": {
        "total_stats": train_stats,
        "layer_stats": train_layer_stats
    },
    "val_stats": {
        "total_stats": val_stats,
        "layer_stats": val_layer_stats
    }
}

final_combined_dataset = {
    "train": train_final_dataset,
    "val": val_final_dataset
}

# === SAVE CONTRASTIVE DATASET ===
print(f"\n--- Saving train/val contrastive dataset to {output_file}... ---")

dataset_to_save = {
    "data": final_combined_dataset,
    "info": dataset_info
}

with open(output_file, 'wb') as f:
    pickle.dump(dataset_to_save, f)

print(f"Train/Val contrastive dataset saved successfully!")

# === SUMMARY STATISTICS ===
print(f"\n--- Dataset Summary ---")
print(f"Total layers: {NUM_LAYERS}")
print(f"Tags included: {CONTRASTIVE_TAGS}")
print(f"Splits: train, val")

print(f"\nTrain sample counts by layer (first 5 layers):")
for layer_idx in range(min(5, NUM_LAYERS)):
    print(f"  Layer {layer_idx}:")
    for tag in CONTRASTIVE_TAGS:
        count = train_layer_stats[layer_idx][tag]
        print(f"    {tag}: {count}")

print(f"\nVal sample counts by layer (first 5 layers):")
for layer_idx in range(min(5, NUM_LAYERS)):
    print(f"  Layer {layer_idx}:")
    for tag in CONTRASTIVE_TAGS:
        count = val_layer_stats[layer_idx][tag]
        print(f"    {tag}: {count}")

print(f"\nTotal train samples across all layers:")
for tag in CONTRASTIVE_TAGS:
    total = sum(train_layer_stats[layer][tag] for layer in range(NUM_LAYERS))
    print(f"  {tag}: {total}")

print(f"\nTotal val samples across all layers:")
for tag in CONTRASTIVE_TAGS:
    total = sum(val_layer_stats[layer][tag] for layer in range(NUM_LAYERS))
    print(f"  {tag}: {total}")

print(f"\n--- Dataset building complete! ---")

# === EXAMPLE: HOW TO LOAD THE DATASET ===
print(f"\n--- Example: How to load this train/val dataset ---")
print(f"""
To load and use this train/val dataset:

import pickle
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
with open('{output_file}', 'rb') as f:
    dataset = pickle.load(f)

data = dataset['data']
info = dataset['info']

# Access activations for a specific layer
layer_idx = 15

# TRAIN DATA
train_faithful = data['train'][layer_idx]['F']      # Shape: [num_samples, hidden_dim]
train_unfaithful = data['train'][layer_idx]['U']    # Shape: [num_samples, hidden_dim]

# Create train labels (0 for faithful, 1 for unfaithful)
train_faithful_labels = torch.zeros(train_faithful.shape[0])
train_unfaithful_labels = torch.ones(train_unfaithful.shape[0])

# Combine train data and labels
X_train = torch.cat([train_faithful, train_unfaithful], dim=0)
y_train = torch.cat([train_faithful_labels, train_unfaithful_labels], dim=0)

# VAL DATA
val_faithful = data['val'][layer_idx]['F']          # Shape: [num_samples, hidden_dim]
val_unfaithful = data['val'][layer_idx]['U']        # Shape: [num_samples, hidden_dim]

# Create val labels
val_faithful_labels = torch.zeros(val_faithful.shape[0])
val_unfaithful_labels = torch.ones(val_unfaithful.shape[0])

# Combine val data and labels
X_val = torch.cat([val_faithful, val_unfaithful], dim=0)
y_val = torch.cat([val_faithful_labels, val_unfaithful_labels], dim=0)

# Train a linear classifier
classifier = LogisticRegression()
classifier.fit(X_train.numpy(), y_train.numpy())

# Test on validation set (avoiding overfitting)
val_predictions = classifier.predict(X_val.numpy())
accuracy = accuracy_score(y_val.numpy(), val_predictions)
print(f"Layer {{layer_idx}} Validation Accuracy: {{accuracy:.3f}}")
""")

print(f"\nNow you can train classifiers on the train set and evaluate on val set!")
print(f"This avoids overfitting and provides reliable performance estimates.")