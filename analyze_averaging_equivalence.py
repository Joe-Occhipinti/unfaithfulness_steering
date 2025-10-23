"""
Mathematical analysis: Is averaging separately-computed steering vectors
equivalent to merging datasets and computing one vector?

Scenario:
- Dataset A: compute vector_A from mean(pos_A) - mean(neg_A)
- Dataset B: compute vector_B from mean(pos_B) - mean(neg_B)
- Merged: compute vector_merged from mean(pos_A + pos_B) - mean(neg_A + neg_B)

Question: Does (vector_A + vector_B) / 2 == vector_merged ?
"""

import pickle
import torch

print("=== LOADING DATASETS ===\n")

# Load both datasets
pos_file = r"data\sprint4_2025-10-21\datasets\acts_pos_bas_psyXprof_2025-10-19.pkl"
neg_file = r"data\sprint4_2025-10-21\datasets\acts_neg_bas_psyXprof_2025-10-19.pkl"

with open(pos_file, 'rb') as f:
    dataset_A = pickle.load(f)

with open(neg_file, 'rb') as f:
    dataset_B = pickle.load(f)

print("Dataset A (pos_bas):")
print(f"  Prompts: {dataset_A['info']['num_prompts']}")
print(f"  F_body: {dataset_A['info']['total_stats']['F_body']}")
print(f"  U_body: {dataset_A['info']['total_stats']['U_body']}")

print("\nDataset B (neg_bas):")
print(f"  Prompts: {dataset_B['info']['num_prompts']}")
print(f"  F_body: {dataset_B['info']['total_stats']['F_body']}")
print(f"  U_body: {dataset_B['info']['total_stats']['U_body']}")

print("\n=== MATHEMATICAL ANALYSIS ===\n")

# Extract train split activations for both datasets
def extract_train_activations(dataset, pos_tags, neg_tags, layer_idx=0):
    """Extract positive and negative activations from train split for a specific layer"""
    pos_acts = []
    neg_acts = []

    for prompt_idx, prompt_data in dataset['data'].items():
        if prompt_data['metadata'].get('split') != 'train':
            continue

        layers = prompt_data['layers']
        if layer_idx not in layers:
            continue

        layer = layers[layer_idx]

        for tag in pos_tags:
            if tag in layer and layer[tag].numel() > 0:
                pos_acts.append(layer[tag])

        for tag in neg_tags:
            if tag in layer and layer[tag].numel() > 0:
                neg_acts.append(layer[tag])

    pos_tensor = torch.cat(pos_acts, dim=0) if pos_acts else torch.empty(0, dataset['info']['hidden_dim'])
    neg_tensor = torch.cat(neg_acts, dim=0) if neg_acts else torch.empty(0, dataset['info']['hidden_dim'])

    return pos_tensor, neg_tensor

# Use layer 0 for demonstration (same logic applies to all layers)
LAYER = 0
POS_TAGS = ['F_body']
NEG_TAGS = ['U_body']

print(f"Testing on Layer {LAYER} with {POS_TAGS} vs {NEG_TAGS}\n")

# Extract activations from both datasets (train split only)
pos_A, neg_A = extract_train_activations(dataset_A, POS_TAGS, NEG_TAGS, LAYER)
pos_B, neg_B = extract_train_activations(dataset_B, POS_TAGS, NEG_TAGS, LAYER)

print(f"Dataset A (train split only):")
print(f"  Positive samples: {pos_A.shape[0]}")
print(f"  Negative samples: {neg_A.shape[0]}")

print(f"\nDataset B (train split only):")
print(f"  Positive samples: {pos_B.shape[0]}")
print(f"  Negative samples: {neg_B.shape[0]}")

print("\n=== COMPUTING STEERING VECTORS ===\n")

# Method 1: Separate computation then average
print("Method 1: Compute separately, then average")

vector_A = pos_A.mean(dim=0) - neg_A.mean(dim=0)
print(f"  vector_A = mean(pos_A) - mean(neg_A)")
print(f"    Norm: {vector_A.norm().item():.6f}")

vector_B = pos_B.mean(dim=0) - neg_B.mean(dim=0)
print(f"  vector_B = mean(pos_B) - mean(neg_B)")
print(f"    Norm: {vector_B.norm().item():.6f}")

averaged_vector = (vector_A + vector_B) / 2
print(f"\n  averaged_vector = (vector_A + vector_B) / 2")
print(f"    Norm: {averaged_vector.norm().item():.6f}")

# Method 2: Merge datasets then compute
print("\n\nMethod 2: Merge datasets, then compute")

merged_pos = torch.cat([pos_A, pos_B], dim=0)
merged_neg = torch.cat([neg_A, neg_B], dim=0)

print(f"  Merged positive samples: {merged_pos.shape[0]} (= {pos_A.shape[0]} + {pos_B.shape[0]})")
print(f"  Merged negative samples: {merged_neg.shape[0]} (= {neg_A.shape[0]} + {neg_B.shape[0]})")

merged_vector = merged_pos.mean(dim=0) - merged_neg.mean(dim=0)
print(f"\n  merged_vector = mean(merged_pos) - mean(merged_neg)")
print(f"    Norm: {merged_vector.norm().item():.6f}")

print("\n=== EQUIVALENCE CHECK ===\n")

# Check if they're equal
difference = (averaged_vector - merged_vector).norm().item()
max_magnitude = max(averaged_vector.norm().item(), merged_vector.norm().item())
relative_error = difference / max_magnitude if max_magnitude > 0 else 0

print(f"||averaged_vector - merged_vector|| = {difference:.10f}")
print(f"Relative error: {relative_error:.10f}")

if difference < 1e-6:
    print("\n[OK] VECTORS ARE EQUAL (within numerical tolerance)")
    print("  -> Averaging is mathematically equivalent to merging!")
else:
    print(f"\n[FAIL] VECTORS ARE DIFFERENT")
    print("  -> Averaging is NOT equivalent to merging")

# Mathematical explanation
print("\n=== MATHEMATICAL PROOF ===\n")

n_pos_A = pos_A.shape[0]
n_neg_A = neg_A.shape[0]
n_pos_B = pos_B.shape[0]
n_neg_B = neg_B.shape[0]

print("Averaged vector:")
print(f"  = (1/2) * [mean(pos_A) - mean(neg_A)] + (1/2) * [mean(pos_B) - mean(neg_B)]")
print(f"  = (1/2) * mean(pos_A) + (1/2) * mean(pos_B) - (1/2) * mean(neg_A) - (1/2) * mean(neg_B)")

print("\nMerged vector:")
print(f"  = mean(pos_A U pos_B) - mean(neg_A U neg_B)")
print(f"  = (n_pos_A * mean(pos_A) + n_pos_B * mean(pos_B)) / (n_pos_A + n_pos_B)")
print(f"    - (n_neg_A * mean(neg_A) + n_neg_B * mean(neg_B)) / (n_neg_A + n_neg_B)")

print("\nSubstituting actual counts:")
print(f"  = ({n_pos_A} * mean(pos_A) + {n_pos_B} * mean(pos_B)) / {n_pos_A + n_pos_B}")
print(f"    - ({n_neg_A} * mean(neg_A) + {n_neg_B} * mean(neg_B)) / {n_neg_A + n_neg_B}")

print("\nThese are EQUAL if and only if:")
print(f"  (1/2) * mean(pos_A) + (1/2) * mean(pos_B) = ({n_pos_A} * mean(pos_A) + {n_pos_B} * mean(pos_B)) / {n_pos_A + n_pos_B}")
print(f"  AND")
print(f"  (1/2) * mean(neg_A) + (1/2) * mean(neg_B) = ({n_neg_A} * mean(neg_A) + {n_neg_B} * mean(neg_B)) / {n_neg_A + n_neg_B}")

print("\nSimplifying the positive condition:")
print(f"  (n_pos_A + n_pos_B) / 2 = n_pos_A  AND  (n_pos_A + n_pos_B) / 2 = n_pos_B")
print(f"  ==> n_pos_A = n_pos_B = {n_pos_A}")

print("\nSimplifying the negative condition:")
print(f"  (n_neg_A + n_neg_B) / 2 = n_neg_A  AND  (n_neg_A + n_neg_B) / 2 = n_neg_B")
print(f"  ==> n_neg_A = n_neg_B = {n_neg_A}")

print("\n=== CONCLUSION ===\n")

pos_equal = (n_pos_A == n_pos_B)
neg_equal = (n_neg_A == n_neg_B)

if pos_equal and neg_equal:
    print("[OK] Sample counts are EQUAL:")
    print(f"  n_pos_A = n_pos_B = {n_pos_A}")
    print(f"  n_neg_A = n_neg_B = {n_neg_A}")
    print("\n[OK] Therefore: Averaging == Merging (mathematically equivalent)")
else:
    print("[FAIL] Sample counts are NOT equal:")
    print(f"  n_pos_A = {n_pos_A}, n_pos_B = {n_pos_B} -> Equal: {pos_equal}")
    print(f"  n_neg_A = {n_neg_A}, n_neg_B = {n_neg_B} -> Equal: {neg_equal}")
    print("\n[FAIL] Therefore: Averaging != Merging")
    print("\nTo achieve equivalence, you need WEIGHTED averaging:")
    print(f"  weighted_vector = ({n_pos_A} * vector_A + {n_pos_B} * vector_B) / {n_pos_A + n_pos_B}")
    print(f"                    [Approximate - exact formula is more complex]")
