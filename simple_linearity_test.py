"""
Simple test to check if the basic functionality works
"""

import torch
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
print("Loading dataset...")
with open(r"C:\Users\l440\Desktop\unfaithfulness_steering-1\datasets\sprint_2_contrastive_dataset_full_sweep_all_(un)faithful_tags_mmlu_psychology_train_2025-09-04.pkl", 'rb') as f:
    dataset = pickle.load(f)

print("Dataset loaded successfully!")

# Test just layer 31 (the best layer from previous analysis)
layer_idx = 31
data = dataset['data'][layer_idx]

print(f"\nTesting layer {layer_idx}...")

# Get activations
positive_acts = []
negative_acts = []

# Faithful tags
for tag in ["F_str", "F_wk"]:
    if tag in data and data[tag].numel() > 0:
        positive_acts.append(data[tag])
        print(f"Added {data[tag].shape[0]} samples from {tag}")

# Unfaithful tags  
for tag in ["U_str", "U_wk"]:
    if tag in data and data[tag].numel() > 0:
        negative_acts.append(data[tag])
        print(f"Added {data[tag].shape[0]} samples from {tag}")

if not positive_acts or not negative_acts:
    print("ERROR: No activations found!")
    exit()

# Combine
positive_combined = torch.cat(positive_acts, dim=0)
negative_combined = torch.cat(negative_acts, dim=0)

print(f"Positive samples: {positive_combined.shape[0]}")
print(f"Negative samples: {negative_combined.shape[0]}")

# Convert to numpy (handle bfloat16)
X = torch.cat([positive_combined.float(), negative_combined.float()], dim=0).numpy()
y = torch.cat([
    torch.zeros(positive_combined.shape[0]),  # 0 for faithful
    torch.ones(negative_combined.shape[0])    # 1 for unfaithful  
]).numpy()

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Train classifier
print("Training linear classifier...")
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, classifier.predict(X_train))
test_acc = accuracy_score(y_test, classifier.predict(X_test))

print(f"\n=== RESULTS ===")
print(f"Training accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

# Interpretation
if test_acc > 0.8:
    print("✅ EXCELLENT linear separability - faithfulness is strongly linearly encoded!")
elif test_acc > 0.7:
    print("✅ GOOD linear separability - faithfulness is moderately linearly encoded")
elif test_acc > 0.6:  
    print("⚠️ MODERATE linear separability - steering may work but not optimally")
else:
    print("❌ POOR linear separability - faithfulness may not be linearly encoded")

print(f"\nConclusion: Layer {layer_idx} shows {test_acc:.1%} linear classification accuracy")