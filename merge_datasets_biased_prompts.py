"""
merge_annotated_datasets.py

Merges positive and negative baseline annotated datasets into a single file.
"""

import json
from pathlib import Path

# Input files
POS_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\pos_bas_annotated_histXmeta_2025-10-19.jsonl"
NEG_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\neg_bas_annotated_histXmeta_2025-10-19.jsonl"

# Output file
OUTPUT_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\annotated_histXmeta_2025-10-19.jsonl"

def load_jsonl(filepath):
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, filepath):
    """Save JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Load both datasets
print(f"Loading {POS_FILE}...")
pos_data = load_jsonl(POS_FILE)
print(f"Loaded {len(pos_data)} records from positive baseline")

print(f"\nLoading {NEG_FILE}...")
neg_data = load_jsonl(NEG_FILE)
print(f"Loaded {len(neg_data)} records from negative baseline")

# Merge datasets
merged_data = pos_data + neg_data
print(f"\nMerged total: {len(merged_data)} records")

# Save merged dataset
save_jsonl(merged_data, OUTPUT_FILE)
print(f"\nSaved merged dataset to {OUTPUT_FILE}")

# Print summary statistics
splits = {}
for item in merged_data:
    split = item.get('split', 'unknown')
    splits[split] = splits.get(split, 0) + 1

print("\nSplit distribution in merged dataset:")
for split, count in sorted(splits.items()):
    print(f"  {split}: {count}")