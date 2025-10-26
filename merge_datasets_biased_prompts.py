"""
merge_annotated_datasets.py

Merges positive and negative baseline annotated datasets into a single file.
"""

import json
from pathlib import Path

# Input files
FILE_1 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\annotated_1_science_hist_psy_2025-10-25.jsonl"
FILE_2 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\annotated_2_science_hist_psy_2025-10-25.jsonl"
FILE_3 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\pos_bas_annotated_scienceXgrader_2025-10-21.jsonl"
FILE_4 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\neg_bas_annotated_scienceXgrader_2025-10-21.jsonl"
FILE_5 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\pos_bas_annotated_histXmeta_2025-10-19.jsonl"
FILE_6 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\neg_bas_annotated_histXmeta_2025-10-19.jsonl"
FILE_7 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\pos_bas_annotated_psyXprof_2025-10-19.jsonl"
FILE_8 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\neg_bas_annotated_psyXprof_2025-10-19.jsonl"

# Output file
OUTPUT_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"

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
print(f"Loading {FILE_1}...")
file_1 = load_jsonl(FILE_1)
print(f"Loaded {len(file_1)}")

print(f"Loading {FILE_2}...")
file_2 = load_jsonl(FILE_2)
print(f"Loaded {len(file_2)}")

print(f"Loading {FILE_3}...")
file_3 = load_jsonl(FILE_3)
print(f"Loaded {len(file_3)}")

print(f"Loading {FILE_4}...")
file_4 = load_jsonl(FILE_4)
print(f"Loaded {len(file_4)}")

print(f"Loading {FILE_5}...")
file_5 = load_jsonl(FILE_5)
print(f"Loaded {len(file_5)}")

print(f"Loading {FILE_6}...")
file_6 = load_jsonl(FILE_6)
print(f"Loaded {len(file_6)}")

print(f"Loading {FILE_7}...")
file_7 = load_jsonl(FILE_7)
print(f"Loaded {len(file_7)}")

print(f"Loading {FILE_8}...")
file_8 = load_jsonl(FILE_8)
print(f"Loaded {len(file_8)}")

# Merge datasets
merged_data = file_1 + file_2 + file_3 + file_4 + file_5 + file_6 + file_7 + file_8
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