"""
analyze_jsonl_stratification.py

Count records in the annotated JSONL file stratified by:
- split (train vs val)
- hint_template
- global_faithfulness_classification
- hint_correctness (hint_letter == ground_truth_letter vs !=)
"""

import json
from collections import defaultdict

# Input file
JSONL_FILE = "data/sprint_4_2025-10-15/annotated/touse_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"

print("=" * 80)
print("JSONL FILE STRATIFICATION ANALYSIS")
print("=" * 80)
print(f"\nFile: {JSONL_FILE}")

# Initialize counters
total_records = 0
split_counts = defaultdict(int)
hint_template_counts = defaultdict(lambda: defaultdict(int))
full_stratification = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
hint_correctness_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
full_stratification_with_hint = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

# Parse file
print("\nParsing file...")
with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
            total_records += 1

            # Extract fields
            split = record.get('split', 'unknown')
            hint_template = record.get('hint_template', 'unknown')
            global_faithfulness = record.get('global_faithfulness_classification', 'unknown')
            hint_letter = record.get('hint_letter', 'unknown')
            ground_truth_letter = record.get('ground_truth_letter', 'unknown')

            # Determine hint correctness
            if hint_letter == ground_truth_letter:
                hint_correctness = 'correct_hint'
            else:
                hint_correctness = 'incorrect_hint'

            # Update counters
            split_counts[split] += 1
            hint_template_counts[split][hint_template] += 1
            full_stratification[split][hint_template][global_faithfulness] += 1
            hint_correctness_counts[split][hint_template][hint_correctness] += 1
            full_stratification_with_hint[split][hint_template][hint_correctness][global_faithfulness] += 1

        except json.JSONDecodeError as e:
            print(f"WARNING: Could not parse line {line_num}: {e}")
            continue

print(f"Total records parsed: {total_records}")

# Print results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

# 1. Split counts
print("\n=== Split Distribution ===")
for split in sorted(split_counts.keys()):
    count = split_counts[split]
    percentage = (count / total_records) * 100
    print(f"{split:10s}: {count:5d} records ({percentage:5.2f}%)")

# 2. Split × Hint Template
print("\n=== Split × Hint Template ===")
for split in sorted(hint_template_counts.keys()):
    print(f"\n{split}:")
    split_total = split_counts[split]
    for hint in sorted(hint_template_counts[split].keys()):
        count = hint_template_counts[split][hint]
        percentage = (count / split_total) * 100
        print(f"  {hint:20s}: {count:5d} records ({percentage:5.2f}%)")

# 3. Hint Correctness Distribution (Split × Hint Template × Correct/Incorrect)
print("\n=== Split × Hint Template × Hint Correctness ===")
for split in sorted(hint_correctness_counts.keys()):
    print(f"\n{split}:")
    for hint in sorted(hint_correctness_counts[split].keys()):
        print(f"  {hint}:")
        hint_total = hint_template_counts[split][hint]
        for correctness in sorted(hint_correctness_counts[split][hint].keys()):
            count = hint_correctness_counts[split][hint][correctness]
            percentage = (count / hint_total) * 100
            print(f"    {correctness:20s}: {count:5d} records ({percentage:5.2f}%)")

# 4. Full Stratification (Split × Hint Template × Global Faithfulness)
print("\n=== Full Stratification: Split × Hint Template × Global Faithfulness ===")
for split in sorted(full_stratification.keys()):
    print(f"\n{split}:")
    for hint in sorted(full_stratification[split].keys()):
        print(f"  {hint}:")
        hint_total = hint_template_counts[split][hint]
        for faithfulness in sorted(full_stratification[split][hint].keys()):
            count = full_stratification[split][hint][faithfulness]
            percentage = (count / hint_total) * 100
            print(f"    {faithfulness:30s}: {count:5d} records ({percentage:5.2f}%)")

# 5. Full Stratification with Hint Correctness (Split × Hint Template × Hint Correctness × Global Faithfulness)
print("\n=== Full Stratification: Split × Hint Template × Hint Correctness × Global Faithfulness ===")
for split in sorted(full_stratification_with_hint.keys()):
    print(f"\n{split}:")
    for hint in sorted(full_stratification_with_hint[split].keys()):
        print(f"  {hint}:")
        for correctness in sorted(full_stratification_with_hint[split][hint].keys()):
            print(f"    {correctness}:")
            correctness_total = hint_correctness_counts[split][hint][correctness]
            for faithfulness in sorted(full_stratification_with_hint[split][hint][correctness].keys()):
                count = full_stratification_with_hint[split][hint][correctness][faithfulness]
                percentage = (count / correctness_total) * 100 if correctness_total > 0 else 0
                print(f"      {faithfulness:28s}: {count:5d} records ({percentage:5.2f}%)")

# 6. Global Faithfulness Distribution (across all splits)
print("\n=== Global Faithfulness Distribution (all splits) ===")
global_faithfulness_counts = defaultdict(int)
for split in full_stratification:
    for hint in full_stratification[split]:
        for faithfulness, count in full_stratification[split][hint].items():
            global_faithfulness_counts[faithfulness] += count

for faithfulness in sorted(global_faithfulness_counts.keys()):
    count = global_faithfulness_counts[faithfulness]
    percentage = (count / total_records) * 100
    print(f"{faithfulness:30s}: {count:5d} records ({percentage:5.2f}%)")

# 7. Global Hint Correctness Distribution (across all splits)
print("\n=== Global Hint Correctness Distribution (all splits) ===")
global_hint_correctness_counts = defaultdict(int)
for split in hint_correctness_counts:
    for hint in hint_correctness_counts[split]:
        for correctness, count in hint_correctness_counts[split][hint].items():
            global_hint_correctness_counts[correctness] += count

for correctness in sorted(global_hint_correctness_counts.keys()):
    count = global_hint_correctness_counts[correctness]
    percentage = (count / total_records) * 100
    print(f"{correctness:20s}: {count:5d} records ({percentage:5.2f}%)")

# 8. Summary table by split
print("\n=== Summary Table: Global Faithfulness by Split ===")
split_faithfulness_counts = defaultdict(lambda: defaultdict(int))
for split in full_stratification:
    for hint in full_stratification[split]:
        for faithfulness, count in full_stratification[split][hint].items():
            split_faithfulness_counts[split][faithfulness] += count

# Get all unique faithfulness labels
all_faithfulness_labels = set()
for split in split_faithfulness_counts:
    all_faithfulness_labels.update(split_faithfulness_counts[split].keys())

# Print header
print(f"{'Faithfulness':30s} ", end="")
for split in sorted(split_counts.keys()):
    print(f"{split:>10s} ", end="")
print()
print("-" * 80)

# Print rows
for faithfulness in sorted(all_faithfulness_labels):
    print(f"{faithfulness:30s} ", end="")
    for split in sorted(split_counts.keys()):
        count = split_faithfulness_counts[split][faithfulness]
        print(f"{count:10d} ", end="")
    print()

# 9. Summary table: Hint Correctness by Split
print("\n=== Summary Table: Hint Correctness by Split ===")
split_hint_correctness_counts = defaultdict(lambda: defaultdict(int))
for split in hint_correctness_counts:
    for hint in hint_correctness_counts[split]:
        for correctness, count in hint_correctness_counts[split][hint].items():
            split_hint_correctness_counts[split][correctness] += count

# Print header
print(f"{'Hint Correctness':30s} ", end="")
for split in sorted(split_counts.keys()):
    print(f"{split:>10s} ", end="")
print()
print("-" * 80)

# Print rows
for correctness in ['correct_hint', 'incorrect_hint']:
    print(f"{correctness:30s} ", end="")
    for split in sorted(split_counts.keys()):
        count = split_hint_correctness_counts[split][correctness]
        print(f"{count:10d} ", end="")
    print()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
