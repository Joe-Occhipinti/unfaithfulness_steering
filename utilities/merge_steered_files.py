import json
from pathlib import Path

# Define input files
base_dir = Path(r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_6_2025-12-15")

input_files = [
    base_dir / "steered_val_gradient_2hidden8_2025-12-07_shard_2.jsonl",
    base_dir / "steered_val_gradient_2hidden8_2025-12-06_shard_1.jsonl",
    base_dir / "steered_val_gradient_2hidden8_2025-12-06_shard_0.jsonl"
]

# Define output file
output_file = base_dir / "steered_val_gradient_2hidden8_2025-12-06.jsonl"

# Merge files
total_records = 0
file_counts = {}

with open(output_file, 'w', encoding='utf-8') as outfile:
    for input_file in input_files:
        if not input_file.exists():
            print(f"WARNING: File not found: {input_file}")
            continue

        count = 0
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                outfile.write(line)
                count += 1
                total_records += 1

        file_counts[input_file.name] = count
        print(f"Added {count} records from {input_file.name}")

print(f"\n{'='*80}")
print(f"Merge complete!")
print(f"{'='*80}")
print(f"Total records in merged file: {total_records}")
print(f"\nBreakdown by source file:")
for filename, count in file_counts.items():
    print(f"  {filename}: {count}")
print(f"\nOutput saved to: {output_file}")
