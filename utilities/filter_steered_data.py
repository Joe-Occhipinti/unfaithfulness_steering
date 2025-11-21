import json
from pathlib import Path

# Input and output file paths
input_file = Path(r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_5_2025-11-15\steered\steered_val_1_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl")
output_file = input_file.parent / f"{input_file.stem}_filtered.jsonl"

# Counters for statistics
total_records = 0
filtered_out_count = 0
kept_records = 0

# Open input and output files
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:

    for line in infile:
        total_records += 1

        # Parse JSON line
        record = json.loads(line.strip())

        # Check filtering conditions
        coefficient = record.get('steering_coefficient')
        layer = record.get('steering_layer')

        # Filter out if coefficient is 2 OR layer is 15
        if coefficient == -2 or coefficient == -2.0 or coefficient == 2 or coefficient == 2.0 or layer == 15:
            filtered_out_count += 1
            continue

        # Keep this record
        kept_records += 1
        outfile.write(line)

# Print statistics
print(f"Filtering complete!")
print(f"Total records processed: {total_records}")
print(f"Records filtered out: {filtered_out_count}")
print(f"Records kept: {kept_records}")
print(f"\nOutput saved to: {output_file}")
