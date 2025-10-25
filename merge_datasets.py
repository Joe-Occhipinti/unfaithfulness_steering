import json
from pathlib import Path

# Define target subjects
TARGET_SUBJECTS = {
    "high_school_psychology",
    "college_biology",
    "high_school_biology",
    "college_chemistry",
    "high_school_chemistry",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history"
}

# Input file paths
input_files = [
    r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\behavioural\baseline\baseline_economics_history_psychology_2025-10-19.jsonl",
    r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\behavioural\baseline_stem_2025-10-15.jsonl"
]

# Output directory and file
output_dir = Path(r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\behavioural")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "baseline_filtered_2025-10-25.jsonl"

# Collect filtered records
filtered_records = []

# Process each input file
for input_file in input_files:
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                if record.get("subject") in TARGET_SUBJECTS:
                    filtered_records.append(record)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {input_file}: {e}")

# Write filtered records to output file
print(f"\nWriting {len(filtered_records)} records to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for record in filtered_records:
        f.write(json.dumps(record) + '\n')

# Print summary statistics
subject_counts = {}
for record in filtered_records:
    subject = record.get("subject")
    subject_counts[subject] = subject_counts.get(subject, 0) + 1

print(f"\nSummary:")
print(f"Total records: {len(filtered_records)}")
print(f"\nRecords per subject:")
for subject in sorted(subject_counts.keys()):
    print(f"  {subject}: {subject_counts[subject]}")

print(f"\nOutput saved to: {output_file}")
