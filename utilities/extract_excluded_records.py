"""
Extract records that were excluded by the filter_hinted_by_length.py script.
Creates a dataset of: original_file MINUS filtered_file
"""

import json
from pathlib import Path

# ============================================================================
# TUNABLE PARAMETERS
# ============================================================================

# Input files
ORIGINAL_FILE = "data/annotated/hinted/psychology_professor_2025-08-15/annotated_global_biased_high_school_psychology_2025-08-15.jsonl"
FILTERED_FILE = "data/annotated/hinted_cut/psycholody_professor_2025-08-15/cut_annotated_global_biased_psychology_professsor_2025-10-12.jsonl"

# Output file
OUTPUT_FILE = "data/annotated/hinted_excluded/psychology_professor_2025-08-15/excluded_annotated_global_biased_high_school_psychology_2025-08-15.jsonl"

# ============================================================================
# MAIN LOGIC
# ============================================================================

def load_jsonl(filepath):
    """Load all records from a JSONL file."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def create_record_key(record):
    """Create a unique key for a record to identify duplicates."""
    # Use question text as the primary identifier
    return record.get('question', '')

def extract_excluded_records(original_file, filtered_file, output_file):
    """Extract records that are in original but not in filtered."""

    print(f"Loading original file: {original_file}")
    original_records = load_jsonl(original_file)
    print(f"  Loaded {len(original_records)} records")

    print(f"\nLoading filtered file: {filtered_file}")
    filtered_records = load_jsonl(filtered_file)
    print(f"  Loaded {len(filtered_records)} records")

    # Create a set of keys from filtered records
    filtered_keys = {create_record_key(rec) for rec in filtered_records}

    # Find excluded records
    excluded_records = [
        rec for rec in original_records
        if create_record_key(rec) not in filtered_keys
    ]

    print(f"\nFound {len(excluded_records)} excluded records")
    print(f"Verification: {len(original_records)} = {len(filtered_records)} + {len(excluded_records)}")

    # Write to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in excluded_records:
            f.write(json.dumps(record) + '\n')

    print(f"\nExcluded records saved to: {output_file}")

    return excluded_records

def main():
    print("Extracting excluded records\n")
    print("=" * 70)

    excluded = extract_excluded_records(ORIGINAL_FILE, FILTERED_FILE, OUTPUT_FILE)

    print("=" * 70)
    print("\nDone!")

if __name__ == "__main__":
    main()
