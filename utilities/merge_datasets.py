import json

def load_jsonl(file_path):
    """Load JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def get_subjects_from_records(records):
    """Extract unique subjects from records."""
    subjects = set()
    for record in records:
        if 'subject' in record:
            subjects.add(record['subject'])
    return subjects

def merge_datasets():
    # Define file paths
    path_1 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\annotated\hinted\annotated_local_biased_all_subjects_no_argument.jsonl"
    path_2 = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\annotated\hinted\annotated_local_biased_argument_only.jsonl"
    output_path = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\merged_annotated_hinted_2025-10-15.jsonl"
    
    print("Loading datasets...")

    # Load run1 and run2 datasets
    records_1 = load_jsonl(path_1)
    records_2 = load_jsonl(path_2)

    # Collect all records from shared subjects
    merged_records = []

    # Add all records from run1 and run2
    merged_records.extend(records_1)
    merged_records.extend(records_2)

    # Save merged dataset
    print(f"\nSaving merged dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return output_path, merged_records

if __name__ == "__main__":
    output_path, records = merge_datasets()
    print(f"\nMerge complete! Output saved to: {output_path}")
