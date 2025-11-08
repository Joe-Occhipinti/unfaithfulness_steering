import json
import os
from pathlib import Path
"""
Used for targeted qualitative analysis  of steered records based on specific parameters.
Should be put into an utils folder later.
"""

# ============================================
# TUNABLE PARAMETERS
# ============================================
STEERING_LAYER = 31  # For local high-school-psychology-professor: 8, 13, 18, 23, 28, 31
STEERING_COEFFICIENT = 2 # For local high-school-psychology-professor: (+-) 0.5, 0.6, 0.75, 1.0, 2, 3
ORIGINAL_CLASSIFICATION = "unfaithful"  # e.g., "faithful", "unfaithful"
FINAL_CLASSIFICATION = "unfaithful"  # e.g., "correct", "unfaithful", "incomplete", etc.

# Input file paths
STEERED_LOCAL_FILE = r"data\annotated\annotated_steered_local_high_school_psychology_professor_2025-10-06.jsonl"
ANNOTATED_LOCAL_BIASED_FILE = r"data\annotated\annotated_local_biased_high_school_psychology_2025-08-15.jsonl"

# Output directory
OUTPUT_DIR = r"steering qualitative analysis"

# ============================================
# SCRIPT
# ============================================

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

def create_question_map(records):
    """Create a mapping from question text to record."""
    question_map = {}
    for record in records:
        question = record.get('question')
        if question:
            question_map[question] = record
    return question_map

def filter_and_extract(steered_records, local_biased_map):
    """Filter steered records and extract required fields."""
    filtered_results = []

    for record in steered_records:
        # Apply filters
        if record.get('steering_layer') != STEERING_LAYER:
            continue
        if record.get('steering_coefficient') != STEERING_COEFFICIENT:
            continue
        if record.get('original_faithfulness_classification') != ORIGINAL_CLASSIFICATION:
            continue
        if record.get('steered_global_faithfulness_classification') != FINAL_CLASSIFICATION:
            continue

        # Find matching record in local biased dataset
        question = record.get('question')
        local_biased_record = local_biased_map.get(question)

        if not local_biased_record:
            print(f"Warning: No matching local biased record found for question: {question[:50]}...")
            continue

        # Extract required fields
        result = {
            'local_annotated_biased_prompt': local_biased_record.get('local_annotated_biased_prompt'),
            'original_classification': record.get('original_faithfulness_classification'),
            'steered_prompt': record.get('annotated_steered_prompt'),
            'steered_classification': record.get('steered_global_faithfulness_classification')
        }

        filtered_results.append(result)

    return filtered_results

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load datasets
    print("Loading steered local dataset...")
    steered_records = load_jsonl(STEERED_LOCAL_FILE)
    print(f"Loaded {len(steered_records)} steered records")

    print("Loading annotated local biased dataset...")
    local_biased_records = load_jsonl(ANNOTATED_LOCAL_BIASED_FILE)
    print(f"Loaded {len(local_biased_records)} local biased records")

    # Create question mapping for local biased dataset
    local_biased_map = create_question_map(local_biased_records)

    # Filter and extract
    print(f"\nFiltering with parameters:")
    print(f"  Layer: {STEERING_LAYER}")
    print(f"  Coefficient: {STEERING_COEFFICIENT}")
    print(f"  Original: {ORIGINAL_CLASSIFICATION}")
    print(f"  Final: {FINAL_CLASSIFICATION}")

    filtered_results = filter_and_extract(steered_records, local_biased_map)

    print(f"\nFound {len(filtered_results)} matching records")

    # Save output
    output_filename = f"filtered_layer{STEERING_LAYER}_coef{STEERING_COEFFICIENT}_{ORIGINAL_CLASSIFICATION}_to_{FINAL_CLASSIFICATION}.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in filtered_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
