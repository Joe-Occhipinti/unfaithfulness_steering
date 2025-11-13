"""
Filter annotated steered records based on faithfulness classification criteria.

This script reads the annotated steered dataset and filters records into 4 categories
where each category contains records that satisfy BOTH positive and negative steering conditions.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import date
from collections import defaultdict

def load_jsonl(file_path: str) -> List[Dict]:
    """Load records from a JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def save_jsonl(records: List[Dict], file_path: str):
    """Save records to a JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    print(f"Saved {len(records)} records to {file_path}")

def get_record_key(record: Dict) -> Tuple:
    """Generate a unique key for matching positive/negative coefficient pairs."""
    return (
        record.get('question_id'),
        record.get('prompt_index'),
        record.get('ground_truth_letter'),
        record.get('hint_letter'),
        record.get('original_faithfulness_classification')
    )

def filter_records(records: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Filter records into 4 categories where each record satisfies BOTH conditions.

    Category 1: GT==Hint & Unfaithful (both coeff=+0.75→Faithful AND coeff=-0.75→Unfaithful)
    Category 2: GT==Hint & Faithful (both coeff=-0.75→Unfaithful AND coeff=+0.75→Faithful)
    Category 3: GT!=Hint & Unfaithful (both coeff=+0.75→Faithful AND coeff=-0.75→Unfaithful)
    Category 4: GT!=Hint & Faithful (both coeff=-0.75→Unfaithful AND coeff=+0.75→Faithful)
    """

    # First, organize records by their key and coefficient
    records_by_key = defaultdict(lambda: {'positive': None, 'negative': None})

    for record in records:
        ground_truth = record.get('ground_truth_letter')
        hint = record.get('hint_letter')
        original_faith = record.get('original_faithfulness_classification')
        steered_faith = record.get('steered_global_faithfulness_classification')
        coeff = record.get('steering_coefficient', 0)
        layer = record.get('steering_layer')

        # Skip if missing required fields
        if not all([ground_truth, hint, original_faith, steered_faith]):
            continue

        # Additional constraints: layer must be 8, coefficient must be ±0.75
        if layer != 8:
            continue
        if abs(coeff) != 0.75:
            continue

        key = get_record_key(record)

        if coeff > 0:
            records_by_key[key]['positive'] = record
        else:
            records_by_key[key]['negative'] = record

    # Now find records that have BOTH positive and negative versions
    categories = {
        'category_1': [],  # GT==Hint & Unfaithful (both conditions)
        'category_2': [],  # GT==Hint & Faithful (both conditions)
        'category_3': [],  # GT!=Hint & Unfaithful (both conditions)
        'category_4': [],  # GT!=Hint & Faithful (both conditions)
    }

    for key, pair in records_by_key.items():
        pos_record = pair['positive']
        neg_record = pair['negative']

        # Skip if we don't have both positive and negative versions
        if not pos_record or not neg_record:
            continue

        # Extract common fields
        ground_truth = key[2]
        hint = key[3]
        original_faith = key[4]

        # Normalize faithfulness classifications
        is_original_faithful = original_faith.lower() == 'faithful'
        ground_matches_hint = (ground_truth == hint)

        # Get steered faithfulness for both
        pos_steered_faith = pos_record.get('steered_global_faithfulness_classification', '').lower()
        neg_steered_faith = neg_record.get('steered_global_faithfulness_classification', '').lower()

        is_pos_steered_faithful = pos_steered_faith == 'faithful'
        is_neg_steered_unfaithful = neg_steered_faith != 'faithful'

        # Category 1: GT==Hint & Originally Unfaithful
        # Must have: coeff=+0.75→Faithful AND coeff=-0.75→Unfaithful
        if (ground_matches_hint and
            not is_original_faithful and
            is_pos_steered_faithful and
            is_neg_steered_unfaithful):
            categories['category_1'].extend([pos_record, neg_record])

        # Category 2: GT==Hint & Originally Faithful
        # Must have: coeff=-0.75→Unfaithful AND coeff=+0.75→Faithful
        elif (ground_matches_hint and
              is_original_faithful and
              is_neg_steered_unfaithful and
              is_pos_steered_faithful):
            categories['category_2'].extend([pos_record, neg_record])

        # Category 3: GT!=Hint & Originally Unfaithful
        # Must have: coeff=+0.75→Faithful AND coeff=-0.75→Unfaithful
        elif (not ground_matches_hint and
              not is_original_faithful and
              is_pos_steered_faithful and
              is_neg_steered_unfaithful):
            categories['category_3'].extend([pos_record, neg_record])

        # Category 4: GT!=Hint & Originally Faithful
        # Must have: coeff=-0.75→Unfaithful AND coeff=+0.75→Faithful
        elif (not ground_matches_hint and
              is_original_faithful and
              is_neg_steered_unfaithful and
              is_pos_steered_faithful):
            categories['category_4'].extend([pos_record, neg_record])

    return categories

def generate_summary(categories: Dict[str, List[Dict]]) -> Dict:
    """Generate a summary of the filtering results."""
    summary = {
        'total_records': sum(len(records) for records in categories.values()),
        'total_unique_prompts': sum(len(records) // 2 for records in categories.values()),
        'categories': {}
    }

    category_descriptions = {
        'category_1': 'GT==Hint & Unfaithful (both +0.75→Faithful AND -0.75→Unfaithful)',
        'category_2': 'GT==Hint & Faithful (both -0.75→Unfaithful AND +0.75→Faithful)',
        'category_3': 'GT!=Hint & Unfaithful (both +0.75→Faithful AND -0.75→Unfaithful)',
        'category_4': 'GT!=Hint & Faithful (both -0.75→Unfaithful AND +0.75→Faithful)',
    }

    for cat_name, records in categories.items():
        num_pairs = len(records) // 2
        summary['categories'][cat_name] = {
            'description': category_descriptions[cat_name],
            'total_records': len(records),
            'unique_prompts': num_pairs,
            'percentage_of_prompts': f"{num_pairs / summary['total_unique_prompts'] * 100:.2f}%" if summary['total_unique_prompts'] > 0 else "0%"
        }

    return summary

def main():
    # Input file
    input_file = r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_4_2025-10-15\annotated\steered\annotated_steered_sprint4_2025-10-27.jsonl"

    # Output directory
    output_dir = Path(r"C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_4_2025-10-15\annotated\steered\filtered")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load records
    print(f"Loading records from {input_file}...")
    records = load_jsonl(input_file)
    print(f"Loaded {len(records)} total records")

    # Filter records
    print("\nFiltering records (matching positive and negative coefficient pairs)...")
    categories = filter_records(records)

    # Save each category to a separate file
    today = date.today().strftime("%Y-%m-%d")
    for cat_name, cat_records in categories.items():
        if cat_records:  # Only save non-empty categories
            output_file = output_dir / f"{cat_name}_{today}.jsonl"
            save_jsonl(cat_records, str(output_file))

    # Generate and save summary
    summary = generate_summary(categories)
    summary_file = output_dir / f"filtering_summary_{today}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")

    # Print summary
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)
    print(f"Total records (including both +/- coefficients): {summary['total_records']}")
    print(f"Total unique prompts (with both coefficients): {summary['total_unique_prompts']}")
    print("\nCategory breakdown:")
    for cat_name, cat_info in summary['categories'].items():
        print(f"\n{cat_name}:")
        print(f"  Description: {cat_info['description']}")
        print(f"  Total records: {cat_info['total_records']} (pos + neg)")
        print(f"  Unique prompts: {cat_info['unique_prompts']}")
        print(f"  Percentage of prompts: {cat_info['percentage_of_prompts']}")

if __name__ == "__main__":
    main()
