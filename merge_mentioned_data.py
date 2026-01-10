"""
Merge hint_mentioned and rule_classification from mentioned files into normalized files.

This script:
1. Loads the mentioned file (has per-record hint_mentioned and steered_global_faithfulness_classification)
2. Loads the normalized file (has null hint_mentioned and rule_classification)
3. Matches records by (prompt_index/question_id, steering_layer, steering_coefficient)
4. Fills in hint_mentioned and derives rule_classification
5. Saves the merged file
"""

import json
from pathlib import Path
from datetime import datetime


def derive_rule_classification(steered_global_classification: str) -> str:
    """
    Derive rule_classification from steered_global_faithfulness_classification.
    
    Mapping:
        wrong_to_correct, hint_error -> changed
        faithful, unfaithful -> stable (answer didn't change category)
        incomplete -> incomplete
        error -> error
    """
    if steered_global_classification is None:
        return None
    
    classification = steered_global_classification.lower()
    
    # Changed answers (transitions)
    if classification in ['wrong_to_correct', 'hint_error']:
        return 'changed'
    
    # Stable answers (direct classifications)
    if classification in ['faithful', 'unfaithful']:
        return 'stable'
    
    # Incomplete
    if 'incomplete' in classification:
        return 'incomplete'
    
    # Error cases
    if classification == 'error':
        return 'error'
    
    # Default fallback
    return 'error'


def derive_faithfulness(steered_global_classification: str) -> str:
    """
    Derive faithfulness from steered_global_faithfulness_classification.
    
    Direct values:
        faithful -> faithful
        unfaithful -> unfaithful
        
    Transition values (need interpretation):
        wrong_to_correct -> depends on context (could be faithful now)
        hint_error -> unfaithful (followed the wrong hint)
        incomplete -> None (can't determine)
        error -> None (can't determine)
    """
    if steered_global_classification is None:
        return None
    
    classification = steered_global_classification.lower()
    
    # Direct faithfulness values
    if classification == 'faithful':
        return 'faithful'
    if classification == 'unfaithful':
        return 'unfaithful'
    
    # Transition interpretations
    if classification == 'wrong_to_correct':
        return 'faithful'  # Model corrected itself, now faithful
    if classification == 'hint_error':
        return 'unfaithful'  # Model followed wrong hint
    
    # Can't determine faithfulness for incomplete/error
    return None


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: Path) -> None:
    """Save records to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def build_lookup_key(record: dict) -> tuple:
    """Build lookup key from record."""
    # Get ID from various possible field names
    record_id = record.get('prompt_index', record.get('question_id', record.get('hinted_id')))
    layer = record.get('steering_layer')
    coeff = record.get('steering_coefficient')
    return (record_id, layer, coeff)


def merge_files(mentioned_path: str, normalized_path: str, output_path: str = None) -> dict:
    """
    Merge hint_mentioned and rule_classification from mentioned file into normalized file.
    
    Returns statistics about the merge.
    """
    mentioned_path = Path(mentioned_path)
    normalized_path = Path(normalized_path)
    
    if output_path is None:
        # Replace _NORMALIZED with _MERGED
        output_path = normalized_path.parent / normalized_path.name.replace('_NORMALIZED', '_MERGED')
    else:
        output_path = Path(output_path)
    
    print(f"Loading mentioned file: {mentioned_path.name}")
    mentioned_records = load_jsonl(mentioned_path)
    print(f"  Loaded {len(mentioned_records)} records")
    
    print(f"Loading normalized file: {normalized_path.name}")
    normalized_records = load_jsonl(normalized_path)
    print(f"  Loaded {len(normalized_records)} records")
    
    # Build lookup from mentioned file
    print("Building lookup table...")
    mentioned_lookup = {}
    for record in mentioned_records:
        key = build_lookup_key(record)
        mentioned_lookup[key] = {
            'hint_mentioned': record.get('hint_mentioned'),
            'steered_global_faithfulness_classification': record.get('steered_global_faithfulness_classification'),
            'original_faithfulness_classification': record.get('original_faithfulness_classification'),
        }
    print(f"  Built lookup with {len(mentioned_lookup)} unique keys")
    
    # Merge into normalized records
    print("Merging records...")
    stats = {
        'total': len(normalized_records),
        'matched': 0,
        'unmatched': 0,
        'hint_mentioned_filled': 0,
        'rule_classification_filled': 0,
        'faithfulness_filled': 0,
    }
    
    merged_records = []
    for record in normalized_records:
        key = build_lookup_key(record)
        mentioned_data = mentioned_lookup.get(key)
        
        if mentioned_data:
            stats['matched'] += 1
            steered_class = mentioned_data.get('steered_global_faithfulness_classification')
            
            # Fill hint_mentioned
            if record.get('hint_mentioned') is None and mentioned_data.get('hint_mentioned') is not None:
                record['hint_mentioned'] = mentioned_data['hint_mentioned']
                stats['hint_mentioned_filled'] += 1
            
            # Derive and fill rule_classification
            if record.get('rule_classification') is None and steered_class:
                record['rule_classification'] = derive_rule_classification(steered_class)
                stats['rule_classification_filled'] += 1
            
            # Derive and fill faithfulness
            if record.get('faithfulness') is None and steered_class:
                faithfulness_val = derive_faithfulness(steered_class)
                if faithfulness_val:
                    record['faithfulness'] = faithfulness_val
                    stats['faithfulness_filled'] += 1
        else:
            stats['unmatched'] += 1
        
        merged_records.append(record)
    
    # Save merged file
    print(f"Saving merged file: {output_path.name}")
    save_jsonl(merged_records, output_path)
    print(f"  Saved {len(merged_records)} records")
    
    return stats


def main():
    print("=" * 60)
    print("MERGE MENTIONED DATA INTO NORMALIZED FILES")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Define file pairs to merge
    base_dir = Path('data/definitive_pipeline_data/DeepSeek-Llama-8B')
    
    file_pairs = [
        {
            'mentioned': base_dir / 'mentioned_annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5.jsonl',
            'normalized': base_dir / 'annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_FIXED_NORMALIZED.jsonl',
        },
        {
            'mentioned': base_dir / 'mentioned_annotated_steered_val_gradient_2hidden8_2025-12-06.jsonl',
            'normalized': base_dir / 'annotated_steered_val_gradient_2hidden8_2025-12-06_FIXED_NORMALIZED.jsonl',
        },
        {
            'mentioned': base_dir / 'mentioned_annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl',
            'normalized': base_dir / 'annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25_FIXED_NORMALIZED.jsonl',
        },
    ]
    
    all_stats = []
    for pair in file_pairs:
        print(f"\n{'*' * 60}")
        stats = merge_files(pair['mentioned'], pair['normalized'])
        all_stats.append(stats)
        
        print(f"\nSTATISTICS:")
        print(f"  Total records: {stats['total']}")
        print(f"  Matched: {stats['matched']} ({100*stats['matched']/stats['total']:.1f}%)")
        print(f"  Unmatched: {stats['unmatched']}")
        print(f"  hint_mentioned filled: {stats['hint_mentioned_filled']}")
        print(f"  rule_classification filled: {stats['rule_classification_filled']}")
        print(f"  faithfulness filled: {stats['faithfulness_filled']}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
