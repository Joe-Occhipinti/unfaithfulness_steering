"""
Script to fix DeepSeek dataset fields to match eval_steered_global_faithfulness.py output schema.

Transformations:
1. Rename fields:
   - completeness -> steered_completeness
   - compliance -> steered_compliance
   - validation_date -> steered_validation_date
   - prompt_index or question_id -> hinted_id (use existing value)

2. Add missing fields with null/derived values:
   - backend: null (no data available)
   - steering_mode: derive from filename or set based on dataset type
   - faithfulness: null (needs to be computed by script)
   - hint_mentioned: null (needs to be computed by script)
   - rule_classification: null (needs to be computed by script)
"""

import json
from pathlib import Path
from datetime import datetime

# Files to process
FILES = [
    {
        'path': 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_FIXED.jsonl',
        'steering_mode': 'off_policy'
    },
    {
        'path': 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_val_gradient_2hidden8_2025-12-06_FIXED.jsonl',
        'steering_mode': 'mlp'  # gradient -> mlp
    },
    {
        'path': 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25_FIXED.jsonl',
        'steering_mode': 'linear'  # hintweighting -> linear
    },
]


def transform_record(record: dict, steering_mode: str) -> dict:
    """Transform a single record to match expected schema."""
    new_record = {}
    
    # === Core identification ===
    # Get hinted_id from prompt_index or question_id
    new_record['hinted_id'] = record.get('prompt_index', record.get('question_id'))
    new_record['steering_layer'] = record.get('steering_layer')
    new_record['steering_coefficient'] = record.get('steering_coefficient')
    
    # === Content fields ===
    new_record['steered_prompt'] = record.get('steered_prompt')
    new_record['hint_template'] = record.get('hint_template')
    new_record['ground_truth_letter'] = record.get('ground_truth_letter')
    new_record['hint_letter'] = record.get('hint_letter')
    new_record['biased_answer_letter'] = record.get('biased_answer_letter')
    
    # === Metadata ===
    new_record['split'] = record.get('split')
    new_record['date'] = record.get('date')
    new_record['model'] = record.get('model')
    new_record['steering_mode'] = steering_mode  # Derived from filename
    new_record['backend'] = None  # No data available
    
    # === Steered answer evaluation (rename fields) ===
    new_record['steered_answer_letter'] = record.get('steered_answer_letter')
    new_record['steered_compliance'] = record.get('compliance')  # Renamed
    new_record['steered_completeness'] = record.get('completeness')  # Renamed
    new_record['steered_accuracy'] = record.get('steered_accuracy')
    new_record['steered_validation_date'] = record.get('validation_date')  # Renamed
    
    # === Output fields from eval_steered_global_faithfulness.py ===
    # These need to be computed by the script, leave as null
    new_record['rule_classification'] = None
    new_record['faithfulness'] = None
    new_record['hint_mentioned'] = None
    
    return new_record


def process_file(file_info: dict) -> None:
    """Process a single file and save the transformed version."""
    path = Path(file_info['path'])
    steering_mode = file_info['steering_mode']
    
    print(f"\n{'='*60}")
    print(f"Processing: {path.name}")
    print(f"Steering mode: {steering_mode}")
    print('='*60)
    
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return
    
    # Read all records
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records")
    
    # Transform records
    transformed = []
    for record in records:
        transformed.append(transform_record(record, steering_mode))
    
    # Create output path (overwrite original with _NORMALIZED suffix)
    output_path = path.parent / f"{path.stem}_NORMALIZED.jsonl"
    
    # Save transformed records
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in transformed:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(transformed)} records to: {output_path}")
    
    # Verify fields
    if transformed:
        print(f"\nVerification - Fields in output ({len(transformed[0])} fields):")
        for field in sorted(transformed[0].keys()):
            value = transformed[0][field]
            if value is None:
                print(f"  - {field}: null")
            else:
                print(f"  - {field}: has value")


def main():
    print("="*60)
    print("DEEPSEEK DATASET FIELD NORMALIZATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    for file_info in FILES:
        process_file(file_info)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
