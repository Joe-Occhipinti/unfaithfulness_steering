import json
from pathlib import Path

# Expected fields from eval_steered_global_faithfulness.py output
# Based on Qwen3-32B annotated files
EXPECTED_FIELDS = {
    # Core identification
    'hinted_id',
    'steering_layer', 
    'steering_coefficient',
    
    # Content fields
    'steered_prompt',
    'hint_template',
    'ground_truth_letter',
    'hint_letter',
    'biased_answer_letter',
    
    # Metadata
    'split',
    'date',
    'model',
    'steering_mode',
    'backend',
    
    # Steered answer evaluation
    'steered_answer_letter',
    'steered_compliance',
    'steered_completeness',
    'steered_accuracy',
    'steered_validation_date',
    
    # OUTPUT from eval_steered_global_faithfulness.py
    'rule_classification',
    'faithfulness',
    'hint_mentioned',
}

# DeepSeek files to analyze
files = [
    ('off_policy', 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_FIXED.jsonl'),
    ('gradient', 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_val_gradient_2hidden8_2025-12-06_FIXED.jsonl'),
    ('hintweighting', 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25_FIXED.jsonl'),
]

print("="*80)
print("DEEPSEEK DATASET FIELD ANALYSIS")
print("="*80)

for name, path in files:
    print(f"\n{'='*80}")
    print(f"DATASET: {name}")
    print(f"File: {Path(path).name}")
    print("="*80)
    
    try:
        with open(path) as f:
            # Get first record
            first_record = json.loads(f.readline())
            actual_fields = set(first_record.keys())
            
            # Count total records
            f.seek(0)
            total_records = sum(1 for _ in f)
            print(f"Total records: {total_records}")
            
        print(f"\nActual fields ({len(actual_fields)}):")
        for field in sorted(actual_fields):
            status = "OK" if field in EXPECTED_FIELDS else "EXTRA"
            print(f"  - {field} [{status}]")
        
        # Missing fields
        missing = EXPECTED_FIELDS - actual_fields
        if missing:
            print(f"\nMISSING FIELDS ({len(missing)}):")
            for field in sorted(missing):
                print(f"  * {field}")
        else:
            print("\nNo missing fields!")
        
        # Extra fields (not in expected)
        extra = actual_fields - EXPECTED_FIELDS
        if extra:
            print(f"\nEXTRA FIELDS ({len(extra)}):")
            for field in sorted(extra):
                print(f"  + {field}")
        
        # Check for null/missing values in key fields
        print("\nCHECKING VALUES IN FIRST RECORD:")
        key_output_fields = ['rule_classification', 'faithfulness', 'hint_mentioned']
        for field in key_output_fields:
            if field in first_record:
                val = first_record[field]
                print(f"  {field}: {val} (type: {type(val).__name__})")
            else:
                print(f"  {field}: FIELD MISSING")
                
    except FileNotFoundError:
        print(f"  ERROR: File not found!")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
