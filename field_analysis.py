import json
import sys

EXPECTED = {
    'hinted_id', 'steering_layer', 'steering_coefficient', 'steered_prompt', 
    'hint_template', 'ground_truth_letter', 'hint_letter', 'biased_answer_letter', 
    'split', 'date', 'model', 'steering_mode', 'backend', 'steered_answer_letter', 
    'steered_compliance', 'steered_completeness', 'steered_accuracy', 
    'steered_validation_date', 'rule_classification', 'faithfulness', 'hint_mentioned'
}

files = [
    ('off_policy', 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_FIXED.jsonl'),
    ('gradient', 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_val_gradient_2hidden8_2025-12-06_FIXED.jsonl'),
    ('hintweighting', 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25_FIXED.jsonl'),
]

results = []
for name, path in files:
    with open(path, encoding='utf-8') as f:
        rec = json.loads(f.readline())
    actual = set(rec.keys())
    missing = sorted(EXPECTED - actual)
    extra = sorted(actual - EXPECTED)
    
    # Get sample values for key output fields
    key_values = {}
    for field in ['rule_classification', 'faithfulness', 'hint_mentioned', 'steering_mode', 'backend']:
        key_values[field] = rec.get(field, 'FIELD_MISSING')
    
    results.append({
        'name': name,
        'actual_count': len(actual),
        'actual_fields': sorted(actual),
        'missing': missing,
        'extra': extra,
        'key_values': key_values
    })

with open('field_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("Analysis saved to field_analysis_results.json")
