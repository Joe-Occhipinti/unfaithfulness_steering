import json

EXPECTED = {'hinted_id', 'steering_layer', 'steering_coefficient', 'steered_prompt', 
            'hint_template', 'ground_truth_letter', 'hint_letter', 'biased_answer_letter', 
            'split', 'date', 'model', 'steering_mode', 'backend', 'steered_answer_letter', 
            'steered_compliance', 'steered_completeness', 'steered_accuracy', 
            'steered_validation_date', 'rule_classification', 'faithfulness', 'hint_mentioned'}

files = [
    'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_FIXED_NORMALIZED.jsonl',
    'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_val_gradient_2hidden8_2025-12-06_FIXED_NORMALIZED.jsonl',
    'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25_FIXED_NORMALIZED.jsonl',
]

for path in files:
    with open(path, encoding='utf-8') as f:
        rec = json.loads(f.readline())
    actual = set(rec.keys())
    missing = EXPECTED - actual
    extra = actual - EXPECTED
    name = path.split('/')[-1][:40]
    print(name)
    if missing:
        print("  Missing:", sorted(missing))
    else:
        print("  Missing: NONE")
    if extra:
        print("  Extra:", sorted(extra))
    else:
        print("  Extra: NONE")
    print()
