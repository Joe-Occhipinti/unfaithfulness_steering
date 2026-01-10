import json
from pathlib import Path

base_dir = Path('data/definitive_pipeline_data/DeepSeek-Llama-8B')

files = [
    ('off_policy', base_dir / 'annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_FIXED_MERGED.jsonl'),
    ('gradient', base_dir / 'annotated_steered_val_gradient_2hidden8_2025-12-06_FIXED_MERGED.jsonl'),
    ('hintweighting', base_dir / 'annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25_FIXED_MERGED.jsonl'),
]

results = []
for name, path in files:
    if not path.exists():
        results.append({'name': name, 'error': 'FILE NOT FOUND'})
        continue
    
    with open(path, encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    total = len(records)
    hint_count = sum(1 for r in records if r.get('hint_mentioned') is not None)
    rule_count = sum(1 for r in records if r.get('rule_classification') is not None)
    faith_count = sum(1 for r in records if r.get('faithfulness') is not None)
    
    rule_dist = {}
    for r in records:
        rc = r.get('rule_classification') or 'NULL'
        rule_dist[rc] = rule_dist.get(rc, 0) + 1
    
    results.append({
        'name': name,
        'total': total,
        'hint_mentioned_pct': round(100*hint_count/total, 1),
        'rule_classification_pct': round(100*rule_count/total, 1),
        'faithfulness_pct': round(100*faith_count/total, 1),
        'rule_distribution': rule_dist
    })

with open('merged_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to merged_results.json")
