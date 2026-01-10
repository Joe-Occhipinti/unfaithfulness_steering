import json
from collections import Counter

path = 'data/definitive_pipeline_data/DeepSeek-Llama-8B/mentioned_annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5.jsonl'

with open(path, encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

# Find all fields with "faith" in the name
faith_fields = [k for k in records[0].keys() if 'faith' in k.lower()]

# Get distinct values for each
result = {}
for field in faith_fields:
    vals = Counter(r.get(field) for r in records)
    result[field] = dict(vals.most_common())

with open('faith_values.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Saved to faith_values.json")
