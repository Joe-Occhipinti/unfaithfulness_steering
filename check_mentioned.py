import json

path = 'data/definitive_pipeline_data/DeepSeek-Llama-8B/mentioned_annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5.jsonl'

with open(path, encoding='utf-8') as f:
    rec = json.loads(f.readline())
    f.seek(0)
    total = sum(1 for _ in f)

output = {
    "total_records": total,
    "num_fields": len(rec),
    "fields": sorted(rec.keys()),
    "sample_values": {k: (str(v)[:100] if isinstance(v, str) else v) for k, v in rec.items()}
}

with open('mentioned_structure.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print("Saved to mentioned_structure.json")
