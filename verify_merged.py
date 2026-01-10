import json

# Check the merged file
path = 'data/definitive_pipeline_data/DeepSeek-Llama-8B/annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_FIXED_MERGED.jsonl'

with open(path, encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]

# Sample record
rec = records[0]
print(f"Total records: {len(records)}")
print(f"\nSAMPLE RECORD KEY FIELDS:")
print(f"  hinted_id: {rec.get('hinted_id')}")
print(f"  steering_layer: {rec.get('steering_layer')}")
print(f"  steering_coefficient: {rec.get('steering_coefficient')}")
print(f"  steering_mode: {rec.get('steering_mode')}")
print(f"  hint_mentioned: {rec.get('hint_mentioned')}")
print(f"  rule_classification: {rec.get('rule_classification')}")
print(f"  faithfulness: {rec.get('faithfulness')}")

# Count non-null values for key fields
hint_mentioned_count = sum(1 for r in records if r.get('hint_mentioned') is not None)
rule_class_count = sum(1 for r in records if r.get('rule_classification') is not None)
faithfulness_count = sum(1 for r in records if r.get('faithfulness') is not None)

print(f"\nFIELD FILL RATES:")
print(f"  hint_mentioned: {hint_mentioned_count}/{len(records)} ({100*hint_mentioned_count/len(records):.1f}%)")
print(f"  rule_classification: {rule_class_count}/{len(records)} ({100*rule_class_count/len(records):.1f}%)")
print(f"  faithfulness: {faithfulness_count}/{len(records)} ({100*faithfulness_count/len(records):.1f}%)")

# Check rule_classification distribution
rule_dist = {}
for r in records:
    rc = r.get('rule_classification', 'NULL')
    rule_dist[rc] = rule_dist.get(rc, 0) + 1
print(f"\nRULE_CLASSIFICATION DISTRIBUTION:")
for k, v in sorted(rule_dist.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
