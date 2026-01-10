import json

files = [
    ('linear', 'data/definitive_pipeline_data/Qwen3-32B/annotated_steered_linear_Qwen3-32B_2026-01-04.jsonl'),
    ('mlp', 'data/definitive_pipeline_data/Qwen3-32B/annotated_steered_mlp_Qwen3-32B_2026-01-05.jsonl'),
    ('off_policy', 'data/definitive_pipeline_data/Qwen3-32B/annotated_steered_off_policy_Qwen3-32B_2026-01-04.jsonl')
]

all_fields = {}
for name, path in files:
    with open(path) as f:
        data = json.loads(f.readline())
        all_fields[name] = set(data.keys())
        print(f"{name.upper()} fields ({len(data.keys())}):")
        for k in sorted(data.keys()):
            print(f"  - {k}")
        print()

# Find differences
print("="*60)
print("FIELD DIFFERENCES")
print("="*60)

common = all_fields['linear'] & all_fields['mlp'] & all_fields['off_policy']
print(f"Common ({len(common)}):")
for f in sorted(common):
    print(f"  - {f}")

# Linear vs MLP
linear_vs_mlp = all_fields['linear'].symmetric_difference(all_fields['mlp'])
if linear_vs_mlp:
    print(f"\nLinear vs MLP differences: {sorted(linear_vs_mlp)}")

# Linear vs off_policy
linear_vs_off = all_fields['linear'].symmetric_difference(all_fields['off_policy'])
if linear_vs_off:
    print(f"\nLinear vs Off-policy differences: {sorted(linear_vs_off)}")
