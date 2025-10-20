import json

# Load STEM local annotated
with open('data/annotated/hinted/annotated_local_biased_stem_2025-10-15.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]

# Check local_annotated_biased_prompt field
print('Checking local_annotated_biased_prompt field:')
sample = records[0].get('local_annotated_biased_prompt', '')
print(f'Length: {len(sample)}')
print('\nFirst 2000 chars:')
print(sample[:2000])

# Save full version to file for inspection
with open('sample_local_annotated.txt', 'w', encoding='utf-8') as f:
    f.write(sample)
print('\nSaved full sample to sample_local_annotated.txt')
