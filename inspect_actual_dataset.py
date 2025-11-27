import pickle

# Load the actual dataset
with open(r'data\sprint_4_2025-10-15\datasets\new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl', 'rb') as f:
    data = pickle.load(f)

print("="*80)
print("ACTUAL DATASET INSPECTION")
print("="*80)

# 1. Check info section
print("\n=== INFO SECTION ===")
print(f"Keys in info: {list(data['info'].keys())}")
print(f"\nmetadata_fields present: {'metadata_fields' in data['info']}")
if 'metadata_fields' in data['info']:
    print(f"metadata_fields value: {data['info']['metadata_fields']}")
    print(f"'split' in metadata_fields: {'split' in data['info']['metadata_fields']}")

# 2. Check first prompt
print("\n=== FIRST PROMPT METADATA ===")
first_prompt_idx = list(data['data'].keys())[0]
first_metadata = data['data'][first_prompt_idx]['metadata']
print(f"Metadata keys: {list(first_metadata.keys())}")
print(f"Full metadata: {first_metadata}")
print(f"\n'split' field present: {'split' in first_metadata}")
if 'split' in first_metadata:
    print(f"Split value: {first_metadata['split']}")

# 3. Check split distribution across ALL prompts
print("\n=== SPLIT DISTRIBUTION (ALL PROMPTS) ===")
split_counts = {}
prompts_with_split = 0
prompts_without_split = 0

for idx in data['data'].keys():
    metadata = data['data'][idx]['metadata']
    if 'split' in metadata:
        prompts_with_split += 1
        split_val = metadata['split']
        split_counts[split_val] = split_counts.get(split_val, 0) + 1
    else:
        prompts_without_split += 1

print(f"Total prompts: {len(data['data'])}")
print(f"Prompts WITH split field: {prompts_with_split}")
print(f"Prompts WITHOUT split field: {prompts_without_split}")

if split_counts:
    print(f"\nSplit distribution:")
    for split_val, count in sorted(split_counts.items()):
        print(f"  {split_val}: {count} prompts ({count/len(data['data'])*100:.1f}%)")
else:
    print("\nNO SPLIT INFORMATION FOUND IN THIS DATASET")

print("\n" + "="*80)
