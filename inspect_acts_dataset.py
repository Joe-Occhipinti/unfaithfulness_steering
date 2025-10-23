import pickle
import json

# Load the dataset
data = pickle.load(open(r'C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\datasets\acts_pos_bas_psyXprof_2025-10-19.pkl', 'rb'))

print('=== TOP-LEVEL KEYS ===')
print(list(data.keys()))

print('\n=== INFO ===')
print(json.dumps(data['info'], indent=2, default=str))

print('\n=== DATA STRUCTURE ===')
print('Type of data:', type(data['data']))
print('Number of prompts:', len(data['data']))

first_prompt_idx = list(data['data'].keys())[0]
print(f'\nFirst prompt index: {first_prompt_idx}')
print('Keys in first prompt:', list(data['data'][first_prompt_idx].keys()))

print('\n=== CHECKING STRUCTURE ===')
prompt_data = data['data'][first_prompt_idx]

if 'layers' in prompt_data:
    print('Structure: NEW (nested layers)')
    layers = prompt_data['layers']
    print(f'Number of layers: {len(layers)}')
    print(f'Layers available: {sorted(layers.keys())[:5]}...')
    layer_0 = layers[0]
    print(f'\nTags in layer 0: {list(layer_0.keys())}')
    for tag in layer_0:
        tensor = layer_0[tag]
        print(f'  {tag}: shape {tensor.shape if hasattr(tensor, "shape") else "N/A"}')

    if 'metadata' in prompt_data:
        print(f'\n=== METADATA ===')
        print(f'Metadata fields: {list(prompt_data["metadata"].keys())}')
        print('\nMetadata sample:')
        for k, v in list(prompt_data['metadata'].items())[:15]:
            print(f'  {k}: {v}')
else:
    print('Structure: OLD (direct layer access)')
    layer_keys = sorted([k for k in prompt_data.keys() if isinstance(k, int)])
    print(f'Available layers: {layer_keys[:5]}...')
    layer_0 = prompt_data[0]
    print(f'\nTags in layer 0: {list(layer_0.keys())}')
    for tag in layer_0:
        tensor = layer_0[tag]
        print(f'  {tag}: shape {tensor.shape if hasattr(tensor, "shape") else "N/A"}')

# Check a few more prompts to see variation
print('\n=== SAMPLE OF PROMPTS ===')
for i, prompt_idx in enumerate(list(data['data'].keys())[:3]):
    prompt_info = data['data'][prompt_idx]

    # Get layer data
    if 'layers' in prompt_info:
        layer_0 = prompt_info['layers'][0]
    else:
        layer_0 = prompt_info[0]

    tags_present = [tag for tag in layer_0.keys() if layer_0[tag].numel() > 0]
    tag_counts = {tag: layer_0[tag].shape[0] for tag in tags_present}

    print(f'\nPrompt {prompt_idx}:')
    print(f'  Tags with activations: {tags_present}')
    print(f'  Activation counts: {tag_counts}')

    if 'metadata' in prompt_info:
        meta = prompt_info['metadata']
        if 'split' in meta:
            print(f'  Split: {meta["split"]}')
        if 'faithfulness_classification' in meta:
            print(f'  Faithfulness: {meta["faithfulness_classification"]}')
