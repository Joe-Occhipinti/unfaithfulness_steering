import pickle
import torch

# Load the pickle file
with open(r'C:\Users\occhi\Desktop\unfaithfulness_steering\data\sprint_4_2025-10-15\datasets\new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl', 'rb') as f:
    dataset = pickle.load(f)

print("="*80)
print("COMPLETE DATASET STRUCTURE")
print("="*80)

print("\n### TOP LEVEL ###")
print(f"Type: {type(dataset)}")
print(f"Keys: {list(dataset.keys())}")

print("\n### INFO SECTION ###")
info = dataset['info']
for key, value in info.items():
    print(f"\n{key}:")
    if isinstance(value, (list, dict)):
        print(f"  Type: {type(value)}")
        print(f"  Length: {len(value)}")
        if isinstance(value, dict) and len(value) <= 10:
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, list) and len(value) <= 10:
            print(f"  Values: {value}")
        else:
            if isinstance(value, dict):
                print(f"  First 3 items: {dict(list(value.items())[:3])}")
            else:
                print(f"  First 3 items: {value[:3]}")
    else:
        print(f"  Value: {value}")

print("\n### DATA SECTION STRUCTURE ###")
data = dataset['data']
print(f"Number of prompts: {len(data)}")
print(f"Prompt IDs (first 10): {list(data.keys())[:10]}")

# Examine first prompt in detail
first_prompt_id = list(data.keys())[0]
first_prompt = data[first_prompt_id]

print(f"\n### STRUCTURE OF PROMPT {first_prompt_id} ###")
print(f"Keys: {list(first_prompt.keys())}")

print("\n--- Metadata ---")
metadata = first_prompt['metadata']
for key, value in metadata.items():
    print(f"  {key}: {value}")

print("\n--- Layers ---")
layers = first_prompt['layers']
print(f"Number of layers: {len(layers)}")
print(f"Layer IDs: {list(layers.keys())}")

# Examine first layer
first_layer_id = list(layers.keys())[0]
first_layer = layers[first_layer_id]
print(f"\n--- Structure of Layer {first_layer_id} ---")
print(f"Type: {type(first_layer)}")
print(f"Keys: {list(first_layer.keys())}")

for tag_key, tag_value in first_layer.items():
    print(f"\n  Tag: {tag_key}")
    print(f"    Type: {type(tag_value)}")
    if isinstance(tag_value, torch.Tensor):
        print(f"    Shape: {tag_value.shape}")
        print(f"    Dtype: {tag_value.dtype}")
        print(f"    Device: {tag_value.device}")
        print(f"    First 3 values: {tag_value[:3].tolist() if tag_value.numel() >= 3 else tag_value.tolist()}")
    elif isinstance(tag_value, list):
        print(f"    Length: {len(tag_value)}")
        if len(tag_value) > 0 and isinstance(tag_value[0], torch.Tensor):
            print(f"    First element shape: {tag_value[0].shape}")
            print(f"    First element dtype: {tag_value[0].dtype}")

print("\n### SUMMARY ###")
print(f"Dataset contains:")
print(f"  - {len(data)} prompts")
print(f"  - {len(layers)} layers per prompt")
print(f"  - Tags per layer: {list(first_layer.keys())}")
print(f"  - Metadata per prompt: {list(metadata.keys())}")
print(f"  - Info fields: {list(info.keys())}")

print("\n" + "="*80)
