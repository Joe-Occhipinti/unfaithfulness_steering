import pickle
from collections import defaultdict

# Load the dataset
with open(r'data\sprint_4_2025-10-15\datasets\new_scie_hist_psy_X_grader_prof_meta_2025-10-25.pkl', 'rb') as f:
    data = pickle.load(f)

print("="*80)
print("ACTIVATION COUNTS BY CLASS")
print("Filtering: correct_hint='False' only")
print("Layer: 0 (representative layer)")
print("="*80)

# Initialize tracking structure
# Key: (tag, hint_template, split)
# Value: {'num_activations': int, 'num_prompts': int}
counts = defaultdict(lambda: {'num_activations': 0, 'num_prompts': 0})

# Tags we're interested in
target_tags = ['F_body', 'U_body']

# Track unique hint templates
hint_templates_found = set()

# Iterate through all prompts
for prompt_idx, prompt_data in data['data'].items():
    metadata = prompt_data['metadata']
    
    # Filter: only correct_hint = "False"
    if metadata['correct_hint'] != 'False':
        continue
    
    hint_template = metadata['hint_template']
    split = metadata['split']
    hint_templates_found.add(hint_template)
    
    # Access layer 0
    layers = prompt_data['layers']
    if 0 not in layers:
        continue
    
    layer_0 = layers[0]
    
    # Count activations for each target tag
    for tag in target_tags:
        if tag in layer_0:
            tensor = layer_0[tag]
            num_activations = tensor.shape[0]  # Number of rows = number of activations
            
            # Update counts
            class_key = (tag, hint_template, split)
            counts[class_key]['num_activations'] += num_activations
            counts[class_key]['num_prompts'] += 1

# Display results
print(f"\nFound {len(hint_templates_found)} unique hint templates:")
for ht in sorted(hint_templates_found):
    print(f"  - {ht}")

# Group and display by tag
for tag in target_tags:
    print(f"\n{'='*80}")
    print(f"{tag} CLASSES (correct_hint=False)")
    print(f"{'='*80}")
    
    # Get all hint templates for this tag
    tag_templates = set()
    for (t, ht, split) in counts.keys():
        if t == tag:
            tag_templates.add(ht)
    
    # Display each hint template class
    for hint_template in sorted(tag_templates):
        print(f"\n--- Class: {tag} × {hint_template} × correct_hint=False ---")
        
        # Get train and val counts
        train_key = (tag, hint_template, 'train')
        val_key = (tag, hint_template, 'val')
        
        train_acts = counts[train_key]['num_activations']
        train_prompts = counts[train_key]['num_prompts']
        val_acts = counts[val_key]['num_activations']
        val_prompts = counts[val_key]['num_prompts']
        
        total_acts = train_acts + val_acts
        total_prompts = train_prompts + val_prompts
        
        print(f"  Train: {train_acts:,} activations from {train_prompts} prompts")
        print(f"  Val:   {val_acts:,} activations from {val_prompts} prompts")
        print(f"  Total: {total_acts:,} activations from {total_prompts} prompts")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

total_classes = len(hint_templates_found) * len(target_tags)
print(f"Total classes analyzed: {total_classes} ({len(target_tags)} tags × {len(hint_templates_found)} hint_templates)")

# Calculate totals per tag
for tag in target_tags:
    total_acts = sum(counts[(t, ht, s)]['num_activations'] 
                     for (t, ht, s) in counts.keys() if t == tag)
    total_prompts = sum(counts[(t, ht, s)]['num_prompts'] 
                        for (t, ht, s) in counts.keys() if t == tag)
    print(f"{tag}: {total_acts:,} total activations from {total_prompts} total prompts")

# Calculate totals per split
for split in ['train', 'val']:
    total_acts = sum(counts[(t, ht, s)]['num_activations'] 
                     for (t, ht, s) in counts.keys() if s == split)
    total_prompts = sum(counts[(t, ht, s)]['num_prompts'] 
                        for (t, ht, s) in counts.keys() if s == split)
    print(f"{split.capitalize()}: {total_acts:,} total activations from {total_prompts} total prompts")

print("\n" + "="*80)
