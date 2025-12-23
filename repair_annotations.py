"""
Repair script to fix corrupted classifications in the annotated dataset.

This script:
1. Loads the summary JSON which contains correct classifications
2. Loads the original steered dataset 
3. Rebuilds the lookup with the CORRECT key: (qid, hint, layer, coeff, initial_state)
4. Creates a corrected annotated dataset
"""

import json
from collections import defaultdict
import sys
sys.path.insert(0, 'src')
from steered_global_faithfulness import get_initial_joint_state
from data import save_jsonl

print("=== REPAIRING ANNOTATED DATASET ===\n")

# Load summary with correct classifications
print("Loading summary file...")
with open('data/sprint_6_2025-12-15/summary_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5.jsonl', 'r', encoding='utf-8') as f:
    summary = json.load(f)
print("✓ Loaded summary\n")

# Load original records
print("Loading original steered dataset...")
with open('data/sprint_6_2025-12-15/annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5.jsonl', 'r', encoding='utf-8') as f:
    original_records = [json.loads(line) for line in f]
print(f"✓ Loaded {len(original_records)} records\n")

# Build CORRECT classification lookup
print("Building correct classification lookup...")
classification_lookup = {}

configs_by_hint = summary['configurations_by_hint']

for hint_template, configs in configs_by_hint.items():
    for config in configs:
        layer = config['layer']
        coeff_mag = config['coefficient_magnitude']
        
        # Process all 8 groups
        for group_name in ['positive_on_CF', 'positive_on_CU', 'positive_on_WF', 'positive_on_WU',
                          'negative_on_CF', 'negative_on_CU', 'negative_on_WF', 'negative_on_WU']:
            
            if group_name not in config:
                continue
                
            group_data = config[group_name]
            classifications = group_data.get('classifications', {})
            
            # Extract initial_state and direction from group_name
            parts = group_name.split('_on_')
            direction = parts[0]  # 'positive' or 'negative'
            initial_state = parts[1]  # 'CF', 'CU', 'WF', 'WU'
            
            # Determine coefficient sign
            if direction == 'positive':
                coeff = coeff_mag
            else:
                coeff = -coeff_mag
            
            # Store classifications with CORRECT key
            for qid_str, classification in classifications.items():
                qid = int(qid_str)  # Convert string key to int
                key = (qid, hint_template, layer, coeff, initial_state)
                classification_lookup[key] = classification

print(f"✓ Built lookup with {len(classification_lookup)} correct classifications\n")

# Annotate records with correct classifications
print("Annotating records with correct classifications...")
annotated = []
errors = 0
successes = 0

for record in original_records:
    qid = record.get('question_id', record.get('prompt_index'))
    hint_template = record.get('hint_template', 'unknown')
    layer = record['steering_layer']
    coeff = record['steering_coefficient']
    
    # Determine initial state for this record
    initial_state = get_initial_joint_state(record)
    
    # Build lookup key
    key = (qid, hint_template, layer, coeff, initial_state)
    classification = classification_lookup.get(key, 'error')
    
    if classification == 'error':
        errors += 1
    else:
        successes += 1
    
    # Create annotated record
    annotated_record = record.copy()
    annotated_record['steered_global_faithfulness_classification'] = classification
    
    # Add tagged prompt for potential future use
    steered_prompt = record.get('steered_prompt', '')
    if classification == 'faithful':
        annotated_record['annotated_steered_prompt'] = f"[F_final]{steered_prompt}[/F_final]"
    elif classification == 'unfaithful':
        annotated_record['annotated_steered_prompt'] = f"[U_final]{steered_prompt}[/U_final]"
    else:
        annotated_record['annotated_steered_prompt'] = steered_prompt
    
    annotated.append(annotated_record)

print(f"✓ Annotated {len(annotated)} records")
print(f"  Successes: {successes}")
print(f"  Errors: {errors}\n")

# Save corrected dataset
output_path = 'data/sprint_6_2025-12-15/annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5_FIXED.jsonl'
print(f"Saving corrected dataset to: {output_path}")
save_jsonl(annotated, output_path)
print(f"✓ Saved corrected dataset\n")

# Print classification distribution
print("=== Classification Distribution (CORRECTED) ===")
from collections import Counter
classifications = [r['steered_global_faithfulness_classification'] for r in annotated]
distribution = Counter(classifications)
for cls, count in sorted(distribution.items()):
    pct = 100 * count / len(annotated)
    print(f"{cls:30s}: {count:4d} ({pct:5.1f}%)")

print("\n✅ REPAIR COMPLETE!")
print(f"   Original file (corrupted): annotated_steered_val_off_policy_2nd_2025-12-20.jsonl")
print(f"   Corrected file:            annotated_steered_val_off_policy_2nd_2025-12-20_FIXED.jsonl")
