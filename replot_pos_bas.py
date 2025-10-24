"""
Recompute transitions and plot heatmaps for pos_bas_psyXprof steered data
"""

import json
import os
from src.data import load_jsonl
from src.steered_global_faithfulness import group_records_by_config, compute_config_metrics
from src.steered_plots import plot_steering_heatmaps
from src.config import TODAY

# Input
INPUT_JSONL = "data/sprint4_2025-10-21/annotated/steered/annotated_steered_global_val_pos_bas_psyXprof_2025-10-19.jsonl"

# Output
OUTPUT_PLOT_DIR = "data/sprint4_2025-10-21/plots/steering_pos_bas_psyXprof"

print("=== RECOMPUTE HEATMAPS FOR POS_BAS ===")
print(f"Input: {INPUT_JSONL}")
print(f"Output: {OUTPUT_PLOT_DIR}")

# Load annotated records
print("\n=== Loading annotated records ===")
all_records = load_jsonl(INPUT_JSONL)
print(f"Loaded {len(all_records)} records")

# Extract metadata
subject = "psychology_professor"
hint_template = "professor"

# Group records by configuration
print("\n=== Grouping by configuration ===")
grouped = group_records_by_config(all_records)
print(f"Found {len(grouped)} configurations")

# Recompute transitions for each configuration
print("\n=== Recomputing transitions ===")
all_configs = []

for (ht, layer, coeff_mag), config_groups in grouped.items():
    print(f"  Processing layer {layer}, coeff {coeff_mag}...")

    # Create config result with transitions computed from classifications
    config_result = {
        'hint_template': ht,
        'layer': layer,
        'coefficient_magnitude': coeff_mag
    }

    # Process each of the 8 groups
    for group_name in ['positive_on_CF', 'positive_on_CU', 'positive_on_WF', 'positive_on_WU',
                       'negative_on_CF', 'negative_on_CU', 'negative_on_WF', 'negative_on_WU']:

        if group_name not in config_groups or len(config_groups[group_name]) == 0:
            config_result[group_name] = {
                'n': 0,
                'transitions': {}
            }
            continue

        group_records = config_groups[group_name]
        n = len(group_records)

        # Extract classifications from records
        classifications = {}
        for record in group_records:
            qid = record.get('question_id', record.get('prompt_index'))
            classification = record.get('steered_global_faithfulness_classification', 'error')
            classifications[qid] = classification

        # Determine initial state from group name
        initial_state = group_name.split('_')[-1]  # CF, CU, WF, or WU

        # Compute transitions
        from src.steered_global_faithfulness import compute_transitions
        transitions = compute_transitions(group_records, classifications, initial_state)

        config_result[group_name] = {
            'n': n,
            'transitions': transitions,
            'classifications': classifications
        }

    all_configs.append(config_result)

print(f"Computed transitions for {len(all_configs)} configurations")

# Generate heatmaps
print("\n=== Generating Heatmaps ===")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# CORRECT - TRANSITIONS
print("  Creating CORRECT - TRANSITIONS...")
plot_steering_heatmaps(
    all_configs=all_configs,
    subject=subject,
    hint_template=hint_template,
    correctness_group='correct',
    heatmap_type='transitions',
    save_path=os.path.join(OUTPUT_PLOT_DIR, 'heatmaps_correct_transitions.png')
)

# CORRECT - NO CHANGE
print("  Creating CORRECT - NO CHANGE...")
plot_steering_heatmaps(
    all_configs=all_configs,
    subject=subject,
    hint_template=hint_template,
    correctness_group='correct',
    heatmap_type='no_change',
    save_path=os.path.join(OUTPUT_PLOT_DIR, 'heatmaps_correct_no_change.png')
)

# WRONG - TRANSITIONS
print("  Creating WRONG - TRANSITIONS...")
plot_steering_heatmaps(
    all_configs=all_configs,
    subject=subject,
    hint_template=hint_template,
    correctness_group='wrong',
    heatmap_type='transitions',
    save_path=os.path.join(OUTPUT_PLOT_DIR, 'heatmaps_wrong_transitions.png')
)

# WRONG - NO CHANGE
print("  Creating WRONG - NO CHANGE...")
plot_steering_heatmaps(
    all_configs=all_configs,
    subject=subject,
    hint_template=hint_template,
    correctness_group='wrong',
    heatmap_type='no_change',
    save_path=os.path.join(OUTPUT_PLOT_DIR, 'heatmaps_wrong_no_change.png')
)

print("\n=== COMPLETE ===")
print(f"4 heatmaps saved to: {OUTPUT_PLOT_DIR}")
