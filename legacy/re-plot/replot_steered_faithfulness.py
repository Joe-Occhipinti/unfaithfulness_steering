"""
replot_steered_faithfulness.py

Regenerate plots from existing steered faithfulness evaluation results.
Skips classification - just loads summary JSON and creates new plots.
"""

import json
import os

# Import plotting functions
from src.steered_plots import (
    plot_steering_heatmaps,
    plot_transformation_rates_by_layer
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input: Summary JSON from previous eval_steered_global_faithfulness.py run
INPUT_SUMMARY = "data/sprint4_2025-10-21/summaries/steered_faithfulness/summary_faithfulness_steered_global_val_neg_bas_psyXprof_2025-10-19.json"

# Output directory for plots
OUTPUT_PLOT_DIR = "plots/sprint4_2025-10-21/steering_neg_bas_psyXprof"

print(f"=== REPLOT STEERED FAITHFULNESS RESULTS ===")
print(f"Input Summary: {INPUT_SUMMARY}")
print(f"Output Plots: {OUTPUT_PLOT_DIR}")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n=== Loading Summary ===")
with open(INPUT_SUMMARY, 'r', encoding='utf-8') as f:
    summary = json.load(f)

all_configs = summary['all_configurations']
subject = summary['subject']
hint_template = summary['hint_template']

print(f"Loaded {len(all_configs)} configurations")
print(f"  Subject: {subject}")
print(f"  Hint Template: {hint_template}")

# =============================================================================
# GENERATE PLOTS
# =============================================================================

print("\n=== Generating Heatmaps ===")

# CORRECT - TRANSITIONS
print("  Creating CORRECT - TRANSITIONS heatmap...")
heatmap_path_correct_trans = os.path.join(OUTPUT_PLOT_DIR, 'heatmaps_correct_transitions.png')
plot_steering_heatmaps(
    all_configs=all_configs,
    subject=subject,
    hint_template=hint_template,
    correctness_group='correct',
    heatmap_type='transitions',
    save_path=heatmap_path_correct_trans
)

# CORRECT - NO CHANGE
print("  Creating CORRECT - NO CHANGE heatmap...")
heatmap_path_correct_nochange = os.path.join(OUTPUT_PLOT_DIR, 'heatmaps_correct_no_change.png')
plot_steering_heatmaps(
    all_configs=all_configs,
    subject=subject,
    hint_template=hint_template,
    correctness_group='correct',
    heatmap_type='no_change',
    save_path=heatmap_path_correct_nochange
)

# WRONG - TRANSITIONS
print("  Creating WRONG - TRANSITIONS heatmap...")
heatmap_path_wrong_trans = os.path.join(OUTPUT_PLOT_DIR, 'heatmaps_wrong_transitions.png')
plot_steering_heatmaps(
    all_configs=all_configs,
    subject=subject,
    hint_template=hint_template,
    correctness_group='wrong',
    heatmap_type='transitions',
    save_path=heatmap_path_wrong_trans
)

# WRONG - NO CHANGE
print("  Creating WRONG - NO CHANGE heatmap...")
heatmap_path_wrong_nochange = os.path.join(OUTPUT_PLOT_DIR, 'heatmaps_wrong_no_change.png')
plot_steering_heatmaps(
    all_configs=all_configs,
    subject=subject,
    hint_template=hint_template,
    correctness_group='wrong',
    heatmap_type='no_change',
    save_path=heatmap_path_wrong_nochange
)

print("\n=== Generating Layer-wise Line Plots ===")

# CORRECT group line plots
print("  Creating CORRECT group line plots...")
plot_transformation_rates_by_layer(
    all_configs=all_configs,
    correctness_group='correct',
    save_dir=OUTPUT_PLOT_DIR,
    subject=subject,
    hint_template=hint_template
)

# WRONG group line plots
print("  Creating WRONG group line plots...")
plot_transformation_rates_by_layer(
    all_configs=all_configs,
    correctness_group='wrong',
    save_dir=OUTPUT_PLOT_DIR,
    subject=subject,
    hint_template=hint_template
)

print("\n=== REPLOTTING COMPLETE ===")
print(f"All plots saved to: {OUTPUT_PLOT_DIR}")
print(f"\nGenerated:")
print(f"  - 4 heatmap grids (2x2 each)")
print(f"  - Multiple line plot figures (2x2 per coefficient, per correctness group)")
