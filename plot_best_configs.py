"""
Plot best config results: +pos vs -neg correct mentioning hint rates.

Creates a 2x2 panel of grouped bar plots (one per dataset).
For each hint template, shows the best config's:
- Positive steering → correct mentioning hint %
- Negative steering → correct mentioning hint %
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
RESULTS_PATH = "data/definitive_pipeline_data/best_configs_per_dataset.json"
OUTPUT_PATH = "data/definitive_pipeline_data/best_configs_barplot.png"

# Short dataset names for display
DATASET_NAMES = {
    "annotated_steered_val_off_policy_2nd_2025-12-20": "Off-Policy v2",
    "annotated_steered_off_policy_val_scie_hist_psy_X_grader_prof_meta_2025-12-5": "Off-Policy v1",
    "annotated_steered_val_gradient_2hidden8_2025-12-06": "Gradient MLP",
    "annotated_steered_val_hintweighting_scie_hist_psy_X_grader_prof_meta_2025-10-25": "Hint Weighting"
}

# Colors
COLOR_POS = "#4CAF50"  # Green for positive (good)
COLOR_NEG = "#F44336"  # Red for negative (should be low)


def load_results():
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    datasets = list(results.keys())
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = results[dataset]
        
        hints = sorted(data.keys())
        x = np.arange(len(hints))
        width = 0.35
        
        pos_vals = [data[h]['unfaithful_pos_correct_mentioning'] for h in hints]
        neg_vals = [data[h]['unfaithful_neg_correct_mentioning'] for h in hints]
        
        # Bars
        bars1 = ax.bar(x - width/2, pos_vals, width, label='+Pos Steering', color=COLOR_POS, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, neg_vals, width, label='-Neg Steering', color=COLOR_NEG, edgecolor='black', linewidth=0.5)
        
        # Labels on bars
        for bar, val in zip(bars1, pos_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, val in zip(bars2, neg_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add config info below x-axis
        config_labels = []
        for h in hints:
            cfg = data[h]
            config_labels.append(f"{h}\n(L{cfg['layer']}, ±{cfg['strength']})")
        
        ax.set_ylabel('Correct Mentioning Hint (%)', fontsize=11)
        ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, fontsize=9)
        ax.set_ylim(0, max(max(pos_vals), max(neg_vals)) * 1.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add score annotation
        for i, h in enumerate(hints):
            score = data[h]['score']
            ax.annotate(f'Score: {score:.2f}', 
                       xy=(i, max(pos_vals[i], neg_vals[i]) + 3),
                       ha='center', fontsize=8, style='italic', color='gray')
    
    plt.suptitle('Best Config: Correct Mentioning Hint (%) by Steering Direction\n(Unfaithful Answers Only)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot saved to: {OUTPUT_PATH}")
    plt.show()


def main():
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} datasets")
    
    print("Creating plot...")
    plot_results(results)


if __name__ == "__main__":
    main()
