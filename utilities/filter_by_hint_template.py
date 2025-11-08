"""
filter_by_hint_template.py

Filter a hinted dataset to keep only "biased" records with a specific hint template.
"Biased" means the model was influenced by the hint (bias_label == "biased").
Useful for running experiments on individual hint templates.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

# Input dataset path
INPUT_DATASET = "data/behavioural/hinted_stem_2025-10-15.jsonl"

# Hint template to filter (select one)
# Options: "grader_hacking", "reward_hacking", "argument", "self-consistency",
#          "professor", "metadata", "black_square", "user", "unauthorized_access"
HINT_TEMPLATE = "argument"

# Output dataset path (auto-generated if None)
OUTPUT_DATASET = None  # If None, will be auto-generated as: hinted_stem_{hint_template}_biased_2025-10-15.jsonl

# ============================================================================
# END CONFIGURATION
# ============================================================================


def filter_by_hint_template(
    input_path: str,
    hint_template: str,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Filter dataset to keep only "biased" records with specified hint template.
    "Biased" means bias_label == "biased" (model was influenced by wrong hint).

    Args:
        input_path: Path to input JSONL dataset
        hint_template: Hint template to keep
        output_path: Path to output JSONL dataset (auto-generated if None)

    Returns:
        Dictionary with filtering statistics
    """
    input_path = Path(input_path)

    # Auto-generate output path if not provided
    if output_path is None:
        # Extract base name and date from input path
        # E.g., "hinted_stem_2025-10-15.jsonl" -> "hinted_stem_grader_hacking_biased_2025-10-15.jsonl"
        base_name = input_path.stem  # "hinted_stem_2025-10-15"
        parts = base_name.split("_")

        if len(parts) >= 2:
            # Insert hint template and "biased" before date
            # E.g., ["hinted", "stem", "2025-10-15"] -> ["hinted", "stem", "grader_hacking", "biased", "2025-10-15"]
            date_part = parts[-1]  # "2025-10-15"
            prefix_parts = parts[:-1]  # ["hinted", "stem"]
            new_name = "_".join(prefix_parts + [hint_template.replace("-", "_"), "biased", date_part])
        else:
            new_name = f"{base_name}_{hint_template.replace('-', '_')}_biased"

        output_path = input_path.parent / f"{new_name}.jsonl"
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Filtering dataset by hint template (BIASED only)")
    print(f"{'='*80}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Hint template: {hint_template}")
    print()

    # Read and filter records
    filtered_records = []
    all_templates = Counter()
    bias_labels = Counter()
    template_bias_counts = {}
    total_records = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            total_records += 1

            record_template = record.get('hint_template', 'unknown')
            bias_label = record.get('bias_label', 'unknown')

            all_templates[record_template] += 1
            bias_labels[bias_label] += 1

            # Track bias labels per template
            if record_template not in template_bias_counts:
                template_bias_counts[record_template] = Counter()
            template_bias_counts[record_template][bias_label] += 1

            # Keep only records matching hint template AND biased
            if record_template == hint_template and bias_label == "biased":
                filtered_records.append(record)

    # Write filtered records
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in filtered_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Print statistics
    print(f"Total records in input: {total_records}")
    print()

    print("Bias label distribution (overall):")
    for label, count in bias_labels.most_common():
        percentage = count / total_records * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    print()

    print("Hint templates in input dataset:")
    for template, count in all_templates.most_common():
        percentage = count / total_records * 100
        marker = " ← SELECTED" if template == hint_template else ""

        # Show bias breakdown for this template
        biased_count = template_bias_counts[template].get('biased', 0)
        not_biased_count = template_bias_counts[template].get('not-biased', 0)
        biased_pct = (biased_count / count * 100) if count > 0 else 0

        print(f"  {template}: {count} total ({percentage:.1f}%) - {biased_count} biased ({biased_pct:.1f}%){marker}")
    print()

    print(f"✓ Filtered to '{hint_template}' + 'biased' only: {len(filtered_records)} records")
    print(f"  Percentage of total dataset: {len(filtered_records)/total_records*100:.1f}%")
    print(f"✓ Output saved to: {output_path}")
    print(f"{'='*80}\n")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "hint_template": hint_template,
        "total_input_records": total_records,
        "filtered_records": len(filtered_records),
        "percentage": len(filtered_records) / total_records * 100,
        "all_templates": dict(all_templates),
        "bias_labels": dict(bias_labels),
        "template_bias_counts": {k: dict(v) for k, v in template_bias_counts.items()}
    }


if __name__ == "__main__":
    stats = filter_by_hint_template(
        input_path=INPUT_DATASET,
        hint_template=HINT_TEMPLATE,
        output_path=OUTPUT_DATASET
    )
