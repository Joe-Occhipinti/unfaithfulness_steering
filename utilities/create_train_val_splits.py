"""
create_train_val_splits.py

Creates train/val splits from annotated datasets using stratified sampling.
Ensures faithful and unfaithful prompts are proportionally represented in both splits.

Usage:
    python create_train_val_splits.py
"""

import json
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

INPUT_FILE = "data/definitive_pipeline_data/Olmo-3-7B-Think/faithfulness_annotated_Olmo-3-7B-Think_2025-12-29.jsonl"
OUTPUT_FILE = "data/definitive_pipeline_data/Olmo-3-7B-Think/faithfulness_annotated_Olmo-3-7B-Think_2025-12-29.jsonl"
SUMMARY_FILE = "data/definitive_pipeline_data/Olmo-3-7B-Think/faithfulness_summary_Olmo-3-7B-Think_2025-12-29.jsonl"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.3
RANDOM_SEED = 42
CLASSIFICATION_FIELD = "faithfulness_classification"  # Field containing faithfulness labels

# Classification values (standard labels only)
FAITHFUL_LABEL = "faithful"
UNFAITHFUL_LABEL = "unfaithful"


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load records from a JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of record dictionaries
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line {line_num}")
    return records


def save_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save records to a JSONL file.

    Args:
        records: List of record dictionaries
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def group_by_faithfulness_and_hint(
    records: List[Dict[str, Any]],
    faithful_label: str = "faithful",
    unfaithful_label: str = "unfaithful",
    classification_field: str = "global_faithfulness_classification",
    hint_template_field: str = "hint_template"
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Group records by (faithfulness_classification, hint_template) for stratified sampling.

    This ensures both faithful/unfaithful AND different hint templates are
    proportionally represented in train and val splits.

    Standard labels are "faithful" and "unfaithful".

    Args:
        records: List of record dictionaries
        faithful_label: Exact value for faithful classification (default: "faithful")
        unfaithful_label: Exact value for unfaithful classification (default: "unfaithful")
        classification_field: Field name containing faithfulness classification
        hint_template_field: Field name containing hint template

    Returns:
        Dictionary mapping (faithfulness_category, hint_template) -> list of records
    """
    groups = {}

    for record in records:
        classification = record.get(classification_field, "")
        hint_template = record.get(hint_template_field, "unknown")

        # Categorize faithfulness using exact matching
        if classification == faithful_label:
            faith_category = "faithful"
        elif classification == unfaithful_label:
            faith_category = "unfaithful"
        else:
            # Skip records that don't match standard labels
            print(f"  Warning: Skipping record with unexpected classification: '{classification}'")
            continue

        # Create group key as (faithfulness, hint_template)
        group_key = (faith_category, hint_template)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(record)

    return groups


def stratified_split(
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform stratified sampling to create train/val splits.

    Each group (faithfulness × hint_template) is split proportionally
    to ensure balanced representation in both splits.

    Args:
        groups: Dictionary mapping (faithfulness, hint_template) -> records
        train_ratio: Proportion for training set (e.g., 0.6)
        val_ratio: Proportion for validation set (e.g., 0.4)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_records, val_records)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio - 1.0) > 0.001:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must equal 1.0")

    random.seed(random_seed)

    train_records = []
    val_records = []

    # Print group statistics
    print(f"\nFound {len(groups)} groups:")
    for (faith, template), records in sorted(groups.items()):
        print(f"  ({faith}, {template}): {len(records)} prompts")

    # Process each group independently
    for group_key, group_records in groups.items():
        if not group_records:
            continue

        # Shuffle this group
        shuffled = group_records.copy()
        random.shuffle(shuffled)

        # Calculate split point
        num_records = len(shuffled)
        train_count = int(num_records * train_ratio)

        # Warn if group is too small for proper splitting
        if num_records < 3:
            faith, template = group_key
            print(f"  Warning: Group ({faith}, {template}) has only {num_records} samples - may lead to imbalanced splits")

        # Split
        group_train = shuffled[:train_count]
        group_val = shuffled[train_count:]

        # Add to overall splits
        train_records.extend(group_train)
        val_records.extend(group_val)

        faith, template = group_key
        print(f"  ({faith}, {template}): {num_records} total -> {len(group_train)} train, {len(group_val)} val")

    return train_records, val_records


def assign_split_labels(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    split_field: str = "split"
) -> List[Dict[str, Any]]:
    """
    Assign split labels to records and combine into ordered dataset.

    Args:
        train_records: List of training records
        val_records: List of validation records
        split_field: Field name for split label

    Returns:
        Combined list with all train records first, then val records
    """
    # Assign labels
    for record in train_records:
        record[split_field] = "train"

    for record in val_records:
        record[split_field] = "val"

    # Combine: train first, then val
    combined = train_records + val_records

    return combined


def print_split_statistics(
    records: List[Dict[str, Any]],
    split_field: str = "split",
    classification_field: str = "global_faithfulness_classification"
) -> None:
    """
    Print statistics about the splits.

    Args:
        records: List of records with split labels
        split_field: Field name for split label
        classification_field: Field name for classification
    """
    # Count by split
    train_count = sum(1 for r in records if r.get(split_field) == "train")
    val_count = sum(1 for r in records if r.get(split_field) == "val")

    print(f"\n=== SPLIT STATISTICS ===")
    print(f"Total records: {len(records)}")
    print(f"Train: {train_count} ({train_count/len(records)*100:.1f}%)")
    print(f"Val: {val_count} ({val_count/len(records)*100:.1f}%)")

    # Count by split and classification
    for split_name in ["train", "val"]:
        split_records = [r for r in records if r.get(split_field) == split_name]
        if not split_records:
            continue

        # Count using exact matching for standard labels
        faithful = sum(1 for r in split_records if r.get(classification_field) == "faithful")
        unfaithful = sum(1 for r in split_records if r.get(classification_field) == "unfaithful")

        print(f"\n{split_name.capitalize()} split breakdown:")
        print(f"  Faithful: {faithful} ({faithful/len(split_records)*100:.1f}%)")
        print(f"  Unfaithful: {unfaithful} ({unfaithful/len(split_records)*100:.1f}%)")


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def create_stratified_splits(
    input_file: str,
    output_file: str,
    train_ratio: float,
    val_ratio: float,
    random_seed: int = 42,
    faithful_label: str = "faithful",
    unfaithful_label: str = "unfaithful",
    classification_field: str = "global_faithfulness_classification",
    split_field: str = "split",
    summary_file: str = None
) -> Dict[str, Any]:
    """
    Main function to create stratified train/val splits from annotated dataset.

    Standard labels are "faithful" and "unfaithful". Records with other values
    will be skipped with a warning.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        random_seed: Random seed for reproducibility
        faithful_label: Exact value for faithful classification (default: "faithful")
        unfaithful_label: Exact value for unfaithful classification (default: "unfaithful")
        classification_field: Field containing faithfulness classification
        split_field: Field name for split label
        summary_file: Path to save detailed summary statistics JSON (optional)

    Returns:
        Summary dictionary with statistics
    """
    print(f"=== CREATE STRATIFIED TRAIN/VAL SPLITS ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    if summary_file:
        print(f"Summary: {summary_file}")
    print(f"Ratios: {train_ratio:.1%} train, {val_ratio:.1%} val")
    print(f"Random seed: {random_seed}")

    # Step 1: Load records
    print(f"\n--- Step 1: Loading records ---")
    records = load_jsonl(input_file)
    print(f"Loaded {len(records)} records")

    # Step 2: Group by faithfulness AND hint_template (stratified sampling)
    print(f"\n--- Step 2: Grouping by (faithfulness × hint_template) ---")
    groups = group_by_faithfulness_and_hint(
        records,
        faithful_label=faithful_label,
        unfaithful_label=unfaithful_label,
        classification_field=classification_field,
        hint_template_field="hint_template"
    )

    # Count totals by faithfulness category
    total_faithful = sum(len(recs) for (faith, _), recs in groups.items() if faith == "faithful")
    total_unfaithful = sum(len(recs) for (faith, _), recs in groups.items() if faith == "unfaithful")

    print(f"\nTotal by faithfulness category:")
    print(f"  Faithful: {total_faithful} records")
    print(f"  Unfaithful: {total_unfaithful} records")

    # Step 3: Stratified split
    print(f"\n--- Step 3: Performing stratified split ---")
    train_records, val_records = stratified_split(
        groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    # Step 4: Assign split labels and combine
    print(f"\n--- Step 4: Assigning split labels ---")
    final_records = assign_split_labels(
        train_records,
        val_records,
        split_field=split_field
    )
    print(f"Combined dataset: {len(final_records)} records (train first, then val)")

    # Step 5: Print statistics
    print_split_statistics(
        final_records,
        split_field=split_field,
        classification_field=classification_field
    )

    # Step 6: Save output
    print(f"\n--- Step 5: Saving output ---")
    save_jsonl(final_records, output_file)
    print(f"Saved to {output_file}")

    # Step 7: Create detailed summary statistics
    print(f"\n--- Step 6: Creating summary statistics ---")

    # Collect detailed split-wise statistics
    train_stats_by_group = {}
    val_stats_by_group = {}

    for (faith, template), recs in groups.items():
        train_count_group = sum(1 for r in recs if r.get(split_field) == "train")
        val_count_group = sum(1 for r in recs if r.get(split_field) == "val")
        group_key = f"{faith}_{template}"
        train_stats_by_group[group_key] = train_count_group
        val_stats_by_group[group_key] = val_count_group

    summary = {
        "metadata": {
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": input_file,
            "output_file": output_file,
            "summary_file": summary_file if summary_file else "not_saved",
        },
        "configuration": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "random_seed": random_seed,
            "classification_field": classification_field,
            "split_field": split_field
        },
        "overall_statistics": {
            "total_records": len(final_records),
            "train_count": len(train_records),
            "val_count": len(val_records),
            "train_percentage": len(train_records) / len(final_records) * 100,
            "val_percentage": len(val_records) / len(final_records) * 100
        },
        "faithfulness_statistics": {
            "total": {
                "faithful": total_faithful,
                "unfaithful": total_unfaithful
            },
            "train": {
                "faithful": sum(1 for r in train_records if r.get(classification_field) == faithful_label),
                "unfaithful": sum(1 for r in train_records if r.get(classification_field) == unfaithful_label)
            },
            "val": {
                "faithful": sum(1 for r in val_records if r.get(classification_field) == faithful_label),
                "unfaithful": sum(1 for r in val_records if r.get(classification_field) == unfaithful_label)
            }
        },
        "group_statistics": {
            "num_groups": len(groups),
            "groups_total": {f"{faith}_{template}": len(recs) for (faith, template), recs in groups.items()},
            "groups_train": train_stats_by_group,
            "groups_val": val_stats_by_group
        }
    }

    # Save summary to file if path provided
    if summary_file:
        import os
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary statistics saved to {summary_file}")

    print(f"\n=== SPLIT CREATION COMPLETE ===")
    return summary


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    summary = create_stratified_splits(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_seed=RANDOM_SEED,
        faithful_label=FAITHFUL_LABEL,
        unfaithful_label=UNFAITHFUL_LABEL,
        classification_field=CLASSIFICATION_FIELD,
        summary_file=SUMMARY_FILE
    )
