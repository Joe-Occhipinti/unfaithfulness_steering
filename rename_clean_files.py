"""
rename_clean_files.py

Renames all files with '_clean' suffix to remove the suffix,
making the cleaned versions the standard files.
"""

import os
import shutil
from pathlib import Path

print("=" * 70)
print("RENAMING CLEANED FILES")
print("=" * 70)

# Define file mappings: old_name -> new_name
file_mappings = {
    # Baseline files
    "data/behavioural/baseline_high_school_macroeconomics_microeconomics_2025-10-03_clean.jsonl":
        "data/behavioural/baseline_high_school_macroeconomics_microeconomics_2025-10-03.jsonl",

    # Hinted files
    "data/behavioural/hinted_high_school_macroeconomics_microeconomics_2025-10-03_clean.jsonl":
        "data/behavioural/hinted_high_school_macroeconomics_microeconomics_2025-10-03.jsonl",

    # Annotated files
    "data/annotated/annotated_biased_high_school_macroeconomics_microeconomics_2025-10-03_clean.jsonl":
        "data/annotated/annotated_biased_high_school_macroeconomics_microeconomics_2025-10-03.jsonl",

    # Summary files
    "data/summaries/baseline_summary_high_school_macroeconomics_microeconomics_2025-10-03_clean.json":
        "data/summaries/baseline_summary_high_school_macroeconomics_microeconomics_2025-10-03.json",

    "data/summaries/hinted_summary_high_school_macroeconomics_microeconomics_2025-10-03_clean.json":
        "data/summaries/hinted_summary_high_school_macroeconomics_microeconomics_2025-10-03.json",

    "data/summaries/faithfulness_hinted_high_school_macroeconomics_microeconomics_2025-10-03_clean.json":
        "data/summaries/faithfulness_hinted_high_school_macroeconomics_microeconomics_2025-10-03.json",

    # Plot files
    "plots/hint_breakdown_2025-10-04_clean.png":
        "plots/hint_breakdown_2025-10-04.png",

    "plots/faithfulness_distribution_biased_high_school_macroeconomics_microeconomics_2025-10-04_clean.png":
        "plots/faithfulness_distribution_biased_high_school_macroeconomics_microeconomics_2025-10-04.png",

    "plots/faithfulness_by_hint_biased_high_school_macroeconomics_microeconomics_2025-10-04_clean.png":
        "plots/faithfulness_by_hint_biased_high_school_macroeconomics_microeconomics_2025-10-04.png",
}

print("\nFiles to rename:")
renamed_count = 0
skipped_count = 0
error_count = 0

for old_path, new_path in file_mappings.items():
    print(f"\n  {old_path}")
    print(f"  -> {new_path}")

    if not os.path.exists(old_path):
        print(f"     [SKIP] Source file doesn't exist")
        skipped_count += 1
        continue

    if os.path.exists(new_path):
        print(f"     [WARNING] Target file already exists, will be overwritten")
        try:
            os.remove(new_path)
            print(f"     [DELETED] Old target file")
        except Exception as e:
            print(f"     [ERROR] Could not delete target: {e}")
            error_count += 1
            continue

    try:
        shutil.move(old_path, new_path)
        print(f"     [OK] Renamed successfully")
        renamed_count += 1
    except Exception as e:
        print(f"     [ERROR] Failed to rename: {e}")
        error_count += 1

print("\n" + "=" * 70)
print("RENAMING COMPLETE")
print("=" * 70)
print(f"\nRenamed: {renamed_count} files")
print(f"Skipped: {skipped_count} files (didn't exist)")
print(f"Errors: {error_count} files")

if renamed_count > 0:
    print("\nThe cleaned files are now the standard versions.")
    print("Old contaminated files have been replaced.")
