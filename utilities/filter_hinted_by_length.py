"""
filter_hinted_by_length.py

Filter hinted generated texts by word length threshold.

This script filters a JSONL dataset to include only records where the
'hinted_generated_text' field has a word count below a specified threshold.
All other fields in each record are preserved in the filtered output.

Usage:
    1. Configure the parameters below
    2. Run: python filter_hinted_by_length.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_jsonl, save_jsonl
from src.utilities.filters import (
    calculate_median_word_length,
    filter_by_text_length,
    get_length_statistics
)
from src.plots import (
    plot_text_length_histogram,
    plot_text_length_sorted_bar
)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Input/Output files
INPUT_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\annotated\hinted\psychology_professor_2025-08-15\annotated_local_biased_high_school_psychology_2025-08-15.jsonl"
OUTPUT_FILE = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\annotated\hinted_cut\psycholody_professor_2025-08-15\cut_annotated_local_biased_psychology_professor_2025-08-12.jsonl"

# Field to filter by
TEXT_FIELD = "hinted_generated_text"

# Threshold (set to None to use median)
THRESHOLD = None  # None = use median, or specify a number like 150

# Plot settings
GENERATE_PLOTS = True
PLOT_DIR = "plots/length_filtering/psychology_professor"

# =============================================================================


def main():
    """Main execution function."""

    # Validate input file exists
    if not Path(INPUT_FILE).exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    # Load data
    print(f"Loading data from: {INPUT_FILE}")
    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data)} records")

    # Check if the specified field exists
    if not data:
        print("Error: Input file is empty")
        sys.exit(1)

    first_record = data[0]
    if TEXT_FIELD not in first_record:
        print(f"Error: Field '{TEXT_FIELD}' not found in records")
        print(f"Available fields: {list(first_record.keys())}")
        sys.exit(1)

    # Calculate statistics on original data
    print(f"\n{'='*60}")
    print(f"ORIGINAL DATA STATISTICS")
    print(f"{'='*60}")
    original_stats = get_length_statistics(data, TEXT_FIELD)
    print(f"Records: {len(data)}")
    print(f"Mean word length:   {original_stats['mean']:.2f}")
    print(f"Median word length: {original_stats['median']:.2f}")
    print(f"Min word length:    {original_stats['min']}")
    print(f"Max word length:    {original_stats['max']}")

    # Determine threshold
    if THRESHOLD is None:
        threshold = int(calculate_median_word_length(data, TEXT_FIELD))
        print(f"\nUsing median as threshold: {threshold} words")
    else:
        threshold = THRESHOLD
        print(f"\nUsing specified threshold: {threshold} words")

    # Filter data
    print(f"\nFiltering records with {TEXT_FIELD} <= {threshold} words...")
    filtered_data = filter_by_text_length(data, TEXT_FIELD, threshold)

    # Calculate statistics on filtered data
    print(f"\n{'='*60}")
    print(f"FILTERED DATA STATISTICS")
    print(f"{'='*60}")
    filtered_stats = get_length_statistics(filtered_data, TEXT_FIELD)
    print(f"Records: {len(filtered_data)} ({len(filtered_data)/len(data)*100:.1f}% of original)")
    print(f"Mean word length:   {filtered_stats['mean']:.2f}")
    print(f"Median word length: {filtered_stats['median']:.2f}")
    print(f"Min word length:    {filtered_stats['min']}")
    print(f"Max word length:    {filtered_stats['max']}")

    # Save filtered data
    print(f"\nSaving filtered data to: {OUTPUT_FILE}")
    save_jsonl(filtered_data, OUTPUT_FILE)
    print(f"✓ Successfully saved {len(filtered_data)} records")

    # Summary
    removed = len(data) - len(filtered_data)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Original records:  {len(data)}")
    print(f"Filtered records:  {len(filtered_data)}")
    print(f"Removed records:   {removed} ({removed/len(data)*100:.1f}%)")
    print(f"Threshold:         {threshold} words")

    # Generate plots if requested
    if GENERATE_PLOTS:
        print(f"\n{'='*60}")
        print(f"GENERATING PLOTS")
        print(f"{'='*60}")

        import os

        # Create plot directory
        plot_dir = Path(PLOT_DIR)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Generate base filename from input file
        input_name = Path(INPUT_FILE).stem

        # 1. Histogram with threshold and percentages
        print("\nGenerating histogram with threshold cut...")
        plot_text_length_histogram(
            data,
            text_field=TEXT_FIELD,
            threshold=threshold,
            save_path=str(plot_dir / f"{input_name}_histogram.png"),
            show_plot=False,
            title_suffix=f"Total: {len(data)} texts"
        )

        # 2. Sorted bar chart (limited to 100 items for readability)
        print("Generating sorted bar chart...")
        plot_text_length_sorted_bar(
            data,
            text_field=TEXT_FIELD,
            threshold=threshold,
            max_items=100,
            save_path=str(plot_dir / f"{input_name}_sorted.png"),
            show_plot=False,
            title_suffix=f"Showing up to 100 texts"
        )

        print(f"\n✓ All plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
