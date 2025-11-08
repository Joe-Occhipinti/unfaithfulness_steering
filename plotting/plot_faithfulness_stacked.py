import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path to import plotting functions
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from plots import plot_global_faithfulness_stacked

def load_jsonl(file_path):
    """Load JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def main():
    # Define paths
    input_file = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\annotated\touse_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"

    # Create plots directory
    plots_dir = Path(r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\sprint4_2025-10-21\plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_file = plots_dir / f"faithfulness_stacked_by_template_{timestamp}.png"

    print(f"Loading annotated dataset from: {input_file}")

    # Load the annotated results
    annotated_results = load_jsonl(input_file)

    print(f"Loaded {len(annotated_results)} records")

    # Check what hint templates are present
    templates = set(r.get('hint_template', 'unknown') for r in annotated_results)
    print(f"Hint templates found: {sorted(templates)}")

    # Check bias labels present
    bias_labels = set(r.get('bias_label', 'unknown') for r in annotated_results)
    print(f"Bias labels found: {sorted(bias_labels)}")

    # Check faithfulness classifications present
    faithfulness_labels = set(r.get('faithfulness_classification', 'unknown') for r in annotated_results)
    print(f"Faithfulness classifications found: {sorted(faithfulness_labels)}")

    print(f"\nGenerating stacked bar plot...")

    # Create the plot
    plot_global_faithfulness_stacked(
        hinted_results=annotated_results,
        save_path=str(output_file),
        show_plot=False  # Set to False to just save without displaying
    )

    print(f"\nPlot saved to: {output_file}")

if __name__ == "__main__":
    main()
