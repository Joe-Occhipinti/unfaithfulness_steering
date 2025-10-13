"""
Analyze the distribution of answer letters (A, B, C, D) when steering from unfaithful to faithful.
"""

import json
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# TUNABLE PARAMETERS
# ============================================================================
STEERING_LAYER = 23
STEERING_COEFFICIENT = 0.6

# Input file path
INPUT_FILE = "data/annotated/steered/psychology_professor_2025-08-15/annotated_steered_local_val_biased_psychology_professor_2025-10-12.jsonl"

# Output directory for plots
OUTPUT_DIR = "plots/psychology_professor_2025-08-15/steered_letter_distribution/local" # change to /global as needed

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def load_and_filter_data(filepath, layer, coefficient):
    """Load data and filter for relevant examples."""
    filtered_examples = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)

            # Apply filters
            if (example['steering_layer'] == layer and
                example['steering_coefficient'] == coefficient and
                coefficient > 0 and
                example['original_faithfulness_classification'] == 'unfaithful' and
                example['steered_global_faithfulness_classification'] == 'faithful'):

                filtered_examples.append(example)

    return filtered_examples

def extract_letter_distribution(examples):
    """Extract and count answer letter distribution."""
    letters = [ex['steered_answer_letter'] for ex in examples]
    return Counter(letters)

def plot_distribution(counter, layer, coefficient, output_dir):
    """Create bar chart of letter distribution."""
    # Ensure all letters A-D are present
    all_letters = ['A', 'B', 'C', 'D']
    counts = [counter.get(letter, 0) for letter in all_letters]
    total = sum(counts)
    percentages = [100 * c / total if total > 0 else 0 for c in counts]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(all_letters, counts, color='steelblue', alpha=0.8)

    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Answer Letter', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of Answer Letters\n'
                 f'Layer {layer}, Coefficient {coefficient}\n'
                 f'Unfaithful → Faithful (N={total})',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f'letter_distribution_L{layer}_C{coefficient}.png'
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path / filename}")

    plt.show()

def main():
    print(f"Analyzing letter distribution for:")
    print(f"  Layer: {STEERING_LAYER}")
    print(f"  Coefficient: {STEERING_COEFFICIENT}")
    print(f"  Input: {INPUT_FILE}\n")

    # Load and filter data
    examples = load_and_filter_data(INPUT_FILE, STEERING_LAYER, STEERING_COEFFICIENT)
    print(f"Found {len(examples)} examples matching criteria")

    if len(examples) == 0:
        print("No examples found. Check your filters and input file.")
        return

    # Extract letter distribution
    letter_counts = extract_letter_distribution(examples)

    # Print summary
    print("\nLetter Distribution:")
    for letter in ['A', 'B', 'C', 'D']:
        count = letter_counts.get(letter, 0)
        pct = 100 * count / len(examples)
        print(f"  {letter}: {count} ({pct:.1f}%)")

    # Plot
    plot_distribution(letter_counts, STEERING_LAYER, STEERING_COEFFICIENT, OUTPUT_DIR)

if __name__ == "__main__":
    main()
