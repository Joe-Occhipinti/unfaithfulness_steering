"""
Analyze the distribution of hint letters (A, B, C, D) in faithful examples from hinted runs.
Uses training split (70% of data with random seed 42).
"""

import json
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ============================================================================
# TUNABLE PARAMETERS
# ============================================================================

# Input file path
INPUT_FILE = "data/annotated/hinted/psychology_professor_2025-08-15/annotated_global_biased_high_school_psychology_2025-08-15.jsonl"

# Output directory for plots
OUTPUT_DIR = "plots/psychology_professor_2025-08-15/hinted_letter_distribution"

# Train/val split parameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.7

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def load_and_filter_data(filepath):
    """Load data, apply train/val split, and filter for faithful examples in training set."""
    all_examples = []

    # Load all examples
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            all_examples.append(example)

    # Apply train/val split with seed 42
    random.seed(RANDOM_SEED)
    n_total = len(all_examples)
    n_train = int(n_total * TRAIN_RATIO)

    # Shuffle indices
    indices = list(range(n_total))
    random.shuffle(indices)

    # Get train indices
    train_indices = set(indices[:n_train])

    # Filter for training examples that are faithful
    filtered_examples = []
    for idx, example in enumerate(all_examples):
        if idx not in train_indices:
            continue

        # Filter for faithful examples
        if example.get('faithfulness_classification') == 'unfaithful':
            filtered_examples.append(example)

    return filtered_examples, n_train, n_total

def extract_letter_distribution(examples):
    """Extract and count hint letter distribution."""
    letters = [ex['hint_letter'] for ex in examples]
    return Counter(letters)

def plot_distribution(counter, output_dir, n_train, n_total):
    """Create bar chart of hint letter distribution."""
    # Ensure all letters A-D are present
    all_letters = ['A', 'B', 'C', 'D']
    counts = [counter.get(letter, 0) for letter in all_letters]
    total = sum(counts)
    percentages = [100 * c / total if total > 0 else 0 for c in counts]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(all_letters, counts, color='seagreen', alpha=0.8)

    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Hint Letter', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of Hint Letters in Unfaithful Examples (Training Split)\n'
                 f'Unfaithful N={total}, Train N={n_train}/{n_total} (seed={RANDOM_SEED})',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = 'hinted_letter_distribution_unfaithful_train.png'
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path / filename}")

    plt.show()

def main():
    print(f"Analyzing hint letter distribution in faithful examples (training split)")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Train ratio: {TRAIN_RATIO}, Seed: {RANDOM_SEED}\n")

    # Load and filter data
    examples, n_train, n_total = load_and_filter_data(INPUT_FILE)
    print(f"Total examples: {n_total}")
    print(f"Training examples: {n_train}")
    print(f"Faithful training examples: {len(examples)}")

    if len(examples) == 0:
        print("No faithful examples found in training set. Check your input file.")
        return

    # Extract letter distribution
    letter_counts = extract_letter_distribution(examples)

    # Print summary
    print("\nHint Letter Distribution in Faithful Training Examples:")
    for letter in ['A', 'B', 'C', 'D']:
        count = letter_counts.get(letter, 0)
        pct = 100 * count / len(examples)
        print(f"  {letter}: {count} ({pct:.1f}%)")

    # Plot
    plot_distribution(letter_counts, OUTPUT_DIR, n_train, n_total)

if __name__ == "__main__":
    main()
