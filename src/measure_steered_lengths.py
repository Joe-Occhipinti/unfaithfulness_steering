import json
import statistics
import re
from collections import defaultdict
""""
Used to understand if steered faithful answers are generally shorter or longer than unfaithful ones.
Measure and report the lengths of steered answers in tokens, categorized by
the sign of the steering coefficient and the original faithfulness classification.
"""

# File path
FILE_PATH = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\annotated\annotated_steered_local_high_school_psychology_professor_2025-10-06.jsonl"

def count_tokens(text):
    """Approximate token count using word-based estimation."""
    # Split on whitespace and punctuation to approximate tokens
    # GPT-style tokenizers typically produce ~1.3 tokens per word
    words = re.findall(r'\b\w+\b', text)
    return int(len(words) * 1.3)

def load_and_measure():
    """Load JSONL file and measure token lengths by coefficient direction and original classification."""
    # Dictionary: (direction, original_classification) -> list of token counts
    length_data = defaultdict(list)

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())

            # Get the steered response and metadata
            steered_response = record.get('steered_response', '')
            coefficient = record.get('steering_coefficient')
            original_classification = record.get('original_faithfulness_classification', '')

            if not steered_response or coefficient is None:
                continue

            # Determine direction based on coefficient sign
            if coefficient > 0:
                direction = 'positive'
            elif coefficient < 0:
                direction = 'negative'
            else:
                direction = 'zero'

            # Count tokens
            token_count = count_tokens(steered_response)

            # Store by (direction, original_classification)
            key = (direction, original_classification)
            length_data[key].append(token_count)

    return length_data

def main():
    print("Loading data and counting tokens...")
    length_data = load_and_measure()

    print(f"\n{'='*80}")
    print(f"STEERED ANSWER LENGTHS BY COEFFICIENT DIRECTION AND ORIGINAL CLASSIFICATION")
    print(f"{'='*80}\n")

    # Sort keys for consistent output
    sorted_keys = sorted(length_data.keys())

    for key in sorted_keys:
        direction, original_classification = key
        lengths = length_data[key]
        print(f"Direction: {direction.upper():8} | Original: {original_classification:12} (n={len(lengths)})")
        print(f"  Mean:   {statistics.mean(lengths):6.2f} tokens")
        print(f"  Median: {statistics.median(lengths):6.2f} tokens")
        print()

if __name__ == "__main__":
    main()
