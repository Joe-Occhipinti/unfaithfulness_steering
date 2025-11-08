import json
import statistics
import re
"""
**Some parts of it could be re-used for filtering activations based on CoT length.**

Used to understand if biased faithful answers are generally shorter or longer than unfaithful ones.
Measure and report the lengths of biased answers in tokens, categorized by
faithfulness classification.
"""

# File path
FILE_PATH = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\data\annotated\annotated_local_biased_high_school_psychology_2025-08-15.jsonl"

def count_tokens(text):
    """Approximate token count using word-based estimation."""
    # Split on whitespace and punctuation to approximate tokens
    # GPT-style tokenizers typically produce ~1.3 tokens per word
    words = re.findall(r'\b\w+\b', text)
    return int(len(words) * 1.3)

def load_and_measure():
    """Load JSONL file and measure token lengths by faithfulness."""
    faithful_lengths = []
    unfaithful_lengths = []

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())

            # Get the biased prompt and classification
            biased_prompt = record.get('biased_prompt', '')
            classification = record.get('faithfulness_classification', '')

            if not biased_prompt:
                continue

            # Count tokens
            token_count = count_tokens(biased_prompt)

            # Categorize by classification
            if classification == 'faithful':
                faithful_lengths.append(token_count)
            elif classification == 'unfaithful':
                unfaithful_lengths.append(token_count)

    return faithful_lengths, unfaithful_lengths

def main():
    print("Loading data and counting tokens...")
    faithful_lengths, unfaithful_lengths = load_and_measure()

    print(f"\n{'='*60}")
    print(f"FAITHFUL ANSWERS (n={len(faithful_lengths)})")
    print(f"{'='*60}")
    if faithful_lengths:
        print(f"Mean:   {statistics.mean(faithful_lengths):.2f} tokens")
        print(f"Median: {statistics.median(faithful_lengths):.2f} tokens")
    else:
        print("No faithful answers found")

    print(f"\n{'='*60}")
    print(f"UNFAITHFUL ANSWERS (n={len(unfaithful_lengths)})")
    print(f"{'='*60}")
    if unfaithful_lengths:
        print(f"Mean:   {statistics.mean(unfaithful_lengths):.2f} tokens")
        print(f"Median: {statistics.median(unfaithful_lengths):.2f} tokens")
    else:
        print("No unfaithful answers found")

if __name__ == "__main__":
    main()
