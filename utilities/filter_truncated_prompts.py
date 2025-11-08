"""
filter_truncated_prompts.py

Filters out prompts that don't have a period before closing tags [/F_body] or [/U_body].
These are likely truncated prompts that didn't complete properly.

Usage:
    python filter_truncated_prompts.py <input_file> <output_file>

Example:
    python filter_truncated_prompts.py data/sprint4_2025-10-21/annotated/annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl data/sprint4_2025-10-21/annotated/filtered_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl
"""

import json
import re
import sys
import os

# =============================================================================
# CONFIGURATION - Set input and output file paths here
# =============================================================================

INPUT_FILE = "data/sprint4_2025-10-21/annotated/reprocessed_local_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"
OUTPUT_FILE = "data/sprint4_2025-10-21/annotated/clean_reprocessed_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"
DISCARDED_FILE = "data/sprint4_2025-10-21/annotated/discarded__reprocessed_annotated_scie_hist_psy_X_grader_prof_meta_2025-10-25.jsonl"

def has_period_before_closing_tag(text):
    """
    Check if text has a period right before or before a space before [/F_body] or [/U_body] closing tags.

    Returns:
        True if period is found before the closing tag (properly completed)
        False if no period is found (likely truncated)
    """
    # Pattern: period followed by optional whitespace, then closing tag (either [/F_body] or [/U_body])
    pattern = r'\.\s*\[/[FU]_body\]'
    return bool(re.search(pattern, text))

def filter_truncated_prompts(input_file, output_file, discarded_file):
    """
    Filter out truncated prompts from input file and save to output file.
    Also saves discarded prompts to a separate file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (complete prompts)
        discarded_file: Path to discarded JSONL file (truncated prompts)
    """

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(discarded_file), exist_ok=True)

    total_count = 0
    truncated_count = 0
    kept_count = 0

    truncated_examples = []

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(discarded_file, 'w', encoding='utf-8') as discardfile:

        for idx, line in enumerate(infile):
            total_count += 1
            data = json.loads(line)

            # Get the local annotated prompt (contains [/F_body] or [/U_body] tags)
            local_annotated_prompt = data.get('local_annotated_biased_prompt', '')

            # Check if it has proper period before closing tag
            is_complete = has_period_before_closing_tag(local_annotated_prompt)

            if is_complete:
                # Keep this prompt - write to output
                outfile.write(line)
                kept_count += 1
            else:
                # Truncated - write to discarded file
                discardfile.write(line)
                truncated_count += 1

                # Store first 5 examples for reporting
                if len(truncated_examples) < 5:
                    truncated_examples.append({
                        'idx': idx,
                        'id': data.get('question', '')[:50] + '...',
                        'answer': data.get('biased_answer_letter', ''),
                        'hint': data.get('hint_letter', '')
                    })

    # Print summary
    print(f"=== FILTERING SUMMARY ===")
    print(f"Input file: {input_file}")
    print(f"Output file (complete): {output_file}")
    print(f"Discarded file (truncated): {discarded_file}")
    print(f"\nTotal prompts processed: {total_count}")
    print(f"Prompts kept (complete): {kept_count} ({kept_count/total_count*100:.1f}%)")
    print(f"Prompts discarded (truncated): {truncated_count} ({truncated_count/total_count*100:.1f}%)")

    if truncated_examples:
        print(f"\n--- First {len(truncated_examples)} truncated prompts discarded ---")
        for item in truncated_examples:
            print(f"  Line {item['idx']}: {item['id']}")
            print(f"    answer={item['answer']}, hint={item['hint']}")

    print(f"\n✅ Complete dataset saved to: {output_file}")
    print(f"✅ Discarded dataset saved to: {discarded_file}")
    return kept_count, truncated_count

if __name__ == "__main__":
    # Use command-line arguments if provided, otherwise use hardcoded paths
    if len(sys.argv) == 4:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        discarded_file = sys.argv[3]
    else:
        input_file = INPUT_FILE
        output_file = OUTPUT_FILE
        discarded_file = DISCARDED_FILE
        print(f"Using hardcoded paths from configuration:")
        print(f"  Input:     {input_file}")
        print(f"  Output:    {output_file}")
        print(f"  Discarded: {discarded_file}\n")

    filter_truncated_prompts(input_file, output_file, discarded_file)
