#!/usr/bin/env python3
"""
Strip Annotation Tags Helper Script

This script removes annotation tags like [Tag] and [/Tag] from the "biased_prompt_train" field
in a JSONL file while preserving all other data.
"""

import json
import re
from pathlib import Path
from datetime import datetime


def strip_annotation_tags(text: str) -> str:
    """
    Remove annotation tags in the format [Tag] or [/Tag] from text and normalize whitespace.

    Args:
        text: Input text with annotation tags

    Returns:
        Text with annotation tags removed and whitespace normalized
    """
    if not text:
        return text

    # Pattern to match [anything] or [/anything]
    pattern = r'\[/?[^\]]+\]'

    # Remove all matching tags
    cleaned_text = re.sub(pattern, '', text)

    # Normalize whitespace: replace multiple consecutive spaces with single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # Clean up leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def process_file(input_file: str) -> str:
    """
    Process the JSONL file and strip annotation tags from biased_prompt_train field.

    Args:
        input_file: Path to input JSONL file

    Returns:
        Path to output file
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output filename
    date_stamp = datetime.now().strftime("%Y-%m-%d")

    # Clean the stem by removing any .jsonl extensions
    clean_stem = input_path.stem
    if clean_stem.endswith('.jsonl'):
        clean_stem = clean_stem[:-6]  # Remove '.jsonl'

    output_filename = f"cleaned_{clean_stem}_{date_stamp}.jsonl"
    output_path = input_path.parent / output_filename

    print(f"Processing: {input_file}")
    print(f"Output: {output_path}")

    processed_count = 0
    cleaned_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())

                # Check if biased_prompt_train field exists
                if 'biased_prompt_train' in data and data['biased_prompt_train']:
                    original_text = data['biased_prompt_train']
                    cleaned_text = strip_annotation_tags(original_text)

                    # Update the field with cleaned text
                    data['biased_prompt_train'] = cleaned_text

                    # Count if any tags were actually removed
                    if original_text != cleaned_text:
                        cleaned_count += 1
                        print(f"Line {line_num}: Removed annotation tags")
                    else:
                        print(f"Line {line_num}: No tags found")
                else:
                    print(f"Line {line_num}: No biased_prompt_train field")

                # Write the processed line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total lines processed: {processed_count}")
    print(f"Lines with tags removed: {cleaned_count}")
    print(f"Output saved to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    # Hardcoded input file path as specified
    input_file = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\datasets\sprint 2.2 (2025-09-06)\annotated_biased_prompt_mmlu_psychology_train_V3_2025-09-06.jsonl.jsonl"

    try:
        output_path = process_file(input_file)
        print(f"\nTag stripping complete! Cleaned file saved to:")
        print(f"{output_path}")

    except Exception as e:
        print(f"Error: {e}")