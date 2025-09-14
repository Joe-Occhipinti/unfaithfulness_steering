#!/usr/bin/env python3
"""
Prepare Validation Data Helper Script

This script transforms validation data from the sprint 1 format to the format
needed by the annotation processor. It extracts:
- biased_prompt_val: from "biased_prompt" field
- unbiased_answer: extracted from "change_type" (X -> Y format, X is unbiased)
- hinted_answer: extracted from "change_type" (X -> Y format, Y is hinted)

Input: val_biased_answers_2025-08-12.jsonl
Output: annotated_val_biased_answers_2025-09-14.jsonl
"""

import json
import re
from pathlib import Path
from datetime import datetime


def parse_change_type(change_type: str) -> tuple:
    """
    Parse change_type field to extract unbiased and hinted answers.

    Args:
        change_type: String in format "X -> Y" or "X_to_Y"

    Returns:
        Tuple of (unbiased_answer, hinted_answer)
    """
    if not change_type:
        return None, None

    # Handle both "A -> B" and "A_to_B" formats
    if " -> " in change_type:
        parts = change_type.split(" -> ")
    elif "_to_" in change_type:
        parts = change_type.split("_to_")
    else:
        return None, None

    if len(parts) == 2:
        unbiased_answer = parts[0].strip()
        hinted_answer = parts[1].strip()
        return unbiased_answer, hinted_answer

    return None, None


def process_validation_data(input_file: str, output_file: str) -> None:
    """
    Process validation data and create annotation-ready format.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Processing: {input_file}")
    print(f"Output: {output_file}")

    processed_count = 0
    skipped_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse input line
                data = json.loads(line.strip())

                # Extract required fields
                biased_prompt = data.get("biased_prompt")
                change_type = data.get("change_type")

                if not biased_prompt:
                    print(f"Warning: Line {line_num} missing biased_prompt field, skipping")
                    skipped_count += 1
                    continue

                if not change_type:
                    print(f"Warning: Line {line_num} missing change_type field, skipping")
                    skipped_count += 1
                    continue

                # Parse change_type to get unbiased and hinted answers
                unbiased_answer, hinted_answer = parse_change_type(change_type)

                if not unbiased_answer or not hinted_answer:
                    print(f"Warning: Line {line_num} could not parse change_type '{change_type}', skipping")
                    skipped_count += 1
                    continue

                # Create output record with only required fields
                output_record = {
                    "biased_prompt_val": biased_prompt,
                    "unbiased_answer": unbiased_answer,
                    "hinted_answer": hinted_answer
                }

                # Write to output file
                outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                processed_count += 1

                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} entries...")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_count += 1

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} entries")
    print(f"Skipped: {skipped_count} entries")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # Hardcoded paths as specified
    input_file = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\datasets\sprint 1 (2025\07\15-2025\08\15)\val_biased_answers_2025-08-12.jsonl"
    output_file = r"C:\Users\l440\Desktop\unfaithfulness_steering-1\annotated_val_biased_answers_2025-09-14.jsonl"

    try:
        process_validation_data(input_file, output_file)
        print(f"\nValidation data preparation complete!")
        print(f"Ready for annotation processing with:")
        print(f"python sentence_annotation_processor.py \"{output_file}\"")
        print(f"or")
        print(f"python gemini_annotation_processor.py \"{output_file}\"")

    except Exception as e:
        print(f"Error: {e}")