#!/usr/bin/env python3
"""
Letter Choice Extractor

This script extracts letter choices from model answers using regex pattern matching.
It looks for the first capitalized letter after "Therefore, the best answer is" in each answer.

Usage:
    python extract_letter_choices.py

Configuration:
    - INPUT_FILE: Path to input JSONL file with 'generated_answer' field
    - OUTPUT_FILE: Path to output JSONL file with extracted letter choices
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

def extract_letter_choice(answer_text: str) -> Optional[str]:
    """
    Extract the first capitalized letter after "Therefore, the best answer is" using regex.
    
    Args:
        answer_text: The generated answer text to search
        
    Returns:
        The extracted letter (A, B, C, D) or None if not found
    """
    # Regex pattern to find "Therefore, the best answer is" followed by a capitalized letter
    # This pattern is case-insensitive and handles various formatting (parentheses, asterisks, colons, etc.)
    pattern = r"therefore,?\s+the\s+best\s+answer\s+is[:\s]*[\(\*]*([A-D])[\)\*]*"
    
    # Search for the pattern (case-insensitive)
    match = re.search(pattern, answer_text, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()  # Return the letter in uppercase
    
    return None

def process_jsonl_file(input_path: str, output_path: str) -> None:
    """
    Process a JSONL file and extract letter choices from generated answers.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print("-" * 60)
    
    processed_count = 0
    found_count = 0
    not_found_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse input line
                data = json.loads(line.strip())
                
                # Check if generated_answer field exists
                if 'generated_biased_answer' not in data:
                    print(f"Warning: Line {line_num} missing 'generated_answer' field, skipping")
                    continue
                
                generated_answer = data['generated_biased_answer']
                
                # Extract letter choice
                letter_choice = extract_letter_choice(generated_answer)
                
                # Add extraction results to original data
                data['extracted_biased_letter'] = letter_choice
                data['extraction_timestamp'] = datetime.now().isoformat()
                
                if letter_choice:
                    found_count += 1
                    status = f"Found: {letter_choice}"
                else:
                    not_found_count += 1
                    status = "Not found"
                
                print(f"Line {line_num}: {status}")
                
                # Write to output file
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                outfile.flush()  # Force write to disk immediately
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print("-" * 60)
    print(f"Processing complete!")
    print(f"Total processed: {processed_count} entries")
    print(f"Letters found: {found_count} entries")
    print(f"Letters not found: {not_found_count} entries")
    print(f"Success rate: {found_count/processed_count*100:.1f}%" if processed_count > 0 else "N/A")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE VARIABLES
    # ========================================
    
    # Input file path (JSONL file containing 'generated_biased_answer' field)
    INPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_correct_with_biased_merged.jsonl"
    
    # Output file path
    OUTPUT_FILE = r"C:\Users\l440\unfaithfulness_steering\datasets\mmlu_psychology_correct_with_biased_extracted.jsonl"
    
    # ========================================
    # SCRIPT EXECUTION
    # ========================================
    
    print("=" * 60)
    print("Letter Choice Extractor")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)
    
    try:
        process_jsonl_file(INPUT_FILE, OUTPUT_FILE)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")